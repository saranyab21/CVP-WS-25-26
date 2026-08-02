import cv2
import numpy as np
from mtcnn import MTCNN


class FaceDetector:
    """
    Face detection + tracking.

    Design idea:
    - Face detection (MTCNN) -> run it once to initialize and only for recovery.
    - Template matching is fast -> run it every frame to keep the video smooth.

    1. We run MTCNN once to get the initial face bbox and template.
    2. For each new frame, we search only near the last bbox and run fast template matching.
    3. If the similarity score is good, we update bbox; if it’s weak, we count a miss.
    4. Only after 3 consecutive misses (and a cooldown), we re-run detection to recover.
    5. We preprocess to gray+blur for stable matching, and refresh the template only when the match is strong to avoid drift.
    """

    def __init__(self, tm_window_size=25, tm_threshold=0.5, aligned_image_size=224, debug=False):
        # Slow but robust face detector (used for initialization and re-detection).
        self.detector = MTCNN()

        # Template matching settings (these are the ones we tuned in the assignment).
        self.tm_window_size = tm_window_size      # how far we search around the last bbox
        self.tm_threshold = tm_threshold          # min similarity score to accept tracking
        self.aligned_image_size = aligned_image_size
        self.debug = debug

        # Tracking state (updated over time).
        self.reference = None                     # last full frame used to initialize tracking (optional)
        self.template = None                      # the current face template (preprocessed: gray+blur)
        self.bbox = None                          # current bbox [x, y, w, h]

        # Internal tracking rules:
        # We do not run the detector immediately on a single bad match,
        # because one frame can be noisy (motion blur, small occlusion).
        self._MISS_TO_REDETECT = 3                # re-detect only after N consecutive bad matches
        self._REDETECT_COOLDOWN = 5               # after re-detection, wait a few frames before re-detecting again
        self._UPDATE_TEMPLATE_THRESHOLD = 0.7     # update template only when the match is strong (avoids drift)

        # Counters to implement the above rules.
        self.frame_idx = 0
        self.miss_count = 0
        self.last_detect_frame = -999999

    # ---------- helpers ----------
    def _preprocess_for_tm(self, img_bgr):
        """
        Template matching works better in gray scale.
        Slight blur reduces noise (helps stabilize match scores across frames).
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        return gray

    def _maybe_redetect(self, image):
        """
        Decide whether we should run face detection as a recovery step.

        We only re-detect if:
        - tracking failed for several consecutive frames (miss_count >= N), AND
        - we are past a short cooldown since the last detection.

        If we do NOT re-detect, we simply keep the last bbox so the video remains smooth.
        """
        cooldown_ok = (self.frame_idx - self.last_detect_frame) >= self._REDETECT_COOLDOWN
        miss_ok = self.miss_count >= self._MISS_TO_REDETECT

        if miss_ok and cooldown_ok:
            if self.debug:
                print(f"[REDETECT] miss_count={self.miss_count}")
            return self.detect_face(image)

        # No re-detect: return current bbox (best guess) without doing expensive detection.
        # aligned is still produced so downstream code can continue.
        aligned = self.align_face(image, self.bbox) if self.bbox is not None else None
        return {"rect": self.bbox, "image": image, "aligned": aligned, "response": None}

    # ---------- slow detection ----------
    def detect_face(self, image):
        """
        Run MTCNN on the full image.
        This is the slow step -> should not be called every frame.
        """
        detections = self.detector.detect_faces(image)  # (no threshold args: mtcnn package API)
        if not detections:
            # If nothing is detected, reset state so we can try again later.
            self.reference = None
            self.template = None
            self.bbox = None
            return None

        # If multiple faces are present, choose the largest face (most likely the main subject).
        largest_detection = np.argmax([d["box"][2] * d["box"][3] for d in detections])
        face_rect = detections[largest_detection]["box"]  # [x, y, w, h]

        # Save tracking state based on the detection result.
        self.reference = image.copy()
        self.bbox = [int(face_rect[0]), int(face_rect[1]), int(face_rect[2]), int(face_rect[3])]

        # Build a template from the detected face region.
        tpl_bgr = self.crop_face(image, self.bbox)
        if tpl_bgr is None or tpl_bgr.size == 0:
            return None

        # Store the template in the same format we use for matching (gray+blur).
        self.template = self._preprocess_for_tm(tpl_bgr)

        # Detection is our new reliable reference -> reset miss logic.
        self.miss_count = 0
        self.last_detect_frame = self.frame_idx

        # Provide the aligned face crop for recognition.
        aligned = self.align_face(image, self.bbox)
        return {"rect": self.bbox, "image": image, "aligned": aligned, "response": 1.0}

    # ---------- fast tracking ----------
    def track_face(self, image):
        """
        Per-frame tracking using template matching.

        Flow:
        1) If we are not initialized -> do one detection.
        2) Else match template inside a window around last bbox (cheap).
        3) If match fails repeatedly -> call detect_face() to recover.
        """
        self.frame_idx += 1

        # First frame (or after losing track completely): run detection once to initialize.
        if self.template is None or self.bbox is None:
            return self.detect_face(image)

        x, y, w, h = self.bbox
        pad = self.tm_window_size

        # Search only near the previous bbox (faster + avoids matching to random regions).
        x1 = max(x - pad, 0)
        y1 = max(y - pad, 0)
        x2 = min(x + w + pad, image.shape[1] - 1)
        y2 = min(y + h + pad, image.shape[0] - 1)

        search_bgr = image[y1:y2, x1:x2]
        if search_bgr is None or search_bgr.size == 0:
            # If the search region is invalid, count as a miss and maybe re-detect.
            self.miss_count += 1
            return self._maybe_redetect(image)

        # Preprocess search area the same way as the template.
        search = self._preprocess_for_tm(search_bgr)
        tpl = self.template

        # Template matching requires search area to be larger than the template.
        if search.shape[0] < tpl.shape[0] or search.shape[1] < tpl.shape[1]:
            self.miss_count += 1
            return self._maybe_redetect(image)

        # Compute normalized correlation score for every possible template location.
        result = cv2.matchTemplate(search, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        # If similarity is too low, treat it as tracking failure for this frame.
        # We do not re-detect immediately (to avoid lag), we require consecutive failures.
        if max_val < self.tm_threshold:
            self.miss_count += 1
            if self.debug and self.frame_idx % 10 == 0:
                print(f"[TM FAIL] score={max_val:.3f} < {self.tm_threshold} (miss={self.miss_count})")
            return self._maybe_redetect(image)

        # Successful match -> reset failure counter.
        self.miss_count = 0

        # Convert best match position back to image coordinates.
        new_x = x1 + max_loc[0]
        new_y = y1 + max_loc[1]
        new_bbox = [int(new_x), int(new_y), int(w), int(h)]

        # Update tracker state.
        self.bbox = new_bbox
        self.reference = image.copy()

        # Template refresh:
        # Update only when the match is strong, otherwise the template can drift.
        if max_val >= self._UPDATE_TEMPLATE_THRESHOLD:
            new_tpl_bgr = self.crop_face(image, new_bbox)
            if new_tpl_bgr is not None and new_tpl_bgr.size != 0:
                self.template = self._preprocess_for_tm(new_tpl_bgr)

        # Output aligned crop for recognition pipeline.
        aligned = self.align_face(image, new_bbox)
        return {"rect": new_bbox, "image": image, "aligned": aligned, "response": float(max_val)}

    # ---------- alignment + crop ----------
    def align_face(self, image, face_rect):
        """
        Produce a fixed-size face crop for the recognition model.
        (The recognition part expects a consistent input size.)
        """
        crop = self.crop_face(image, face_rect)
        if crop is None or crop.size == 0:
            return None
        return cv2.resize(crop, dsize=(self.aligned_image_size, self.aligned_image_size))

    def crop_face(self, image, face_rect):
        """
        Crop ROI from bbox with boundary checks.
        face_rect is [x, y, w, h].
        """
        x, y, w, h = face_rect
        top = max(int(y), 0)
        left = max(int(x), 0)
        bottom = min(int(y + h), image.shape[0] - 1)
        right = min(int(x + w), image.shape[1] - 1)

        if bottom <= top or right <= left:
            return np.empty((0, 0, 3), dtype=image.dtype)

        return image[top:bottom, left:right, :]
