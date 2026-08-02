"""
misc.py

Purpose:
Creates evidence for assignment completeness:
- 4.1: parameter tuning for template matching tracking (tm_window_size, tm_threshold)
- 4.2: parameter tuning for recognition (k, max_distance, min_prob)
- 4.2: robustness tests (pose/rotation, brightness, blur)
- 4.3: k-means objective over iterations + sensitivity to init
- 4.4: DIR curve + threshold selection for (i) FAR<=1% max ID, (ii) ID>=90% min FAR

Run:
python misc.py

Outputs (created in ./evidence_outputs/):
- tm_tuning.csv
- recog_tuning.csv
- robustness.txt
- kmeans_objective_seed_<seed>.csv
- kmeans_objective_plot.png
- dir_curve_results.csv
- dir_thresholds.txt
"""

import os
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt

from config import Config
from face_detector import FaceDetector
from face_recognition import FaceRecognizer, FaceClustering
from evaluation import OpenSetEvaluation
from classifier import NearestNeighborClassifier


OUT_DIR = "evidence_outputs"


def ensure_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def save_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_some_frames(max_frames=30):
    """
    Loads frames from the folder given by Config.TEST_DATA / first identity folder found.
    Uses your course structure data/test_data/<Person>/<frames>.
    """
    base = Config.TEST_DATA
    if not base.exists():
        print("[ERROR] Config.TEST_DATA not found:", base)
        return []

    # find first folder containing images
    for person_dir in sorted(base.iterdir()):
        if person_dir.is_dir():
            imgs = sorted([p for p in person_dir.iterdir() if p.suffix.lower() in [".jpg", ".png", ".jpeg"]])
            frames = []
            for p in imgs[:max_frames]:
                im = cv2.imread(str(p))
                if im is not None:
                    frames.append(im)
            if frames:
                print("[INFO] Using frames from:", person_dir)
                return frames

    print("[ERROR] No frames found in:", base)
    return []


# -----------------------------
# 4.1: Tune template matching
# -----------------------------
def tune_template_matching(frames, window_sizes, thresholds):
    detector = FaceDetector()
    results = []

    for w in window_sizes:
        for t in thresholds:
            detector.tm_window_size = w
            detector.tm_threshold = t

            # reset state
            detector.template = None
            detector.bbox = None
            detector.reference = None

            ok = 0
            total = 0
            redetect_proxy = 0
            responses = []

            for i, frame in enumerate(frames):
                total += 1
                try:
                    out = detector.track_face(frame)
                    if out is None:
                        redetect_proxy += 1
                        continue

                    ok += 1
                    resp = out.get("response", None)
                    
                    # If response is None, it means we came via detect_face (re-detection)
                    if resp is None:
                        redetect_proxy += 1
                    else:
                        responses.append(float(resp))
                        if float(resp) < t:
                            # Low match score means tracker would consider re-detection
                            redetect_proxy += 1
                            
                except Exception:
                    redetect_proxy += 1

            results.append({
                "tm_window_size": w,
                "tm_threshold": t,
                "frames_total": total,
                "frames_ok": ok,
                "ok_rate": ok / max(1, total),
                "redetect_proxy": redetect_proxy,
                "avg_response": float(np.mean(responses)) if responses else float("nan"),
            })

    save_csv(os.path.join(OUT_DIR, "tm_tuning.csv"), results)
    print("[SAVED] evidence_outputs/tm_tuning.csv")


# -----------------------------
# 4.2: Tune k, max_distance, min_prob
# -----------------------------
def build_gallery_from_test_data(recognizer, detector, max_per_id=5):
    """
    Enroll identities from Config.TEST_DATA/<person>/... images.
    """
    base = Config.TEST_DATA
    labels = []
    for person_dir in sorted(base.iterdir()):
        if not person_dir.is_dir():
            continue
        imgs = sorted([p for p in person_dir.iterdir() if p.suffix.lower() in [".jpg", ".png", ".jpeg"]])
        if not imgs:
            continue

        label = person_dir.name
        labels.append(label)

        for p in imgs[:max_per_id]:
            im = cv2.imread(str(p))
            if im is None:
                continue
            face = detector.detect_face(im)
            if face is None:
                continue
            recognizer.partial_fit(face["aligned"], label)

    return labels

def split_identities_for_unknown(detector, base_dir, max_per_id=3, seed=42):
    """
    Returns:
      known_data: list of (aligned_face, label) for enrolled identities
      known_val:  list of (aligned_face, label) for validation from enrolled identities
      unknown_val:list of aligned_face from identities NOT enrolled (treated as unknown)
    """
    person_dirs = [p for p in sorted(base_dir.iterdir()) if p.is_dir()]
    rng = np.random.RandomState(seed)
    rng.shuffle(person_dirs)
    if len(person_dirs) < 2:
        raise RuntimeError("Need at least 2 identities in provided data to form unknown split.")

    # Example split: first half known, second half unknown
    split = len(person_dirs) // 2
    known_ids = person_dirs[:split]
    unknown_ids = person_dirs[split:]

    known_train = []
    known_val = []
    unknown_val = []

    for pdir in known_ids:
        imgs = sorted([p for p in pdir.iterdir() if p.suffix.lower() in [".jpg",".jpeg",".png"]])
        aligned = []
        for im_path in imgs[:max_per_id*2]:
            im = cv2.imread(str(im_path))
            if im is None:
                continue
            face = detector.detect_face(im)
            if face is None:
                continue
            aligned.append(face["aligned"])

        # use first part for gallery, second part for known-val
        label = pdir.name
        for a in aligned[:max_per_id]:
            known_train.append((a, label))
        for a in aligned[max_per_id:max_per_id*2]:
            known_val.append((a, label))

    for pdir in unknown_ids:
        imgs = sorted([p for p in pdir.iterdir() if p.suffix.lower() in [".jpg",".jpeg",".png"]])
        for im_path in imgs[:max_per_id]:
            im = cv2.imread(str(im_path))
            if im is None:
                continue
            face = detector.detect_face(im)
            if face is None:
                continue
            unknown_val.append(face["aligned"])

    return known_train, known_val, unknown_val


def tune_recognizer_from_provided_data(grid_k, grid_dist, grid_prob, max_per_id=3):
    detector = FaceDetector()
    recognizer = FaceRecognizer()

    # clear any loaded gallery
    recognizer.labels = []
    recognizer.embeddings = np.empty((0, 128), dtype=np.float32)

    known_train, known_val, unknown_val = split_identities_for_unknown(
        detector, Config.TEST_DATA, max_per_id=max_per_id
    )

    # enroll gallery
    for face, label in known_train:
        recognizer.partial_fit(face, label)

    results = []
    for k in grid_k:
        for d in grid_dist:
            for p in grid_prob:
                recognizer.num_neighbours = k
                recognizer.max_distance = d
                recognizer.min_prob = p

                # known ID accuracy
                correct = 0
                total_known = 0
                for face, gt in known_val:
                    pred, post, dist = recognizer.predict(face)
                    total_known += 1
                    if pred == gt:
                        correct += 1
                known_id_rate = correct / max(1, total_known)

                # unknown rejection
                rej = 0
                total_unk = 0
                for face in unknown_val:
                    pred, post, dist = recognizer.predict(face)
                    total_unk += 1
                    if pred == "unknown":
                        rej += 1
                unknown_rejection = rej / max(1, total_unk)

                results.append({
                    "k": k,
                    "max_distance": d,
                    "min_prob": p,
                    "known_id_rate": known_id_rate,
                    "unknown_rejection_rate": unknown_rejection,
                    "known_samples": total_known,
                    "unknown_samples": total_unk,
                })

    save_csv(os.path.join(OUT_DIR, "recog_tuning.csv"), results)
    print("[SAVED] evidence_outputs/recog_tuning.csv")

# -----------------------------
# 4.2: Robustness tests
# -----------------------------
def rotate_image(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def change_brightness(img, factor):
    x = img.astype(np.float32) * factor
    return np.clip(x, 0, 255).astype(np.uint8)


def blur(img, k=7):
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img, (k, k), 0)


def robustness_report(frames):
    detector = FaceDetector()
    recognizer = FaceRecognizer()

    # gallery from first identity folder only
    build_gallery_from_test_data(recognizer, detector, max_per_id=3)

    base = frames[0]
    variants = {
        "frontal": base,
        "rotate_+30": rotate_image(base, 30),
        "rotate_-30": rotate_image(base, -30),
        "bright_x1.5": change_brightness(base, 1.5),
        "dark_x0.6": change_brightness(base, 0.6),
        "blur_k7": blur(base, 7),
    }

    lines = []
    for name, im in variants.items():
        face = detector.detect_face(im)
        if face is None:
            lines.append(f"{name}: detection FAILED")
            continue
        pred, post, dist = recognizer.predict(face["aligned"])
        lines.append(f"{name}: pred={pred}, posterior={post:.4f}, min_dist={dist:.4f}")

    path = os.path.join(OUT_DIR, "robustness.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("[SAVED] evidence_outputs/robustness.txt")


# -----------------------------
# 4.3: k-means objective tracking (external)
# -----------------------------
def kmeans_objective(X, centers, membership):
    diffs = X - centers[membership]
    return float(np.sum(np.sum(diffs ** 2, axis=1)))


def run_kmeans_objective(frames, k=3, max_iter=25, seeds=(1, 42, 99)):
    detector = FaceDetector()
    clustering = FaceClustering(num_clusters=k, max_iter=max_iter)

    # collect embeddings
    for im in frames[:20]:
        face = detector.detect_face(im)
        if face is None:
            continue
        clustering.partial_fit(face["aligned"])

    X = clustering.embeddings
    if X.shape[0] < k:
        print("[WARN] Not enough embeddings for k-means evidence.")
        return

    all_hist = {}

    for seed in seeds:
        np.random.seed(seed)
        idx = np.random.choice(X.shape[0], k, replace=False)
        centers = X[idx].copy()
        hist = []

        for it in range(max_iter):
            dists = np.linalg.norm(X[:, None] - centers[None, :], axis=2)
            membership = np.argmin(dists, axis=1)

            J = kmeans_objective(X, centers, membership)
            hist.append(J)

            for ci in range(k):
                pts = X[membership == ci]
                if len(pts) > 0:
                    centers[ci] = np.mean(pts, axis=0)

        all_hist[seed] = hist
        save_csv(os.path.join(OUT_DIR, f"kmeans_objective_seed_{seed}.csv"),
                 [{"iteration": i, "objective": v} for i, v in enumerate(hist)])

    plt.figure()
    for seed, hist in all_hist.items():
        plt.plot(hist, label=f"seed={seed}")
    plt.xlabel("Iteration")
    plt.ylabel("Objective")
    plt.title("k-means objective convergence")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "kmeans_objective_plot.png"), dpi=150)
    plt.close()
    print("[SAVED] evidence_outputs/kmeans_objective_plot.png")

    print("[SAVED] evidence_outputs/kmeans_objective_seed_*.csv")


# -----------------------------
# 4.3 (extra): Tune num_clusters (k in k-means)
# -----------------------------
def _collect_embeddings_for_clustering(frames, max_frames=20):
    """
    Collects embeddings once, independent of number of clusters.
    We use FaceClustering only as an embedding extractor (it wraps FaceNet).
    """
    detector = FaceDetector()
    tmp = FaceClustering(num_clusters=1, max_iter=1)  # num_clusters irrelevant here
    for im in frames[:max_frames]:
        face = detector.detect_face(im)
        if face is None:
            continue
        tmp.partial_fit(face["aligned"])
    return tmp.embeddings


def _run_kmeans_objective_on_X(X, k, max_iter=25, seed=42):
    """
    Runs plain k-means on a fixed embedding matrix X and returns objective history.
    """
    if X.shape[0] < k:
        return None

    np.random.seed(seed)
    idx = np.random.choice(X.shape[0], k, replace=False)
    centers = X[idx].copy()
    hist = []

    for it in range(max_iter):
        dists = np.linalg.norm(X[:, None] - centers[None, :], axis=2)
        membership = np.argmin(dists, axis=1)

        J = kmeans_objective(X, centers, membership)
        hist.append(J)

        # update centers
        for ci in range(k):
            pts = X[membership == ci]
            if len(pts) > 0:
                centers[ci] = np.mean(pts, axis=0)

    return hist


def tune_num_clusters(frames, cluster_list=(2, 3, 4, 5), max_iter=25, seeds=(1, 42, 99), max_frames=20):
    """
    Tunes num_clusters (k in k-means) by running k-means for different k values
    and reporting mean/std final objective across seeds.

    Outputs:
      - evidence_outputs/num_clusters_tuning.csv
      - evidence_outputs/num_clusters_elbow.png
    """
    X = _collect_embeddings_for_clustering(frames, max_frames=max_frames)
    if X is None or X.shape[0] < 2:
        print("[WARN] Not enough embeddings to tune num_clusters.")
        return

    rows = []
    mean_objs = []
    ks = []

    for k in cluster_list:
        finals = []
        for seed in seeds:
            hist = _run_kmeans_objective_on_X(X, k=k, max_iter=max_iter, seed=seed)
            if hist is None:
                continue
            finals.append(hist[-1])

        if not finals:
            continue

        ks.append(k)
        meanJ = float(np.mean(finals))
        stdJ = float(np.std(finals))
        mean_objs.append(meanJ)

        rows.append({
            "num_clusters": k,
            "mean_final_objective": meanJ,
            "std_final_objective": stdJ,
            "n_seeds": len(finals),
            "n_points": int(X.shape[0]),
            "max_iter": int(max_iter),
        })

    save_csv(os.path.join(OUT_DIR, "num_clusters_tuning.csv"), rows)
    print("[SAVED] evidence_outputs/num_clusters_tuning.csv")

    # Elbow plot (objective vs k): objective always decreases with k,
    # so we look for diminishing returns (elbow).
    if ks:
        plt.figure()
        plt.plot(ks, mean_objs, marker="o")
        plt.xlabel("num_clusters (k in k-means)")
        plt.ylabel("Mean final objective (lower is better)")
        plt.title("Elbow curve for choosing num_clusters")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "num_clusters_elbow.png"), dpi=150)
        plt.close()
        print("[SAVED] evidence_outputs/num_clusters_elbow.png")



# -----------------------------
# 4.4: DIR curve thresholds extraction
# -----------------------------
def dir_curve_and_thresholds():
    false_alarm_rate_range = np.logspace(-3.0, 0, 1000, endpoint=False)
    classifier = NearestNeighborClassifier()
    evaluator = OpenSetEvaluation(classifier=classifier, false_alarm_rate_range=false_alarm_rate_range)

    evaluator.prepare_input_data(Config.EVAL_TRAIN_DATA, Config.EVAL_TEST_DATA)
    results = evaluator.run()

    fars = np.array(results["false_alarm_rates"], dtype=float)
    idr = np.array(results["identification_rates"], dtype=float)
    thr = np.array(results["similarity_thresholds"], dtype=float)

    # (i) FAR <= 1% -> maximize ID
    mask1 = fars <= 0.01
    if np.any(mask1):
        sel1 = np.where(mask1)[0][np.argmax(idr[mask1])]
        op1 = (fars[sel1], idr[sel1], thr[sel1])
    else:
        op1 = (np.nan, np.nan, np.nan)

    # (ii) ID >= 90% -> minimize FAR
    mask2 = idr >= 0.90
    if np.any(mask2):
        sel2 = np.where(mask2)[0][np.argmin(fars[mask2])]
        op2 = (fars[sel2], idr[sel2], thr[sel2])
    else:
        op2 = (np.nan, np.nan, np.nan)

    # save full curve table
    rows = [{"far": float(f), "idr": float(i), "thr": float(t)} for f, i, t in zip(fars, idr, thr)]
    save_csv(os.path.join(OUT_DIR, "dir_curve_results.csv"), rows)

    txt = os.path.join(OUT_DIR, "dir_thresholds.txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write("Operating point (i): FAR<=1% and ID maximized\n")
        f.write(f"FAR={op1[0]:.6f}, ID={op1[1]:.6f}, thr={op1[2]:.6f}\n\n")
        f.write("Operating point (ii): ID>=90% and FAR minimized\n")
        f.write(f"FAR={op2[0]:.6f}, ID={op2[1]:.6f}, thr={op2[2]:.6f}\n")

    print("[SAVED] evidence_outputs/dir_curve_results.csv")
    print("[SAVED] evidence_outputs/dir_thresholds.txt")


def main():
    ensure_dir()
    frames = load_some_frames(max_frames=30)
    if not frames:
        return

    # 4.1 tuning
    tune_template_matching(frames, window_sizes=[15, 25, 35], thresholds=[0.4, 0.5, 0.6])

    # 4.2 tuning
    tune_recognizer_from_provided_data(grid_k=[1, 3, 5], grid_dist=[0.8, 1.0, 1.2], grid_prob=[0.4, 0.5, 0.6])

    # 4.2 robustness
    robustness_report(frames)

    # 4.3 objective tracking
    run_kmeans_objective(frames, k=3, max_iter=25, seeds=(1, 42, 99))

    # 4.3 extra: tune num_clusters
    tune_num_clusters(frames, cluster_list=(2, 3, 4, 5), max_iter=25, seeds=(1, 42, 99))


    # 4.4 DIR thresholds
    dir_curve_and_thresholds()


if __name__ == "__main__":
    main()
