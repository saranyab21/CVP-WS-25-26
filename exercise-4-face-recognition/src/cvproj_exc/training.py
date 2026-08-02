import argparse
import cv2

from dotenv import load_dotenv
load_dotenv()

from config import Config, ReIdMode, enum_choices
from face_detector import FaceDetector
from face_recognition import FaceClustering, FaceRecognizer


# The training module of the face recognition system.
# Workflow:
#   1) Capture new video frame
#   2) Run face detection/tracking (Exercise 5.1)
#   3) Extract face embedding and update face identification/clustering
#   4) Save trained models
def main(args):
    # Setup OpenCV video capture
    if args.video == "none":
        camera = cv2.VideoCapture(-1)
        wait_for_frame = 1
    else:
        camera = cv2.VideoCapture(args.video)
        wait_for_frame = 100
    camera.set(3, 640)
    camera.set(4, 480)

    # Image display
    cv2.namedWindow("Camera")
    cv2.moveWindow("Camera", 0, 0)

    # Prepare detection, identification, clustering
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    clustering = FaceClustering()

    # Video capturing loop
    state = ""
    num_samples = 0
    while True:
        key = cv2.waitKey(wait_for_frame)

        # Stop capturing using ESC
        if (key & 255) == 27:
            break

        # Pause capturing using 'p'
        if key == ord("p"):
            cv2.waitKey(-1)

        # Read next frame
        _, frame = camera.read()
        if frame is None:
            print("End of stream")
            break

        # Resize
        height, width = frame.shape[:2]
        if width < 640:
            s = 640.0 / width
            frame = cv2.resize(frame, (int(s * width), int(s * height)))

        # Flip for webcam
        if args.video == "none":
            frame = cv2.flip(frame, 1)

        # ======== Exercise 5.1: Face Tracking (with alignment) ============
        if (face := detector.track_face(frame)) is not None:
            num_samples += 1

            if args.mode == ReIdMode.IDENT:
                recognizer.partial_fit(face["aligned"], args.label)
                state = f"{args.label} ({num_samples} samples)"
            elif args.mode == ReIdMode.CLUSTER:
                clustering.partial_fit(face["aligned"])
                state = f"{num_samples} samples"

            # Draw rectangle and info
            x, y, w, h = face["rect"]
            response = face["response"]
            cv2.rectangle(frame, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)

            label_text = f"{state}, resp={response:.2f}"
            ((tw, th), _) = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                frame,
                (x - 1, y + h),
                (x + 1 + tw, y + h + th + 4),
                (0, 255, 0),
                -1,
            )
            cv2.putText(
                frame,
                label_text,
                (x, y + h + th),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

            # Optional: show aligned face
            if key == ord("a"):
                cv2.imshow("Aligned Face", face["aligned"])

        # Show annotated frame
        cv2.imshow("Camera", frame)

    # Save trained models
    if args.mode == ReIdMode.IDENT:
        print("Save trained face recognition model")
        recognizer.save()
    if args.mode == ReIdMode.CLUSTER:
        print("Save trained face clustering")
        clustering.fit(visualize=True, random_seed=42)
        clustering.save()


def arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=ReIdMode,
        choices=enum_choices(ReIdMode),
        default=ReIdMode.IDENT,
        help="The training mode.",
    )

    parser.add_argument(
        "--video",
        type=str,
        default=Config.TRAIN_DATA.joinpath("Alan_Ball", "%04d.jpg"),
        help="The video capture input. In case of 'none' the default video capture (webcam) is "
             "used. Use a filename(s) to read video data from image file (see VideoCapture "
             "documentation).",
    )

    parser.add_argument(
        "--label",
        type=str,
        default="Alan_Ball",
        help="Identity label (only required for face identification).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(arguments())