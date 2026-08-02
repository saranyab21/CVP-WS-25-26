import argparse
import cv2
import numpy as np

from config import Config, ReIdMode, enum_choices
from face_detector import FaceDetector
from face_recognition import FaceClustering, FaceRecognizer

from dotenv import load_dotenv
load_dotenv()

def main(args):
    if args.video == "none":
        camera = cv2.VideoCapture(-1)
        wait_for_frame = 200
    else:
        camera = cv2.VideoCapture(args.video)
        wait_for_frame = 100
    camera.set(3, 640)
    camera.set(4, 480)

    cv2.namedWindow("Camera")
    cv2.moveWindow("Camera", 0, 0)

    detector = FaceDetector()
    recognizer = FaceRecognizer()
    clustering = FaceClustering()

    # --- For clustering: count how many faces have been added
    faces_added = 0
    clustering_fitted = False

    on_track = False
    while True:
        key = cv2.waitKey(wait_for_frame)
        if (key & 255) == 27:
            break
        if key == ord("p"):
            cv2.waitKey(-1)

        _, frame = camera.read()
        if frame is None:
            print("End of stream")
            break

        height, width = frame.shape[:2]
        if width < 640:
            s = 640.0 / width
            frame = cv2.resize(frame, (int(s * width), int(s * height)))
        if args.video == "none":
            frame = cv2.flip(frame, 1)

        face = detector.track_face(frame)

        label_str = ""
        confidence_str = ""
        state_str = ""
        predicted_label = ""

        # --- Collect embeddings for clustering
        if face is not None and args.mode == ReIdMode.CLUSTER:
            if not clustering_fitted:
                clustering.partial_fit(face["aligned"])
                faces_added += 1
                # Fit clustering after enough faces have been added (e.g., after 10)
                if faces_added == 10:
                    clustering.fit(visualize=True, random_seed=42)
                    clustering_fitted = True  # Only fit once

        # Make prediction continuously when a face is visible
        if face is not None:
            on_track = True
            if args.mode == ReIdMode.IDENT:
                predicted_label, prob, dist_to_prediction = recognizer.predict(face["aligned"])
                label_str = "{}".format(predicted_label)
                confidence_str = "Prob.: {:1.2f}, Dist.: {:1.2f}".format(prob, dist_to_prediction)
            elif args.mode == ReIdMode.CLUSTER and clustering_fitted:
                # Only predict if clustering has been fitted
                predicted_label, distances_to_clusters = clustering.predict(face["aligned"])
                label_str = f"Cluster {predicted_label}"

                if isinstance(distances_to_clusters, np.ndarray):
                    min_dist = np.min(distances_to_clusters)
                    max_dist = np.max(distances_to_clusters)
                    confidence_str = f"MinDist: {min_dist:.2f}, MaxDist: {max_dist:.2f}"
                else:
                    confidence_str = ""
            elif args.mode == ReIdMode.CLUSTER and not clustering_fitted:
                label_str = "Clustering..."
                confidence_str = ""

            state_str = "{} | {}".format(label_str, confidence_str)
        else:
            on_track = False

        if face is not None:
            face_rect = face["rect"]
            color = (0, 255, 0)
            if isinstance(predicted_label, str) and predicted_label.lower() == "unknown":
                color = (0, 0, 255)
            cv2.rectangle(
                frame,
                (face_rect[0], face_rect[1]),
                (face_rect[0] + face_rect[2] - 1, face_rect[1] + face_rect[3] - 1),
                color,
                2,
            )
            ((tw, th), _) = cv2.getTextSize(state_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                frame,
                (face_rect[0] - 1, face_rect[1] + face_rect[3]),
                (face_rect[0] + 1 + tw, face_rect[1] + face_rect[3] + th + 4),
                color,
                -1,
            )
            cv2.putText(
                frame,
                state_str,
                (face_rect[0], face_rect[1] + face_rect[3] + th),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

        cv2.imshow("Camera", frame)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=ReIdMode,
        choices=enum_choices(ReIdMode),
        default=ReIdMode.IDENT,
        help="The test mode.",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=Config.TEST_DATA.joinpath("Alan_Ball", "%04d.jpg"),
        help="The video capture input. In case of 'none' the default video capture (webcam) is "
        "used. Use a filename(s) to read video data from image file (see VideoCapture "
        "documentation).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(arguments())

