# Exercise 4 — Face Recognition

A complete video-based **face recognition system**: detect and track faces, extract deep
embeddings, and perform **identification** (closed- and open-set), **re-identification** via
clustering, **DIR-curve evaluation**, and an **open-set recognition challenge** (SPL / MPL).
Built on the YouTube Faces database.

> **Numbering note:** the course task sheet is "Sheet 5" and the tasks are numbered 5.1–5.5,
> but in the repository this is **Exercise 4 (Face Recognition)**, matching Lecture 4 in the
> course schedule.

| Track      | Exercises covered | Focus                                                            |
| ---------- | ----------------- | ---------------------------------------------------------------- |
| Group      | 5.1 – 5.4         | Detection/tracking, k-NN identification, k-means clustering, DIR |
| Individual | 5.2 (ext.), 5.5   | Dual color+grayscale embeddings, open-set challenge (SPL/MPL)    |

The whole system is one integrated codebase under [`src/cvproj_exc/`](src/cvproj_exc/);
`training.py` and `test.py` are the entry points (identification vs. clustering mode).

---

## Pipeline

### 5.1 · Detection, Tracking & Alignment — [`face_detector.py`](src/cvproj_exc/face_detector.py)

- **Detection** with **MTCNN** (largest-face bounding box).
- **Tracking** via **template matching** (`cv2.matchTemplate` + `minMaxLoc`) in a small
  search window around the last position — fast, run every frame — with **MTCNN
  re-initialization** when the match score drops below a tuned threshold (recovers from
  pose changes / lost tracks).
- **Alignment** cropping/normalizing faces to 224×224 for the feature extractor.

### 5.2 · Identification & Verification — [`face_recognition.py`](src/cvproj_exc/face_recognition.py)

- **FaceNet** (ResNet-50, ONNX via OpenCV DNN) extracts 128-D L2-normalized embeddings.
- **Closed-set k-NN** identification implemented from scratch (no sklearn): brute-force
  pairwise distances, majority vote, **posterior** `p(Cᵢ|x)=kᵢ/k`, and distance to the
  predicted class.
- **Open-set** protocol: threshold-nearest-neighbor rule — reject as *unknown* when distance
  exceeds `τ_d` or posterior falls below `τ_p`.
- **Individual extension:** stores **two embeddings** per face (color BGR + grayscale) and
  fuses both branches for prediction, posterior, and distance — improving robustness to
  illumination changes.

### 5.3 · Clustering / Re-Identification — [`face_recognition.py`](src/cvproj_exc/face_recognition.py) (`FaceClustering`)

- **k-means from scratch** in embedding space (random init, stored centers + labels).
- Re-identification by nearest cluster center, returning the best match and the full
  distance distribution.
- Convergence and initialization sensitivity analyzed across multiple seeds.

### 5.4 · Evaluation — [`evaluation.py`](src/cvproj_exc/evaluation.py), [`dir_curve.py`](src/cvproj_exc/dir_curve.py)

- **DIR (Detection & Identification Rate) curves** for open-set identification: rank-1
  identification rate, false-alarm-rate thresholding via percentiles, and operating-point
  selection.

### 5.5 · Open-Set Recognition Challenge — [`osr_learning.py`](src/cvproj_exc/osr_learning.py)

- **Single Pseudo Label (SPL)** and **Multi Pseudo Label (MPL)** open-set strategies over
  known classes (KCs) and known-unknown classes (KUCs), benchmarked by AUROC, DIR@FAR,
  and balanced rank-1. Hyperparameters tuned in [`tune_osr.py`](src/cvproj_exc/tune_osr.py);
  tests in [`test_osr_learning.py`](src/cvproj_exc/test_osr_learning.py).

## Selected results

Evidence and figures in [`results/`](results/):

- **Robustness** ([`robustness.txt`](results/robustness.txt)) — identification holds under
  ±30° rotation, brightening, and darkening (posterior 1.0); degrades under heavy blur and
  fails detection at extreme pose, as expected.
- **DIR operating points** ([`dir_thresholds.txt`](results/dir_thresholds.txt)) — at
  FAR ≤ 1 %, identification rate ≈ 0.71; requiring ID ≥ 90 % pushes FAR to ≈ 14 %.
- **k-means convergence** across seeds 1/42/99 and a **cluster-count elbow** analysis
  (`kmeans_objective_plot.png`, `num_clusters_elbow.png`).
- **DIR curve** (`dir_curve.png`) and parameter-tuning sweeps for the tracker, recognizer,
  and clustering (`*_tuning.csv`).

## How to run

```bash
pip install -r exercise-4-face-recognition/src/requirements.txt
cd exercise-4-face-recognition/src

# enroll a person (identification gallery)
python cvproj_exc/training.py --mode ident --video ../data/train_data/NAME/%04d.jpg --label NAME
# identify
python cvproj_exc/test.py --mode ident --video ../data/test_data/NAME/%04d.jpg
# clustering / re-identification
python cvproj_exc/test.py --mode cluster --video ../data/test_data/NAME/%04d.jpg
# DIR evaluation curve
python cvproj_exc/dir_curve.py
# open-set challenge tests
python cvproj_exc/test_osr_learning.py
```

Data (YouTube Faces videos, the ResNet-50 ONNX model, evaluation `.pkl`s, challenge CSV) is
**not** included — see [`data/README.md`](data/README.md).

## Stack

`numpy` · `opencv-python` (DNN, template matching) · `mtcnn` · `torch` / `torchvision` ·
`matplotlib`. Core algorithms — k-NN, k-means, open-set decision rules, DIR evaluation — are
implemented **from scratch** (sklearn is used only in the 5.5 challenge backbone, as the task
permits).

## Files

```
exercise-4-face-recognition/
├── src/
│   ├── cvproj_exc/
│   │   ├── face_detector.py       # MTCNN detection + template-matching tracking
│   │   ├── face_recognition.py    # FaceNet, FaceRecognizer (k-NN), FaceClustering (k-means)
│   │   ├── training.py / test.py  # entry points (ident / cluster modes)
│   │   ├── evaluation.py / dir_curve.py   # DIR-curve evaluation
│   │   ├── classifier.py / eval_osr.py    # nearest-neighbor classifier + OSR eval
│   │   ├── osr_learning.py / tune_osr.py  # SPL/MPL challenge + tuning
│   │   ├── test_osr_learning.py   # challenge tests
│   │   ├── config.py / misc.py    # paths + helpers
│   ├── README.txt                 # run commands + chosen parameters
│   └── requirements.txt           # pinned environment (portable)
├── results/                       # evidence: robustness, DIR, k-means, tuning sweeps
├── data/README.md                 # how to obtain data & the ONNX model
└── exercise-04.pdf                # task sheet
```
