# Computer Vision Project — FAU (Winter 25/26, 10 ECTS)

A collection of computer-vision exercises completed as part of the **Project Computer
Vision** course at FAU (Pattern
Recognition Lab, INF 5). This repository covers the **full 10 ECTS track**, where every classical
method is **implemented from scratch** rather than called from a library.

> **Scope of the 10 ECTS option:** the standard course is 5 ECTS (lectures + group
> exercises). The 10 ECTS option requires completing *all* group **and** individual
> exercises, with the individual work done independently and without relying on
> existing library implementations of the core algorithms.

---

## Exercises

| # | Exercise | Individual extension | Core techniques |
|---|----------|----------------------|-----------------|
| **1** | [**Box Detection**](exercise-1-box-detection/) — 3D box measurement from ToF data | MLESAC · Preemptive RANSAC | RANSAC plane fitting, point clouds, morphology |
| **2** | [**Demosaicing & HDR**](exercise-2-demosaicing-hdr/) — raw imaging pipeline | HDR from JPEG (Debevec–Malik) | Bayer demosaicing, white balance, gamma, iCAM06, CRF |
| **3** | [**Writer Identification & Retrieval**](exercise-3-writer-identification/) — ICDAR17 historical manuscripts | Color-image SIFT + RootSIFT | SIFT, VLAD encoding, power-norm/GMP, E-SVM, mAP |
| **4** | [**Face Recognition**](exercise-4-face-recognition/) — video-based recognition system | Dual-embedding + open-set challenge (SPL/MPL) | MTCNN, template tracking, FaceNet, k-NN, k-means, DIR, OSR |
| **5** | [**Computer Vision in the Humanities**](exercise-5-cv-humanities/) — detection on cultural-heritage imagery | SS→CNN→SVM detection pipeline | Selective Search, Felzenszwalb, LBP, ResNet18, SVM, NMS |

Each exercise includes both the **group** solution and the **individual** extension required
for the 10 ECTS track.

---

## 1 · Box Detection from Time-of-Flight Data

Estimate the physical **height, length, and width** of a box from a single ToF capture
(registered amplitude image, distance image, and point cloud).

- **Group:** a from-scratch **RANSAC** pipeline that fits the floor and box-top planes
  (`n·x = d`), cleans the masks with morphological ops and largest-connected-component
  selection, and derives box dimensions from plane geometry and corner coordinates.
  Includes an elbow-plot study of the inlier threshold and a written discussion of the
  pipeline's failure modes.
- **Individual:** two advanced robust estimators built from scratch —
  **MLESAC** (likelihood-based scoring instead of inlier counting) and
  **Preemptive RANSAC** (Nistér's method for time-constrained hypothesis evaluation),
  benchmarked against the baseline.

→ Full write-up: **[exercise-1-box-detection/README.md](exercise-1-box-detection/)**

## 2 · Demosaicing & HDR

Reconstruct full-color images from a raw **Bayer mosaic**, analyze the camera's radiometric
response, and build **HDR** images — all from scratch in NumPy.

- **Group:** Bayer-pattern demosaicing (RGGB, mask-based convolution), gray-world white
  balance, **linearity & gamma** analysis, logarithmic tone mapping, and a from-scratch
  **iCAM06** tone-mapping operator.
- **Individual:** HDR from 12 bracketed **JPEGs** via the **Debevec–Malik** method —
  camera-response-function estimation, radiance-map merging, and log tone mapping, using only
  NumPy + PIL.

→ Full write-up: **[exercise-2-demosaicing-hdr/README.md](exercise-2-demosaicing-hdr/)**

## 3 · Writer Identification & Retrieval

Identify the **writer** of a historical handwritten page and **retrieve** other pages by
the same hand, on the **ICDAR 2017 Historical-WI** dataset.

- **Group:** encode per-image **SIFT** descriptors into a global **VLAD** representation
  (MiniBatch k-means vocabulary, power normalization, optional Generalized Max Pooling),
  then rank by cosine similarity and re-rank with a **Linear / Exemplar-SVM**. Evaluated
  with top-1 accuracy and **mean Average Precision (mAP)**.
- **Individual:** re-run the pipeline on the **original color images** with direct SIFT
  extraction and **RootSIFT (Hellinger)** normalization, comparing against the binarized
  baseline.

| Method                    | Top-1 accuracy | mAP    |
| ------------------------- | -------------- | ------ |
| VLAD + power-norm         | 0.821          | 0.625  |
| + Exemplar-SVM re-ranking | **0.887**      | **0.749** |

→ Full write-up: **[exercise-3-writer-identification/README.md](exercise-3-writer-identification/)**

## 4 · Face Recognition

A complete video-based face recognition system on the YouTube Faces database, spanning
detection through an open-set recognition challenge.

- **Group (5.1–5.4):** MTCNN detection with **template-matching tracking** (+ re-init),
  224×224 alignment, **FaceNet** (ResNet-50/ONNX) 128-D embeddings, from-scratch **k-NN**
  identification (closed- and open-set), from-scratch **k-means** clustering for
  re-identification, and **DIR-curve** evaluation.
- **Individual (5.2 ext. + 5.5):** **dual color+grayscale embeddings** fused for robustness,
  and the **open-set recognition challenge** with **Single/Multi Pseudo Label** (SPL/MPL)
  strategies over known and known-unknown classes.

Selected results: identification stays robust to ±30° rotation and illumination changes;
DIR analysis gives ≈0.71 ID rate at FAR ≤ 1 %. See the write-up for figures.

→ Full write-up: **[exercise-4-face-recognition/README.md](exercise-4-face-recognition/)**

## 5 · Computer Vision in the Humanities

Object detection for cultural-heritage imagery, built on **Selective Search**.

- **Group:** Selective Search region proposals from scratch — **Felzenszwalb** initial
  segmentation, HSV-color + **LBP-texture** region features, hierarchical merging, and
  proposal filtering — applied to art-historical images.
- **Individual:** a complete classical detection pipeline on the balloon dataset —
  Selective Search proposals → **ResNet18** features → class-weighted **SVM** →
  **non-maximum suppression**, with an honest discussion of the small-data / class-imbalance
  limitations.

→ Full write-up: **[exercise-5-cv-humanities/README.md](exercise-5-cv-humanities/)**

---

## Repository structure

```
cv-project-fau/
├── README.md                        ← you are here
├── LICENSE                          ← MIT (own code only)
├── requirements.txt                 ← shared Python dependencies
├── .gitignore
│
├── exercise-1-box-detection/
│   ├── README.md
│   ├── group/box_detection.ipynb
│   ├── individual/mlesac_preemptive_ransac.ipynb
│   ├── data/README.md               ← datasets are course-provided (git-ignored)
│   ├── exercise-01.pdf
│   └── exercise-01-individual.pdf
│
├── exercise-2-demosaicing-hdr/
│   ├── README.md
│   ├── group/          ← demosaicing + WB + gamma + iCAM06
│   ├── individual/     ← HDR-from-JPEG (Debevec–Malik)
│   ├── results/        ← diagnostic plots & renders
│   ├── data/README.md  ← raw data is course-provided (git-ignored)
│   └── exercise-02.pdf
│
├── exercise-3-writer-identification/
│   ├── README.md
│   ├── group/          ← SIFT → VLAD pipeline + notebooks
│   ├── individual/     ← color-image extension + results
│   ├── results/        ← result figures & run logs
│   ├── data/README.md  ← dataset is course-provided (git-ignored)
│   └── exercise-03.pdf
│
├── exercise-4-face-recognition/
│   ├── README.md
│   ├── src/cvproj_exc/  ← detection, tracking, FaceNet, k-NN, k-means, OSR
│   ├── results/         ← robustness, DIR, k-means, tuning evidence
│   ├── data/README.md   ← data + ONNX model are course-provided (git-ignored)
│   └── exercise-04.pdf
│
└── exercise-5-cv-humanities/
    ├── README.md
    ├── group/          ← Selective Search implementation
    ├── individual/     ← SS→CNN→SVM→NMS detection pipeline
    ├── results/        ← detections & SS visualizations
    ├── data/README.md  ← datasets are course/third-party (git-ignored)
    └── exercise-05.pdf
```

Each exercise is self-contained, with its own `README.md` describing the task, method,
and results, and a `group/` + `individual/` split mirroring the 10 ECTS structure.

## Getting started

```bash
git clone https://github.com/<your-username>/cv-project-fau.git
cd cv-project-fau

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook
```

Datasets are **not** included — they are provided by the course. Each exercise's
`data/README.md` explains how to obtain and place the files.

## Tech stack

Python · NumPy · SciPy · Matplotlib · scikit-learn · Jupyter. Later exercises add
image- and deep-learning libraries as noted in their respective READMEs. A guiding
constraint of the course: **core algorithms are implemented from scratch**, not called
from library black boxes.

## A note on academic integrity

This repository is shared publicly **after** completion and grading of the course, as a
personal portfolio. The **exercise sheets** (`exercise-*.pdf`) remain the intellectual
property of the course instructors and FAU and are included only for context. If you
are a current student, please follow your course's policy and do not submit this work
as your own.

## License

Source code and original notebooks are released under the [MIT License](LICENSE).
Course-provided materials (task sheets, datasets, lecture content) are excluded from
that license and remain the property of their respective owners.
