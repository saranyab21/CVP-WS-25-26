# Computer Vision Project — FAU (Winter 25/26, 10 ECTS)

A collection of computer-vision exercises completed as part of the **Project Computer
Vision** course at Friedrich-Alexander-Universität Erlangen-Nürnberg (Pattern
Recognition Lab, INF 5). This repository covers the **full 10 ECTS track** — both the
**group exercises** and the additional **individual exercises**, where every classical
method is **implemented from scratch** rather than called from a library.

> **Scope of the 10 ECTS option:** the standard course is 5 ECTS (lectures + group
> exercises). The 10 ECTS option requires completing *all* group **and** individual
> exercises, with the individual work done independently and without relying on
> existing library implementations of the core algorithms.

---

## Exercises

| # | Exercise | Group | Individual extension | Core techniques |
|---|----------|-------|----------------------|-----------------|
| **1** | [**Box Detection**](exercise-1-box-detection/) — 3D box measurement from ToF data | ✅ | ✅ MLESAC · Preemptive RANSAC | RANSAC plane fitting, point clouds, morphology |
| **2** | Demosaicing & HDR | ⬜ | ⬜ | Bayer interpolation, tone mapping, exposure fusion |
| **3** | Writer Identification & Retrieval | ⬜ | ⬜ | Feature descriptors, encoding, retrieval metrics |
| **4** | Face Recognition | ⬜ | ⬜ | Eigenfaces / embeddings, classification |
| **5** | Computer Vision in the Humanities | ⬜ | ⬜ | Applied CV on cultural-heritage data |

*(✅ = included · ⬜ = to be added. This table is updated as each exercise is uploaded.)*

---

## 1 · Box Detection from Time-of-Flight Data ✅

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

## 2 · Demosaicing & HDR ⬜

Reconstruct full-color images from a raw Bayer mosaic and build high-dynamic-range
images from bracketed exposures. *(Coming soon.)*

## 3 · Writer Identification & Retrieval ⬜

Identify the writer of a handwriting sample and retrieve documents by the same hand.
*(Coming soon.)*

## 4 · Face Recognition ⬜

Detect and recognize faces from image data. *(Coming soon.)*

## 5 · Computer Vision in the Humanities ⬜

Apply computer-vision methods to cultural-heritage / humanities data. *(Coming soon.)*

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
├── exercise-2-demosaicing-hdr/      (coming soon)
├── exercise-3-writer-identification/(coming soon)
├── exercise-4-face-recognition/     (coming soon)
└── exercise-5-cv-humanities/        (coming soon)
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
