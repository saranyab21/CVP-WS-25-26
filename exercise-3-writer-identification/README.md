# Exercise 3 — Writer Identification & Retrieval

Identifying the **writer** of a historical handwritten document and **retrieving** other
documents by the same hand, on the **ICDAR 2017 Historical-WI** dataset. The pipeline
encodes local **SIFT** descriptors into a global document descriptor and ranks documents
by similarity, evaluated with retrieval metrics.

| Track      | Location                              | Focus                                             |
| ---------- | ------------------------------------- | ------------------------------------------------- |
| Group      | [`group/`](group/)                    | SIFT → VLAD encoding → retrieval + classification |
| Individual | [`individual/`](individual/)          | Extended encodings on the original (color) images |

---

## Problem

Given a large collection of scanned historical manuscript pages — each written by one of
many writers — the task is twofold:

- **Identification (classification):** predict the writer of a query page.
- **Retrieval:** rank all other pages by how likely they share the query's writer, and
  measure ranking quality with **mean Average Precision (mAP)**.

The course provides pre-extracted **SIFT local descriptors** (one `.pkl.gz` per image)
plus writer-label files for train/test. The challenge is turning thousands of variable-
count local descriptors per page into a single fixed-length, discriminative descriptor.

## Group solution — VLAD pipeline

Implemented in [`group/writer_identification.py`](group/writer_identification.py) (built on
the course skeleton, preserved as [`skeleton.py`](group/skeleton.py)):

1. **Load** per-image SIFT descriptors from the label files.
2. **Vocabulary:** cluster a descriptor sample with **MiniBatch k-means** to build a
   visual vocabulary.
3. **VLAD encoding:** aggregate residuals between descriptors and their nearest cluster
   centers into a single global vector per document.
4. **Normalization:** **power normalization** (signed square-root) followed by
   L2-normalization; optional **Generalized Max Pooling (GMP)** with a ridge-regression
   formulation as an alternative aggregation.
5. **Retrieval & classification:** cosine similarity for ranking (→ mAP), plus a
   **Linear SVM** / **Exemplar-SVM (E-SVM)** re-ranking stage for identification.

The notebooks [`extract.ipynb`](group/extract.ipynb) and [`final.ipynb`](group/final.ipynb)
drive feature extraction and the final evaluation.

### Results (binarized dataset)

| Method                    | Top-1 accuracy | mAP    |
| ------------------------- | -------------- | ------ |
| VLAD + power-norm         | 0.821          | 0.625  |
| + Exemplar-SVM re-ranking | **0.887**      | **0.749** |

Exemplar-SVM re-ranking improved retrieval by **+0.12 mAP** over the VLAD baseline.

## Individual extension

Code in [`individual/`](individual/). The bonus task re-runs the pipeline on the **original
(color) ICDAR17 images** rather than the provided binarized descriptors — extracting SIFT
features directly and applying Hellinger (RootSIFT) normalization before VLAD encoding — and
compares the effect against the binarized baseline. See
[`individual/results.txt`](individual/results.txt) for the full write-up and numbers.

## How to run

```bash
# from the repository root
pip install -r requirements.txt   # adds opencv-python, scikit-learn, tqdm, joblib

cd exercise-3-writer-identification/group
python writer_identification.py --labels_train <train.txt> --labels_test <test.txt> \
    --in_train <features/train> --in_test <features/test> --powernorm
```

Key flags: `--powernorm` (power normalization), `--gmp --gamma <g>` (generalized max
pooling), `--C <c>` (SVM regularization), `--overwrite` (recompute encodings instead of
loading cached ones).

The dataset and pre-computed encodings are **not** included — see
[`data/README.md`](data/README.md).

## Stack

`numpy` · `opencv-python` (SIFT) · `scikit-learn` (MiniBatchKMeans, LinearSVC, Ridge, PCA,
`average_precision_score`) · `joblib` / `multiprocessing` (parallel encoding) · `tqdm`.

## Files

```
exercise-3-writer-identification/
├── group/
│   ├── writer_identification.py   # main VLAD pipeline
│   ├── skeleton.py                # course-provided skeleton (reference)
│   ├── skeleton_original.py
│   ├── parmap.py                  # parallel-map helper
│   ├── test.py
│   ├── extract.ipynb              # feature extraction
│   └── final.ipynb                # final evaluation
├── individual/
│   ├── writer_identification_bonus.py   # color-image extension
│   ├── test_label.py
│   ├── color_labels_{train,test}.txt
│   └── results.txt                # bonus write-up + metrics
├── results/                       # result figures & run logs
├── data/README.md                 # how to obtain the dataset
└── exercise-03.pdf                # task sheet
```
