# Data — Exercise 3 (Writer Identification)

The dataset for this exercise is **course-provided** and **not redistributed** here (it is
large — ~8.7 GB of pre-extracted features — and is FAU/ICDAR material).

## What you need

**ICDAR 2017 Historical-WI** local features and label files:

- `icdar17_labels_train.txt`, `icdar17_labels_test.txt` — writer label per image.
- One `.pkl.gz` per image containing its pre-extracted **SIFT** local descriptors
  (suffix `_SIFT_patch_pr.pkl.gz`).

## How to obtain

Download from the link in the task sheet (`exercise-03.pdf`) — the FAUbox
`icdar17_local_features.zip` (~8.7 GB) — or, on the FAU CIP pool, use the shared copy
referenced in the sheet. Then arrange as:

```
exercise-3-writer-identification/data/
└── icdar17_local_features/
    ├── icdar17_labels_train.txt
    ├── icdar17_labels_test.txt
    ├── train/   # per-image *_SIFT_patch_pr.pkl.gz
    └── test/
```

Point the script at these paths with `--labels_train`, `--labels_test`, `--in_train`,
`--in_test`.

## Regenerable artifacts (not committed)

The `*_sift_hellinger.pkl` and `enc_*.pkl.gz` encoding files are **produced by running the
pipeline** and are git-ignored. Delete them any time and rerun with `--overwrite` to
regenerate.

> All dataset files and `.pkl` encodings are git-ignored and will never be pushed.
