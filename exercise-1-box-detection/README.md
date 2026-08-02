# Exercise 1 — Box Detection from Time-of-Flight Data

Estimating the 3D dimensions of a box from a single Time-of-Flight (ToF) capture,
using a **from-scratch RANSAC** plane-fitting pipeline, plus two advanced robust
estimators implemented as individual extensions: **MLESAC** and **Preemptive RANSAC**.

| Track          | Notebook                                                   | Focus                                             |
| -------------- | ---------------------------------------------------------- | ------------------------------------------------- |
| Group          | [`group/box_detection.ipynb`](group/box_detection.ipynb)   | RANSAC plane detection & box measurement          |
| Individual     | [`individual/mlesac_preemptive_ransac.ipynb`](individual/mlesac_preemptive_ransac.ipynb) | MLESAC + Preemptive RANSAC extensions |

---

## Problem

The input is a `.mat` file containing three **registered** representations of the
same scene captured by a ToF camera:

- an **amplitude image** `A`,
- a **distance image** `D`,
- a **point cloud** `PC` (a 3D coordinate per pixel).

The goal is to recover the physical **height, length, and width** of a box sitting
on a floor — robustly, despite sensor noise. Naively subtracting distance
measurements is far too noise-sensitive, so the pipeline instead fits **planes** to
the floor and the box top and reasons geometrically from there.

## Group solution — RANSAC pipeline

The group notebook implements the full measurement pipeline:

1. **Load & visualize** the amplitude image, distance image, and a subsampled 3D
   scatter of the point cloud (`scatter3D`).
2. **Filter** the point cloud to remove invalid / zero-distance returns.
3. **Plane fitting with a hand-written RANSAC** (no `scikit-learn` estimator) using
   a normal/offset plane parameterization `n·x = d`. The estimator takes a point
   cloud, an inlier threshold, and a max-iteration budget, and returns the
   best-supported plane.
4. **Floor detection** → binary inlier mask; **box-top detection** → the second
   dominant plane.
5. **Mask cleanup** with morphological opening/closing and **largest-connected-
   component** selection (`scipy.ndimage.label`) to isolate the true box top.
6. **Measurement**: box **height** from the distance between the two parallel
   planes; **length & width** from the 3D coordinates of the detected box corners.

The notebook also includes an **elbow-plot analysis** of the inlier threshold,
concluding that a threshold of ≈ 0.02 balances inlier count against noise
(higher values like 0.04–0.05 show diminishing returns), and a written
**discussion** of the pipeline's weaknesses and how to make it more robust,
accurate, or faster.

## Individual extensions

Two more sophisticated consensus estimators, implemented from scratch and
benchmarked against the baseline RANSAC.

### MLESAC — Maximum Likelihood Estimation Sample Consensus

Where plain RANSAC scores a hypothesis by *counting* inliers, **MLESAC** maximizes
the likelihood of the data under a mixture (inliers ~ Gaussian error, outliers ~
uniform). In this implementation, points within the inlier threshold ε contribute
their raw residual, while points beyond it are penalized by a fixed constant γ
(chosen slightly above ε). This "soft" penalty discourages large residuals instead
of discarding points outright, giving a more nuanced cost than a hard inlier count.

### Preemptive RANSAC (Nistér)

Targets the real-time / time-constrained setting. Instead of evaluating each
hypothesis against the full dataset, it:

1. generates a fixed number **M** of hypotheses up front,
2. evaluates them in **B-sized chunks** of the data,
3. after each chunk, reorders hypotheses by cost and keeps only the top
   `f(i) = ⌊M · 2^(−⌊i/B⌋)⌋` of them,

repeating until a single hypothesis survives. The notebook compares runtime and
quality across different values of **M** and visualizes the resulting floor
segmentation masks.

## How to run

```bash
# from the repository root
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook exercise-1-box-detection/group/box_detection.ipynb
```

Place the course-provided `.mat` files in `data/` first — see
[`data/README.md`](data/README.md).

## Stack

`numpy` · `scipy` (`ndimage`, `io`) · `matplotlib` (incl. `mplot3d`) — all robust
estimators are implemented from scratch, without library RANSAC.

## Files

```
exercise-1-box-detection/
├── group/box_detection.ipynb                  # group solution (RANSAC pipeline)
├── individual/mlesac_preemptive_ransac.ipynb  # individual extensions
├── data/README.md                             # how to obtain the datasets
├── exercise-01.pdf                            # group task sheet
└── exercise-01-individual.pdf                 # individual task sheet
```
