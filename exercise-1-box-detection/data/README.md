# Data — Exercise 1

The datasets for this exercise are **provided by the course** (FAU StudOn) and are
**not redistributed** in this repository.

## What you need

Four `.mat` example files. Each file contains one Time-of-Flight (ToF) capture with:

| Variable | Shape          | Description                                             |
| -------- | -------------- | ------------------------------------------------------- |
| `A`      | `H × W × …`    | Registered **amplitude** image                          |
| `D`      | `H × W × …`    | Registered **distance** image                           |
| `PC`     | `H × W × 3`    | Registered **point cloud** (3D coordinates per pixel)   |

All three representations are **registered** — pixel `(i, j)` refers to the same
physical point across `A`, `D`, and `PC`.

## How to obtain

1. Download the example `.mat` files from the course's StudOn page
   (*Computer Vision Project → Exercise 1 materials*).
2. Place them in this folder:

   ```
   exercise-1-box-detection/data/
   ├── example2kinect.mat     # loaded by the notebooks by default
   └── … (other example .mat files for testing)
   ```

3. The notebooks load `data/example2kinect.mat` by default. Adjust the path in the
   first code cell if your filename differs.

> These files are git-ignored, so they will never be committed or pushed.
