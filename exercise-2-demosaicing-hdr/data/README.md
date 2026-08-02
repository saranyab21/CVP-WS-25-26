# Data — Exercise 2 (Demosaicing & HDR)

The raw image data for this exercise is **course-provided** and **not redistributed** here
(raw `.CR3` / `.npy` sensor files and bracketed exposure sets are large).

## What you need

- **Raw sensor files** for demosaicing and tone mapping (Canon `.CR3`, plus a `.npy` dump
  used for Bayer-pattern inspection).
- **HDR exposure stack** — a set of bracketed shots of the same scene at different exposure
  times (for the group HDR task).
- **Bonus exposure set** — 12 JPEG images at different exposures (for the individual HDR
  task).

## How to obtain

Download the exercise data from the course's StudOn page (*Computer Vision Project →
Demosaicing / Exercise 2 materials*) and arrange it like this:

```
exercise-2-demosaicing-hdr/data/
├── exercise_4_data/          # raw sensor captures (as provided)
│   ├── 01/ … 05/
│   └── HDR/
└── ex4_additional_exercise_data/   # 12 bracketed JPEGs for the bonus HDR task
```

Then adjust the file paths near the bottom of each script (they use relative `data/…`
paths by default) to match your layout.

> All raw data and generated full-resolution renders are git-ignored and will never be
> pushed. The result images committed here are downscaled previews only.
