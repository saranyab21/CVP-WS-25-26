# Exercise 2 — Demosaicing & HDR

Reconstructing full-color images from a raw camera sensor's **Bayer mosaic**, analyzing the
camera's **radiometric response**, and building **High Dynamic Range (HDR)** images from
bracketed exposures — implemented from scratch in **NumPy**.

| Track      | Location                     | Focus                                                    |
| ---------- | ---------------------------- | -------------------------------------------------------- |
| Group      | [`group/`](group/)           | Demosaicing, white balance, gamma/linearity, iCAM06 tone mapping |
| Individual | [`individual/`](individual/) | HDR from JPEGs via Debevec–Malik camera-response estimation |

---

## Problem

A digital camera sensor doesn't capture full color per pixel — each photosite sits under a
single red, green, or blue filter arranged in a **Bayer pattern** (RGGB). Turning that raw
single-channel mosaic into a viewable color image, and then compressing a high-dynamic-range
scene down to a displayable 8-bit image, is the imaging pipeline this exercise reconstructs.

## Group solution

Implemented in [`group/demosaicing_hdr.py`](group/demosaicing_hdr.py) and
[`group/icam06_tonemap.py`](group/icam06_tonemap.py):

1. **Bayer-pattern demosaicing** — read the raw sensor data (`rawpy`), build per-channel
   masks for the RGGB layout, and interpolate the missing color samples by convolution to
   recover a full RGB image.
2. **White balance** — gray-world correction to neutralize color casts.
3. **Radiometric analysis** — measure the camera's **linearity** and estimate the **gamma**
   curve, with diagnostic plots of the response.
4. **Tone mapping** — a **logarithmic** operator swept across base/scale parameters, plus a
   from-scratch implementation of **iCAM06** for perceptually-informed HDR tone mapping.

### Selected results

Diagnostic figures and representative outputs are in [`results/`](results/):

- `bayer_pattern_region.png` — the raw RGGB mosaic up close
- `demosaiced_image.png`, `demosaiced_image_grayworld_wb2.png` — before/after white balance
- `linearity_plot.png`, `gamma_curves.png`, `gamma_corrected_images_comparison.png` — radiometric response
- `log_correction_curves.png` + two representative `log_corrected_*` tone-mapped outputs
- `icam06_output.png` — iCAM06 tone-mapping result

> Full-resolution renders were downscaled for the repository; the code regenerates them at
> native resolution from the raw data.

## Individual extension — HDR from JPEG

Code and write-up in [`individual/`](individual/). Builds an HDR image from **12 bracketed
JPEG exposures** using the classic **Debevec & Malik** method:

1. Estimate the **camera response function (CRF)** per channel by solving the Debevec–Malik
   least-squares system (`gsolve`) on sampled non-saturated pixels.
2. **Linearize** each exposure via the recovered CRF and known exposure times.
3. **Merge** into a radiance map with a weighted average in the log domain.
4. **Tone map** to a displayable image via log compression.

Implemented with **only NumPy + PIL** (per the assignment constraint). Result:
[`individual/hdr_result.jpg`](individual/hdr_result.jpg); full method notes and known
limitations in [`individual/README.txt`](individual/README.txt).

## How to run

```bash
# from the repository root
pip install -r requirements.txt   # adds rawpy, opencv-python, Pillow

cd exercise-2-demosaicing-hdr/group
python demosaicing_hdr.py         # expects raw files under ./data/ (see data/README.md)
```

Scripts reference raw data under a relative `data/` path — adjust the paths near the bottom
of each script to point at your local copy.

## Stack

`numpy` · `rawpy` (raw sensor decoding) · `scipy.ndimage` (convolution) · `opencv-python` ·
`Pillow` · `matplotlib`. All demosaicing, CRF estimation, and tone-mapping logic is hand-
written — no library demosaic/HDR calls.

## Files

```
exercise-2-demosaicing-hdr/
├── group/
│   ├── demosaicing_hdr.py     # demosaicing + WB + gamma + log tone mapping
│   └── icam06_tonemap.py      # iCAM06 tone mapping
├── individual/
│   ├── hdr_from_jpg.py        # Debevec–Malik HDR from JPEGs (NumPy only)
│   ├── hdr_result.jpg         # bonus output
│   └── README.txt             # method notes + limitations
├── results/                   # diagnostic plots & representative renders
├── data/README.md             # how to obtain the raw data
└── exercise-02.pdf            # task sheet
```
