# Exercise 5 — Computer Vision in the Humanities

Object detection for cultural-heritage imagery, built on **Selective Search**. The group
task implements Selective Search region proposals from scratch on art-historical images; the
individual task extends it into a **full detection pipeline** (proposals → CNN features →
SVM → non-maximum suppression).

> **Numbering note:** the course task sheet labels this "Sheet 3", but the exercise itself is
> numbered **5.1 / 5.2** and unzips to `exercise-5/` — it corresponds to **Lecture 5:
> Computer Vision in the Humanities**.

| Track      | Location                     | Focus                                                       |
| ---------- | ---------------------------- | ----------------------------------------------------------- |
| Group      | [`group/`](group/)           | Selective Search region proposals (Felzenszwalb + merging)  |
| Individual | [`individual/`](individual/) | End-to-end detection pipeline: SS → ResNet18 → SVM → NMS     |

---

## Group solution — Selective Search

Implemented in [`group/selective_search.py`](group/selective_search.py), driven by
[`group/main.py`](group/main.py):

1. **Initial regions** from the **Felzenszwalb** graph-based segmentation
   (`skimage.segmentation.felzenszwalb`).
2. **Region features** — color (HSV histograms) and texture (**Local Binary Patterns**).
3. **Hierarchical grouping** — iteratively merge the most similar adjacent regions, building
   up from fine segments to object-sized proposals.
4. **Proposal filtering** — remove duplicates, drop regions below a minimum size, and reject
   extreme aspect ratios to keep object-like boxes.

Tested on art-historical images (Christian art, classical archaeology, art history). Sample
outputs and the region visualization are in [`results/`](results/); the written answers and
a short write-up are in [`group/answers.txt`](group/answers.txt) and
[`group/selective_search_writeup.docx`](group/selective_search_writeup.docx).

## Individual extension — detection pipeline

A complete classical object-detection pipeline on the **balloon** dataset (COCO format),
documented in [`individual/README.txt`](individual/README.txt). Five modular stages run in
order:

1. **Generate proposals** — Selective Search over every image
   ([`detection_generate_proposals.py`](individual/detection_generate_proposals.py)).
2. **Label proposals** — compute IoU against COCO ground truth; label positive (IoU ≥ 0.75),
   negative (IoU ≤ 0.25), ignore the ambiguous middle
   ([`detection_label_proposals.py`](individual/detection_label_proposals.py)).
3. **Extract features** — a pre-trained **ResNet18** as a feature extractor over each
   proposal ([`detection_extract_features.py`](individual/detection_extract_features.py)).
4. **Train SVM** — a linear SVM with class weighting to counter heavy background/object
   imbalance ([`detection_train_svm.py`](individual/detection_train_svm.py)).
5. **Inference + NMS** — classify test proposals, apply non-maximum suppression, and draw
   final boxes ([`detection_inference_and_visualize.py`](individual/detection_inference_and_visualize.py)).

The write-up discusses the design honestly: why **two IoU thresholds** give cleaner training
data than one, how the small imbalanced dataset limits recall, and augmentation strategies
that would help — see [`individual/answers.txt`](individual/answers.txt). Sample detections
are in [`results/detections/`](results/detections/).

## How to run

```bash
# from the repository root
pip install -r requirements.txt   # adds torch, torchvision, scikit-image

# Group: Selective Search on the art images
python exercise-5-cv-humanities/group/main.py

# Individual: run the 5 stages in order (see individual/README.txt)
cd exercise-5-cv-humanities/individual
python detection_generate_proposals.py
python detection_label_proposals.py
python detection_extract_features.py
python detection_train_svm.py
python detection_inference_and_visualize.py
```

Datasets go under `data/` — see [`data/README.md`](data/README.md).

## Stack

`numpy` · `scikit-image` (felzenszwalb, LBP) · `torch` / `torchvision` (ResNet18) ·
`scikit-learn` (SVC, class weighting, metrics) · `Pillow` · `matplotlib` · `joblib`.

## Files

```
exercise-5-cv-humanities/
├── group/
│   ├── selective_search.py        # Selective Search implementation
│   ├── main.py                    # driver + proposal filtering
│   ├── answers.txt                # question answers
│   └── selective_search_writeup.docx
├── individual/
│   ├── detection_generate_proposals.py
│   ├── detection_label_proposals.py
│   ├── detection_extract_features.py
│   ├── detection_train_svm.py
│   ├── detection_inference_and_visualize.py
│   ├── README.txt                 # step-by-step pipeline guide
│   └── answers.txt                # question answers
├── results/
│   ├── detections/                # balloon detection outputs
│   ├── selective_search_regions.png
│   └── *.jpg                       # SS proposals on art images
├── data/README.md                  # how to obtain the datasets
└── exercise-05.pdf                 # task sheet
```
