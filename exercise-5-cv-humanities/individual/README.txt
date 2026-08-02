Computer Vision Project – Exercise 3.2 (Bonus Detection Pipeline)
Summer Term 2025
Author: Saranya Bhattacharjee (aj81owid)

-------------------------------------------------------------
HOW TO RUN THE PIPELINE (Step by Step)
-------------------------------------------------------------

This code implements a full object detection pipeline based on Selective Search and a linear SVM, as required by Exercise 3.2.  
It is organized into modular scripts, each of which should be run separately, in order:

1. ----------- Generate Region Proposals -----------
    Script: code/detection_generate_proposals.py

    - This script generates Selective Search region proposals for every image in the balloon dataset (train/valid/test).
    - Proposals are saved as .npy files in results/proposals/
    - To run:
        python code/detection_generate_proposals.py

2. ----------- Label Proposals (Positive/Negative) -----------
    Script: code/detection_label_proposals.py

    - Loads proposals and COCO annotations, computes IoU, labels as positive (IoU ≥ 0.75), negative (IoU ≤ 0.25), or ignores ambiguous regions.
    - Output: .npz files per image in results/labeled_proposals/
    - To run:
        python code/detection_label_proposals.py

3. ----------- Feature Extraction -----------
    Script: code/detection_extract_features.py

    - For every positive and negative proposal, extracts a feature vector using a pre-trained ResNet18 CNN.
    - Features (X) and labels (y) are saved for each split in results/features/
    - To run:
        python code/detection_extract_features.py

4. ----------- SVM Training -----------
    Script: code/detection_train_svm.py

    - Trains a linear SVM with class weighting to handle imbalance.
    - Saves the trained model as results/svm_model.joblib.
    - To run:
        python code/detection_train_svm.py

5. ----------- Inference and Visualization -----------
    Script: code/detection_inference_and_visualize.py

    - Classifies proposals in the test set using the trained SVM.
    - Applies non-maximum suppression to remove overlapping boxes.
    - Draws final detections on each test image and saves to results/detections/.
    - To run:
        python code/detection_inference_and_visualize.py

-------------------------------------------------------------
NOTES AND LIMITATIONS
-------------------------------------------------------------

- The approach is based on classical Selective Search proposals + SVM detection with generic CNN features.
- On this small balloon dataset, the pipeline detects some balloons but misses others.
    - **Main limitations:**
        - **Proposal coverage:** Selective Search may not generate a region matching every balloon, so some objects can't be detected even with perfect classification.
        - **Class imbalance:** There are thousands of background proposals, but only a handful of positive/balloon proposals, making SVM training difficult.
        - **Feature limitations:** The CNN is pre-trained on generic images, not specifically on balloons, so features are not fully optimized for this task.
- **Non-maximum suppression** is used to remove duplicate detections per object, but if proposal regions heavily overlap, sometimes true positives may also be suppressed.
- Due to the above, **100% detection is not achievable with this classical approach** on a small, imbalanced dataset using generic CNN features.

-------------------------------------------------------------
FILES AND STRUCTURE
-------------------------------------------------------------

- code/:      All scripts for proposals, labeling, feature extraction, training, and inference
- data/:      Balloon dataset (COCO format, split into train, valid, test)
- results/:   Outputs for proposals, features, labels, detections, and model

-------------------------------------------------------------
HOW TO REPRODUCE (Environment)
-------------------------------------------------------------

- All code tested on Python 3.9+ (using a conda environment).
- Main dependencies: numpy, scikit-learn, torch, torchvision, Pillow, scikit-image

-------------------------------------------------------------

For questions or a demonstration, please contact me :)).

-------------------------------------------------------------
