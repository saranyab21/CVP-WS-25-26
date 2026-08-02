import os
import json
import numpy as np

# ==== CONFIGURATION ====
TRAIN_DIR = "../data/balloon_dataset/train"
VALID_DIR = "../data/balloon_dataset/valid"
PROPOSALS_DIR = "../results/proposals"
LABELED_DIR = "../results/labeled_proposals"
TRAIN_JSON = "../data/balloon_dataset/train/_annotations.coco.json"
VALID_JSON = "../data/balloon_dataset/valid/_annotations.coco.json"

TP = 0.75   # Positive IoU threshold
TN = 0.25   # Negative IoU threshold

os.makedirs(LABELED_DIR, exist_ok=True)

# ==== STEP 2 FUNCTIONS ====

def parse_coco_annotations(json_path):
    with open(json_path, 'r') as f:
        coco = json.load(f)
    # Map image id to filename
    id_to_filename = {img['id']: img['file_name'] for img in coco['images']}
    filename_to_boxes = {fn: [] for fn in id_to_filename.values()}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        bbox = ann['bbox']  # [x, y, width, height]
        fn = id_to_filename[img_id]
        filename_to_boxes[fn].append(bbox)
    return filename_to_boxes

def compute_iou(boxA, boxB):
    # Boxes: [x, y, w, h]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    denom = boxAArea + boxBArea - interArea
    if denom <= 0:
        return 0.0
    iou = interArea / denom
    return iou

def label_proposals_for_image(proposals, gt_boxes, tp=0.75, tn=0.25):
    labels = []
    for prop in proposals:
        ious = [compute_iou(prop, gt) for gt in gt_boxes]
        max_iou = max(ious) if ious else 0
        if max_iou >= tp:
            labels.append(1)   # Positive (balloon)
        elif max_iou <= tn:
            labels.append(0)   # Negative (background)
        else:
            labels.append(-1)  # Ignore (ambiguous)
    return np.array(labels)

def process_and_label_dataset(split_name, img_dir, coco_json, proposals_dir, out_dir, tp=0.75, tn=0.25):
    filename_to_boxes = parse_coco_annotations(coco_json)
    img_list = [f for f in os.listdir(img_dir) if f.lower().endswith('.jpg')]
    for fname in img_list:
        proposals_path = os.path.join(proposals_dir, f"{split_name}_{fname.replace('.jpg','')}_proposals.npy")
        if not os.path.exists(proposals_path):
            print(f"Warning: No proposals for {fname}")
            continue
        proposals = np.load(proposals_path)
        gt_boxes = filename_to_boxes.get(fname, [])
        labels = label_proposals_for_image(proposals, gt_boxes, tp, tn)
        # Save: proposals, labels
        out_path = os.path.join(out_dir, f"{split_name}_{fname.replace('.jpg','')}_labeled.npz")
        np.savez_compressed(out_path, proposals=proposals, labels=labels)
        print(f"{fname}: {np.sum(labels==1)} positive, {np.sum(labels==0)} negative, {np.sum(labels==-1)} ignored --> {out_path}")

if __name__ == "__main__":
    print("Processing TRAIN split...")
    process_and_label_dataset('train', TRAIN_DIR, TRAIN_JSON, PROPOSALS_DIR, LABELED_DIR, TP, TN)
    print("Processing VALID split...")
    process_and_label_dataset('valid', VALID_DIR, VALID_JSON, PROPOSALS_DIR, LABELED_DIR, TP, TN)
    print("Done: labeled proposals created and saved for all train/valid images.")
