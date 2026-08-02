import os
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image, ImageDraw
import joblib

# === Paths ===
SVM_PATH = "../results/svm_model.joblib"
TEST_DIR = "../data/balloon_dataset/test"
PROPOSALS_DIR = "../results/proposals"
OUT_DIR = "../results/detections"
LABELED_DIR = "../results/labeled_proposals" # for valid/train demos
os.makedirs(OUT_DIR, exist_ok=True)

# === Load Model ===
svm = joblib.load(SVM_PATH)

# === Pretrained CNN (feature extractor) ===
cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
cnn = torch.nn.Sequential(*list(cnn.children())[:-1])
cnn.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn = cnn.to(device)
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def non_max_suppression_fast(boxes, scores, overlapThresh=0.3):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    pick = []

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,0] + boxes[:,2]   # x + w
    y2 = boxes[:,1] + boxes[:,3]   # y + h
    area = (x2 - x1) * (y2 - y1)
    idxs = np.argsort(scores)[::-1]

    while len(idxs) > 0:
        i = idxs[0]  # index of the current best box
        pick.append(i)  # keeping it
        # Computing the intersection (vectorised) btw the best box i and every lower-scored box
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        overlap = (w * h) / (area[idxs[1:]] + 1e-8)
        # Keep only those lower-scored boxes whose overlap <= 0.3 (threshold)
        idxs = idxs[np.where(overlap <= overlapThresh)[0] + 1]
    return pick  #return indices of the boxes to keep


def extract_feature(img_crop):
    img_tensor = preprocess(img_crop).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = cnn(img_tensor)
    return feat.cpu().numpy().flatten()

def detect_and_visualize(img_path, proposals_path, out_path, threshold=0.5):
    img = np.array(Image.open(img_path).convert("RGB"))
    base = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(base)
    proposals = np.load(proposals_path)
    pos_count = 0

    pos_boxes = []
    pos_scores = []
    for prop in proposals:
        x, y, w, h = map(int, prop)
        crop = img[y:y+h, x:x+w, :]
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            continue
        feat = extract_feature(crop)
        score = svm.decision_function([feat])[0]
        label = svm.predict([feat])[0]
        if label == 1 and score > threshold:
            pos_boxes.append([x, y, w, h])
            pos_scores.append(score)

    # Apply NMS
    picks = non_max_suppression_fast(pos_boxes, pos_scores, overlapThresh=0.3)
    for i in picks:
        x, y, w, h = pos_boxes[i]
        draw.rectangle([x, y, x+w, y+h], outline="red", width=2)
    pos_count = len(picks)

    base.save(out_path)
    print(f"{os.path.basename(img_path)}: {pos_count} detections -> {out_path}")

if __name__ == "__main__":
    # Run on all test images
    for fname in os.listdir(TEST_DIR):
        if not fname.lower().endswith(".jpg"):
            continue
        img_path = os.path.join(TEST_DIR, fname)
        proposals_path = os.path.join(PROPOSALS_DIR, f"test_{fname.replace('.jpg','')}_proposals.npy")
        if not os.path.exists(proposals_path):
            print(f"Missing proposals for {fname}, skipping!")
            continue
        out_path = os.path.join(OUT_DIR, fname.replace(".jpg", "_detected.jpg"))
        detect_and_visualize(img_path, proposals_path, out_path)
