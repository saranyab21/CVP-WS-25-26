import os
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image

LABELED_DIR = "../results/labeled_proposals"
TRAIN_DIR = "../data/balloon_dataset/train"
VALID_DIR = "../data/balloon_dataset/valid"
FEATURES_DIR = "../results/features"
os.makedirs(FEATURES_DIR, exist_ok=True)

# Using ResNet18 (removing the final classification layer for features)
cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
cnn = torch.nn.Sequential(*list(cnn.children())[:-1])  # Removing final FC
cnn.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn = cnn.to(device)

# Preprocessing for torchvision models
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_feature(img_crop):
    img_tensor = preprocess(img_crop).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = cnn(img_tensor)
    return feat.cpu().numpy().flatten()

def process_split(split_name, img_dir):
    all_features = []
    all_labels = []
    img_list = [f for f in os.listdir(img_dir) if f.lower().endswith('.jpg')]
    for fname in img_list:
        labeled_path = os.path.join(LABELED_DIR, f"{split_name}_{fname.replace('.jpg','')}_labeled.npz")
        if not os.path.exists(labeled_path):
            print(f"Warning: no labeled proposals for {fname}")
            continue
        arr = np.load(labeled_path)
        proposals, labels = arr["proposals"], arr["labels"]
        img = np.array(Image.open(os.path.join(img_dir, fname)).convert("RGB"))
        for prop, lab in zip(proposals, labels):
            if lab not in [0, 1]:
                continue  # Only using positive/negative
            x, y, w, h = map(int, prop)
            crop = img[y:y+h, x:x+w, :]  # region
            if crop.shape[0] == 0 or crop.shape[1] == 0:  # skip invalid
                continue
            feat = extract_feature(crop)
            all_features.append(feat)
            all_labels.append(lab)
        print(f"Processed {fname}: {len(all_features)} regions so far.")

    X = np.array(all_features)
    y = np.array(all_labels)
    np.save(os.path.join(FEATURES_DIR, f"{split_name}_X.npy"), X)
    np.save(os.path.join(FEATURES_DIR, f"{split_name}_y.npy"), y)
    print(f"Saved features for {split_name}: {X.shape}, {y.shape}")

if __name__ == "__main__":
    process_split('train', TRAIN_DIR)
    process_split('valid', VALID_DIR)
    print("Done: Feature extraction complete.")
