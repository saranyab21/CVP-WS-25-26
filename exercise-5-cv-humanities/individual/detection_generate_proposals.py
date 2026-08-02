import os
import skimage.io
import numpy as np
import sys

sys.path.append(os.path.dirname(__file__))  
from selective_search import selective_search  

# Set directories
TRAIN_DIR = "../data/balloon_dataset/train"
VALID_DIR = "../data/balloon_dataset/valid"
TEST_DIR = "../data/balloon_dataset/test"

PROPOSALS_DIR = "../results/proposals"
os.makedirs(PROPOSALS_DIR, exist_ok=True)

def run_selective_search_on_image(img_path):
    img = skimage.io.imread(img_path)
    # The function below returns list of boxes: [x, y, w, h] 
    _, regions = selective_search(img, scale=500, sigma=0.9, min_size=10)
    # regions: [{'rect': (x, y, w, h), ...}, ...]
    # Converting to a numpy array of boxes
    boxes = np.array([r['rect'] for r in regions])
    return boxes

def process_dataset(img_dir, split_name):
    img_list = [f for f in os.listdir(img_dir) if f.lower().endswith('.jpg')]
    for fname in img_list:
        img_path = os.path.join(img_dir, fname)
        boxes = run_selective_search_on_image(img_path)
        # Saving proposals as .npy file, one per image
        out_name = f"{split_name}_{fname.replace('.jpg','')}_proposals.npy"
        out_path = os.path.join(PROPOSALS_DIR, out_name)
        np.save(out_path, boxes)
        print(f"Saved {len(boxes)} proposals for {fname} --> {out_path}")

if __name__ == "__main__":
    process_dataset(TRAIN_DIR, 'train')
    process_dataset(VALID_DIR, 'valid')
    process_dataset(TEST_DIR, 'test')
    print("Done: proposals generated for all train/valid/test images.")
