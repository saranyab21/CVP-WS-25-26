# To extract writer ID from filename:

import os
import random

IMG_DIR = r"C:\Users\admin\OneDrive\Desktop\SEM-2\Computer Vision project\Exercise 2\icdar17-historicalwi-training-color\icdar2017-training-color"
all_imgs = [f for f in os.listdir(IMG_DIR) if f.endswith(('.png','.jpg','.jpeg'))]

# Get mapping from writer id to files
writer_to_imgs = {}
for fn in all_imgs:
    writer_id = fn.split('-')[0]
    writer_to_imgs.setdefault(writer_id, []).append(fn)

# For each writer, put 50% of their images into train, 50% into test (randomly)
train_lines, test_lines = [], []
for writer, imgs in writer_to_imgs.items():
    if len(imgs) == 1:
        # Only 1 image: put in train (or you can choose to put in test)
        train_lines.append(f"{imgs[0]} {writer}\n")
    else:
        random.shuffle(imgs)
        split = len(imgs) // 2
        for fn in imgs[:split]:
            train_lines.append(f"{fn} {writer}\n")
        for fn in imgs[split:]:
            test_lines.append(f"{fn} {writer}\n")

with open('color_labels_train.txt', 'w') as f:
    f.writelines(train_lines)
with open('color_labels_test.txt', 'w') as f:
    f.writelines(test_lines)

print("Done.")
