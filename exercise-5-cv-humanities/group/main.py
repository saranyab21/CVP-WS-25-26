'''
@author: Prathmesh R Madhu.
For educational purposes only
'''

# -*- coding: utf-8 -*-
from __future__ import (
    division,
    print_function,
)

import os
import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from selective_search import selective_search

    
# loading a test image from '../data' folder
# Add folders with images
image_dirs = [
    '../data/chrisarch',
    '../data/arthist',
    '../data/classarch'
]


def process_image(image_path, output_dir):
    image = skimage.io.imread(image_path)
    print(f'Processing {image_path}, shape: {image.shape}')

    image_label, regions = selective_search(
        image,
        scale=500,
        min_size=20
    )

    candidates = set()
    for r in regions:
        if r['rect'] in candidates:
            continue
        if r['size'] < 2000:
            continue
        x, y, w, h = r['rect']
        if w / h > 1.2 or h / w > 1.2:
            continue
        candidates.add(r['rect'])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    for x, y, w, h in candidates:
        rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
    plt.axis('off')

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, os.path.basename(image_path))
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f'Saved: {save_path}')

def main():
    for directory in image_dirs:
        for filename in os.listdir(directory):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(directory, filename)
                process_image(image_path, '../results')


if __name__ == '__main__':
    main()