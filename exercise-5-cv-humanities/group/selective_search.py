'''
@author: Prathmesh R Madhu.
For educational purposes only
'''
# -*- coding: utf-8 -*-
from __future__ import division

import skimage.io
import skimage.feature
import skimage.color
import skimage.transform
import skimage.util
import skimage.segmentation
import numpy as np
from skimage.segmentation import felzenszwalb
from skimage.feature import local_binary_pattern
from skimage.util import img_as_ubyte

def generate_segments(im_orig, scale, sigma, min_size):
    """
    Task 1: Segment smallest regions by the algorithm of Felzenswalb.
    1.1. Generate the initial image mask using felzenszwalb algorithm
    1.2. Merge the image mask to the image as a 4th channel
    """
    img_lbl = felzenszwalb(im_orig, scale=scale, sigma=sigma, min_size=min_size)
    im_mask = np.zeros((im_orig.shape[0], im_orig.shape[1], 4), dtype=np.float32)
    im_mask[:, :, :3] = im_orig
    im_mask[:, :, 3] = img_lbl
    
    return im_mask

def sim_colour(r1, r2):
    """
    2.1. calculate the sum of histogram intersection of colour
    """
    return np.sum(np.minimum(r1["colour_hist"], r2["colour_hist"]))
    

def sim_texture(r1, r2):
    """
    2.2. calculate the sum of histogram intersection of texture
    """
    return np.sum(np.minimum(r1["texture_hist"], r2["texture_hist"]))


def sim_size(r1, r2, imsize):
    """
    2.3. calculate the size similarity over the image
    """
    return 1.0 - (r1["size"] + r2["size"]) / imsize


def sim_fill(r1, r2, imsize):
    """
    2.4. calculate the fill similarity over the image
    """
    bbsize = (
        max(r1["max_x"], r2["max_x"]) - min(r1["min_x"], r2["min_x"])
    ) * (
        max(r1["max_y"], r2["max_y"]) - min(r1["min_y"], r2["min_y"])
    )
    return 1.0 - (bbsize - r1["size"] - r2["size"]) / imsize

def calc_sim(r1, r2, imsize):
    return (sim_colour(r1, r2) + sim_texture(r1, r2)
            + sim_size(r1, r2, imsize) + sim_fill(r1, r2, imsize))

def calc_colour_hist(img):
    """
    Task 2.5.1
    calculate colour histogram for each region
    the size of output histogram will be BINS * COLOUR_CHANNELS(3)
    number of bins is 25 as same as [uijlings_ijcv2013_draft.pdf]
    extract HSV
    """
    
    BINS = 25
    hsv = skimage.color.rgb2hsv(img[:, :, :3])
    hist = []

    for channel in range(3):  # HSV channels
        c = hsv[:, :, channel]
        hist_channel, _ = np.histogram(c, bins=BINS, range=(0, 1), density=True)
        hist.extend(hist_channel)

    return np.array(hist)


def calc_texture_gradient(img):
    """
    Task 2.5.2
    calculate texture gradient for entire image
    The original SelectiveSearch algorithm proposed Gaussian derivative
    for 8 orientations, but we will use LBP instead.
    output will be [height(*)][width(*)]
    Useful function: Refer to skimage.feature.local_binary_pattern documentation
    """
    
    P = 8  # Number of circularly symmetric neighbor set points
    R = 1  # Radius
    METHOD = 'uniform'

    ret = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=np.float32)

    for c in range(img.shape[2]):
        ret[:, :, c] = local_binary_pattern(img[:, :, c], P, R, METHOD)

    return ret

def calc_texture_hist(img):
    """
    Task 2.5.3
    calculate texture histogram for each region
    calculate the histogram of gradient for each colours
    the size of output histogram will be
        BINS * ORIENTATIONS * COLOUR_CHANNELS(3)
    Do not forget to L1 Normalize the histogram
    """
    
    BINS = 10
    hist = []

    for c in range(img.shape[2]):
        channel = img[:, :, c]
        hist_channel, _ = np.histogram(channel, bins=BINS, range=(0, 256), density=True)
        hist.extend(hist_channel)

    hist = np.array(hist)
    hist = hist / (np.sum(hist) + 1e-6)  # L1 Normalization

    return hist

def extract_regions(img):
    '''
    Task 2.5: Generate regions denoted as datastructure R
    - Convert image to hsv color map
    - Count pixel positions
    - Calculate the texture gradient
    - calculate color and texture histograms
    - Store all the necessary values in R.
    '''
    hsv = skimage.color.rgb2hsv(img[:, :, :3])
    img_size = img.shape[0] * img.shape[1]
    img_mask = img[:, :, 3].astype(int)  # segmentation mask
    texture_gradient = calc_texture_gradient(img[:, :, :3])

    R = {}
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            label = img_mask[y, x]
            if label not in R:
                R[label] = {
                    "min_x": x, "min_y": y,
                    "max_x": x, "max_y": y,
                    "labels": [label],
                    "size": 0,
                    "colour_hist": np.zeros(25 * 3),
                    "texture_hist": np.zeros(10 * 3),
                    "pixels": []
                }
            R[label]["min_x"] = min(R[label]["min_x"], x)
            R[label]["min_y"] = min(R[label]["min_y"], y)
            R[label]["max_x"] = max(R[label]["max_x"], x)
            R[label]["max_y"] = max(R[label]["max_y"], y)
            R[label]["size"] += 1
            R[label]["pixels"].append((y, x))

    for k, v in R.items():
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
        for y, x in v["pixels"]:
            mask[y, x] = True
        region_rgb = img[:, :, :3] * mask[:, :, None]
        region_texture = texture_gradient * mask[:, :, None]

        R[k]["colour_hist"] = calc_colour_hist(region_rgb)
        R[k]["texture_hist"] = calc_texture_hist(region_texture)

    return R


def extract_neighbours(regions):

    def intersect(a, b):
        if (a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]):
            return True
        return False

    neighbours = []
    keys = list(regions.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            r1 = regions[keys[i]]
            r2 = regions[keys[j]]
            if intersect(r1, r2):
                neighbours.append(((keys[i], r1), (keys[j], r2)))

    return neighbours


def merge_regions(r1, r2):
    new_size = r1["size"] + r2["size"]
    rt = {
        "min_x": min(r1["min_x"], r2["min_x"]),
        "min_y": min(r1["min_y"], r2["min_y"]),
        "max_x": max(r1["max_x"], r2["max_x"]),
        "max_y": max(r1["max_y"], r2["max_y"]),
        "size": new_size,
        "labels": [*r1["labels"], *r2["labels"]][:5],  # Limit to avoid infinite growth
        "colour_hist": (r1["colour_hist"] * r1["size"] + r2["colour_hist"] * r2["size"]) / new_size,
        "texture_hist": (r1["texture_hist"] * r1["size"] + r2["texture_hist"] * r2["size"]) / new_size,
    }


    return rt


def selective_search(image_orig, scale=1.0, sigma=0.8, min_size=50):
    '''
    Selective Search for Object Recognition" by J.R.R. Uijlings et al.
    :arg:
        image_orig: np.ndarray, Input image
        scale: int, determines the cluster size in felzenszwalb segmentation
        sigma: float, width of Gaussian kernel for felzenszwalb segmentation
        min_size: int, minimum component size for felzenszwalb segmentation

    :return:
        image: np.ndarray,
            image with region label
            region label is stored in the 4th value of each pixel [r,g,b,(region)]
        regions: array of dict
            [
                {
                    'rect': (left, top, width, height),
                    'labels': [...],
                    'size': component_size
                },
                ...
            ]
    '''

    # Checking the 3 channel of input image
    assert image_orig.shape[2] == 3, "Please use image with three channels."
    imsize = image_orig.shape[0] * image_orig.shape[1]

    # Task 1: Load image and get smallest regions. Refer to `generate_segments` function.
    image = generate_segments(image_orig, scale, sigma, min_size)

    if image is None:
        return None, {}

    # Task 2: Extracting regions from image
    # Task 2.1-2.4: Refer to functions "sim_colour", "sim_texture", "sim_size", "sim_fill"
    # Task 2.5: Refer to function "extract_regions". You would also need to fill "calc_colour_hist",
    # "calc_texture_hist" and "calc_texture_gradient" in order to finish task 2.5.
    R = extract_regions(image)

    # Task 3: Extracting neighbouring information
    # Refer to function "extract_neighbours"
    neighbours = extract_neighbours(R)

    # Calculating initial similarities
    S = {}
    for (ai, ar), (bi, br) in neighbours:
        S[(ai, bi)] = calc_sim(ar, br, imsize)

    merge_limit = 200  # Max number of merges
    merge_count = 0

    # Hierarchical search for merging similar regions
    while S != {} and merge_count < merge_limit:

        # Get highest similarity pair (i, j)
        i, j = sorted(S.items(), key=lambda i: i[1])[-1][0]

        # Task 4: Merge corresponding regions. Refer to function "merge_regions"
        t = max(R.keys()) + 1.0  # new region ID
        R[t] = merge_regions(R[i], R[j])

        # Task 5: Mark similarities for regions to be removed
        keys_to_remove = []
        for k in S.keys():
            if i in k or j in k:
                keys_to_remove.append(k)


        # Task 6: Remove old similarities of related regions
        for k in keys_to_remove:
            del S[k]


        # Task 7: Calculate similarities with the new region
        for k in R.keys():
            if k == t:
                continue
            S[(t, k)] = calc_sim(R[t], R[k], imsize)

        merge_count += 1


    # Task 8: Generating the final regions from R
    regions = []
    for k, v in R.items():
        rect = (
            v["min_x"],
            v["min_y"],
            v["max_x"] - v["min_x"],
            v["max_y"] - v["min_y"],
        )
        regions.append({
            "rect": rect,
            "size": v["size"],
            "labels": v["labels"]
        })


    return image, regions


