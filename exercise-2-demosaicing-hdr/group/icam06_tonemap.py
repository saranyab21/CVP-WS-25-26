
import numpy as np
import matplotlib.pyplot as plt
import cv2
import rawpy
from scipy.ndimage import convolve
import os

test_path = "data/exercise_4_data/exercise_4_data/02/IMG_4782.CR3"
print("Exists?", os.path.exists(test_path))

# --- Load raw image ---
def load_raw_image(file_path):
    with rawpy.imread(file_path) as raw:
        raw_image = np.array(raw.raw_image_visible)
    return raw_image

# --- Generate masks for RGGB Bayer pattern ---
def generate_masks(shape):
    mask_red = np.zeros(shape)
    mask_green = np.zeros(shape)
    mask_blue = np.zeros(shape)
    mask_red[0::2, 0::2] = 1
    mask_green[0::2, 1::2] = 1
    mask_green[1::2, 0::2] = 1
    mask_blue[1::2, 1::2] = 1
    return mask_red, mask_green, mask_blue

# --- Simple demosaicing via convolution ---
def demosaic_image(raw_data):
    kernel = np.ones((3, 3))
    mask_red, mask_green, mask_blue = generate_masks(raw_data.shape)
    red_channel = convolve(mask_red * raw_data, kernel) / (convolve(mask_red, kernel) + 1e-8)
    green_channel = convolve(mask_green * raw_data, kernel) / (convolve(mask_green, kernel) + 1e-8)
    blue_channel = convolve(mask_blue * raw_data, kernel) / (convolve(mask_blue, kernel) + 1e-8)
    return np.stack([red_channel, green_channel, blue_channel], axis=-1)

# --- Apply gray world white balance ---
def apply_gray_world_white_balance(image):
    mean_r, mean_g, mean_b = np.mean(image[..., 0]), np.mean(image[..., 1]), np.mean(image[..., 2])
    mean_total = np.mean(image)
    balanced = np.empty_like(image)
    balanced[..., 0] = image[..., 0] * (mean_total / mean_r)
    balanced[..., 1] = image[..., 1] * (mean_total / mean_g)
    balanced[..., 2] = image[..., 2] * (mean_total / mean_b)
    return np.clip(balanced, 0, np.max(image))

# --- Apply iCAM06 tone mapping (simplified) ---
def apply_icam06(image, sigma_color=0.2, sigma_space=15):
    luminance = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
    base = cv2.bilateralFilter(luminance.astype(np.float32), d=5, sigmaColor=sigma_color*255, sigmaSpace=sigma_space)
    detail = luminance / (base + 1e-8)
    tone_mapped = np.log1p(base) * detail
    tone_mapped = np.clip(tone_mapped / np.max(tone_mapped), 0, 1)
    result = image * tone_mapped[..., None]
    result = np.clip(result / np.max(result), 0, 1)
    plt.imsave("icam06_output.png", result)
    print("iCAM06 output saved as: icam06_output.png")

# --- MAIN ---
if __name__ == "__main__":
    input_path = "data/exercise_4_data/exercise_4_data/02/IMG_4782.CR3"  # Update if needed
    raw = load_raw_image(input_path)
    demo = demosaic_image(raw)
    wb = apply_gray_world_white_balance(demo)
    apply_icam06(wb)
