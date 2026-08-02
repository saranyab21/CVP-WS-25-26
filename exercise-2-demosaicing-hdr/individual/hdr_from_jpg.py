
# Bonus HDR from JPGs using CRF estimation 

import numpy as np
from PIL import Image
import glob
import os

# ---- 1. Setting up paths and exposure times ----
DATA_DIR = '../ex4_additional_exercise_data/ex4_additional_exercise_data'

# Collect JPG files in alphabetical order (brightest to darkest exposures)
img_paths = sorted(glob.glob(os.path.join(DATA_DIR, '*.jpg')))
print("Image file order:")
for i, p in enumerate(img_paths):
    print(f"{i}: {os.path.basename(p)}")

# Define exposure times (shortest->longest), then reverse to match file order
exposure_times = np.array([
    1/4096, 1/2048, 1/1024, 1/512, 1/256, 1/128,
    1/64, 1/32, 1/16, 1/8, 1/4, 1/2
], dtype=np.float32)[::-1]

assert len(img_paths) == len(exposure_times), "Number of images and exposure times must match!"

# ---- 2. Loading image stack ----
print("Loading images...")
images = [np.array(Image.open(p)).astype(np.uint8) for p in img_paths]
images = np.stack(images, axis=0)  # (N, H, W, 3)
num_images, H, W, C = images.shape
print(f"Loaded {num_images} images of shape {H}x{W}.")

# ---- 3. Weighting function for Debevec's algorithm ----
def weighting(z):
    # Triangular weight: favors mid-tone pixels, ignores saturated/dark
    z = np.asarray(z)
    return np.minimum(z, 255-z)

# ---- 4. Random sample selection (avoid saturated/too-dark pixels) ----
np.random.seed(42)
num_samples = 30  
print(f"Sampling {num_samples} random non-saturated pixel coordinates...")
coords = []
tries = 0
while len(coords) < num_samples:
    tries += 1
    if tries > 100000:
        raise RuntimeError("Few valid pixels found, try even lower num_samples or relax sampling further")
    # Pick a random pixel location
    y = np.random.randint(0, H)
    x = np.random.randint(0, W)
    px = images[:, y, x, :]
    # Pixel must have at least 6 valid exposures in each channel
    valid = ((px > 5) & (px < 250))  # (12, 3)
    if np.all(np.sum(valid, axis=0) >= 6):  # all channels have at least 6 valid exposures
        coords.append((y, x))
coords = np.array(coords)
print("Selected sample coordinates:", coords.shape)

# ---- 5. Debevec & Malik CRF solver (per channel) ----
def gsolve(Z, B, l=100, w=None):
    """Solves for camera response curve g and log scene irradiance lE."""
    n = 256
    num_samples, num_images = Z.shape
    A = np.zeros((num_samples*num_images + n + 1, n + num_samples), dtype=np.float64)
    b = np.zeros((A.shape[0],), dtype=np.float64)
    k = 0
    if w is None:
        w = np.ones(256)
    # Data-fitting equations
    for i in range(num_samples):
        for j in range(num_images):
            z = Z[i, j]
            wij = w[z]
            A[k, z] = wij       # coefficient for g(z)
            A[k, n + i] = -wij  # coefficient for log irradiance
            b[k] = wij * B[j]   # RHS = weighted log exposure time
            k += 1
    # Fix g(128)=0 : # Anchor: To avoid trivial shift
    A[k,128] = 1
    k += 1
    # Smoothness constraints: encourage smooth g(z)
    for z in range(1, n-1):
        A[k, z-1] = l * w[z]
        A[k, z  ] = -2 * l * w[z]
        A[k, z+1] = l * w[z]
        k += 1
    # Solve least-squares system
    print(f"  Solving linear system of size {A.shape} ...")
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    g = x[:n]
    lE = x[n:]
    return g, lE

# ---- 6. Estimate g(z) for R, G, B separately ----
Z_channels = []
for c in range(3):
    # Build Z: sampled pixel values for this channel
    Z = np.zeros((num_samples, num_images), dtype=np.uint8)
    for i, (y, x) in enumerate(coords):
        Z[i, :] = images[:, y, x, c]
    Z_channels.append(Z)
    
# Log exposure times
B = np.log(exposure_times)

# Solve CRF for each channel
g_curves = []
for c in range(3):
    print(f"Preparing to solve CRF for channel {c} ...")
    g, _ = gsolve(Z_channels[c], B, l=100, w=weighting(np.arange(256)))
    print(f"Solved CRF for channel {c}.")
    g_curves.append(g)

# ---- 7. Linearize all images ----
print("Linearizing all images ...")
# Map pixel values through g, subtract log shutter time
log_images = np.zeros_like(images, dtype=np.float64)
for c in range(3):
    g = g_curves[c]
    for j in range(num_images):
        img = images[j, :, :, c]
        log_images[j, :, :, c] = g[img] - B[j]  # shape: (H, W)

# ---- 8. HDR merge (weighted average in log domain) ----
print("Merging HDR ...")
w = weighting(np.arange(256))
hdr = np.zeros((H, W, 3), dtype=np.float64)
for c in range(3):
    num = np.zeros((H, W), dtype=np.float64)
    den = np.zeros((H, W), dtype=np.float64)
    for j in range(num_images):
        img = images[j, :, :, c]
        wij = w[img]
        num += wij * log_images[j, :, :, c]
        den += wij
    # Final HDR radiance map = exp(weighted avg of log irradiance)
    hdr[:, :, c] = np.exp(num / (den + 1e-8))  # shape: (H, W)

# ---- 9. Tone mapping (log compression) ----
print("Tone mapping ...")
# Log compression, normalize, scale to 8-bit
hdr_tonemapped = np.log(1 + hdr)
hdr_tonemapped = hdr_tonemapped / np.max(hdr_tonemapped)
hdr_tonemapped = np.clip(hdr_tonemapped * 255, 0, 255).astype(np.uint8)

# ---- 10. Save result ----
results_dir = os.path.join(os.path.dirname(__file__), 'bonus_result')
os.makedirs(results_dir, exist_ok=True)
out_path = os.path.join(results_dir, 'hdr_jpg_bonus_result.png')
Image.fromarray(hdr_tonemapped).save(out_path)
print(f"HDR result saved to {out_path}")


