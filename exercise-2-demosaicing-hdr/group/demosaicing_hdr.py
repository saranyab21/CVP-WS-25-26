import numpy as np
import rawpy
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

# Load the raw sensor data
def load_raw_image(file_path):
    with rawpy.imread(file_path) as raw:
        raw_image = np.array(raw.raw_image_visible)
    return raw_image

# Generate masks for each color channel based on the Bayer pattern
def generate_masks(shape):
    mask_red = np.zeros(shape)
    mask_green = np.zeros(shape)
    mask_blue = np.zeros(shape)  
    
    mask_red[0::2, 0::2] = 1  # Red pixels
    mask_green[0::2, 1::2] = 1  # Green pixels on even rows
    mask_green[1::2, 0::2] = 1  # Green pixels on odd rows
    mask_blue[1::2, 1::2] = 1  # Blue pixels


    return mask_red, mask_green, mask_blue

# Perform demosaicing using convolution and masks
def demosaic_image(raw_data):
    # Define a 3x3 convolution kernel (simple averaging kernel)
    kernel = np.ones((3, 3)) 

    # Generate masks
    mask_red, mask_green, mask_blue = generate_masks(raw_data.shape)

    print("Red Mask (6x6):\n", mask_red[:16, :16])
    print("Green Mask (6x6):\n", mask_green[:16, :16])
    print("Blue Mask (6x6):\n", mask_blue[:16, :16])

    # Compute each channel
    red_channel = (convolve(mask_red * raw_data, kernel) / convolve(mask_red, kernel))
    green_channel = (convolve(mask_green * raw_data, kernel) / convolve(mask_green, kernel))
    blue_channel = (convolve(mask_blue * raw_data, kernel) / convolve(mask_blue, kernel))

    # Stack channels to form the RGB image
    rgb_image = np.stack([red_channel, green_channel, blue_channel], axis=-1)
    return rgb_image

# Apply gamma correction
def apply_gamma_correction(image, gamma=0.3):
    # Use percentiles to normalize
    a = np.percentile(image, 0.01)
    b = np.percentile(image, 99.99)
    normalized = (image - a) / (b - a)
    normalized[normalized < 0] = 0
    normalized[normalized > 1] = 1

    # Apply gamma correction
    gamma_corrected = normalized ** gamma

    # Scale back to original range
    #corrected_image = gamma_corrected * (b - a) + a
    return gamma_corrected


def apply_log_correction(image, log_base=2, scale=1.0, offset=0.0):
    a = np.percentile(image, 0.01)
    b = np.percentile(image, 99.99)
    normalized = (image - a) / (b - a)
    normalized = np.clip(normalized, 0, 1)
    log_corrected = np.log1p(scale * (normalized + offset)) / np.log(log_base)
    return log_corrected

# Plot log curves for different parameters
def plot_log_curves(bases, scales, offset=0.0, save_path=None):
    x = np.linspace(0, 1, 500)
    plt.figure(figsize=(8, 6))
    for base in bases:
        for scale in scales:
            y = np.log1p(scale * (x + offset)) / np.log(base)
            label = f'base={base}, scale={scale}'
            plt.plot(x, y, label=label)
    plt.xlabel('Input Intensity')
    plt.ylabel('Output Intensity')
    plt.title('Logarithmic Correction Curves')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def apply_sigmoid_curve(image, a=10, b=0.5):
    """
    Sigmoid tone mapping: output = 1 / (1 + exp(-a*(x - b)))
    """
    #image = image.astype(np.float32)

    p1 = np.percentile(image, 0.01)
    p2 = np.percentile(image, 99.99)

    image_norm = (image - p1) / (p2 - p1)
    image_norm = np.clip(image_norm, 0, 1)

    image_sigmoid = 1 / (1 + np.exp(-a * (image_norm - b)))

    # Rescale to original range
    #image_mapped = image_sigmoid * (p2 - p1) + p1

    return image_sigmoid

# Display the resulting image for verification
def display_image(rgb_image, title="Image"):
    # Normalize for display purposes
    rgb_image_normalized = rgb_image / np.max(rgb_image)
    plt.imshow(rgb_image_normalized)
    plt.title(title)
    plt.axis('off')
    #plt.show()

def plot_gamma(save_path=None):

    # Create x values from 0 to 1
    x = np.linspace(0, 1, 500)

    # Gamma values
    gammas = [0.1, 0.3, 0.5, 1, 2, 3]
    colors = ['black', 'lime', 'blue', 'red', 'orange', 'purple']
    labels = ['0.1', '0.3', '0.5', '1', '2', '3']

    # Plot gamma curves
    plt.figure(figsize=(6, 5))
    for gamma, color, label in zip(gammas, colors, labels):
        y = x ** gamma
        plt.plot(x, y, color=color)
        # Place label at x=0.5
        y_label_pos = (0.5) ** gamma
        plt.text(0.52, y_label_pos, label, color=color, fontsize=12)

    # Plot formatting
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Input Intensity')
    plt.ylabel('Output Intensity')
    plt.title('Gamma Correction Curves')
    plt.grid(True)
    plt.axis('square')

    if save_path:
        plt.savefig(save_path)
    #plt.show()


def apply_gray_world_white_balance(image):
    """
    Applies Gray World white balance to an RGB image.
    """
    # Compute mean of each channel
    mean_r = np.mean(image[..., 0])
    mean_g = np.mean(image[..., 1])
    mean_b = np.mean(image[..., 2])
    # Compute overall mean
    mean_total = np.mean(image)
    # Scale each channel
    balanced = np.empty_like(image)
    balanced[..., 0] = image[..., 0] * (mean_total / mean_r)
    balanced[..., 1] = image[..., 1] * (mean_total / mean_g)
    balanced[..., 2] = image[..., 2] * (mean_total / mean_b)
    # Clip to [0, max of input] to avoid out-of-bounds
    balanced = np.clip(balanced, 0, np.max(image))
    return balanced
    


def prove_linearity_of_sensor_response():
    
    exposure_times = [1/10, 1/20, 1/40, 1/80, 1/160, 1/320]
    file_names = [
        "IMG_3044.CR3",
        "IMG_3045.CR3",
        "IMG_3046.CR3",
        "IMG_3047.CR3",
        "IMG_3048.CR3",
        "IMG_3049.CR3"
    ]

    mean_rs, mean_gs, mean_bs = [], [], []
    base_path = "data/exercise_4_data/exercise_4_data/05"

    for fname in file_names:
        full_file_path = f"{base_path}\\{fname}"
        raw_data = load_raw_image(full_file_path)  # Update path as needed
        demosaiced = demosaic_image(raw_data)
        # Optionally apply white balance here if you want to compare after WB
        mean_rs.append(np.mean(demosaiced[..., 0]))
        mean_gs.append(np.mean(demosaiced[..., 1]))
        mean_bs.append(np.mean(demosaiced[..., 2]))

    plt.figure(figsize=(8, 6))
    plt.plot(exposure_times, mean_rs, 'r-o', label='Red')
    plt.plot(exposure_times, mean_gs, 'g-o', label='Green')
    plt.plot(exposure_times, mean_bs, 'b-o', label='Blue')
    plt.xlabel('Exposure Time (s)')
    plt.ylabel('Average Channel Value')
    plt.title('Linearity of Sensor Response')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("linearity_plot.png")
    plt.show()

# Main execution
if __name__ == "__main__":
    raw_file_path = "data/exercise_4_data/exercise_4_data/02/IMG_4782.CR3"  # Update with your local file path

    # Load raw data
    raw_data = load_raw_image(raw_file_path)

    # Display original raw image
    plt.imshow(raw_data, cmap='gray')
    plt.title("Original Raw Image")
    plt.axis('off')
    #plt.show()

    # Perform demosaicing
    demosaiced_image = demosaic_image(raw_data)
    plt.imsave("demosaiced_image.png", demosaiced_image / np.max(demosaiced_image))

    # Apply gamma correction with different values
    gamma_corrected_image_01 = apply_gamma_correction(demosaiced_image, gamma=0.1)
    gamma_corrected_image_03 = apply_gamma_correction(demosaiced_image, gamma=0.3)
    gamma_corrected_image_05 = apply_gamma_correction(demosaiced_image, gamma=0.5)
    gamma_corrected_image_1 = apply_gamma_correction(demosaiced_image, gamma=1)
    gamma_corrected_image_2 = apply_gamma_correction(demosaiced_image, gamma=2) 
    gamma_corrected_image_3 = apply_gamma_correction(demosaiced_image, gamma=3)



    images = [
        (gamma_corrected_image_01, "Gamma=0.1"),
        (gamma_corrected_image_03, "Gamma=0.3"),
        (gamma_corrected_image_05, "Gamma=0.5"),
        (gamma_corrected_image_1, "Gamma=1"),
        (gamma_corrected_image_2, "Gamma=2"),
        (gamma_corrected_image_3, "Gamma=3"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    for ax, (img, title) in zip(axes, images):
        ax.imshow(img / np.max(img))
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("gamma_corrected_images_comparison.png")
    #plt.show()
     # Apply alternate curve adjustment

    plot_gamma(save_path="gamma_curves.png")


    #log_corrected_image = apply_log_correction(demosaiced_image)

    #plt.imsave("log_corrected_image.png", log_corrected_image / np.max(log_corrected_image))
    wb_image = apply_gray_world_white_balance(demosaiced_image)

    print("Means before WB:", np.mean(demosaiced_image[..., 0]), np.mean(demosaiced_image[..., 1]), np.mean(demosaiced_image[..., 2]))
    print("Means after WB:", np.mean(wb_image[..., 0]), np.mean(wb_image[..., 1]), np.mean(wb_image[..., 2]))    
    
    plt.imsave("demosaiced_image_grayworld_wb2.png", wb_image / np.max(wb_image))

    log_bases = [2, 5, 10]
    scales = [0.5, 1, 2]
    offset = 0.0

    # Try all combinations and save images
    for base in log_bases:
        for scale in scales:
            log_img = apply_log_correction(demosaiced_image, log_base=base, scale=scale, offset=offset)
            filename = f"log_corrected_base{base}_scale{str(scale).replace('.','_')}.png"
            plt.imsave(filename, log_img / np.max(log_img))

    # Plot and save the log curves
    plot_log_curves(log_bases, scales, offset=offset, save_path="log_correction_curves.png")
    # Display the results
    #display_image(demosaiced_image, title="Demosaiced Image")
    #display_image(gamma_corrected_image_03, title="Gamma Corrected Image (Gamma=0.3)")
    #display_image(gamma_corrected_image_05, title="Gamma Corrected Image (Gamma=0.5)")
    #display_image(log_corrected_image, title="Logarithmic Corrected Image")

    prove_linearity_of_sensor_response()

    



# ----------------------------------
# Part 1: Investigate Bayer Pattern (Manual Visual Check)
# ----------------------------------
def investigate_bayer_pattern(npy_path):
    data = np.load(npy_path)
    region = data[100:116, 100:116]
    plt.imshow(region, cmap='gray')
    plt.title("Zoomed Bayer Pattern Region")
    plt.colorbar()
    plt.savefig("bayer_pattern_region.png")
    print("Patch for manual inspection:")
    print(region[:8, :8])

# ----------------------------------
# Part 6: Initial HDR Implementation
# ----------------------------------
def initial_hdr_merge(image_paths, exposure_times):
    hdr_stack = []
    for path, exposure in zip(image_paths, exposure_times):
        raw_data = load_raw_image(path)
        hdr_stack.append(raw_data / exposure)
    hdr_raw = np.mean(hdr_stack, axis=0)
    demosaiced = demosaic_image(hdr_raw)
    white_balanced = apply_gray_world_white_balance(demosaiced)
    log_hdr = np.log1p(white_balanced)
    scaled_hdr = 255 * (log_hdr / np.max(log_hdr))
    scaled_hdr = np.clip(scaled_hdr, 0, 255).astype(np.uint8)
    plt.imsave("initial_hdr_output.png", scaled_hdr)
    return scaled_hdr

# ----------------------------------
# Part 7: iCAM06 (Simplified Version)
# ----------------------------------
def apply_icam06(image, sigma_color=0.2, sigma_space=15):
    luminance = 0.2126 * image[...,0] + 0.7152 * image[...,1] + 0.0722 * image[...,2]
    base = cv2.bilateralFilter(luminance.astype(np.float32), d=5, sigmaColor=sigma_color*255, sigmaSpace=sigma_space)
    detail = luminance / (base + 1e-8)
    tone_mapped = np.log1p(base) * detail
    tone_mapped = np.clip(tone_mapped / np.max(tone_mapped), 0, 1)
    result = image * tone_mapped[..., None]
    result = np.clip(result / np.max(result), 0, 1)
    plt.imsave("icam06_output.png", result)
    return result

# ----------------------------------
# Part 8: Final Demosaicing Function for Submission
# ----------------------------------
from PIL import Image
def process_raw(input_path, output_path):
    raw_data = load_raw_image(input_path)
    demosaiced = demosaic_image(raw_data)
    white_balanced = apply_gray_world_white_balance(demosaiced)
    corrected = apply_gamma_correction(white_balanced, gamma=0.3)
    corrected = (corrected * 255).astype(np.uint8)
    img = Image.fromarray(corrected)
    img.save(output_path, quality=99)


# ---------- Additional Task Executions ----------

# Task 1: Bayer pattern investigation
# Uncomment and ensure IMG_9939.npy exists in working directory
investigate_bayer_pattern("data/exercise_4_data/exercise_4_data/01/IMG_9939.npy")

# Task 6: Initial HDR merge (requires 00.CR3 to 10.CR3 in correct path)
hdr_image_paths = [f"00{i}.CR3" if i < 10 else f"0{i}.CR3" for i in range(11)]
hdr_image_paths = [f"data/exercise_4_data/HDR/{p}" for p in hdr_image_paths]
hdr_exposures = [1 / (2 ** i) for i in range(11)]

try:
    initial_hdr_merge(hdr_image_paths, hdr_exposures)
except Exception as e:
    print("HDR Merge (Task 6) failed:", e)

# Task 7: iCAM06 tone mapping
try:
    icam_output = apply_icam06(wb_image)
except Exception as e:
    print("iCAM06 (Task 7) failed:", e)

# Task 8: Final demosaicing function
try:
    process_raw(
        "data/exercise_4_data/exercise_4_data/02/IMG_4782.CR3",
        "processed_output_final.jpg"
    )
except Exception as e:
    print("process_raw (Task 8) failed:", e)
