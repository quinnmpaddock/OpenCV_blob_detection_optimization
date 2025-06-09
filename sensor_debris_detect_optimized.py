import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from tqdm import tqdm
from scipy.spatial import cKDTree 
import os
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from dataclasses import dataclass
from typing import List, Dict, Tuple
import glob

Image.MAX_IMAGE_PIXELS = 2000000000

@dataclass
class CategorizedKeypoint:
    """Wrapper to preserve keypoint category information"""
    keypoint: cv2.KeyPoint
    category: str

    @property
    def pt(self):
        return self.keypoint.pt

    @property
    def size(self):
        return self.keypoint.size

    def scale(self, factor):
        """Scale keypoint coordinates and size"""
        self.keypoint.pt = (self.keypoint.pt[0] * factor, self.keypoint.pt[1] * factor)
        self.keypoint.size *= factor

def get_unique_filename(base_path, extension):
    """
    Generate a unique filename by adding a number suffix if the file already exists.

    Args:
        base_path (str): Base path without extension (e.g., "/path/to/file")
        extension (str): File extension with dot (e.g., ".csv", ".png")

    Returns:
        str: Unique filename with number suffix if needed
    """
    full_path = base_path + extension

    # If file doesn't exist, return original name
    if not os.path.exists(full_path):
        return full_path

    # Find existing numbered files
    pattern = f"{base_path}_*{extension}"
    existing_files = glob.glob(pattern)

    # Extract numbers from existing files
    numbers = []
    base_name = os.path.basename(base_path)

    for file_path in existing_files:
        filename = os.path.basename(file_path)
        # Remove extension and base name to get the number part
        name_without_ext = filename.replace(extension, '')
        if name_without_ext.startswith(base_name + '_'):
            suffix = name_without_ext[len(base_name + '_'):]
            try:
                numbers.append(int(suffix))
            except ValueError:
                continue

    # Find next available number
    if numbers:
        next_number = max(numbers) + 1
    else:
        next_number = 1

    return f"{base_path}_{next_number}{extension}"

def create_unique_output_paths(save_dir, filename):
    """
    Create unique output paths for all three output files.

    Args:
        save_dir (str): Output directory path
        filename (str): Base filename without extension

    Returns:
        tuple: (csv_path, hist_path, img_path) with unique names
    """
    # Create base paths
    csv_base = os.path.join(save_dir, f"{filename}_keypoints")
    hist_base = os.path.join(save_dir, f"{filename}_histogram")
    img_base = os.path.join(save_dir, f"{filename}_keypoints_image")

    # Get unique paths for each file type
    csv_path = get_unique_filename(csv_base, ".csv")
    hist_path = get_unique_filename(hist_base, ".png")
    img_path = get_unique_filename(img_base, ".png")

    return csv_path, hist_path, img_path

def subdivide_image(image_path, rows, cols, overlap):
    """Optimized image subdivision with better memory handling"""
    img_full = Image.open(image_path).convert("RGB")
    img_width, img_height = img_full.size

    sub_images = []
    coords = []
    base_lengths = []

    base_tile_width = img_width // cols
    base_tile_height = img_height // rows

    for row in range(rows):
        for col in range(cols):
            if row == rows - 1:
                tile_height = (img_height // rows)
                upper = row * (tile_height)
            else:
                tile_height = (img_height // rows) + overlap
                upper = row * (tile_height - overlap)

            if col == cols - 1:
                tile_width = (img_width // cols)
                left = col * (tile_width)
            else:
                tile_width = (img_width // cols) + overlap
                left = col * (tile_width - overlap)

            right = left + tile_width
            lower = upper + tile_height

            sub_image = img_full.crop((left, upper, right, lower))

            sub_images.append(np.array(sub_image))
            coords.append((left, upper))
            base_lengths.append((base_tile_width, base_tile_height))

    return sub_images, coords, base_lengths, (img_width, img_height), img_full

def estimate_background_intensity(image, patch_size=50):
    """Improved background estimation using multiple patches"""
    h, w = image.shape
    patches = []

    # Sample from multiple locations for better estimation
    for i in range(3):
        for j in range(3):
            y = h * (i + 1) // 4
            x = w * (j + 1) // 4
            half = patch_size // 2

            # Ensure we don't go out of bounds
            y_start = max(0, y - half)
            y_end = min(h, y + half)
            x_start = max(0, x - half)
            x_end = min(w, x + half)

            patch = image[y_start:y_end, x_start:x_end]
            if patch.size > 0:
                patches.append(np.median(patch))

    return int(np.median(patches)) if patches else 128

def create_enhanced_masks(gray, bg_intensity, mask_cushion):
    """Create enhanced masks with morphological operations"""
    # Basic intensity-based masks
    white_mask = cv2.inRange(gray, bg_intensity + mask_cushion, 255)
    black_mask = cv2.inRange(gray, 0, bg_intensity - mask_cushion)

    # Enhance masks with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)

    return white_mask, black_mask

def detect_blobs(sub_image, coords, local_coord_number, params, params_thin):
    """Optimized blob detection with category preservation"""
    mask_cushion = config["mask_cushion"]

    # Convert to grayscale once
    gray = cv2.cvtColor(sub_image, cv2.COLOR_BGR2GRAY)

    # Improved background estimation
    background_intensity = estimate_background_intensity(gray)

    # Create enhanced masks
    white_mask, black_mask = create_enhanced_masks(gray, background_intensity, mask_cushion)

    # Create detectors
    detector = cv2.SimpleBlobDetector_create(params)
    detector_thin = cv2.SimpleBlobDetector_create(params_thin)

    # Detect blobs for each category
    white_keypoints = detector.detect(white_mask)
    black_keypoints = detector.detect(black_mask)
    thin_white_keypoints = detector_thin.detect(white_mask)
    thin_black_keypoints = detector_thin.detect(black_mask)

    # Adjust coordinates and create categorized keypoints
    offset_x, offset_y = coords[local_coord_number]
    categorized_keypoints = []

    for kp in white_keypoints:
        kp.pt = (kp.pt[0] + offset_x, kp.pt[1] + offset_y)
        categorized_keypoints.append(CategorizedKeypoint(kp, 'white'))

    for kp in black_keypoints:
        kp.pt = (kp.pt[0] + offset_x, kp.pt[1] + offset_y)
        categorized_keypoints.append(CategorizedKeypoint(kp, 'black'))

    for kp in thin_white_keypoints:
        kp.pt = (kp.pt[0] + offset_x, kp.pt[1] + offset_y)
        categorized_keypoints.append(CategorizedKeypoint(kp, 'thin_white'))

    for kp in thin_black_keypoints:
        kp.pt = (kp.pt[0] + offset_x, kp.pt[1] + offset_y)
        categorized_keypoints.append(CategorizedKeypoint(kp, 'thin_black'))

    return categorized_keypoints

def filter_keypoints(categorized_keypoints: List[CategorizedKeypoint], threshold_distance):
    """Improved filtering that preserves categories and selects best keypoint in clusters"""
    if len(categorized_keypoints) < 2:
        return categorized_keypoints, len(categorized_keypoints)

    # Extract coordinates for spatial indexing
    coords = np.array([ckp.pt for ckp in categorized_keypoints])
    tree = cKDTree(coords)

    keep_mask = np.ones(len(categorized_keypoints), dtype=bool)

    for i in range(len(categorized_keypoints)):
        if not keep_mask[i]:
            continue

        # Find neighbors within threshold
        neighbors = tree.query_ball_point(coords[i], threshold_distance)

        if len(neighbors) > 1:
            # Among neighbors, keep the one with largest size
            sizes = [categorized_keypoints[j].size for j in neighbors]
            best_idx = neighbors[np.argmax(sizes)]

            # Mark others for removal
            for j in neighbors:
                if j != best_idx:
                    keep_mask[j] = False

    filtered = [categorized_keypoints[i] for i in range(len(categorized_keypoints)) if keep_mask[i]]
    return filtered, len(filtered)

def draw_keypoints(orig_img, categorized_keypoints):
    """Draw keypoints with different colors for different categories"""
    # Convert categorized keypoints back to regular keypoints for drawing
    regular_keypoints = [ckp.keypoint for ckp in categorized_keypoints]

    processed_img = cv2.drawKeypoints(
        orig_img, regular_keypoints, np.array([]), (255, 0, 0),
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    return processed_img

def create_keypoint_dataframe(categorized_keypoints):
    """Create pandas DataFrame from categorized keypoints"""
    keypoint_dict = {}

    for i, ckp in enumerate(categorized_keypoints):
        keypoint_dict[i] = {
            'coordinates': ckp.pt,
            'size': ckp.size,
            'type': ckp.category
        }

    return pd.DataFrame.from_dict(keypoint_dict, orient='index')

def save_results_with_unique_names(keypoint_df, resized_stitched, filename, save_dir):
    """
    Save all results with unique filenames to avoid overwrites.

    Args:
        keypoint_df (pd.DataFrame): DataFrame containing keypoint data
        resized_stitched (np.array): Processed image with keypoints
        filename (str): Base filename
        save_dir (str): Output directory

    Returns:
        tuple: (csv_path, hist_path, img_path) - actual saved file paths
    """
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Get unique file paths
    csv_path, hist_path, img_path = create_unique_output_paths(save_dir, filename)

    # Save CSV file
    keypoint_df.to_csv(csv_path, index=False)
    print(f"Keypoints saved to: {os.path.basename(csv_path)}")

    # Create and save histogram
    if len(keypoint_df) > 0:
        plt.figure(figsize=(10, 6))
        for debris_type in keypoint_df['type'].unique():
            subset = keypoint_df[keypoint_df['type'] == debris_type]
            plt.hist(subset['size'], bins=10, alpha=0.7, label=debris_type)

        plt.legend()
        plt.title(f'Blob Size by Type for: {filename}')
        plt.xlabel('Size')
        plt.ylabel('Count')
        plt.savefig(hist_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close to free memory
        print(f"Histogram saved to: {os.path.basename(hist_path)}")
    else:
        # Create empty histogram for consistency
        plt.figure(figsize=(10, 6))
        plt.title(f'No blobs detected for: {filename}')
        plt.xlabel('Size')
        plt.ylabel('Count')
        plt.savefig(hist_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Empty histogram saved to: {os.path.basename(hist_path)}")

    # Save processed image
    plt.figure(figsize=(12, 8))
    plt.imshow(resized_stitched)
    plt.axis('off')
    plt.title(f'Detected Debris: {filename}')
    plt.savefig(img_path, bbox_inches='tight', dpi=300)
    plt.close()  # Close to free memory
    print(f"Processed image saved to: {os.path.basename(img_path)}")

    return csv_path, hist_path, img_path

# Get image from console input
parser = argparse.ArgumentParser(description="Open and display an image.")
parser.add_argument("image_path", help="Path to the image file")
parser.add_argument("-c", "--config", help="Path to the config file")
args = parser.parse_args()

image_path = args.image_path
filename = os.path.splitext(os.path.basename(image_path))[0]

# Load configuration
with open(args.config, 'r') as f:
    config = json.load(f)

# Extract parameters
rows, cols = config["rows"], config["columns"]
overlap = config["overlap"]
scale_factor = config["scale_factor"]
threshold_distance = config["threshold_distance"]
save_dir = config["save_directory"]

# Set up blob detection parameters
params = cv2.SimpleBlobDetector_Params()
params.filterByColor = False
params.minThreshold = 0
params.maxThreshold = 255
params.thresholdStep = 5
params.filterByArea = True
params.minArea = config["general_params"]["min_blob_area"]
params.maxArea = config["general_params"]["max_blob_area"]
params.filterByCircularity = False
params.minCircularity = 0.5
params.filterByConvexity = False
params.minConvexity = 0.87
params.filterByInertia = True
params.minInertiaRatio = config["normal_debris_params"]["min_inertia_ratio"]
params.maxInertiaRatio = config["normal_debris_params"]["max_inertia_ratio"]

params_thin = cv2.SimpleBlobDetector_Params()
params_thin.filterByColor = False
params_thin.minThreshold = 0
params_thin.maxThreshold = 255
params_thin.thresholdStep = 5
params_thin.filterByArea = True
params_thin.minArea = config["general_params"]["min_blob_area"]
params_thin.maxArea = config["general_params"]["max_blob_area"]
params_thin.filterByCircularity = False
params_thin.minCircularity = 0.5
params_thin.filterByConvexity = False
params_thin.minConvexity = 0.87
params_thin.filterByInertia = True
params_thin.minInertiaRatio = config["thin_debris_params"]["min_inertia_ratio"]
params_thin.maxInertiaRatio = config["thin_debris_params"]["max_inertia_ratio"]

# Process image
print("Subdividing image...")
sub_images, coords, base_lengths, full_size, img_full = subdivide_image(image_path, rows, cols, overlap)

print("Detecting blobs...")
# Process sub-images with progress bar and category preservation
all_categorized_keypoints = []
local_coord_number = 0

for sub_image in tqdm(sub_images, desc="Processing sub-images"):
    categorized_keypoints = detect_blobs(sub_image, coords, local_coord_number, params, params_thin)
    all_categorized_keypoints.extend(categorized_keypoints)
    local_coord_number += 1

print("Filtering keypoints...")
# Filter keypoints while preserving categories
total_filtered_keypoints, total_num_of_keypoints = filter_keypoints(
    all_categorized_keypoints, threshold_distance
)

print(f"Number of keypoints after filtering: {total_num_of_keypoints}")

# Draw keypoints on original image
processed_full_img = draw_keypoints(np.array(img_full), total_filtered_keypoints)

# Resize image
resized_stitched = cv2.resize(processed_full_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

# Scale keypoints to match resized image
for ckp in total_filtered_keypoints:
    ckp.scale(scale_factor)

# Create DataFrame with categories preserved
keypoint_df = create_keypoint_dataframe(total_filtered_keypoints)

# Save all results with unique filenames
csv_path, hist_path, img_path = save_results_with_unique_names(
    keypoint_df, resized_stitched, filename, save_dir
)

# Print summary by category
if len(keypoint_df) > 0:
    print("\nDetection Summary:")
    category_counts = keypoint_df['type'].value_counts()
    for category, count in category_counts.items():
        print(f"  {category}: {count} detections")
else:
    print("\nNo debris detected in this image.")

print(f"\nAll results saved to: {save_dir}")
print(f"Files created:")
print(f"  - {os.path.basename(csv_path)}")
print(f"  - {os.path.basename(hist_path)}")
print(f"  - {os.path.basename(img_path)}")
