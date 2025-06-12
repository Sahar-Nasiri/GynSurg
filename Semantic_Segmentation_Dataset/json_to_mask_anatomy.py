# This script processes a JSON file containing anatomy annotations
# and generates multi class mask images for each anatomy class defined in the mapping.
import os
import json
import cv2
import numpy as np
from collections import defaultdict


# --- Define the mask label mapping for the 3 anatomy classes ---
mask_label_mapping = {
    # "organ": 50,
    "uterus": 85,
    "tube": 170,
    "ovary": 255,
}

# --- Define the effective mapping for anatomy annotations ---
# These mappings merge the raw anatomy category IDs into the effective names.
anatomy_mapping = {
    # 19: "organ",
    20: "uterus",
    22: "tube",
    23: "ovary",    
}

# --- Load the anatomy annotation JSON file ---
json_file = "/home/itec/sahar/Domain_Adaptation/Lap_Segmentation/anatomy.json"  # Adjust the path as needed.
with open(json_file, "r") as f:
    data = json.load(f)

# --- Build a mapping from image id to image information ---
images_info = {}
for img in data.get("images", []):
    images_info[img["id"]] = img

# --- Group annotations by image_id ---
# Only include annotations with category_ids present in our anatomy_mapping.
annotations_grouped = defaultdict(list)
for ann in data.get("annotations", []):
    cat_id = ann.get("category_id")
    if cat_id in anatomy_mapping:
        annotations_grouped[ann["image_id"]].append(ann)

# --- Define base directories ---
# 'input_base' is where the original images are stored.
input_base = "/home/itec/sahar/Domain_Adaptation/Lap_Segmentation/ganseg"  # Adjust if needed.
# 'mask_output_base' is where the anatomy masks will be saved.
mask_output_base = "anatomy_mask"
os.makedirs(mask_output_base, exist_ok=True)

# --- Process each image that has anatomy annotations ---
for image_id, ann_list in annotations_grouped.items():
    img_info = images_info.get(image_id)
    if not img_info:
        print(f"Warning: Image id {image_id} not found.")
        continue

    # Get image details from JSON.
    img_path = img_info["path"]  # e.g., "ganseg/GANSEG_01/..."
    file_name = img_info.get("file_name", os.path.basename(img_path))
    width = img_info.get("width", None)
    height = img_info.get("height", None)

    # Load the original image to determine dimensions (if not provided)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not load image at {img_path}")
        continue
    if width is None or height is None:
        height, width = img.shape[:2]

    # Create a blank mask (background = 0)
    mask = np.zeros((height, width), dtype=np.uint8)

    # For each annotation, draw the segmentation polygon with its label value.
    for ann in ann_list:
        cat_id = ann.get("category_id")
        effective_name = anatomy_mapping.get(cat_id)
        # Get the label value from our mask label mapping.
        label_value = mask_label_mapping.get(effective_name, 0)
        segs = ann.get("segmentation", [])
        for seg in segs:
            pts = np.array(seg).reshape((-1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [pts], color=int(label_value))

    # Determine the relative directory structure from the input_base.
    if img_path.startswith(input_base + os.sep):
        rel_path = img_path[len(input_base + os.sep):]
    else:
        rel_path = img_path
    rel_dir = os.path.dirname(rel_path)

    # Create the corresponding output directory for the masks.
    out_dir = os.path.join(mask_output_base, rel_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Build the output file name: append "_mask" to the original file name.
    base, ext = os.path.splitext(file_name)
    out_file_name = f"{base}_mask.png"
    out_path = os.path.join(out_dir, out_file_name)

    # Save the mask image (8-bit single channel)
    cv2.imwrite(out_path, mask)
    print(f"Saved anatomy mask: {out_path}")