
import os
import json
import cv2
import numpy as np
from collections import defaultdict

#----------------------
# These are all the anatomys in the "anatomy.json" file, which we consider them as two seperate categories, "anatomy" and "Auxiliary tool"
# anatomy counts:
#----------------------
# Anatomy counts:
# organ (ID: 19): 132
# uterus (ID: 20): 478
# ovary (ID: 23): 401
# tube (ID: 22): 154
#_________________________________

def hex2bgr(hex_color):
    """Convert hex color (e.g. '#3fe50f') to a BGR tuple for OpenCV."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        return (255, 255, 255)
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)

# --- Define the effective mapping for anatomy categories ---
# These are the only categories we consider for the "anatomy" dataset.
# For merged classes, all annotations belonging to any of the IDs below
# will be labeled as the merged class name.
anatomy_mapping = {
    # 19: "organ",
    20: "uterus",
    22: "tube",
    23: "ovary",    
}

# --- Load JSON file ---
json_file = "/home/itec/sahar/Domain_Adaptation/Lap_Segmentation/anatomy.json"  # Adjust this path as needed.
with open(json_file, 'r') as f:
    data = json.load(f)

# --- Build a mapping from image id to image info ---
images_info = {}
for img in data.get("images", []):
    images_info[img["id"]] = img

# --- Group annotations by image_id ---
# Only include annotations whose category_id is in our anatomy_mapping.
annotations_grouped = defaultdict(list)
for ann in data.get("annotations", []):
    cat_id = ann.get("category_id")
    if cat_id in anatomy_mapping:
        annotations_grouped[ann["image_id"]].append(ann)

# --- Define base directories ---
# 'input_base' is where the original images are stored.
input_base = "/home/itec/sahar/Domain_Adaptation/Lap_Segmentation/ganseg"  # Adjust this path as needed.
# Output directories for overlays and for copying the original images.
overlay_output_base = "anatomy_overlays"
original_output_base = "anatomy_originals"
os.makedirs(overlay_output_base, exist_ok=True)
os.makedirs(original_output_base, exist_ok=True)

# Blending factor for the overlay polygons.
alpha = 0.4

# --- Process each image that has annotations ---
for image_id, ann_list in annotations_grouped.items():
    img_info = images_info.get(image_id)
    if not img_info:
        print(f"Warning: Image id {image_id} not found in images info.")
        continue

    # Get image details from JSON.
    img_path = img_info["path"]  
    file_name = img_info.get("file_name", os.path.basename(img_path))
    width = img_info.get("width", None)
    height = img_info.get("height", None)

    # Load the original image.
    original_img = cv2.imread(img_path)
    if original_img is None:
        print(f"Warning: Could not load image at {img_path}")
        continue
    if width is None or height is None:
        height, width = original_img.shape[:2]

    # Create an overlay image as a copy of the original.
    overlay_img = original_img.copy()

    # Collect unique effective anatomy names in this image (for naming).
    unique_anatomys = set()

    # Process each annotation on the image.
    for ann in ann_list:
        cat_id = ann.get("category_id")
        effective_name = anatomy_mapping.get(cat_id)
        unique_anatomys.add(effective_name)

        # Get the annotation's hex color (default white if not provided).
        hex_color = ann.get("color", "#FFFFFF")
        bgr_color = hex2bgr(hex_color)

        # Process each segmentation polygon.
        segs = ann.get("segmentation", [])
        for seg in segs:
            pts = np.array(seg).reshape((-1, 2)).astype(np.int32)
            # Create a temporary overlay to fill the polygon.
            temp_overlay = overlay_img.copy()
            cv2.fillPoly(temp_overlay, [pts], bgr_color)
            # Blend the filled polygon with the overlay image.
            overlay_img = cv2.addWeighted(temp_overlay, alpha, overlay_img, 1 - alpha, 0)
            
            # # Compute the centroid to annotate with the anatomy name.
            # if pts.shape[0] > 0:
            #     centroid = pts.mean(axis=0).astype(int)
            #     # Draw text with a dark outline for contrast.
            #     cv2.putText(overlay_img, effective_name, (centroid[0], centroid[1]),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
            #     cv2.putText(overlay_img, effective_name, (centroid[0], centroid[1]),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    # Determine the relative directory (subfolder structure) from the input_base.
    if img_path.startswith(input_base + os.sep):
        rel_path = img_path[len(input_base + os.sep):]
    else:
        rel_path = img_path
    rel_dir = os.path.dirname(rel_path)

    # Create corresponding output directories for overlays and originals.
    overlay_out_dir = os.path.join(overlay_output_base, rel_dir)
    os.makedirs(overlay_out_dir, exist_ok=True)
    original_out_dir = os.path.join(original_output_base, rel_dir)
    os.makedirs(original_out_dir, exist_ok=True)

    # Build the output file name for the overlay image.
    sorted_anatomys = "_".join(sorted(unique_anatomys))
    base, ext = os.path.splitext(file_name)
    overlay_file_name = f"{base}_annotated{ext}"
    overlay_out_path = os.path.join(overlay_out_dir, overlay_file_name)
    
    # Save the overlay image.
    cv2.imwrite(overlay_out_path, overlay_img)
    print(f"Saved anatomy overlay: {overlay_out_path}")
    
    # Also save (or copy) the original image in the new folder structure.
    original_out_path = os.path.join(original_out_dir, file_name)
    cv2.imwrite(original_out_path, original_img)
    print(f"Copied original anatomy image: {original_out_path}")