
import os
import json
import cv2
import numpy as np
from collections import defaultdict

#----------------------
# These are all the instruments in the "instrument.json" file, which we consider them as two seperate categories, "Instrument" and "Auxiliary tool"
# Instrument counts:
#----------------------
# grasper (ID: 2): 3382             # "Instrument"
# glove (ID: 31): 62
# scissors (ID: 10): 477            # "Instrument"
# colpotomizer (ID: 30): 76
# irrigator (ID: 3): 1107           # "Instrument"
# in-cannula (ID: 14): 131          # "Auxiliary tool"
# needle-holder (ID: 12): 957       # "Instrument"
# needle (ID: 6): 572               # "Instrument"
# thread (ID: 9): 1470              # "Auxiliary tool"
# bipolar-forceps (ID: 5): 630      # "Instrument"
# thread-fragment (ID: 18): 3717    # "Auxiliary tool"
# trocar-sleeve (ID: 27): 84        # "Auxiliary tool"
# knot-pusher (ID: 13): 90          # "Instrument"
# suture-carrier (ID: 16): 28       # "Instrument"
# sealer-divider (ID: 7): 1210      # "Instrument"
# hook (ID: 11): 241                # "Instrument"
# clip (ID: 17): 300
# clip-applier (ID: 15): 54
# cannula (ID: 28): 2               # "Auxiliary tool"
# corkscrew (ID: 29): 23
# trocar (ID: 8): 8
# morcellator (ID: 4): 292          # "Auxiliary tool"
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

# --- Define the effective mapping for instrument categories ---
# These are the only categories we consider for the "Instrument" dataset.
# For merged classes, all annotations belonging to any of the IDs below
# will be labeled as the merged class name.
Instrument_mapping = {
    # Remain as individual classes:
    2: "grasper",
    3: "irrigator",
    5: "bipolar-forceps",
    7: "sealer-divider",
    10: "scissors",
    11: "hook",
    # Merged as "suturing-instrument":
    12: "suturing-instrument",
    6:  "suturing-instrument",
    13: "suturing-instrument",
    16: "suturing-instrument",
}

# --- Load JSON file ---
json_file = "/home/itec/sahar/Domain_Adaptation/Lap_Segmentation_1/instruments.json"  # Adjust this path as needed.
with open(json_file, 'r') as f:
    data = json.load(f)

# --- Build a mapping from image id to image info ---
images_info = {}
for img in data.get("images", []):
    images_info[img["id"]] = img

# --- Group annotations by image_id ---
# Only include annotations whose category_id is in our Instrument_mapping.
annotations_grouped = defaultdict(list)
for ann in data.get("annotations", []):
    cat_id = ann.get("category_id")
    if cat_id in Instrument_mapping:
        annotations_grouped[ann["image_id"]].append(ann)

# --- Define base directories ---
# 'input_base' is where the original images are stored.
input_base = "/home/itec/sahar/Domain_Adaptation/Lap_Segmentation_1/insseg"  # Adjust this path as needed.
# Output directories for overlays and for copying the original images.
overlay_output_base = "instrument_overlays"
original_output_base = "instrument_originals"
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

    # Collect unique effective instrument names in this image (for naming).
    unique_instruments = set()

    # Process each annotation on the image.
    for ann in ann_list:
        cat_id = ann.get("category_id")
        effective_name = Instrument_mapping.get(cat_id)
        unique_instruments.add(effective_name)

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
            
            # # Compute the centroid to annotate with the instrument name.
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
    sorted_instruments = "_".join(sorted(unique_instruments))
    base, ext = os.path.splitext(file_name)
    overlay_file_name = f"{base}_annotated{ext}"
    overlay_out_path = os.path.join(overlay_out_dir, overlay_file_name)
    
    # Save the overlay image.
    cv2.imwrite(overlay_out_path, overlay_img)
    print(f"Saved instrument overlay: {overlay_out_path}")
    
    # Also save (or copy) the original image in the new folder structure.
    original_out_path = os.path.join(original_out_dir, file_name)
    cv2.imwrite(original_out_path, original_img)
    print(f"Copied original instrument image: {original_out_path}")