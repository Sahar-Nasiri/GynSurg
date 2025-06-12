import os
import random
import csv
import numpy as np

# Set the base dataset folder.
dataset_base = "Lap_anatomy_dataset"

# Define the image and mask base directories.
images_base = os.path.join(dataset_base, "ganseg")
masks_base = os.path.join(dataset_base, "ganseg_mask")

# --- Step 1. Collect all patient directories ---
# Patient directories are assumed to be subdirectories within each GANSEG_* folder.
patient_dirs = []

# List all subfolders (e.g., "GANSEG_01", "GANSEG_02", etc.) in images_base.
genseg_subfolders = [os.path.join(images_base, d) for d in os.listdir(images_base)
                     if os.path.isdir(os.path.join(images_base, d))]

# For each GANSEG_* folder, list the patient (sub-subfolder) directories.
for folder in genseg_subfolders:
    patients = [os.path.join(folder, subfolder) for subfolder in os.listdir(folder)
                if os.path.isdir(os.path.join(folder, subfolder))]
    patient_dirs.extend(patients)

print(f"Found {len(patient_dirs)} patients.")

# --- Step 2. Shuffle and divide patient directories into 4 folds ---
random.shuffle(patient_dirs)
# Use numpy's array_split to deal with uneven splits.
folds = np.array_split(patient_dirs, 4)

# --- Helper function to get CSV rows from a set of patient directories ---
def collect_rows(patient_list, images_base, masks_base, valid_ext=(".jpg", ".jpeg", ".png", ".bmp")):
    """
    For each patient in the list, this function lists image files (if they have a valid
    extension) and constructs a row of [index, image_path, mask_path].
    
    It assumes the folder structure is mirrored in the mask folder.
    """
    rows = []
    index = 0
    for patient_dir in sorted(patient_list):
        # Compute the relative path from images_base, to locate the corresponding mask folder.
        rel_path = os.path.relpath(patient_dir, images_base)
        mask_patient_dir = os.path.join(masks_base, rel_path)
        
        # List and sort image files (modify the valid_ext tuple as needed).
        for file in sorted(os.listdir(patient_dir)):
            # Decode filename if needed.
            if isinstance(file, bytes):
                file = file.decode("utf-8")
            if file.lower().endswith(valid_ext):
                image_path = os.path.join(patient_dir, file)
                mask_path = os.path.join(mask_patient_dir, file)
                base, ext = os.path.splitext(file)
                mask_file = f"{base}_mask{ext}"
                mask_path = os.path.join(mask_patient_dir, mask_file)
                # Convert to absolute paths.
                image_path_abs = os.path.abspath(image_path)
                mask_path_abs = os.path.abspath(mask_path)
                rows.append([index, image_path_abs, mask_path_abs])
                index += 1
    return rows

# --- Step 3. Create CSV files for each fold ---
# For each fold, use the patients in that fold as the test set and the rest as the train set.
for i in range(4):
    test_patients = list(folds[i])
    # Combine all patients in the other folds for training.
    train_patients = []
    for j in range(4):
        if j != i:
            train_patients.extend(list(folds[j]))

    # Collect CSV rows.
    test_rows = collect_rows(test_patients, images_base, masks_base)
    train_rows = collect_rows(train_patients, images_base, masks_base)
    
    # Define CSV filenames. These files will be saved in the current working directory.
    train_csv = f"Lap_anatomy_train_{i}.csv"
    test_csv  = f"Lap_anatomy_test_{i}.csv"
    
    # Write train CSV.
    with open(train_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["", "imgs", "masks"])
        writer.writerows(train_rows)
    
    # Write test CSV.
    with open(test_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["", "imgs", "masks"])
        writer.writerows(test_rows)
    
    print(f"Fold {i+1} CSV files saved: {train_csv} (train), {test_csv} (test)")

    # Print summary for fold i.
    print(f"========== Fold {i} ==========")
    
    # Print Test set details.
    print(f"Test Set: {len(test_patients)} patient directories")
    for patient in sorted(test_patients):
        # Convert full path to relative path with respect to images_base.
        rel_path = os.path.relpath(patient, images_base)
        # The relative path will typically be: GANSEG_xx/0.mp4_ etc.
        print(f"  {rel_path}")
    
    # Print Train set details.
    print(f"\nTrain Set: {len(train_patients)} patient directories")
    for patient in sorted(train_patients):
        rel_path = os.path.relpath(patient, images_base)
        print(f"  {rel_path}")
    print("\n")