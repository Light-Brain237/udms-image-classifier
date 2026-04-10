"""
UDMS Image Classifier — Dataset Download Script
Week 1, Day 1-2

What this script does:
Downloads all training datasets needed for the UDMS classifier.
- HuggingFace datasets download automatically
- Kaggle datasets download automatically  
- Mendeley datasets print manual instructions
"""

# --- IMPORTS ---
import os
from pathlib import Path
from datasets import load_dataset  # for HuggingFace
import subprocess                   # for Kaggle CLI

# --- PROJECT PATHS ---
# Path(__file__) = this file's location (src/data/)
# .parent.parent = goes up two levels to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# --- IMPORT CATEGORY MAPPING ---
# Always import categories from the single source of truth
# Never hardcode category names anywhere else
from src.data.category_mapping import (
    ROAD_ISSUES_HF_MAP,
    URBAN_VISUAL_POLLUTION_MAP,
    ROAD_HAZARDS_MAP,
    CIVIC_ISSUES_MAP,
    FLOOD_CLASSIFICATION_MAP,
    DAMAGED_SIGNS_MAP,
)


# ============================================================
# DATASET 1: Road Issues Detection (HuggingFace) — CRITICAL
# Covers categories: illegal_dumping, pothole_road, 
#                    damaged_signage, other
# ============================================================

def download_road_issues_hf():
    """
    Downloads the Road Issues Detection dataset from HuggingFace.
    
    This is the single most important dataset for UDMS.
    It contains 9,660 images across 7 classes — all public domain (CC0).
    
    Images are saved to: data/raw/road_issues_hf/<class_name>/
    """
    print("\n" + "=" * 60)
    print("DATASET 1: Road Issues Detection (HuggingFace)")
    print("Priority: CRITICAL | Images: 9,660 | License: CC0")
    print("=" * 60)

    # Where to save the images
    save_dir = RAW_DATA_DIR / "road_issues_hf"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded to avoid re-downloading
    existing = list(save_dir.rglob("*.jpg")) + list(save_dir.rglob("*.png"))
    if len(existing) > 100:
        print(f"✅ Already downloaded ({len(existing)} images found). Skipping.")
        return

    print("📥 Loading dataset from HuggingFace...")
    print("   This may take a few minutes on first download...")

    try:
        # Load the dataset using the datasets library
        # This is the correct method — NOT snapshot_download
        ds = load_dataset(
            "Programmer-RD-AI/road-issues-detection-dataset",
            split="train"
        )

        print(f"✅ Dataset loaded: {len(ds)} images")
        print("💾 Saving images to disk...")

        saved_count = 0
        skipped_count = 0

        # Loop through every image in the dataset
        for i, item in enumerate(ds):
            # Get the class label for this image
            # Different datasets store labels differently
            label = None
            if "label" in item:
                # Some datasets store label as integer index
                if isinstance(item["label"], int):
                    label = ds.features["label"].int2str(item["label"])
                else:
                    label = str(item["label"])
            elif "labels" in item:
                label = str(item["labels"])

            if label is None:
                skipped_count += 1
                continue

            # Check if this label maps to a UDMS category
            # If it maps to None, we discard it (e.g. Illegal Parking)
            udms_category = ROAD_ISSUES_HF_MAP.get(label)
            if udms_category is None:
                skipped_count += 1
                continue

            # Create folder for this class
            # e.g. data/raw/road_issues_hf/illegal_dumping/
            class_dir = save_dir / label.replace("/", "_").replace(" ", "_")
            class_dir.mkdir(parents=True, exist_ok=True)

            # Save the image
            img_path = class_dir / f"road_issues_{i:05d}.jpg"
            if not img_path.exists():
                item["image"].save(img_path)

            saved_count += 1

            # Show progress every 500 images
            if saved_count % 500 == 0:
                print(f"   Saved {saved_count} images so far...")

        print(f"✅ Done! Saved {saved_count} images, skipped {skipped_count}")
        print(f"📁 Location: {save_dir}")

    except Exception as e:
        print(f"❌ Error downloading Road Issues dataset: {e}")
        print("   Please check your internet connection and try again")


# ============================================================
# DATASET 2-5: Kaggle Datasets
# ============================================================

def download_kaggle_dataset(kaggle_ref, dataset_name, save_folder, priority, notes):
    """
    Downloads a single dataset from Kaggle.
    
    kaggle_ref:   the dataset path e.g. "abhranta/urban-visual-pollution-dataset"
    dataset_name: human readable name for display
    save_folder:  where to save inside data/raw/
    priority:     CRITICAL/HIGH/MEDIUM for display
    notes:        any special notes about this dataset
    """
    print("\n" + "=" * 60)
    print(f"KAGGLE DATASET: {dataset_name}")
    print(f"Priority: {priority}")
    print(f"Notes: {notes}")
    print("=" * 60)

    # Where to save
    save_dir = RAW_DATA_DIR / save_folder
    save_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    existing = list(save_dir.rglob("*.jpg")) + \
               list(save_dir.rglob("*.png")) + \
               list(save_dir.rglob("*.jpeg"))
    if len(existing) > 50:
        print(f"✅ Already downloaded ({len(existing)} images found). Skipping.")
        return

    print(f"📥 Downloading from Kaggle: {kaggle_ref}")
    print("   This may take a few minutes...")

    # This runs the kaggle CLI command:
    # kaggle datasets download -d <ref> --unzip -p <folder>
    command = [
        "kaggle", "datasets", "download",
        "-d", kaggle_ref,
        "--unzip",
        "-p", str(save_dir)
    ]

    result = subprocess.run(command, capture_output=False, text=True)

    if result.returncode == 0:
        # Count downloaded images
        downloaded = list(save_dir.rglob("*.jpg")) + \
                     list(save_dir.rglob("*.png")) + \
                     list(save_dir.rglob("*.jpeg"))
        print(f"✅ Successfully downloaded: {len(downloaded)} images")
        print(f"📁 Location: {save_dir}")
    else:
        print(f"❌ Failed to download: {kaggle_ref}")
        print("   Please check your internet connection and Kaggle credentials")


# ============================================================
# MENDELEY DATASETS — Cannot be automated
# Must be downloaded manually in the browser
# ============================================================

def print_mendeley_instructions():
    """
    Prints manual download instructions for Mendeley datasets.
    
    Mendeley does not have an API for automated downloads.
    The user must download these manually in their browser.
    """
    print("\n" + "=" * 60)
    print("⚠️  MANUAL DOWNLOAD REQUIRED — Mendeley Dataset")
    print("=" * 60)
    print("""
DATASET: Urban Civic Issues — Potholes and Garbage (QR4Change)
Priority: HIGH | Images: 4,937 | Most realistic citizen images

WHY THIS MATTERS:
These images were taken by citizens on smartphones — exactly
like the photos UDMS users will submit. This makes them the
most realistic training data we have.

HOW TO DOWNLOAD:
1. Open your browser and go to:
   https://data.mendeley.com/datasets/zndzygc3p3/2

2. Click the "Download All" button on that page

3. A ZIP file will download to your Downloads folder

4. Extract the ZIP file

5. Copy the extracted folder to:
   data/raw/civic_issues_qr4change/

6. Come back here and continue with the next step
""")


# ============================================================
# MAIN FUNCTION — runs everything in the correct order
# ============================================================

def main():
    """
    Main function that downloads all datasets in priority order.
    
    Order follows the recommended download sequence from the
    Dataset Sources Reference Guide:
    1. Road Issues HF (CRITICAL)
    2. Urban Visual Pollution (CRITICAL) 
    3. Road Hazards (HIGH)
    4. Flood Classification (MEDIUM)
    5. Damaged Signs (MEDIUM)
    6. Mendeley instructions (HIGH - manual)
    """

    print("=" * 60)
    print("UDMS Image Classifier — Dataset Downloader")
    print("Week 1, Day 1-2")
    print("=" * 60)
    print(f"📁 Saving all datasets to: {RAW_DATA_DIR}")

    # Create the main raw data folder
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # --- STEP 1: HuggingFace (CRITICAL) ---
    download_road_issues_hf()

    # --- STEP 2: Kaggle Datasets ---
    download_kaggle_dataset(
        kaggle_ref="abhranta/urban-visual-pollution-dataset",
        dataset_name="Urban Visual Pollution Dataset",
        save_folder="urban_visual_pollution",
        priority="CRITICAL — only source for broken_lighting category",
        notes="Contains BAD_STREETLIGHT class essential for Category 3"
    )

    download_kaggle_dataset(
        kaggle_ref="sabidrahman/pothole-cracks-and-openmanhole",
        dataset_name="Road Hazards — Potholes, Cracks, Manholes",
        save_folder="road_hazards",
        priority="HIGH",
        notes="Covers potholes AND open manholes (water/sewage)"
    )

    download_kaggle_dataset(
        kaggle_ref="dhawalsrivastava2583/flood-classification-dataset",
        dataset_name="Flood Classification Dataset",
        save_folder="flood_classification",
        priority="MEDIUM",
        notes="Water and sewage proxy — filter for urban scenes"
    )

    download_kaggle_dataset(
        kaggle_ref="danielvareta/damaged-signs-dataset",
        dataset_name="Damaged Signs Dataset",
        save_folder="damaged_signs",
        priority="MEDIUM",
        notes="Damaged and deteriorated signage imagery"
    )

    # --- STEP 3: Mendeley (manual) ---
    print_mendeley_instructions()

    # --- DONE ---
    print("\n" + "=" * 60)
    print("✅ Automated downloads complete!")
    print("⚠️  Remember to manually download the Mendeley dataset")
    print("📁 Check data/raw/ folder for all downloaded images")
    print("=" * 60)


# Only run main() if this script is called directly
# not if it is imported by another script
if __name__ == "__main__":
    main()