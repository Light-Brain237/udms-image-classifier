"""
UDMS Image Classifier — Dataset Cleaner
Week 1, Day 2-3

What this script does:
Takes all raw downloaded images and sorts them into
our 7 UDMS category folders under data/processed/all/

Think of it like sorting a huge pile of unsorted
photos into 7 labelled folders.

Run from project root:
python -m src.data.clean_dataset
"""

import os
import shutil
import csv
from pathlib import Path
from PIL import Image
import imagehash

# --- PROJECT PATHS ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# --- IMPORT CATEGORY MAPPINGS ---
# Always import from single source of truth
from src.data.category_mapping import (
    UDMS_CATEGORIES,
    URBAN_VISUAL_POLLUTION_MAP,
    ROAD_HAZARDS_MAP,
    CIVIC_ISSUES_MAP,
    DAMAGED_SIGNS_MAP,
)

# --- GARBAGE CLASSIFICATION MAP ---
# Maps garbage dataset folder names to UDMS categories
GARBAGE_MAP = {
    "biological":  "illegal_dumping",
    "cardboard":   "illegal_dumping",
    "metal":       "illegal_dumping",
    "plastic":     "illegal_dumping",
    "trash":       "illegal_dumping",
    "shoes":       "illegal_dumping",
    "clothes":     "illegal_dumping",
    "brown-glass": "illegal_dumping",
    "green-glass": "illegal_dumping",
    "white-glass": "illegal_dumping",
    "paper":       "illegal_dumping",
    "battery":     None,  # not relevant to street disorder
}

# --- QUALITY CHECKS ---
# Minimum image size — images smaller than this are
# too low quality to be useful for training
MIN_SIZE = 100  # pixels (both width and height)

# Maximum file size to avoid corrupted large files
MAX_SIZE_MB = 20


def is_valid_image(img_path: Path) -> bool:
    """
    Checks if an image is valid and good enough quality.

    Returns True if the image:
    - Can be opened without errors
    - Is at least 100x100 pixels
    - Is not too large (corrupted files are often huge)
    - Is RGB or convertible to RGB
    """
    try:
        # Check file size first (quick check)
        file_size_mb = img_path.stat().st_size / (1024 * 1024)
        if file_size_mb > MAX_SIZE_MB:
            return False

        # Try to open the image
        with Image.open(img_path) as img:
            width, height = img.size

            # Check minimum size
            if width < MIN_SIZE or height < MIN_SIZE:
                return False

        return True

    except Exception:
        # If we cannot open it, it is corrupted
        return False


def copy_image(src_path: Path, category: str, counter: int) -> bool:
    """
    Copies a single image to the correct UDMS category folder.

    src_path: where the image currently is
    category: which UDMS category it belongs to
              e.g. "illegal_dumping"
    counter:  unique number to avoid filename conflicts

    Returns True if successful, False if skipped.
    """
    # Validate the image first
    if not is_valid_image(src_path):
        return False

    # Create destination folder if it does not exist
    # We put ALL images into a single "all" folder first
    # The split_dataset.py script will later divide into
    # train/val/test
    dest_dir = PROCESSED_DIR / "all" / category
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Create a unique filename
    # Format: category_00001.jpg
    dest_filename = f"{category}_{counter:06d}.jpg"
    dest_path = dest_dir / dest_filename

    # Skip if already exists
    if dest_path.exists():
        return True

    try:
        # Open, convert to RGB, and save
        # Converting to RGB handles grayscale and RGBA images
        with Image.open(src_path) as img:
            rgb_img = img.convert("RGB")
            rgb_img.save(dest_path, "JPEG", quality=95)
        return True

    except Exception as e:
        print(f"   ⚠️  Could not copy {src_path.name}: {e}")
        return False


# ============================================================
# DATASET CLEANERS — one function per dataset
# ============================================================

def clean_urban_visual_pollution():
    """
    Cleans the Urban Visual Pollution dataset.

    This dataset stores all images in one flat folder.
    A CSV file (train.csv) tells us what class each image is.

    Structure:
    urban_visual_pollution/
      images/
        abc123.jpg
        def456.jpg
        ...
      train.csv  ← contains: filename, label
    """
    print("\n" + "=" * 60)
    print("Cleaning: Urban Visual Pollution Dataset")
    print("=" * 60)

    dataset_dir = RAW_DATA_DIR / "urban_visual_pollution"
    images_dir = dataset_dir / "images"
    csv_path = dataset_dir / "train.csv"

    if not csv_path.exists():
        print(f"❌ train.csv not found at {csv_path}")
        return 0, 0

    if not images_dir.exists():
        print(f"❌ images folder not found at {images_dir}")
        return 0, 0

    saved = 0
    skipped = 0
    counter_by_category = {cat: 0 for cat in UDMS_CATEGORIES}
    processed_images = set()  # track already processed images

    print(f"📖 Reading CSV file...")

    # Read the CSV file to get image labels
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # This CSV uses "image_path" for filename
            # and "name" for the class label
            filename = row.get("image_path")
            label = row.get("name")

            if not filename or not label:
                skipped += 1
                continue

            # Clean up the label
            label = label.strip().upper()

            # Map to UDMS category
            udms_category = URBAN_VISUAL_POLLUTION_MAP.get(label)
            if udms_category is None:
                skipped += 1
                continue

            # Find the image file
            img_path = images_dir / filename
            if not img_path.exists():
                # Try with .jpg extension
                img_path = images_dir / f"{filename}.jpg"
            if not img_path.exists():
                skipped += 1
                continue

            # Skip if we already processed this image
            # Same image can appear multiple times in CSV
            # because each row is a bounding box detection
            if filename in processed_images:
                continue
            processed_images.add(filename)

            # Copy to processed folder
            counter_by_category[udms_category] += 1
            success = copy_image(
                img_path,
                udms_category,
                counter_by_category[udms_category]
            )

            if success:
                saved += 1
            else:
                skipped += 1

            # Show progress every 1000 images
            if (saved + skipped) % 1000 == 0:
                print(f"   Processed {saved + skipped} images...")

    print(f"✅ Saved: {saved} | Skipped: {skipped}")
    return saved, skipped


def clean_road_hazards():
    """
    Cleans the Road Hazards dataset.

    This dataset has a classes/ folder with subfolders
    named by class. Images are inside each class folder.

    Structure:
    road_hazards/
      classes/
        pothole/      → pothole_road
        cracks/       → pothole_road
        open_manhole/ → water_sewage
        good_road/    → discard
    """
    print("\n" + "=" * 60)
    print("Cleaning: Road Hazards Dataset")
    print("=" * 60)

    classes_dir = RAW_DATA_DIR / "road_hazards" / "classes"

    if not classes_dir.exists():
        print(f"❌ classes folder not found at {classes_dir}")
        return 0, 0

    saved = 0
    skipped = 0
    counter_by_category = {cat: 1000 for cat in UDMS_CATEGORIES}

    # Loop through each class folder
    for class_folder in classes_dir.iterdir():
        if not class_folder.is_dir():
            continue

        class_name = class_folder.name.lower()

        # Map to UDMS category
        udms_category = ROAD_HAZARDS_MAP.get(class_name)
        if udms_category is None:
            print(f"   ⏭️  Discarding class: {class_name}")
            continue

        print(f"   📁 Processing: {class_name} → {udms_category}")

        # Copy all images in this folder
        for img_path in class_folder.rglob("*.jpg"):
            counter_by_category[udms_category] += 1
            success = copy_image(
                img_path,
                udms_category,
                counter_by_category[udms_category]
            )
            if success:
                saved += 1
            else:
                skipped += 1

        # Also check for PNG images
        for img_path in class_folder.rglob("*.png"):
            counter_by_category[udms_category] += 1
            success = copy_image(
                img_path,
                udms_category,
                counter_by_category[udms_category]
            )
            if success:
                saved += 1
            else:
                skipped += 1

    print(f"✅ Saved: {saved} | Skipped: {skipped}")
    return saved, skipped


def clean_garbage_classification():
    """
    Cleans the Garbage Classification dataset.

    Images are already sorted into folders by class name.
    We map each folder name to illegal_dumping category.

    Structure:
    garbage_classification_2/
      garbage_classification/
        biological/
        cardboard/
        metal/
        plastic/
        trash/
        ...
    """
    print("\n" + "=" * 60)
    print("Cleaning: Garbage Classification Dataset")
    print("=" * 60)

    garbage_dir = RAW_DATA_DIR / "garbage_classification_2" / \
                  "garbage_classification"

    if not garbage_dir.exists():
        print(f"❌ Folder not found at {garbage_dir}")
        return 0, 0

    saved = 0
    skipped = 0
    counter = 5000  # start from 5000 to avoid conflicts

    # Loop through each class folder
    for class_folder in garbage_dir.iterdir():
        if not class_folder.is_dir():
            continue

        class_name = class_folder.name.lower()

        # Map to UDMS category
        udms_category = GARBAGE_MAP.get(class_name)
        if udms_category is None:
            print(f"   ⏭️  Discarding class: {class_name}")
            continue

        print(f"   📁 Processing: {class_name} → {udms_category}")

        # Copy all images
        for img_path in class_folder.rglob("*.jpg"):
            counter += 1
            success = copy_image(img_path, udms_category, counter)
            if success:
                saved += 1
            else:
                skipped += 1

    print(f"✅ Saved: {saved} | Skipped: {skipped}")
    return saved, skipped


def clean_damaged_signs():
    """
    Cleans the Damaged Signs dataset.

    All images in this dataset are damaged signs.
    Everything maps to damaged_signage category.

    Structure:
    damaged_signs/
      damaged_signs_dataset/
        dataset/
          ← all images here
    """
    print("\n" + "=" * 60)
    print("Cleaning: Damaged Signs Dataset")
    print("=" * 60)

    dataset_dir = RAW_DATA_DIR / "damaged_signs" / \
                  "damaged_signs_dataset" / "dataset"

    if not dataset_dir.exists():
        print(f"❌ Dataset folder not found at {dataset_dir}")
        return 0, 0

    saved = 0
    skipped = 0
    counter = 10000  # start from 10000 to avoid conflicts

    # All images map to damaged_signage
    for img_path in dataset_dir.rglob("*.jpg"):
        counter += 1
        success = copy_image(img_path, "damaged_signage", counter)
        if success:
            saved += 1
        else:
            skipped += 1

    # Also check PNG
    for img_path in dataset_dir.rglob("*.png"):
        counter += 1
        success = copy_image(img_path, "damaged_signage", counter)
        if success:
            saved += 1
        else:
            skipped += 1

    print(f"✅ Saved: {saved} | Skipped: {skipped}")
    return saved, skipped


# ============================================================
# REPORT GENERATOR
# ============================================================

def generate_summary_report(total_saved, total_skipped):
    """
    Counts images per category and prints a summary report.
    Tells us if we have enough images per category.
    """
    print("\n" + "=" * 60)
    print("DATASET SUMMARY REPORT")
    print("=" * 60)

    all_dir = PROCESSED_DIR / "all"

    # Minimum images needed per category
    minimums = {
        "illegal_dumping": 150,
        "pothole_road":    150,
        "broken_lighting": 100,
        "water_sewage":    100,
        "damaged_signage":  80,
        "vegetation":       80,
        "other":            50,
    }

    total = 0
    for category in UDMS_CATEGORIES:
        cat_dir = all_dir / category
        if cat_dir.exists():
            count = len(list(cat_dir.glob("*.jpg")))
        else:
            count = 0

        minimum = minimums.get(category, 100)
        status = "✅" if count >= minimum else "⚠️ NEEDS MORE DATA"

        print(f"{status} {category:<20} {count:>5} images "
              f"(min: {minimum})")
        total += count

    print(f"\n{'─' * 60}")
    print(f"Total images: {total}")
    print(f"Total saved:  {total_saved}")
    print(f"Total skipped: {total_skipped}")
    print("=" * 60)

    # Warn about categories with low data
    print("\n⚠️  CATEGORIES NEEDING ATTENTION:")
    for category in UDMS_CATEGORIES:
        cat_dir = all_dir / category
        count = len(list(cat_dir.glob("*.jpg"))) \
                if cat_dir.exists() else 0
        minimum = minimums.get(category, 100)
        if count < minimum:
            shortage = minimum - count
            print(f"   {category}: needs {shortage} more images")


# ============================================================
# MAIN FUNCTION
# ============================================================

def clean_web_scraped():
    """
    Adds manually reviewed web scraped images to the dataset.

    These images were scraped from Bing for weak categories:
    - vegetation (overgrown urban areas)
    - broken_lighting (damaged streetlights)

    All images have been manually reviewed before this step.
    """
    print("\n" + "=" * 60)
    print("Cleaning: Web Scraped Images")
    print("=" * 60)

    scraped_dir = RAW_DATA_DIR / "web_scraped"

    if not scraped_dir.exists():
        print(f"❌ Web scraped folder not found at {scraped_dir}")
        return 0, 0

    saved = 0
    skipped = 0

    # Map scraper folder names to UDMS categories
    category_map = {
        "vegetation": "vegetation",
        "broken_lighting": "broken_lighting",
        "water_sewage": "water_sewage",
    }

    for folder_name, udms_category in category_map.items():
        category_dir = scraped_dir / folder_name

        if not category_dir.exists():
            print(f"   ⚠️  No folder found for: {folder_name}")
            continue

        print(f"\n   📁 Processing: {folder_name} → {udms_category}")

        counter = 20000  # start high to avoid conflicts

        # Loop through all subfolders
        for subfolder in category_dir.iterdir():
            if not subfolder.is_dir():
                continue

            for img_path in subfolder.rglob("*.jpg"):
                counter += 1
                success = copy_image(img_path, udms_category, counter)
                if success:
                    saved += 1
                else:
                    skipped += 1

            for img_path in subfolder.rglob("*.jpeg"):
                counter += 1
                success = copy_image(img_path, udms_category, counter)
                if success:
                    saved += 1
                else:
                    skipped += 1

            for img_path in subfolder.rglob("*.png"):
                counter += 1
                success = copy_image(img_path, udms_category, counter)
                if success:
                    saved += 1
                else:
                    skipped += 1

    print(f"✅ Saved: {saved} | Skipped: {skipped}")
    return saved, skipped


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    """
    Runs all dataset cleaners in sequence.
    """
    print("=" * 60)
    print("UDMS Image Classifier — Dataset Cleaner")
    print("Week 1, Day 2-3")
    print("=" * 60)
    print(f"Output folder: {PROCESSED_DIR / 'all'}")

    # Create output folder
    (PROCESSED_DIR / "all").mkdir(parents=True, exist_ok=True)

    # Create subfolders for all 7 categories
    for category in UDMS_CATEGORIES:
        (PROCESSED_DIR / "all" / category).mkdir(
            parents=True, exist_ok=True
        )

    total_saved = 0
    total_skipped = 0

    # Run each cleaner in order
    saved, skipped = clean_urban_visual_pollution()
    total_saved += saved
    total_skipped += skipped

    saved, skipped = clean_road_hazards()
    total_saved += saved
    total_skipped += skipped

    saved, skipped = clean_garbage_classification()
    total_saved += saved
    total_skipped += skipped

    saved, skipped = clean_damaged_signs()
    total_saved += saved
    total_skipped += skipped

    # Add web scraped images for weak categories
    saved, skipped = clean_web_scraped()
    total_saved += saved
    total_skipped += skipped

    # Generate summary report
    generate_summary_report(total_saved, total_skipped)

    print("\n✅ Cleaning complete!")
    print(f"📁 Check data/processed/all/ for sorted images")


if __name__ == "__main__":
    main()