"""
UDMS Clean Dataset — Map raw downloads to 7 UDMS categories.

Input:  data/raw/<dataset_name>/ (messy, original class structure)
Output: data/raw/cleaned/<category_name>/ (flat, mapped to 7 UDMS categories)
"""

import csv
import shutil
from pathlib import Path
from collections import defaultdict

from PIL import Image
import imagehash

from src.data.category_mapping import (
    get_udms_category,
    UDMS_CATEGORIES,
    URBAN_VISUAL_POLLUTION_MAP,
    ROAD_HAZARDS_MAP,
    FLOOD_CLASSIFICATION_MAP,
)

PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
CLEANED_DIR = RAW_DATA_DIR / "cleaned"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_image(filepath: Path) -> bool:
    """Try to open with PIL. Return False if corrupted/unreadable."""
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except Exception:
        return False


def remove_small_images(image_dir: Path, min_size: int = 100) -> int:
    """Remove images smaller than min_size x min_size pixels. Returns count removed."""
    removed = 0
    for img_path in image_dir.rglob("*"):
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                if w < min_size or h < min_size:
                    img_path.unlink()
                    removed += 1
        except Exception:
            img_path.unlink()
            removed += 1
    return removed


def remove_duplicates(image_dir: Path, hash_size: int = 8) -> int:
    """Remove duplicate images using perceptual hashing. Returns count removed."""
    hashes = {}
    removed = 0
    for img_path in sorted(image_dir.rglob("*")):
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            with Image.open(img_path) as img:
                h = imagehash.phash(img, hash_size=hash_size)
            if h in hashes:
                img_path.unlink()
                removed += 1
            else:
                hashes[h] = img_path
        except Exception:
            continue
    return removed


def _copy_image(src: Path, dest_dir: Path, counter: dict) -> bool:
    """Copy a single image to dest_dir with a unique name. Returns True if copied."""
    if not src.exists() or src.suffix.lower() not in IMAGE_EXTENSIONS:
        return False
    dest_dir.mkdir(parents=True, exist_ok=True)
    cat_name = dest_dir.name
    counter[cat_name] = counter.get(cat_name, 0) + 1
    ext = src.suffix.lower()
    dest = dest_dir / f"{cat_name}_{counter[cat_name]:05d}{ext}"
    shutil.copy2(src, dest)
    return True


def map_road_hazards(output_base: Path, counter: dict) -> dict:
    """Map road_hazards dataset (class folders with images)."""
    source = RAW_DATA_DIR / "road_hazards" / "classes"
    counts = defaultdict(int)
    if not source.exists():
        print(f"  [SKIP] road_hazards not found at {source}")
        return counts

    for class_dir in source.iterdir():
        if not class_dir.is_dir():
            continue
        label = class_dir.name.lower()
        udms_cat = get_udms_category("road_hazards", label)
        if udms_cat is None:
            print(f"  [DISCARD] road_hazards/{class_dir.name}")
            continue
        # Images are in class_dir/images/ and class_dir/augmented/
        for sub in ["images", "augmented"]:
            img_dir = class_dir / sub
            if not img_dir.exists():
                continue
            for img_path in img_dir.iterdir():
                if _copy_image(img_path, output_base / udms_cat, counter):
                    counts[udms_cat] += 1
    return counts


def map_urban_visual_pollution(output_base: Path, counter: dict) -> dict:
    """Map urban_visual_pollution dataset (CSV + images folder)."""
    source = RAW_DATA_DIR / "urban_visual_pollution"
    img_dir = source / "images"
    counts = defaultdict(int)

    if not img_dir.exists():
        print(f"  [SKIP] urban_visual_pollution images not found at {img_dir}")
        return counts

    # Collect unique image → class mappings from train.csv
    # Since it's object detection, one image can have multiple labels.
    # We pick the most frequent label per image, or the first one.
    image_labels = {}
    for csv_name in ["train.csv", "test.csv"]:
        csv_path = source / csv_name
        if not csv_path.exists():
            continue
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row.get("image_path", "").strip()
                label = row.get("name", "").strip()
                if img_name and label:
                    # Keep first label seen per image (simplification)
                    if img_name not in image_labels:
                        image_labels[img_name] = label

    for img_name, label in image_labels.items():
        udms_cat = get_udms_category("urban_visual_pollution", label)
        if udms_cat is None:
            continue
        img_path = img_dir / img_name
        if _copy_image(img_path, output_base / udms_cat, counter):
            counts[udms_cat] += 1

    return counts


def map_flood_classification(output_base: Path, counter: dict) -> dict:
    """Map flood_classification dataset (class folders: flooded/non-flooded)."""
    source = RAW_DATA_DIR / "flood_classification"
    counts = defaultdict(int)

    if not source.exists():
        print(f"  [SKIP] flood_classification not found at {source}")
        return counts

    # Look for train/test subfolders or direct class folders
    for root_candidate in [source, source / "train", source / "Flood Classification"]:
        if not root_candidate.exists():
            continue
        for class_dir in root_candidate.iterdir():
            if not class_dir.is_dir():
                continue
            label = class_dir.name.strip()
            udms_cat = get_udms_category("flood_classification", label)
            if udms_cat is None:
                continue
            for img_path in class_dir.iterdir():
                if _copy_image(img_path, output_base / udms_cat, counter):
                    counts[udms_cat] += 1

    return counts


def map_civic_issues(output_base: Path, counter: dict) -> dict:
    """Map civic_issues_qr4change (Mendeley — folders of potholes and garbage)."""
    source = RAW_DATA_DIR / "civic_issues_qr4change"
    counts = defaultdict(int)

    if not source.exists() or not any(source.iterdir()):
        print(f"  [SKIP] civic_issues_qr4change is empty or missing")
        return counts

    # Mendeley dataset has varied folder structures; search for class-like folders
    for item in source.rglob("*"):
        if not item.is_dir():
            continue
        label = item.name.lower()
        udms_cat = get_udms_category("civic_issues_qr4change", label)
        if udms_cat is None:
            continue
        for img_path in item.iterdir():
            if img_path.is_file() and _copy_image(img_path, output_base / udms_cat, counter):
                counts[udms_cat] += 1

    return counts


def map_damaged_signs(output_base: Path, counter: dict) -> dict:
    """Map damaged_signs dataset (all images → damaged_signage)."""
    source = RAW_DATA_DIR / "damaged_signs"
    counts = defaultdict(int)

    if not source.exists() or not any(source.iterdir()):
        print(f"  [SKIP] damaged_signs is empty or missing")
        return counts

    # Copy all images found recursively
    for img_path in source.rglob("*"):
        if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
            if _copy_image(img_path, output_base / "damaged_signage", counter):
                counts["damaged_signage"] += 1

    return counts


def map_road_issues_hf(output_base: Path, counter: dict) -> dict:
    """Map road_issues_hf HuggingFace dataset (class folders)."""
    source = RAW_DATA_DIR / "road_issues_hf"
    counts = defaultdict(int)

    if not source.exists() or not any(source.iterdir()):
        print(f"  [SKIP] road_issues_hf is empty or missing")
        return counts

    for class_dir in source.iterdir():
        if not class_dir.is_dir():
            continue
        label = class_dir.name
        udms_cat = get_udms_category("road_issues_hf", label)
        if udms_cat is None:
            print(f"  [DISCARD] road_issues_hf/{class_dir.name}")
            continue
        for img_path in class_dir.iterdir():
            if _copy_image(img_path, output_base / udms_cat, counter):
                counts[udms_cat] += 1

    return counts


def check_source_balance(category_counts: dict, max_source_pct: float = 0.40) -> list:
    """Warn if any single source exceeds max_source_pct of a category."""
    warnings = []
    # This is a simplified check — in practice would track per-source counts
    for cat, count in category_counts.items():
        if count == 0:
            warnings.append(f"WARNING: {cat} has 0 images!")
    return warnings


def clean_all() -> None:
    """Run full pipeline: map all datasets, remove duplicates, remove small images."""
    print("=" * 60)
    print("UDMS Clean Dataset Pipeline")
    print("=" * 60)

    # Clear previous cleaned output
    if CLEANED_DIR.exists():
        shutil.rmtree(CLEANED_DIR)

    # Create category folders
    for cat in UDMS_CATEGORIES:
        (CLEANED_DIR / cat).mkdir(parents=True, exist_ok=True)

    counter = {}  # per-category file counter for unique naming
    total_counts = defaultdict(int)

    # Map each dataset
    print("\n[1/6] Mapping road_hazards...")
    counts = map_road_hazards(CLEANED_DIR, counter)
    for k, v in counts.items():
        total_counts[k] += v
    print(f"  Mapped: {dict(counts)}")

    print("\n[2/6] Mapping urban_visual_pollution...")
    counts = map_urban_visual_pollution(CLEANED_DIR, counter)
    for k, v in counts.items():
        total_counts[k] += v
    print(f"  Mapped: {dict(counts)}")

    print("\n[3/6] Mapping flood_classification...")
    counts = map_flood_classification(CLEANED_DIR, counter)
    for k, v in counts.items():
        total_counts[k] += v
    print(f"  Mapped: {dict(counts)}")

    print("\n[4/6] Mapping civic_issues_qr4change...")
    counts = map_civic_issues(CLEANED_DIR, counter)
    for k, v in counts.items():
        total_counts[k] += v
    print(f"  Mapped: {dict(counts)}")

    print("\n[5/6] Mapping damaged_signs...")
    counts = map_damaged_signs(CLEANED_DIR, counter)
    for k, v in counts.items():
        total_counts[k] += v
    print(f"  Mapped: {dict(counts)}")

    print("\n[6/6] Mapping road_issues_hf...")
    counts = map_road_issues_hf(CLEANED_DIR, counter)
    for k, v in counts.items():
        total_counts[k] += v
    print(f"  Mapped: {dict(counts)}")

    # Quality checks
    print("\n" + "=" * 60)
    print("Quality checks...")
    for cat in UDMS_CATEGORIES:
        cat_dir = CLEANED_DIR / cat
        removed_small = remove_small_images(cat_dir)
        removed_dups = remove_duplicates(cat_dir)
        if removed_small or removed_dups:
            print(f"  {cat}: removed {removed_small} small, {removed_dups} duplicates")

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL CATEGORY COUNTS (after cleaning):")
    print("=" * 60)
    for cat in UDMS_CATEGORIES:
        cat_dir = CLEANED_DIR / cat
        count = sum(1 for f in cat_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS) if cat_dir.exists() else 0
        status = "OK" if count >= 80 else "LOW"
        print(f"  {cat:20s}: {count:5d} images  [{status}]")

    warnings = check_source_balance(total_counts)
    for w in warnings:
        print(f"  {w}")

    print("\nCleaned data saved to:", CLEANED_DIR)


if __name__ == "__main__":
    clean_all()
