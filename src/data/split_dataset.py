"""
UDMS Split Dataset — Split cleaned images into train/val/test (70/15/15).

Input:  data/processed/all/<category>/ (flat image folders from clean_dataset.py)
Output: data/processed/train/<category>/, val/<category>/, test/<category>/

CRITICAL: Uses stratified split to maintain class proportions.
CRITICAL: random_state=42 for reproducibility.
"""

import shutil
from pathlib import Path
from collections import defaultdict

from sklearn.model_selection import train_test_split

from src.data.category_mapping import UDMS_CATEGORIES

PROJECT_ROOT = Path(__file__).parent.parent.parent
SOURCE_DIR = PROJECT_ROOT / "data" / "processed" / "all"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42


def clear_split_dirs(output_dir: Path = OUTPUT_DIR) -> None:
    """Remove and recreate train/val/test category folders.

    This prevents data leakage from stale images left over by previous runs.
    Only the train/, val/, test/ subdirectories are cleared — the all/ source
    directory is never touched.
    """
    for split_name in ["train", "val", "test"]:
        split_dir = output_dir / split_name
        if split_dir.exists():
            shutil.rmtree(split_dir)
        for cat in UDMS_CATEGORIES:
            (split_dir / cat).mkdir(parents=True, exist_ok=True)
    print("  Cleared existing train/val/test folders.")


def split_dataset(source_dir: Path = SOURCE_DIR, output_dir: Path = OUTPUT_DIR) -> dict:
    """Split each category 70/15/15 into train/val/test.

    Reads images from data/processed/all/<category>/ and copies them
    into data/processed/train/, val/, test/ subdirectories.
    Uses scikit-learn train_test_split with random_state=42.
    Images are copied (not moved) to preserve originals.

    IMPORTANT: Clears existing train/val/test folders first to prevent
    data leakage from stale images left by previous runs.

    Returns: {category: {train: N, val: N, test: N}}
    """
    clear_split_dirs(output_dir)

    results = {}

    for cat in UDMS_CATEGORIES:
        cat_dir = source_dir / cat
        if not cat_dir.exists():
            print(f"  [SKIP] {cat} — folder not found")
            results[cat] = {"train": 0, "val": 0, "test": 0}
            continue

        images = sorted([
            f for f in cat_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ])

        if len(images) < 3:
            print(f"  [SKIP] {cat} — only {len(images)} images (need ≥3)")
            results[cat] = {"train": 0, "val": 0, "test": 0}
            continue

        # First split: 70% train, 30% temp
        train_imgs, temp_imgs = train_test_split(
            images, test_size=(1 - TRAIN_RATIO), random_state=RANDOM_STATE
        )

        # Second split: 50/50 on temp → 15% val, 15% test
        val_imgs, test_imgs = train_test_split(
            temp_imgs, test_size=0.5, random_state=RANDOM_STATE
        )

        # Copy to output directories
        for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            dest = output_dir / split_name / cat
            dest.mkdir(parents=True, exist_ok=True)
            for img in split_imgs:
                shutil.copy2(img, dest / img.name)

        results[cat] = {
            "train": len(train_imgs),
            "val": len(val_imgs),
            "test": len(test_imgs),
        }
        total = len(train_imgs) + len(val_imgs) + len(test_imgs)
        print(f"  {cat:20s}: train={len(train_imgs):4d}  val={len(val_imgs):4d}  test={len(test_imgs):4d}  total={total}")

    return results


def verify_split(output_dir: Path = OUTPUT_DIR) -> bool:
    """Verify no image appears in more than one split and all categories exist."""
    all_ok = True

    for cat in UDMS_CATEGORIES:
        split_files = {}
        for split_name in ["train", "val", "test"]:
            split_dir = output_dir / split_name / cat
            if not split_dir.exists():
                print(f"  [MISSING] {split_name}/{cat}")
                all_ok = False
                continue
            files = {f.name for f in split_dir.iterdir() if f.is_file()}
            split_files[split_name] = files

        # Check for overlap
        for s1 in ["train", "val", "test"]:
            for s2 in ["train", "val", "test"]:
                if s1 >= s2:
                    continue
                if s1 in split_files and s2 in split_files:
                    overlap = split_files[s1] & split_files[s2]
                    if overlap:
                        print(f"  [LEAK] {cat}: {len(overlap)} images in both {s1} and {s2}")
                        all_ok = False

    if all_ok:
        print("  All checks passed — no data leakage detected.")
    else:
        raise RuntimeError("Data leakage detected! See messages above.")
    return all_ok


def main():
    print("=" * 60)
    print("UDMS Dataset Splitter (70/15/15)")
    print("=" * 60)

    print(f"\nSource: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}\n")

    results = split_dataset()

    print("\nVerifying split integrity...")
    verify_split()

    print("\nDone. Processed data is in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
