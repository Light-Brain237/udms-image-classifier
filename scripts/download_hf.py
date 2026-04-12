"""
Quick script to download the Road Issues HuggingFace dataset.
Run this from the project root:
python scripts/download_hf.py
"""

from datasets import load_dataset
from pathlib import Path

# Where to save images
save_dir = Path("data/raw/road_issues_hf")
save_dir.mkdir(parents=True, exist_ok=True)

# Check if already downloaded
existing = list(save_dir.rglob("*.jpg"))
if len(existing) > 100:
    print(f"Already downloaded {len(existing)} images. Done!")
else:
    print("Loading dataset from HuggingFace...")
    ds = load_dataset(
        "Programmer-RD-AI/road-issues-detection-dataset",
        split="train"
    )
    print(f"Dataset loaded: {len(ds)} images")
    print("Saving images to disk...")

    saved = 0
    for i, item in enumerate(ds):

        # Get the label
        if isinstance(item.get("label"), int):
            label = ds.features["label"].int2str(item["label"])
        else:
            label = str(item.get("label", "unknown"))

        # Clean label for folder name
        clean_label = label.replace("/", "_").replace(" ", "_")[:50]

        # Create folder for this class
        class_dir = save_dir / clean_label
        class_dir.mkdir(parents=True, exist_ok=True)

        # Save image
        img_path = class_dir / f"img_{i:05d}.jpg"
        if not img_path.exists():
            item["image"].save(img_path)

        saved += 1

        # Show progress every 500 images
        if saved % 500 == 0:
            print(f"Saved {saved} images...")

    print(f"Done! Saved {saved} images")
    print(f"Location: {save_dir}")