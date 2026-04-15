"""
UDMS Dataset Report Generator

Input:  data/processed/train|val|test/<category>/
Output: data/dataset_report.md  (with embedded charts)

Contents:
- Per-category image counts (train, val, test)
- Bar chart of class distribution
- Sample grid: 3 random images per category
- Warnings for imbalanced categories
"""

import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.data.category_mapping import UDMS_CATEGORIES

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Minimum images needed per category
MINIMUMS = {
    "illegal_dumping": 150,
    "pothole_road": 150,
    "broken_lighting": 100,
    "water_sewage": 100,
    "damaged_signage": 80,
    "vegetation": 80,
    "other": 50,
}


def count_images(data_dir: Path) -> dict:
    """Count images per category per split. Returns nested dict."""
    counts = {}
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        counts[split] = {}
        for cat in UDMS_CATEGORIES:
            cat_dir = split_dir / cat
            if cat_dir.exists():
                n = sum(1 for f in cat_dir.iterdir()
                        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)
            else:
                n = 0
            counts[split][cat] = n
    return counts


def plot_distribution(counts: dict, output_path: Path) -> None:
    """Bar chart: x=categories, y=counts, grouped by split. Save as PNG."""
    splits = ["train", "val", "test"]
    x = np.arange(len(UDMS_CATEGORIES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, split in enumerate(splits):
        values = [counts[split].get(cat, 0) for cat in UDMS_CATEGORIES]
        ax.bar(x + i * width, values, width, label=split)

    ax.set_xlabel("Category")
    ax.set_ylabel("Image Count")
    ax.set_title("UDMS Dataset — Class Distribution by Split")
    ax.set_xticks(x + width)
    ax.set_xticklabels([c.replace("_", "\n") for c in UDMS_CATEGORIES],
                       fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def create_sample_grid(data_dir: Path, output_path: Path,
                       samples_per_class: int = 3) -> None:
    """Grid of sample images, one row per category. Save as PNG."""
    fig, axes = plt.subplots(
        len(UDMS_CATEGORIES), samples_per_class,
        figsize=(samples_per_class * 3, len(UDMS_CATEGORIES) * 3),
    )

    for row, cat in enumerate(UDMS_CATEGORIES):
        cat_dir = data_dir / "train" / cat
        images = []
        if cat_dir.exists():
            images = [f for f in cat_dir.iterdir()
                      if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]

        sampled = random.sample(images, min(samples_per_class, len(images)))

        for col in range(samples_per_class):
            ax = axes[row][col] if len(UDMS_CATEGORIES) > 1 else axes[col]
            if col < len(sampled):
                img = Image.open(sampled[col]).convert("RGB").resize((224, 224))
                ax.imshow(img)
            ax.axis("off")
            if col == 0:
                ax.set_title(cat.replace("_", " "), fontsize=9, loc="left")

    fig.suptitle("UDMS Dataset — Sample Images per Category", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def check_imbalance(counts: dict) -> list:
    """Return list of warnings for categories below minimum thresholds."""
    warnings = []
    for cat in UDMS_CATEGORIES:
        total = sum(counts[split].get(cat, 0) for split in ["train", "val", "test"])
        minimum = MINIMUMS.get(cat, 80)
        if total < minimum:
            shortage = minimum - total
            warnings.append(
                f"⚠️  **{cat}**: {total} images (need {minimum}, short by {shortage})"
            )
    return warnings


def generate_report(data_dir: Path = DATA_DIR,
                    output_path: Path | None = None) -> None:
    """Orchestrate: count, plot, sample, check, write markdown."""
    if output_path is None:
        output_path = PROJECT_ROOT / "data" / "dataset_report.md"

    counts = count_images(data_dir)

    # Generate chart
    chart_path = output_path.parent / "class_distribution.png"
    plot_distribution(counts, chart_path)

    # Generate sample grid
    grid_path = output_path.parent / "sample_grid.png"
    create_sample_grid(data_dir, grid_path)

    # Check for imbalance
    warnings = check_imbalance(counts)

    # Write report
    lines = [
        "# UDMS Dataset Report\n",
        f"Generated from: `{data_dir}`\n",
        "## Class Distribution\n",
        f"![Class Distribution](class_distribution.png)\n",
        "| Category | Train | Val | Test | Total |",
        "|----------|------:|----:|-----:|------:|",
    ]

    for cat in UDMS_CATEGORIES:
        tr = counts["train"].get(cat, 0)
        va = counts["val"].get(cat, 0)
        te = counts["test"].get(cat, 0)
        total = tr + va + te
        lines.append(f"| {cat} | {tr} | {va} | {te} | {total} |")

    grand = sum(
        counts[s].get(c, 0)
        for s in ["train", "val", "test"]
        for c in UDMS_CATEGORIES
    )
    lines.append(f"| **TOTAL** | | | | **{grand}** |")

    # Text bar chart showing distribution
    lines.append("\n## Distribution Bar Chart\n")
    lines.append("```")
    max_total = max(
        sum(counts[s].get(c, 0) for s in ["train", "val", "test"])
        for c in UDMS_CATEGORIES
    ) or 1
    bar_width = 40
    for cat in UDMS_CATEGORIES:
        total = sum(counts[s].get(cat, 0) for s in ["train", "val", "test"])
        bar_len = int((total / max_total) * bar_width)
        bar = "█" * bar_len + "░" * (bar_width - bar_len)
        lines.append(f"{cat:20s} |{bar}| {total}")
    lines.append("```")

    lines.append("\n## Sample Images\n")
    lines.append("![Sample Grid](sample_grid.png)\n")

    if warnings:
        lines.append("\n## ⚠️  Imbalance Warnings\n")
        for w in warnings:
            lines.append(f"- {w}")
    else:
        lines.append("\n## ✅ All categories meet minimum thresholds.\n")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"📄 Report written to: {output_path}")


def main():
    print("=" * 60)
    print("UDMS Dataset Report Generator")
    print("=" * 60)
    generate_report()


if __name__ == "__main__":
    main()

