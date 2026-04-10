# UDMS Image Classifier — Copilot Project Instructions

> **This file is the single source of truth for this project.**
> Place it at the root of the repo as `COPILOT_INSTRUCTIONS.md`.
> Copilot and any AI assistant should read this before generating any code.

---

## 1. PROJECT IDENTITY

**Name:** UDMS Image Classifier
**Purpose:** Automatically classify citizen-submitted photos of urban disorder into 7 categories for the Urban Disorder Monitoring System (UDMS).
**Company:** Light Brain Technologies
**Duration:** 4 weeks
**Target deployment:** REST API microservice, Dockerized, CPU-only inference

### 1.1 The 7 Classification Categories

| Index | Category ID         | Human Label                        | Priority |
|-------|---------------------|------------------------------------|----------|
| 0     | illegal_dumping     | Illegal Dumping / Garbage          | HIGH     |
| 1     | pothole_road        | Pothole / Road Damage              | HIGH     |
| 2     | broken_lighting     | Broken / Missing Street Lighting   | MEDIUM   |
| 3     | water_sewage        | Water / Sewage Issues              | MEDIUM   |
| 4     | damaged_signage     | Damaged Signage / Infrastructure   | LOWER    |
| 5     | vegetation          | Vegetation Overgrowth              | LOWER    |
| 6     | other               | Other / Unclassified               | LOW      |

### 1.2 Hardware Constraints

- CPU: Intel i5-4210U @ 1.70–2.60 GHz (dual-core)
- RAM: 6 GB
- GPU: None in production — all inference must run on CPU
- Training: Google Colab free tier GPU

### 1.3 Success Criteria

| Metric                          | Minimum   | Stretch   |
|---------------------------------|-----------|-----------|
| Overall accuracy                | ≥75%      | ≥85%      |
| Per-class F1 (top 3 categories) | ≥0.70     | ≥0.80     |
| Inference time (single image)   | <500ms    | <200ms    |
| API response time (end-to-end)  | <1 second | <500ms    |
| Model file size                 | <50 MB    | <20 MB    |
| Test coverage                   | ≥80%      | ≥90%      |

---

## 2. PROJECT STRUCTURE

**This is the exact directory tree. Create it exactly as shown. Do not add, rename, or reorganize folders.**

```
udms-image-classifier/
│
├── COPILOT_INSTRUCTIONS.md          ← THIS FILE
├── README.md
├── pyproject.toml
├── requirements.txt                 # Pinned production dependencies
├── requirements-dev.txt             # Dev/test dependencies
├── .env.example                     # MODEL_PATH, CONFIDENCE_THRESHOLD, LOG_LEVEL
├── .gitignore
├── Dockerfile
├── docker-compose.yml
│
├── data/
│   ├── raw/                         # Downloaded datasets (gitignored)
│   │   ├── road_issues_hf/          # From HuggingFace
│   │   ├── urban_visual_pollution/  # From Kaggle
│   │   ├── civic_issues_qr4change/  # From Mendeley
│   │   ├── road_hazards/            # From Kaggle
│   │   ├── flood_classification/    # From Kaggle
│   │   ├── damaged_signs/           # From Kaggle
│   │   └── web_scraped/             # Manual scraping results
│   ├── processed/
│   │   ├── train/
│   │   │   ├── illegal_dumping/
│   │   │   ├── pothole_road/
│   │   │   ├── broken_lighting/
│   │   │   ├── water_sewage/
│   │   │   ├── damaged_signage/
│   │   │   ├── vegetation/
│   │   │   └── other/
│   │   ├── val/                     # Same 7 subdirectories
│   │   └── test/                    # Same 7 subdirectories
│   └── dataset_report.md
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_augmentation_experiments.ipynb
│   ├── 03_train_classifier.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_export_model.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download_datasets.py
│   │   ├── category_mapping.py
│   │   ├── clean_dataset.py
│   │   ├── split_dataset.py
│   │   ├── preprocessing.py
│   │   ├── augmentation_pipeline.py
│   │   ├── scraper.py               # Web scraping for weak categories
│   │   └── generate_report.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── callbacks.py
│   │   ├── evaluate.py
│   │   └── export.py
│   └── inference/
│       ├── __init__.py
│       └── classifier.py           # Uses src.data.preprocessing (no duplicate)
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── dependencies.py
│   ├── middleware.py
│   ├── schemas.py
│   └── routes/
│       ├── __init__.py
│       ├── classify.py
│       └── health.py
│
├── models/
│   ├── classifier.tflite
│   ├── classifier.onnx
│   ├── classifier.h5
│   └── label_map.json
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_preprocessing.py
│   ├── test_augmentation.py
│   ├── test_model_build.py
│   ├── test_classifier.py
│   ├── test_api_classify.py
│   ├── test_api_health.py
│   ├── test_edge_cases.py
│   └── fixtures/
│       ├── sample_pothole.jpg
│       ├── sample_garbage.jpg
│       ├── sample_lighting.jpg
│       └── corrupted.bin
│
├── demo/
│   └── streamlit_app.py
│
├── docs/
│   ├── api_spec.yaml
│   ├── architecture_diagram.png
│   ├── evaluation_report.md
│   ├── integration_guide.md
│   ├── retraining_guide.md
│   ├── model_card.md
│   ├── performance_report.md
│   └── final_presentation.pdf
│
└── scripts/
    ├── setup_project.sh
    ├── download_datasets.sh
    ├── run_training.sh
    └── benchmark_inference.py
```

---

## 3. DATASET DOWNLOAD SPECIFICATION

### 3.1 Dataset Sources — Exact URLs and Category Mapping

**Download order matters. Follow this sequence.**

#### Dataset 1: Road Issues Detection (CRITICAL — download first)
- **Source:** HuggingFace
- **URL:** https://huggingface.co/datasets/Programmer-RD-AI/road-issues-detection-dataset
- **Images:** 9,660 across 7 classes
- **License:** CC0 1.0 (public domain)
- **Download to:** `data/raw/road_issues_hf/`
- **Download method:** SCRIPTABLE — use the `datasets` library, NOT `snapshot_download`:
  ```python
  from datasets import load_dataset
  ds = load_dataset("Programmer-RD-AI/road-issues-detection-dataset", split="train")
  # Then iterate ds and save images to data/raw/road_issues_hf/<class_name>/
  ```
- **Category mapping:**
  - `Littering/Garbage` → `illegal_dumping`
  - `Damaged Road Issues` → `pothole_road`
  - `Pothole Issues` → `pothole_road`
  - `Broken Road Sign Issues` → `damaged_signage`
  - `Mixed Issues` → `other`
  - `Vandalism/Graffiti` → `other`
  - `Illegal Parking` → discard (not a UDMS category)

#### Dataset 2: Urban Visual Pollution (CRITICAL — only source for broken lighting)
- **Source:** Kaggle
- **URL:** https://www.kaggle.com/datasets/abhranta/urban-visual-pollution-dataset
- **Images:** 9,966 across 11 classes
- **Download to:** `data/raw/urban_visual_pollution/`
- **Category mapping:**
  - `GARBAGE` → `illegal_dumping`
  - `POTHOLE` → `pothole_road`
  - `ROAD_CONSTRUCTION` → `pothole_road`
  - `BAD_STREETLIGHT` → `broken_lighting` ← **only readily available source**
  - `BROKEN_SIGNAGE` → `damaged_signage`
  - `FADED_SIGNAGE` → `damaged_signage`
  - `CLUTTERED_SIDEWALK` → `other`
  - `BAD_BILLBOARD` → discard
  - `GRAFFITI` → `other`
  - `SAND_ON_ROADS` → discard
  - `UNKEPT_FACADE` → discard

#### Dataset 3: Civic Issues QR4Change (HIGH — most realistic images)
- **Source:** Mendeley Data
- **URL:** https://data.mendeley.com/datasets/zndzygc3p3/2
- **Images:** 4,937 (1,004 pothole + 1,962 plain road + 712 garbage + 1,259 non-garbage)
- **Download to:** `data/raw/civic_issues_qr4change/`
- **Download method:** ⚠️ MANUAL — Mendeley has no CLI. Download the ZIP from the browser, extract into `data/raw/civic_issues_qr4change/`. The download script should print instructions and skip this dataset if the folder is empty.
- **Category mapping:**
  - Pothole images → `pothole_road`
  - Plain road images → discard (negative samples, not a UDMS category)
  - Garbage dump images → `illegal_dumping`
  - Non-garbage images → discard
- **Note:** These are from field surveys in Pune, India — closest visual match to African cities

#### Dataset 4: Road Hazards — Potholes, Cracks and Manholes (HIGH)
- **Source:** Kaggle
- **URL:** https://www.kaggle.com/datasets/sabidrahman/pothole-cracks-and-openmanhole
- **Download to:** `data/raw/road_hazards/`
- **Category mapping:**
  - Pothole images → `pothole_road`
  - Crack images → `pothole_road`
  - Open manhole images → `water_sewage` ← **important for sewage/drain coverage**

#### Dataset 5: Flood Classification (MEDIUM — water/sewage proxy)
- **Source:** Kaggle
- **URL:** https://www.kaggle.com/datasets/dhawalsrivastava2583/flood-classification-dataset
- **Images:** ~13,000 (9,296 flood + 3,748 non-flood)
- **Download to:** `data/raw/flood_classification/`
- **Download method:** SCRIPTABLE — `kaggle datasets download -d dhawalsrivastava2583/flood-classification-dataset`
- **Category mapping:**
  - Flooded images → `water_sewage`
  - Non-flood images → discard
  - **FILTER:** Keep only ground-level urban scenes. Discard aerial/satellite/rural imagery.

#### Dataset 6: Damaged Signs (MEDIUM)
- **Source:** Kaggle
- **URL:** https://www.kaggle.com/datasets/danielvareta/damaged-signs-dataset
- **Download to:** `data/raw/damaged_signs/`
- **Category mapping:**
  - All damaged sign images → `damaged_signage`

#### Dataset 7 (optional): Garbage Classification 12-class
- **Source:** Kaggle
- **URL:** https://www.kaggle.com/datasets/mostafaabla/garbage-classification
- **Download to:** `data/raw/garbage_extra/`
- **Use only if:** Category 1 needs more data after datasets 1–3

### 3.2 Web Scraping Targets — For Weak Categories

These categories have no dedicated dataset and MUST be supplemented with scraping.

#### Category 2 (broken_lighting) — WEAK coverage (~800 available)
Search terms:
- "broken streetlight"
- "damaged lamp post urban Africa"
- "faulty street light pole"
- "non-functional streetlight developing country"
- "broken street light Lagos"
- "damaged lamppost Nairobi"

#### Category 3 (water_sewage) — MODERATE coverage (~1,200 available)
Search terms:
- "burst water pipe street"
- "sewage overflow road Africa"
- "open manhole cover urban"
- "stagnant water urban street"

#### Category 5 (vegetation) — WEAKEST coverage (~300 available)
Search terms:
- "overgrown sidewalk Africa"
- "tree blocking road sign"
- "unmaintained public park urban"
- "overgrown road median"
- "bush overgrown pavement developing country"

**Fallback:** If vegetation stays below 100 images after scraping, merge into `other` category.

### 3.3 Data Quality Rules

When processing downloaded images, enforce these checks:

1. Remove duplicates using perceptual hashing (imagehash library)
2. Remove images smaller than 100×100 pixels
3. Remove clearly mislabelled or irrelevant images
4. Remove watermarked, composited, or cartoon/illustration images
5. No single source may exceed 40% of any category (prevents style overfitting)
6. Keep only citizen-perspective images (phone camera, street-level) — discard aerial/satellite
7. Final minimum per category: 150 images for HIGH priority, 80 for LOWER priority

### 3.4 Target Dataset Size

| Category          | Minimum | Target | With Augmentation |
|-------------------|---------|--------|-------------------|
| illegal_dumping   | 150     | 300    | 600+              |
| pothole_road      | 150     | 300    | 600+              |
| broken_lighting   | 100     | 200    | 400+              |
| water_sewage      | 100     | 200    | 400+              |
| damaged_signage   | 80      | 150    | 300+              |
| vegetation        | 80      | 150    | 300+              |
| other             | 50      | 100    | 200+              |

---

## 4. MODULE SPECIFICATIONS

**Every file below must be created with exactly the functions, classes, inputs, outputs, and logic described. Do not add extra functions. Do not rename anything.**

---

### 4.1 `src/data/download_datasets.py`

**Purpose:** Automate downloading of all training datasets.

```python
"""
Dependencies: kaggle, huggingface_hub, requests, pathlib
"""

# CONSTANTS
RAW_DATA_DIR = Path("data/raw")
DATASETS = {
    "road_issues_hf": {
        "source": "huggingface",
        "repo_id": "Programmer-RD-AI/road-issues-detection-dataset",
        "target_dir": RAW_DATA_DIR / "road_issues_hf",
        "priority": "CRITICAL"
    },
    "urban_visual_pollution": {
        "source": "kaggle",
        "dataset_id": "abhranta/urban-visual-pollution-dataset",
        "target_dir": RAW_DATA_DIR / "urban_visual_pollution",
        "priority": "CRITICAL"
    },
    "road_hazards": {
        "source": "kaggle",
        "dataset_id": "sabidrahman/pothole-cracks-and-openmanhole",
        "target_dir": RAW_DATA_DIR / "road_hazards",
        "priority": "HIGH"
    },
    "damaged_signs": {
        "source": "kaggle",
        "dataset_id": "danielvareta/damaged-signs-dataset",
        "target_dir": RAW_DATA_DIR / "damaged_signs",
        "priority": "MEDIUM"
    },
    "garbage_classification": {
        "source": "kaggle",
        "dataset_id": "mostafaabla/garbage-classification",
        "target_dir": RAW_DATA_DIR / "garbage_extra",
        "priority": "LOW"
    }
}

def download_huggingface(repo_id: str, target_dir: Path) -> None:
    """Download dataset from HuggingFace using the datasets library.
    DO NOT use snapshot_download — it may not work for all dataset formats.
    Use: from datasets import load_dataset
    ds = load_dataset(repo_id, split='train')
    Then iterate and save each image to target_dir/<label>/<filename>.jpg
    """

def download_kaggle(dataset_id: str, target_dir: Path) -> None:
    """Download dataset from Kaggle using kaggle CLI.
    Use: subprocess.run(["kaggle", "datasets", "download", "-d", dataset_id, "-p", str(target_dir), "--unzip"])
    Requires: KAGGLE_USERNAME and KAGGLE_KEY env vars (from ~/.kaggle/kaggle.json)
    """

def print_manual_instructions(dataset_name: str, url: str, target_dir: Path) -> None:
    """Print instructions for datasets that cannot be automated.
    Used for: Mendeley (civic_issues_qr4change).
    Message: 'Please download manually from {url} and extract to {target_dir}'
    Then check if target_dir exists and has files — if yes, skip with 'Already downloaded.'
    """

def download_all() -> None:
    """Download all datasets in priority order.
    For each dataset:
    1. Check if target_dir already exists and has files — if yes, skip.
    2. If source is 'huggingface', call download_huggingface.
    3. If source is 'kaggle', call download_kaggle.
    4. If source is 'mendeley', call print_manual_instructions.
    5. Print summary: downloaded N datasets, M need manual download, K already existed.
    """

def main():
    """Entry point. Call download_all() and print summary."""
```

**Validation:** After running, every `target_dir` should exist and contain image files. Print count per directory.

---

### 4.2 `src/data/category_mapping.py`

**Purpose:** Map source dataset class names to the 7 UDMS categories. This is the single source of truth for all label translations.

```python
"""
This module defines how each source dataset's class labels map to UDMS categories.
NEVER hardcode category mappings elsewhere — always import from here.
"""

UDMS_CATEGORIES = [
    "illegal_dumping",
    "pothole_road",
    "broken_lighting",
    "water_sewage",
    "damaged_signage",
    "vegetation",
    "other"
]

CATEGORY_LABELS = {
    "illegal_dumping": "Illegal Dumping / Garbage",
    "pothole_road": "Pothole / Road Damage",
    "broken_lighting": "Broken / Missing Street Lighting",
    "water_sewage": "Water / Sewage Issues",
    "damaged_signage": "Damaged Signage / Infrastructure",
    "vegetation": "Vegetation Overgrowth",
    "other": "Other / Unclassified"
}

NUM_CLASSES = 7

# Source → UDMS mapping. None means discard.
ROAD_ISSUES_HF_MAP = {
    "Littering/Garbage": "illegal_dumping",
    "Damaged Road Issues": "pothole_road",
    "Pothole Issues": "pothole_road",
    "Broken Road Sign Issues": "damaged_signage",
    "Mixed Issues": "other",
    "Vandalism/Graffiti": "other",
    "Illegal Parking": None  # discard
}

URBAN_VISUAL_POLLUTION_MAP = {
    "GARBAGE": "illegal_dumping",
    "POTHOLE": "pothole_road",
    "ROAD_CONSTRUCTION": "pothole_road",
    "BAD_STREETLIGHT": "broken_lighting",
    "BROKEN_SIGNAGE": "damaged_signage",
    "FADED_SIGNAGE": "damaged_signage",
    "CLUTTERED_SIDEWALK": "other",
    "GRAFFITI": "other",
    "BAD_BILLBOARD": None,
    "SAND_ON_ROADS": None,
    "UNKEPT_FACADE": None
}

ROAD_HAZARDS_MAP = {
    "pothole": "pothole_road",
    "crack": "pothole_road",
    "open_manhole": "water_sewage"
}

CIVIC_ISSUES_MAP = {
    "pothole": "pothole_road",
    "garbage": "illegal_dumping",
    "plain_road": None,  # negative sample, discard
    "non_garbage": None   # negative sample, discard
}

def get_udms_category(source_dataset: str, source_label: str) -> str | None:
    """Return the UDMS category for a given source dataset and label, or None to discard."""
    maps = {
        "road_issues_hf": ROAD_ISSUES_HF_MAP,
        "urban_visual_pollution": URBAN_VISUAL_POLLUTION_MAP,
        "road_hazards": ROAD_HAZARDS_MAP,
        "civic_issues_qr4change": CIVIC_ISSUES_MAP,
    }
    mapping = maps.get(source_dataset, {})
    return mapping.get(source_label, None)

def get_label_index(category: str) -> int:
    """Return integer index for a UDMS category string."""
    return UDMS_CATEGORIES.index(category)

def get_category_from_index(index: int) -> str:
    """Return UDMS category string from integer index."""
    return UDMS_CATEGORIES[index]
```

**Validation:** `len(UDMS_CATEGORIES) == 7`. Every mapping dict value is either a string in `UDMS_CATEGORIES` or `None`.

---

### 4.3 `src/data/clean_dataset.py`

**Purpose:** Take raw downloads, apply category mapping, remove bad images, output clean per-category folders.

```python
"""
Input:  data/raw/<dataset_name>/ (messy, original class structure)
Output: data/raw/cleaned/<category_name>/ (flat, mapped to 7 UDMS categories)

Functions:
"""

def remove_duplicates(image_dir: Path, hash_size: int = 8) -> int:
    """Remove duplicate images using perceptual hashing (imagehash library).
    Returns count of removed images."""

def remove_small_images(image_dir: Path, min_size: int = 100) -> int:
    """Remove images smaller than min_size x min_size pixels. Returns count removed."""

def validate_image(filepath: Path) -> bool:
    """Try to open with PIL. Return False if corrupted/unreadable."""

def map_and_copy(source_dir: Path, dataset_name: str, output_base: Path) -> dict:
    """Read images from source_dir, look up UDMS category via category_mapping.py,
    copy to output_base/<category>/. Returns dict of {category: count}."""

def check_source_balance(category_counts: dict, max_source_pct: float = 0.40) -> list:
    """Warn if any single source exceeds max_source_pct of a category."""

def clean_all() -> None:
    """Run full pipeline: map_and_copy all datasets, remove_duplicates,
    remove_small_images, validate, print summary."""
```

**Validation:** No image in output should be smaller than 100x100. No duplicates (run imagehash check). Print per-category counts and warn if any category is below minimum.

---

### 4.4 `src/data/split_dataset.py`

**Purpose:** Split cleaned images into train/val/test with 70/15/15 ratio.

```python
"""
Input:  data/raw/cleaned/<category>/ (flat image folders)
Output: data/processed/train/<category>/, val/<category>/, test/<category>/

CRITICAL: Use stratified split to maintain class proportions.
CRITICAL: Set random_state=42 for reproducibility.
"""

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42

def split_dataset(cleaned_dir: Path, output_dir: Path) -> dict:
    """For each category subfolder in cleaned_dir:
    1. List all image files
    2. Use sklearn.model_selection.train_test_split twice:
       - First split: 70% train, 30% temp
       - Second split: 50/50 on temp → 15% val, 15% test
    3. Copy (not move) images to output_dir/train|val|test/<category>/
    Returns: {category: {train: N, val: N, test: N}}
    """

def verify_split(output_dir: Path) -> bool:
    """Verify: no image appears in more than one split.
    Verify: all 7 category folders exist in each split.
    Verify: ratios are approximately 70/15/15."""
```

**Validation:** Run `verify_split` after every split. Assert no data leakage between splits.

---

### 4.5 `src/data/preprocessing.py`

**Purpose:** Image preprocessing for both training and inference.

```python
"""
Standard preprocessing matching MobileNetV2 expectations.
This module is used by BOTH training pipeline AND inference runtime.
"""

IMG_SIZE = (224, 224)
PIXEL_RANGE = (0.0, 1.0)  # Normalize to [0, 1]

def load_and_preprocess(image_path: str | Path) -> np.ndarray:
    """Load image from disk, preprocess for model input.
    1. Open with PIL
    2. Convert to RGB (handle grayscale, RGBA)
    3. Resize to 224x224 using LANCZOS resampling
    4. Convert to numpy float32 array
    5. Normalize pixel values to [0, 1] by dividing by 255.0
    Returns: numpy array shape (224, 224, 3), dtype float32, values in [0, 1]
    """

def preprocess_bytes(image_bytes: bytes) -> np.ndarray:
    """Same as load_and_preprocess but accepts raw bytes (for API endpoint).
    1. Open with PIL from BytesIO
    2. Same steps as load_and_preprocess
    Returns: numpy array shape (224, 224, 3), dtype float32
    """

def add_batch_dimension(image: np.ndarray) -> np.ndarray:
    """Expand dims to (1, 224, 224, 3) for single-image inference."""
    return np.expand_dims(image, axis=0)
```

**Validation:**
- Output shape is always `(224, 224, 3)`
- Output dtype is `float32`
- Output values are in `[0.0, 1.0]`
- Grayscale input → 3-channel RGB output
- RGBA input → 3-channel RGB output

---

### 4.6 `src/data/augmentation_pipeline.py`

**Purpose:** Data augmentation applied ONLY to training images. Never to val/test.

```python
"""
Uses Albumentations library.
These specific transforms are chosen per the project plan.
"""
import albumentations as A

def get_training_augmentation() -> A.Compose:
    """Return the training augmentation pipeline.

    Transforms (in order):
    1. HorizontalFlip(p=0.5)
    2. Rotate(limit=15, p=0.5)              # ±15 degrees
    3. RandomBrightnessContrast(
           brightness_limit=0.2,
           contrast_limit=0.2, p=0.5)        # ±20%
    4. RandomScale(scale_limit=0.2, p=0.3)   # 0.8x–1.2x zoom
    5. GaussNoise(var_limit=(10, 50), p=0.3) # Simulates phone camera noise
    6. Resize(224, 224)                       # Always resize last
    """

def get_validation_transform() -> A.Compose:
    """Return validation/test transform. Resize ONLY, no augmentation.
    Transforms:
    1. Resize(224, 224)
    """

def augment_image(image: np.ndarray, transform: A.Compose) -> np.ndarray:
    """Apply augmentation to a single image. Returns augmented image."""
```

**CRITICAL RULE:** Augmentation is NEVER applied to validation or test data. Only to training data.

**Validation:** Augmented images must remain valid (no NaN, no values outside [0, 255] before normalization, correct shape).

---

### 4.7 `src/data/generate_report.py`

**Purpose:** Generate `data/dataset_report.md` with class distribution stats and sample visualizations.

```python
"""
Input: data/processed/train|val|test/<category>/
Output: data/dataset_report.md

Contents:
- Per-category image counts (train, val, test)
- Bar chart of class distribution (saved as PNG, embedded in markdown)
- Sample grid: 3 random images per category
- Warnings for imbalanced categories (any category below minimum threshold)
"""

def count_images(data_dir: Path) -> dict:
    """Count images per category per split. Returns nested dict."""

def plot_distribution(counts: dict, output_path: Path) -> None:
    """Bar chart: x=categories, y=counts, grouped by split. Save as PNG."""

def create_sample_grid(data_dir: Path, output_path: Path, samples_per_class: int = 3) -> None:
    """Grid of sample images, one row per category. Save as PNG."""

def check_imbalance(counts: dict) -> list:
    """Return list of warnings for categories below minimum thresholds."""

def generate_report(data_dir: Path, output_path: Path) -> None:
    """Orchestrate: count, plot, sample, check, write markdown."""
```

---

### 4.8 `src/training/model.py`

**Purpose:** Define the MobileNetV2 transfer learning model architecture.

```python
"""
Architecture:
  MobileNetV2 (ImageNet weights, include_top=False)
  → GlobalAveragePooling2D
  → Dropout(0.3)
  → Dense(128, activation='relu')
  → Dropout(0.2)
  → Dense(7, activation='softmax')

Input shape: (224, 224, 3)
Output shape: (7,) — probability distribution over 7 categories
"""
import tensorflow as tf

INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 7
DENSE_UNITS = 128
DROPOUT_1 = 0.3
DROPOUT_2 = 0.2

def build_model(num_classes: int = NUM_CLASSES, freeze_backbone: bool = True) -> tf.keras.Model:
    """Build MobileNetV2 transfer learning model.

    Args:
        num_classes: Number of output classes (default 7)
        freeze_backbone: If True, freeze all MobileNetV2 layers (Phase 1).
                        If False, all layers are trainable (call unfreeze_layers after).

    Steps:
    1. Load MobileNetV2 with:
       - input_shape=(224, 224, 3)
       - include_top=False
       - weights='imagenet'
    2. Set base_model.trainable = freeze_backbone (inverted!)
    3. Build Sequential or Functional model:
       Input → base_model → GlobalAveragePooling2D →
       Dropout(0.3) → Dense(128, relu) → Dropout(0.2) → Dense(num_classes, softmax)
    4. Return compiled model (DO NOT compile here — compile in train.py)

    Returns: tf.keras.Model
    """

def unfreeze_top_layers(model: tf.keras.Model, num_layers: int = 30) -> tf.keras.Model:
    """Unfreeze the last `num_layers` layers of the backbone for fine-tuning (Phase 2).

    Steps:
    1. Access the MobileNetV2 base (model.layers[0] or by name)
    2. Set base.trainable = True
    3. For each layer in base.layers[:-num_layers]: layer.trainable = False
    4. Return the model (needs recompilation in train.py)
    """

def get_model_summary(model: tf.keras.Model) -> str:
    """Return model summary as string."""
```

**Validation:**
- Total parameters should be ~3.4M (MobileNetV2) + small head
- Output shape of final layer is `(None, 7)`
- When frozen: only ~67K trainable parameters (the head)
- When top 30 unfrozen: ~1.2M+ trainable parameters

---

### 4.9 `src/training/train.py`

**Purpose:** Training loop for both Phase 1 (head only) and Phase 2 (fine-tuning).

```python
"""
Phase 1: Freeze backbone, train classification head only
  - Optimizer: Adam(learning_rate=1e-3)  — also try 3e-4 and 1e-4
  - Loss: CategoricalCrossentropy (or SparseCategoricalCrossentropy if using integer labels)
  - Epochs: 15–20
  - Callbacks: ModelCheckpoint, EarlyStopping(patience=5)

Phase 2: Unfreeze last 30 layers, fine-tune
  - Optimizer: Adam(learning_rate=1e-5)  — MUST be very low to prevent catastrophic forgetting
  - Loss: same as Phase 1
  - Epochs: 10–15
  - Callbacks: ModelCheckpoint, EarlyStopping(patience=5), ReduceLROnPlateau

Data loading: Use tf.keras.utils.image_dataset_from_directory
  - directory: data/processed/train/ (or val/)
  - image_size: (224, 224)
  - batch_size: 32
  - label_mode: 'categorical'
"""

BATCH_SIZE = 32
PHASE1_LR = 1e-3
PHASE1_EPOCHS = 20
PHASE2_LR = 1e-5
PHASE2_EPOCHS = 15

def load_data(data_dir: str, split: str = "train", batch_size: int = BATCH_SIZE):
    """Load image dataset from directory structure.
    Returns: tf.data.Dataset"""

def train_phase1(model, train_ds, val_ds, epochs: int = PHASE1_EPOCHS) -> tf.keras.callbacks.History:
    """Compile with Adam(lr=PHASE1_LR), fit with callbacks.
    Save best model to models/phase1_best.h5"""

def train_phase2(model, train_ds, val_ds, epochs: int = PHASE2_EPOCHS) -> tf.keras.callbacks.History:
    """Call unfreeze_top_layers(model, 30), recompile with Adam(lr=PHASE2_LR), fit.
    Save best model to models/phase2_best.h5"""

def train_full_pipeline(data_dir: str = "data/processed") -> tf.keras.Model:
    """Orchestrate: build_model → train_phase1 → unfreeze → train_phase2 → return model"""
```

**CRITICAL:** Phase 2 learning rate MUST be 1e-5 or lower. Higher rates cause catastrophic forgetting of ImageNet features.

---

### 4.10 `src/training/evaluate.py`

**Purpose:** Generate evaluation metrics on the test set.

```python
"""
Input: Trained model + data/processed/test/
Output: docs/evaluation_report.md

Metrics to compute:
1. Overall accuracy
2. Per-class precision, recall, F1-score (sklearn.metrics.classification_report)
3. Confusion matrix (sklearn.metrics.confusion_matrix)
4. Misclassification analysis: top 10 most confused class pairs
5. Confidence distribution histogram
"""

def evaluate_model(model, test_ds) -> dict:
    """Run model.predict on test set, compute all metrics. Return results dict."""

def plot_confusion_matrix(y_true, y_pred, class_names, output_path: Path) -> None:
    """Plot and save confusion matrix as PNG."""

def find_misclassifications(y_true, y_pred, image_paths, top_n: int = 10) -> list:
    """Return the top_n most confident wrong predictions with image paths."""

def generate_evaluation_report(results: dict, output_path: Path) -> None:
    """Write docs/evaluation_report.md with all metrics, charts, and analysis."""
```

**Validation:** If overall accuracy < 75%, the report MUST flag this and suggest remediation steps.

---

### 4.11 `src/training/export.py`

**Purpose:** Convert trained Keras model to TFLite and ONNX for production inference.

```python
"""
Input: models/phase2_best.h5 (or whatever the best model is)
Output: models/classifier.tflite, models/classifier.onnx, models/label_map.json

TFLite conversion:
1. Load .h5 model
2. tf.lite.TFLiteConverter.from_keras_model(model)
3. converter.optimizations = [tf.lite.Optimize.DEFAULT]  # dynamic range quantization
4. tflite_model = converter.convert()
5. Save to models/classifier.tflite

ONNX conversion:
1. pip install tf2onnx
2. python -m tf2onnx.convert --saved-model <path> --output models/classifier.onnx

Label map:
1. Write models/label_map.json from category_mapping.UDMS_CATEGORIES
"""

def export_tflite(model_path: str, output_path: str, quantize: bool = True) -> int:
    """Convert to TFLite. Return file size in bytes.
    If quantize=True, apply dynamic range quantization (INT8)."""

def export_onnx(model_path: str, output_path: str) -> int:
    """Convert to ONNX. Return file size in bytes."""

def create_label_map(output_path: str) -> None:
    """Write label_map.json:
    {
      "0": {"category": "illegal_dumping", "label": "Illegal Dumping / Garbage"},
      "1": {"category": "pothole_road", "label": "Pothole / Road Damage"},
      ...
    }
    """

def benchmark_tflite(tflite_path: str, sample_image: np.ndarray, num_runs: int = 100) -> float:
    """Run inference num_runs times, return average time in ms.
    Must be < 500ms on CPU."""

def export_all(model_path: str = "models/phase2_best.h5") -> None:
    """Orchestrate: export_tflite, export_onnx, create_label_map, benchmark."""
```

**Validation:**
- TFLite file size < 50MB (stretch: <20MB)
- Benchmark latency < 500ms on CPU
- Label map has exactly 7 entries matching UDMS_CATEGORIES

---

### 4.12 `src/inference/classifier.py`

**Purpose:** Production inference class. Used by API and demo app. Loaded ONCE at startup.

```python
"""
This is the core inference module. It wraps the TFLite model.
IMPORTANT: This module imports preprocessing from src.data.preprocessing.
There is NO separate src/inference/preprocessing.py — that would be a duplicate.

Pattern reference: Study the Predictor class from github.com/CVxTz/FastImageClassification
Adapt their pattern: load model once, expose a predict(image_bytes) method.

Pattern reference: Study TFLite serving from github.com/robmarkcole/tensorflow-lite-rest-server
Adapt their pattern: TFLite interpreter setup, invoke, get output tensor.
"""
from src.data.preprocessing import preprocess_bytes, add_batch_dimension

class UDMSClassifier:
    def __init__(self, model_path: str, label_map_path: str, confidence_threshold: float = 0.6):
        """Load TFLite model and label map once.

        Steps:
        1. Load TFLite interpreter: tf.lite.Interpreter(model_path=model_path)
        2. Allocate tensors: interpreter.allocate_tensors()
        3. Get input/output details
        4. Load label_map.json into self.label_map
        5. Store confidence_threshold
        """

    def predict(self, image_bytes: bytes) -> dict:
        """Run inference on raw image bytes.

        Steps:
        1. Call preprocessing.preprocess_bytes(image_bytes)  → (224,224,3)
        2. Add batch dimension → (1,224,224,3)
        3. Set input tensor on interpreter
        4. Invoke interpreter
        5. Get output tensor → (1, 7) softmax probabilities
        6. Find top prediction (argmax)
        7. Build response dict

        Returns:
        {
            "prediction": {
                "category": "illegal_dumping",
                "category_label": "Illegal Dumping / Garbage",
                "confidence": 0.87,
                "requires_review": False  # True if confidence < threshold
            },
            "alternatives": [
                {"category": "water_sewage", "confidence": 0.08},
                {"category": "vegetation", "confidence": 0.03}
            ],
            "model_version": "1.0.0",
            "inference_time_ms": 142
        }
        """

    def _get_alternatives(self, probabilities: np.ndarray, top_n: int = 3) -> list:
        """Return top_n predictions excluding the primary, sorted by confidence desc."""

    @property
    def model_info(self) -> dict:
        """Return model metadata: version, categories, input shape, file size."""
```

**Validation:**
- `predict()` always returns a dict with exactly the keys shown above
- `confidence` is a float between 0.0 and 1.0
- `requires_review` is True when confidence < threshold
- `alternatives` has at most `top_n` entries, sorted descending by confidence
- `inference_time_ms` is measured with `time.perf_counter()`

---

### 4.13 `app/config.py`

**Purpose:** All configuration loaded from environment variables.

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MODEL_PATH: str = "models/classifier.tflite"
    LABEL_MAP_PATH: str = "models/label_map.json"
    CONFIDENCE_THRESHOLD: float = 0.6
    MAX_FILE_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: set = {".jpg", ".jpeg", ".png", ".webp"}
    MODEL_VERSION: str = "1.0.0"
    LOG_LEVEL: str = "INFO"
    API_PREFIX: str = "/api/v1"

    class Config:
        env_file = ".env"

settings = Settings()
```

---

### 4.14 `app/schemas.py`

**Purpose:** Pydantic response models matching the API contract.

```python
from pydantic import BaseModel

class PredictionResult(BaseModel):
    category: str              # e.g. "illegal_dumping"
    category_label: str        # e.g. "Illegal Dumping / Garbage"
    confidence: float          # 0.0–1.0
    requires_review: bool      # True if confidence < threshold

class AlternativeResult(BaseModel):
    category: str
    confidence: float

class ClassifyResponse(BaseModel):
    prediction: PredictionResult
    alternatives: list[AlternativeResult]
    model_version: str
    inference_time_ms: float

class HealthResponse(BaseModel):
    status: str                # "healthy"
    model_loaded: bool
    model_version: str

class ModelInfoResponse(BaseModel):
    version: str
    categories: list[str]
    input_shape: list[int]     # [224, 224, 3]
    model_size_mb: float

class ErrorResponse(BaseModel):
    error: str
    detail: str
```

---

### 4.15 `app/dependencies.py`

**Purpose:** Model singleton — loaded once at app startup, shared across all requests.

```python
"""
The UDMSClassifier is expensive to load (reads TFLite file, allocates tensors).
Load it ONCE and reuse for every request.
"""
from src.inference.classifier import UDMSClassifier
from app.config import settings

classifier: UDMSClassifier | None = None

def get_classifier() -> UDMSClassifier:
    global classifier
    if classifier is None:
        classifier = UDMSClassifier(
            model_path=settings.MODEL_PATH,
            label_map_path=settings.LABEL_MAP_PATH,
            confidence_threshold=settings.CONFIDENCE_THRESHOLD
        )
    return classifier

def startup_load_model() -> None:
    """Called during FastAPI startup event. Forces model loading."""
    get_classifier()
```

---

### 4.16 `app/main.py`

**Purpose:** FastAPI application factory.

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.dependencies import startup_load_model
from app.routes import classify, health

def create_app() -> FastAPI:
    app = FastAPI(
        title="UDMS Image Classifier API",
        description="Classifies citizen-submitted images of urban disorder",
        version=settings.MODEL_VERSION
    )

    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    app.include_router(classify.router, prefix=settings.API_PREFIX)
    app.include_router(health.router)

    @app.on_event("startup")
    async def on_startup():
        startup_load_model()

    return app

app = create_app()
```

---

### 4.17 `app/routes/classify.py`

**Purpose:** The main classification endpoint.

```python
"""
POST /api/v1/classify
- Accepts: multipart/form-data with field "image"
- Validates: file extension, file size
- Returns: ClassifyResponse

Pattern reference: Study endpoint design from github.com/CVxTz/FastImageClassification
Their /scorefile/ endpoint is almost identical to what we need.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from app.schemas import ClassifyResponse, ErrorResponse
from app.dependencies import get_classifier
from app.config import settings

router = APIRouter()

@router.post("/classify", response_model=ClassifyResponse, responses={400: {"model": ErrorResponse}})
async def classify_image(image: UploadFile = File(...)):
    """
    Steps:
    1. Validate file extension is in ALLOWED_EXTENSIONS
    2. Read file bytes
    3. Validate file size <= MAX_FILE_SIZE_MB * 1024 * 1024
    4. Call classifier.predict(image_bytes)
    5. Return ClassifyResponse
    6. On any error: raise HTTPException(400) with ErrorResponse
    """
```

---

### 4.18 `app/routes/health.py`

```python
router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Return service health status and whether model is loaded."""

@router.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Return model metadata: version, categories, input shape, size."""
```

---

### 4.19 `demo/streamlit_app.py`

**Purpose:** Simple upload-and-predict demo UI.

```python
"""
Pattern reference: Study Streamlit app from github.com/arpanpramanik2003/smart-waste-classification

UI layout:
1. Title: "UDMS Image Classifier Demo"
2. File uploader: accepts jpg, png, webp
3. On upload:
   a. Display the uploaded image
   b. Run classifier.predict(image_bytes)
   c. Display: category label, confidence bar, requires_review badge
   d. Display: alternatives as a ranked list
   e. Display: inference time
"""
```

---

### 4.20 Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system deps for Pillow/OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

### 4.21 docker-compose.yml

```yaml
version: "3.8"
services:
  classifier:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/classifier.tflite
      - LABEL_MAP_PATH=/app/models/label_map.json
      - CONFIDENCE_THRESHOLD=0.6
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/app/models
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
```

---

### 4.22 `scripts/setup_project.sh`

**Purpose:** One-command project scaffold. Run this once to create the full directory tree.

```bash
#!/bin/bash
set -e

echo "Creating UDMS Image Classifier project structure..."

# Root files
touch README.md pyproject.toml requirements.txt requirements-dev.txt .env.example .gitignore Dockerfile docker-compose.yml

# Data directories
mkdir -p data/raw/{road_issues_hf,urban_visual_pollution,civic_issues_qr4change,road_hazards,flood_classification,damaged_signs,web_scraped}
mkdir -p data/processed/{train,val,test}/{illegal_dumping,pothole_road,broken_lighting,water_sewage,damaged_signage,vegetation,other}

# Source code
mkdir -p src/data src/training src/inference
touch src/__init__.py src/data/__init__.py src/training/__init__.py src/inference/__init__.py
touch src/data/{download_datasets,category_mapping,clean_dataset,split_dataset,preprocessing,augmentation_pipeline,scraper,generate_report}.py
touch src/training/{model,train,callbacks,evaluate,export}.py
touch src/inference/{__init__,classifier}.py

# API
mkdir -p app/routes
touch app/__init__.py app/{main,config,dependencies,middleware,schemas}.py
touch app/routes/__init__.py app/routes/{classify,health}.py

# Models
mkdir -p models
touch models/label_map.json

# Tests
mkdir -p tests/fixtures
touch tests/__init__.py tests/conftest.py
touch tests/{test_preprocessing,test_augmentation,test_model_build,test_classifier,test_api_classify,test_api_health,test_edge_cases}.py

# Demo
mkdir -p demo
touch demo/streamlit_app.py

# Docs
mkdir -p docs
touch docs/{api_spec.yaml,evaluation_report.md,integration_guide.md,retraining_guide.md,model_card.md,performance_report.md}

# Notebooks
mkdir -p notebooks
touch notebooks/{01_data_exploration,02_augmentation_experiments,03_train_classifier,04_evaluation,05_export_model}.ipynb

# Scripts
mkdir -p scripts
touch scripts/{setup_project.sh,download_datasets.sh,run_training.sh,benchmark_inference.py}

echo "Done. Project structure created."
echo "Next: run src/data/download_datasets.py to fetch training data."
```

---

### 4.23 `src/training/callbacks.py`

**Purpose:** Custom and configured Keras callbacks for training.

```python
"""
Centralizes all callback configuration so train.py stays clean.
"""
import tensorflow as tf

def get_phase1_callbacks(checkpoint_dir: str = "models/") -> list:
    """Return callbacks for Phase 1 (head training).
    
    Callbacks:
    1. ModelCheckpoint(
           filepath=checkpoint_dir + "phase1_best.h5",
           monitor='val_accuracy', save_best_only=True, mode='max')
    2. EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    3. CSVLogger(checkpoint_dir + "phase1_training_log.csv")
    """

def get_phase2_callbacks(checkpoint_dir: str = "models/") -> list:
    """Return callbacks for Phase 2 (fine-tuning).
    
    Callbacks:
    1. ModelCheckpoint(
           filepath=checkpoint_dir + "phase2_best.h5",
           monitor='val_accuracy', save_best_only=True, mode='max')
    2. EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    3. ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
    4. CSVLogger(checkpoint_dir + "phase2_training_log.csv")
    """
```

---

### 4.24 `app/middleware.py`

**Purpose:** Request logging and input validation middleware.

```python
"""
Middleware applied to all incoming requests.
"""
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import logging
import time

logger = logging.getLogger("udms_classifier")

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request: method, path, status code, duration.
    
    On each request:
    1. Record start time
    2. Call next handler
    3. Record end time
    4. Log: "{method} {path} → {status_code} in {duration_ms}ms"
    """

class FileSizeValidationMiddleware(BaseHTTPMiddleware):
    """Reject requests with bodies exceeding MAX_FILE_SIZE_MB.
    
    Only applies to POST requests to /api/v1/classify.
    Check Content-Length header first (fast path).
    If no Content-Length, read body and check size (slow path).
    Return 413 Payload Too Large if exceeded.
    """

# Register in app/main.py:
# app.add_middleware(RequestLoggingMiddleware)
# app.add_middleware(FileSizeValidationMiddleware)
```

---

### 4.25 `src/data/scraper.py`

**Purpose:** Web scraping for weak categories (broken_lighting, water_sewage, vegetation).

```python
"""
Used for categories with insufficient dataset coverage.
Categories 2 (broken_lighting), 3 (water_sewage), and 5 (vegetation) need scraping.

IMPORTANT: This is a helper tool for manual use during Week 1.
It is NOT part of the automated pipeline.
Review all scraped images manually before including in training data.

Dependencies: requests, beautifulsoup4, Pillow
Optional: icrawler (easier than raw scraping for Google Images)
"""

# Search queries per category — these are specific to African urban contexts
SCRAPE_QUERIES = {
    "broken_lighting": [
        "broken streetlight",
        "damaged lamp post urban Africa",
        "faulty street light pole",
        "non-functional streetlight developing country",
        "broken street light Lagos",
        "damaged lamppost Nairobi",
        "broken streetlight Harare",
    ],
    "water_sewage": [
        "burst water pipe street",
        "sewage overflow road Africa",
        "open manhole cover urban",
        "stagnant water urban street",
        "flooded road Africa city",
        "blocked drain urban Africa",
    ],
    "vegetation": [
        "overgrown sidewalk Africa",
        "tree blocking road sign",
        "unmaintained public park urban",
        "overgrown road median",
        "bush overgrown pavement developing country",
        "vegetation encroaching road Africa",
    ],
}

def scrape_google_images(query: str, output_dir: Path, max_images: int = 50) -> int:
    """Download images from Google Images for a given query.
    
    Recommended: Use icrawler library (pip install icrawler)
    from icrawler.builtin import GoogleImageCrawler
    crawler = GoogleImageCrawler(storage={'root_dir': str(output_dir)})
    crawler.crawl(keyword=query, max_num=max_images)
    
    Returns: number of images downloaded.
    """

def scrape_flickr_cc(query: str, output_dir: Path, max_images: int = 50) -> int:
    """Download Creative Commons licensed images from Flickr.
    Requires Flickr API key (free to obtain).
    Returns: number of images downloaded.
    """

def scrape_all_weak_categories(output_base: Path) -> dict:
    """Run all scraping queries for weak categories.
    Save to output_base/<category>/<query_slug>/<images>.
    Returns: {category: total_images_scraped}
    
    After running:
    1. Manually review all scraped images
    2. Remove irrelevant, low-quality, watermarked images
    3. Move approved images to data/raw/web_scraped/<category>/
    """
```

---

## 4A. COLAB INTEGRATION

**Training runs on Google Colab, not locally. Here's how the code bridges the two environments.**

### How notebooks use `src/` modules

The Colab notebooks in `notebooks/` need to import from `src/`. There are two approaches:

**Approach A (recommended): Upload repo to Google Drive**
```python
# At the top of every Colab notebook:
from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.insert(0, '/content/drive/MyDrive/udms-image-classifier')

# Now imports work:
from src.training.model import build_model
from src.training.train import train_phase1, train_phase2
from src.data.category_mapping import UDMS_CATEGORIES
```

**Approach B: pip install from GitHub**
```python
!pip install git+https://github.com/<your-username>/udms-image-classifier.git
```
This requires a `pyproject.toml` with `[project]` metadata and `packages = ["src"]`.

### Data on Colab

```python
# Upload processed dataset to Google Drive at:
# /content/drive/MyDrive/udms-data/processed/train/
# /content/drive/MyDrive/udms-data/processed/val/
# /content/drive/MyDrive/udms-data/processed/test/

DATA_DIR = "/content/drive/MyDrive/udms-data/processed"
```

### Saving checkpoints from Colab

```python
# Always save to Google Drive, not Colab local (which resets on disconnect):
CHECKPOINT_DIR = "/content/drive/MyDrive/udms-image-classifier/models/"

# After training, download the best model locally:
# Copy models/phase2_best.h5 from Drive to your local models/ directory
# Then run src/training/export.py locally to create .tflite and .onnx
```

### What runs where

| Task | Environment | Why |
|------|------------|-----|
| Data download + cleaning | Local | Needs disk space, manual review |
| Data splitting | Local | One-time operation |
| Model training (Phase 1 & 2) | Google Colab | Needs GPU |
| Model evaluation | Google Colab | Needs GPU for fast inference on full test set |
| Model export (.tflite, .onnx) | Local | Final artifacts for deployment |
| API development | Local | VS Code + Copilot |
| Docker build + testing | Local | Docker required |
| Demo app | Local | Streamlit |

---

## 4B. TESTING STRATEGY BY PHASE

**Not all tests can run from Day 1. Here's what's testable when.**

### Week 1 (no model exists yet) — test data pipeline only

```python
# tests/test_preprocessing.py — CAN run
# tests/test_augmentation.py — CAN run
# tests/test_model_build.py — CAN run (builds model, doesn't need trained weights)
# tests/test_classifier.py — CANNOT run (needs .tflite file)
# tests/test_api_*.py — CANNOT run (needs loaded model)
```

### Week 2 (model trained, .tflite exported) — test inference

```python
# tests/test_classifier.py — NOW runs with real .tflite
# tests/test_api_*.py — STILL blocked until API is built (Week 3)
```

### Week 3 (API built) — test everything

```python
# All tests can now run
# Run: pytest --cov=src --cov=app --cov-report=term-missing
```

### Mock strategy for early testing

For `test_classifier.py` before the real model exists, create a dummy TFLite model:

```python
# tests/conftest.py

import numpy as np
import json

@pytest.fixture
def mock_tflite_model(tmp_path):
    """Create a minimal valid TFLite model for testing.
    This model outputs random 7-class probabilities.
    It validates that the classifier wrapper works correctly
    WITHOUT needing a trained model.
    """
    import tensorflow as tf
    # Build a tiny model with same I/O signature
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    outputs = tf.keras.layers.Dense(7, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_bytes = converter.convert()
    
    model_path = tmp_path / "test_model.tflite"
    model_path.write_bytes(tflite_bytes)
    return str(model_path)

@pytest.fixture
def mock_label_map(tmp_path):
    """Create a valid label_map.json for testing."""
    label_map = {
        str(i): {"category": cat, "label": f"Test {cat}"}
        for i, cat in enumerate([
            "illegal_dumping", "pothole_road", "broken_lighting",
            "water_sewage", "damaged_signage", "vegetation", "other"
        ])
    }
    path = tmp_path / "label_map.json"
    path.write_text(json.dumps(label_map))
    return str(path)

@pytest.fixture
def sample_image_bytes():
    """Create a valid JPEG image as bytes for testing."""
    from PIL import Image
    import io
    img = Image.new('RGB', (640, 480), color='red')
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    return buffer.getvalue()

@pytest.fixture
def corrupted_bytes():
    """Bytes that are NOT a valid image."""
    return b"this is not an image file at all"

@pytest.fixture
def classifier_instance(mock_tflite_model, mock_label_map):
    """UDMSClassifier loaded with mock model for testing."""
    from src.inference.classifier import UDMSClassifier
    return UDMSClassifier(
        model_path=mock_tflite_model,
        label_map_path=mock_label_map,
        confidence_threshold=0.6
    )
```

**This means tests for the classifier and API can run from Week 1 using mocks, validating the code structure before the real model exists.**

---

## 5. EXECUTION ORDER

**Copilot must generate files in this order. Each step depends on the previous.**

| Step | File to create | Depends on | What to test |
|------|----------------|------------|--------------|
| 1 | `scripts/setup_project.sh` | Nothing | Directories exist |
| 2 | `src/data/category_mapping.py` | Nothing | 7 categories, maps are valid |
| 3 | `models/label_map.json` | category_mapping | 7 entries, correct format |
| 4 | `src/data/download_datasets.py` | category_mapping | Datasets download successfully |
| 5 | `src/data/scraper.py` | Nothing | Scrape queries defined, functions run |
| 6 | `src/data/clean_dataset.py` | category_mapping | Images sorted into 7 folders |
| 7 | `src/data/preprocessing.py` | Nothing | Output shape (224,224,3), float32 |
| 8 | `src/data/augmentation_pipeline.py` | Nothing | Augmented images valid |
| 9 | `src/data/split_dataset.py` | clean_dataset | 70/15/15 split, no leakage |
| 10 | `src/data/generate_report.py` | split_dataset | Report generated with charts |
| 11 | `src/training/model.py` | preprocessing | Model builds, correct output shape |
| 12 | `src/training/callbacks.py` | Nothing | Callbacks instantiate without error |
| 13 | `src/training/train.py` | model, callbacks | Training runs, loss decreases |
| 14 | `src/training/evaluate.py` | train | Metrics computed, report generated |
| 15 | `src/training/export.py` | evaluate, category_mapping | .tflite exists, <500ms inference |
| 16 | `src/inference/classifier.py` | export, preprocessing | predict() returns correct dict |
| 17 | `app/config.py` | Nothing | Settings load from env |
| 18 | `app/schemas.py` | Nothing | Pydantic models validate |
| 19 | `app/middleware.py` | config | Middleware instantiates |
| 20 | `app/dependencies.py` | classifier | Model loads at startup |
| 21 | `app/routes/classify.py` | dependencies, schemas | Endpoint returns 200 |
| 22 | `app/routes/health.py` | dependencies | Health returns "healthy" |
| 23 | `app/main.py` | routes, middleware, deps | App starts, endpoints reachable |
| 24 | `Dockerfile` | main | Container builds |
| 25 | `docker-compose.yml` | Dockerfile | Container runs, API responds |
| 26 | `tests/conftest.py` | classifier, schemas | Fixtures create mock model |
| 27 | `tests/*` | conftest + all above | ≥80% coverage |
| 28 | `demo/streamlit_app.py` | classifier | Upload → prediction works |

---

## 6. REFERENCE REPOS — WHAT TO EXTRACT

**These repos are read in the browser, never cloned. Extract only the listed patterns.**

| Repo | What to extract | Apply to which file |
|------|----------------|---------------------|
| `arpanpramanik2003/smart-waste-classification` | MobileNetV2 model build pattern: `GlobalAveragePooling2D → Dense(128) → Dropout → Softmax`. Study their `model.py`. | `src/training/model.py` |
| `arpanpramanik2003/smart-waste-classification` | Streamlit app layout: upload widget, prediction display, confidence bar | `demo/streamlit_app.py` |
| `CVxTz/FastImageClassification` | Predictor class pattern: load model once in `__init__`, expose `predict(image)` method. Study their `predictor.py`. API endpoint pattern: `/scorefile/` accepts file upload, returns JSON scores. | `src/inference/classifier.py`, `app/routes/classify.py` |
| `robmarkcole/tensorflow-lite-rest-server` | TFLite interpreter setup: `Interpreter(model_path)`, `allocate_tensors()`, `set_tensor()`, `invoke()`, `get_tensor()`. CPU-optimized serving pattern. | `src/inference/classifier.py` |
| `eRuaro/food-vision-api` | Simplest Dockerfile + FastAPI pattern. Clean, minimal. | `Dockerfile`, `app/main.py` |
| `monatis/mobilenetv2-tf2` | Training pipeline structure: directory-based data loading, clean train script. | `src/training/train.py` |
| `akathedeveloper/Trash-Classifier` | Multi-model benchmarking: how to compare accuracy across experiments. Evaluation metrics pattern. | `src/training/evaluate.py` |
| TensorFlow official tutorial | Two-phase transfer learning: freeze → train head → unfreeze → fine-tune. The definitive reference for the exact training procedure. | `src/training/model.py`, `src/training/train.py` |

---

## 7. ANTI-HALLUCINATION RULES

**Copilot must follow these rules to avoid generating incorrect code.**

1. **Never invent dataset URLs.** Only use the exact URLs listed in Section 3.1.
2. **Never change the 7 categories.** They are fixed. Do not add, remove, or rename them.
3. **Never change the model architecture.** It is MobileNetV2 → GAP → Dropout(0.3) → Dense(128, relu) → Dropout(0.2) → Dense(7, softmax). No other architecture.
4. **Never set Phase 2 learning rate above 1e-5.** Higher rates destroy pretrained features.
5. **Never apply augmentation to validation or test data.** Only training data.
6. **Never hardcode category names.** Always import from `src/data/category_mapping.py`.
7. **Never load the model per-request.** Load once at startup via `app/dependencies.py`.
8. **Image size is always 224×224.** Do not change this — it matches MobileNetV2 input.
9. **Confidence threshold default is 0.6.** Below this, `requires_review = True`.
10. **The API response schema is fixed.** Do not modify the JSON structure in Section 4.12.
11. **Use `tf.lite.Interpreter` for inference, not full TensorFlow.** The full TF runtime is too heavy for production.
12. **Use `float32` inputs for TFLite.** Do not use `int8` input — only the model weights are quantized.
13. **Train/val/test split is 70/15/15 with random_state=42.** Do not change these ratios.
14. **Do not use `model.predict()` in production.** Use TFLite interpreter `invoke()` instead.
15. **Do not use PyTorch.** This project uses TensorFlow/Keras exclusively.
16. **Do not create `src/inference/preprocessing.py`.** All preprocessing lives in `src/data/preprocessing.py` and is imported by the classifier. No duplicate.
17. **Mendeley datasets cannot be automated.** The download script must print manual instructions for Mendeley, not attempt to download programmatically.
18. **Use the `datasets` library for HuggingFace, not `snapshot_download`.** `load_dataset()` handles all formats correctly.
19. **Synthetic images must not exceed 20–30% of any category.** They supplement real data, never replace it.
20. **Training code runs on Colab, not locally.** Notebooks must include Drive mount + sys.path setup. Never assume GPU is available locally.
21. **Save Colab checkpoints to Google Drive, not Colab local storage.** Colab local resets on disconnect.

---

## 8. DEPENDENCIES

### requirements.txt (production)
```
tensorflow-cpu>=2.13,<2.17
fastapi>=0.100,<1.0
uvicorn[standard]>=0.23
pillow>=10.0
pydantic>=2.0,<3.0
pydantic-settings>=2.0
python-multipart>=0.0.6
numpy>=1.24,<2.0
```

### requirements-dev.txt
```
-r requirements.txt
albumentations>=1.3
scikit-learn>=1.3
matplotlib>=3.7
pandas>=2.0
seaborn>=0.12
pytest>=7.4
pytest-cov>=4.1
httpx>=0.24
locust>=2.16
kaggle>=1.5
huggingface-hub>=0.17
imagehash>=4.3
tf2onnx>=1.14
black>=23.0
ruff>=0.1
streamlit>=1.28
```

### .gitignore
```
data/raw/
*.h5
*.tflite
*.onnx
.env
__pycache__/
*.pyc
.ipynb_checkpoints/
references/
*.egg-info/
dist/
build/
.pytest_cache/
htmlcov/
```

---

## 9. TESTING SPECIFICATION

### `tests/conftest.py`

**See Section 4B above for the complete conftest.py with all fixtures.** It includes:
- `mock_tflite_model`: Creates a dummy TFLite model with correct I/O shape (allows testing before real model exists)
- `mock_label_map`: Valid 7-entry label_map.json
- `sample_image_bytes`: Valid JPEG as bytes
- `corrupted_bytes`: Invalid bytes for edge case testing
- `classifier_instance`: UDMSClassifier loaded with mock model

### Test assertions that must pass

| Test file | Assertion |
|-----------|-----------|
| test_preprocessing | `output.shape == (224, 224, 3)` |
| test_preprocessing | `output.dtype == np.float32` |
| test_preprocessing | `0.0 <= output.min() and output.max() <= 1.0` |
| test_preprocessing | Grayscale input → 3-channel output |
| test_augmentation | Augmented image has same shape as input |
| test_augmentation | Augmented image has no NaN values |
| test_model_build | `model.output_shape == (None, 7)` |
| test_model_build | Frozen model trainable params < 100K |
| test_classifier | `predict()` returns dict with key "prediction" |
| test_classifier | `prediction.confidence` is between 0 and 1 |
| test_classifier | `prediction.category` is in UDMS_CATEGORIES |
| test_api_classify | POST /api/v1/classify with valid image → 200 |
| test_api_classify | POST /api/v1/classify with .txt file → 400 |
| test_api_classify | POST /api/v1/classify with >10MB file → 400 |
| test_api_health | GET /health → 200, body contains "healthy" |
| test_edge_cases | Corrupted bytes → 400, not 500 |
| test_edge_cases | 10x10 pixel image → 200 (resized, not rejected) |
