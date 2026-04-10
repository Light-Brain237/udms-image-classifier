# UDMS Image Classifier — Complete Workflow Guide

**Your setup:** VS Code + GitHub Copilot + Google Colab + Browser  
**Your project:** One single repo — `udms-image-classifier/`  
**Reference repos:** Read in browser tabs, never cloned locally  
**Datasets:** Downloaded into `data/raw/`, the only external files on disk  

---

## Phase 0: Project Setup (Day 0)

### Step 1 — Create the project skeleton

Open your terminal and scaffold the entire directory structure in one go:

```bash
mkdir udms-image-classifier && cd udms-image-classifier
git init

# Core directories
mkdir -p data/{raw,processed/{train,val,test}/{illegal_dumping,pothole_road,broken_lighting,water_sewage,damaged_signage,vegetation,other}}
mkdir -p notebooks
mkdir -p src/{data,training,inference}
mkdir -p app/routes
mkdir -p models
mkdir -p tests/fixtures
mkdir -p demo
mkdir -p docs
mkdir -p scripts

# Init files
touch src/__init__.py src/data/__init__.py src/training/__init__.py src/inference/__init__.py
touch app/__init__.py app/routes/__init__.py
touch tests/__init__.py tests/conftest.py
```

### Step 2 — Set up .gitignore

Create `.gitignore` with these entries:

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
```

### Step 3 — Create requirements files

`requirements.txt` (production):
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

`requirements-dev.txt` (development):
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
datasets>=2.14
imagehash>=4.3
icrawler>=0.6
tf2onnx>=1.14
black>=23.0
ruff>=0.1
streamlit>=1.28
```

### Step 4 — Push initial commit

```bash
git add .
git commit -m "chore: scaffold project structure"
git remote add origin <your-github-url>
git push -u origin main
```

---

## Phase 1: Data Collection & Preparation (Week 1)

### Step 5 — Download datasets

**Use `src/data/download_datasets.py` (from COPILOT_INSTRUCTIONS.md Section 4.1), NOT a bash script.** The Python script handles HuggingFace, Kaggle, and manual Mendeley instructions in one place.

```bash
# Prerequisites
pip install kaggle huggingface-hub datasets

# Ensure Kaggle credentials exist
# Download kaggle.json from kaggle.com/settings → place at ~/.kaggle/kaggle.json

# Run the download script
python -m src.data.download_datasets
```

**What happens:**
1. HuggingFace Road Issues dataset downloads automatically (9,660 images) → `data/raw/road_issues_hf/`
2. Kaggle datasets download automatically:
   - Urban Visual Pollution (9,966 images) → `data/raw/urban_visual_pollution/`
   - Road Hazards (potholes, cracks, manholes) → `data/raw/road_hazards/`
   - Flood Classification (13,000 images) → `data/raw/flood_classification/`
   - Damaged Signs → `data/raw/damaged_signs/`
3. **Mendeley prints manual instructions:** Download ZIP from https://data.mendeley.com/datasets/zndzygc3p3/2 in your browser, extract to `data/raw/civic_issues_qr4change/`

**After downloading, run the scraper for weak categories:**
```bash
python -m src.data.scraper
# This scrapes Google Images/Flickr for: broken_lighting, water_sewage, vegetation
# Results go to data/raw/web_scraped/<category>/
# ⚠️ MANUALLY REVIEW all scraped images before proceeding
```

**Then clean and map everything to UDMS categories:**
```bash
python -m src.data.clean_dataset
# Maps source labels → 7 UDMS categories using category_mapping.py
# Removes duplicates, small images, corrupted files
```

### Step 6 — Build the preprocessing pipeline

Open `src/data/preprocessing.py` in VS Code.

**Browser tab to open:** `github.com/arpanpramanik2003/smart-waste-classification` — look at how they handle image resizing and normalization.

Write your preprocessing code with Copilot. Start with this comment:

```python
# Resize image to 224x224, normalize pixel values to [0,1],
# convert to RGB if grayscale, return numpy array
```

Copilot will generate the function. Review it, adjust it.

### Step 7 — Build the augmentation pipeline

Open `src/data/augmentation_pipeline.py`.

**Browser tab to open:** `github.com/albumentations-team/albumentations` — look at their example notebooks for classification augmentation.

Your pipeline should include: horizontal flip, rotation ±15°, brightness/contrast ±20%, zoom 0.8–1.2x, Gaussian noise. Write the comment, let Copilot generate, review.

### Step 8 — Build the dataset splitter

Open `src/data/split_dataset.py`.

This script reads `data/raw/`, applies your preprocessing, splits 70/15/15 into `data/processed/train|val|test/`, and generates `data/dataset_report.md` with class distribution charts.

**No browser tab needed** — this is standard scikit-learn `train_test_split` logic. Copilot handles it.

### Step 9 — Run the pipeline and verify

```bash
bash scripts/download_datasets.sh
python src/data/split_dataset.py
```

**Check:** Open `data/dataset_report.md`. Do you have ≥150 images per major category? If not, go back to Step 5 and expand your web scraping for the weak categories.

### Step 10 — Commit Week 1 deliverables

```bash
git add src/data/ scripts/ data/dataset_report.md data/processed/.gitkeep
git commit -m "feat: data pipeline — cleaned dataset with 7 categories"
```

---

## Phase 2: Model Training (Week 2)

### Step 11 — Set up the Colab training notebook

**First: Upload your project and data to Google Drive.**

1. Push your repo to GitHub (you've been committing locally)
2. Upload `data/processed/` to Google Drive at: `My Drive/udms-data/processed/`
3. Clone or sync your repo to Google Drive at: `My Drive/udms-image-classifier/`

Create `notebooks/03_train_classifier.ipynb` in Google Colab.

**Every Colab notebook must start with this setup cell:**
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Add project to Python path so imports work
import sys
sys.path.insert(0, '/content/drive/MyDrive/udms-image-classifier')

# Verify imports
from src.training.model import build_model
from src.data.category_mapping import UDMS_CATEGORIES, NUM_CLASSES
print(f"Categories: {UDMS_CATEGORIES}")

# Set data and checkpoint paths (ALWAYS on Drive, not Colab local)
DATA_DIR = "/content/drive/MyDrive/udms-data/processed"
CHECKPOINT_DIR = "/content/drive/MyDrive/udms-image-classifier/models/"
```

**Browser tabs to open (read these BEFORE writing any code):**
1. `tensorflow.org/tutorials/images/transfer_learning` — the official MobileNetV2 tutorial. Read end to end.
2. `github.com/arpanpramanik2003/smart-waste-classification` — look at their model architecture. It's nearly identical to what you need.
3. `github.com/trekhleb/machine-learning-experiments` — their MobileNetV2 notebook for a clean walkthrough.

### Step 12 — Build the model architecture

In `src/training/model.py` (locally in VS Code), write:

```python
# Build MobileNetV2 transfer learning model for 7-class urban disorder classification
# Architecture: MobileNetV2 (ImageNet) → GlobalAveragePooling → Dropout(0.3) →
# Dense(128, ReLU) → Dropout(0.2) → Dense(7, Softmax)
# Phase 1: freeze all conv layers, train head only
# Phase 2: unfreeze last 30 layers, fine-tune at 1e-5
```

Let Copilot generate. Compare with what you read from the reference repos. Adjust.

### Step 13 — Phase 1 training (classification head only)

In Colab notebook: load MobileNetV2 with frozen backbone, train for 15–20 epochs.

Experiment with learning rates: 1e-3, 3e-4, 1e-4. Monitor accuracy/loss curves.

**Save checkpoint to Drive:** `{CHECKPOINT_DIR}/phase1_best.h5`
⚠️ Never save to Colab local `/content/` — it resets on disconnect.

### Step 14 — Phase 2 training (fine-tuning)

Unfreeze last 30 layers. Set learning rate to **exactly 1e-5** (higher destroys pretrained features). Add ReduceLROnPlateau + EarlyStopping. Train for 10–15 epochs.

**Save checkpoint to Drive:** `{CHECKPOINT_DIR}/phase2_best.h5`

### Step 15 — Evaluate

Open `src/training/evaluate.py`.

**Browser tab:** `github.com/akathedeveloper/Trash-Classifier` — study their evaluation metrics and multi-model benchmarking approach.

Generate: confusion matrix, per-class precision/recall/F1, misclassification analysis.

Save results to `docs/evaluation_report.md`.

**Check:** Is overall accuracy ≥75%? If not:
- Are certain categories dragging you down? → Get more data for them, or merge into "Other"
- Is domain mismatch the issue? → Add more African-city images via web scraping
- Try EfficientNet-B0 as backup if MobileNetV2 plateaus

### Step 16 — Export model

**First: Copy the trained model from Google Drive to your local machine.**
```bash
# Download phase2_best.h5 from Google Drive → local models/ directory
# (Use Drive sync, manual download, or gdown CLI)
```

Then open `src/training/export.py` locally and run:

Convert the best `.h5` model to:
- `models/classifier.tflite` (primary — for CPU inference)
- `models/classifier.onnx` (alternative)

`models/label_map.json` should already exist from Step 2 (generated by `category_mapping.py`). Verify it has exactly 7 entries.

Benchmark inference speed: must be <500ms on CPU.

### Step 17 — Commit Week 2 deliverables

```bash
git add src/training/ models/label_map.json docs/evaluation_report.md notebooks/
git commit -m "feat: trained classifier — 78% accuracy, exported TFLite + ONNX"
```

---

## Phase 3: API Development (Week 3)

### Step 18 — Build the Predictor class

Open `src/inference/classifier.py`.

**Browser tabs to open:**
1. `github.com/CVxTz/FastImageClassification` — study the predictor class pattern. This is your primary reference.
2. `github.com/robmarkcole/tensorflow-lite-rest-server` — study the TFLite inference code.

```python
# UDMSClassifier: loads TFLite model + label map once,
# accepts raw image bytes, preprocesses, runs inference,
# returns {category, label, confidence, requires_review, alternatives}
# Confidence threshold configurable via config
```

### Step 19 — Build the FastAPI service

Open `app/main.py`.

```python
# FastAPI app for UDMS Image Classification
# Startup: load UDMSClassifier singleton
# Endpoints: POST /api/v1/classify, GET /health, GET /model/info
# Middleware: request logging, CORS
```

Open `app/routes/classify.py`:
```python
# POST /api/v1/classify
# Accepts: multipart/form-data with image file
# Validates: file type (jpg/png/webp), file size (≤10MB)
# Returns: ClassifyResponse with prediction, alternatives, model_version, inference_time_ms
```

Open `app/schemas.py`:
```python
# Pydantic models matching the API response schema from the project plan
# ClassifyResponse, PredictionResult, AlternativeResult, ErrorResponse
```

Open `app/config.py`:
```python
# Settings loaded from environment variables
# MODEL_PATH, LABEL_MAP_PATH, CONFIDENCE_THRESHOLD (default 0.6),
# MAX_FILE_SIZE_MB (default 10), LOG_LEVEL
```

### Step 20 — Dockerize

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app/
COPY src/ ./src/
COPY models/ ./models/
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Create `docker-compose.yml`:
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
    volumes:
      - ./models:/app/models
```

**Browser tab:** `github.com/eRuaro/food-vision-api` — simplest Docker + FastAPI reference if you get stuck.

### Step 21 — Write the integration guide

Create `docs/integration_guide.md` explaining how the UDMS backend should call your service: endpoint URL, request format, response parsing, error handling, retry strategy.

### Step 22 — Test the container

```bash
docker-compose up --build
# In another terminal:
curl -X POST http://localhost:8000/api/v1/classify \
  -F "image=@tests/fixtures/sample_pothole.jpg"
```

### Step 23 — Commit Week 3 deliverables

```bash
git add app/ src/inference/ Dockerfile docker-compose.yml docs/integration_guide.md docs/api_spec.yaml
git commit -m "feat: FastAPI microservice — Dockerized, documented API"
```

---

## Phase 4: Testing, Docs & Demo (Week 4)

### Step 24 — Write tests

Open `tests/test_preprocessing.py`, `tests/test_classifier.py`, `tests/test_api_classify.py`.

Target: ≥80% code coverage.

```bash
pytest --cov=src --cov=app --cov-report=term-missing
```

### Step 25 — Edge case testing

Open `tests/test_edge_cases.py`. Test with:
- Corrupted image files
- Wrong file types (.pdf, .txt)
- Extremely large files (>10MB)
- Completely ambiguous images
- Very small images (10x10 pixels)

Save results to `docs/edge_case_results.md`.

### Step 26 — Stress test

Create `scripts/benchmark_inference.py` using Locust:
- Measure: requests/second, p50/p95/p99 latency
- Verify: API response time <1 second under concurrent load

Save results to `docs/performance_report.md`.

### Step 27 — Build the demo UI

Open `demo/streamlit_app.py`.

**Browser tabs:**
1. `github.com/arpanpramanik2003/smart-waste-classification` — their Streamlit app
2. `github.com/akathedeveloper/Trash-Classifier` — their Gradio app

Build a simple UI: upload image → see prediction with confidence bar + category label.

```bash
streamlit run demo/streamlit_app.py
```

### Step 28 — Complete documentation

Create/finalize these files:
- `docs/model_card.md` — model metadata, intended use, limitations, bias considerations
- `docs/retraining_guide.md` — step-by-step instructions for adding new data and retraining
- `docs/architecture_diagram.png` — export from the diagram I created above
- `README.md` — project overview, quickstart, API usage, architecture summary

### Step 29 — Final commit and cleanup

```bash
# Clean up any unused files
black src/ app/ tests/ demo/
ruff check src/ app/ tests/

git add .
git commit -m "feat: complete project — tested, documented, demo-ready"
git tag v1.0.0
git push --tags
```

### Step 30 — Prepare final presentation

Create `docs/final_presentation.pdf` covering:
- Problem statement (urban disorder classification for African cities)
- Technical approach (MobileNetV2, transfer learning, confidence gating)
- Results (accuracy, F1 scores, inference speed)
- Demo (live Streamlit walkthrough)
- What's next (Phase 2 roadmap: multi-label, YOLO detection, severity scoring)

---

## Quick Reference: Which Browser Tab, Which Step

| Step | Repo to READ in browser | What to look at |
|------|------------------------|-----------------|
| 6 | smart-waste-classification | Image preprocessing |
| 7 | albumentations (examples) | Augmentation pipeline |
| 11 | TensorFlow transfer learning tutorial | End-to-end MobileNetV2 |
| 11 | smart-waste-classification | Model architecture |
| 11 | trekhleb/machine-learning-experiments | Clean notebook walkthrough |
| 15 | Trash-Classifier (akathedeveloper) | Evaluation + benchmarking |
| 18 | FastImageClassification (CVxTz) | Predictor class pattern |
| 18 | tensorflow-lite-rest-server | TFLite inference serving |
| 20 | food-vision-api (eRuaro) | Simple Docker + FastAPI |
| 27 | smart-waste-classification | Streamlit demo |
| 27 | Trash-Classifier | Gradio demo |

**Additional repos (from my recommendations):**

| When to check | Repo | What for |
|---------------|------|----------|
| Step 5 (data) | sekilab/RoadDamageDetector | Multi-country road damage images |
| Step 5 (data) | IllegalDumpSiteDetection (UNICEF) | Dump site images from developing countries |
| Step 12 (model) | nive927/Pothole-and-Plain-Road-Classification | Comparing classifier heads |
| Step 15 (eval) | Sshanu/civic_issue_dataset | Crop around bounding boxes for more images |
| Step 21 (integrate) | clean-city-watch | End-to-end system architecture reference |

---

## The Rule

**One repo on your machine. Reference repos in browser tabs. Datasets in `data/raw/`. Your own code written with Copilot.**
