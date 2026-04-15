# UDMS Image Classifier

Automatically classifies citizen-submitted photos of urban disorder into 7 categories for the Urban Disorder Monitoring System (UDMS).

**Company:** Light Brain Technologies

## Categories

| # | Category ID | Human Label |
|---|-------------|-------------|
| 0 | illegal_dumping | Illegal Dumping / Garbage |
| 1 | pothole_road | Pothole / Road Damage |
| 2 | broken_lighting | Broken / Missing Street Lighting |
| 3 | water_sewage | Water / Sewage Issues |
| 4 | damaged_signage | Damaged Signage / Infrastructure |
| 5 | vegetation | Vegetation Overgrowth |
| 6 | other | Other / Unclassified |

## Architecture

- **Model:** MobileNetV2 transfer learning (224x224 input, 7-class softmax)
- **Inference:** TFLite on CPU (<500ms per image)
- **API:** FastAPI REST microservice, Dockerized
- **Training:** Google Colab free tier GPU

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Or with Docker
docker-compose up
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/classify` | Classify an uploaded image |
| GET | `/health` | Health check |
| GET | `/model/info` | Model metadata |

## Project Structure

See [COPILOT_INSTRUCTIONS.md](COPILOT_INSTRUCTIONS.md) for the full project specification.

## Data Pipeline

```bash
# 1. Download datasets
python -m src.data.download_datasets

# 2. Clean and map to UDMS categories
python -m src.data.clean_dataset

# 3. Split into train/val/test (70/15/15)
python -m src.data.split_dataset

# 4. Generate dataset report
python -m src.data.generate_report
```

## Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest --cov=src --cov=app --cov-report=term-missing

# Format code
black src/ app/ tests/
ruff check src/ app/ tests/
```

## License

Proprietary — Light Brain Technologies

