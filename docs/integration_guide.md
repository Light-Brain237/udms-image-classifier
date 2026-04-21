# UDMS Classifier Integration Guide (Week 3)

This guide explains how UDMS backend services can call the image classifier API.

## 1. Overview

The UDMS Image Classifier service receives an uploaded image and returns:

- The predicted UDMS category
- A confidence score
- A review flag for low-confidence predictions
- Alternative category candidates
- Model and inference metadata

Base URL:

```text
http://localhost:8000
```

## 2. Endpoints

### GET /health

Purpose: Verify that the service is running and the model is loaded.

Example response:

```json
{
	"status": "healthy",
	"model_loaded": true,
	"model_version": "1.0.0"
}
```

### POST /api/v1/classify

Purpose: Classify one image into a UDMS category.

## 3. Request Format for /classify

- Method: POST
- Content-Type: multipart/form-data
- Parameter name: file
- Accepted formats: jpg, jpeg, png, webp
- Max file size: 10MB

Example request (curl):

```bash
curl -X POST "http://localhost:8000/api/v1/classify" \
	-H "accept: application/json" \
	-H "Content-Type: multipart/form-data" \
	-F "file=@sample.jpg"
```

## 4. Response Format

The full success response JSON:

```json
{
	"prediction": {
		"category": "pothole_road",
		"category_label": "Pothole / Road Damage",
		"confidence": 0.8742,
		"requires_review": false
	},
	"alternatives": [
		{
			"category": "damaged_signage",
			"confidence": 0.0821
		},
		{
			"category": "other",
			"confidence": 0.0264
		},
		{
			"category": "illegal_dumping",
			"confidence": 0.0107
		}
	],
	"model_version": "1.0.0",
	"inference_time_ms": 143.7
}
```

Field explanation:

- category: Predicted UDMS category ID. Located at prediction.category.
- category_label: Human-readable label for the predicted category. Located at prediction.category_label.
- confidence: Probability score for the top prediction, from 0.0 to 1.0. Located at prediction.confidence.
- requires_review: True when confidence is below the review threshold. Located at prediction.requires_review.
- alternatives: List of runner-up categories with confidence scores.
- model_version: Version string of the model used for inference.
- inference_time_ms: Time spent in model inference in milliseconds.

## 5. Error Responses

The API returns HTTP 400 for input validation issues and HTTP 500 for inference errors.

### 400 - Invalid file type

Example:

```json
{
	"detail": "Invalid file type 'application/pdf'. Allowed: jpg, jpeg, png, webp."
}
```

### 400 - File too large

Example:

```json
{
	"detail": "File size 12400000 bytes exceeds limit of 10 MB."
}
```

### 500 - Model error

Example:

```json
{
	"detail": "Model inference error: <error details>"
}
```

## 6. Confidence Thresholding

The service uses a confidence threshold of 0.6.

Rule:

- If confidence < 0.6, then requires_review is true.
- If confidence >= 0.6, then requires_review is false.

How backend should handle requires_review: true:

- Do not auto-close the report based only on the classifier.
- Route the record to a manual review queue.
- Store both top prediction and alternatives for reviewer context.
- Allow reviewer override as the final decision.

## 7. Code Examples

### Python requests example

```python
import requests

BASE_URL = "http://localhost:8000"
IMAGE_PATH = "sample.jpg"

with open(IMAGE_PATH, "rb") as image_file:
    files = {"file": ("sample.jpg", image_file, "image/jpeg")}
    response = requests.post(f"{BASE_URL}/api/v1/classify", files=files, timeout=30)

try:
    response.raise_for_status()
    data = response.json()

    prediction = data["prediction"]
    category = prediction["category"]
    label = prediction["category_label"]
    confidence = prediction["confidence"]
    requires_review = prediction["requires_review"]

    print("Category:", category)
    print("Label:", label)
    print("Confidence:", confidence)
    print("Requires review:", requires_review)
    print("Alternatives:", data.get("alternatives", []))
    print("Model version:", data.get("model_version"))
    print("Inference time (ms):", data.get("inference_time_ms"))

except requests.HTTPError:
    error_body = {}
    try:
        error_body = response.json()
    except ValueError:
        pass

    print("Request failed")
    print("Status:", response.status_code)
    print("Error body:", error_body or response.text)
```

### Error handling notes

- Check status code before using response fields.
- For 400, treat as client input issue and return actionable validation feedback.
- For 500, retry if appropriate and log with request trace ID.

## 8. Deployment Options

### Run locally with uvicorn

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Run with Docker Compose

```bash
docker-compose up --build
```

Service is exposed at http://localhost:8000.

### Environment variables reference

Application settings:

- UDMS_MODEL_PATH: Path to the .tflite model file.
- UDMS_LABEL_MAP_PATH: Path to the label map JSON file.
- UDMS_CONFIDENCE_THRESHOLD: Float threshold for requires_review.
- UDMS_MAX_FILE_SIZE_MB: Maximum upload size in MB (default 10).
- UDMS_LOG_LEVEL: Logging level (for example INFO, DEBUG).
- UDMS_MODEL_VERSION: Version string returned in API responses.

Default values in code:

- UDMS_MODEL_PATH=models/classifier.tflite
- UDMS_LABEL_MAP_PATH=models/label_map.json
- UDMS_CONFIDENCE_THRESHOLD=0.6
- UDMS_MAX_FILE_SIZE_MB=10
- UDMS_LOG_LEVEL=INFO
- UDMS_MODEL_VERSION=1.0.0
