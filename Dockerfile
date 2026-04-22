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

ENV MODEL_PATH=/app/models/classifier.tflite
ENV LABEL_MAP_PATH=/app/models/label_map.json
ENV CONFIDENCE_THRESHOLD=0.6

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


