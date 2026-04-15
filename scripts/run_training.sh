#!/bin/bash
set -e

echo "UDMS Image Classifier — Training Pipeline"
echo "==========================================="
echo ""
echo "NOTE: Training should be run on Google Colab (GPU required)."
echo "This script is for reference / local testing only."
echo ""
echo "To train on Colab:"
echo "  1. Upload data/processed/ to Google Drive"
echo "  2. Open notebooks/03_train_classifier.ipynb in Colab"
echo "  3. Follow the notebook instructions"
echo ""

python -c "
from src.training.model import build_model
model = build_model(num_classes=7, freeze_backbone=True)
model.summary()
print('Model built successfully. Ready for training on Colab.')
"
