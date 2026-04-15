#!/bin/bash
set -e

echo "UDMS Image Classifier — Dataset Downloader"
echo "============================================"
echo "Running Python download script..."

python -m src.data.download_datasets

echo ""
echo "Running web scraper for weak categories..."
python -m src.data.scraper

echo ""
echo "Done. Check data/raw/ for downloaded files."
echo "Next: python -m src.data.clean_dataset"
