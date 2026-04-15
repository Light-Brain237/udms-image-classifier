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
