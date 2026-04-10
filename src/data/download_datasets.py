# ============================================================
# UDMS Image Classifier - Dataset Downloader
# Week 1, Day 1-2
# 
# What this script does:
# Downloads training image datasets from Kaggle automatically.
# Think of it like an automatic shopping cart that fetches
# all our ingredients (images) for us.
# ============================================================

# --- IMPORTS ---
# These are tools we installed earlier that we need to use

import os           # lets us work with folders and file paths
import subprocess   # lets us run command line commands from Python
from pathlib import Path  # a cleaner way to work with folder paths

# --- PROJECT PATHS ---
# Here we define where everything lives on our computer
# Path(__file__) means "the folder where THIS script is located"

# This gets the root project folder (udms_classifier)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# This is where raw downloaded images will be saved
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# --- DATASET DEFINITIONS ---
# This is our "shopping list" of datasets to download
# Format is: "kaggle-username/dataset-name"
# We are starting with the most important ones first
# as recommended in the Dataset Sources Reference Guide

DATASETS = [
    {
        "name": "Road Issues Detection Dataset",        # human friendly name
        "source": "huggingface",                        # where it comes from
        "kaggle_ref": None,                             # not on kaggle
        "priority": "CRITICAL",                         # from our reference doc
        "categories": [1, 2, 5, 7],                    # which UDMS categories it covers
        "notes": "Download manually from HuggingFace"  # special instructions
    },
    {
        "name": "Urban Visual Pollution Dataset",
        "source": "kaggle",
        "kaggle_ref": "abhranta/urban-visual-pollution-dataset",
        "priority": "CRITICAL",
        "categories": [1, 2, 3, 5],
        "notes": "Only source for Category 3 (Street Lighting)"
    },
    {
        "name": "Civic Issues - Potholes and Garbage",
        "source": "mendeley",
        "kaggle_ref": None,
        "priority": "HIGH",
        "categories": [1, 2],
        "notes": "Download manually from Mendeley"
    },
    {
        "name": "Road Hazards - Potholes Cracks Manholes",
        "source": "kaggle",
        "kaggle_ref": "sabidrahman/pothole-cracks-and-openmanhole",
        "priority": "HIGH",
        "categories": [2, 4],
        "notes": "Also covers open manholes for Category 4"
    },
    {
        "name": "Flood Classification Dataset",
        "source": "kaggle",
        "kaggle_ref": "dhawalsrivastava2583/flood-classification-dataset",
        "priority": "MEDIUM",
        "categories": [4],
        "notes": "Water and sewage proxy dataset"
    },
    {
        "name": "Damaged Signs Dataset",
        "source": "kaggle",
        "kaggle_ref": "danielvareta/damaged-signs-dataset",
        "priority": "MEDIUM",
        "categories": [5],
        "notes": "Signage supplement"
    },
]

# --- FOLDER CREATION ---
# This function creates a folder for each dataset
# so images stay organised and do not mix together

def create_dataset_folder(dataset_name):
    """
    Creates a folder inside data/raw/ for a specific dataset.
    
    For example:
    data/raw/urban_visual_pollution/
    data/raw/road_hazards/
    """
    # Convert dataset name to a folder-friendly format
    # e.g. "Urban Visual Pollution Dataset" becomes "urban_visual_pollution_dataset"
    folder_name = dataset_name.lower().replace(" ", "_").replace("-", "_")
    
    # Create the full path
    folder_path = RAW_DATA_DIR / folder_name
    
    # Create the folder if it does not already exist
    folder_path.mkdir(parents=True, exist_ok=True)
    
    return folder_path


# --- KAGGLE DOWNLOADER ---
# This function downloads a dataset from Kaggle automatically

def download_from_kaggle(kaggle_ref, destination_folder):
    """
    Downloads a dataset from Kaggle using the kaggle CLI tool.
    
    kaggle_ref: the dataset reference e.g. "abhranta/urban-visual-pollution-dataset"
    destination_folder: where to save the downloaded files
    """
    print(f"\n📥 Downloading: {kaggle_ref}")
    print(f"📁 Saving to: {destination_folder}")
    
    # This is the same as typing in Command Prompt:
    # kaggle datasets download -d abhranta/urban-visual-pollution-dataset --unzip -p data/raw/...
    command = [
        "kaggle",
        "datasets",
        "download",
        "-d", kaggle_ref,      # the dataset reference
        "--unzip",             # automatically unzip the downloaded file
        "-p", str(destination_folder)  # where to save it
    ]
    
    # Run the command and show output
    result = subprocess.run(command, capture_output=False, text=True)
    
    # Check if it worked
    if result.returncode == 0:
        print(f"✅ Successfully downloaded: {kaggle_ref}")
    else:
        print(f"❌ Failed to download: {kaggle_ref}")
        print("Please check your internet connection and try again")


# --- MAIN FUNCTION ---
# This is the main function that runs everything
# Think of it as the manager that coordinates all the steps

def main():
    """
    Main function that:
    1. Creates folders for each dataset
    2. Downloads all Kaggle datasets automatically
    3. Lists manual download instructions for non-Kaggle datasets
    """
    
    print("=" * 60)
    print("UDMS Image Classifier - Dataset Downloader")
    print("Week 1, Day 1-2")
    print("=" * 60)
    
    # Create the raw data folder if it does not exist
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Keep track of manual downloads needed
    manual_downloads = []
    
    # Loop through every dataset in our shopping list
    for dataset in DATASETS:
        
        # Create a folder for this dataset
        folder = create_dataset_folder(dataset["name"])
        
        print(f"\n{'=' * 40}")
        print(f"Dataset  : {dataset['name']}")
        print(f"Priority : {dataset['priority']}")
        print(f"Categories covered: {dataset['categories']}")
        
        # Check if it is a Kaggle dataset
        if dataset["source"] == "kaggle" and dataset["kaggle_ref"]:
            download_from_kaggle(dataset["kaggle_ref"], folder)
            
        else:
            # Not a Kaggle dataset - needs manual download
            manual_downloads.append(dataset)
            print(f"⚠️  Manual download required")
            print(f"📝 Note: {dataset['notes']}")
    
    # At the end, show instructions for manual downloads
    if manual_downloads:
        print("\n" + "=" * 60)
        print("MANUAL DOWNLOADS REQUIRED")
        print("The following datasets need to be downloaded manually:")
        print("=" * 60)
        
        for dataset in manual_downloads:
            print(f"\n📌 {dataset['name']}")
            print(f"   Priority  : {dataset['priority']}")
            print(f"   Source    : {dataset['source']}")
            print(f"   Notes     : {dataset['notes']}")
            
            if dataset["source"] == "huggingface":
                print(f"   URL: https://huggingface.co/datasets/Programmer-RD-AI/road-issues-detection-dataset")
            elif dataset["source"] == "mendeley":
                print(f"   URL: https://data.mendeley.com/datasets/zndzygc3p3/2")
    
    print("\n" + "=" * 60)
    print("Download session complete!")
    print(f"Check your data/raw/ folder for downloaded images")
    print("=" * 60)


# --- ENTRY POINT ---
# This means "only run main() if this script is run directly"
# (not if it is imported by another script)

if __name__ == "__main__":
    main()