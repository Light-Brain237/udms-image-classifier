"""
UDMS Image Classifier — Web Scraper
Week 1, Day 1-2

What this script does:
Scrapes images from the web for categories that do not
have enough data from Kaggle/HuggingFace datasets.

Currently targeting:
- vegetation (0 images — needs 80+)
- broken_lighting (1 image — needs 100+)

Run from project root:
python -m src.data.scraper

NOTE: Always manually review scraped images before
using them for training. Web scraping can return
irrelevant or low quality images.
"""

import os
import time
import requests
from pathlib import Path
from PIL import Image
import io

# --- PROJECT PATHS ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "web_scraped"

# --- SEARCH QUERIES ---
# These are the search terms we use to find images
# Localised to African cities as per the project plan
# to reduce domain mismatch

SCRAPE_QUERIES = {
    "vegetation": [
        "overgrown sidewalk Africa street",
        "overgrown vegetation urban road Africa",
        "tree blocking road sign Africa",
        "unmaintained public park urban Africa",
        "overgrown road median Africa",
        "bush overgrown pavement developing country",
        "vegetation encroachment urban street",
        "overgrown fence urban Africa",
    ],
    "broken_lighting": [
        "broken streetlight urban Africa",
        "damaged lamp post street Africa",
        "non functional street light developing country",
        "broken street light pole urban",
        "damaged streetlight Nigeria",
        "broken street lamp Lagos",
        "faulty street light Ghana",
        "damaged light pole Africa road",
    ],
    "water_sewage": [
        "burst water pipe street Africa",
        "sewage overflow road Africa",
        "open manhole cover urban",
        "stagnant water urban street Africa",
        "flooded road Africa city",
        "blocked drain urban Africa",
    ],
}

# How many images to try to get per query
IMAGES_PER_QUERY = 20

# Minimum image size to save
MIN_SIZE = 100


def download_image(url: str, save_path: Path) -> bool:
    """
    Downloads a single image from a URL and saves it.

    url: the image URL
    save_path: where to save it

    Returns True if successful, False if failed.
    """
    try:
        # Download the image with a timeout
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                         "AppleWebKit/537.36"
        }
        response = requests.get(url, timeout=10, headers=headers)

        if response.status_code != 200:
            return False

        # Check it is actually an image
        content_type = response.headers.get("content-type", "")
        if "image" not in content_type:
            return False

        # Open with PIL to validate
        img = Image.open(io.BytesIO(response.content))

        # Check minimum size
        if img.width < MIN_SIZE or img.height < MIN_SIZE:
            return False

        # Convert to RGB and save
        rgb_img = img.convert("RGB")
        rgb_img.save(save_path, "JPEG", quality=95)
        return True

    except Exception:
        return False


def scrape_with_icrawler(category: str, query: str,
                          save_dir: Path, max_images: int):
    """
    Uses icrawler to scrape images from Google Images.

    icrawler is a Python library that automates image
    searching — like having a robot type into Google Images
    and save all the photos it finds.

    category: e.g. "vegetation"
    query: search term e.g. "overgrown sidewalk Africa"
    save_dir: where to save the images
    max_images: how many images to try to get
    """
    try:
        from icrawler.builtin import BingImageCrawler

        # Create a subfolder for this specific query
        # so we can track where each image came from
        query_folder = query.replace(" ", "_")[:30]
        output_dir = save_dir / query_folder
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check if already scraped
        existing = list(output_dir.glob("*.jpg")) + \
                   list(output_dir.glob("*.jpeg")) + \
                   list(output_dir.glob("*.png"))
        if len(existing) >= max_images:
            print(f"   ✅ Already have {len(existing)} images "
                  f"for: {query[:40]}")
            return len(existing)

        print(f"   🔍 Searching: {query[:50]}")

        # Set up the Bing Image crawler
        crawler = BingImageCrawler(
            storage={"root_dir": str(output_dir)},
            log_level=50
        )

        # Start crawling
        crawler.crawl(
            keyword=query,
            max_num=max_images,
            file_idx_offset=len(existing)
        )

        # Count what we got
        downloaded = list(output_dir.glob("*.jpg")) + \
                     list(output_dir.glob("*.jpeg")) + \
                     list(output_dir.glob("*.png"))
        print(f"   ✅ Got {len(downloaded)} images")
        return len(downloaded)

    except ImportError:
        print("   ❌ icrawler not installed")
        return 0
    except Exception as e:
        print(f"   ❌ Error scraping '{query}': {e}")
        return 0


def scrape_category(category: str):
    """
    Scrapes images for a single UDMS category.

    Loops through all search queries for that category
    and saves results to data/raw/web_scraped/<category>/
    """
    print(f"\n{'=' * 60}")
    print(f"Scraping category: {category}")
    print(f"{'=' * 60}")

    queries = SCRAPE_QUERIES.get(category, [])
    if not queries:
        print(f"❌ No queries defined for category: {category}")
        return

    # Create save directory
    save_dir = RAW_DATA_DIR / category
    save_dir.mkdir(parents=True, exist_ok=True)

    total_images = 0

    for i, query in enumerate(queries):
        print(f"\n[{i+1}/{len(queries)}] Query: {query}")

        count = scrape_with_icrawler(
            category=category,
            query=query,
            save_dir=save_dir,
            max_images=IMAGES_PER_QUERY
        )
        total_images += count

        # Small delay between searches to be polite
        time.sleep(2)

    print(f"\n✅ Total scraped for {category}: {total_images} images")
    print(f"📁 Saved to: {save_dir}")
    print(f"\n⚠️  IMPORTANT: Please manually review these images!")
    print(f"   Delete any that are irrelevant or low quality")
    print(f"   before running clean_dataset.py again")


def main():
    """
    Scrapes images for all weak categories.
    """
    print("=" * 60)
    print("UDMS Image Classifier — Web Scraper")
    print("Week 1, Day 1-2")
    print("=" * 60)
    print("Targeting weak categories:")
    print("  - vegetation (needs 80+ images)")
    print("  - broken_lighting (needs 100+ images)")
    print("  - water_sewage (needs more images)")

    # Create base directory
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Scrape each weak category
    scrape_category("vegetation")
    scrape_category("broken_lighting")
    scrape_category("water_sewage")

    print("\n" + "=" * 60)
    print("✅ Scraping complete!")
    print("=" * 60)
    print("\nNEXT STEPS:")
    print("1. Open data/raw/web_scraped/ in File Explorer")
    print("2. Manually review all downloaded images")
    print("3. Delete irrelevant or low quality images")
    print("4. Re-run: python -m src.data.clean_dataset")
    print("=" * 60)


if __name__ == "__main__":
    main()
