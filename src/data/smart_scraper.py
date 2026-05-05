"""
UDMS Smart Image Scraper
========================
Searches Google and Bing for urban-disorder images, verifies each download
against the trained TFLite classifier, deduplicates across runs, and saves
verified images organised by category.

Usage examples
--------------
# Scrape all 5 categories (base queries)
python -m src.data.smart_scraper

# Scrape specific categories only
python -m src.data.smart_scraper --categories potholes vegetation_overgrowth

# Second run — different search terms avoid re-downloading the same pages
python -m src.data.smart_scraper --variation 1

# Higher confidence threshold (keep only very-sure predictions)
python -m src.data.smart_scraper --confidence 0.8

# Download without model verification
python -m src.data.smart_scraper --skip-verification
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# ── Internal imports ──────────────────────────────────────────────────────────
from src.data.category_mapping import UDMS_CATEGORIES, CATEGORY_LABELS

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR   = PROJECT_ROOT / "models"
TFLITE_PATH  = MODELS_DIR / "classifier.tflite"
LABEL_MAP_PATH = MODELS_DIR / "label_map.json"

# Directories that already contain training/verified images — used when
# hashing the existing dataset so we never re-download what we already have.
EXISTING_DATA_DIRS = [
    PROJECT_ROOT / "data" / "processed" / "all",
]

# ── Model constants ───────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)          # MobileNetV2 input resolution

# ── CLIP verification prompts ─────────────────────────────────────────────────
# Keep prompts describe images we WANT for each category.
# Reject prompts are shared and filter obvious non-urban-disorder content.
# An image passes if its top-scoring CLIP prompt is one of the keep prompts.
CLIP_KEEP_PROMPTS: dict[str, list[str]] = {
    "potholes": [
        "a pothole on a road",
        "a damaged road surface with holes",
        "cracked asphalt pavement",
        "road damage in an urban street",
    ],
    "illegal_dumping": [
        "illegal dumping of garbage on a street",
        "a pile of waste on the sidewalk",
        "rubbish dumped on an urban road",
        "garbage scattered on a city street",
    ],
    "bad_drainage": [
        "an open or broken manhole cover on a road",
        "sewage overflow flooding a street",
        "a burst water pipe on a road",
        "stagnant water flooding a city street",
    ],
    "damaged_signage": [
        "a damaged or broken road sign",
        "a vandalized traffic sign",
        "a fallen or bent street sign",
        "a missing or illegible road sign",
    ],
    "vegetation_overgrowth": [
        "overgrown vegetation blocking a sidewalk",
        "trees or bushes encroaching on a road",
        "overgrown weeds on an urban pavement",
        "plants blocking road infrastructure",
    ],
}

CLIP_REJECT_PROMPTS: list[str] = [
    "a cartoon or illustration or drawing",
    "a text infographic on a white background",
    "a close-up portrait of a person's face",
    "a car interior or vehicle dashboard",
    "a pristine clean road with no damage",
    "an aerial or satellite view",
    "an indoor room or building interior",
    "a food or restaurant scene",
    "a map or diagram or chart",
]

# ── Search queries per category (Africa-specific) ────────────────────────────
BASE_QUERIES: dict[str, list[str]] = {
    "illegal_dumping": [
        "illegal dumping Lagos Nigeria",
        "garbage on street Africa",
        "waste dumping urban Africa",
        "rubbish pile sidewalk Nigeria",
    ],
    "potholes": [
        "pothole road Lagos Nigeria",
        "damaged road surface Africa",
        "road cracks potholes Nigeria",
        "broken road infrastructure Africa",
    ],
    "bad_drainage": [
        "open manhole cover Lagos",
        "sewage overflow street Africa",
        "burst pipe road Nigeria",
        "stagnant water road Africa",
    ],
    "damaged_signage": [
        "damaged road sign Nigeria",
        "broken traffic sign Africa",
        "vandalised road sign Lagos",
        "damaged guardrail road Africa",
    ],
    "vegetation_overgrowth": [
        "overgrown sidewalk Lagos Nigeria",
        "tree blocking road sign Africa",
        "bush overgrown pavement Nigeria",
        "vegetation blocking road Africa",
    ],
}

# Suffixes appended to every query when --variation N is used.
# Each variation produces different search-engine result pages, so successive
# runs retrieve new images rather than the same URLs as before.
QUERY_VARIATIONS: dict[int, str] = {
    0: "",
    1: " 2024",
    2: " street level",
    3: " citizen report",
    4: " phone camera",
    5: " daytime",
}

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_interpreter(model_path: Path) -> "Interpreter":  # type: ignore[name-defined]
    """Load the TFLite interpreter once at startup.

    Tries ai_edge_litert first (the production runtime used by the API),
    falls back to the bundled TensorFlow Lite interpreter.

    Args:
        model_path: Absolute path to ``classifier.tflite``.

    Returns:
        An allocated ``Interpreter`` instance ready for inference.
    """
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter  # type: ignore[no-redef]

    interpreter = Interpreter(
        model_path=str(model_path),
        experimental_default_delegate_latest_features=True,
    )
    interpreter.allocate_tensors()
    logger.info("TFLite model loaded from %s", model_path)
    return interpreter


def load_label_map(label_map_path: Path) -> dict[str, str]:
    """Load index-to-category mapping from label_map.json.

    Args:
        label_map_path: Path to ``label_map.json``.

    Returns:
        Dict mapping string index → category name, e.g. ``{"4": "pothole_road"}``.
    """
    with open(label_map_path, encoding="utf-8") as fh:
        raw: dict = json.load(fh)
    # label_map.json stores {"0": {"category": "...", "label": "..."}, ...}
    return {k: v["category"] for k, v in raw.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# HASHING & DUPLICATE TRACKING
# ═══════════════════════════════════════════════════════════════════════════════

def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's raw bytes.

    Calculated on the raw bytes on disk — not on a processed numpy array —
    so the hash is stable regardless of how the image is decoded later.

    Args:
        path: Path to any file.

    Returns:
        64-character hex string.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_download_log(log_path: Path) -> dict[str, dict]:
    """Load the persistent download log from JSON.

    The log tracks every image ever downloaded so we never store duplicates
    across multiple scraping sessions.

    Args:
        log_path: Path to ``download_log.json``.

    Returns:
        Dict keyed by SHA-256 hash → metadata dict.
    """
    if log_path.exists():
        try:
            with open(log_path, encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read download log (%s) — starting fresh.", exc)
    return {}


def save_download_log(log_path: Path, log: dict[str, dict]) -> None:
    """Atomically write the download log back to disk.

    Args:
        log_path: Destination path.
        log: The full log dict (hash → metadata).
    """
    tmp = log_path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(log, fh, indent=2)
        tmp.replace(log_path)
    except OSError as exc:
        logger.error("Failed to save download log: %s", exc)


def load_url_history(history_path: Path) -> set[str]:
    """Load the set of all previously downloaded URLs.

    Args:
        history_path: Path to ``url_history.txt``.

    Returns:
        Set of URL strings.
    """
    if history_path.exists():
        try:
            return set(history_path.read_text(encoding="utf-8").splitlines())
        except OSError as exc:
            logger.warning("Could not read URL history (%s) — starting fresh.", exc)
    return set()


def append_urls(history_path: Path, new_urls: list[str]) -> None:
    """Append newly downloaded URLs to the persistent URL history file.

    Args:
        history_path: Path to ``url_history.txt``.
        new_urls: URLs collected during this session.
    """
    if not new_urls:
        return
    try:
        with open(history_path, "a", encoding="utf-8") as fh:
            fh.write("\n".join(new_urls) + "\n")
    except OSError as exc:
        logger.error("Failed to append URLs to history: %s", exc)


def hash_existing_dataset(existing_dirs: list[Path]) -> set[str]:
    """Compute SHA-256 hashes of all images already in the dataset.

    Searches every sub-folder of each supplied directory so we never add
    an image that was already collected during a previous data-preparation run.

    Args:
        existing_dirs: Directories to walk, e.g. ``data/processed/all/``.

    Returns:
        Set of hex-digest strings.
    """
    hashes: set[str] = set()
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    total = 0
    for root_dir in existing_dirs:
        if not root_dir.exists():
            continue
        for img_path in root_dir.rglob("*"):
            if img_path.suffix.lower() not in extensions:
                continue
            try:
                hashes.add(sha256_file(img_path))
                total += 1
            except OSError:
                pass
    logger.info("Hashed %d existing dataset images.", total)
    return hashes


# ═══════════════════════════════════════════════════════════════════════════════
# CRAWLING
# ═══════════════════════════════════════════════════════════════════════════════

def crawl_query(
    query: str,
    save_dir: Path,
    max_images: int,
    *,
    min_size: tuple[int, int] = (200, 200),
) -> int:
    """Download images for a single search query using BingImageCrawler.

    Google Images is not used: since mid-2023 Google's bot-detection rejects
    icrawler's requests, causing the parser thread to receive HTML it cannot
    parse and raise ``TypeError: 'NoneType' object is not iterable`` on every
    single query.  Bing returns results reliably and is used as the sole engine.

    Args:
        query:      The search string to pass to the image crawler.
        save_dir:   Directory where raw downloads land.
        max_images: Maximum number of images to request.
        min_size:   ``(width, height)`` minimum accepted image dimensions.

    Returns:
        Number of new files present in ``save_dir`` after crawling.
    """
    from icrawler.builtin import BingImageCrawler  # type: ignore[import]

    save_dir.mkdir(parents=True, exist_ok=True)
    before = len(list(save_dir.iterdir()))

    try:
        logger.info("  Bing ← \"%s\"", query)
        bing_crawler = BingImageCrawler(
            storage={"root_dir": str(save_dir)},
            log_level=logging.WARNING,
        )
        bing_crawler.crawl(
            keyword=query,
            max_num=max_images,
            min_size=min_size,
            filters={"type": "photo"},
        )
    except Exception as exc:
        logger.warning("  Bing crawler error: %s", exc)

    after = len(list(save_dir.iterdir()))
    got = after - before
    if got == 0:
        logger.warning("  Bing returned 0 images for: \"%s\"", query)
    else:
        logger.info("  Bing returned %d images.", got)
    return got


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def classify_image(
    img_path: Path,
    interpreter: "Interpreter",  # type: ignore[name-defined]
    input_index: int,
    output_index: int,
    label_map: dict[str, str],
) -> Optional[tuple[str, float, np.ndarray]]:
    """Run TFLite inference on a single image file.

    Preprocessing matches the training pipeline:
    - PIL open → RGB → resize 224×224
    - Convert to float32 in **[0, 255]** — the model's Lambda layer applies
      ``mobilenet_v2.preprocess_input`` (→ [-1, 1]) internally.
    - Add batch dimension → shape (1, 224, 224, 3).

    Args:
        img_path:     Path to the image file to classify.
        interpreter:  Pre-allocated TFLite Interpreter.
        input_index:  Index of the model's input tensor.
        output_index: Index of the model's output tensor.
        label_map:    Mapping of string index → category name.

    Returns:
        ``(top_category, top_confidence, probs)`` tuple where ``probs`` is the
        full softmax output array, or ``None`` if inference fails.
    """
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img = img.resize(IMG_SIZE, Image.LANCZOS)
            arr = np.array(img, dtype=np.float32)   # [0, 255] — model scales internally

        input_data = np.expand_dims(arr, axis=0)     # (1, 224, 224, 3)
        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()
        probs = interpreter.get_tensor(output_index)[0]

        top_idx = int(np.argmax(probs))
        confidence = float(probs[top_idx])
        category = label_map[str(top_idx)]
        return category, confidence, probs
    except Exception as exc:
        logger.debug("Inference failed on %s: %s", img_path.name, exc)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# QUALITY CHECK
# ═══════════════════════════════════════════════════════════════════════════════

def quality_check(img_path: Path) -> bool:
    """Return True if the image passes all quality requirements.

    Rules:
    - Must be openable by PIL (not corrupt).
    - Minimum 100×100 pixels.
    - Maximum 20 MB on disk.

    Args:
        img_path: Path to the image file to inspect.

    Returns:
        ``True`` if the image passes, ``False`` otherwise.
    """
    max_bytes = 20 * 1024 * 1024  # 20 MB
    min_dim = 100

    try:
        size_bytes = img_path.stat().st_size
        if size_bytes > max_bytes:
            logger.debug("Quality fail (>20 MB): %s", img_path.name)
            return False

        with Image.open(img_path) as img:
            w, h = img.size
        if w < min_dim or h < min_dim:
            logger.debug("Quality fail (<%dpx): %s", min_dim, img_path.name)
            return False

        return True
    except Exception as exc:
        logger.debug("Quality fail (corrupt): %s — %s", img_path.name, exc)
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SCRAPER CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class SmartScraper:
    """Orchestrates downloading, verification, deduplication and reporting.

    Load once, then call :py:meth:`run` to execute the full pipeline.

    Args:
        output_dir:         Root directory for all scraped artefacts.
        confidence_threshold: Minimum model confidence to keep an image.
        max_per_query:      Max images to request per search query.
        variation:          Query-variation index (0–5).
        skip_verification:  If ``True``, skip model verification entirely.
        error_log_path:     Where to write per-image error lines.
    """

    def __init__(
        self,
        output_dir: Path,
        confidence_threshold: float = 0.7,
        max_per_query: int = 30,
        variation: int = 0,
        skip_verification: bool = False,
        skip_verify_categories: Optional[list[str]] = None,
        use_clip: bool = False,
        error_log_path: Optional[Path] = None,
    ) -> None:
        self.output_dir = output_dir
        self.confidence_threshold = confidence_threshold
        self.max_per_query = max_per_query
        self.variation = variation
        self.suffix = QUERY_VARIATIONS.get(variation, "")
        self.skip_verification = skip_verification
        # Categories exempted from model verification (use for classes the
        # current model is too biased to evaluate, e.g. pothole_road before
        # retraining with balanced data).
        self.skip_verify_categories: set[str] = set(skip_verify_categories or [])
        # When True, use CLIP ViT-B/32 for relevance filtering instead of the
        # TFLite classifier — avoids the chicken-and-egg problem where a biased
        # model rejects the images needed to retrain it.
        self.use_clip = use_clip

        # Persistent state files
        self.download_log_path = output_dir / "download_log.json"
        self.url_history_path  = output_dir / "url_history.txt"
        self.report_path       = output_dir / "scrape_report.md"
        self.error_log_path    = error_log_path or (output_dir / "errors.log")

        # In-memory state (populated during run)
        self.download_log: dict[str, dict] = {}
        self.url_history:  set[str]        = set()
        self.existing_hashes: set[str]     = set()
        self.session_hashes:  set[str]     = set()   # hashes seen THIS session

        # TFLite interpreter (loaded lazily — only when verification is needed)
        self._interpreter   = None
        self._input_index:  int = 0
        self._output_index: int = 0
        self._label_map:    dict[str, str] = {}
        self._cat_to_idx:   dict[str, int] = {}   # populated in _load_model

        # CLIP model (loaded lazily — only when --use-clip is set)
        self._clip_model       = None
        self._clip_preprocess  = None
        self._clip_device: str = "cpu"
        self._clip_module      = None   # imported clip module
        self._clip_text_tokens: dict = {}   # pre-tokenised prompts per category

        # Session counters
        self._stats: dict[str, int] = {
            "downloaded":         0,
            "skipped_existing":   0,
            "skipped_session_dup":0,
            "deleted_model":      0,
            "deleted_quality":    0,
            "deleted_clip":       0,
        }
        self._per_category: dict[str, int] = {c: 0 for c in UDMS_CATEGORIES}

        # Ensure output dir exists before the FileHandler tries to open errors.log
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # File-based error logger
        fh = logging.FileHandler(self.error_log_path, encoding="utf-8")
        fh.setLevel(logging.WARNING)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logging.getLogger().addHandler(fh)

    # ── Setup ──────────────────────────────────────────────────────────────────

    def _setup_dirs(self) -> None:
        """Create all required output directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for cat in UDMS_CATEGORIES:
            (self.output_dir / "verified" / cat).mkdir(parents=True, exist_ok=True)

    def _load_state(self) -> None:
        """Load persistent download log, URL history, and existing-dataset hashes."""
        self.download_log = load_download_log(self.download_log_path)
        self.url_history  = load_url_history(self.url_history_path)

        # Pre-populate session_hashes from the download log so that images
        # already logged from previous runs are treated as duplicates.
        self.session_hashes = set(self.download_log.keys())

        # Hash every image already in the training dataset.
        dirs_to_hash = list(EXISTING_DATA_DIRS)
        for cat in UDMS_CATEGORIES:
            p = self.output_dir / "verified" / cat
            if p.exists():
                dirs_to_hash.append(p)
        self.existing_hashes = hash_existing_dataset(dirs_to_hash)
        logger.info("Loaded %d known hashes from existing dataset.", len(self.existing_hashes))

    def _load_model(self) -> None:
        """Load TFLite interpreter and label map (called once, lazily)."""
        if self._interpreter is not None:
            return
        if not TFLITE_PATH.exists():
            raise FileNotFoundError(
                f"TFLite model not found at {TFLITE_PATH}. "
                "Train the model first or use --skip-verification."
            )
        self._interpreter = load_interpreter(TFLITE_PATH)
        self._input_index  = self._interpreter.get_input_details()[0]["index"]
        self._output_index = self._interpreter.get_output_details()[0]["index"]
        self._label_map    = load_label_map(LABEL_MAP_PATH)
        self._cat_to_idx   = {v: int(k) for k, v in self._label_map.items()}
        logger.info(
            "Model ready — %d classes: %s",
            len(self._label_map),
            list(self._label_map.values()),
        )

    def _load_clip(self) -> None:
        """Load CLIP ViT-B/32 and pre-tokenise all category prompts.

        Weights are downloaded once (~340 MB) and cached by torch hub.
        Pre-tokenisation means every call to ``_clip_filter`` only needs a
        single image-encoder forward pass — no per-image text overhead.

        Raises:
            ImportError: If ``torch`` or ``clip`` (openai-clip) are not installed.
        """
        if self._clip_model is not None:
            return
        try:
            import torch
            import clip as clip_module
        except ImportError as exc:
            raise ImportError(
                "CLIP is not installed. Run:\n"
                "  pip install torch torchvision\n"
                "  pip install git+https://github.com/openai/CLIP.git"
            ) from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip_module.load("ViT-B/32", device=device)
        model.eval()

        # Pre-tokenise every category's prompt list once
        text_tokens: dict = {}
        for cat, keep_prompts in CLIP_KEEP_PROMPTS.items():
            all_prompts_cat = keep_prompts + CLIP_REJECT_PROMPTS
            text_tokens[cat] = clip_module.tokenize(all_prompts_cat).to(device)

        self._clip_model       = model
        self._clip_preprocess  = preprocess
        self._clip_device      = device
        self._clip_module      = clip_module
        self._clip_text_tokens = text_tokens
        logger.info("CLIP ViT-B/32 loaded on %s.", device)

    # ── Per-image helpers ──────────────────────────────────────────────────────

    def _is_duplicate(self, img_path: Path) -> tuple[bool, str]:
        """Check whether an image is a duplicate.

        Args:
            img_path: Path to the downloaded image.

        Returns:
            ``(is_duplicate, sha256_hash)`` — the hash is always returned so
            the caller can register it in the log without re-computing it.
        """
        try:
            h = sha256_file(img_path)
        except OSError as exc:
            logger.warning("Could not hash %s: %s", img_path.name, exc)
            return True, ""   # treat unreadable files as duplicates to be safe

        if h in self.existing_hashes:
            self._stats["skipped_existing"] += 1
            return True, h
        if h in self.session_hashes:
            return True, h
        return False, h

    def _register_image(
        self,
        h: str,
        img_path: Path,
        category: str,
    ) -> None:
        """Record a new image in the download log and session hash set.

        Args:
            h:        SHA-256 hash of the file.
            img_path: Final path where the image lives on disk.
            category: UDMS category assigned to this image.
        """
        self.download_log[h] = {
            "hash":     h,
            "filepath": str(img_path),
            "url":      "",   # icrawler does not expose per-image URLs
            "date":     datetime.utcnow().isoformat(),
            "category": category,
        }
        self.session_hashes.add(h)

    # ── Verification ───────────────────────────────────────────────────────────

    def _clip_filter(self, img_path: Path, expected_category: str) -> bool:
        """Return True if the image is semantically relevant to ``expected_category``.

        Computes CLIP cosine similarity between the image and a combined prompt
        list (category keep prompts + shared reject prompts) in a single forward
        pass.  Accepts the image when the highest-scoring prompt is a keep prompt.

        Args:
            img_path:          Path to the image file.
            expected_category: UDMS category currently being scraped.

        Returns:
            ``True`` to keep the image, ``False`` to discard it.
        """
        import torch

        keep_prompts = CLIP_KEEP_PROMPTS.get(expected_category, [])
        if not keep_prompts:
            return True  # no prompts defined for category → pass through

        n_keep = len(keep_prompts)
        text_tokens = self._clip_text_tokens.get(expected_category)
        if text_tokens is None:
            # Fallback: tokenise on the fly (shouldn't happen after _load_clip)
            all_prompts = keep_prompts + CLIP_REJECT_PROMPTS
            text_tokens = self._clip_module.tokenize(all_prompts).to(self._clip_device)

        try:
            img = Image.open(img_path).convert("RGB")
            image_tensor = self._clip_preprocess(img).unsqueeze(0).to(self._clip_device)
            with torch.no_grad():
                logits, _ = self._clip_model(image_tensor, text_tokens)
                probs = logits.softmax(dim=-1).cpu().numpy()[0]

            best_idx = int(np.argmax(probs))
            passed = best_idx < n_keep
            if not passed:
                all_prompts = keep_prompts + CLIP_REJECT_PROMPTS
                logger.info(
                    "  CLIP rejected: %s  (matched: %r, score=%.3f)",
                    img_path.name, all_prompts[best_idx], float(probs[best_idx]),
                )
            return passed
        except Exception as exc:
            logger.debug("CLIP filter error on %s: %s", img_path.name, exc)
            return True  # on error, pass through

    def _verify_and_move(
        self,
        raw_files: list[Path],
        expected_category: str,
        timestamp: str,
    ) -> tuple[int, int]:
        """Verify, deduplicate, quality-check and move images to verified/.

        Args:
            raw_files:         List of image files from the crawler output dir.
            expected_category: Category the crawler was searching for.
            timestamp:         ``YYYYMMDD`` string for output filenames.

        Returns:
            ``(kept, deleted)`` counts.
        """
        verified_dir = self.output_dir / "verified" / expected_category
        verified_dir.mkdir(parents=True, exist_ok=True)

        existing_count = len(list(verified_dir.iterdir()))
        counter = existing_count + 1
        kept = 0
        deleted = 0

        for img_path in raw_files:
            if not img_path.is_file():
                continue

            # ── Quality check ────────────────────────────────────────────────
            if not quality_check(img_path):
                self._stats["deleted_quality"] += 1
                deleted += 1
                try:
                    img_path.unlink()
                except OSError:
                    pass
                continue

            # ── Duplicate check ──────────────────────────────────────────────
            is_dup, h = self._is_duplicate(img_path)
            if is_dup:
                self._stats["skipped_session_dup"] += 1
                deleted += 1
                try:
                    img_path.unlink()
                except OSError:
                    pass
                continue

            # ── Verification: CLIP or TFLite ─────────────────────────────────
            if self.use_clip:
                if not self._clip_filter(img_path, expected_category):
                    self._stats["deleted_clip"] += 1
                    deleted += 1
                    try:
                        img_path.unlink()
                    except OSError:
                        pass
                    continue
            else:
                verify_this = (
                    not self.skip_verification
                    and expected_category not in self.skip_verify_categories
                )
                if verify_this:
                    result = classify_image(
                        img_path,
                        self._interpreter,
                        self._input_index,
                        self._output_index,
                        self._label_map,
                    )
                    if result is None:
                        # Inference error — skip image
                        deleted += 1
                        try:
                            img_path.unlink()
                        except OSError:
                            pass
                        continue

                    pred_cat, top_confidence, probs = result

                    # Use the target class's own score rather than requiring
                    # top-1 match — keeps images where the model assigns enough
                    # probability to the expected category even if another class
                    # scores higher.
                    target_idx = self._cat_to_idx.get(expected_category)
                    if target_idx is not None:
                        target_score = float(probs[target_idx])
                    else:
                        # Fallback: use top-1 match if category not found in map
                        target_score = top_confidence if pred_cat == expected_category else 0.0

                    if target_score < self.confidence_threshold:
                        logger.info(
                            "  Model rejected: %s → top=%s(%.3f) target=%s(%.3f) threshold=%.2f",
                            img_path.name, pred_cat, top_confidence,
                            expected_category, target_score, self.confidence_threshold,
                        )
                        self._stats["deleted_model"] += 1
                        deleted += 1
                        try:
                            img_path.unlink()
                        except OSError:
                            pass
                        continue

            # ── Accept image ─────────────────────────────────────────────────
            dest_name = f"{expected_category}_smart_{timestamp}_{counter:04d}.jpg"
            dest_path = verified_dir / dest_name
            try:
                # Convert and save as JPEG to normalise format
                with Image.open(img_path) as img:
                    img.convert("RGB").save(dest_path, "JPEG", quality=95)
                try:
                    img_path.unlink()
                except OSError:
                    pass   # Windows: file still locked; safe to leave raw copy
            except Exception as exc:
                # Catches OSError (corrupt), PIL.UnidentifiedImageError,
                # struct.error (truncated download), DecompressionBombError, etc.
                logger.error(
                    "Failed to save %s: %s: %s",
                    dest_name, type(exc).__name__, exc,
                )
                try:
                    dest_path.unlink()   # remove any partial write
                except OSError:
                    pass
                continue

            self._register_image(h, dest_path, expected_category)
            self._per_category[expected_category] += 1
            counter += 1
            kept += 1

        return kept, deleted

    # ── Category scraping loop ─────────────────────────────────────────────────

    def _scrape_category(self, category: str, timestamp: str) -> None:
        """Run the full download → verify → organise pipeline for one category.

        Args:
            category:  UDMS category key, e.g. ``"pothole_road"``.
            timestamp: ``YYYYMMDD`` string used in output filenames.
        """
        queries = [q + self.suffix for q in BASE_QUERIES.get(category, [])]
        if not queries:
            logger.warning("No queries defined for category: %s", category)
            return

        label = CATEGORY_LABELS.get(category, category)
        logger.info("")
        logger.info("══ %s (%s) — %d queries ══", category, label, len(queries))

        total_kept    = 0
        total_deleted = 0
        session_urls: list[str] = []

        for query in queries:
            logger.info(" Query: \"%s\"", query)
            raw_dir = self.output_dir / "raw" / category / query.replace(" ", "_")
            raw_dir.mkdir(parents=True, exist_ok=True)

            # Download
            try:
                crawl_query(query, raw_dir, self.max_per_query)
            except Exception as exc:
                logger.error("Crawl failed for \"%s\": %s", query, exc)
                continue

            raw_files = sorted(raw_dir.iterdir())
            self._stats["downloaded"] += len(raw_files)
            session_urls.append(query)   # queries logged as-is (no raw URLs from icrawler)

            # Verify + move
            kept, deleted = self._verify_and_move(raw_files, category, timestamp)
            total_kept    += kept
            total_deleted += deleted

            logger.info(
                "  → kept %d | deleted %d", kept, deleted
            )
            time.sleep(0.5)   # be polite to search engines

        logger.info(
            "Category %s done — %d new verified images.", category, total_kept
        )
        append_urls(self.url_history_path, session_urls)

    # ── Public entry point ─────────────────────────────────────────────────────

    def run(self, categories: Optional[list[str]] = None) -> None:
        """Execute the full smart-scraping pipeline.

        Args:
            categories: Subset of UDMS_CATEGORIES to scrape.
                        ``None`` (default) scrapes all 7 categories.
        """
        target_cats = categories or UDMS_CATEGORIES
        timestamp   = datetime.utcnow().strftime("%Y%m%d")

        logger.info("═" * 60)
        logger.info("UDMS Smart Scraper — %s", timestamp)
        logger.info("Categories : %s", target_cats)
        logger.info("Variation  : %d (\"%s\")", self.variation, self.suffix)
        logger.info("Confidence : %.2f", self.confidence_threshold)
        logger.info("Max/query  : %d", self.max_per_query)
        logger.info("Verify     : %s", not self.skip_verification)
        if self.use_clip:
            logger.info("Verifier   : CLIP ViT-B/32")
        if self.skip_verify_categories:
            logger.info("Skip-verify: %s", sorted(self.skip_verify_categories))
        logger.info("═" * 60)

        self._setup_dirs()
        self._load_state()

        if self.use_clip:
            self._load_clip()
        else:
            needs_model = not self.skip_verification and any(
                c not in self.skip_verify_categories for c in target_cats
            )
            if needs_model:
                self._load_model()

        for cat in target_cats:
            if cat not in UDMS_CATEGORIES:
                logger.warning("Unknown category %r — skipping.", cat)
                continue
            try:
                self._scrape_category(cat, timestamp)
            except Exception as exc:
                logger.error("Unexpected error in category %s: %s", cat, exc)

        # Persist state
        save_download_log(self.download_log_path, self.download_log)

        self._write_report(timestamp, target_cats)
        self._print_summary()

    # ── Reporting ─────────────────────────────────────────────────────────────

    def _write_report(self, timestamp: str, categories: list[str]) -> None:
        """Write a Markdown summary report to ``scrape_report.md``.

        Args:
            timestamp:  ``YYYYMMDD`` string for the report header.
            categories: Categories that were scraped this session.
        """
        s = self._stats
        total_new = sum(self._per_category[c] for c in categories)

        lines = [
            f"# UDMS Smart Scraper Report — {timestamp}",
            "",
            "## Session Summary",
            "",
            f"| Metric | Count |",
            f"|--------|------:|",
            f"| Images downloaded this session | {s['downloaded']:,} |",
            f"| Duplicates skipped (already in dataset) | {s['skipped_existing']:,} |",
            f"| Duplicates skipped (same hash this session) | {s['skipped_session_dup']:,} |",
            f"| Deleted by model verification | {s['deleted_model']:,} |",
            f"| Deleted by CLIP filter | {s['deleted_clip']:,} |",
            f"| Deleted by quality check | {s['deleted_quality']:,} |",
            f"| **New unique verified images** | **{total_new:,}** |",
            "",
            "## New Images per Category",
            "",
            "| Category | Label | New Images |",
            "|----------|-------|----------:|",
        ]
        for cat in UDMS_CATEGORIES:
            label = CATEGORY_LABELS.get(cat, cat)
            lines.append(f"| `{cat}` | {label} | {self._per_category[cat]:,} |")

        lines += [
            "",
            "## Verified Directory Totals",
            "",
            "| Category | Total Verified Images |",
            "|----------|-----------------------:|",
        ]
        for cat in UDMS_CATEGORIES:
            verified_dir = self.output_dir / "verified" / cat
            total = len(list(verified_dir.iterdir())) if verified_dir.exists() else 0
            lines.append(f"| `{cat}` | {total:,} |")

        lines += ["", f"_Report generated {datetime.utcnow().isoformat()} UTC_", ""]

        try:
            self.report_path.write_text("\n".join(lines), encoding="utf-8")
            logger.info("Report saved to %s", self.report_path)
        except OSError as exc:
            logger.error("Could not write report: %s", exc)

    def _print_summary(self) -> None:
        """Print the final session summary to stdout."""
        s = self._stats
        total_new = sum(self._per_category.values())

        print("\n" + "═" * 60)
        print("  UDMS Smart Scraper — Session Complete")
        print("═" * 60)
        print(f"  Downloaded this session     : {s['downloaded']:>6,}")
        print(f"  Skipped (existing dataset)  : {s['skipped_existing']:>6,}")
        print(f"  Skipped (session duplicate) : {s['skipped_session_dup']:>6,}")
        print(f"  Deleted by model            : {s['deleted_model']:>6,}")
        print(f"  Deleted by CLIP filter      : {s['deleted_clip']:>6,}")
        print(f"  Deleted by quality check    : {s['deleted_quality']:>6,}")
        print(f"  New verified images         : {total_new:>6,}")
        print("─" * 60)
        print("  Per category:")
        for cat, count in self._per_category.items():
            label = CATEGORY_LABELS.get(cat, cat)
            print(f"    {cat:<22} {count:>5,}  ({label})")
        print("═" * 60)
        print(f"  Report → {self.report_path}")
        print("═" * 60 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct and return the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m src.data.smart_scraper",
        description="UDMS Smart Image Scraper — download, verify and deduplicate "
                    "urban-disorder images using icrawler + TFLite verification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=UDMS_CATEGORIES,
        default=None,
        metavar="CATEGORY",
        help=(
            "Categories to scrape. Omit to scrape all 7. "
            f"Choices: {UDMS_CATEGORIES}"
        ),
    )
    parser.add_argument(
        "--max-per-query",
        type=int,
        default=30,
        dest="max_per_query",
        help="Maximum images to download per search query.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.7,
        help="Minimum model confidence [0, 1] to accept a verified image.",
    )
    parser.add_argument(
        "--variation",
        type=int,
        default=0,
        choices=list(QUERY_VARIATIONS.keys()),
        help=(
            "Search query variation to differentiate successive runs. "
            f"Suffixes: {QUERY_VARIATIONS}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "smart_scraped",
        dest="output_dir",
        help="Root directory for all scraped artefacts.",
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        dest="skip_verification",
        help="Skip TFLite model verification — download only.",
    )
    parser.add_argument(
        "--skip-verify-categories",
        nargs="+",
        choices=UDMS_CATEGORIES,
        default=None,
        dest="skip_verify_categories",
        metavar="CATEGORY",
        help=(
            "Exempt specific categories from model verification. "
            "Use for classes the current model is too biased to evaluate "
            "(e.g. pothole_road before retraining with balanced data)."
        ),
    )
    parser.add_argument(
        "--use-clip",
        action="store_true",
        dest="use_clip",
        help=(
            "Use CLIP ViT-B/32 for relevance filtering instead of the TFLite "
            "classifier.  Works for all categories without bias.  Requires: "
            "pip install torch torchvision and "
            "pip install git+https://github.com/openai/CLIP.git"
        ),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    """Entry point for ``python -m src.data.smart_scraper``.

    Args:
        argv: Argument list (uses ``sys.argv`` when ``None``).
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    scraper = SmartScraper(
        output_dir=args.output_dir,
        confidence_threshold=args.confidence,
        max_per_query=args.max_per_query,
        variation=args.variation,
        skip_verification=args.skip_verification,
        skip_verify_categories=args.skip_verify_categories,
        use_clip=args.use_clip,
    )
    scraper.run(categories=args.categories)


if __name__ == "__main__":
    main()
