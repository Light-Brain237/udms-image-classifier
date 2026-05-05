"""
pothole_scraper.py
Category-targeted image scraper with three-layer filtering.

Pipeline:
  1. Multi-engine scrape (Google, Bing, DuckDuckGo)
     - Layer 1: URL/filename token filter  (DDG: pre-download)
     - Layer 2: Alt-text signal filter     (DDG: pre-download)
  2. Perceptual-hash deduplication
  3. CLIP keep/reject scoring on pixels   (Layer 3: the real filter)

Install:
  pip install ddgs icrawler torch torchvision ftfy regex tqdm pillow imagehash requests
  pip install git+https://github.com/openai/CLIP.git

Run:
  python pothole_scraper.py

Change CATEGORY block at the top to scrape a different class.
"""

import hashlib
import re
import shutil
from pathlib import Path

import imagehash
import requests
import torch
from PIL import Image
from tqdm import tqdm

import clip
from ddgs import DDGS
from icrawler.builtin import BingImageCrawler, GoogleImageCrawler


# ============================================================
# CONFIG — edit this block to scrape a different category
# ============================================================

CATEGORY = "pothole"

# ------------------------------------------------------------
# Query generator — produces ~150 queries from city × modifier grids
# ------------------------------------------------------------

# (city_or_country, language) pairs. Language controls which modifier set applies.
PLACES = [
    # Francophone
    ("Douala", "fr"), ("Yaoundé", "fr"), ("Bafoussam", "fr"),
    ("Dakar", "fr"), ("Abidjan", "fr"), ("Bamako", "fr"),
    ("Kinshasa", "fr"), ("Ouagadougou", "fr"), ("Conakry", "fr"),
    ("Libreville", "fr"), ("Cotonou", "fr"), ("Lomé", "fr"),
    ("Cameroun", "fr"), ("Sénégal", "fr"), ("Côte d'Ivoire", "fr"),
    # Anglophone
    ("Lagos", "en"), ("Abuja", "en"), ("Kano", "en"), ("Ibadan", "en"),
    ("Benin City", "en"), ("Port Harcourt", "en"),
    ("Accra", "en"), ("Kumasi", "en"),
    ("Nairobi", "en"), ("Mombasa", "en"),
    ("Kampala", "en"), ("Dar es Salaam", "en"),
    ("Johannesburg", "en"), ("Cape Town", "en"), ("Durban", "en"),
    ("Harare", "en"), ("Lusaka", "en"), ("Kigali", "en"),
    ("Addis Ababa", "en"), ("Banjul", "en"), ("Freetown", "en"),
    ("Nigeria", "en"), ("Ghana", "en"), ("Kenya", "en"), ("Uganda", "en"),
    # Lusophone
    ("Luanda", "pt"), ("Maputo", "pt"), ("Beira", "pt"), ("Bissau", "pt"),
]

MODIFIERS = {
    "en": ["pothole", "potholes", "bad road", "damaged road",
           "road damage", "pothole rainy season"],
    "fr": ["nid de poule", "nids de poule", "route dégradée",
           "mauvaise route", "chaussée abîmée", "trou route"],
    "pt": ["buraco na estrada", "buracos estrada", "estrada danificada",
           "estrada esburacada"],
}

# Generic queries not tied to a place (catch cross-border news coverage).
GENERIC_EXTRAS = [
    "potholes rainy season Africa",
    "laterite road damage Africa",
    "dirt road erosion Africa",
    "African street pothole",
    "unpaved road damage sub-saharan",
    # User-generated content via site: operators (much higher image-caption match rate)
    "pothole site:reddit.com",
    "pothole Africa site:flickr.com",
    "potholes site:twitter.com Nigeria",
    "nid de poule site:twitter.com",
]


def build_queries() -> list[str]:
    queries = []
    for place, lang in PLACES:
        for mod in MODIFIERS[lang]:
            if lang == "en":
                queries.append(f"{mod} in {place}")
            else:
                queries.append(f"{mod} {place}")
    queries.extend(GENERIC_EXTRAS)
    return queries


QUERIES = build_queries()

# What the KEPT images should look like. Covers paved + unpaved African roads.
KEEP_PROMPTS = [
    "a close-up photograph of a pothole on a road",
    "a photograph of a damaged asphalt road with holes",
    "a photograph of a cracked and broken road surface",
    "a photograph of a pothole on a red laterite dirt road",
    "a photograph of an eroded unpaved road with gullies",
    "a photograph of a damaged tropical urban street with potholes",
    "a photograph of a flooded pothole on a road",
]

# What junk looks like. Make this specific to your observed failure modes.
REJECT_PROMPTS = [
    "a portrait photograph of a person",
    "a photograph of a politician at a podium giving a speech",
    "a photograph of bridge or building construction",
    "an illustration, cartoon, diagram, or infographic",
    "a screenshot of text, a news headline, or a tweet",
    "a logo, icon, or stock photo watermark",
    "a map, satellite image, or aerial view",
    "a car interior, dashboard, or steering wheel",
    "a photograph of a smooth undamaged road",
    "a photograph of road construction machinery",
    # out-of-distribution geography — soft reject
    "a snowy road or winter street scene",
    "a road covered in snow, ice, or frost",
]

# Soft ranking prompts — NOT used to reject, only to sort the kept folder.
# Helps you review the most in-distribution images first when labeling.
AFRICA_PROMPTS = [
    "a street scene in an African city with palm trees or tropical vegetation",
    "a road in a sub-Saharan African town with market stalls or motorbike taxis",
    "a red dirt laterite road in tropical Africa",
]
NON_AFRICA_PROMPTS = [
    "a street scene in a European city with historic stone buildings",
    "a suburban American street with neat sidewalks and maple trees",
    "a cold northern country road with pine forests",
]

# Tokens that appear in image filenames/URLs of known junk.
# Hit = skip before download (DDG) or post-download reject-flag (icrawler).
SUSPICIOUS_IMG_TOKENS = [
    # people / politicians
    "ramaphosa", "biden", "trump", "macron", "minister", "president",
    "ceo", "mayor", "portrait", "headshot", "speech", "podium",
    "press-conference", "interview",
    # non-photo / UI chrome
    "logo", "icon", "banner", "header", "footer", "thumbnail",
    "avatar", "profile", "sprite", "favicon",
    # non-road content
    "infographic", "chart", "graph", "diagram", "map", "satellite",
    # stock / watermark
    "stock", "watermark", "getty", "shutter",
]

# Domains to block at the URL level — these serve mostly watermarked stock.
BAD_URL_SUBSTR = [
    "shutterstock", "gettyimages", "alamy", "istockphoto",
    "dreamstime", "depositphotos", "123rf", ".svg",
]

# Alt-text words that signal THE IMAGE is about the topic (strong).
# Covers EN, FR, PT, SW.
ALT_STRONG = [
    # EN
    "pothole", "potholes", "road damage", "damaged road",
    "broken road", "asphalt damage", "crater", "road surface",
    # FR
    "nid de poule", "nid-de-poule", "nids de poule",
    "route dégradée", "route degradee", "chaussée dégradée",
    "mauvaise route", "trou dans la route",
    # PT
    "buraco", "buracos", "estrada danificada", "estrada esburacada",
    # SW
    "shimo", "barabara", "shimo barabarani",
]

# Alt-text words that signal AN ARTICLE about the topic, not the image.
ALT_ARTICLE_ONLY = [
    "budget", "billion", "parliament", "government", "minister",
    "announce", "speech", "statement", "report", "policy",
]

# CLIP thresholds
KEEP_MARGIN = 1.05      # best keep score must exceed best reject by this ratio
MIN_KEEP_SCORE = 0.18   # absolute floor — reject low-confidence regardless

# Scraping volume
PER_QUERY = 80        # per query per engine; with ~150 queries × 3 engines = ~36k attempts
MIN_W, MIN_H = 400, 300

# If you've already populated _raw/ from another source (e.g. Mapillary),
# set True to skip web scraping and go straight to dedup + CLIP.
SKIP_SCRAPE = True

# Output
OUT_DIR = Path("try2Ima")
RAW_DIR = OUT_DIR / "_raw"       # never touched after download
DEDUP_DIR = OUT_DIR / "_dedup"    # unique images copied here
KEEP_DIR = OUT_DIR / "keep"
REJECT_DIR = OUT_DIR / "reject"


# ============================================================
# LAYER 1 & 2 — cheap filters (before download, DDG only)
# ============================================================

def url_looks_bad(img_url: str) -> bool:
    """Layer 1: reject based on URL or filename tokens."""
    u = img_url.lower()
    if any(bad in u for bad in BAD_URL_SUBSTR):
        return True
    fname = u.split("/")[-1].split("?")[0]
    if any(tok in fname for tok in SUSPICIOUS_IMG_TOKENS):
        return True
    return False


def alt_text_ok(result: dict) -> bool:
    """Layer 2: alt-text should mention the visual subject, not just the article."""
    alt = (result.get("title", "") + " " + result.get("source", "")).lower()
    if not alt.strip():
        # no alt text — can't reject, pass through to CLIP
        return True
    has_strong = any(s in alt for s in ALT_STRONG)
    looks_like_article = any(w in alt for w in ALT_ARTICLE_ONLY)
    # If alt mentions article topics but no visual subject → probably a stock
    # header image for an article about potholes, not an image OF a pothole.
    if looks_like_article and not has_strong:
        return False
    return True


# ============================================================
# SCRAPING
# ============================================================

def download(url: str, out_dir: Path) -> Path | None:
    try:
        r = requests.get(
            url, timeout=10,
            headers={"User-Agent": "Mozilla/5.0 (compatible; DatasetBot/1.0)"},
        )
        if r.status_code != 200 or len(r.content) < 10_000:
            return None
        # content-type sanity
        ctype = r.headers.get("content-type", "").lower()
        if not ctype.startswith("image/"):
            return None
        h = hashlib.md5(r.content).hexdigest()[:12]
        ext = url.split("?")[0].rsplit(".", 1)[-1].lower()
        if ext not in ("jpg", "jpeg", "png", "webp"):
            ext = "jpg"
        path = out_dir / f"ddg_{h}.{ext}"
        path.write_bytes(r.content)
        return path
    except Exception:
        return None


def scrape_ddg(query: str, n: int, out_dir: Path) -> int:
    """DuckDuckGo — full 3-layer filtering available."""
    kept = 0
    try:
        with DDGS() as ddgs:
            for r in ddgs.images(query, max_results=n, safesearch="off"):
                url = r.get("image", "")
                if not url:
                    continue
                # Layer 1: URL/filename
                if url_looks_bad(url):
                    continue
                # Layer 2: alt text
                if not alt_text_ok(r):
                    continue
                # Size gate
                if r.get("width", 0) < MIN_W or r.get("height", 0) < MIN_H:
                    continue
                if download(url, out_dir):
                    kept += 1
    except Exception as e:
        print(f"    DDG error: {e}")
    return kept


def scrape_icrawler(query: str, n: int, out_dir: Path) -> None:
    """Google + Bing via icrawler. Pre-download metadata is limited,
    so we rely on their built-in filters + CLIP at the end."""
    for Crawler, tag in ((GoogleImageCrawler, "google"), (BingImageCrawler, "bing")):
        try:
            c = Crawler(
                storage={"root_dir": str(out_dir)},
                downloader_threads=4,
                feeder_threads=1,
                parser_threads=1,
                log_level=40,  # suppress info logs
            )
            c.crawl(
                keyword=query,
                max_num=n,
                min_size=(MIN_W, MIN_H),
                filters={"type": "photo"},
                file_idx_offset="auto",
            )
        except Exception as e:
            print(f"    {tag} error: {e}")


def prune_by_filename(raw_dir: Path) -> int:
    """Post-hoc Layer 1 for icrawler results — inspect preserved filenames."""
    removed = 0
    for p in raw_dir.iterdir():
        if any(tok in p.name.lower() for tok in SUSPICIOUS_IMG_TOKENS):
            p.unlink(missing_ok=True)
            removed += 1
    return removed


# ============================================================
# DEDUP
# ============================================================

def dedup(raw_dir: Path, dedup_dir: Path) -> int:
    """Copy unique images from raw_dir into dedup_dir. raw_dir is never modified."""
    seen = {}
    for p in list(raw_dir.iterdir()):
        try:
            with Image.open(p) as im:
                h = str(imagehash.phash(im))
            if h not in seen:
                seen[h] = p
                shutil.copy2(p, dedup_dir / p.name)
        except Exception:
            pass  # skip unreadable files, leave them in _raw
    return len(seen)


# ============================================================
# LAYER 3 — CLIP (the real filter)
# ============================================================

def clip_filter(raw_dir: Path, keep_dir: Path, reject_dir: Path) -> tuple[int, int]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  using device: {device}")
    model, preprocess = clip.load("ViT-B/32", device=device)

    def encode_prompts(prompts):
        with torch.no_grad():
            tok = clip.tokenize(prompts).to(device)
            feat = model.encode_text(tok)
            feat /= feat.norm(dim=-1, keepdim=True)
        return feat

    keep_feat = encode_prompts(KEEP_PROMPTS)
    rej_feat = encode_prompts(REJECT_PROMPTS)
    afr_feat = encode_prompts(AFRICA_PROMPTS)
    nonafr_feat = encode_prompts(NON_AFRICA_PROMPTS)

    kept = rejected = 0
    files = list(raw_dir.iterdir())
    for p in tqdm(files, desc="  CLIP scoring"):
        try:
            img = Image.open(p).convert("RGB")
            t = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                f = model.encode_image(t)
                f /= f.norm(dim=-1, keepdim=True)
                best_keep = (f @ keep_feat.T).max().item()
                best_rej = (f @ rej_feat.T).max().item()
                best_afr = (f @ afr_feat.T).max().item()
                best_nonafr = (f @ nonafr_feat.T).max().item()
                # Africa relevance: positive if image looks more African than not.
                # Range roughly -0.15 to +0.15. Used only for ranking, not filtering.
                africa_score = best_afr - best_nonafr

            # Filename encodes: africa-score, keep-score, reject-score
            # Sort keep/ alphabetically → most in-distribution African roads first.
            # 'a' prefix = Africa-high, 'z' prefix = Africa-low (sorts last).
            tier = "a" if africa_score > 0.02 else ("m" if africa_score > -0.02 else "z")
            stem = f"{tier}_afr{africa_score:+.3f}_k{best_keep:.2f}_r{best_rej:.2f}_{p.name}"

            if best_keep >= MIN_KEEP_SCORE and best_keep >= KEEP_MARGIN * best_rej:
                shutil.copy2(p, keep_dir / stem)
                kept += 1
            else:
                shutil.copy2(p, reject_dir / stem)
                rejected += 1
        except Exception:
            pass

    return kept, rejected


# ============================================================
# MAIN
# ============================================================

def main():
    for d in (RAW_DIR, DEDUP_DIR, KEEP_DIR, REJECT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print(f"=== Scraping '{CATEGORY}' ===\n")

    if SKIP_SCRAPE:
        print("[1/4] Skipping web scrape (SKIP_SCRAPE=True). Using existing _raw/.\n")
    else:
        print(f"[1/4] Scraping {len(QUERIES)} queries × ~3 engines...")
        for q in QUERIES:
            print(f"  • {q}")
            scrape_icrawler(q, PER_QUERY, RAW_DIR)
            ddg_kept = scrape_ddg(q, PER_QUERY, RAW_DIR)
            print(f"      ddg saved: {ddg_kept}")

    total_raw = len(list(RAW_DIR.iterdir()))
    print(f"  raw total: {total_raw}\n")

    print("[2/4] Perceptual-hash dedup...")
    unique = dedup(RAW_DIR, DEDUP_DIR)
    print(f"  unique: {unique}  →  {DEDUP_DIR}")

    pruned = prune_by_filename(DEDUP_DIR)
    print(f"  filename-pruned from dedup: {pruned}\n")

    print("[3/4] CLIP filtering (pixels)...")
    # Clear previous keep/reject so old results don't mix with new ones
    for d in (KEEP_DIR, REJECT_DIR):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
    kept, rejected = clip_filter(DEDUP_DIR, KEEP_DIR, REJECT_DIR)

    print("\n[4/4] Done.")
    total = kept + rejected
    precision = kept / total if total else 0
    print(f"  kept:     {kept:5d}  →  {KEEP_DIR}")
    print(f"  rejected: {rejected:5d}  →  {REJECT_DIR}")
    print(f"  precision: {precision:.1%}")
    print("\nFilename legend:  a_/m_/z_ = Africa-high / mid / low relevance")
    print("                  afr±N.NNN = africa_score (positive = more African)")
    print("                  kN.NN     = CLIP keep-score")
    print("                  rN.NN     = CLIP reject-score")
    print("\nNext steps:")
    print(f"  1. `ls {KEEP_DIR}` — sorted alphabetically, the 'a_' prefixed")
    print(f"     images are your highest-priority African-context potholes.")
    print(f"  2. Spot-check {REJECT_DIR} — any real potholes? lower MIN_KEEP_SCORE.")
    print(f"  3. Human-label high-tier keeps first (label-studio / CVAT).")
    print(f"  4. If you end up with too few 'a_' tier images, add more city-")
    print(f"     specific queries and re-run — QUERIES list at top of script.")


if __name__ == "__main__":
    main()
