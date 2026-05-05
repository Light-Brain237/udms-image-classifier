"""
bad_drainage_scraper.py
Category-targeted image scraper with three-layer filtering.

Category: water_sewage / bad drainage

Pipeline:
  1. Multi-engine scrape (Bing, DuckDuckGo)
     - Layer 1: URL/filename token filter  (DDG: pre-download)
     - Layer 2: Alt-text signal filter     (DDG: pre-download)
  2. Perceptual-hash deduplication
  3. CLIP keep/reject scoring on pixels   (Layer 3: the real filter)

Target: ~10,000 raw downloads

Install:
  pip install ddgs icrawler torch torchvision ftfy regex tqdm pillow imagehash requests
  pip install git+https://github.com/openai/CLIP.git

Run:
  python bad_drainage_scraper.py
"""

import subprocess
import sys
from pathlib import Path

# ── Auto-relaunch with venv Python if not already running in it ──────────────
_VENV_PYTHON = Path(__file__).parent / "venv" / "Scripts" / "python.exe"
if _VENV_PYTHON.exists() and Path(sys.executable).resolve() != _VENV_PYTHON.resolve():
    print(f"[venv] Re-launching with {_VENV_PYTHON}")
    sys.exit(subprocess.run([str(_VENV_PYTHON), __file__] + sys.argv[1:]).returncode)
# ─────────────────────────────────────────────────────────────────────────────

import hashlib
import shutil

import imagehash
import requests
import torch
from PIL import Image
from tqdm import tqdm

import clip
from ddgs import DDGS
from icrawler.builtin import BingImageCrawler


# ============================================================
# CONFIG
# ============================================================

CATEGORY = "water_sewage"

# ============================================================
# PLACES — African city/country grid
# ============================================================

PLACES = [
    # Francophone
    ("Douala", "fr"), ("Yaoundé", "fr"), ("Bafoussam", "fr"),
    ("Dakar", "fr"), ("Abidjan", "fr"), ("Bamako", "fr"),
    ("Kinshasa", "fr"), ("Ouagadougou", "fr"), ("Conakry", "fr"),
    ("Libreville", "fr"), ("Cotonou", "fr"), ("Lomé", "fr"),
    ("Cameroun", "fr"), ("Sénégal", "fr"), ("Côte d'Ivoire", "fr"),
    ("Brazzaville", "fr"), ("Niamey", "fr"),
    # Anglophone
    ("Lagos", "en"), ("Abuja", "en"), ("Kano", "en"), ("Ibadan", "en"),
    ("Benin City", "en"), ("Port Harcourt", "en"),
    ("Accra", "en"), ("Kumasi", "en"),
    ("Nairobi", "en"), ("Mombasa", "en"),
    ("Kampala", "en"), ("Dar es Salaam", "en"),
    ("Johannesburg", "en"), ("Cape Town", "en"), ("Durban", "en"),
    ("Harare", "en"), ("Lusaka", "en"), ("Kigali", "en"),
    ("Addis Ababa", "en"), ("Freetown", "en"),
    ("Nigeria", "en"), ("Ghana", "en"), ("Kenya", "en"), ("Uganda", "en"),
    ("Tanzania", "en"), ("Cameroon", "en"),
    # Lusophone
    ("Luanda", "pt"), ("Maputo", "pt"), ("Beira", "pt"), ("Bissau", "pt"),
]

MODIFIERS = {
    "en": [
        "flooded street bad drainage",
        "overflowing sewer urban street",
        "blocked drain flooding road",
        "sewage overflow street",
        "waterlogged road bad drainage",
        "open sewer street",
        "flooded gutter street",
        "broken drain flooding",
        "stagnant water road",
        "drainage problem street flooding",
    ],
    "fr": [
        "rue inondée mauvais drainage",
        "égout débordant rue",
        "caniveau bouché inondation",
        "eaux usées débordement rue",
        "route inondée drainage",
        "fossé plein eau stagnante",
        "inondation urbaine rue",
        "canalisation cassée inondation",
    ],
    "pt": [
        "rua alagada drenagem ruim",
        "esgoto transbordando rua",
        "bueiro entupido inundação",
        "rua inundada drenagem",
        "água estagnada rua",
        "calçada alagada",
    ],
}

GENERIC_EXTRAS = [
    # English generic
    "flooded street bad drainage Africa",
    "overflowing sewer Africa urban",
    "blocked drain flood Africa",
    "sewage overflow African city",
    "open sewer Africa street",
    "waterlogged road Africa",
    "urban flooding bad drainage Africa",
    "stagnant water street Africa",
    "drainage failure Africa",
    "flooded gutter Africa",
    "sewage in street Africa",
    "open drain overflow Africa",
    "broken culvert flooding Africa",
    "street flooding rainy season Africa",
    "flood water urban Africa",
    # Social media
    "flooded street Africa site:reddit.com",
    "sewage overflow Africa site:flickr.com",
    "inondation urbaine site:twitter.com",
    "flooded road site:twitter.com Nigeria",
    # City-specific
    "flooded street Lagos drainage",
    "sewage overflow Nairobi",
    "blocked drain Accra flooding",
    "waterlogged road Kampala",
    "flooded street Douala",
    "open sewer Kinshasa",
    "drainage problem Dar es Salaam",
    "flooded road Abidjan",
    "sewage street Johannesburg",
    "urban flooding Kumasi",
    # Swahili
    "mafuriko barabarani",
    "maji machafu mtaani",
    "bomba la maji kuvunjika",
]


def build_queries() -> list[str]:
    queries = []
    seen = set()
    for place, lang in PLACES:
        for mod in MODIFIERS[lang]:
            q = f"{mod} in {place}" if lang == "en" else f"{mod} {place}"
            if q not in seen:
                seen.add(q)
                queries.append(q)
    for q in GENERIC_EXTRAS:
        if q not in seen:
            seen.add(q)
            queries.append(q)
    return queries


QUERIES = build_queries()

# ============================================================
# CLIP PROMPTS — tuned for bad drainage / water sewage
# ============================================================

KEEP_PROMPTS = [
    "a photograph of a flooded street with water overflowing from a blocked drain",
    "a photograph of sewage water overflowing onto a road in an urban area",
    "a photograph of an open sewer running along a street in an African city",
    "a photograph of a waterlogged road with stagnant water due to bad drainage",
    "a photograph of a blocked gutter or culvert causing flooding on a street",
    "a photograph of brown dirty flood water covering a road in a city",
    "a photograph of an overflowing drainage channel next to a road",
    "a photograph of a broken sewer pipe with water gushing onto a street",
    "a photograph of a flooded urban street during or after heavy rain",
    "a photograph of standing sewage water on a residential street",
]

REJECT_PROMPTS = [
    "a portrait photograph of a person",
    "a photograph of a politician giving a speech",
    "a photograph of a clean dry road with no water",
    "a photograph of a swimming pool or recreational water facility",
    "a photograph of a river, lake, or natural body of water",
    "a photograph of a flood affecting fields or farmland",
    "an illustration, cartoon, diagram, or infographic",
    "a screenshot of text, a news headline, or a tweet",
    "a logo, icon, or stock photo watermark",
    "a map, satellite image, or aerial view of flooding",
    "a photograph of water treatment infrastructure or pumping station",
    "a photograph of a pothole or damaged road with no water",
    "a photograph of garbage dumped on a dry street",
    "a snowy outdoor scene",
]

# Africa relevance ranking prompts
AFRICA_PROMPTS = [
    "a street scene in an African city with palm trees or tropical vegetation",
    "a road in a sub-Saharan African town with market stalls or motorbike taxis",
    "a flooded urban street in tropical Africa during rainy season",
]
NON_AFRICA_PROMPTS = [
    "a street scene in a European city with historic stone buildings",
    "a suburban American street with neat sidewalks and maple trees",
    "a cold northern country road with pine forests",
]

# ============================================================
# LAYER 1 — URL / filename token filter
# ============================================================

SUSPICIOUS_IMG_TOKENS = [
    "ramaphosa", "biden", "trump", "macron", "minister", "president",
    "ceo", "mayor", "portrait", "headshot", "speech", "podium",
    "press-conference", "interview",
    "logo", "icon", "banner", "header", "footer", "thumbnail",
    "avatar", "profile", "sprite", "favicon",
    "infographic", "chart", "graph", "diagram", "map", "satellite",
    "swimming-pool", "swimming_pool", "pool", "lake", "river",
    "stock", "watermark", "getty", "shutter",
]

BAD_URL_SUBSTR = [
    "shutterstock", "gettyimages", "alamy", "istockphoto",
    "dreamstime", "depositphotos", "123rf", ".svg",
]

# ============================================================
# LAYER 2 — Alt-text filter
# ============================================================

ALT_STRONG = [
    # EN
    "flooded street", "flooded road", "flood", "flooding",
    "sewage overflow", "overflowing sewer", "blocked drain",
    "waterlogged", "open sewer", "stagnant water", "drainage",
    "waterlogged road", "sewer", "drain overflow", "gutter",
    # FR
    "rue inondée", "inondation", "égout débordant", "caniveau bouché",
    "eaux usées", "eau stagnante", "drainage", "fossé plein",
    # PT
    "rua alagada", "inundação", "esgoto transbordando",
    "bueiro entupido", "água estagnada",
    # SW
    "mafuriko", "maji machafu",
]

ALT_ARTICLE_ONLY = [
    "budget", "billion", "parliament", "government", "minister",
    "announce", "speech", "statement", "report", "policy",
    "drainage masterplan", "flood management scheme",
]

# ============================================================
# THRESHOLDS & VOLUME
# ============================================================

KEEP_MARGIN = 1.05
MIN_KEEP_SCORE = 0.18

# ~160 unique queries × 80 per engine × 2 engines ≈ 25,600 attempts → ~10,000 raw
PER_QUERY = 80
MIN_W, MIN_H = 400, 300

SKIP_SCRAPE = True   # set True if _raw/ is already populated

# ============================================================
# OUTPUT DIRS — stored inside the project under data/raw/
# ============================================================

OUT_DIR    = Path(__file__).parent / "data" / "raw" / "bad_drainage_scraped"
RAW_DIR    = OUT_DIR / "_raw"
DEDUP_DIR  = OUT_DIR / "_dedup"
KEEP_DIR   = OUT_DIR / "keep"
REJECT_DIR = OUT_DIR / "reject"


# ============================================================
# LAYER 1 & 2 HELPERS
# ============================================================

def url_looks_bad(img_url: str) -> bool:
    u = img_url.lower()
    if any(bad in u for bad in BAD_URL_SUBSTR):
        return True
    fname = u.split("/")[-1].split("?")[0]
    if any(tok in fname for tok in SUSPICIOUS_IMG_TOKENS):
        return True
    return False


def alt_text_ok(result: dict) -> bool:
    alt = (result.get("title", "") + " " + result.get("source", "")).lower()
    if not alt.strip():
        return True
    has_strong = any(s in alt for s in ALT_STRONG)
    looks_like_article = any(w in alt for w in ALT_ARTICLE_ONLY)
    if looks_like_article and not has_strong:
        return False
    return True


# ============================================================
# SCRAPING
# ============================================================

def download(url: str, out_dir: Path) -> "Path | None":
    try:
        r = requests.get(
            url, timeout=10,
            headers={"User-Agent": "Mozilla/5.0 (compatible; DatasetBot/1.0)"},
        )
        if r.status_code != 200 or len(r.content) < 10_000:
            return None
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
    kept = 0
    try:
        with DDGS() as ddgs:
            for r in ddgs.images(query, max_results=n, safesearch="off"):
                url = r.get("image", "")
                if not url:
                    continue
                if url_looks_bad(url):
                    continue
                if not alt_text_ok(r):
                    continue
                # Cast to int — DDG can return width/height as strings
                if int(r.get("width") or 0) < MIN_W or int(r.get("height") or 0) < MIN_H:
                    continue
                if download(url, out_dir):
                    kept += 1
    except Exception as e:
        print(f"    DDG error: {e}")
    return kept


def scrape_icrawler(query: str, n: int, out_dir: Path) -> None:
    # Google Images has blocked icrawler since mid-2023 — Bing only.
    try:
        c = BingImageCrawler(
            storage={"root_dir": str(out_dir)},
            downloader_threads=4,
            feeder_threads=1,
            parser_threads=1,
            log_level=40,
        )
        c.crawl(
            keyword=query,
            max_num=n,
            min_size=(MIN_W, MIN_H),
            filters={"type": "photo"},
            file_idx_offset="auto",
        )
    except Exception as e:
        print(f"    bing error: {e}")


def prune_by_filename(dedup_dir: Path) -> int:
    removed = 0
    for p in dedup_dir.iterdir():
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
            pass
    return len(seen)


# ============================================================
# LAYER 3 — CLIP
# ============================================================

def clip_filter(dedup_dir: Path, keep_dir: Path, reject_dir: Path) -> tuple[int, int]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  using device: {device}")
    model, preprocess = clip.load("ViT-B/32", device=device)

    def encode_prompts(prompts):
        with torch.no_grad():
            tok = clip.tokenize(prompts).to(device)
            feat = model.encode_text(tok)
            feat /= feat.norm(dim=-1, keepdim=True)
        return feat

    keep_feat   = encode_prompts(KEEP_PROMPTS)
    rej_feat    = encode_prompts(REJECT_PROMPTS)
    afr_feat    = encode_prompts(AFRICA_PROMPTS)
    nonafr_feat = encode_prompts(NON_AFRICA_PROMPTS)

    kept = rejected = 0
    files = list(dedup_dir.iterdir())
    for p in tqdm(files, desc="  CLIP scoring"):
        try:
            img = Image.open(p).convert("RGB")
            t = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                f = model.encode_image(t)
                f /= f.norm(dim=-1, keepdim=True)
                best_keep   = (f @ keep_feat.T).max().item()
                best_rej    = (f @ rej_feat.T).max().item()
                best_afr    = (f @ afr_feat.T).max().item()
                best_nonafr = (f @ nonafr_feat.T).max().item()
                africa_score = best_afr - best_nonafr

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
    print(f"    Total queries : {len(QUERIES)}")
    print(f"    PER_QUERY     : {PER_QUERY} × ~2 engines = ~{len(QUERIES) * PER_QUERY * 2:,} attempts\n")

    if SKIP_SCRAPE:
        print("[1/4] Skipping web scrape (SKIP_SCRAPE=True). Using existing _raw/.\n")
    else:
        print(f"[1/4] Scraping {len(QUERIES)} queries × ~2 engines...")
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
    for d in (KEEP_DIR, REJECT_DIR):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
    kept, rejected = clip_filter(DEDUP_DIR, KEEP_DIR, REJECT_DIR)

    print("\n[4/4] Done.")
    total = kept + rejected
    precision = kept / total if total else 0
    print(f"  kept:      {kept:5d}  →  {KEEP_DIR}")
    print(f"  rejected:  {rejected:5d}  →  {REJECT_DIR}")
    print(f"  precision: {precision:.1%}")
    print("\nFilename legend:  a_/m_/z_ = Africa-high / mid / low relevance")
    print("                  afr±N.NNN = africa_score (positive = more African)")
    print("                  kN.NN     = CLIP keep-score")
    print("                  rN.NN     = CLIP reject-score")
    print("\nNext steps:")
    print(f"  1. Review {KEEP_DIR}")
    print(f"     'a_' prefixed = highest African-context relevance.")
    print(f"  2. Spot-check {REJECT_DIR} — real drainage images missed? lower MIN_KEEP_SCORE.")
    print(f"  3. Move verified keeps → data/processed/train/water_sewage/")


if __name__ == "__main__":
    main()
