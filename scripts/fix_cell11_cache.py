"""Fix cell 11 — wipe tf_cache dir before recreating it to avoid lockfile errors."""
import json

with open('notebooks/colab_train.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
cells = nb['cells']

cells[11]['source'] = [
    "import os, shutil\n",
    "\n",
    "# Always wipe the cache dir so stale lockfiles never block a new run.\n",
    "CACHE_DIR = '/content/tf_cache'\n",
    "if os.path.exists(CACHE_DIR):\n",
    "    shutil.rmtree(CACHE_DIR)\n",
    "os.makedirs(CACHE_DIR, exist_ok=True)\n",
    "print(f'tf.data cache dir cleared and ready: {CACHE_DIR}')\n",
    "\n",
    "# Scan and remove corrupt/unreadable image files before training.\n",
    "from PIL import Image as _PIL_Image\n",
    "import pathlib as _pl\n",
    "\n",
    "_bad_files = []\n",
    "for _p in _pl.Path(DATA_DIR).rglob('*'):\n",
    "    if not _p.is_file():\n",
    "        continue\n",
    "    try:\n",
    "        with _PIL_Image.open(_p) as _im:\n",
    "            _im.verify()\n",
    "    except Exception:\n",
    "        _bad_files.append(_p)\n",
    "\n",
    "if _bad_files:\n",
    "    print(f'Removing {len(_bad_files)} corrupt/unreadable files...')\n",
    "    for _p in _bad_files:\n",
    "        _p.unlink()\n",
    "        print(f'  removed: {_p.name}')\n",
    "else:\n",
    "    print('No corrupt files found — dataset is clean.')\n",
]

with open('notebooks/colab_train.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Cell 11 fixed.')
