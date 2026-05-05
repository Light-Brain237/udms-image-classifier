"""Fix cell 4 — remove PAT placeholder, use public URL."""
import json

with open('notebooks/colab_train.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
cells = nb['cells']

cells[4]['source'] = [
    "import os, subprocess, sys\n",
    "\n",
    "GITHUB_USERNAME = 'Light-Brain237'\n",
    "GITHUB_REPO     = 'udms-image-classifier'\n",
    "BRANCH          = 'main'\n",
    "\n",
    "REPO_PATH = f'/content/{GITHUB_REPO}'\n",
    "REPO_URL  = f'https://github.com/{GITHUB_USERNAME}/{GITHUB_REPO}.git'\n",
    "\n",
    "if os.path.isdir(os.path.join(REPO_PATH, '.git')):\n",
    "    print(f'Pulling latest changes into {REPO_PATH} ...')\n",
    "    result = subprocess.run(\n",
    "        ['git', '-C', REPO_PATH, 'pull', 'origin', BRANCH],\n",
    "        capture_output=True, text=True,\n",
    "    )\n",
    "    print(result.stdout or result.stderr)\n",
    "else:\n",
    "    print(f'Cloning {GITHUB_USERNAME}/{GITHUB_REPO} ...')\n",
    "    subprocess.run(['git', 'clone', '--branch', BRANCH, REPO_URL, REPO_PATH], check=True)\n",
    "    print('Repository cloned.')\n",
    "\n",
    "if REPO_PATH not in sys.path:\n",
    "    sys.path.insert(0, REPO_PATH)\n",
    "os.chdir(REPO_PATH)\n",
    "print(f'Working directory: {os.getcwd()}')\n",
    "\n",
    "result = subprocess.run(\n",
    "    ['git', '-C', REPO_PATH, 'log', '--oneline', '-3'],\n",
    "    capture_output=True, text=True,\n",
    ")\n",
    "print(f'Recent commits:\\n{result.stdout}')\n",
]

with open('notebooks/colab_train.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Cell 4 fixed.')
