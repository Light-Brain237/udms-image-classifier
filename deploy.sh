#!/usr/bin/env bash
set -euo pipefail
BRANCH="${BRANCH:-main}"
cd "$(dirname "$0")"

git fetch origin "${BRANCH}"
git reset --hard "origin/${BRANCH}"

source .venv/bin/activate
pip install -r requirements.txt

pm2 reload udms-classifier --update-env