#!/usr/bin/env bash
# Padel OCR - lancement macOS / Linux (crée l'environnement au premier lancement)
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "Première installation, quelques minutes..."
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
else
  source .venv/bin/activate
fi
python app.py
