#!/bin/bash
# Double-cliquer pour lancer Padel OCR en service local (utilisé par Consularis).
cd "$(dirname "$0")"
exec .venv/bin/python api.py
