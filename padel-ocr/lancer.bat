@echo off
REM Padel OCR - lancement Windows (crée l'environnement au premier lancement)
cd /d "%~dp0"
if not exist .venv (
    echo Premiere installation, quelques minutes...
    py -3 -m venv .venv || python -m venv .venv
    call .venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)
python app.py
pause
