"""
Padel OCR en service local (API HTTP), pour les applications qui l'appellent
(ex. Consularis ouvert depuis le disque).

    .venv/bin/python api.py            -> http://127.0.0.1:8765

  GET  /sante      : le service répond-il ?
  POST /analyser   : un fichier (PDF ou image, champ « fichier »)
                     -> lignes du tableau (structurées, avec alertes),
                        lignes de texte (pour lire l'en-tête : date, n°…),
                        texte brut, rapport de lecture.

Le service n'écoute que sur 127.0.0.1 : rien ne sort de l'ordinateur.
Un seul document est lu à la fois (l'OCR prend beaucoup de mémoire).
"""
from __future__ import annotations

import json
import math
import re
import tempfile
import threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from extraction import _rangees, extraire
from ocr_engine import ErreurMemoire, IMAGE_EXT, PDF_EXT, analyser_isole, texte_complet

HOTE, PORT = "127.0.0.1", 8765
VERSION = "1.0"
TAILLE_MAX = 30 * 1024 * 1024        # 30 Mo
MAX_PAGES = 10

app = FastAPI(title="Padel OCR", version=VERSION)
# Origines admises : une page ouverte depuis le disque (origine « null ») ou
# une page servie en local. Les sites internet ne peuvent pas appeler le service.
app.add_middleware(CORSMiddleware,
                   allow_origin_regex=r"^(null|https?://(localhost|127\.0\.0\.1)(:\d+)?)$",
                   allow_methods=["GET", "POST"], allow_headers=["*"],
                   allow_private_network=True)      # Chrome : accès au serveur local


_verrou = threading.Lock()


def _json(obj) -> Response:
    def conv(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    def propre(o):                       # NaN / infini -> null (JSON valide)
        if isinstance(o, float) and not math.isfinite(o):
            return None
        if isinstance(o, dict):
            return {k: propre(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [propre(v) for v in o]
        return o

    return Response(json.dumps(propre(obj), default=conv, ensure_ascii=False),
                    media_type="application/json")


@app.get("/sante")
def sante():
    return {"service": "padel-ocr", "version": VERSION, "ok": True}


@app.post("/analyser")
def analyser(fichier: UploadFile = File(...)):
    nom = Path(fichier.filename or "document").name
    ext = Path(nom).suffix.lower()
    if ext not in IMAGE_EXT | PDF_EXT:
        raise HTTPException(415, f"Format non pris en charge ({ext or 'sans extension'}) : PDF ou image.")
    contenu = fichier.file.read(TAILLE_MAX + 1)
    if len(contenu) > TAILLE_MAX:
        raise HTTPException(413, "Fichier trop gros (30 Mo au plus).")
    if not _verrou.acquire(timeout=300):
        raise HTTPException(503, "Un autre document est en cours de lecture, réessayez dans un instant.")
    try:
        with tempfile.TemporaryDirectory(prefix="padel_api_") as dossier:
            chemin = Path(dossier) / ("document" + ext)
            chemin.write_bytes(contenu)
            try:
                pages, images = analyser_isole(chemin, mode="rapide", max_pages=MAX_PAGES)
            except ErreurMemoire as e:
                raise HTTPException(507, str(e))
            except ValueError as e:
                raise HTTPException(422, str(e))
    finally:
        _verrou.release()

    candidats, rapport = extraire(pages, images)
    lignes_texte = []
    for p in pages:
        for rangee in _rangees(p.lignes):        # rangées serrées : une ligne du document
            cells = [re.sub(r"\s+", " ", l.texte).strip() for l in sorted(rangee, key=lambda l: l.boite[0])]
            cells = [c for c in cells if c]
            if cells:
                lignes_texte.append({"page": p.numero, "cells": cells, "text": " ".join(cells)})
    a_verifier = sum(1 for c in candidats if c["alertes_liste"])
    return _json({
        "fichier": nom,
        "pages": [{"numero": p.numero, "largeur": p.largeur, "hauteur": p.hauteur, "source": p.source}
                  for p in pages],
        "candidats": [{k: v for k, v in c.items() if k not in ("alertes",)} for c in candidats],
        "lignes_texte": lignes_texte,
        "texte": texte_complet(pages),
        "rapport": {**rapport, "lignes": len(candidats), "a_verifier": a_verifier,
                    "ocr": any(p.source == "ocr" for p in pages)},
    })


if __name__ == "__main__":
    import uvicorn
    print(f"Padel OCR — service local sur http://{HOTE}:{PORT} (Ctrl+C pour arrêter)")
    uvicorn.run(app, host=HOTE, port=PORT, log_level="warning")
