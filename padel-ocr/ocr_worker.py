"""
Processus d'OCR isolé, lancé par ocr_engine.analyser_isole() pour un document.

    python ocr_worker.py <fichier> <sortie.pkl> '<options JSON>'

Écrit (pages, images) dans sortie.pkl puis se termine (mémoire libérée).
Communique avec le parent par stdout : "@@PROG i n" (progression) et
"@@ERREUR message" (erreur prévisible, ex. format non pris en charge).
"""
import json
import pickle
import sys

from ocr_engine import MoteurOCR


def main():
    chemin, sortie, options = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])

    def progression(i, n):
        print(f"@@PROG {i} {n}", flush=True)

    try:
        resultat = MoteurOCR().analyser_fichier(chemin, progression=progression, **options)
    except ValueError as e:
        print(f"@@ERREUR {e}", flush=True)
        sys.exit(2)
    with open(sortie, "wb") as f:
        pickle.dump(resultat, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
