"""
Padel OCR — version ligne de commande (traitement par lot).

Exemples :
  python cli.py facture.pdf
  python cli.py scans/*.jpg --lang de --mode precis --out resultats
  python cli.py dossier_scans/ --annoter
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ocr_engine import IMAGE_EXT, LANGUES, MODES, PDF_EXT, MoteurOCR, dessiner, texte_complet, vers_dict


def lister_entrees(chemins: list[str]) -> list[Path]:
    fichiers = []
    for c in map(Path, chemins):
        if c.is_dir():
            fichiers += sorted(p for p in c.iterdir() if p.suffix.lower() in IMAGE_EXT | PDF_EXT)
        elif c.exists():
            fichiers.append(c)
        else:
            print(f"[!] Introuvable : {c}", file=sys.stderr)
    return fichiers


def main():
    ap = argparse.ArgumentParser(description="OCR d'images et de PDF avec PaddleOCR")
    ap.add_argument("entrees", nargs="+", help="fichiers ou dossiers")
    ap.add_argument("--lang", default="fr", choices=list(LANGUES), help="langue (défaut : fr)")
    ap.add_argument("--mode", default="rapide", choices=list(MODES))
    ap.add_argument("--orientation", action="store_true", help="corriger les pages tournées")
    ap.add_argument("--seuil", type=float, default=0.5, help="confiance minimale (0-1)")
    ap.add_argument("--dpi", type=int, default=200, help="résolution de rendu des PDF")
    ap.add_argument("--filigrane", type=int, nargs="?", const=160, default=None,
                    help="supprimer les filigranes gris (seuil optionnel 80-240, défaut 160)")
    ap.add_argument("--out", default="resultats_ocr", help="dossier de sortie")
    ap.add_argument("--annoter", action="store_true", help="sauver aussi les images annotées")
    args = ap.parse_args()

    fichiers = lister_entrees(args.entrees)
    if not fichiers:
        sys.exit("Aucun fichier à traiter.")

    sortie = Path(args.out)
    sortie.mkdir(parents=True, exist_ok=True)
    moteur = MoteurOCR()

    # si deux fichiers ont le même nom de base (scan.png + scan.pdf), on garde l'extension
    stems = [f.stem for f in fichiers]

    for f in fichiers:
        base = f.stem if stems.count(f.stem) == 1 else f"{f.stem}_{f.suffix[1:].lower()}"
        t0 = time.time()
        print(f"→ {f.name} … ", end="", flush=True)
        try:
            pages, images = moteur.analyser_fichier(
                f, langue=args.lang, mode=args.mode, orientation=args.orientation,
                seuil=args.seuil, dpi=args.dpi, filigrane=args.filigrane,
            )
        except Exception as e:  # on continue le lot même si un fichier échoue
            print(f"ERREUR : {e}")
            continue

        (sortie / f"{base}.txt").write_text(texte_complet(pages), encoding="utf-8")
        (sortie / f"{base}.json").write_text(
            json.dumps(vers_dict(f.name, pages, args.lang), ensure_ascii=False, indent=2),
            encoding="utf-8")
        if args.annoter:
            for p, img in zip(pages, images):
                suffixe = f"_p{p.numero}" if len(pages) > 1 else ""
                dessiner(img, p).save(sortie / f"{base}{suffixe}_annote.jpg", quality=90)

        nb = sum(len(p.lignes) for p in pages)
        print(f"{len(pages)} page(s), {nb} lignes, {time.time() - t0:.1f} s")

    print(f"\nRésultats dans : {sortie.resolve()}")


if __name__ == "__main__":
    main()
