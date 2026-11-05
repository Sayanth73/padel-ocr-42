"""
Mesure la justesse de l'extraction sur les documents de référence.

    python factures_test/evaluer.py

Pour chaque <doc>.pdf ayant un <doc>.verite.json (résultat attendu, vérifié à
la main), lance OCR + extraction et compare champ par champ. À relancer après
toute modification du moteur ou de l'extraction.
Pour ajouter un document : déposer le PDF ici, lancer avec --generer <doc>.pdf
pour créer un brouillon de .verite.json, puis le CORRIGER à la main.
"""
import json
import sys
from pathlib import Path

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI.parent))

from extraction import extraire          # noqa: E402
from ocr_engine import analyser_isole    # noqa: E402

CHAMPS_DEFAUT = ["nom", "ean", "ancien_ean", "prix_negocie", "tva_pct", "arret"]


def analyser(pdf: Path, options: dict | None = None):
    # options propres au document (ex. {"texte_integre": false} : forcer l'OCR
    # sur un PDF numérique, pour tester la lecture d'un scan)
    pages, images = analyser_isole(pdf, **{"mode": "rapide", "filigrane": 160, **(options or {})})
    return extraire(pages, images)


def egal(a, b):
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and abs(a - b) < 0.005
    return (a or "") == (b or "")


def evaluer(pdf: Path, verite: dict) -> bool:
    candidats, rapport = analyser(pdf, verite.get("options"))
    CHAMPS = verite.get("champs", CHAMPS_DEFAUT)
    attendu = verite["lignes"]
    # appariement : par EAN si possible, sinon par (page, nom)
    restants = list(candidats)
    erreurs, manquantes = [], []
    par_champ = {c: 0 for c in CHAMPS}
    for v in attendu:
        trouve = next((c for c in restants if v["ean"] and c["ean"] == v["ean"]
                       and c["page"] == v["page"] and egal(c["prix_negocie"], v.get("prix_negocie", c["prix_negocie"]))), None) \
            or next((c for c in restants if v["ean"] and c["ean"] == v["ean"] and c["page"] == v["page"]), None) \
            or next((c for c in restants if c["page"] == v["page"] and c["nom"] == v["nom"]), None)
        if not trouve:
            manquantes.append(v)
            continue
        restants.remove(trouve)
        for ch in CHAMPS:
            if egal(trouve[ch], v[ch]):
                par_champ[ch] += 1
            else:
                erreurs.append((v["nom"], ch, v[ch], trouve[ch]))
    n = len(attendu)
    print(f"\n=== {pdf.name} — méthode : {', '.join(rapport['methode'])}")
    print(f"lignes attendues {n} · extraites {len(candidats)} · manquantes {len(manquantes)} "
          f"· en trop {len(restants)}")
    for ch in CHAMPS:
        print(f"  {ch:13} {par_champ[ch]:4}/{n}  ({par_champ[ch] / n:.1%})")
    lignes_parfaites = n - len(manquantes) - len({e[0] for e in erreurs})
    print(f"  lignes 100 % justes : {lignes_parfaites}/{n}")
    for nom, ch, att, obt in erreurs:
        print(f"  ✗ {nom[:40]:40} {ch}: attendu {att!r}, obtenu {obt!r}")
    for v in manquantes:
        print(f"  ✗ MANQUANTE : {v['nom']} ({v['ean']})")
    for c in restants:
        print(f"  ✗ EN TROP : {c['nom']} ({c['ean']})")
    # le critère qui compte : aucun prix ni EAN faux
    critiques = [e for e in erreurs if e[1] in ("ean", "prix_negocie", "prix_net", "prix_tarif", "qte", "montant", "remise_pct")]
    return not critiques and not manquantes and not restants


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--generer":
        pdf = ICI / sys.argv[2]
        candidats, _ = analyser(pdf)
        brouillon = {"fichier": pdf.name, "verifie": "NON — à corriger à la main",
                     "lignes": [{k: c[k] for k in ["page"] + CHAMPS_DEFAUT} for c in candidats]}
        sortie = pdf.with_suffix(".verite.json")
        sortie.write_text(json.dumps(brouillon, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Brouillon écrit : {sortie} — vérifie chaque ligne contre le document !")
        return
    ok = True
    for v in sorted(ICI.glob("*.verite.json")):
        verite = json.loads(v.read_text(encoding="utf-8"))
        ok &= evaluer(ICI / verite["fichier"], verite)
    print("\nRÉSULTAT :", "OK — aucun prix ni EAN faux" if ok else "ÉCHEC — voir les ✗ ci-dessus")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
