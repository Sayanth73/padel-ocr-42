"""
Transforme les lignes OCR d'un tarif / d'une facture en lignes « produit + prix »
candidates, avec des CONTRÔLES sur chaque ligne. Le résultat est toujours
présenté à l'utilisateur pour validation avant enregistrement.

Deux méthodes :
  1. Lecture par colonnes (prioritaire) : on repère la ligne d'en-tête du
     tableau (DESIGNATION, PRIX, EAN, TVA…), les traits verticaux du tableau
     dans l'image, puis chaque morceau de texte est rangé dans sa colonne
     selon sa position. Le prix est ce qui est dans la colonne Prix : on ne
     devine pas.
  2. Secours (aucun en-tête reconnu) : heuristique ligne par ligne, toutes les
     lignes sont signalées « format non reconnu ».
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

import numpy as np

from ocr_engine import Ligne, Page, trier_lecture

RE_EAN = re.compile(r"(?<!\d)(\d{13})(?!\d)")
RE_PRIX = re.compile(r"(?<![\d.,])(\d{1,3}(?:['’ ]\d{3})*[.,]\d{2})(?![\d%])")
RE_PCT = re.compile(r"(\d{1,2}(?:[.,]\d{1,2})?)\s?%")
RE_DATE = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})\b")
RE_ARRET = re.compile(r"arr[eê]t\w*\s+de\s+commerciali|arr[eê]t[ée]?\b|supprim|discontinu", re.I)

SEUIL_CONFIANCE = 0.90       # en dessous : « lecture incertaine »
PRIX_MAX = 5000.0            # au-dessus : « prix aberrant »

# Champ -> motif reconnu dans le libellé de colonne. L'ordre compte :
# « Ancien code EAN » avant « Code EAN », « Prix public » avant « Prix ».
COLONNES_CONNUES = [
    ("remise", r"REMISE|RABAIS|DISCOUNT"),
    ("ancien_ean", r"ANCIEN"),
    ("montant", r"MONTANT|\bTOTAL"),
    ("condition", r"CONDITION|PALIER"),
    ("prix_public", r"PUBLIC|CONSEILL|\bPPC\b|\bTTC\b"),
    ("prix_net", r"\bNET\b"),                                     # « P.U. net HT », « Prix net HT »
    ("prix_tarif", r"TARIF|BRUT|P\.\s?U\b|\bPU\b|UNITAIRE"),        # « P.U. HT », « Prix tarif HT »
    ("prix", r"PRIX"),                                              # prix unique (« Prix pharmacien HT »)
    ("ean", r"EAN|GTIN|\bCIP\b|CODE"),
    ("tva", r"TVA|MWST|\bVAT\b"),
    ("groupage", r"GROUPAGE|COLISAGE|\bUVC\b"),
    ("qte", r"\bQT[EÉ]|QUANTIT|\bQTY\b"),
    ("designation", r"D[EÉ]SIGNATION|LIBELL|ARTICLE|PRODUIT|DESCRIPTION|D[EÉ]NOMINATION"),
    ("info", r"\bINFO|REMARQ|OBSERV|COMMENT|STATUT"),
]
CHAMPS_PRIX = ("prix", "prix_net", "prix_tarif")
RE_FIN_TABLEAU = re.compile(r"\s*(TOTAL|SOUS[- ]TOTAL|TAUX\b|BASE HT|NET [AÀ] PAYER|MONTANT TOTAL)", re.I)
RE_PALIER = re.compile(r"(\d+(?:[.,]\d+)?)\s*%\s*(?:d[èe]s|à partir de|a partir de|pour)\s*(\d+)", re.I)

COLONNES_SORTIE = ["garder", "alertes", "nom", "ean", "prix_negocie", "prix_public", "remise_pct",
                   "tva_pct", "ancien_ean", "groupage", "arret", "categorie", "confiance",
                   "page", "ligne_ocr"]


# --------------------------------------------------------------------------- #
# Utilitaires
# --------------------------------------------------------------------------- #
def ean_valide(code: str) -> bool:
    """Clé de contrôle EAN-13 / GTIN : un chiffre mal lu rend presque toujours le code invalide."""
    if not re.fullmatch(r"\d{13}", code or ""):
        return False
    d = [int(c) for c in code]
    return (10 - sum(x * (3 if i % 2 else 1) for i, x in enumerate(d[:12])) % 10) % 10 == d[12]


def suggerer_ean(code: str, autres: list[str]) -> tuple[str, str] | None:
    """Pour un EAN à clé invalide : essaie chaque correction d'UN chiffre qui
    donne un code valide, et garde celle qui ressemble le plus aux autres codes
    du document (même préfixe fabricant, codes voisins). Retourne
    (code proposé, explication) ou None si aucune proposition nette."""
    if not re.fullmatch(r"\d{13}", code or "") or ean_valide(code):
        return None
    autres = [a for a in autres if a and a != code and ean_valide(a)]
    if not autres:
        return None

    def prefixe(a, b):
        n = 0
        while n < 13 and a[n] == b[n]:
            n += 1
        return n

    candidats = []
    for i in range(13):
        for d in "0123456789":
            if d == code[i]:
                continue
            c = code[:i] + d + code[i + 1:]
            if ean_valide(c):
                proche = max(autres, key=lambda a: prefixe(c, a))
                candidats.append((prefixe(c, proche), c, i, proche))
    candidats.sort(reverse=True)
    if not candidats or candidats[0][0] < 7:
        return None
    if len(candidats) > 1 and candidats[1][0] == candidats[0][0]:
        return None                                   # deux propositions aussi bonnes : on ne choisit pas
    score, c, i, proche = candidats[0]
    return c, (f"1 seul chiffre changé ({i + 1}e position : {code[i]} → {c[i]}), "
               f"le code devient valide et ressemble aux autres codes du document (ex. {proche})")


def _prix(s: str) -> float:
    return float(re.sub(r"['’ ]", "", s).replace(",", "."))


def _pct(s: str) -> float:
    return float(s.replace(",", "."))


def detecter_date(texte: str) -> str | None:
    """Première date trouvée dans le document -> AAAA-MM-JJ (date du deal)."""
    for j, m, a in RE_DATE.findall(texte):
        j, m, a = int(j), int(m), int(a)
        if a < 100:
            a += 2000
        if 1 <= j <= 31 and 1 <= m <= 12 and 2000 <= a <= 2100:
            return f"{a:04d}-{m:02d}-{j:02d}"
    return None


def _hauteur(l: Ligne) -> int:
    return l.boite[3] - l.boite[1]


def _yc(l: Ligne) -> float:
    return (l.boite[1] + l.boite[3]) / 2


def _nettoyer(lignes: list[Ligne]) -> list[Ligne]:
    """Retire les débris de filigrane : petits fragments peu sûrs, ou petits
    fragments anormalement hauts (texte en diagonale)."""
    if not lignes:
        return []
    h_med = statistics.median(_hauteur(l) for l in lignes)
    propres = []
    for l in lignes:
        court = len(l.texte.replace(" ", "")) <= 5
        if court and (l.confiance < 0.8 or _hauteur(l) > 1.5 * h_med):
            continue
        propres.append(l)
    return propres


# --------------------------------------------------------------------------- #
# Structure du tableau
# --------------------------------------------------------------------------- #
@dataclass
class Colonne:
    champ: str
    libelle: str
    x1: float
    x2: float


@dataclass
class Gabarit:
    colonnes: list[Colonne]
    y_debut: float              # fin de l'en-tête (en pixels)
    largeur_page: int
    lignes_entete: list[Ligne] = field(default_factory=list)

    def a_l_echelle(self, largeur: int) -> "Gabarit":
        k = largeur / self.largeur_page
        return Gabarit([Colonne(c.champ, c.libelle, c.x1 * k, c.x2 * k) for c in self.colonnes],
                       0, largeur)


def _champ_de(libelle: str) -> str | None:
    lib = libelle.upper()
    for champ, motif in COLONNES_CONNUES:
        if re.search(motif, lib):
            return champ
    return None


def _decouper_libelle(l: Ligne) -> list[Ligne]:
    """Une boîte d'en-tête peut contenir plusieurs intitulés collés par le PDF
    ou l'OCR : on la coupe à chaque mot qui commence un intitulé d'un autre
    champ, la position de chaque morceau étant estimée dans la boîte."""
    mots = list(re.finditer(r"\S+", l.texte))
    if len(mots) < 2:
        return [l]
    segments: list[list] = []
    for m in mots:
        champ_mot = _champ_de(m.group())
        if segments:
            champ_seg = _champ_de(" ".join(x.group() for x in segments[-1]))
            prix = set(CHAMPS_PRIX) | {"prix_public"}
            if champ_mot and champ_seg and champ_mot != champ_seg and not (
                    champ_seg in prix and champ_mot in prix) and not (
                    # « ANCIEN CODE EAN » est un seul intitulé
                    _champ_de(" ".join(x.group() for x in segments[-1]) + " " + m.group()) == champ_seg):
                segments.append([m])
                continue
            segments[-1].append(m)
        else:
            segments.append([m])
    if len(segments) == 1:
        return [l]
    x1, x2, n = l.boite[0], l.boite[2], max(1, len(l.texte))
    morceaux = []
    for seg in segments:
        debut, fin = seg[0].start(), seg[-1].end()
        morceaux.append(Ligne(texte=l.texte[debut:fin], confiance=l.confiance,
                              boite=[int(x1 + debut / n * (x2 - x1)), l.boite[1],
                                     int(x1 + fin / n * (x2 - x1)), l.boite[3]]))
    return morceaux


def _traits_verticaux(image: np.ndarray, y1: int, y2: int) -> list[int]:
    """Abscisses des traits verticaux du tableau entre y1 et y2 (colonnes de
    pixels sombres sur plus de 60 % de la hauteur)."""
    if image is None or y2 - y1 < 50:
        return []
    zone = image[y1:y2]
    gris = zone.mean(axis=2) if zone.ndim == 3 else zone
    fraction = (gris < 140).mean(axis=0)
    xs = np.where(fraction > 0.6)[0]
    traits, groupe = [], []
    for x in xs:
        if groupe and x - groupe[-1] > 3:
            traits.append(int(np.mean(groupe)))
            groupe = []
        groupe.append(x)
    if groupe:
        traits.append(int(np.mean(groupe)))
    return traits


def detecter_gabarit(page: Page, image: np.ndarray | None = None) -> Gabarit | None:
    lignes = _nettoyer(page.lignes)
    if not lignes:
        return None
    h_med = statistics.median(_hauteur(l) for l in lignes)
    # boîtes qui ressemblent à un libellé de colonne (texte court + mot-clé)
    morceaux = [m for l in lignes if len(l.texte) <= 60 for m in _decouper_libelle(l)]
    cles = [l for l in morceaux if len(l.texte) <= 40 and _champ_de(l.texte)]
    if not cles:
        return None
    # la rangée d'en-tête = la bande verticale qui contient le plus de libellés distincts
    meilleur = None
    for ref in cles:
        bande = [l for l in cles if abs(_yc(l) - _yc(ref)) <= 2.0 * h_med]
        champs = {_champ_de(l.texte) for l in bande}
        if len(champs) >= 3 and champs & set(CHAMPS_PRIX) and (
                meilleur is None or len(champs) > len(meilleur[1])):
            meilleur = (bande, champs)
    if not meilleur:
        return None
    bande = meilleur[0]
    y_min = min(_yc(l) for l in bande) - 0.6 * h_med
    y_max = max(_yc(l) for l in bande) + 0.6 * h_med
    # toutes les boîtes de la zone d'en-tête, regroupées par colonne (recouvrement en x)
    entete = sorted([m for l in lignes if y_min <= _yc(l) <= y_max for m in _decouper_libelle(l)],
                    key=lambda l: l.boite[0])
    groupes: list[list[Ligne]] = []
    for l in entete:
        for g in groupes:
            gx1, gx2 = min(b.boite[0] for b in g), max(b.boite[2] for b in g)
            # recouvrement franc (les morceaux d'un intitulé collé sont estimés)
            recouvre = min(l.boite[2], gx2) - max(l.boite[0], gx1)
            if recouvre > 0.3 * min(l.boite[2] - l.boite[0], gx2 - gx1):
                g.append(l)
                break
        else:
            groupes.append([l])
    libelles = []
    for g in groupes:
        texte = " ".join(b.texte for b in sorted(g, key=_yc))
        champ = _champ_de(texte)
        if champ:
            libelles.append((champ, texte, min(b.boite[0] for b in g), max(b.boite[2] for b in g)))
    libelles.sort(key=lambda c: c[2])

    # limites des colonnes : traits verticaux du tableau s'il y en a, sinon milieux
    traits = _traits_verticaux(image, int(y_max), int(y_max + 40 * h_med)) if image is not None else []
    colonnes = []
    for i, (champ, texte, x1, x2) in enumerate(libelles):
        centre = (x1 + x2) / 2
        if len(traits) >= 3:
            gauche = max([t for t in traits if t < centre], default=0)
            droite = min([t for t in traits if t > centre], default=page.largeur)
        else:
            gauche = (libelles[i - 1][3] + x1) / 2 if i else 0
            droite = (x2 + libelles[i + 1][2]) / 2 if i + 1 < len(libelles) else page.largeur
            if champ in ("designation", "info") and i + 1 < len(libelles):
                # texte aligné à gauche : il peut s'étendre jusqu'à l'intitulé suivant
                droite = max(droite, libelles[i + 1][2] - 0.5 * h_med)
        colonnes.append(Colonne(champ, texte, gauche, droite))
    # la 1re colonne (désignation, alignée à gauche) commence au bord du tableau
    if colonnes:
        colonnes[0].x1 = min(colonnes[0].x1, min(l.boite[0] for l in lignes if _yc(l) > y_max) - 5)
    return Gabarit(colonnes, y_max, page.largeur, entete)


# --------------------------------------------------------------------------- #
# Lecture par colonnes
# --------------------------------------------------------------------------- #
def _repartir(l: Ligne, colonnes: list[Colonne]) -> list[tuple[str, str, float]]:
    """(champ, texte, confiance) pour une boîte OCR. Une boîte qui chevauche
    plusieurs colonnes (ex. « 10% ARRET DE COMMERCIALISATION ») est coupée en
    mots, chaque mot étant placé selon sa position estimée dans la boîte."""
    x1, x2 = l.boite[0], l.boite[2]

    def colonne_de(x):
        for c in colonnes:
            if c.x1 <= x < c.x2:
                return c.champ
        return None

    touchees = {c.champ for c in colonnes if x1 < c.x2 - 3 and x2 > c.x1 + 3}
    if len(touchees) <= 1:
        return [(colonne_de((x1 + x2) / 2), l.texte, l.confiance)]
    morceaux = []
    # mots collés par l'OCR : « 20%ARRET », « 3400…|3401… »
    texte = re.sub(r"%(?=\S)", "% ", l.texte).replace("|", " ")
    n = max(1, len(texte))
    for m in re.finditer(r"\S+", texte):
        if m.group() in {"|", "/"}:
            continue
        x = x1 + (m.start() + m.end()) / 2 / n * (x2 - x1)
        morceaux.append((colonne_de(x), m.group(), l.confiance))
    # recolle les mots consécutifs d'une même colonne
    fusion: list[list] = []
    for champ, mot, conf in morceaux:
        if fusion and fusion[-1][0] == champ:
            fusion[-1][1] += " " + mot
        else:
            fusion.append([champ, mot, conf])
    return [tuple(f) for f in fusion]


def _rangees(lignes: list[Ligne]) -> list[list[Ligne]]:
    """Regroupe les boîtes par rangée de tableau (centre vertical proche)."""
    if not lignes:
        return []
    h_med = statistics.median(_hauteur(l) for l in lignes)
    tol = 0.45 * h_med
    rangees: list[list[Ligne]] = []
    for l in sorted(lignes, key=_yc):
        if rangees and abs(_yc(l) - statistics.median(_yc(b) for b in rangees[-1])) <= tol:
            rangees[-1].append(l)
        else:
            rangees.append([l])
    return rangees


def _boite(lignes: list[Ligne]) -> list[int]:
    """Rectangle englobant d'une rangée (pour la montrer sur le document)."""
    return [min(l.boite[0] for l in lignes), min(l.boite[1] for l in lignes),
            max(l.boite[2] for l in lignes), max(l.boite[3] for l in lignes)]


def _nom_propre(texte: str) -> str:
    texte = re.sub(r"\(\d\)", " ", texte)                 # renvois de note (5)
    return " ".join(texte.split()).strip(" -:;|")


def lire_tableau(page: Page, gabarit: Gabarit) -> tuple[list[dict], dict]:
    lignes = [l for l in _nettoyer(page.lignes) if _yc(l) > gabarit.y_debut]
    stats = {"rangees": 0, "sections": 0, "sans_prix": 0, "hors_colonne": 0}
    candidats = []
    categorie = ""
    for rangee in _rangees(lignes):
        cellules: dict[str, list[tuple[str, float]]] = {}
        for l in sorted(rangee, key=lambda b: b.boite[0]):
            for champ, texte, conf in _repartir(l, gabarit.colonnes):
                cellules.setdefault(champ or "?", []).append((texte, conf))
        # débris de filigrane (« a », « ma », « 58 »…) à côté d'un vrai texte
        for champ in ("designation", "info", "?"):
            if champ in cellules and len(cellules[champ]) > 1:
                gardes = [(t, c) for t, c in cellules[champ] if len(t.replace(" ", "")) > 3]
                cellules[champ] = gardes or cellules[champ]
        if all(len(t.replace(" ", "")) <= 3 for v in cellules.values() for t, _ in v):
            continue                                   # rangée faite uniquement de débris
        if RE_FIN_TABLEAU.match(" ".join(l.texte for l in sorted(rangee, key=lambda b: b.boite[0]))):
            break                                      # totaux / récapitulatif TVA : fin du tableau
        texte_brut = " | ".join(f"{k}: {' '.join(t for t, _ in v)}" for k, v in cellules.items())
        txt = {k: " ".join(t for t, _ in v) for k, v in cellules.items()}
        conf = {k: min(c for _, c in v) for k, v in cellules.items()}

        # prix : une liste par colonne de prix présente (net, prix, tarif)
        par_col = {k: RE_PRIX.findall(txt.get(k, "")) for k in CHAMPS_PRIX}
        prix_trouves = par_col["prix_net"] or par_col["prix"] or par_col["prix_tarif"]
        eans = RE_EAN.findall(txt.get("ean", ""))
        nom = _nom_propre(txt.get("designation", ""))
        stats["rangees"] += 1
        if "?" in cellules:
            stats["hors_colonne"] += 1

        # rangée de section (titre de catégorie) : texte seul, ni prix ni code
        if not prix_trouves and not eans and not RE_EAN.findall(txt.get("ancien_ean", "")):
            if nom:
                stats["sections"] += 1
                categorie = nom
            continue

        alertes = []
        arret = bool(RE_ARRET.search(" ".join(txt.values())))
        remise = RE_PCT.findall(txt.get("remise", "")) or (
            re.findall(r"\d{1,2}(?:[.,]\d{1,2})?", txt.get("remise", "")) if "remise" in txt else [])
        remise_pct = _pct(remise[0]) if remise else None
        tarif = _prix(par_col["prix_tarif"][0]) if par_col["prix_tarif"] else None
        if tarif is None and par_col["prix_net"] and par_col["prix"]:
            tarif = _prix(par_col["prix"][0])           # « Prix » + « Net » : « Prix » est le tarif
        net = None
        if par_col["prix_net"]:
            net = _prix(par_col["prix_net"][0])
        elif par_col["prix"] and not (par_col["prix_tarif"] and remise_pct):
            net = _prix(par_col["prix"][0])
        elif tarif is not None:
            net = round(tarif * (1 - (remise_pct or 0) / 100), 4)
        if net is None:
            stats["sans_prix"] += 1
            if not arret:
                alertes.append("prix manquant")
        for k, v in par_col.items():
            if len(v) > 1:
                alertes.append(f"{len(v)} prix dans la case : lignes fusionnées ?")
                break
        if len(eans) > 1:
            alertes.append(f"{len(eans)} EAN dans la case : lignes fusionnées ?")
        ean = eans[0] if eans else ""
        if not ean:
            alertes.append("EAN manquant")
        elif not ean_valide(ean):
            alertes.append("clé EAN invalide (code mal lu ou faux dans le document)")
        anciens = RE_EAN.findall(txt.get("ancien_ean", ""))
        ancien = anciens[0] if anciens else ""
        if ancien and not ean_valide(ancien):
            alertes.append("ancien EAN : clé invalide")
        if not nom:
            alertes.append("nom manquant")
        prix = net
        if prix is not None and not (0 < prix <= PRIX_MAX):
            alertes.append("prix aberrant")
        prix_public = None
        pp = RE_PRIX.findall(txt.get("prix_public", ""))
        if pp:
            prix_public = _prix(pp[0])
            if prix is not None and prix > prix_public:
                alertes.append("prix négocié > prix public")
        # contrôles croisés : le document se vérifie lui-même
        if tarif is not None and net is not None:
            if net > tarif + 0.005:
                alertes.append("prix net > prix tarif")
            elif remise_pct is not None and par_col["prix_net"] and \
                    abs(tarif * (1 - remise_pct / 100) - net) > 0.011:
                alertes.append(f"écart : tarif × (1 − remise) = {tarif * (1 - remise_pct / 100):.2f} ≠ net {net:.2f}")
        qtes = re.findall(r"(?<![\d.,])\d+(?![\d.,])", txt.get("qte", ""))
        qte = int(qtes[0]) if qtes else None
        mts = RE_PRIX.findall(txt.get("montant", ""))
        montant = _prix(mts[-1]) if mts else None
        if qte is not None and net is not None and montant is not None and abs(qte * net - montant) > 0.05:
            alertes.append(f"écart : quantité × net = {qte * net:.2f} ≠ montant {montant:.2f}")
        mp = RE_PALIER.search(txt.get("condition", ""))
        palier = {"qte_min": int(mp.group(2)), "remise": _pct(mp.group(1))} if mp else None
        tva = RE_PCT.findall(txt.get("tva", ""))
        groupage = re.findall(r"\d+", txt.get("groupage", ""))
        champs_conf = ("designation", "ean", "qte", "remise", "montant") + CHAMPS_PRIX
        confiance = min([conf[k] for k in champs_conf if k in conf] or [0])
        if confiance < SEUIL_CONFIANCE:
            alertes.append(f"lecture incertaine ({confiance:.0%})")
        if arret:
            alertes.insert(0, "arrêt de commercialisation")

        candidats.append({
            "nom": nom, "ean": ean, "ancien_ean": ancien,
            "prix_negocie": prix, "prix_public": prix_public,
            "remise_pct": remise_pct, "qte": qte, "prix_tarif": tarif, "prix_net": net,
            "montant": montant, "palier": palier,
            "tva_pct": _pct(tva[0]) if tva else None,
            "groupage": int(groupage[0]) if groupage else None,
            "arret": arret, "categorie": categorie,
            "confiance": round(confiance, 2), "page": page.numero,
            "ligne_ocr": texte_brut, "alertes_liste": alertes, "boite": _boite(rangee),
        })
    return candidats, stats


# --------------------------------------------------------------------------- #
# Secours : heuristique ligne par ligne (format sans en-tête reconnu)
# --------------------------------------------------------------------------- #
MOTS_EXCLUS = re.compile(
    r"\b(total|sous-total|tva|mwst|iva|tax|montant|solde|report|frais|port|"
    r"rabais total|net à payer|à payer|iban|bic|téléphone|tél|facture|date)\b", re.I)
MOTS_PARASITES = re.compile(r"\b(chf|eur|€|fr\.?|pce?s?|stk|ex\.?)\b", re.I)
RE_REMISE_EXPLICITE = re.compile(r"(?:remise|rabais|-)\s?(\d{1,2}(?:[.,]\d{1,2})?)\s?%", re.I)


def _heuristique(page: Page) -> list[dict]:
    candidats = []
    for rangee in trier_lecture(_nettoyer(page.lignes)):
        texte = " ".join(l.texte for l in rangee)
        if MOTS_EXCLUS.search(texte):
            continue
        sans_date = RE_DATE.sub(" ", texte)
        prix = [_prix(p) for p in RE_PRIX.findall(sans_date)]
        if not prix:
            continue
        eans = RE_EAN.findall(sans_date)
        remise = RE_REMISE_EXPLICITE.search(sans_date)   # « 10% » seul = souvent la TVA
        nom = RE_EAN.sub(" ", sans_date)
        for motif in (RE_PRIX, RE_PCT, MOTS_PARASITES):
            nom = motif.sub(" ", nom)
        nom = re.sub(r"^\s*\d+\s+|\s\d+\s*$", " ", nom)
        nom = _nom_propre(nom)
        if len(re.sub(r"[^A-Za-zÀ-ÿ]", "", nom)) < 3:
            continue
        unitaires = prix[:-1] if len(prix) >= 2 else prix
        ean = eans[-1] if eans else ""
        alertes = ["format non reconnu : vérifier la ligne"]
        if ean and not ean_valide(ean):
            alertes.append("clé EAN invalide")
        if len(eans) > 1:
            alertes.append("plusieurs EAN sur la ligne")
        confiance = min(l.confiance for l in rangee)
        if confiance < SEUIL_CONFIANCE:
            alertes.append(f"lecture incertaine ({confiance:.0%})")
        arret = bool(RE_ARRET.search(texte))
        if arret:
            alertes.insert(0, "arrêt de commercialisation")
        candidats.append({
            "nom": nom, "ean": ean, "ancien_ean": "",
            "prix_negocie": min(unitaires),
            "prix_public": max(unitaires) if len(unitaires) >= 2 else None,
            "remise_pct": _pct(remise.group(1)) if remise else None,
            "tva_pct": None, "groupage": None, "arret": arret, "categorie": "",
            "confiance": round(confiance, 2), "page": page.numero,
            "ligne_ocr": texte, "alertes_liste": alertes, "boite": _boite(rangee),
        })
    return candidats


# --------------------------------------------------------------------------- #
# Point d'entrée
# --------------------------------------------------------------------------- #
# Alertes qui décochent la ligne par défaut (l'utilisateur doit la valider lui-même)
BLOQUANTES = ("arrêt", "fusionnées", "clé EAN invalide", "incertaine", "aberrant",
              "manquant", "format non reconnu", "écart", "double")


def finaliser(candidats: list[dict]) -> list[dict]:
    """Contrôles sur l'ensemble du document + colonnes garder / alertes."""
    vus: dict[str, int] = {}
    for c in candidats:
        if c["ean"]:
            vus[c["ean"]] = vus.get(c["ean"], 0) + 1
    for c in candidats:
        if c["ean"] and vus[c["ean"]] > 1:
            c["alertes_liste"].append("EAN en double dans le document")
        c["garder"] = not any(any(b in a for b in BLOQUANTES) for a in c["alertes_liste"])
        c["alertes"] = "✅" if not c["alertes_liste"] else "⚠️ " + " ; ".join(c["alertes_liste"])
    return candidats


def extraire(pages: list[Page], images: list | None = None) -> tuple[list[dict], dict]:
    """Retourne (candidats, rapport). `images` (optionnel) : images des pages,
    utilisées pour repérer les traits verticaux du tableau."""
    images = images or [None] * len(pages)
    candidats, rapport = [], {"methode": [], "rangees": 0, "sections": 0, "sans_prix": 0,
                              "hors_colonne": 0}
    gabarit_prec = None
    for page, image in zip(pages, images):
        gabarit = detecter_gabarit(page, image)
        if gabarit is None and gabarit_prec is not None:
            gabarit = gabarit_prec.a_l_echelle(page.largeur)     # page suivante sans en-tête
        c = None
        if gabarit is not None:
            c, stats = lire_tableau(page, gabarit)
            # garde-fou : si la majorité des produits n'ont pas de prix, les
            # colonnes sont mal comprises -> méthode de secours pour cette page
            if c and sum(x["prix_negocie"] is None for x in c) > len(c) / 2:
                c = None
            else:
                for k, v in stats.items():
                    rapport[k] += v
                rapport["methode"].append("colonnes")
                rapport.setdefault("colonnes", [f"{col.champ} ← « {col.libelle} »"
                                                for col in gabarit.colonnes])
                gabarit_prec = gabarit
        if c is None:
            c = _heuristique(page)
            rapport["methode"].append("heuristique")
        candidats.extend(c)
    return finaliser(candidats), rapport


def extraire_candidats(pages: list[Page], images: list | None = None) -> list[dict]:
    """Compatibilité : liste des candidats seulement."""
    return extraire(pages, images)[0]
