"""
Moteur OCR basé sur PaddleOCR 3.x.

Rôle : charger le modèle (une fois par langue/mode), transformer images et PDF
en pages, lancer l'OCR, remettre les lignes dans l'ordre de lecture et dessiner
les boîtes détectées.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Évite un appel réseau de vérification à chaque démarrage.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
PDF_EXT = {".pdf"}
# PaddleOCR redimensionne de toute façon au-delà de 4000 px : on réduit dès le
# chargement pour économiser la mémoire (une page A4 à 600 DPI = ~100 Mo en RAM).
COTE_MAX = 4000
TAILLE_DETECTION = 2048


def _reduire(img: Image.Image) -> Image.Image:
    if max(img.size) > COTE_MAX:
        img = img.copy()
        img.thumbnail((COTE_MAX, COTE_MAX), Image.LANCZOS)
    return img

# Langues proposées dans l'interface (code PaddleOCR -> libellé)
LANGUES = {
    "fr": "Français",
    "en": "Anglais",
    "de": "Allemand",
    "it": "Italien",
    "es": "Espagnol",
    "pt": "Portugais",
    "ch": "Chinois + anglais",
    "japan": "Japonais",
    "korean": "Coréen",
    "ar": "Arabe",
    "ru": "Russe",
}

MODES = {
    "rapide": "PP-OCRv5_mobile_det",   # détection légère, idéale sur CPU
    "precis": "PP-OCRv5_server_det",   # détection lourde, meilleure sur docs denses
}


@dataclass
class Ligne:
    texte: str
    confiance: float
    boite: list[int]            # [x1, y1, x2, y2]
    polygone: list[list[int]] = field(default_factory=list)


@dataclass
class Page:
    numero: int
    largeur: int
    hauteur: int
    lignes: list[Ligne]
    source: str = "ocr"          # "ocr" ou "pdf" (texte lu directement dans le PDF)

    @property
    def texte(self) -> str:
        return assembler_texte(self.lignes)


# --------------------------------------------------------------------------- #
# Chargement des fichiers
# --------------------------------------------------------------------------- #
# En dessous de ce nombre de caractères, une page PDF est considérée comme un
# scan (pas de vrai texte intégré) et passe par l'OCR.
MIN_CARACTERES_TEXTE = 20


def _fusion_mots(mots, echelle: float) -> Ligne:
    return Ligne(texte=" ".join(m[4] for m in mots), confiance=1.0,
                 boite=[int(min(m[0] for m in mots) * echelle), int(min(m[1] for m in mots) * echelle),
                        int(max(m[2] for m in mots) * echelle), int(max(m[3] for m in mots) * echelle)])


def lignes_texte_integre(page_pdf, echelle: float) -> list[Ligne] | None:
    """Lit le texte déjà présent dans une page PDF (PDF numérique, pas un scan).
    Retourne None si la page n'a pas assez de texte -> il faudra l'OCR.
    Les mots d'une même ligne sont regroupés ; un grand espace horizontal
    (= colonne de tableau) coupe la ligne en plusieurs boîtes, comme l'OCR."""
    mots = page_pdf.get_text("words")
    if sum(len(m[4]) for m in mots) < MIN_CARACTERES_TEXTE:
        return None
    groupes: dict[tuple, list] = {}
    for m in mots:
        groupes.setdefault((m[5], m[6]), []).append(m)       # (bloc, ligne)
    lignes = []
    for ms in groupes.values():
        ms.sort(key=lambda m: m[0])
        courant = [ms[0]]
        for m in ms[1:]:
            hauteur = max(1.0, courant[-1][3] - courant[-1][1])
            if m[0] - courant[-1][2] > 1.5 * hauteur:
                lignes.append(_fusion_mots(courant, echelle))
                courant = [m]
            else:
                courant.append(m)
        lignes.append(_fusion_mots(courant, echelle))
    return lignes


def charger_document(chemin: str | Path, dpi: int = 200, max_pages: int | None = None,
                     texte_integre: bool = True) -> list[tuple[np.ndarray, list[Ligne] | None]]:
    """Liste de (image RGB, lignes du texte intégré ou None si OCR nécessaire)."""
    chemin = Path(chemin)
    if chemin.suffix.lower() not in PDF_EXT or not texte_integre:
        return [(img, None) for img in charger_pages(chemin, dpi, max_pages)]
    import pymupdf

    resultat = []
    with pymupdf.open(chemin) as doc:
        for i, page in enumerate(doc):
            if max_pages and i >= max_pages:
                break
            pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            img = np.array(_reduire(Image.fromarray(arr)))
            echelle = img.shape[1] / page.rect.width          # points PDF -> pixels
            resultat.append((img, lignes_texte_integre(page, echelle)))
    return resultat


def pdf_entierement_texte(chemin: str | Path, max_pages: int | None = None) -> bool:
    """True si toutes les pages (à traiter) d'un PDF ont du vrai texte : pas besoin d'OCR."""
    chemin = Path(chemin)
    if chemin.suffix.lower() not in PDF_EXT:
        return False
    import pymupdf

    with pymupdf.open(chemin) as doc:
        n = min(len(doc), max_pages) if max_pages else len(doc)
        return n > 0 and all(
            sum(len(m[4]) for m in doc[i].get_text("words")) >= MIN_CARACTERES_TEXTE for i in range(n))


def charger_pages(chemin: str | Path, dpi: int = 200, max_pages: int | None = None) -> list[np.ndarray]:
    """Retourne une liste d'images RGB (numpy) : 1 pour une image, N pour un PDF."""
    chemin = Path(chemin)
    ext = chemin.suffix.lower()
    if ext in IMAGE_EXT:
        img = Image.open(chemin)
        img = _reduire(img.convert("RGB"))
        return [np.array(img)]
    if ext in PDF_EXT:
        import pymupdf

        pages = []
        with pymupdf.open(chemin) as doc:
            for i, page in enumerate(doc):
                if max_pages and i >= max_pages:
                    break
                pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
                pages.append(np.array(_reduire(Image.fromarray(arr))))
        return pages
    raise ValueError(f"Format non pris en charge : {ext} (images ou PDF uniquement)")


def supprimer_filigrane(image_rgb: np.ndarray, seuil: int = 160, tolerance_couleur: int = 40) -> np.ndarray:
    """Efface les filigranes gris : tout pixel gris (peu saturé) plus clair que `seuil`
    (0 = noir, 255 = blanc) devient blanc. Le texte noir/foncé et les éléments en
    couleur sont conservés. Baisser `seuil` si le filigrane est foncé ; le monter si
    le vrai texte devient trop fin ou disparaît."""
    img = image_rgb.astype(np.int16)
    luminance = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    chroma = img.max(axis=2) - img.min(axis=2)
    masque = (luminance > seuil) & (chroma < tolerance_couleur)
    propre = image_rgb.copy()
    propre[masque] = 255
    return propre


# --------------------------------------------------------------------------- #
# Ordre de lecture
# --------------------------------------------------------------------------- #
def bandes_sombres(image_rgb: np.ndarray, seuil: int = 110) -> list[tuple[int, int]]:
    """Bandes horizontales majoritairement foncées, de 12 à 250 px de haut
    (bandeaux d'en-tête de tableau, en texte clair sur fond foncé)."""
    sombre = (image_rgb.mean(axis=2) < seuil).mean(axis=1) > 0.5
    bandes, y, n = [], 0, len(sombre)
    while y < n:
        if not sombre[y]:
            y += 1
            continue
        z = y
        while z < n and sombre[z]:
            z += 1
        if 12 <= z - y <= 250:
            bandes.append((y, z))
        y = z
    return bandes


def inverser_bandes_sombres(image_rgb: np.ndarray, bandes=None) -> np.ndarray:
    """Passe les bandes foncées en négatif -> texte foncé sur fond clair,
    que l'OCR lit bien. Le reste de l'image est inchangé."""
    bandes = bandes_sombres(image_rgb) if bandes is None else bandes
    if not bandes:
        return image_rgb
    out = image_rgb.copy()
    for y, z in bandes:
        out[y:z] = 255 - out[y:z]
    return out


def trier_lecture(lignes: list[Ligne]) -> list[list[Ligne]]:
    """Regroupe les boîtes en rangées (même hauteur) puis trie de gauche à droite."""
    if not lignes:
        return []
    restantes = sorted(lignes, key=lambda l: (l.boite[1] + l.boite[3]) / 2)
    rangees: list[list[Ligne]] = []
    for l in restantes:
        y1, y2 = l.boite[1], l.boite[3]
        centre = (y1 + y2) / 2
        place = False
        if rangees:
            ref = rangees[-1]
            ry1 = min(r.boite[1] for r in ref)
            ry2 = max(r.boite[3] for r in ref)
            h = max(1, min(y2 - y1, ry2 - ry1))
            # même rangée si le centre tombe dans la bande verticale de la rangée
            if ry1 - 0.25 * h <= centre <= ry2 + 0.25 * h:
                ref.append(l)
                place = True
        if not place:
            rangees.append([l])
    return [sorted(r, key=lambda l: l.boite[0]) for r in rangees]


def assembler_texte(lignes: list[Ligne]) -> str:
    out = []
    for rangee in trier_lecture(lignes):
        morceaux = [rangee[0].texte]
        for prec, cur in zip(rangee, rangee[1:]):
            ecart = cur.boite[0] - prec.boite[2]
            hauteur = max(1, prec.boite[3] - prec.boite[1])
            # grand espace horizontal = colonne différente (ex. tableau, facture)
            morceaux.append("\t" if ecart > 1.5 * hauteur else " ")
            morceaux.append(cur.texte)
        out.append("".join(morceaux))
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Moteur
# --------------------------------------------------------------------------- #
class MoteurOCR:
    """Garde en cache un modèle PaddleOCR par (langue, mode, orientation)."""

    def __init__(self):
        self._cache: dict[tuple, object] = {}

    def _modele(self, langue: str, mode: str, orientation: bool):
        cle = (langue, mode, orientation)
        if cle not in self._cache:
            from paddleocr import PaddleOCR
            import gc

            # Un seul modèle en mémoire : le mode précis seul prend ~10 Go,
            # cumulé avec un autre modèle il fait planter une machine 16 Go.
            self._cache.clear()
            gc.collect()

            self._cache[cle] = PaddleOCR(
                lang=langue,
                text_detection_model_name=MODES.get(mode, MODES["rapide"]),
                use_doc_orientation_classify=orientation,
                use_doc_unwarping=False,
                use_textline_orientation=orientation,
                # Le détecteur travaille sur une image réduite (plus grand côté
                # <= 2048 px) : sans ça, une page A4 haute résolution consomme
                # 7 à 10+ Go de RAM et le processus est tué (code 137).
                text_det_limit_type="max",
                text_det_limit_side_len=TAILLE_DETECTION,
            )
        return self._cache[cle]

    def analyser_image(self, image_rgb: np.ndarray, langue="fr", mode="rapide",
                       orientation=False, seuil=0.5, numero=1) -> Page:
        modele = self._modele(langue, mode, orientation)
        bandes = bandes_sombres(image_rgb)
        image = inverser_bandes_sombres(image_rgb, bandes)
        resultats = modele.predict(image[:, :, ::-1].copy())   # PaddleOCR attend du BGR
        lignes = []
        for res in resultats:
            lignes.extend(_extraire_lignes(res, seuil))
        # Bandeaux d'en-tête : intitulés serrés que la détection pleine page
        # manque souvent. On relit chaque bandeau seul, avec une marge blanche,
        # et cette lecture remplace celle de la page pour la bande.
        h_img = image.shape[0]
        for y, z in bandes:
            y0, z0 = max(0, y - 4), min(h_img, z + 4)
            marge = 40
            bande = np.full((z0 - y0 + 2 * marge, image.shape[1] + 2 * marge, 3), 255, np.uint8)
            bande[marge:-marge, marge:-marge] = image[y0:z0]
            relues = []
            for res in modele.predict(bande[:, :, ::-1].copy()):
                for l in _extraire_lignes(res, seuil):
                    x1, y1, x2, y2 = l.boite
                    relues.append(Ligne(l.texte, l.confiance,
                                        [x1 - marge, y1 - marge + y0, x2 - marge, y2 - marge + y0],
                                        [[px - marge, py - marge + y0] for px, py in l.polygone]))
            if relues:
                lignes = [l for l in lignes if not (y0 <= (l.boite[1] + l.boite[3]) / 2 <= z0)] + relues
        # numérotation = ordre de lecture (haut→bas, gauche→droite)
        lignes = [l for rangee in trier_lecture(lignes) for l in rangee]
        h, w = image_rgb.shape[:2]
        return Page(numero=numero, largeur=w, hauteur=h, lignes=lignes)

    def analyser_fichier(self, chemin, langue="fr", mode="rapide", orientation=False,
                         seuil=0.5, dpi=200, max_pages=None, progression=None,
                         filigrane: int | None = None, texte_integre: bool = True):
        """Retourne (pages_resultats, images_rgb).
        `filigrane` : seuil de suppression des filigranes gris (None = désactivé).
        `texte_integre` : pour les PDF numériques, lit le texte directement (sans OCR)."""
        document = charger_document(chemin, dpi=dpi, max_pages=max_pages, texte_integre=texte_integre)
        pages, images = [], []
        for i, (img, integre) in enumerate(document, start=1):
            if progression:
                progression(i, len(document))
            h, w = img.shape[:2]
            if integre is not None:
                lignes = [l for rangee in trier_lecture(integre) for l in rangee]
                pages.append(Page(numero=i, largeur=w, hauteur=h, lignes=lignes, source="pdf"))
            else:
                if filigrane:
                    img = supprimer_filigrane(img, filigrane)
                pages.append(self.analyser_image(img, langue, mode, orientation, seuil, numero=i))
            images.append(img)
        return pages, images


# --------------------------------------------------------------------------- #
# Analyse dans un processus séparé
# --------------------------------------------------------------------------- #
class ErreurMemoire(RuntimeError):
    pass


def analyser_isole(chemin, progression=None, **options):
    """Comme MoteurOCR().analyser_fichier(), mais l'OCR tourne dans un processus
    séparé (ocr_worker.py) qui se termine après le document :
      - toute la mémoire est rendue au système à la fin ;
      - si l'OCR manque de mémoire et est tué, l'appelant (le serveur) survit
        et reçoit une ErreurMemoire au lieu de planter.
    Un PDF entièrement numérique est lu directement, sans lancer l'OCR."""
    if options.get("texte_integre", True) and pdf_entierement_texte(chemin, options.get("max_pages")):
        return MoteurOCR().analyser_fichier(chemin, progression=progression, **options)

    with tempfile.TemporaryDirectory(prefix="padel_ocr_") as dossier:
        sortie = Path(dossier) / "resultat.pkl"
        journal = Path(dossier) / "worker.log"
        cmd = [sys.executable, str(Path(__file__).with_name("ocr_worker.py")),
               str(chemin), str(sortie), json.dumps(options)]
        message_erreur = None
        with journal.open("w") as log:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log, text=True,
                                    cwd=Path(__file__).parent)
            try:
                for ligne in proc.stdout:
                    if ligne.startswith("@@PROG ") and progression:
                        i, n = ligne.split()[1:3]
                        progression(int(i), int(n))
                    elif ligne.startswith("@@ERREUR "):
                        message_erreur = ligne[len("@@ERREUR "):].strip()
                code = proc.wait()
            except BaseException:                 # requête annulée, Ctrl+C...
                proc.kill()
                proc.wait()
                raise
        if message_erreur:
            raise ValueError(message_erreur)
        if code in (-signal.SIGKILL, 137):
            raise ErreurMemoire(
                "Le moteur OCR a été arrêté par le système (mémoire insuffisante). "
                "Essaie le mode rapide, un DPI plus bas, ou moins de pages à la fois.")
        if code != 0 or not sortie.exists():
            fin = journal.read_text(errors="replace").strip().splitlines()[-8:]
            raise RuntimeError(f"Le moteur OCR a échoué (code {code}) :\n" + "\n".join(fin))
        import pickle

        with sortie.open("rb") as f:
            return pickle.load(f)


def _champ(res, cle):
    try:
        return res[cle]
    except (KeyError, TypeError):
        return res.json["res"][cle]


def _extraire_lignes(res, seuil: float) -> list[Ligne]:
    textes = list(_champ(res, "rec_texts"))
    scores = list(_champ(res, "rec_scores"))
    boites = np.asarray(_champ(res, "rec_boxes")).reshape(-1, 4) if len(textes) else []
    try:
        polys = _champ(res, "rec_polys")
    except Exception:
        polys = [None] * len(textes)
    lignes = []
    for t, s, b, p in zip(textes, scores, boites, polys):
        t = str(t).strip()
        if not t or float(s) < seuil:
            continue
        poly = np.asarray(p).astype(int).tolist() if p is not None else []
        lignes.append(Ligne(texte=t, confiance=round(float(s), 4),
                            boite=[int(v) for v in b], polygone=poly))
    return lignes


# --------------------------------------------------------------------------- #
# Visualisation & export
# --------------------------------------------------------------------------- #
def couleur_confiance(c: float) -> tuple[int, int, int]:
    if c >= 0.9:
        return (22, 163, 74)     # vert
    if c >= 0.7:
        return (234, 179, 8)     # jaune
    return (220, 38, 38)         # rouge


def dessiner(image_rgb: np.ndarray, page: Page) -> Image.Image:
    img = Image.fromarray(image_rgb).convert("RGB")
    calque = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(calque)
    epaisseur = max(2, img.width // 600)
    police = None
    for nom in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf", "Helvetica.ttc"):
        try:
            police = ImageFont.truetype(nom, max(14, img.width // 80))
            break
        except OSError:
            continue
    police = police or ImageFont.load_default()
    for idx, l in enumerate(page.lignes, start=1):
        col = couleur_confiance(l.confiance)
        if l.polygone:
            pts = [tuple(p) for p in l.polygone]
            d.polygon(pts, fill=col + (40,), outline=col + (255,), width=epaisseur)
        else:
            d.rectangle(l.boite, fill=col + (40,), outline=col + (255,), width=epaisseur)
        d.text((l.boite[0], max(0, l.boite[1] - police.size - 2)), str(idx),
               fill=col + (255,), font=police)
    return Image.alpha_composite(img.convert("RGBA"), calque).convert("RGB")


def vers_dict(nom_fichier: str, pages: list[Page], langue: str) -> dict:
    return {
        "fichier": nom_fichier,
        "langue": langue,
        "pages": [
            {
                "numero": p.numero,
                "largeur": p.largeur,
                "hauteur": p.hauteur,
                "source": p.source,
                "texte": p.texte,
                "lignes": [asdict(l) for l in p.lignes],
            }
            for p in pages
        ],
    }


def texte_complet(pages: list[Page]) -> str:
    if len(pages) == 1:
        return pages[0].texte
    return "\n\n".join(f"===== Page {p.numero} =====\n{p.texte}" for p in pages)
