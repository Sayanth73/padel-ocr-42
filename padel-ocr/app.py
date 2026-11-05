"""
Padel OCR — interface web locale (Gradio) autour de PaddleOCR.

Lancer :  python app.py      puis ouvrir http://127.0.0.1:7860
"""
from __future__ import annotations

import csv
import json
import re
import tempfile
import time
from pathlib import Path

import gradio as gr

import base
from extraction import BLOQUANTES, COLONNES_SORTIE, detecter_date, ean_valide, extraire, suggerer_ean
from PIL import Image, ImageDraw

from ocr_engine import LANGUES, analyser_isole, charger_pages, dessiner, texte_complet, vers_dict

DOSSIER_SORTIE = Path(tempfile.gettempdir()) / "padel_ocr"
DOSSIER_SORTIE.mkdir(exist_ok=True)

CSS = """
#titre h1 {margin-bottom: 0}
#titre p {margin-top: 4px; opacity: .75}
.legende span {padding: 2px 8px; border-radius: 6px; margin-right: 6px; font-size: 13px}
"""


def lancer_ocr(fichiers, langue, mode, orientation, seuil, dpi, max_pages,
               filigrane_actif, filigrane_seuil, texte_integre, progress=gr.Progress()):
    if not fichiers:
        raise gr.Error("Ajoute au moins une image ou un PDF.")

    t0 = time.time()
    galerie, lignes_tableau, textes, export_json = [], [], [], []
    pages_pdf = 0
    for f in fichiers:
        chemin = Path(f if isinstance(f, str) else f.name)
        progress(0, desc=f"Chargement du modèle / lecture de {chemin.name}…")

        def prog(i, n, nom=chemin.name):
            progress(i / n, desc=f"{nom} — page {i}/{n}")

        try:
            pages, images = analyser_isole(
                chemin, langue=langue, mode=mode, orientation=orientation,
                seuil=seuil, dpi=int(dpi), max_pages=int(max_pages) or None,
                progression=prog,
                filigrane=int(filigrane_seuil) if filigrane_actif else None,
                texte_integre=texte_integre,
            )
        except (ValueError, RuntimeError) as e:
            raise gr.Error(f"{chemin.name} : {e}")
        pages_pdf += sum(p.source == "pdf" for p in pages)

        for page, img in zip(pages, images):
            origine = "texte du PDF" if page.source == "pdf" else "OCR"
            legende = f"{chemin.name} — p.{page.numero} ({len(page.lignes)} lignes, {origine})"
            galerie.append((dessiner(img, page), legende))
            for idx, l in enumerate(page.lignes, start=1):
                lignes_tableau.append([chemin.name, page.numero, idx, l.texte, l.confiance])

        entete = f"########## {chemin.name} ##########\n" if len(fichiers) > 1 else ""
        textes.append(entete + texte_complet(pages))
        export_json.append(vers_dict(chemin.name, pages, langue))

    texte_final = "\n\n".join(textes)

    # Fichiers à télécharger
    horodatage = time.strftime("%Y%m%d-%H%M%S")
    f_txt = DOSSIER_SORTIE / f"ocr_{horodatage}.txt"
    f_json = DOSSIER_SORTIE / f"ocr_{horodatage}.json"
    f_csv = DOSSIER_SORTIE / f"ocr_{horodatage}.csv"
    f_txt.write_text(texte_final, encoding="utf-8")
    f_json.write_text(json.dumps(export_json, ensure_ascii=False, indent=2), encoding="utf-8")
    with f_csv.open("w", newline="", encoding="utf-8-sig") as fh:   # utf-8-sig : accents OK dans Excel
        w = csv.writer(fh, delimiter=";")
        w.writerow(["fichier", "page", "n°", "texte", "confiance"])
        w.writerows(lignes_tableau)

    nb = len(lignes_tableau)
    moy = sum(r[4] for r in lignes_tableau) / nb if nb else 0
    faibles = sum(1 for r in lignes_tableau if r[4] < 0.7)
    stats = (f"**{len(galerie)} page(s)** · **{nb} lignes** · confiance moyenne "
             f"**{moy:.1%}** · {faibles} ligne(s) < 70 % · {time.time() - t0:.1f} s")
    if pages_pdf:
        stats += f" · {pages_pdf} page(s) lue(s) directement dans le PDF (sans OCR)"

    return galerie, texte_final, lignes_tableau, [str(f_txt), str(f_json), str(f_csv)], stats


# --------------------------------------------------------------------------- #
# Deals pharmacies
# --------------------------------------------------------------------------- #
COLONNES_DEAL = COLONNES_SORTIE
TYPES_DEAL = ["bool", "str", "str", "str", "number", "number", "number", "number", "str",
              "number", "bool", "str", "number", "number", "str"]
assert len(TYPES_DEAL) == len(COLONNES_DEAL)
ECART_MAX = 0.30          # alerte si le prix change de plus de 30 % vs le dernier prix connu
COLONNES_PRIX = ["pharmacie", "produit", "gamme", "ean", "prix_negocie", "remise_%", "prix_public",
                 "depuis"]
COLONNES_HISTO = ["produit", "pharmacie", "prix_negocie", "remise_%", "du", "au", "document"]


def _num(v):
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    try:
        f = float(str(v).replace(",", "."))
    except ValueError:
        return None
    return None if f != f else f          # NaN -> None


def _choix_pharmacies():
    with base.connexion() as con:
        return base.liste_pharmacies(con)


# Alertes venant de la lecture (non recalculables depuis la ligne). Elles restent
# affichées tant que l'utilisateur n'a pas corrigé la ligne lui-même.
ALERTES_LECTURE = ("incertaine", "fusionnées", "format non reconnu", "ancien EAN")
# Erreurs « dures » : si une modification en fait apparaître une, la ligne est décochée
ERREURS_DURES = ("clé EAN invalide", "prix aberrant", "EAN en double", "prix manquant",
                 "nom manquant", "écart")
CHAMPS_SUIVIS = {"nom": "nom", "ean": "EAN", "prix_negocie": "prix", "prix_public": "prix public",
                 "remise_pct": "remise", "tva_pct": "TVA", "ancien_ean": "ancien EAN"}


def _vrai(v):
    return str(v).lower() not in ("false", "0", "", "none", "nan")


def _texte(v):
    return "" if v is None or (isinstance(v, float) and v != v) else str(v).strip()


def _format(v):
    t = _texte(v)
    if re.fullmatch(r"\d{13}", t):
        return t
    n = _num(v)
    return f"{n:.2f}" if n is not None and t else (t or "∅")


def _egal(a, b):
    na, nb = _num(a), _num(b)
    if na is not None and nb is not None and not re.fullmatch(r"\d{13}", _texte(a)):
        return abs(na - nb) < 0.005
    return _texte(a) == _texte(b)


def controler(lignes: list[dict], pharmacie: str, etat: dict, initial: bool = False):
    """(Re)calcule la colonne « alertes » de chaque ligne. Appelé à l'analyse ET à
    chaque modification du tableau. Retourne (lignes, nb_modifiees, details)."""
    originaux = etat.get("originaux", [])
    alertes_lecture = etat.get("alertes_lecture", [])
    pharmacie = (pharmacie or "").strip() or None
    eans = [_texte(d.get("ean")) for d in lignes]
    details = []
    with base.connexion() as con:
        for i, d in enumerate(lignes):
            orig = originaux[i] if i < len(originaux) else None
            if orig is None:
                modifs = ["ligne ajoutée à la main"]
            else:
                modifs = [f"{lib} {_format(orig.get(ch))} → {_format(d.get(ch))}"
                          for ch, lib in CHAMPS_SUIVIS.items() if not _egal(d.get(ch), orig.get(ch))]
            alertes = []
            if _vrai(d.get("arret")):
                alertes.append("arrêt de commercialisation")
            if orig is not None and not modifs and i < len(alertes_lecture):
                alertes += alertes_lecture[i]
            nom, ean, prix = _texte(d.get("nom")), _texte(d.get("ean")), _num(d.get("prix_negocie"))
            if not nom:
                alertes.append("nom manquant")
            if not ean:
                alertes.append("EAN manquant")
            elif not ean_valide(ean):
                alertes.append("clé EAN invalide")
            elif eans.count(ean) > 1:
                alertes.append("EAN en double dans le document")
            if prix is None:
                if not _vrai(d.get("arret")):
                    alertes.append("prix manquant")
            elif not 0 < prix <= 5000:
                alertes.append("prix aberrant")
            pp = _num(d.get("prix_public"))
            if prix is not None and pp is not None and prix > pp:
                alertes.append("prix négocié > prix public")
            if ean and prix:
                prec = base.dernier_prix(con, ean, pharmacie)
                if prec and prec[0] and abs(prix - prec[0]) / prec[0] > ECART_MAX:
                    alertes.append(f"écart {prix / prec[0] - 1:+.0%} vs dernier prix {prec[0]:.2f} "
                                   f"({prec[1]}, {prec[2]})")
            if modifs:
                alertes.insert(0, "✏️ modifié : " + ", ".join(modifs))
                details.append(f"{nom or '(sans nom)'} — " + ", ".join(modifs))

            avant = _texte(d.get("alertes"))
            if initial:
                d["garder"] = not any(any(b in a for b in BLOQUANTES) for a in alertes)
            elif any(e in " ".join(alertes) and e not in avant for e in ERREURS_DURES):
                d["garder"] = False          # une modification vient d'introduire une erreur
            problemes = [a for a in alertes if not a.startswith("✏️")]
            d["alertes"] = ("✅" if not problemes else "⚠️ " + " ; ".join(problemes))
            if modifs:
                d["alertes"] = alertes[0] + ("" if not problemes else " | ⚠️ " + " ; ".join(problemes))
    return lignes, len(details), details


def _resume_modifs(n, details):
    if not n:
        return ""
    liste = "\n".join(f"- {d}" for d in details[:15])
    plus = f"\n- … et {n - 15} autre(s)" if n > 15 else ""
    return f"✏️ **{n} ligne(s) modifiée(s) à la main** (par rapport à la lecture du document) :\n{liste}{plus}"


def _vers_dicts(tableau):
    lignes = tableau.values.tolist() if hasattr(tableau, "values") else (tableau or [])
    return [dict(zip(COLONNES_DEAL, l)) for l in lignes]


def revalider(tableau, pharmacie, etat):
    """Appelé à chaque modification du tableau par l'utilisateur."""
    lignes = _vers_dicts(tableau)
    if not lignes:
        return gr.update(), ""
    lignes, n, details = controler(lignes, pharmacie, etat or {})
    return [[d.get(k) for k in COLONNES_DEAL] for d in lignes], _resume_modifs(n, details)


# --------------------------------------------------------------------------- #
# Vérification visuelle : ligne du tableau <-> passage du document original
# --------------------------------------------------------------------------- #
def _sauver_pages(chemin: Path, images_analyse) -> list[str]:
    """Enregistre les pages ORIGINALES (avec filigrane, telles que dans le
    document) pour la vérification. Même résolution que l'analyse, donc les
    coordonnées des lignes s'appliquent telles quelles."""
    try:
        originales = charger_pages(chemin, dpi=200)
    except Exception:
        originales = images_analyse
    dossier = DOSSIER_SORTIE / f"verif_{time.strftime('%Y%m%d-%H%M%S')}_{chemin.stem[:30]}"
    dossier.mkdir(parents=True, exist_ok=True)
    chemins = []
    for i, (orig, ana) in enumerate(zip(originales, images_analyse), start=1):
        img = Image.fromarray(orig)
        if img.size != (ana.shape[1], ana.shape[0]):          # sécurité : même taille que l'analyse
            img = img.resize((ana.shape[1], ana.shape[0]))
        f = dossier / f"page_{i}.jpg"
        img.save(f, quality=88)
        chemins.append(str(f))
    return chemins


def _couleur(alertes: str, garder) -> tuple:
    if _texte(alertes) == "✅":
        return (22, 163, 74)          # vert
    if "⚠️" in _texte(alertes):
        return (234, 88, 12) if not _vrai(garder) else (234, 179, 8)   # orange / jaune si coché
    return (37, 99, 235)              # bleu : modifié à la main sans problème


def apercu(etat, tableau, idx):
    """Zoom sur la ligne `idx` + page entière avec toutes les lignes colorées."""
    lignes = _vers_dicts(tableau)
    pages = (etat or {}).get("pages_img", [])
    boites = (etat or {}).get("boites", [])
    if not pages or idx is None or idx < 0 or idx >= len(lignes):
        return None, None, "Clique sur une ligne du tableau pour la voir dans le document."
    d = lignes[idx]
    if idx >= len(boites) or not boites[idx]:
        return None, None, f"Ligne {idx + 1} : ajoutée à la main, pas de position dans le document."
    num_page, boite = boites[idx]
    page = Image.open(pages[num_page - 1]).convert("RGB")
    W, H = page.size
    x1, y1, x2, y2 = boite
    h = max(20, y2 - y1)

    # zoom : la ligne + quelques lignes au-dessus/en dessous, sur toute la largeur
    zoom = page.crop((0, max(0, y1 - 5 * h), W, min(H, y2 + 5 * h)))
    dz = ImageDraw.Draw(zoom, "RGBA")
    oy = max(0, y1 - 5 * h)
    dz.rectangle([2, y1 - oy - 6, W - 3, y2 - oy + 6], outline=(37, 99, 235, 255), width=6,
                 fill=(37, 99, 235, 30))

    # page entière : toutes les lignes extraites de cette page, colorées par statut
    vue = page.copy()
    dv = ImageDraw.Draw(vue, "RGBA")
    for j, (dj, bj) in enumerate(zip(lignes, boites)):
        if not bj or bj[0] != num_page:
            continue
        c = _couleur(dj.get("alertes"), dj.get("garder"))
        bx1, by1, bx2, by2 = bj[1]
        dv.rectangle([bx1 - 4, by1 - 2, bx2 + 4, by2 + 2], outline=c + (255,), width=4, fill=c + (40,))
    dv.rectangle([0, y1 - 14, W - 1, y2 + 14], outline=(37, 99, 235, 255), width=10)
    vue.thumbnail((1100, 1600))

    n_alertes = sum("⚠️" in _texte(x.get("alertes")) for x in lignes)
    legende = (f"**Ligne {idx + 1}/{len(lignes)}** — page {num_page} — **{_texte(d.get('nom'))}** · "
               f"prix lu : **{_format(d.get('prix_negocie'))}** · EAN : `{_texte(d.get('ean'))}`\n\n"
               f"{_texte(d.get('alertes'))}\n\n"
               f"<small>Page : 🟩 sans alerte · 🟧 alerte (décochée) · 🟨 alerte mais cochée · "
               f"🟦 ligne affichée — {n_alertes} ligne(s) avec alerte au total</small>")
    return zoom, vue, legende


def _a_verifier(etat):
    return (etat or {}).get("a_verifier", [])


def panneau(etat, tableau, idx):
    """Contenu du panneau de vérification. idx=None après la dernière alerte :
    écran final avec le récapitulatif et le bouton d'enregistrement."""
    lignes = _vers_dicts(tableau)
    etat = etat or {}
    a_verif, decisions = _a_verifier(etat), etat.get("decisions", {})
    faites = len([i for i in a_verif if str(i) in decisions])
    fin = bool(lignes) and (idx is None or not (0 <= idx < len(lignes)))
    cache_form, montre_form = gr.update(visible=False), gr.update(visible=True)

    if fin:
        valides = sum(1 for v in decisions.values() if v)
        ignorees = sum(1 for v in decisions.values() if not v)
        a_enregistrer = sum(_vrai(d.get("garder")) and bool(_texte(d.get("nom")))
                            and _num(d.get("prix_negocie")) is not None for d in lignes)
        recap = (f"## ✅ Vérification terminée\n\n"
                 f"- **{valides}** alerte(s) validée(s) · **{ignorees}** ignorée(s)\n"
                 f"- **{len(lignes) - len(a_verif)}** ligne(s) sans alerte\n"
                 f"- **{a_enregistrer} deal(s) seront enregistrés**\n\n"
                 "Vérifie la **pharmacie** et la **date** à gauche, puis enregistre.")
        if not a_verif:
            recap = (f"## ✅ Aucune alerte\n\n**{a_enregistrer} deal(s)** prêts. Vérifie la "
                     "**pharmacie** et la **date** à gauche, puis enregistre.")
        _, vue, _ = apercu(etat, tableau, 0)
        return (None, vue, "", gr.update(value=""), gr.update(value=""), gr.update(value=None), "",
                gr.update(visible=False), "", None, cache_form, gr.update(visible=True), recap)

    if not lignes:
        return (None, None, "Analyse un document pour commencer.", gr.update(value=""), gr.update(value=""),
                gr.update(value=None), "", gr.update(visible=False), "", None, cache_form,
                gr.update(visible=False), "")

    zoom, vue, legende = apercu(etat, tableau, idx)
    position = (a_verif.index(idx) + 1) if idx in a_verif else None
    progres = (f"**Alerte {position} / {len(a_verif)}** · {faites} traitée(s)" if position
               else f"Ligne {idx + 1} (sans alerte) · {faites} / {len(a_verif)} alerte(s) traitée(s)")
    d = lignes[idx]
    ean = _texte(d.get("ean"))
    sugg_md, sugg_btn = "", gr.update(visible=False)
    if ean and not ean_valide(ean):
        autres = [_texte(x.get("ean")) for k, x in enumerate(lignes) if k != idx]
        sugg = suggerer_ean(ean, autres)
        if sugg:
            sugg_md = f"💡 **Suggestion : `{sugg[0]}`** — {sugg[1]}."
            sugg_btn = gr.update(visible=True, value=f"Appliquer {sugg[0]}")
        else:
            sugg_md = ("💡 Pas de correction évidente : compare avec le zoom, ou laisse le code "
                       "tel quel si le document est lui-même faux.")
    if str(idx) in decisions:
        legende = (f"☑️ *Déjà traitée ({'validée' if decisions[str(idx)] else 'ignorée'}) — "
                   "tu peux changer d'avis.*\n\n" + legende)
    return (zoom, vue, legende, gr.update(value=_texte(d.get("nom"))), gr.update(value=ean),
            gr.update(value=_num(d.get("prix_negocie"))), sugg_md, sugg_btn, progres, idx,
            montre_form, gr.update(visible=False), "")


def _prochaine(etat, idx):
    """Alerte suivante NON traitée, dans l'ordre, sans jamais boucler.
    None quand il n'en reste plus -> écran final."""
    a_verif, decisions = _a_verifier(etat), (etat or {}).get("decisions", {})
    restantes = [i for i in a_verif if str(i) not in decisions]
    if not restantes:
        return None
    idx = -1 if idx is None else idx
    return next((i for i in restantes if i > idx), restantes[0])


def selection_ligne(etat, tableau, evt: gr.SelectData):
    idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    return panneau(etat, tableau, idx)


def decider(etat, tableau, idx, nom, ean, prix, garder, pharmacie):
    """Boutons Valider / Ignorer : applique les champs du panneau à la ligne,
    recontrôle, puis passe à l'alerte suivante non traitée."""
    lignes = _vers_dicts(tableau)
    etat = etat or {}
    if idx is None or not (0 <= idx < len(lignes)):
        return (gr.update(), gr.update(), *panneau(etat, tableau, idx), etat)
    d = lignes[idx]
    ean_avant, prix_avant = _texte(d.get("ean")), _num(d.get("prix_negocie"))
    d["nom"], d["ean"], d["prix_negocie"], d["garder"] = (nom or "").strip(), (ean or "").strip(), prix, garder
    lignes, n, details = controler(lignes, pharmacie, etat)
    tableau = [[x.get(k) for k in COLONNES_DEAL] for x in lignes]
    # tu as corrigé un champ mais la nouvelle valeur est toujours fausse -> refus
    nouvelle_ean, nouveau_prix = (ean or "").strip(), _num(prix)
    if garder and nouvelle_ean != ean_avant and nouvelle_ean and not ean_valide(nouvelle_ean):
        lignes[idx]["garder"] = False
    if garder and not _egal(nouveau_prix, prix_avant) and (nouveau_prix is None or not 0 < nouveau_prix <= 5000):
        lignes[idx]["garder"] = False
    # même EAN déjà validé sur une autre ligne : enregistrer les deux écraserait l'un par l'autre
    if garder and nouvelle_ean:
        autre = next((k for k, x in enumerate(lignes) if k != idx and _vrai(x.get("garder"))
                      and _texte(x.get("ean")) == nouvelle_ean), None)
        if autre is not None:
            lignes[idx]["garder"] = False
            tableau = [[x.get(k) for k in COLONNES_DEAL] for x in lignes]
            p = list(panneau(etat, tableau, idx))
            p[2] = (f"⛔ **Même EAN que la ligne {autre + 1} (« {_texte(lignes[autre].get('nom'))} »), "
                    "déjà validée.** Les deux s'écraseraient. Corrige l'EAN ou clique sur « Ignorer ».\n\n"
                    + p[2])
            return (tableau, _resume_modifs(n, details), *p, etat)
    sans_prix = garder and nouveau_prix is None
    if sans_prix:
        lignes[idx]["garder"] = False
    tableau = [[x.get(k) for k in COLONNES_DEAL] for x in lignes]
    if sans_prix:
        p = list(panneau(etat, tableau, idx))
        p[2] = ("⛔ **Pas de prix : impossible d'enregistrer un deal sans prix.** Saisis le prix "
                "(voir le document) ou clique sur « Ignorer ».\n\n" + p[2])
        return (tableau, _resume_modifs(n, details), *p, etat)
    if garder and not _vrai(lignes[idx]["garder"]):
        # la correction vient d'introduire une erreur : on reste sur la ligne
        p = list(panneau(etat, tableau, idx))
        p[2] = "⛔ **Pas validée : ta modification introduit une erreur.**\n\n" + p[2]
        return (tableau, _resume_modifs(n, details), *p, etat)
    etat.setdefault("decisions", {})[str(idx)] = bool(garder)
    return (tableau, _resume_modifs(n, details), *panneau(etat, tableau, _prochaine(etat, idx)), etat)


def appliquer_suggestion(tableau, idx, ean):
    lignes = _vers_dicts(tableau)
    autres = [_texte(x.get("ean")) for k, x in enumerate(lignes) if k != idx]
    sugg = suggerer_ean((ean or "").strip(), autres)
    return gr.update(value=sugg[0]) if sugg else gr.update()


def analyser_deal(fichier, pharmacie, langue, mode, filigrane_actif, filigrane_seuil,
                  texte_integre, progress=gr.Progress()):
    if not fichier:
        raise gr.Error("Ajoute la facture / l'offre (image ou PDF).")
    chemin = Path(fichier if isinstance(fichier, str) else fichier.name)
    progress(0, desc=f"OCR de {chemin.name}…")
    try:
        pages, images = analyser_isole(
            chemin, langue=langue, mode=mode, seuil=0.5,
            filigrane=int(filigrane_seuil) if filigrane_actif else None,
            texte_integre=texte_integre,
            progression=lambda i, n: progress(i / n, desc=f"{chemin.name} — page {i}/{n}"))
    except (ValueError, RuntimeError) as e:
        raise gr.Error(str(e))
    texte = texte_complet(pages)
    cands, rapport = extraire(pages, images)

    etat = {"fichier": chemin.name, "texte": texte,
            "pages_img": _sauver_pages(chemin, images),
            "boites": [(c["page"], c["boite"]) if c.get("boite") else None for c in cands],
            "originaux": [{k: c[k] for k in COLONNES_DEAL} for c in cands],
            "alertes_lecture": [[a for a in c["alertes_liste"] if any(k in a for k in ALERTES_LECTURE)]
                                for c in cands]}
    controler(cands, pharmacie, etat, initial=True)
    lignes = [[c[k] for k in COLONNES_DEAL] for c in cands]
    date_doc = detecter_date(texte)
    avert_date = ""
    if not date_doc:
        date_doc = time.strftime("%Y-%m-%d")
        avert_date = ("\n\n⚠️ **Aucune date trouvée dans le document** : la date du jour est "
                      "proposée. Mets la date de validité du tarif / du deal avant d'enregistrer.")
    n_ok = sum(c["alertes"] == "✅" for c in cands)
    n_coches = sum(c["garder"] for c in cands)
    n_arret = sum(c["arret"] for c in cands)
    if "colonnes" in rapport["methode"]:
        methode = ("**Tableau reconnu** — colonnes : "
                   + ", ".join(rapport.get("colonnes", [])) + ".")
    else:
        methode = ("⚠️ **Aucun en-tête de tableau reconnu** : extraction approximative, "
                   "vérifie chaque ligne.")
    origine = "texte du PDF" if all(p.source == "pdf" for p in pages) else "OCR"
    msg = (f"{methode}\n\n**{len(cands)} produit(s)** lus ({origine}) sur "
           f"{rapport['rangees']} rangées ({rapport['sections']} titres de section ignorés). "
           f"✅ {n_ok} sans alerte · ⚠️ {len(cands) - n_ok} avec alerte (dont {n_arret} en arrêt "
           f"de commercialisation). **{n_coches} ligne(s) cochée(s)** pour l'enregistrement : "
           "les lignes avec alerte sont décochées, vérifie-les et coche-les si elles sont justes."
           + avert_date)
    if not lignes:
        msg = ("Aucune ligne « produit + prix » reconnue. Vérifie le texte OCR ci-dessous ; "
               "tu peux aussi ajouter des lignes à la main dans le tableau.")
        lignes = [[True, "", "", "", None, None, None, None, "", None, False, "", None, None, ""]]
    etat["a_verifier"] = [i for i, c in enumerate(cands) if "⚠️" in c["alertes"]]
    etat["decisions"] = {}
    premier = etat["a_verifier"][0] if etat["a_verifier"] else None
    return (lignes, date_doc, texte, etat, msg, "", *panneau(etat, lignes, premier))


def enregistrer_deals(tableau, pharmacie, date_deal, etat):
    if (etat or {}).get("enregistre"):
        raise gr.Error("Ce document est déjà enregistré. Analyse un nouveau document pour continuer.")
    pharmacie = (pharmacie or "").strip()
    if not pharmacie:
        raise gr.Error("Choisis ou tape le nom de la pharmacie.")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", (date_deal or "").strip()):
        raise gr.Error("Date au format AAAA-MM-JJ, ex. 2026-11-05.")
    lignes = tableau.values.tolist() if hasattr(tableau, "values") else (tableau or [])

    vrai = _vrai
    retenues = [dict(zip(COLONNES_DEAL, l)) for l in lignes if vrai(dict(zip(COLONNES_DEAL, l)).get("garder"))]
    # deux lignes cochées avec le même EAN : la 2e écraserait la 1re -> refus
    eans = [str(d.get("ean") or "").strip() for d in retenues if str(d.get("ean") or "").strip()]
    doublons = sorted({e for e in eans if eans.count(e) > 1})
    if doublons:
        raise gr.Error("Même EAN coché sur plusieurs lignes : " + ", ".join(doublons)
                       + ". Décoche ou corrige l'une d'elles.")
    n_deals = n_nouveaux = 0
    ignores, changements = [], []
    n_inchanges = 0
    with base.connexion() as con:
        doc_id = base.ajouter_document(con, etat.get("fichier", "saisie manuelle"),
                                       etat.get("texte", "")) if etat else None
        for d in retenues:
            nom, prix = str(d.get("nom") or "").strip(), _num(d.get("prix_negocie"))
            if not nom or prix is None:
                ignores.append(nom or "(sans nom)")
                continue
            pid, cree = base.trouver_ou_creer_produit(
                con, nom, ean=str(d.get("ean") or ""), prix_public=_num(d.get("prix_public")),
                ancien_ean=str(d.get("ancien_ean") or ""), tva_pct=_num(d.get("tva_pct")),
                gamme=str(d.get("categorie") or ""), arret=vrai(d.get("arret")))
            ean = str(d.get("ean") or "").strip()
            prec = base.dernier_prix(con, ean, pharmacie) if ean else None
            if prec and abs(prec[0] - prix) >= 0.005:
                changements.append(f"{nom} : {prec[0]:.2f} → {prix:.2f} ({prix / prec[0] - 1:+.0%}, "
                                   f"ancien prix du {prec[2]})")
            ajoute = base.ajouter_deal(con, pharmacie, pid, prix, date_debut=date_deal.strip(),
                                       remise_pct=_num(d.get("remise_pct")), document_id=doc_id)
            if ajoute is None:
                n_inchanges += 1
            else:
                n_deals += 1
            n_nouveaux += cree
    msg = f"✅ **{n_deals} deal(s) enregistré(s)** pour {pharmacie} ({n_nouveaux} nouveau(x) produit(s))."
    if n_inchanges:
        msg += (f" **{n_inchanges}** prix identique(s) à ceux déjà en base : rien ajouté pour eux "
                "(pas de doublon).")
    if changements:
        msg += (f"\n\n🔄 **{len(changements)} prix changé(s)** par rapport au deal précédent :\n"
                + "\n".join(f"- {c}" for c in changements[:30])
                + (f"\n- … et {len(changements) - 30} autre(s)" if len(changements) > 30 else ""))
    elif not n_inchanges:
        msg += " Aucun prix existant n'a changé."
    if ignores:
        msg += f"\n\nIgnorées (nom ou prix manquant) : {', '.join(ignores)}."
    if etat:
        etat["enregistre"] = True
    return msg, gr.update(choices=_choix_pharmacies(), value=pharmacie), \
        gr.update(choices=[""] + _choix_pharmacies()), msg, etat


def consulter(pharmacie, recherche):
    with base.connexion() as con:
        actuels = base.prix_actuels(con, pharmacie or None, (recherche or "").strip() or None)
        histo = base.historique(con, recherche.strip()) if recherche and recherche.strip() else []
    return actuels, histo


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #
with gr.Blocks(title="Padel OCR") as demo:
    gr.Markdown("# 🏓 Padel OCR\nExtraction de texte d'images et de PDF scannés avec PaddleOCR — 100 % local.",
                elem_id="titre")

    with gr.Tabs():
        with gr.Tab("OCR"):
            with gr.Row():
                with gr.Column(scale=1):
                    fichiers = gr.File(label="Images ou PDF", file_count="multiple",
                                       file_types=["image", ".pdf"])
                    langue = gr.Dropdown(choices=[(v, k) for k, v in LANGUES.items()], value="fr",
                                         label="Langue du document")
                    mode = gr.Radio(choices=[("Rapide (CPU)", "rapide"), ("Précis (~10 Go de RAM !)", "precis")],
                                    value="rapide", label="Mode")
                    orientation = gr.Checkbox(value=False, label="Corriger les pages tournées / texte à l'envers")
                    seuil = gr.Slider(0, 1, value=0.5, step=0.05, label="Confiance minimale")
                    filigrane_actif = gr.Checkbox(value=True, label="Supprimer les filigranes gris")
                    filigrane_seuil = gr.Slider(
                        80, 240, value=160, step=5, label="Seuil filigrane",
                        info="Plus bas = efface aussi les gris foncés. Si le vrai texte s'abîme, monte la valeur. "
                             "Vérifie le résultat dans l'onglet Visualisation.")
                    with gr.Accordion("Options PDF", open=False):
                        dpi = gr.Slider(100, 400, value=200, step=50, label="Résolution de rendu (DPI)")
                        max_pages = gr.Number(value=0, precision=0, label="Nombre max de pages (0 = toutes)")
                        texte_integre = gr.Checkbox(value=True, label="PDF numérique : lire le texte directement (sans OCR)",
                                        info="Décoche si le texte intégré du PDF est faux (ex. scan avec mauvais OCR intégré).")
                    bouton = gr.Button("Lancer l'OCR", variant="primary")
                    stats = gr.Markdown()
                    gr.HTML('<div class="legende">Boîtes : '
                            '<span style="background:#16a34a33">≥ 90 %</span>'
                            '<span style="background:#eab30833">70–90 %</span>'
                            '<span style="background:#dc262633">&lt; 70 %</span></div>')

                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.Tab("Texte"):
                            texte = gr.Textbox(label="Texte extrait (ordre de lecture)", lines=22,
                                               buttons=["copy"])
                        with gr.Tab("Visualisation"):
                            galerie = gr.Gallery(label="Pages annotées", columns=1, height=650,
                                                 object_fit="contain")
                        with gr.Tab("Lignes détaillées"):
                            tableau = gr.Dataframe(headers=["fichier", "page", "n°", "texte", "confiance"],
                                                   wrap=True, interactive=False)
                        with gr.Tab("Téléchargements"):
                            sorties = gr.File(label="TXT · JSON · CSV", file_count="multiple")

            bouton.click(lancer_ocr,
                         inputs=[fichiers, langue, mode, orientation, seuil, dpi, max_pages,
                                 filigrane_actif, filigrane_seuil, texte_integre],
                         outputs=[galerie, texte, tableau, sorties, stats])


        with gr.Tab("Deals pharmacies"):
            etat_doc = gr.State({})
            d_idx = gr.State(None)
            with gr.Row():
                with gr.Column(scale=1, min_width=260):
                    d_fichier = gr.File(label="Facture / offre de la pharmacie",
                                        file_types=["image", ".pdf"])
                    d_pharmacie = gr.Dropdown(choices=_choix_pharmacies(), allow_custom_value=True,
                                              label="Pharmacie", info="Choisis-en une ou tape un nouveau nom")
                    d_date = gr.Textbox(label="Date du deal (AAAA-MM-JJ)",
                                        info="Détectée dans le document si possible")
                    with gr.Accordion("Options de lecture", open=False):
                        d_langue = gr.Dropdown(choices=[(v, k) for k, v in LANGUES.items()], value="fr",
                                               label="Langue du document")
                        d_mode = gr.Radio(choices=[("Rapide (CPU)", "rapide"),
                                                   ("Précis (~10 Go de RAM !)", "precis")],
                                          value="rapide", label="Mode")
                        d_fili = gr.Checkbox(value=True, label="Supprimer les filigranes gris")
                        d_fili_seuil = gr.Slider(80, 240, value=160, step=5, label="Seuil filigrane")
                        d_texte_integre = gr.Checkbox(
                            value=True, label="PDF numérique : lire le texte directement (sans OCR)",
                            info="Décoche si le texte intégré du PDF est faux.")
                    d_analyser = gr.Button("1. Analyser le document", variant="primary")
                    d_enregistrer = gr.Button("2. Enregistrer les deals", variant="secondary")
                    d_msg = gr.Markdown()

                with gr.Column(scale=4):
                    # --- Panneau de vérification : document + correction côte à côte
                    gr.Markdown("### 🔍 Vérification des alertes")
                    d_progres = gr.Markdown()
                    with gr.Row():
                        with gr.Column(scale=3):
                            d_zoom = gr.Image(label="Document original (filigrane compris)", type="pil",
                                              interactive=False)
                            d_legende = gr.Markdown("Analyse un document pour commencer.")
                        with gr.Column(scale=2, min_width=260, visible=False) as r_form:
                            r_nom = gr.Textbox(label="Nom du produit")
                            r_ean = gr.Textbox(label="EAN")
                            r_sugg = gr.Markdown()
                            r_btn_sugg = gr.Button("Appliquer la suggestion", size="sm", visible=False)
                            r_prix = gr.Number(label="Prix négocié")
                            with gr.Row():
                                r_valider = gr.Button("✅ Valider et suivante", variant="primary")
                                r_ignorer = gr.Button("❌ Ignorer et suivante", variant="stop")
                            gr.Markdown("<small>Valider = la ligne sera enregistrée (avec tes corrections). "
                                        "Ignorer = elle ne sera pas enregistrée.</small>")
                    with gr.Column(visible=False) as r_fin:
                        r_recap = gr.Markdown()
                        r_enregistrer = gr.Button("💾 Enregistrer les deals", variant="primary", size="lg")
                    with gr.Accordion("Voir la page entière", open=False):
                        d_page = gr.Image(label="Page entière — 🟩 sans alerte · 🟧 alerte décochée · "
                                                "🟨 alerte cochée · 🟦 ligne affichée",
                                          type="pil", interactive=False, height=800)

                    # --- Vue d'ensemble
                    gr.Markdown("### Toutes les lignes")
                    d_modifs = gr.Markdown()
                    d_tableau = gr.Dataframe(headers=COLONNES_DEAL, datatype=TYPES_DEAL,
                                             interactive=True, wrap=True, type="array",
                                             row_count=(1, "dynamic"), col_count=(len(COLONNES_DEAL), "fixed"),
                                             label="Clique sur une ligne pour l'afficher dans le panneau "
                                                   "ci-dessus · modifiable directement aussi")
                    with gr.Accordion("Texte OCR brut", open=False):
                        d_texte = gr.Textbox(lines=15, buttons=["copy"], show_label=False)

        with gr.Tab("Prix actuels"):
            with gr.Row():
                p_pharmacie = gr.Dropdown(choices=[""] + _choix_pharmacies(), value="",
                                          label="Pharmacie (vide = toutes)")
                p_recherche = gr.Textbox(label="Produit ou EAN", placeholder="ex. Dafalgan")
                p_bouton = gr.Button("Rechercher", variant="primary")
            p_actuels = gr.Dataframe(headers=COLONNES_PRIX, interactive=False, wrap=True,
                                     label="Prix en vigueur (le moins cher en premier par produit)")
            p_histo = gr.Dataframe(headers=COLONNES_HISTO, interactive=False, wrap=True,
                                   label="Historique des négociations (si un produit est recherché)")

    PANNEAU = [d_zoom, d_page, d_legende, r_nom, r_ean, r_prix, r_sugg, r_btn_sugg, d_progres, d_idx,
               r_form, r_fin, r_recap]
    d_analyser.click(analyser_deal, inputs=[d_fichier, d_pharmacie, d_langue, d_mode, d_fili, d_fili_seuil,
                                            d_texte_integre],
                     outputs=[d_tableau, d_date, d_texte, etat_doc, d_msg, d_modifs, *PANNEAU])
    d_tableau.select(selection_ligne, inputs=[etat_doc, d_tableau], outputs=PANNEAU,
                     show_progress="hidden")
    for bouton_decision, garder in ((r_valider, True), (r_ignorer, False)):
        bouton_decision.click(
            lambda e, t, i, n, ea, pr, ph, g=garder: decider(e, t, i, n, ea, pr, g, ph),
            inputs=[etat_doc, d_tableau, d_idx, r_nom, r_ean, r_prix, d_pharmacie],
            outputs=[d_tableau, d_modifs, *PANNEAU, etat_doc], show_progress="hidden")
    r_btn_sugg.click(appliquer_suggestion, inputs=[d_tableau, d_idx, r_ean], outputs=r_ean,
                     show_progress="hidden")
    d_tableau.input(revalider, inputs=[d_tableau, d_pharmacie, etat_doc],
                    outputs=[d_tableau, d_modifs], show_progress="hidden")
    for bouton_enreg in (d_enregistrer, r_enregistrer):
        bouton_enreg.click(enregistrer_deals, inputs=[d_tableau, d_pharmacie, d_date, etat_doc],
                           outputs=[d_msg, d_pharmacie, p_pharmacie, r_recap, etat_doc])
    p_bouton.click(consulter, inputs=[p_pharmacie, p_recherche], outputs=[p_actuels, p_histo])
    p_recherche.submit(consulter, inputs=[p_pharmacie, p_recherche], outputs=[p_actuels, p_histo])
    demo.load(consulter, inputs=[p_pharmacie, p_recherche], outputs=[p_actuels, p_histo])


if __name__ == "__main__":
    demo.queue().launch(theme=gr.themes.Soft(), css=CSS, inbrowser=True)
