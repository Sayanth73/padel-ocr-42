"""
Base de données des deals pharmacies (SQLite, fichier local pharma.db).

Principe : un prix négocié appartient au couple (pharmacie, produit) sur une
période. Chaque nouvelle négociation AJOUTE une ligne dans `deals` et clôture
l'ancienne -> l'historique complet est conservé.
"""
from __future__ import annotations

import sqlite3
import unicodedata
from datetime import date
from pathlib import Path

CHEMIN_DB = Path(__file__).with_name("pharma.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS pharmacies (
    id        INTEGER PRIMARY KEY,
    nom       TEXT NOT NULL UNIQUE,
    adresse   TEXT
);
CREATE TABLE IF NOT EXISTS produits (
    id           INTEGER PRIMARY KEY,
    nom          TEXT NOT NULL,
    nom_norm     TEXT NOT NULL,          -- nom normalisé pour le rapprochement
    ean          TEXT UNIQUE,            -- code-barres 13 chiffres (si connu)
    pharmacode   TEXT UNIQUE,            -- code 7 chiffres (CH), si connu
    prix_public  REAL
);
CREATE INDEX IF NOT EXISTS idx_produits_nom ON produits(nom_norm);
CREATE TABLE IF NOT EXISTS documents (
    id         INTEGER PRIMARY KEY,
    fichier    TEXT NOT NULL,
    date_scan  TEXT NOT NULL,
    texte_ocr  TEXT
);
CREATE TABLE IF NOT EXISTS deals (
    id            INTEGER PRIMARY KEY,
    pharmacie_id  INTEGER NOT NULL REFERENCES pharmacies(id),
    produit_id    INTEGER NOT NULL REFERENCES produits(id),
    prix_negocie  REAL NOT NULL,
    remise_pct    REAL,
    date_debut    TEXT NOT NULL,         -- AAAA-MM-JJ
    date_fin      TEXT,                  -- NULL = deal en cours
    document_id   INTEGER REFERENCES documents(id),
    note          TEXT
);
CREATE INDEX IF NOT EXISTS idx_deals_couple ON deals(pharmacie_id, produit_id, date_debut);

"""

# Vue recréée à chaque connexion (sa définition peut évoluer)
VUE = """
DROP VIEW IF EXISTS prix_actuels;
-- Prix en vigueur aujourd'hui pour chaque (pharmacie, produit)
CREATE VIEW prix_actuels AS
SELECT ph.nom AS pharmacie, pr.nom AS produit, pr.gamme, pr.ean, d.prix_negocie,
       d.remise_pct, pr.prix_public, d.date_debut, d.id AS deal_id
FROM deals d
JOIN pharmacies ph ON ph.id = d.pharmacie_id
JOIN produits pr   ON pr.id = d.produit_id
WHERE d.date_debut <= date('now')
  AND (d.date_fin IS NULL OR d.date_fin > date('now'));
"""


def connexion(chemin: Path | str | None = None) -> sqlite3.Connection:
    # chemin résolu à l'appel (et non à l'import) : base.CHEMIN_DB peut être changé
    con = sqlite3.connect(chemin or CHEMIN_DB)
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    # migrations : colonnes ajoutées après la première version
    existantes = {r[1] for r in con.execute("PRAGMA table_info(produits)")}
    for col, typ in (("ancien_ean", "TEXT"), ("tva_pct", "REAL"), ("gamme", "TEXT"),
                     ("arret", "INTEGER DEFAULT 0")):
        if col not in existantes:
            con.execute(f"ALTER TABLE produits ADD COLUMN {col} {typ}")
    con.executescript(VUE)
    return con


def normaliser(nom: str) -> str:
    """'Dafalgan 500mg  cpr' -> 'DAFALGAN 500MG CPR' (sans accents, espaces uniques)."""
    s = unicodedata.normalize("NFKD", nom).encode("ascii", "ignore").decode()
    return " ".join(s.upper().split())


# --------------------------------------------------------------------------- #
# Pharmacies / produits
# --------------------------------------------------------------------------- #
def liste_pharmacies(con) -> list[str]:
    return [r[0] for r in con.execute("SELECT nom FROM pharmacies ORDER BY nom")]


def pharmacie_id(con, nom: str) -> int:
    nom = nom.strip()
    if not nom:
        raise ValueError("Nom de pharmacie vide")
    con.execute("INSERT OR IGNORE INTO pharmacies(nom) VALUES (?)", (nom,))
    return con.execute("SELECT id FROM pharmacies WHERE nom = ?", (nom,)).fetchone()[0]


def trouver_ou_creer_produit(con, nom: str, ean: str | None = None,
                             pharmacode: str | None = None,
                             prix_public: float | None = None,
                             ancien_ean: str | None = None, tva_pct: float | None = None,
                             gamme: str | None = None, arret: bool = False) -> tuple[int, bool]:
    """Rapprochement d'un produit existant. Retourne (id, cree).
    Avec un EAN : on ne rapproche QUE par code (EAN, ancien EAN, pharmacode).
    Le nom n'est utilisé que si la ligne n'a aucun code : beaucoup de produits
    portent le même nom (« Boîte de 10 pansements 15 cm x 20 cm » existe dans
    plusieurs gammes), un rapprochement par nom créerait de fausses données."""
    ean = (ean or "").strip() or None
    ancien_ean = (ancien_ean or "").strip() or None
    pharmacode = (pharmacode or "").strip() or None
    gamme = (gamme or "").strip() or None
    nom_norm = normaliser(nom)
    recherches = [("ean = ?", ean), ("ancien_ean = ?", ean), ("ean = ?", ancien_ean),
                  ("pharmacode = ?", pharmacode)]
    if not (ean or ancien_ean or pharmacode):
        recherches.append(("nom_norm = ? AND ean IS NULL", nom_norm))
    for condition, val in recherches:
        if not val:
            continue
        r = con.execute(f"SELECT id, ean FROM produits WHERE {condition}", (val,)).fetchone()
        if r:
            pid, ean_actuel = r
            if ean and ean_actuel and ean_actuel != ean:
                # le produit a changé d'EAN : on garde l'ancien en mémoire
                con.execute("UPDATE produits SET ancien_ean = ean, ean = ? WHERE id = ?", (ean, pid))
            con.execute("""UPDATE produits SET
                             ean = COALESCE(ean, ?), ancien_ean = COALESCE(ancien_ean, ?),
                             pharmacode = COALESCE(pharmacode, ?),
                             prix_public = COALESCE(?, prix_public), tva_pct = COALESCE(?, tva_pct),
                             gamme = COALESCE(gamme, ?), arret = ?
                           WHERE id = ?""", (ean, ancien_ean, pharmacode, prix_public, tva_pct,
                                             gamme, int(bool(arret)), pid))
            return pid, False
    cur = con.execute(
        """INSERT INTO produits(nom, nom_norm, ean, pharmacode, prix_public, ancien_ean,
                                tva_pct, gamme, arret) VALUES (?,?,?,?,?,?,?,?,?)""",
        (nom.strip(), nom_norm, ean, pharmacode, prix_public, ancien_ean, tva_pct, gamme,
         int(bool(arret))))
    return cur.lastrowid, True


def dernier_prix(con, ean: str, pharmacie: str | None = None) -> tuple[float, str, str] | None:
    """Dernier prix connu (deal en cours) d'un produit : (prix, pharmacie, date)."""
    sql = ("SELECT prix_negocie, pharmacie, date_debut FROM prix_actuels WHERE ean = ?"
           + (" AND pharmacie = ?" if pharmacie else "") + " ORDER BY date_debut DESC LIMIT 1")
    return con.execute(sql, (ean, pharmacie) if pharmacie else (ean,)).fetchone()


# --------------------------------------------------------------------------- #
# Documents / deals
# --------------------------------------------------------------------------- #
def ajouter_document(con, fichier: str, texte_ocr: str) -> int:
    cur = con.execute("INSERT INTO documents(fichier, date_scan, texte_ocr) VALUES (?,?,?)",
                      (fichier, date.today().isoformat(), texte_ocr))
    return cur.lastrowid


def ajouter_deal(con, pharmacie: str, produit_id: int, prix: float,
                 date_debut: str | None = None, remise_pct: float | None = None,
                 document_id: int | None = None, note: str | None = None) -> int | None:
    """Ajoute un deal à l'historique du couple (pharmacie, produit).
    - Même prix (et même remise) que le deal en vigueur à cette date : rien
      n'est ajouté (retourne None) -> enregistrer deux fois le même document
      ne crée pas de doublons.
    - Sinon le deal en vigueur est clôturé à la date du nouveau.
    - Deal plus ancien que le deal en cours (ancien tarif importé après un
      récent) : il est inséré dans l'historique, clôturé à la date du suivant,
      sans toucher au prix actuel."""
    ph_id = pharmacie_id(con, pharmacie)
    date_debut = date_debut or date.today().isoformat()
    en_vigueur = con.execute(
        """SELECT id, prix_negocie, remise_pct FROM deals
           WHERE pharmacie_id = ? AND produit_id = ? AND date_debut <= ?
             AND (date_fin IS NULL OR date_fin > ?)
           ORDER BY date_debut DESC, id DESC LIMIT 1""",
        (ph_id, produit_id, date_debut, date_debut)).fetchone()
    if en_vigueur and abs(en_vigueur[1] - prix) < 0.005 and en_vigueur[2] == remise_pct:
        return None
    suivant = con.execute(
        """SELECT MIN(date_debut) FROM deals
           WHERE pharmacie_id = ? AND produit_id = ? AND date_debut > ?""",
        (ph_id, produit_id, date_debut)).fetchone()[0]
    if en_vigueur:
        con.execute("UPDATE deals SET date_fin = ? WHERE id = ?", (date_debut, en_vigueur[0]))
    cur = con.execute(
        """INSERT INTO deals(pharmacie_id, produit_id, prix_negocie, remise_pct,
                             date_debut, date_fin, document_id, note) VALUES (?,?,?,?,?,?,?,?)""",
        (ph_id, produit_id, prix, remise_pct, date_debut, suivant, document_id, note))
    return cur.lastrowid


def prix_actuels(con, pharmacie: str | None = None, recherche: str | None = None):
    sql = ("SELECT pharmacie, produit, gamme, ean, prix_negocie, remise_pct, prix_public, date_debut "
           "FROM prix_actuels WHERE 1=1")
    params: list = []
    if pharmacie:
        sql += " AND pharmacie = ?"
        params.append(pharmacie)
    if recherche:
        sql += " AND (produit LIKE ? OR ean LIKE ? OR gamme LIKE ?)"
        params += [f"%{recherche}%"] * 3
    sql += " ORDER BY produit, prix_negocie"
    return [list(r) for r in con.execute(sql, params)]


def historique(con, recherche: str):
    """Tous les deals (passés et en cours) d'un produit, toutes pharmacies."""
    sql = """SELECT pr.nom, ph.nom, d.prix_negocie, d.remise_pct, d.date_debut,
                    COALESCE(d.date_fin, 'en cours'), doc.fichier
             FROM deals d JOIN pharmacies ph ON ph.id = d.pharmacie_id
             JOIN produits pr ON pr.id = d.produit_id
             LEFT JOIN documents doc ON doc.id = d.document_id
             WHERE pr.nom LIKE ? OR pr.ean LIKE ?
             ORDER BY pr.nom, d.date_debut DESC"""
    return [list(r) for r in con.execute(sql, (f"%{recherche}%",) * 2)]
