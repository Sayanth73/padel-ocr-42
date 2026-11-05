# 🏓 Padel OCR

Logiciel OCR local basé sur **PaddleOCR 3.x** (modèles PP-OCRv5).
Glisse des images ou des PDF scannés → texte dans l'ordre de lecture, boîtes colorées par confiance, export TXT / JSON / CSV.
Tout tourne sur ta machine : aucun document n'est envoyé en ligne.

## Installation & lancement

Prérequis : **Python 3.9 à 3.13** (64 bits).

- **Windows** : double-clic sur `lancer.bat`
- **macOS / Linux** : `chmod +x lancer.sh && ./lancer.sh`

Le premier lancement crée un environnement `.venv` et installe les dépendances (~1 Go).
La première analyse télécharge les modèles (~20–200 Mo selon langue/mode) dans `~/.paddlex/` ; ensuite tout fonctionne hors ligne.
L'interface s'ouvre sur http://127.0.0.1:7860.

Installation manuelle :
```bash
python -m venv .venv
# Windows : .venv\Scripts\activate    —    macOS/Linux : source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Interface web (`app.py`)

| Réglage | Effet |
|---|---|
| Langue | choisit le modèle de reconnaissance (fr, en, de, it, es, pt, ch, japan, korean, ar, ru) |
| Mode rapide / précis | détecteur `PP-OCRv5_mobile_det` (CPU, rapide) ou `PP-OCRv5_server_det` (docs denses, plus lent) |
| Corriger pages tournées | active la classification d'orientation page + ligne (photos de travers, texte à 180°) |
| Confiance minimale | élimine les lignes dont le score de reconnaissance est sous le seuil |
| DPI (PDF) | résolution du rendu des pages ; 200 suffit, 300 pour petites polices |

Onglets : **Texte** (copiable, tabulation = colonne séparée, utile pour factures/tableaux) · **Visualisation** (vert ≥ 90 %, jaune 70–90 %, rouge < 70 %) · **Lignes détaillées** · **Téléchargements**.
Le CSV utilise `;` et l'encodage UTF-8 BOM → s'ouvre directement dans Excel (version CH/FR) avec les accents.

## Ligne de commande (`cli.py`) — traitement par lot

```bash
python cli.py facture.pdf
python cli.py scans/ --lang de --mode precis --out resultats --annoter
python cli.py photo1.jpg photo2.jpg --orientation --seuil 0.7
```
Produit un `.txt` et un `.json` par fichier (+ images annotées avec `--annoter`).

## Structure

```
ocr_engine.py   moteur : chargement images/PDF, appel PaddleOCR, ordre de lecture, dessin, export
app.py          interface Gradio
cli.py          traitement par lot
```

Utiliser le moteur dans ton propre code :
```python
from ocr_engine import MoteurOCR, texte_complet
pages, _ = MoteurOCR().analyser_fichier("facture.pdf", langue="fr")
print(texte_complet(pages))
for l in pages[0].lignes:
    print(l.texte, l.confiance, l.boite)   # boite = [x1, y1, x2, y2] en pixels
```

## GPU NVIDIA (optionnel)

Remplace `paddlepaddle` par la version GPU correspondant à ta version de CUDA
(commande exacte sur https://www.paddlepaddle.org.cn/en/install/quick). Aucun changement de code.

## Dépannage

- **`No model source is available`** → pas d'accès internet au premier lancement (ou proxy/pare-feu). Les modèles viennent de HuggingFace / BOS / ModelScope ; on peut forcer une source : `set PADDLE_PDX_MODEL_SOURCE=huggingface` (Windows) / `export PADDLE_PDX_MODEL_SOURCE=huggingface`.
- **Lent sur gros PDF** → mode rapide, DPI 150, ou limite le nombre de pages.
- **Texte mal ordonné sur mises en page complexes** (multi-colonnes, magazines) → l'ordre de lecture est heuristique (haut→bas, gauche→droite). Pour ce cas, PaddleOCR propose `PPStructureV3` (analyse de mise en page), non intégré ici.
