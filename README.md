
# Framework de défense adversariale pour IDS

Reproduction et extension de l'article :

**Awad, Z., Zakaria, M., & Hassan, R. (2025).** *An Enhanced Ensemble Defense Framework for Boosting Adversarial Robustness of Intrusion Detection Systems.* Scientific Reports, 15, 14177.
DOI : [10.1038/s41598-025-94023-z](https://doi.org/10.1038/s41598-025-94023-z)

---

## Contexte

Projet de stage de recherche.
Date de début : juillet 2026.

## Objectif

Reproduire et évaluer un framework de défense par ensemble contre les attaques adversariales sur des systèmes de détection d'intrusion (IDS) basés sur l'apprentissage profond.

Le projet suit les 9 étapes du framework proposé par Awad et al. (2025) :

1. Collection des données
2. Preprocessing
3. Sélection de features
4. Séparation train / test
5. Entraînement du DNN baseline
6. Génération des attaques adversariales
7. Test de vulnérabilité du baseline
8. Application des 4 mécanismes de défense
9. Agrégation par ensemble

---

## Statut d'avancement

| Étape | Description | Statut |
|---|---|---|
| 1 | Collection des données | Terminée |
| 2 | Preprocessing | Terminée |
| 3 | Feature selection | Terminée |
| 4 | Split train / test + normalisation + SMOTE | En cours |
| 5 | Baseline DNN | À faire |
| 6 | Attaques adversariales | À faire |
| 7 | Test de vulnérabilité | À faire |
| 8 | Mécanismes de défense | À faire |
| 9 | Agrégation par ensemble | À faire |

---

## Journal des étapes réalisées

### Étape 1 — Collection des données

- Dataset **CIC-IDS 2017** téléchargé depuis Kaggle (`chethuhn/network-intrusion-dataset`)
- **8 fichiers CSV** correspondant à 5 jours de capture (lundi bénin, mardi-vendredi avec attaques)
- **2 830 743 lignes** au total, **79 colonnes** (78 features + 1 label)
- **15 classes** : BENIGN (80.3%) + 14 types d'attaques
- Stockage sur `~/scratch/` (Alliance Canada) via lien symbolique vers `data/raw/`

### Étape 2 — Preprocessing

Nettoyage des données brutes en 4 opérations :

1. **Nettoyage des noms de colonnes** : 65 colonnes avaient un espace parasite en début de nom (`' Destination Port'` au lieu de `'Destination Port'`), artefact du séparateur `', '` utilisé par CICFlowMeter
2. **Traitement des valeurs Inf et NaN** : 2 867 lignes supprimées (0.101%), provenant de divisions par zéro dans `Flow Bytes/s` et `Flow Packets/s` (flux instantanés avec Flow Duration = 0)
3. **Suppression des doublons** : **307 078 doublons éliminés (10.86%)**, majoritairement issus d'attaques automatisées répétitives (PortScan, DoS Hulk, SSH-Patator)
4. **Encodage des labels** : les 15 classes textuelles converties en entiers 0-14 via `sklearn.preprocessing.LabelEncoder`

**Résultat** : dataset propre de **2 520 798 lignes × 79 colonnes**, sauvegardé en format pickle (`data/processed/cicids2017_clean.pkl`, 1.5 GB).

**Note méthodologique** : la suppression des doublons n'est pas explicitement documentée dans le papier Awad et al. Cette décision peut expliquer des écarts potentiels avec leurs résultats. C'est un choix conservateur pour éviter le data leakage entre train et test.

### Étape 3 — Feature selection

Sélection des 58 features les plus discriminantes via **Random Forest importance**, comme décrit dans le papier.

**Méthodologie** :
- **Random Forest** entraîné sur les **2.52M lignes complètes** (fidélité au papier, pas d'échantillonnage)
- **100 arbres**, critère de Gini, `random_state=42` pour la reproductibilité
- Exécution sur Alliance Canada via SLURM (16 CPUs, 32 GB RAM)
- **Durée d'exécution** : 52 secondes

**Résultats clés** :
- **Importance cumulée des 58 features retenues : 99.34%**
- Les 20 features rejetées ne représentent que 0.66% de l'information discriminante
- **6 features avec importance strictement nulle** (`Bwd Avg Bulk Rate`, `Fwd Avg Bulk Rate`, etc.) : artefacts de CICFlowMeter pour mesurer les transferts en rafale, non pertinents dans ce dataset

**Top 10 des features les plus importantes** :

| Rang | Feature | Importance |
|---|---|---:|
| 1 | Packet Length Variance | 0.0618 |
| 2 | Packet Length Std | 0.0598 |
| 3 | Avg Bwd Segment Size | 0.0565 |
| 4 | Max Packet Length | 0.0491 |
| 5 | Bwd Packet Length Max | 0.0422 |
| 6 | Bwd Packet Length Std | 0.0405 |
| 7 | Average Packet Size | 0.0397 |
| 8 | Total Length of Bwd Packets | 0.0379 |
| 9 | Fwd Packet Length Max | 0.0342 |
| 10 | Total Length of Fwd Packets | 0.0333 |

**Interprétation** : les 10 features les plus discriminantes concernent toutes la **distribution statistique des tailles de paquets** (variance, écart-type, moyenne, max). Cela reflète le fait que les attaques automatisées génèrent des paquets aux tailles quasi-identiques (faible variance), tandis que le trafic bénin présente une grande diversité de tailles.

**Notes méthodologiques** :
- Le papier ne précise pas les 58 features exactes qu'il a retenues, donc une comparaison directe n'est pas possible
- Random Forest ne gère pas les features fortement corrélées : plusieurs paires du top 10 mesurent des choses similaires (ex: `Variance` et `Std`). Le papier n'applique pas de pré-filtrage sur les corrélations, nous non plus.
- Dataset réduit sauvegardé : `data/processed/cicids2017_selected.pkl` (1.15 GB, 59 colonnes)

---

## Structure du projet

```
IDS-Adversarial-Defense/
├── src/                   Code source réutilisable
│   ├── data/              Chargement et preprocessing des données
│   ├── models/            Architectures DNN
│   ├── attacks/           Implémentations des attaques adversariales
│   ├── defenses/          Implémentations des mécanismes de défense
│   └── utils/             Fonctions utilitaires
├── notebooks/             Notebooks Jupyter pour l'exploration
├── scripts/               Scripts exécutables (numérotés par étape)
├── configs/               Fichiers de configuration YAML
├── data/                  Datasets (non versionnés)
│   ├── raw/               Fichiers CSV originaux
│   └── processed/         Données prétraitées (pickle)
├── results/               Sorties (logs, checkpoints, figures)
│   ├── logs/              Logs d'exécution
│   ├── checkpoints/       Modèles entraînés
│   └── figures/           Graphiques pour le rapport
└── tests/                 Tests unitaires
```

---

## Installation

Nécessite Python 3.12 sur Linux (ou Python 3.11 sur les serveurs Alliance Canada).

```bash
# Créer et activer l'environnement virtuel
python -m venv venv
source venv/bin/activate

# Installer PyTorch (version CPU pour le développement local)
pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Installer les autres dépendances
pip install -r requirements.txt

# Installer torchattacks sans ses dépendances
# (évite un conflit sur la version de la librairie requests)
pip install torchattacks==3.5.1 --no-deps
```

### Téléchargement du dataset

Le dataset CIC-IDS 2017 est téléchargé via Kaggle CLI :

```bash
# Configurer les credentials Kaggle (voir https://www.kaggle.com/docs/api)
mkdir -p ~/.kaggle
# Placer votre kaggle.json dans ~/.kaggle/ puis :
chmod 600 ~/.kaggle/kaggle.json

# Télécharger et décompresser
kaggle datasets download \
    -d chethuhn/network-intrusion-dataset \
    -p data/raw \
    --unzip
```

Le dataset fait environ 850 MB décompressé (8 fichiers CSV, 2.8 millions de lignes).

---

## Utilisation

Les scripts sont numérotés dans l'ordre d'exécution du pipeline.

### Scripts exécutables

```bash
# Vérifier l'environnement
python scripts/00_test_environment.py

# Explorer le dataset
python scripts/02_explore_dataset.py

# Preprocessing
python scripts/03_preprocess_dataset.py

# Feature selection (via SLURM sur Alliance Canada)
sbatch scripts/04_feature_selection.sh
```

### Scripts à venir

- `05_split_data.py` : séparation train/test + normalisation + SMOTE
- `06_train_baseline.py` : entraînement du DNN de référence
- `07_generate_attacks.py` : génération des 6 attaques adversariales
- `08_train_defenses.py` : entraînement des 4 mécanismes de défense
- `09_ensemble.py` : agrégation et optimisation de l'ensemble

---

## Attaques adversariales implémentées

Six attaques standards de la littérature :

| Attaque | Référence | Type | Norme |
|---|---|---|---|
| FGSM | Goodfellow et al., 2014 | Single-step | L∞ |
| BIM | Kurakin et al., 2016 | Iterative | L∞ |
| PGD | Madry et al., 2017 | Iterative | L∞ |
| DeepFool | Moosavi-Dezfooli et al., 2015 | Iterative | L2 |
| JSMA | Papernot et al., 2015 | Feature-based | L0 |
| C&W | Carlini & Wagner, 2016 | Optimization | L2 |

## Mécanismes de défense

Quatre défenses combinées dans un ensemble :

- **Adversarial Training (AT)** : entraîner le modèle sur des exemples adversariaux
- **Gaussian Augmentation (GA)** : ajout de bruit gaussien pendant l'entraînement
- **Label Smoothing (LS)** : lissage des labels pour éviter la sur-confiance
- **Denoising Autoencoder (DAE)** : autoencodeur qui nettoie les inputs avant classification

**Agrégation par ensemble** : Majority Voting et Weighted Average, tous deux optimisés par optimisation bayésienne.

---

## Datasets

- **CIC-IDS 2017** : https://www.unb.ca/cic/datasets/ids-2017.html
- **CIC-IDS 2018** : https://www.unb.ca/cic/datasets/ids-2018.html

Le CIC-IDS 2017 contient 2.8 millions de flux réseau étiquetés en 15 classes (14 attaques + trafic bénin), avec 78 features statistiques extraites par CICFlowMeter.

---

## Résultats de référence (issus du papier, CIC-IDS 2017)

| Configuration | Accuracy |
|---|---:|
| Baseline (données propres) | 98.11% |
| Baseline sous attaque C&W | 36.00% |
| Label Smoothing (seul) | 85.90% |
| Adversarial Training (seul) | 80.25% |
| Ensemble simple (Majority Voting) | 84.35% |
| **Ensemble optimisé (Majority Voting)** | **87.49%** |

---

## Environnement de développement

Le projet utilise une architecture hybride :

- **Développement local** : Arch Linux, Python 3.12, VSCode (édition du code, tests rapides)
- **Exécution intensive** : Alliance Canada (serveur nibi), Python 3.11, jobs SLURM
- **Synchronisation** : Git + GitHub (dépôt privé)

## Références principales

- Awad et al. (2025). *Enhanced Ensemble Defense Framework.* Scientific Reports.
- Goodfellow et al. (2014). *Explaining and Harnessing Adversarial Examples.* ICLR.
- Madry et al. (2017). *Towards Deep Learning Models Resistant to Adversarial Attacks.* ICLR.
- Carlini & Wagner (2016). *Towards Evaluating the Robustness of Neural Networks.* IEEE S&P.

---

## Licence

Projet académique. Utilisation à des fins de recherche uniquement.