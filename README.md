
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
| 3 | Feature selection | En cours |
| 4 | Split train / test | À faire |
| 5 | Baseline DNN | À faire |
| 6 | Attaques adversariales | À faire |
| 7 | Test de vulnérabilité | À faire |
| 8 | Mécanismes de défense | À faire |
| 9 | Agrégation par ensemble | À faire |

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

### 1. Vérifier l'environnement

```bash
python scripts/00_test_environment.py
```

### 2. Explorer le dataset

```bash
python scripts/02_explore_dataset.py
```

Analyse la distribution des classes, les valeurs manquantes et les problèmes de format. Génère un rapport dans `results/logs/`.

### 3. Preprocessing

```bash
python scripts/03_preprocess_dataset.py
```

Nettoie les données et sauvegarde le résultat dans `data/processed/cicids2017_clean.pkl`.

### Scripts à venir

- `04_feature_selection.py` : sélection des 58 features les plus importantes
- `05_split_data.py` : séparation train/test/validation
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
- **Exécution intensive** : Alliance Canada (serveur nibi), Python 3.11, entraînements GPU
- **Synchronisation** : Git + GitHub (dépôt privé)

## Références principales

- Awad et al. (2025). *Enhanced Ensemble Defense Framework.* Scientific Reports.
- Goodfellow et al. (2014). *Explaining and Harnessing Adversarial Examples.* ICLR.
- Madry et al. (2017). *Towards Deep Learning Models Resistant to Adversarial Attacks.* ICLR.
- Carlini & Wagner (2016). *Towards Evaluating the Robustness of Neural Networks.* IEEE S&P.

---

## Licence

Projet académique. Utilisation à des fins de recherche uniquement.