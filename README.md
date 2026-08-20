
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
| 4 | Split train/test + Normalisation + SMOTE | Terminée |
| 5 | Baseline DNN | Terminée |
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

### Étape 4 — Split train/test + Normalisation + SMOTE

Cette étape combine trois opérations dans un ordre méthodologiquement crucial pour éviter le data leakage.

**Ordre d'exécution** :
1. Split stratifié train/test (67/33)
2. Normalisation avec StandardScaler (fit sur train uniquement, transform sur train et test)
3. SMOTE sur le train uniquement (avec stratégie custom)

**Résultats du split** :

| Split | Nombre de lignes | Pourcentage |
|---|---:|---:|
| Train | 1 688 934 | 67.0% |
| Test | 831 864 | 33.0% |

Les 15 classes sont préservées dans les deux splits grâce à la stratification.

**Résultats de la normalisation** :
- Train : `mean = 0.0000`, `std = 1.0000` (par construction)
- Test : `mean = -0.0001`, `std = 1.1645` (écart normal, confirme l'absence de data leakage)

#### SMOTE : évolution de la stratégie

Deux versions de SMOTE ont été implémentées et comparées.

**Version initiale — équilibrage total (défaut d'imbalanced-learn)** :

Toutes les classes montées au niveau de la classe majoritaire (BENIGN à 1 403 688) :

| Classe | Avant | Après | Ratio d'expansion |
|---|---:|---:|---:|
| BENIGN | 1 403 688 | 1 403 688 | 1× |
| DoS Hulk | 115 807 | 1 403 688 | 12× |
| Heartbleed | 7 | 1 403 688 | **200 527×** |
| Web Attack SQL | 14 | 1 403 688 | 100 263× |

**Problèmes identifiés** :
- Dataset train de 21M lignes (train × 12)
- 19M exemples synthétiques créés
- Ratio d'expansion extrême sur classes ultra-rares (200 000×) → bruit
- Modèle sur-apprend les interpolations synthétiques

**Version finale — stratégie custom avec cap par classe** :

Chaque classe atteint un plafond raisonnable proportionnel au nombre d'exemples réels :

| Classe | Avant | Après | Ratio d'expansion |
|---|---:|---:|---:|
| BENIGN | 1 403 688 | 1 403 688 | 1× (inchangé) |
| DoS Hulk | 115 807 | 115 807 | 1× (inchangé) |
| DDoS | 85 769 | 85 769 | 1× (inchangé) |
| PortScan | 60 765 | 60 765 | 1× (inchangé) |
| DoS GoldenEye | 6 891 | 50 000 | 7× |
| FTP-Patator | 3 974 | 50 000 | 13× |
| DoS slowloris | 3 608 | 50 000 | 14× |
| DoS Slowhttptest | 3 503 | 50 000 | 14× |
| SSH-Patator | 2 157 | 30 000 | 14× |
| Bot | 1 305 | 30 000 | 23× |
| Web Attack Brute Force | 985 | 10 000 | 10× |
| Web Attack XSS | 437 | 5 000 | 11× |
| Infiltration | 24 | 1 000 | 42× |
| Web Attack SQL | 14 | 1 000 | 71× |
| Heartbleed | 7 | 500 | 71× |

**Résultats de la stratégie custom** :
- Train final : **1 943 529 lignes** (au lieu de 21M)
- **10× moins de données** que la version équilibrage total
- Ratios d'expansion plafonnés à 71× (au lieu de 200 000×)
- Représentation raisonnable des classes minoritaires sans sur-génération bruitée

**Livrables** (version finale) :
- `X_train.pkl` : 0.9 GB (features train après SMOTE custom et scaling)
- `X_test.pkl` : 375 MB (features test après scaling, sans SMOTE)
- `y_train.pkl` : ~15 MB (labels train)
- `y_test.pkl` : 19 MB (labels test)
- `scaler.pkl` : 3.3 KB (StandardScaler entraîné, sauvegardé pour usage futur)

**Notes méthodologiques critiques** :

1. **Ordre des opérations** : le papier ne détaille pas explicitement l'ordre exact (split, normalisation, SMOTE). Nous avons appliqué l'ordre méthodologiquement correct pour éviter le data leakage. Toute normalisation ou SMOTE appliquée avant le split contaminerait les statistiques du test dans le train.

2. **Le paramètre `n_jobs` de SMOTE** a été supprimé dans `imbalanced-learn` version 0.14+. Ne pas l'utiliser sur des versions récentes de la librairie.

3. **Stratégie SMOTE custom** : le papier applique SMOTE avec équilibrage total sans discuter les problèmes que cela pose sur les classes ultra-rares (Heartbleed avec 7 exemples réels → 1.4M synthétiques). Notre stratégie avec cap par classe est une amélioration méthodologique justifiée par la littérature qui recommande au minimum 100 exemples réels par classe pour SMOTE. Ce choix permet un meilleur compromis entre représentation des classes rares et fidélité statistique.

4. **Test std = 1.1645** : le fait que la standardisation du test ne donne pas exactement `std = 1.0` (comme le train) est **normal et souhaitable**. Une std exactement égale à 1.0 sur le test signalerait un data leakage. L'écart observé (16%) confirme l'indépendance des deux splits.

### Étape 5 — Entraînement du DNN baseline

Reproduction et amélioration itérative du DNN baseline décrit dans le papier.

**Architecture** :
- Input : 58 features → Dense(512) + ReLU → Dense(256) + ReLU → Dense(15)
- **165 391 paramètres**
- Softmax implicite via CrossEntropyLoss de PyTorch

**Hyperparamètres finaux** :

| Paramètre | Valeur | Choix |
|---|---|---|
| Learning rate initial | 0.001 | Ajusté depuis 0.01 du papier |
| Scheduler | ReduceLROnPlateau | factor=0.5, patience=5, min_lr=1e-5 |
| Optimizer | Adam | Comme le papier |
| Loss | CrossEntropyLoss | Comme le papier |
| Batch size | 128 | Comme le papier |
| Epochs | 50 | Étendu depuis 30 pour convergence |
| Random state | 42 | Reproductibilité |

**Environnement d'exécution** :
- Alliance Canada nibi
- GPU NVIDIA H100 80GB via SLURM
- Durée totale : environ 20 minutes

#### Itérations et raisonnement méthodologique

Trois versions successives du baseline ont été entraînées pour identifier la configuration optimale.

**Version 1 — Fidèle au papier (lr=0.01 fixe, 30 epochs)** :

Reproduction stricte des hyperparamètres du papier :
- **Accuracy : 90.69%** (papier : 98.11%)
- **F1 macro : 47.13%**
- **F1 weighted : 93.98%**

**Problème identifié** : instabilité de l'entraînement. Le modèle atteint son pic à l'epoch 2 (90.69%) puis se dégrade progressivement jusqu'à 68% à l'epoch 30. Le lr=0.01 est trop élevé pour un dataset de 21M lignes (5M itérations d'optimizer), causant une divergence des poids.

**Version 2 — Ajout d'un scheduler agressif (lr=0.01 initial, factor=0.1, patience=2, 30 epochs)** :

Introduction du ReduceLROnPlateau pour stabiliser l'entraînement.
- **Accuracy : 95.59%** (+4.90 points)
- **F1 macro : 44.66%** (légèrement moins bon car modèle plus conservateur)
- **F1 weighted : 97.05%**

**Problème identifié** : le scheduler trop agressif descend le lr jusqu'à 0.000000 (littéralement zéro) à partir de l'epoch 17. Le modèle est bloqué sans plus pouvoir apprendre.

**Version 3 — SMOTE custom + scheduler ajusté (lr=0.001 initial, factor=0.5, patience=5, min_lr=1e-5, 30 epochs)** :

Deux améliorations combinées : nouvelle stratégie SMOTE et scheduler moins agressif.
- **Accuracy : 99.60%** (+4.01 points, dépasse le papier)
- **F1 macro : 78.15%** (grosse amélioration sur classes rares)
- **F1 weighted : 99.65%**
- Durée : 12 minutes (vs 1h40 en v1/v2)

**Version 4 (finale) — Extension à 50 epochs** :

Après analyse de la courbe, le modèle n'avait pas encore convergé à l'epoch 30. Extension à 50 epochs pour identifier la vraie convergence.
- **Accuracy : 99.69%** (+0.09 points)
- **F1 macro : 80.17%** (+2.02 points, meilleure généralisation)
- **F1 weighted : 99.72%**
- Meilleur epoch : 49
- Convergence confirmée par stagnation sur les 5 dernières epochs

#### Résultats finaux (baseline v4)

**Métriques globales** :

| Métrique | Baseline v4 | Papier Awad et al. |
|---|---:|---:|
| Accuracy | **99.69%** | 98.11% |
| F1 macro | 80.17% | Non détaillé |
| F1 weighted | 99.72% | Non détaillé |

**Performance par classe (extrait)** :

| Classe | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| BENIGN | 99.93% | 99.74% | 99.84% | 691 369 |
| DDoS | 99.95% | 99.96% | 99.95% | 42 245 |
| DoS Hulk | 99.86% | 99.38% | 99.62% | 57 039 |
| DoS GoldenEye | 99.11% | 98.59% | 98.85% | 3 395 |
| DoS slowloris | 98.88% | 99.10% | 98.99% | 1 777 |
| DoS Slowhttptest | 91.57% | 99.48% | 95.36% | 1 725 |
| FTP-Patator | 99.85% | 99.59% | 99.72% | 1 957 |
| SSH-Patator | 90.21% | 98.96% | 94.39% | 1 062 |
| PortScan | 98.93% | 99.90% | 99.41% | 29 929 |
| Web Attack Brute Force | 68.03% | 96.08% | 79.66% | 485 |
| Bot | 36.01% | 94.87% | 52.20% | 643 |
| Heartbleed | 100.00% | 75.00% | 85.71% | 4 |
| Infiltration | 75.00% | 75.00% | 75.00% | 12 |
| Web Attack XSS | 73.33% | 5.12% | 9.57% | 215 |
| Web Attack SQL Injection | 14.29% | 14.29% | 14.29% | 7 |

**Interprétation** :
- Les grandes classes (BENIGN, DDoS, DoS Hulk, PortScan) atteignent > 99% F1
- Les classes intermédiaires (Bot, Web Attack Brute Force) sont raisonnablement bien classées
- **Web Attack XSS** : precision haute (73%) mais recall très bas (5%) — le modèle est très conservateur pour cette classe
- **Web Attack SQL** et **Heartbleed** : classes fondamentalement limitées par le nombre d'exemples réels dans le dataset (7 et 4 respectivement)

#### Notes méthodologiques importantes

1. **Écart avec le papier** : nous obtenons 99.69% contre 98.11% pour le papier. Cet écart favorable (+1.58 points) s'explique probablement par la stratégie SMOTE custom qui réduit le bruit d'interpolation, et par l'extension à 50 epochs.

2. **Le F1 macro reste bas (80%) malgré la haute accuracy** : cela révèle que certaines classes ultra-rares (Web Attack XSS, Web Attack SQL) restent difficiles à apprendre correctement. Le papier ne détaille pas de F1 macro, ce qui empêche la comparaison directe.

3. **Learning rate initial** : passé de 0.01 (papier) à 0.001 après analyse des courbes de la v1 qui montraient une instabilité claire. Le papier ne mentionne pas de scheduler, ce qui suggère qu'ils ont soit implicitement gardé le meilleur modèle (comportement observé en v1), soit utilisé un dataset plus petit sans SMOTE massif.

4. **Extension à 50 epochs** : décision méthodologique justifiée par l'observation que le modèle progressait encore à l'epoch 30. La convergence est prouvée par la stagnation de test_acc à 99.69% sur les 5 dernières epochs.

### Étape 6 — Génération des attaques adversariales

Six attaques adversariales ont été implémentées et évaluées sur le baseline v4 :

**Résultats obtenus** (test complet, 831 864 échantillons) :

| Attaque | Accuracy | F1 macro | F1 weighted | Chute vs baseline |
|---|---:|---:|---:|---|
| Baseline (données propres) | 99.69% | 80.17% | 99.72% | référence |
| FGSM | 83.25% | 6.34% | 75.88% | -16.4 pts |
| BIM | 72.10% | 5.61% | 69.86% | -27.6 pts |
| PGD | 78.48% | 5.86% | 73.09% | -21.2 pts |
| DeepFool | 16.71% | 2.86% | 25.14% | **-82.9 pts** |
| JSMA | En cours | — | — | — |
| C&W | En attente | — | — | — |

**Observations clés** :

- **DeepFool** est l'attaque la plus dévastatrice (accuracy chute à 16.71%), cohérent avec la littérature
- Les attaques L∞ (FGSM, BIM, PGD) sont efficaces sur les classes minoritaires mais moins que sur les majoritaires
- Le F1 macro chute drastiquement (~6% vs 80%) : les classes minoritaires sont presque totalement échouées sous toutes les attaques

**Défis techniques rencontrés** :

**1. Bug de compatibilité NumPy 2.x avec ART 1.18**

`SaliencyMapMethod` (JSMA) utilise `np.product` qui a été supprimé dans NumPy 2.x. Correction appliquée via monkey-patch :

```python
if not hasattr(np, 'product'):
    np.product = np.prod
```

Le patch doit être appliqué **avant** l'import d'ART.

**2. Coût computationnel prohibitif de JSMA**

L'implémentation JSMA de ART calcule pour chaque échantillon un jacobien complet (58 features × 15 classes = 870 dérivées partielles), puis modifie itérativement 2 features à la fois jusqu'à faire changer la classification.

**Mesure empirique** : sur H100, JSMA traite environ **5.5 batches/heure** (batch_size=512), soit 2 800 échantillons/heure.

**Extrapolation** : pour 831 864 échantillons, temps estimé = **12.4 jours** de calcul GPU continu.

**Contrainte SLURM** : la partition GPU maximale de nibi (`gpubase_bygpu_b5`) permet jusqu'à 7 jours par job. Il est donc **techniquement impossible** de compléter JSMA sur le full test set en un seul job.

**Choix méthodologique** :

Un job de 7 jours (168h) a été lancé pour obtenir une preuve empirique de cette limitation, atteignant environ 57% de progression avant timeout. Suite à cela, JSMA sera appliqué sur un **échantillon stratifié** de taille réduite, pratique standard dans la littérature adversariale pour les attaques computationnellement coûteuses.

**Note sur le papier Awad et al.** : le papier ne précise pas la méthodologie exacte d'application de JSMA sur leur dataset. Notre limitation est probablement partagée par les auteurs (utilisation implicite d'échantillonnage ou d'une implémentation optimisée non publique).

**3. Sauvegarde des résultats**

ART ne fait pas de checkpointing pendant l'exécution de `attack.generate()`. Si le job SLURM est terminé avant complétion, les résultats intermédiaires sont perdus. Cette limitation empêche de "reprendre" une attaque partielle avec ART.

Une solution alternative (implémentation custom avec checkpoints périodiques) serait envisageable mais dépasse le cadre de ce projet de reproduction.

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
pip install torchattacks==3.5.1 --no-deps
```

### Téléchargement du dataset

```bash
mkdir -p ~/.kaggle
# Placer votre kaggle.json dans ~/.kaggle/ puis :
chmod 600 ~/.kaggle/kaggle.json

kaggle datasets download \
    -d chethuhn/network-intrusion-dataset \
    -p data/raw \
    --unzip
```

---

## Utilisation

Les scripts sont numérotés dans l'ordre d'exécution du pipeline.

```bash
# Vérification de l'environnement
python scripts/00_test_environment.py

# Exploration
python scripts/02_explore_dataset.py

# Preprocessing
python scripts/03_preprocess_dataset.py

# Feature selection (via SLURM sur Alliance Canada)
sbatch scripts/04_feature_selection.sh

# Split + Normalisation + SMOTE custom (via SLURM)
sbatch scripts/05_split_and_prepare.sh

# Entraînement du DNN baseline (via SLURM avec GPU H100)
sbatch scripts/06_train_baseline.sh

# Génération des graphiques (post-entraînement)
python scripts/07_plot_results.py
```

### Scripts à venir

- `08_generate_attacks.py` : génération des 6 attaques adversariales
- `09_test_vulnerability.py` : évaluation de la vulnérabilité du baseline
- `10_train_defenses.py` : entraînement des 4 mécanismes de défense
- `11_ensemble.py` : agrégation et optimisation de l'ensemble

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

## Nos résultats actuels (baseline v4, CIC-IDS 2017)

| Métrique | Notre baseline | Papier | Écart |
|---|---:|---:|---:|
| **Accuracy** | **99.69%** | 98.11% | +1.58 |
| F1 weighted | 99.72% | Non détaillé | — |
| F1 macro | 80.17% | Non détaillé | — |

---

## Environnement de développement

Le projet utilise une architecture hybride :

- **Développement local** : Arch Linux, Python 3.12, VSCode
- **Exécution intensive** : Alliance Canada (serveur nibi), Python 3.11, jobs SLURM
- **Synchronisation** : Git + GitHub (dépôt privé)

## Références principales

- Awad et al. (2025). *Enhanced Ensemble Defense Framework.* Scientific Reports.
- Goodfellow et al. (2014). *Explaining and Harnessing Adversarial Examples.* ICLR.
- Madry et al. (2017). *Towards Deep Learning Models Resistant to Adversarial Attacks.* ICLR.
- Carlini & Wagner (2016). *Towards Evaluating the Robustness of Neural Networks.* IEEE S&P.
- Chawla et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* JAIR.

---

## Licence

Projet académique. Utilisation à des fins de recherche uniquement.