
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
| 4 | Split train/val/test + Normalisation + SMOTE | **Régénérée le 5 sept. 2026** (MinMaxScaler, vrai split validation) |
| 5 | Baseline DNN | **Réentraînée le 5 sept. 2026** (v5, 100 epochs, sélection sur validation) |
| 6 | Attaques adversariales | Réécrite (substitut, clip_values) — **régénération en cours** |
| 7 | Test de vulnérabilité | Intégrée à l'étape 6, dépend de sa régénération |
| 8 | Mécanismes de défense | Scripts prêts, dépendent de l'étape 6 |
| 9 | Agrégation par ensemble | Script prêt, dépend de l'étape 8 |

> **Les résultats d'attaques (étapes 6-7) documentés dans ce fichier viennent
> encore du pipeline précédent** (white box sans substitut, sans `clip_values`).
> Ils sont marqués comme provisoires et seront remplacés dès que
> `08_generate_attacks.py` aura tourné sur le pipeline corrigé. Les résultats
> de baseline (étapes 4-5), eux, sont à jour.

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

**Note technique** : les libellés de classes issus des CSV contiennent un caractère
mal encodé dans les trois classes `Web Attack – *` (tiret cadratin lu comme
caractère de remplacement). Les scripts de tracé le nettoient à l'affichage
plutôt que de modifier le `label_encoder`, qui doit rester identique à celui
utilisé pour l'entraînement.

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

### Étape 4 — Split train/val/test + Normalisation + SMOTE

Cette étape a connu deux versions. La première (août 2026) utilisait
`StandardScaler` et ne produisait qu'un split train/test. La seconde
(5 septembre 2026) corrige les deux points ; voir « Correctifs
méthodologiques » pour le raisonnement.

#### Version actuelle

**Ordre d'exécution** (crucial pour éviter le data leakage) :
1. Split stratifié train/test, puis train/validation prélevé sur le reste
2. Normalisation `MinMaxScaler` (fit sur train uniquement, transform sur les trois)
3. Bornage explicite de val et test dans [0,1]
4. SMOTE sur le train uniquement, avec stratégie custom (cap par classe)

**Résultats du split** :

| Split | Nombre de lignes | Pourcentage |
|---|---:|---:|
| Train | 1 562 894 | 62.0% |
| Validation | 126 040 | 5.0% |
| Test | 831 864 | 33.0% |

`val_size` est fixé à 0.05 et non 0.15 : une sélection d'epoch n'a pas besoin de
378 000 exemples de validation, et chaque point retiré au train pèse sur des
classes déjà ultra-rares (Heartbleed n'a que 6 exemples dans le train).

**Débordement du domaine après normalisation** :

`MinMaxScaler.transform()` ne garantit pas que val et test tombent dans
l'intervalle appris sur le train. Le script compte et rapporte les valeurs
hors domaine avant de les borner :

| Split | Valeurs hors [0,1] | Proportion | Dépassement max |
|---|---:|---:|---:|
| Train | 1 / 90 647 852 | 0.0000% | 0.0000 |
| Validation | 2 / 7 310 320 | 0.0000% | 0.0377 |
| Test | 55 / 48 248 112 | 0.0001% | 4.5568 |

**Interprétation** : le dépassement est réel en amplitude (jusqu'à 4.56 sur le
test, donc bien au-delà d'un « léger débordement ») mais négligeable en
prévalence — 55 valeurs sur 48 millions. Le bornage reste nécessaire pour
garantir que la référence propre et les échantillons adversariaux portent sur
le même domaine d'entrée, mais il ne change pas les résultats de façon
mesurable. C'est le passage à `MinMaxScaler` lui-même qui compte, pas le clip.

#### SMOTE : évolution de la stratégie

Deux versions ont été implémentées et comparées.

**Version initiale — équilibrage total (défaut d'imbalanced-learn)** :

Toutes les classes montées au niveau de la majoritaire (BENIGN) :

| Classe | Avant | Après | Ratio d'expansion |
|---|---:|---:|---:|
| BENIGN | 1 403 688 | 1 403 688 | 1× |
| DoS Hulk | 115 807 | 1 403 688 | 12× |
| Heartbleed | 7 | 1 403 688 | **200 527×** |
| Web Attack SQL | 14 | 1 403 688 | 100 263× |

**Problèmes identifiés** : train de 21M lignes, 19M exemples synthétiques,
ratios d'expansion extrêmes sur les classes ultra-rares, modèle qui
sur-apprend les interpolations.

**Version retenue — cap par classe** :

Les plafonds vivent maintenant dans `configs/config.yaml` →
`dataset.smote_strategy`, plus en dur dans le script. Effectifs du split
actuel :

| Classe | Avant | Après | Ratio |
|---|---:|---:|---:|
| BENIGN | 1 298 935 | 1 298 935 | 1× (inchangé) |
| DoS Hulk | 107 165 | 107 165 | 1× (inchangé) |
| DDoS | 79 368 | 79 368 | 1× (inchangé) |
| PortScan | 56 230 | 56 230 | 1× (inchangé) |
| DoS GoldenEye | 6 377 | 50 000 | 8× |
| FTP-Patator | 3 677 | 50 000 | 14× |
| DoS slowloris | 3 339 | 50 000 | 15× |
| DoS Slowhttptest | 3 242 | 50 000 | 15× |
| SSH-Patator | 1 996 | 30 000 | 15× |
| Bot | 1 208 | 30 000 | 25× |
| Web Attack Brute Force | 912 | 10 000 | 11× |
| Web Attack XSS | 404 | 5 000 | 12× |
| Infiltration | 22 | 1 000 | 45× |
| Web Attack SQL Injection | 13 | 1 000 | 77× |
| Heartbleed | 6 | 500 | 83× |

**Résultat** : train final de **1 819 198 lignes**, soit 256 304 exemples
synthétiques ajoutés (expansion globale 1.16×), contre 21M pour l'équilibrage
total.

**Point de vigilance** : `k_neighbors` vaut 5, et Heartbleed n'a que 6 exemples
réels dans le train — soit exactement le minimum requis par SMOTE
(`k_neighbors + 1`). Un exemple de moins et l'étape échouerait. Le script
prévoit un repli (filtrage des classes à moins de 2 exemples, avertissement
explicite), mais la marge est d'un seul échantillon.

**Livrables** :

| Fichier | Taille |
|---|---:|
| `X_train.pkl` | 805.0 MB |
| `X_val.pkl` | 56.7 MB |
| `X_test.pkl` | 374.5 MB |
| `y_train.pkl` | 13.9 MB |
| `y_val.pkl` | 1.0 MB |
| `y_test.pkl` | 6.3 MB |
| `scaler.pkl` | 4.3 KB |
| `data_fingerprint.json` | empreinte de configuration |

**Notes méthodologiques** :

1. **Ordre des opérations** : le papier ne détaille pas l'ordre exact. Nous
   appliquons l'ordre méthodologiquement correct — toute normalisation ou
   SMOTE appliquée avant le split contaminerait les statistiques du test.

2. **Le paramètre `n_jobs` de SMOTE** a été supprimé dans `imbalanced-learn`
   0.14+. Ne pas l'utiliser sur les versions récentes.

3. **Stratégie SMOTE custom** : le papier équilibre totalement sans discuter
   le problème des classes ultra-rares. Le cap par classe est une déviation
   assumée, justifiée par la littérature qui recommande au minimum 100
   exemples réels par classe pour SMOTE. C'est probablement la principale
   cause de l'écart d'accuracy avec le papier.

4. **Reproductibilité inter-cluster** : le script a été exécuté sur nibi et
   narval à partir du même `cicids2017_selected.pkl`, et produit exactement
   les mêmes effectifs, les mêmes comptages hors domaine et la même empreinte
   de configuration (`cd8cbf56a4206ed7`). Cela valide le déterminisme du code
   et de la configuration entre machines, pas celui du pipeline en amont
   (étapes 3 et 4), qui n'a pas été rejoué indépendamment.

### Étape 5 — Entraînement du DNN baseline

**Architecture** (identique dans toutes les versions) :
- Input : 58 features → Dense(512) + ReLU → Dense(256) + ReLU → Dense(15)
- **165 391 paramètres**
- Softmax implicite via `CrossEntropyLoss` de PyTorch

#### Cheminement : cinq versions successives

Le baseline a été entraîné cinq fois. Les quatre premières explorent les
hyperparamètres sous le pipeline initial ; la cinquième reprend tout sous le
pipeline corrigé.

**v1 — Fidèle au papier** (`baseline_v1_lr_fixe.pth`)
`lr=0.01` fixe, 30 epochs, StandardScaler.
Accuracy 90.69 %, F1 macro 47.13 %.
*Problème* : instabilité franche, pic à l'epoch 2 puis dégradation. Le
learning rate de la Table 3 est trop élevé pour cette architecture sur ce
jeu de données.

**v2 — Scheduler agressif** (`baseline_v2_scheduler.pth`)
`lr=0.01`, ReduceLROnPlateau avec `factor=0.1, patience=2`.
Accuracy 95.59 %, F1 macro 44.66 %.
*Problème* : le scheduler compense mal un lr initial trop grand et finit par
le faire descendre jusqu'à 0.000000. L'accuracy monte mais le F1 macro se
dégrade encore : les classes minoritaires sont sacrifiées.

**v3 — SMOTE custom + scheduler ajusté** (`baseline_v3_final.pth`)
`lr=0.001`, `factor=0.5, patience=5`, SMOTE avec cap par classe.
Accuracy 99.60 %, F1 macro 78.15 %. Durée réduite à 12 minutes.
*Enseignement* : c'est le passage de l'équilibrage total au cap par classe qui
débloque le F1 macro (+33 points), pas le learning rate seul.

**v4 — Extension à 50 epochs** (`baseline_v4_final.pth`)
Mêmes hyperparamètres que v3, 50 epochs.
Accuracy 99.69 %, F1 macro 80.17 %, F1 weighted 99.72 %.
*Limite découverte plus tard* : le meilleur epoch était sélectionné sur
l'accuracy du **test set**, qui servait donc à la fois à choisir le modèle et
à l'évaluer. Le chiffre de 99.69 % était donc optimiste par construction.

**v5 — Pipeline corrigé** (`baseline_best.pth`)
MinMaxScaler, vrai split de validation, sélection sur `val_acc`, 100 epochs.
Accuracy 99.79 %, F1 macro 84.11 %, F1 weighted 99.79 %.

> **Nomenclature** : les checkpoints v1 à v4 sont conservés dans
> `results/checkpoints/` à titre de trace des itérations. Le modèle courant
> est `baseline_best.pth`, écrit et revalidé par empreinte de configuration ;
> il n'y a pas de fichier `baseline_v5_*.pth`, la numérotation par version
> ayant été remplacée par le mécanisme d'empreintes.

#### v5 en détail

**Hyperparamètres** :

| Paramètre | Valeur | Choix |
|---|---|---|
| Learning rate initial | 0.001 | Ajusté depuis 0.01 du papier (v1/v2) |
| Scheduler | ReduceLROnPlateau | factor=0.5, patience=5, min_lr=1e-5, sur `val_loss` |
| Optimizer | Adam | Comme le papier |
| Loss | CrossEntropyLoss | Comme le papier |
| Batch size | 128 | Comme le papier |
| Epochs | 100 | Étendu depuis 50 (voir ci-dessous) |
| Sélection du modèle | `val_acc` | Sur validation, jamais sur test |
| Random state | 42 | Reproductibilité |

**Environnement d'exécution** : Alliance Canada narval, GPU NVIDIA A100-SXM4-40GB
via SLURM. Durée totale : **37 minutes** (21.8 s/epoch en moyenne).

**Pourquoi 100 epochs.** Un premier réentraînement du pipeline corrigé à
50 epochs donnait 99.689 % d'accuracy et 81.83 % de F1 macro. Les courbes
montraient que `val_loss` descendait encore et que `val_acc` progressait
toujours à l'epoch 50 : le modèle n'avait pas convergé. La « convergence »
annoncée pour v4 reposait sur la stagnation de `test_acc`, une métrique qui
n'aurait pas dû être surveillée.

À 100 epochs, le scheduler se déclenche quatre fois :

| Epoch | Learning rate |
|---:|---:|
| 1 → 32 | 1.0e-3 |
| 33 → 62 | 5.0e-4 |
| 64 → 70 | 2.5e-4 |
| 71 → 91 | 1.25e-4 |
| 92 → 100 | 6.3e-5 |

Le plateau est atteint vers l'epoch 81. Entre les epochs 92 et 100,
`train_loss` reste bloqué à 0.0113-0.0114 et `val_acc` oscille entre 0.9976 et
0.9979 sans tendance. Le meilleur epoch est le 98 (`val_acc` = 0.9979).
Aucun surapprentissage : `val_loss` finit à 0.0093 sans jamais remonter, et la
courbe de validation reste sous celle d'entraînement du début à la fin.

**Gain apporté par le passage de 50 à 100 epochs** :

| Métrique | 50 epochs | 100 epochs | Écart |
|---|---:|---:|---:|
| Accuracy | 99.689% | **99.789%** | +0.10 |
| F1 macro | 81.83% | **84.11%** | +2.28 |
| F1 weighted | 99.694% | **99.788%** | +0.09 |

#### Résultats finaux (baseline v5)

| Métrique | Baseline v5 | Papier Awad et al. |
|---|---:|---:|
| Accuracy | **99.79%** | 98.11% |
| F1 macro | 84.11% | Non détaillé |
| F1 weighted | 99.79% | Non détaillé |

**Performance par classe** :

| Classe | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| BENIGN | 99.94% | 99.84% | 99.89% | 691 369 |
| DDoS | 99.93% | 99.98% | 99.96% | 42 245 |
| DoS Hulk | 99.65% | 99.44% | 99.55% | 57 039 |
| PortScan | 98.91% | 99.90% | 99.40% | 29 929 |
| FTP-Patator | 99.95% | 99.74% | 99.85% | 1 957 |
| DoS slowloris | 99.44% | 99.10% | 99.27% | 1 777 |
| DoS GoldenEye | 99.26% | 98.73% | 99.00% | 3 395 |
| DoS Slowhttptest | 97.78% | 99.54% | 98.65% | 1 725 |
| SSH-Patator | 94.61% | 99.15% | 96.83% | 1 062 |
| Web Attack Brute Force | 69.21% | 99.18% | 81.53% | 485 |
| Bot | 60.25% | 97.82% | 74.57% | 643 |
| Infiltration | 64.29% | 75.00% | 69.23% | 12 |
| Web Attack SQL Injection | 23.53% | 57.14% | 33.33% | 7 |
| Web Attack XSS | 44.83% | 6.05% | 10.66% | 215 |
| Heartbleed | 100.00% | 100.00% | 100.00% | 4 |

**Évolution du F1 macro à travers les versions** :

| Version | Accuracy | F1 macro | Changement principal |
|---|---:|---:|---|
| v1 | 90.69% | 47.13% | Fidèle au papier (lr=0.01, 30 ep) |
| v2 | 95.59% | 44.66% | Scheduler agressif |
| v3 | 99.60% | 78.15% | SMOTE custom + lr=0.001 |
| v4 | 99.69% | 80.17% | 50 epochs |
| **v5** | **99.79%** | **84.11%** | Pipeline corrigé + 100 epochs |

**Notes méthodologiques** :

1. **Écart avec le papier** : 99.79 % contre 98.11 %, soit +1.68 point. Cet
   écart s'explique principalement par la stratégie SMOTE custom, qui réduit
   le bruit d'interpolation par rapport à l'équilibrage total du papier, et
   par l'extension à 100 epochs. Ce chiffre est maintenant obtenu sans que le
   test set ait servi à la sélection du modèle, contrairement à v4.

2. **Le biais de sélection de v4 était négligeable en pratique** : v4 donnait
   99.69 % avec sélection sur test, et le pipeline corrigé donne 99.689 % à
   nombre d'epochs égal. La correction était nécessaire pour la validité du
   protocole, pas parce qu'elle changeait le résultat.

3. **Heartbleed à 100 % n'a aucune signification statistique** : 4 échantillons
   dans le test. Idem pour SQL Injection (7) et Infiltration (12). Les figures
   affichent le support entre parenthèses pour éviter cette lecture erronée.

4. **Web Attack XSS reste la classe la plus problématique** (F1 10.66 %). La
   matrice de confusion montre que 93 % de ses échantillons sont classés en
   `Web Attack - Brute Force`, et non en trafic bénin : le modèle détecte bien
   une attaque web mais ne discrimine pas entre les deux. Les 58 features de
   flux ne capturent probablement pas ce qui les distingue — c'est une limite
   du jeu de données, pas du modèle.

5. **Bot a une précision faible pour un rappel élevé** (60 % / 98 %) : le
   modèle sur-prédit cette classe. Effet plausible de l'expansion SMOTE de
   1 208 à 30 000 exemples.

**Figures produites** (`results/figures/final_baseline_training_20260905_154435_*`) :
courbes d'apprentissage, évolution du learning rate, métriques globales,
F1 par classe avec support, matrice de confusion normalisée avec support.

### Étape 6 — Génération des attaques adversariales

**Régénération en cours.** Le script `08_generate_attacks.py` a été réécrit
(modèle substitut, `clip_values`, périmètre unifié, paramètres lus depuis la
configuration). Les résultats ci-dessous viennent du **pipeline précédent** et
ne sont plus représentatifs.

#### Résultats provisoires (ancien pipeline, white box, StandardScaler, baseline v4)

| Attaque | Accuracy | F1 macro | F1 weighted | Chute vs baseline |
|---|---:|---:|---:|---:|
| Baseline (clean) | 99.69% | 80.17% | 99.72% | — |
| FGSM | 83.25% | 6.34% | 75.88% | -16.4 pts |
| BIM | 72.10% | 5.61% | 69.86% | -27.6 pts |
| PGD | 78.48% | 5.86% | 73.09% | -21.2 pts |
| DeepFool | 16.71% | 2.86% | 25.14% | **-82.9 pts** |
| JSMA | 83.45% | 17.96% | 81.91% | -16.2 pts |
| C&W | 67.15% | 5.37% | 66.93% | -32.5 pts |

Ces six mesures portent bien sur le test set complet (831 864 échantillons)
pour les six attaques — vérifié dans `attacks_results_20260827_135218.pkl`.

**Ce qui devrait changer après régénération** : en semi-white box, l'attaquant
ne dispose plus des gradients du baseline mais de ceux d'un substitut. Les
attaques devraient être moins destructrices et donc plus proches des chiffres
du papier. La chute extrême de DeepFool (16.71 %) était très probablement un
artefact du white box combiné à l'absence de bornes sur les attaques ART.

#### Défis techniques rencontrés et résolus

**1. Bug de compatibilité NumPy 2.x avec ART 1.18**

`SaliencyMapMethod` (JSMA) utilise `np.product`, supprimé dans NumPy 2.x.
Correction par monkey-patch, appliquée **avant** l'import d'ART :

```python
if not hasattr(np, 'product'):
    np.product = np.prod
```

**2. Coût computationnel de JSMA**

Avec `theta=0.1, gamma=1.0` (Table 2 de l'article), le débit mesuré était de
5.5 batches/heure sur H100, soit environ 12 jours pour le test set complet.
Le détour historique par `theta=0.3, gamma=0.15` réglait le temps de calcul
mais s'écartait de l'article. La configuration est revenue aux valeurs de la
Table 2, et le périmètre d'évaluation est passé à un échantillon stratifié
partagé par les six attaques. `08_generate_attacks.py` imprime une estimation
de durée sur échantillon stratifié avant de lancer JSMA en grandeur réelle.

**3. Mode targeted vs untargeted**

Les implémentations ART de JSMA et C&W sont targeted par défaut. Passer les
vrais labels comme cible rend l'attaque triviale — le modèle prédit déjà ces
classes correctement. Correction : `y=None` pour forcer l'untargeted, ce qui
correspond à l'hypothèse annoncée dans l'article.

**4. Volume des exemples adversariaux**

Chaque `X_adv_*.pkl` fait ~185 MB sur le test complet (831 864 × 58 × float32),
soit ~1.1 GB pour les six. Stockage dans `results/attacks/`, non versionné.

### Étape 7 — Test de vulnérabilité du baseline

Intégrée à l'étape 6 : chaque attaque générée est immédiatement évaluée par
transfert sur le baseline. Dépend de la régénération de l'étape 6.

### Étape 8 — Mécanismes de défense

**Scripts prêts, en attente de l'étape 6.**

| Script | Défense | Approche | Hyperparamètres |
|---|---|---|---|
| `11_defense_adversarial_training.py` | Adversarial Training | FGSM à la volée, mix 50/50 | eps=0.05, ratio=0.5 |
| `12_defense_label_smoothing.py` | Label Smoothing | Cross-entropy adoucie | alpha=0.1 |
| `13_defense_gaussian_augmentation.py` | Gaussian Augmentation | Bruit gaussien à l'entraînement | sigma=0.1 |
| `14_defense_denoising_autoencoder.py` | Denoising Autoencoder | Autoencodeur 58→32→58 en amont | sigma=0.1, L1=1e-4 |

**Choix communs** : même architecture que le baseline (512→256), mêmes
hyperparamètres d'entraînement, évaluation intégrée (clean + 6 attaques dans
un seul run), checkpoints séparés.

### Étape 9 — Agrégation par ensemble

**Script prêt.** Combine les 4 défenses via 3 méthodes :

| Méthode | Description |
|---|---|
| Majority Voting | Chaque défense vote, classe majoritaire retenue |
| Weighted Average (égal) | Moyenne des softmax, poids 0.25 chacun |
| Weighted Average (optimisé) | Poids optimisés par Nelder-Mead |

**Déviation par rapport au papier** : il utilise scikit-optimize (optimisation
bayésienne), nous utilisons Nelder-Mead de scipy — équivalent sur un problème
à 4 dimensions, sans dépendance externe à installer sur les clusters.

---

## Correctifs méthodologiques du 3 septembre 2026

Une relecture du pipeline a mis au jour plusieurs écarts avec le protocole de
l'article, indépendants des choix méthodologiques déjà documentés. Ces écarts
n'avaient pas été détectés parce que chaque script redéfinissait ses propres
constantes au lieu de lire `configs/config.yaml` — deux sources de vérité qui
ont divergé sans que ce soit visible.

**1. Absence de modèle substitut (l'écart le plus important).** Les six
attaques étaient générées directement sur le baseline
(`torchattacks.FGSM(model, ...)`, `PyTorchClassifier(model=model)`),
c'est-à-dire en white box complet : l'attaquant disposait des vrais gradients
du modèle qu'il attaque. L'article place l'attaquant en semi-white box via un
substitut (58 → 100 → 100 → 15, entraîné séparément, 17 515 paramètres). Le
substitut est implémenté dans `src/models/substitute.py` et sert désormais de
source aux attaques ; le baseline ne fait plus qu'évaluer la transférabilité
des exemples générés.

**2. `clip_values` non défini sur le classifieur ART.** FGSM, BIM et PGD (via
`torchattacks`) étaient bornées dans [0,1] par un clamp interne à la
bibliothèque ; DeepFool, JSMA et C&W (via ART) ne l'étaient pas.
`create_art_classifier` passe maintenant `clip_values=(0.0, 1.0)`.

**3. `StandardScaler` au lieu de `MinMaxScaler`.** Avec des features
centrées-réduites, donc en partie négatives, le clamp interne de
`torchattacks` (`torch.clamp(x, 0, 1)`) écrasait à zéro toutes les valeurs
négatives de l'échantillon lui-même, pas seulement de la perturbation — et
seulement pour FGSM, BIM et PGD. C'est probablement ce qui expliquait
l'écart de comportement entre les attaques L∞ et DeepFool. L'article ramène
les features dans [0,1] ; `05_split_and_prepare.py` utilise maintenant
`MinMaxScaler`, cohérent avec `clip_values`.

**4. Sélection du meilleur epoch sur le test set.** `06_train_baseline.py`
choisissait le checkpoint et pilotait le scheduler sur l'accuracy du test, qui
servait donc à la fois à choisir le modèle et à l'évaluer. Un vrai split de
validation existe maintenant ; le test n'est touché qu'une fois, après
l'entraînement, avec le modèle déjà figé. *Constat après régénération : le
biais était négligeable — 99.69 % pour v4, 99.689 % pour le pipeline corrigé à
nombre d'epochs égal. La correction reste nécessaire pour la validité du
protocole.*

**5. Paramètres d'attaque divergents de la Table 2.** JSMA utilisait
`theta=0.3` (Table 2 : 0.1) et C&W `max_iter=10` (Table 2 : 9), les autres
hyperparamètres de C&W étant laissés aux défauts d'ART. Cause racine :
`configs/config.yaml` contenait déjà les bonnes valeurs mais n'était lu par
aucun script de calcul. `src/utils/config.py` centralise le chargement et
valide la configuration avant tout calcul.

**6. Doublon entre les scripts 08 et 09.** `09_generate_attacks_jsma_sample.py`
générait JSMA sur un échantillon de 30 000 et C&W sur le test complet, en
white box et sans `clip_values`. Ses sorties (`*_sample30k.pkl`) n'étaient
consommées par aucun autre script — vérifié : `10` et `11-14` ne chargent que
les fichiers produits par `08`. Le script est désactivé (il refuse de
s'exécuter) et `08` couvre les six attaques avec un périmètre unique, piloté
par `evaluation.scope`.

> *Correction apportée à ce point le 5 septembre : une version antérieure de
> ce README affirmait que le tableau de résultats mélangeait deux périmètres
> d'évaluation. C'est faux. La relecture des logs
> (`attacks_results_20260827_135218.pkl`) confirme que les six attaques
> portaient bien sur les 831 864 échantillons du test complet. Le run sur
> 30 000 existait mais ses chiffres n'ont jamais été présentés.*

**7. `SMOTE_STRATEGY` en dur, et un bug de `k_neighbors`.** Les plafonds par
classe vivaient en dur dans `05_split_and_prepare.py` et pouvaient changer
sans qu'aucune empreinte ne bouge ; ils sont maintenant dans la configuration.
Au passage, `k_neighbors` se calculait sur la classe la plus petite de tout
`y_train`, y compris des classes non concernées par SMOTE — une classe
ultra-rare hors stratégie faisait chuter `k_neighbors` pour toutes les autres.
Il se calcule maintenant uniquement sur les classes réellement
suréchantillonnées.

### Garde-fou : empreintes de configuration

`src/utils/config.py` calcule une empreinte pour les données (`05`), le
baseline (`06`) et le substitut / les attaques (`08`). Chaque artefact mis en
cache porte l'empreinte sous laquelle il a été produit, et est invalidé si la
configuration a changé depuis — avec un message indiquant quelle clé diffère,
plutôt que deux hashes opaques.

L'empreinte des données est écrite dans `data/processed/data_fingerprint.json`
à la fin de `05`, et revalidée en tête de `06` et `08`. Sans elle, modifier
`val_size` sans relancer `05` produirait un checkpoint cohérent avec sa propre
empreinte mais entraîné sur des données périmées — des chiffres faux sous une
étiquette juste.

Politique par artefact : le baseline lève une erreur bloquante (le réentraîner
est le rôle de `06`), le substitut est réentraîné automatiquement (cache bon
marché), les `X_adv_*.pkl` sont régénérés.

Ce mécanisme remplace la numérotation manuelle des versions de checkpoints
(`baseline_v1` à `v4`) : l'identité d'un modèle est désormais portée par
l'empreinte de la configuration qui l'a produit, pas par un nom de fichier.

---

## Structure du projet

```
IDS-Adversarial-Defense/
├── src/                   Code source réutilisable
│   ├── data/              Chargement et preprocessing
│   ├── models/            dnn.py (baseline), substitute.py
│   ├── attacks/           Implémentations des attaques
│   ├── defenses/          Implémentations des défenses
│   └── utils/             config.py (chargement + empreintes)
├── notebooks/             Exploration
├── scripts/               Scripts exécutables (numérotés par étape)
├── configs/               config.yaml — source unique de vérité
├── data/                  Datasets (non versionnés, lien vers scratch)
│   ├── raw/               CSV originaux
│   └── processed/         Données prétraitées + data_fingerprint.json
├── results/
│   ├── logs/              Logs d'exécution
│   ├── checkpoints/       Modèles entraînés (v1-v4 archivés, baseline_best courant)
│   ├── attacks/           Exemples adversariaux
│   └── figures/           Graphiques
└── tests/                 Tests unitaires
```

---

## Installation

Nécessite Python 3.12 sur Linux (ou Python 3.11 sur les serveurs Alliance Canada).

```bash
python -m venv venv
source venv/bin/activate

pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
pip install torchattacks==3.5.1 --no-deps
pip install adversarial-robustness-toolbox --no-deps
```

### Téléchargement du dataset

```bash
mkdir -p ~/.kaggle
chmod 600 ~/.kaggle/kaggle.json

kaggle datasets download \
    -d chethuhn/network-intrusion-dataset \
    -p data/raw \
    --unzip
```

---

## Utilisation

Les scripts sont numérotés dans l'ordre d'exécution.

```bash
python scripts/00_test_environment.py
python scripts/02_explore_dataset.py
python scripts/03_preprocess_dataset.py

sbatch scripts/04_feature_selection.sh
sbatch scripts/05_split_and_prepare.sh
sbatch scripts/06_train_baseline.sh
sbatch scripts/07_plot_results.sh

sbatch scripts/08_generate_attacks.sh
# 09_generate_attacks_jsma_sample.sh est desactive : ne pas le soumettre.
sbatch scripts/10_evaluate_and_plot_attacks.sh

sbatch scripts/11_defense_adversarial_training.sh
sbatch scripts/12_defense_label_smoothing.sh
sbatch scripts/13_defense_gaussian_augmentation.sh
sbatch scripts/14_defense_denoising_autoencoder.sh

sbatch scripts/16_ensemble_aggregation.sh
```

Les étapes 05, 06 et 08 se chaînent avec `--dependency=afterok` pour ne pas
lancer une étape si la précédente a échoué :

```bash
J05=$(sbatch --parsable scripts/05_split_and_prepare.sh)
J06=$(sbatch --parsable --dependency=afterok:$J05 scripts/06_train_baseline.sh)
J08=$(sbatch --parsable --dependency=afterok:$J06 scripts/08_generate_attacks.sh)
```

Les scripts `.sh` contiennent des directives `--account` et `--gres`
spécifiques au cluster. Sur narval, les comptes sont `def-smoolak_cpu` et
`def-smoolak_gpu` (le compte sans suffixe n'existe pas) et les GPU sont des
A100 ; sur nibi, l'alias sans suffixe est résolu automatiquement et les GPU
sont des H100. Ces différences ne sont pas versionnées.

---

## Attaques adversariales implémentées

Paramètres alignés sur la Table 2 de l'article.

| Attaque | Référence | Type | Norme | Hyperparamètres |
|---|---|---|---|---|
| FGSM | Goodfellow et al., 2014 | Single-step | L∞ | eps=0.2 |
| BIM | Kurakin et al., 2016 | Iterative | L∞ | eps=0.3, alpha=0.01, 100 iter |
| PGD | Madry et al., 2017 | Iterative | L∞ | eps=0.3, alpha=0.01, 100 iter |
| DeepFool | Moosavi-Dezfooli et al., 2015 | Iterative | L2 | epsilon=1e-6 (overshoot), max_iter=100 |
| JSMA | Papernot et al., 2015 | Feature-based | L0 | theta=0.1, gamma=1.0, untargeted |
| C&W | Carlini & Wagner, 2016 | Optimization | L2 | max_iter=9, confidence=0.0, untargeted |

Pour C&W, les paramètres laissés implicites par l'article sont maintenant
explicites dans la configuration (`binary_search_steps=10`,
`initial_const=0.01`, `learning_rate=0.01`) plutôt que subis comme défauts
de bibliothèque.

## Mécanismes de défense

Quatre défenses combinées dans un ensemble :

- **Adversarial Training (AT)** : entraîner le modèle sur des exemples adversariaux
- **Gaussian Augmentation (GA)** : ajout de bruit gaussien pendant l'entraînement
- **Label Smoothing (LS)** : lissage des labels pour éviter la sur-confiance
- **Denoising Autoencoder (DAE)** : autoencodeur qui nettoie les inputs avant classification

**Agrégation par ensemble** : Majority Voting et Weighted Average, optimisés par Nelder-Mead.

---

## Résultats de référence (papier Awad et al., CIC-IDS 2017)

| Configuration | Accuracy |
|---|---:|
| Baseline (données propres) | 98.11% |
| Baseline sous attaque C&W | 36.00% |
| Label Smoothing (seul) | 85.90% |
| Adversarial Training (seul) | 80.25% |
| Ensemble simple (Majority Voting) | 84.35% |
| **Ensemble optimisé (Majority Voting)** | **87.49%** |

## Nos résultats

### Baseline v5 (CIC-IDS 2017) — à jour

| Métrique | Notre baseline | Papier | Écart |
|---|---:|---:|---:|
| **Accuracy** | **99.79%** | 98.11% | +1.68 |
| F1 weighted | 99.79% | Non détaillé | — |
| F1 macro | 84.11% | Non détaillé | — |

### Vulnérabilité sous attaques — provisoire

Chiffres de l'ancien pipeline (white box, `StandardScaler`, sans
`clip_values`, baseline v4). Voir le tableau de l'étape 6. À remplacer dès que
`08_generate_attacks.py` aura tourné sur le pipeline corrigé.

### Résultats des défenses

**En attente** de la régénération de l'étape 6.

---

## Environnement de développement

- **Développement local** : Arch Linux (Python 3.12) + Windows PowerShell
- **Exécution intensive** : Alliance Canada (nibi et narval), Python 3.11,
  jobs SLURM avec GPU H100 / A100
- **Synchronisation** : Git + GitHub

---

## Références principales

- Awad et al. (2025). *Enhanced Ensemble Defense Framework.* Scientific Reports.
- Goodfellow et al. (2014). *Explaining and Harnessing Adversarial Examples.* ICLR.
- Madry et al. (2017). *Towards Deep Learning Models Resistant to Adversarial Attacks.* ICLR.
- Carlini & Wagner (2016). *Towards Evaluating the Robustness of Neural Networks.* IEEE S&P.
- Papernot et al. (2015). *The Limitations of Deep Learning in Adversarial Settings.* IEEE EuroS&P.
- Moosavi-Dezfooli et al. (2015). *DeepFool: a simple and accurate method to fool deep neural networks.* CVPR.
- Chawla et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* JAIR.
- Tsipras et al. (2019). *Robustness May Be at Odds with Accuracy.* ICLR.

---

## Licence

Projet académique. Utilisation à des fins de recherche uniquement.