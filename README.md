
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
| 4 | Split train/val/test + Normalisation + SMOTE | Réécrite le 3 sept. 2026 (MinMaxScaler, vrai split validation) — à régénérer |
| 5 | Baseline DNN | Réécrite le 3 sept. 2026 (sélection sur validation) — à réentraîner |
| 6 | Attaques adversariales | Réécrite le 3 sept. 2026 (substitut, clip_values) — à régénérer |
| 7 | Test de vulnérabilité | Intégrée à l'étape 6, dépend de sa régénération |
| 8 | Mécanismes de défense | Scripts prêts, dépendent de la régénération des étapes 4-6 |
| 9 | Agrégation par ensemble | Script prêt, dépend de l'étape 8 |

**Voir la section "Correctifs méthodologiques du 3 septembre 2026" plus bas avant de lire les résultats ci-dessous : les chiffres de baseline et d'attaques actuellement documentés dans ce fichier viennent du pipeline PRÉCÉDENT (StandardScaler, pas de substitut) et sont provisoires. Ils seront remplacés dès que les scripts corrigés auront tourné sur le cluster.**

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

Quatre versions successives du baseline ont été entraînées pour identifier la configuration optimale.

**Version 1 — Fidèle au papier (lr=0.01 fixe, 30 epochs)** :
- Accuracy : 90.69%, F1 macro : 47.13%
- Problème : instabilité, pic à epoch 2 puis dégradation

**Version 2 — Scheduler agressif (lr=0.01, factor=0.1, patience=2)** :
- Accuracy : 95.59%, F1 macro : 44.66%
- Problème : lr descend jusqu'à 0.000000

**Version 3 — SMOTE custom + scheduler ajusté (lr=0.001, factor=0.5, patience=5)** :
- Accuracy : 99.60%, F1 macro : 78.15%
- Durée réduite à 12 minutes

**Version 4 (finale) — Extension à 50 epochs** :
- Accuracy : 99.69%, F1 macro : 80.17%, F1 weighted : 99.72%
- Convergence confirmée

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

**Notes méthodologiques importantes** :

1. **Écart avec le papier** : nous obtenons 99.69% contre 98.11% pour le papier. Cet écart favorable (+1.58 points) s'explique probablement par la stratégie SMOTE custom qui réduit le bruit d'interpolation, et par l'extension à 50 epochs.

2. **Le F1 macro reste bas (80%) malgré la haute accuracy** : cela révèle que certaines classes ultra-rares (Web Attack XSS, Web Attack SQL) restent difficiles à apprendre correctement. Le papier ne détaille pas de F1 macro, ce qui empêche la comparaison directe.

3. **Learning rate initial** : passé de 0.01 (papier) à 0.001 après analyse des courbes de la v1 qui montraient une instabilité claire.

4. **Extension à 50 epochs** : décision méthodologique justifiée par l'observation que le modèle progressait encore à l'epoch 30. La convergence est prouvée par la stagnation de test_acc à 99.69% sur les 5 dernières epochs.

### Étape 6 — Génération des attaques adversariales

Six attaques adversariales implémentées et évaluées sur le baseline v4.

#### Résultats obtenus sur le test complet (831 864 échantillons)

| Attaque | Accuracy | Precision macro | Recall macro | F1 macro | F1 weighted | Chute vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (clean) | 99.69% | 83.00% | 83.67% | 80.17% | 99.72% | — |
| FGSM | 83.25% | 14.30% | 6.81% | 6.34% | 75.88% | -16.4 pts |
| BIM | 72.10% | 5.47% | 5.79% | 5.61% | 69.86% | -27.6 pts |
| PGD | 78.48% | 5.49% | 6.30% | 5.86% | 73.09% | -21.2 pts |
| DeepFool | 16.71% | 4.64% | 8.29% | 2.86% | 25.14% | **-82.9 pts** |
| JSMA | 83.45% | 27.32% | 25.44% | 17.96% | 81.91% | -16.2 pts |
| CW | 67.15% | 5.35% | 5.39% | 5.37% | 66.93% | -32.5 pts |

**Observations clés** :
- **DeepFool** est l'attaque la plus dévastatrice sur le test complet (accuracy chute à 16.71%)
- **C&W** est la 2e plus efficace (67.15%), efficace sur toutes les classes non-BENIGN
- **JSMA** touche moins les classes minoritaires que les attaques L∞ (F1 macro 18% vs ~6%)
- Les attaques L∞ (FGSM, BIM, PGD) sont efficaces sur les classes minoritaires
- Le F1 macro chute drastiquement (~6% vs 80%) : les classes minoritaires sont presque totalement échouées sous toutes les attaques

#### Défis techniques rencontrés et résolus

**1. Bug de compatibilité NumPy 2.x avec ART 1.18**

`SaliencyMapMethod` (JSMA) utilise `np.product` qui a été supprimé dans NumPy 2.x. Correction appliquée via monkey-patch :

```python
if not hasattr(np, 'product'):
    np.product = np.prod
```

Le patch doit être appliqué **avant** l'import d'ART.

**2. Coût computationnel de JSMA**

Une première tentative avec `theta=0.1, gamma=1.0` (paramètres par défaut ART) donnait 5.5 batches/heure sur H100, soit environ 12 jours pour le full test set.

**Solution retenue** : ajustement des paramètres à `theta=0.3, gamma=0.15` (perturbation plus forte, jusqu'à 15% des features modifiées), permettant de traiter le test complet en ~15 minutes. Ces paramètres restent conformes à la littérature JSMA (Papernot 2015).

**3. Mode targeted vs untargeted**

Les implémentations ART de JSMA et C&W sont **targeted par défaut**. Passer les vrais labels comme cible rendait l'attaque triviale (le modèle prédit déjà correctement ces classes). Correction : passer `y=None` pour forcer le mode untargeted.

**4. Sauvegarde des exemples adversariaux**

Chaque `X_adv_*.pkl` fait ~185 MB (831 864 × 58 features × float32). Total pour les 6 attaques : ~1.1 GB. Stockage dans `results/attacks/`, non versionné dans Git.

### Étape 7 — Test de vulnérabilité du baseline

Intégrée à l'étape 6 : chaque attaque générée est immédiatement évaluée sur le baseline v4.

**Interprétation** : le baseline v4 est **très vulnérable** aux attaques adversariales, particulièrement à DeepFool (accuracy 16.71%) et C&W (67.15%). Cette vulnérabilité motive l'implémentation des 4 mécanismes de défense de l'étape 8.

**Paradoxe robustesse/précision (Tsipras et al. 2019)** : le baseline atteint 99.69% sur clean (vs 98.11% pour le papier), mais s'effondre à 16.71% sous DeepFool (vs 53.40% pour le papier). Un modèle très confiant est structurellement plus vulnérable aux attaques ciblées qui exploitent la netteté des frontières de décision.

### Étape 8 — Mécanismes de défense

**Scripts prêts, jobs à soumettre.**

Quatre défenses implémentées, chacune évaluée sur données propres + 6 attaques adversariales :

| Script | Défense | Approche | Hyperparamètres clés |
|---|---|---|---|
| `11_defense_adversarial_training.py` | Adversarial Training (AT) | FGSM à la volée, mix 50/50 clean/adv | eps=0.05, ratio=0.5 |
| `12_defense_label_smoothing.py` | Label Smoothing (LS) | Cross-entropy adoucie | alpha=0.1 |
| `13_defense_gaussian_augmentation.py` | Gaussian Augmentation (GA) | Bruit gaussien sur inputs pendant training | sigma=0.1 |
| `14_defense_denoising_autoencoder.py` | Denoising Autoencoder (DAE) | Autoencodeur 58→32→58 en amont du baseline | sigma=0.1, L1=1e-5 |

**Choix méthodologiques communs à toutes les défenses** :
- Même architecture que baseline v4 (BaselineDNN 512→256)
- Mêmes hyperparamètres d'entraînement (lr=0.001, ReduceLROnPlateau, 50 epochs)
- Évaluation intégrée : chaque script produit clean + 6 attaques dans un seul run
- Sauvegarde des checkpoints séparés (`defense_at_best.pth`, `defense_ls_best.pth`, etc.)

### Étape 9 — Agrégation par ensemble

**Script prêt.** Combine les 4 défenses via 3 méthodes d'agrégation :

| Méthode | Description |
|---|---|
| Majority Voting | Chaque défense vote, classe majoritaire retenue |
| Weighted Average (égal) | Moyenne des probabilités softmax, poids 0.25 chacun |
| Weighted Average (optimisé) | Poids optimisés par Nelder-Mead (scipy.optimize) pour maximiser accuracy sur attaques |

**Différence méthodologique avec le papier** : le papier utilise scikit-optimize (Bayesian optimization). Nous utilisons **Nelder-Mead** de scipy.optimize pour :
- Éviter une dépendance externe complexe à installer sur les clusters
- Résultats équivalents pour un problème à 4 dimensions
- Optimisation plus rapide (< 1 min)

Cette différence est documentée dans le rapport final.

---

## Correctifs méthodologiques du 3 septembre 2026

Une relecture du pipeline a mis au jour plusieurs écarts avec le protocole
de l'article, indépendants des choix méthodologiques déjà documentés
(SMOTE custom, ordre split/normalisation/SMOTE). Ces écarts n'avaient pas
été détectés parce que chaque script redéfinissait ses propres constantes
au lieu de lire `configs/config.yaml` — deux sources de vérité qui ont
divergé sans que ce soit visible.

1. **Absence de modèle substitut (l'écart le plus important).** Les six
   attaques étaient générées directement sur le baseline
   (`torchattacks.FGSM(model, ...)`, `PyTorchClassifier(model=model)`),
   c'est-à-dire en white box complet : l'attaquant disposait des vrais
   gradients du modèle qu'il attaque. L'article place l'attaquant en
   semi-white box via un modèle substitut (58 → 100 → 100 → 15, entraîné
   séparément). Un modèle substitut est maintenant implémenté
   (`src/models/substitute.py`) et utilisé par
   `scripts/08_generate_attacks.py` comme source des attaques ; le baseline
   ne sert plus qu'à évaluer la transférabilité des exemples générés.

2. **`clip_values` non défini sur le classifieur ART.** Trois attaques
   (FGSM, BIM, PGD, via `torchattacks`) étaient bornées dans [0,1] par un
   clamp interne à la bibliothèque ; les trois autres (DeepFool, JSMA, C&W,
   via ART) ne l'étaient pas. `create_art_classifier` passe désormais
   `clip_values=(0.0, 1.0)` (`configs/config.yaml` → `dataset.clip_values`).

3. **`StandardScaler` au lieu de `MinMaxScaler`.** Avec des features
   centrées-réduites (donc en partie négatives), le clamp interne de
   `torchattacks` (`torch.clamp(x, 0, 1)`) écrasait à zéro toutes les
   valeurs négatives de l'échantillon lui-même, pas seulement de la
   perturbation — pour FGSM, BIM et PGD uniquement, ce qui rendait leur
   comportement incomparable à DeepFool/JSMA/C&W. L'article ramène les
   features dans [0,1] ; `scripts/05_split_and_prepare.py` utilise
   maintenant `MinMaxScaler`, cohérent avec `clip_values` ci-dessus.

4. **Sélection du meilleur epoch sur le test set.** `06_train_baseline.py`
   choisissait le checkpoint et pilotait le scheduler sur l'accuracy du
   test, qui servait donc à la fois à choisir le modèle et à l'évaluer —
   biais à la hausse de l'accuracy rapportée. Un vrai split de validation
   (`dataset.val_size` dans `configs/config.yaml`) existe maintenant ; le
   test n'est plus touché qu'une fois, pour le rapport final.

5. **Paramètres d'attaque divergents de la Table 2 de l'article.** JSMA
   utilisait `theta=0.3` (Table 2 : 0.1) et C&W `max_iter=10` (Table 2 : 9),
   avec les autres hyperparamètres de C&W laissés aux défauts d'ART. Cause
   racine : `configs/config.yaml` contenait déjà les bonnes valeurs mais
   n'était lu par aucun script de calcul. `src/utils/config.py` centralise
   maintenant le chargement (`load_config()`) et valide la cohérence de la
   configuration avant tout calcul (scaler, clip_values, clés d'attaque
   requises, dimensions du substitut).

6. **Périmètre d'évaluation hétérogène.** JSMA tournait sur un échantillon
   stratifié de 30 000 (`09_generate_attacks_jsma_sample.py`) pendant que
   les cinq autres attaques tournaient sur les 831 864 échantillons du
   test set complet — mélange non signalé dans les résultats. Ce script
   est désactivé (il refuse de s'exécuter) ; `08_generate_attacks.py`
   couvre maintenant les six attaques avec un périmètre unique, piloté par
   `evaluation.scope` dans la configuration.

7. **`SMOTE_STRATEGY` en dur, et un bug de `k_neighbors`.** Les plafonds
   par classe vivaient en dur dans `05_split_and_prepare.py` et pouvaient
   changer sans que rien ne le signale ailleurs ; ils sont maintenant dans
   `configs/config.yaml` → `dataset.smote_strategy`. Au passage, `k_neighbors`
   se calculait sur la classe la plus petite de tout `y_train`, y compris
   des classes non concernées par SMOTE — une classe ultra-rare hors
   stratégie faisait chuter `k_neighbors` pour toutes les autres. Il se
   calcule maintenant uniquement sur les classes réellement suréchantillonnées.

**Garde-fou ajouté en plus** : `src/utils/config.py` calcule une empreinte
de configuration pour les données (`05`), le baseline (`06`) et le
substitut/les attaques (`08`), et invalide un artefact mis en cache
(checkpoint, `X_adv_*.pkl`) si la configuration a changé depuis sa
génération — avec un message qui dit quelle clé a changé plutôt que deux
hashes opaques. Ça évite de réutiliser silencieusement un vieux résultat
après un changement de config, sans plus de cérémonie que ça.

**Risque opérationnel à surveiller** : revenir aux paramètres JSMA fidèles
à l'article (`theta=0.1, gamma=1.0`) réintroduit potentiellement le
problème de durée qui avait motivé le détour vers `theta=0.3, gamma=0.15`
(~12 jours estimés sur le test set complet avec le baseline). Le substitut
est plus petit, mais rien ne garantit que cela suffise.
`08_generate_attacks.py` imprime maintenant une estimation de durée avant
de lancer JSMA en grandeur réelle ; si elle est trop élevée, repasser
`evaluation.scope` à `"sample"` dans `configs/config.yaml` plutôt que de
laisser tourner le job à l'aveugle.

**Conséquence** : les résultats de baseline (99.69%) et d'attaques
documentés plus bas dans ce fichier viennent de l'ancien pipeline et ne
sont plus représentatifs. Le pipeline doit être ré-exécuté dans l'ordre
05 → 06 → 08 pour produire des chiffres comparables au protocole de
l'article. Non vérifié dans cette passe (nécessite l'environnement
cluster complet — `sklearn`, `imblearn`, `torchattacks`, `art` ne sont pas
installés dans l'environnement de développement local) : le comportement
réel de `MinMaxScaler` + SMOTE, l'entraînement du substitut de bout en
bout, et les six attaques via ART/torchattacks. Vérifié localement :
`src/utils/config.py` charge et valide `configs/config.yaml` sans erreur,
et `src/models/substitute.py` instancie un modèle de 17 515 paramètres
avec les bonnes formes d'entrée/sortie.

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
│   ├── attacks/           Exemples adversariaux générés
│   └── figures/           Graphiques pour le rapport
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

Les scripts sont numérotés dans l'ordre d'exécution du pipeline.

```bash
python scripts/00_test_environment.py
python scripts/02_explore_dataset.py
python scripts/03_preprocess_dataset.py

sbatch scripts/04_feature_selection.sh
sbatch scripts/05_split_and_prepare.sh
sbatch scripts/06_train_baseline.sh
python scripts/07_plot_results.py

sbatch scripts/08_generate_attacks.sh
# 09_generate_attacks_jsma_sample.sh est desactive (voir Correctifs du 3
# septembre 2026) : ne pas le soumettre, 08 couvre desormais JSMA et C&W.
sbatch scripts/10_evaluate_and_plot_attacks.sh

sbatch scripts/11_defense_adversarial_training.sh
sbatch scripts/12_defense_label_smoothing.sh
sbatch scripts/13_defense_gaussian_augmentation.sh
sbatch scripts/14_defense_denoising_autoencoder.sh

sbatch scripts/16_ensemble_aggregation.sh
```

---

## Attaques adversariales implémentées

Six attaques standards de la littérature :

| Attaque | Référence | Type | Norme | Hyperparamètres |
|---|---|---|---|---|
| FGSM | Goodfellow et al., 2014 | Single-step | L∞ | eps=0.2 |
| BIM | Kurakin et al., 2016 | Iterative | L∞ | eps=0.3, alpha=0.01, 100 iter |
| PGD | Madry et al., 2017 | Iterative | L∞ | eps=0.3, alpha=0.01, 100 iter |
| DeepFool | Moosavi-Dezfooli et al., 2015 | Iterative | L2 | max_iter=100 |
| JSMA | Papernot et al., 2015 | Feature-based | L0 | theta=0.3, gamma=0.15, untargeted |
| C&W | Carlini & Wagner, 2016 | Optimization | L2 | max_iter=10, confidence=0.0, untargeted |

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

### Baseline v4 (CIC-IDS 2017)

| Métrique | Notre baseline | Papier | Écart |
|---|---:|---:|---:|
| **Accuracy** | **99.69%** | 98.11% | +1.58 |
| F1 weighted | 99.72% | Non détaillé | — |
| F1 macro | 80.17% | Non détaillé | — |

### Vulnérabilité du baseline sous attaques (test complet 831k)

| Attaque | Accuracy | F1 macro | F1 weighted | Chute vs baseline |
|---|---:|---:|---:|---:|
| Baseline (clean) | 99.69% | 80.17% | 99.72% | — |
| FGSM | 83.25% | 6.34% | 75.88% | -16.4 pts |
| BIM | 72.10% | 5.61% | 69.86% | -27.6 pts |
| PGD | 78.48% | 5.86% | 73.09% | -21.2 pts |
| DeepFool | 16.71% | 2.86% | 25.14% | -82.9 pts |
| JSMA | 83.45% | 17.96% | 81.91% | -16.2 pts |
| CW | 67.15% | 5.37% | 66.93% | -32.5 pts |

### Résultats des défenses

**En attente.** Les scripts des 4 défenses individuelles et de l'ensemble sont prêts. Les jobs seront soumis dès que la maintenance des clusters se termine.

---

## Environnement de développement

Le projet utilise une architecture hybride :

- **Développement local** : Arch Linux (Python 3.12) + Windows PowerShell (édition secondaire)
- **Exécution intensive** : Alliance Canada (serveurs nibi et narval), Python 3.11, jobs SLURM avec GPU H100/A100
- **Synchronisation** : Git + GitHub (dépôt privé)

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