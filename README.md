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
| 4 | Split train/val/test + Normalisation + SMOTE | Terminée (5 sept. 2026) |
| 5 | Baseline DNN | Terminée (v5, 100 epochs, sélection sur validation) |
| 6 | Attaques adversariales | Terminée (semi-white box, test complet) |
| 7 | Test de vulnérabilité | Terminée, plus analyse sur test rééquilibré |
| 8 | Mécanismes de défense | **Terminée** (7-8 sept. 2026) |
| 9 | Agrégation par ensemble | **Terminée** (8 sept. 2026) |

Le pipeline complet a été exécuté de bout en bout. Une version 6 du DAE, avec
loss normalisée par groupe et réseau élargi, est en file d'attente ; les
résultats documentés ici sont ceux de la v5.

---

## Résumé des résultats

**Baseline sur données propres** : accuracy 99.79 %, F1 macro 0.8411.

**Vulnérabilité en semi-white box** : les six attaques ramènent le F1 macro
entre 0.06 et 0.33. Les accuracies restent entre 82 et 88 %, mais c'est un
artefact du plancher fixé par la proportion de BENIGN dans le test (83.1 %) —
voir l'étape 7.

**Défenses individuelles**, gain moyen en F1 macro sur les six attaques et
coût sur les données propres :

| Défense | Gain attaques | Coût clean | Bilan net |
|---|---:|---:|---:|
| **Adversarial Training (PGD) + LS** | +0.1015 | −0.0481 | **+0.053** |
| Gaussian Augmentation + LS | +0.1027 | −0.1349 | −0.032 |
| Label Smoothing seul | +0.0001 | −0.0052 | −0.005 |
| Denoising Autoencoder v5 | +0.0737 | −0.3913 | −0.318 |

**Agrégation par ensemble** :

| Méthode | Gain attaques | Coût clean | Bilan net |
|---|---:|---:|---:|
| **Weighted Average, poids égaux** | **+0.1103** | −0.0482 | **+0.062** |
| Majority Voting | +0.0950 | −0.0569 | +0.038 |
| Weighted Average optimisé | +0.0721 | −0.0019 | +0.070 |

**Le résultat central de l'article est reproduit** : l'ensemble à poids égaux
dépasse la meilleure défense individuelle (+0.1103 contre +0.1015) pour un
coût identique sur les données propres.

**Le résultat le plus robuste de cette étude** : JSMA résiste à tout. Cinq
défenses, trois méthodes d'agrégation, quinze combinaisons — aucune ne
dépasse 0.065 de F1 macro contre les 0.0605 du baseline non défendu.

---

## Journal des étapes réalisées

### Étape 1 — Collection des données

- Dataset **CIC-IDS 2017** téléchargé depuis Kaggle (`chethuhn/network-intrusion-dataset`)
- **8 fichiers CSV** correspondant à 5 jours de capture
- **2 830 743 lignes**, **79 colonnes** (78 features + 1 label), **15 classes**
- Stockage sur `~/scratch/` via lien symbolique vers `data/raw/`

### Étape 2 — Preprocessing

1. **Nettoyage des noms de colonnes** : 65 colonnes avaient un espace parasite en début de nom, artefact du séparateur `', '` de CICFlowMeter
2. **Valeurs Inf et NaN** : 2 867 lignes supprimées (0.101 %), divisions par zéro dans `Flow Bytes/s` et `Flow Packets/s`
3. **Doublons** : **307 078 éliminés (10.86 %)**, issus d'attaques automatisées répétitives
4. **Encodage des labels** : 15 classes textuelles → entiers 0-14

**Résultat** : **2 520 798 lignes × 79 colonnes**.

**Note méthodologique** : la suppression des doublons n'est pas documentée
dans le papier. C'est un choix conservateur pour éviter le data leakage.

**Note technique** : les libellés des trois classes `Web Attack – *`
contiennent un caractère mal encodé (tiret cadratin). Les scripts de tracé le
nettoient à l'affichage plutôt que de modifier le `label_encoder`, qui doit
rester identique à celui utilisé pour l'entraînement.

**Six paires de features sont dupliquées** dans le dataset, artefact de
CICFlowMeter identifié en septembre : `Avg Bwd Segment Size` = `Bwd Packet
Length Mean`, `Total Length of Fwd Packets` = `Subflow Fwd Bytes`, `Avg Fwd
Segment Size` = `Fwd Packet Length Mean`, `Subflow Fwd Packets` = `Total Fwd
Packets`, `Fwd Header Length.1` = `Fwd Header Length`, `Subflow Bwd Packets` =
`Total Backward Packets`. Random Forest les a toutes conservées à l'étape 3
puisqu'il évalue chaque feature indépendamment. Sur 58 features, 52 sont donc
réellement indépendantes.

### Étape 3 — Feature selection

Sélection des 58 features via **Random Forest importance**, sur les 2.52M
lignes complètes, 100 arbres, `random_state=42`. Durée : 52 secondes.

- **Importance cumulée des 58 features retenues : 99.34 %**
- **6 features d'importance strictement nulle** parmi les rejetées

**Top 5** : Packet Length Variance (0.0618), Packet Length Std (0.0598), Avg
Bwd Segment Size (0.0565), Max Packet Length (0.0491), Bwd Packet Length Max
(0.0422).

**Interprétation** : les features les plus discriminantes concernent toutes la
distribution statistique des tailles de paquets. Les attaques automatisées
génèrent des paquets aux tailles quasi identiques, le trafic bénin présente
une grande diversité.

### Étape 4 — Split train/val/test + Normalisation + SMOTE

**Ordre d'exécution** (crucial pour éviter le data leakage) :
1. Split stratifié train/test, puis train/validation prélevé sur le reste
2. `MinMaxScaler` (fit sur train uniquement)
3. Bornage explicite de val et test dans [0,1]
4. SMOTE sur le train uniquement, avec cap par classe

| Split | Lignes | Pourcentage |
|---|---:|---:|
| Train | 1 562 894 | 62.0 % |
| Validation | 126 040 | 5.0 % |
| Test | 831 864 | 33.0 % |

**Composition du test set** — chiffre déterminant pour toute l'interprétation
des résultats d'attaques :

| | Lignes | Proportion |
|---|---:|---:|
| BENIGN | 691 369 | **83.1 %** |
| Attaques (14 classes) | 140 495 | 16.9 % |

**Débordement après normalisation** : `MinMaxScaler.transform()` ne garantit
pas que val et test tombent dans l'intervalle appris sur le train.

| Split | Hors [0,1] | Proportion | Dépassement max |
|---|---:|---:|---:|
| Train | 1 / 90 647 852 | 0.0000 % | 0.0000 |
| Validation | 2 / 7 310 320 | 0.0000 % | 0.0377 |
| Test | 55 / 48 248 112 | 0.0001 % | 4.5568 |

Le dépassement est réel en amplitude mais négligeable en prévalence. Le
bornage garantit l'invariant sans changer les résultats de façon mesurable.

**SMOTE avec cap par classe** (plafonds dans `dataset.smote_strategy`) :
train final de **1 819 198 lignes**, 256 304 exemples synthétiques,
expansion globale 1.16×. La version initiale à équilibrage total produisait
21M de lignes avec des ratios jusqu'à 200 000× sur Heartbleed.

**Point de vigilance** : `k_neighbors` vaut 5 et Heartbleed n'a que 6 exemples
réels dans le train, soit exactement le minimum requis par SMOTE.

**Caractéristiques des features**, mesurées sur 300 000 lignes du train — ces
chiffres servent d'échelle à toutes les analyses ultérieures :

| Grandeur | Valeur |
|---|---:|
| Écart-type moyen | 0.10308 |
| Écart-type médian | 0.08190 |
| Écart-type min / max | 0.00054 / 0.47965 |
| Variance moyenne | 0.022818 |
| Features avec écart-type < 0.01 | 14 / 58 |
| Features avec écart-type < 0.05 | 26 / 58 |

La dispersion est extrême, d'un facteur 900 entre la feature la moins et la
plus variable. Les 14 features quasi constantes ont été vérifiées : leurs
écarts entre classes ne dépassent pas 10 écarts-types et concernent des
classes à moins de 20 échantillons, donc elles ne portent pas de signal
exploitable.

### Étape 5 — Entraînement du DNN baseline

**Architecture** : 58 → Dense(512) + ReLU → Dense(256) + ReLU → Dense(15),
**165 391 paramètres**.

#### Cheminement : cinq versions

| Version | Accuracy | F1 macro | Changement principal |
|---|---:|---:|---|
| v1 | 90.69 % | 47.13 % | Fidèle au papier (lr=0.01, 30 epochs) |
| v2 | 95.59 % | 44.66 % | Scheduler agressif |
| v3 | 99.60 % | 78.15 % | SMOTE custom + lr=0.001 |
| v4 | 99.69 % | 80.17 % | 50 epochs |
| **v5** | **99.79 %** | **84.11 %** | Pipeline corrigé + 100 epochs |

v1 était instable : le learning rate de la Table 3 est trop élevé pour cette
architecture. v3 montre que c'est le passage de l'équilibrage total au cap par
classe qui débloque le F1 macro (+33 points), pas le learning rate seul. v4
sélectionnait le meilleur epoch sur l'accuracy du **test set**, ce qui était un
biais méthodologique corrigé en v5.

> **Nomenclature** : les checkpoints v1 à v4 sont conservés à titre de trace.
> Le modèle courant est `baseline_best.pth`, revalidé par empreinte de
> configuration. Il n'y a pas de `baseline_v5_*.pth` : la numérotation
> manuelle a été remplacée par le mécanisme d'empreintes.

#### v5 en détail

| Paramètre | Valeur | Choix |
|---|---|---|
| Learning rate initial | 0.001 | Ajusté depuis 0.01 du papier |
| Scheduler | ReduceLROnPlateau | factor=0.5, patience=5, sur `val_loss` |
| Optimizer / Loss | Adam / CrossEntropy | Comme le papier |
| Batch size | 128 | Comme le papier |
| Epochs | 100 | Étendu depuis 50 |
| Sélection du modèle | `val_acc` | Sur validation, jamais sur test |

Exécution sur narval, A100-SXM4-40GB, **37 minutes** (21.8 s/epoch).

**Pourquoi 100 epochs.** À 50 epochs sous `MinMaxScaler`, `val_loss`
descendait encore et `val_acc` progressait toujours : le modèle n'avait pas
convergé. Le scheduler se déclenche quatre fois (epochs 32, 63, 70, 91),
descendant le learning rate à 6.3e-5. Le plateau est atteint vers l'epoch 81,
le meilleur epoch est le 98. Aucun surapprentissage : la courbe de validation
reste sous celle d'entraînement du début à la fin.

| Métrique | 50 epochs | 100 epochs |
|---|---:|---:|
| Accuracy | 99.689 % | **99.789 %** |
| F1 macro | 81.83 % | **84.11 %** |

**Performance par classe** :

| Classe | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| BENIGN | 99.94 % | 99.84 % | 99.89 % | 691 369 |
| DDoS | 99.93 % | 99.98 % | 99.96 % | 42 245 |
| DoS Hulk | 99.65 % | 99.44 % | 99.55 % | 57 039 |
| PortScan | 98.91 % | 99.90 % | 99.40 % | 29 929 |
| FTP-Patator | 99.95 % | 99.74 % | 99.85 % | 1 957 |
| DoS slowloris | 99.44 % | 99.10 % | 99.27 % | 1 777 |
| DoS GoldenEye | 99.26 % | 98.73 % | 99.00 % | 3 395 |
| DoS Slowhttptest | 97.78 % | 99.54 % | 98.65 % | 1 725 |
| SSH-Patator | 94.61 % | 99.15 % | 96.83 % | 1 062 |
| Web Attack Brute Force | 69.21 % | 99.18 % | 81.53 % | 485 |
| Bot | 60.25 % | 97.82 % | 74.57 % | 643 |
| Infiltration | 64.29 % | 75.00 % | 69.23 % | 12 |
| Web Attack SQL Injection | 23.53 % | 57.14 % | 33.33 % | 7 |
| Web Attack XSS | 44.83 % | 6.05 % | 10.66 % | 215 |
| Heartbleed | 100.00 % | 100.00 % | 100.00 % | 4 |

**Notes** :

1. **Écart avec le papier** : 99.79 % contre 98.11 %, soit +1.68 point. Cet
   écart vient principalement de la stratégie SMOTE custom et de l'extension à
   100 epochs.
2. **Le biais de sélection de v4 était négligeable en pratique** : 99.69 %
   avec sélection sur test contre 99.689 % pour le pipeline corrigé à nombre
   d'epochs égal. La correction restait nécessaire pour la validité du
   protocole.
3. **Heartbleed à 100 % n'a aucune signification** : 4 échantillons. Idem SQL
   Injection (7) et Infiltration (12). Les figures affichent le support.
4. **Web Attack XSS est la classe la plus problématique** (F1 10.66 %) : 93 %
   de ses échantillons sont classés en `Web Attack - Brute Force`, pas en
   trafic bénin. Le modèle détecte une attaque web mais ne discrimine pas
   entre les deux — limite du jeu de données, pas du modèle.

### Étape 6 — Génération des attaques adversariales

**Protocole semi-white box.** Les six attaques sont générées sur un **modèle
substitut** (58 → 100 → 100 → 15, 17 515 paramètres), pas sur le baseline.
Les exemples produits sont ensuite évalués par **transfert** sur le baseline.

**Validation du substitut** : F1 pondéré de **0.9826** sur validation, contre
0.98 annoncé par l'article. Le critère est atteint. Le gate est évalué sur la
validation, jamais sur le test.

**Exécution** : test complet (831 864 échantillons), A100, **2 h 53**.

| Attaque | Accuracy | F1 macro | F1 pondéré | Recall BENIGN | Durée |
|---|---:|---:|---:|---:|---:|
| **Clean** | **99.79 %** | **0.8411** | 0.9979 | 0.9984 | — |
| FGSM | 86.75 % | 0.1502 | 0.8251 | 0.9889 | 1.3 s |
| BIM | 87.42 % | 0.1621 | 0.8293 | 0.9952 | 1.6 min |
| PGD | 82.10 % | 0.0605 | 0.7498 | 0.9876 | 1.2 min |
| DeepFool | 86.92 % | 0.3291 | 0.8512 | 0.9719 | 44.8 min |
| JSMA | 83.05 % | 0.0605 | 0.7544 | 0.9993 | 52.2 min |
| C&W | 84.14 % | 0.3004 | 0.8652 | 0.8810 | 72.0 min |

**Classement réel des attaques, par F1 macro** :

| Rang | Attaque | F1 macro | Interprétation |
|---:|---|---:|---|
| 1 | PGD | 0.0605 | Détection quasi nulle |
| 1 | JSMA | 0.0605 | Évasion totale (recall 0.0000 sur les 14 classes) |
| 3 | FGSM | 0.1502 | Détection résiduelle sur DoS Hulk |
| 4 | BIM | 0.1621 | Idem |
| 5 | C&W | 0.3004 | Seule attaque qui dégrade aussi BENIGN |
| 6 | DeepFool | 0.3291 | La moins destructrice |

#### Comparaison avec l'ancien pipeline (white box, StandardScaler)

| Attaque | Ancien | Nouveau | Écart |
|---|---:|---:|---:|
| DeepFool | 16.71 % | 86.92 % | **+70.2** |
| C&W | 67.15 % | 84.14 % | +17.0 |
| BIM | 72.10 % | 87.42 % | +15.3 |
| PGD | 78.48 % | 82.10 % | +3.6 |
| FGSM | 83.25 % | 86.75 % | +3.5 |
| JSMA | 83.45 % | 83.05 % | −0.4 |

**Le chiffre de 16.71 % pour DeepFool était un artefact.** L'explication
avancée à l'époque (paradoxe robustesse/précision de Tsipras et al.) n'a plus
lieu d'être : le phénomène disparaît dès que le protocole est corrigé.

#### Défis techniques résolus

**1. NumPy 2.x et ART 1.18.** `SaliencyMapMethod` utilise `np.product`,
supprimé dans NumPy 2.x. Monkey-patch appliqué **avant** l'import d'ART :

```python
if not hasattr(np, 'product'):
    np.product = np.prod
```

**2. Coût de JSMA.** Avec `theta=0.1, gamma=1.0` sur le **baseline** en white
box, le débit était de 5.5 batches/heure, soit ~12 jours pour le test complet.
Sur le **substitut** (17 515 paramètres contre 165 391), JSMA prend
**52 minutes**. Le problème venait de la taille du modèle attaqué, pas de
JSMA.

**3. Mode targeted vs untargeted.** ART génère JSMA et C&W en targeted par
défaut. Passer les vrais labels comme cible rend l'attaque triviale.
Correction : `y=None`.

**4. Volume.** Chaque `X_adv_*.pkl` fait 184 MB, soit ~1.1 GB pour les six.
Non versionné.

### Étape 7 — Test de vulnérabilité du baseline

#### Le plancher d'accuracy

Le test contient 83.1 % de trafic BENIGN. Une attaque adversariale masque les
intrusions mais ne touche pas au trafic légitime, qui reste correctement
classé. **L'accuracy ne peut donc pas descendre sous 83.1 % tant que BENIGN
tient.** C'est un plancher arithmétique, pas une propriété du modèle.

Un classifieur qui prédirait « BENIGN » pour tout obtiendrait exactement le
même 83.1 %. Les six accuracies mesurées entre 82 et 88 % ne signifient donc
pas que le baseline résiste — elles signifient que les intrusions sont
devenues invisibles.

**Le F1 macro est la métrique à lire** : de 0.8411 à 0.0605 sous PGD et JSMA,
soit une chute de 93 %.

#### Vérification : test rééquilibré 50/50

Les six jeux d'exemples adversariaux déjà générés ont été réévalués sur un
sous-ensemble à parts égales : toutes les lignes d'attaque conservées
(140 495) et autant de BENIGN tirés au hasard, soit **280 990 lignes à
exactement 50 % BENIGN**. Aucune classe d'attaque n'est perdue. Même modèle,
mêmes exemples : seul le sous-ensemble évalué change.

Script : `scripts/15_balanced_evaluation.py`.

| Attaque | Test complet | Test équilibré | Papier (Table 5) | Écart final |
|---|---:|---:|---:|---:|
| Clean | 99.79 % | 99.68 % | 98.11 % | +1.6 |
| PGD | 82.10 % | **49.43 %** | 46.00 % | **+3.4** |
| FGSM | 86.75 % | **62.96 %** | 54.50 % | +8.5 |
| DeepFool | 86.92 % | **66.77 %** | 53.00 % | +13.8 |
| BIM | 87.42 % | **63.67 %** | 45.00 % | +18.7 |
| JSMA | 83.05 % | **49.97 %** | 81.00 % | −31.0 |
| C&W | 84.14 % | 76.37 % | 36.00 % | +40.4 |

**Écart absolu moyen avec le papier** : 32.5 points sur le test complet,
**19.3 points** sur le test équilibré. Rapprochement de 13.2 points.

**Conclusions** :

1. **La composition du test explique la majeure partie de l'écart** pour PGD,
   FGSM et DeepFool. PGD tombe à 3.4 points du chiffre du papier, qui
   mentionne un test de 20 000 échantillons dont la moitié de trafic régulier.
2. **L'accuracy est trompeuse, et c'est démontrable** : les mêmes exemples
   évalués par le même modèle donnent 83.05 % ou 49.97 % selon le
   sous-ensemble.
3. **JSMA bascule de l'autre côté.** Il coïncidait avec le papier sur le test
   complet et s'en éloigne de 31 points une fois rééquilibré. Son recall BENIGN
   de 0.9993 et son F1 macro de 0.0445 montrent une évasion totale : il
   atterrit mécaniquement sur le plancher, quel qu'il soit. **La coïncidence
   était fortuite.**
4. **C&W résiste au rééquilibrage** (76.37 %). Son F1 macro remonte à 0.3787,
   le meilleur des six : il conserve de vraies performances sur DDoS, DoS Hulk
   et PortScan. L'attaque est **sous-itérée** (`max_iter=9` de la Table 2 plus
   les défauts d'ART).

### Étape 8 — Mécanismes de défense

Quatre défenses, chacune évaluée sur les données propres et les six attaques
du test complet. Le module partagé `src/defenses/common.py` factorise le
chargement, l'évaluation et les métriques.

**Label smoothing appliqué aux quatre défenses**, comme le prescrit
l'article. L'Algorithme 3 dit « train the IDS classifier with the adversarial
augmented dataset and smooth train labels », l'Algorithme 4 « with the
Gaussian augmented dataset and smooth train labels », et la section « Defense
strategies » précise que les défenses sont améliorées « particularly when
trained with smoothed labels ». Le lissage n'est donc pas une défense
parallèle aux trois autres chez eux, c'est une couche appliquée en dessous.

*Note : les Algorithmes 3 et 4 sont intervertis dans l'article — leurs titres
ne correspondent pas à leur contenu.*

#### Résultats des quatre défenses

Gain moyen en F1 macro sur les six attaques, et F1 macro sur données propres :

| Défense | Gain attaques | F1 clean | Coût clean | Bilan net |
|---|---:|---:|---:|---:|
| **AT (PGD) + LS** | +0.1015 | 0.7930 | −0.0481 | **+0.053** |
| GA + LS | +0.1027 | 0.7062 | −0.1349 | −0.032 |
| LS seul | +0.0001 | 0.8359 | −0.0052 | −0.005 |
| DAE v5 | +0.0737 | 0.4498 | −0.3913 | −0.318 |

**F1 macro par attaque** :

| Attaque | Baseline | LS | GA+LS | AT+LS | DAE v5 |
|---|---:|---:|---:|---:|---:|
| Clean | 0.8411 | 0.8359 | 0.7062 | 0.7930 | 0.4498 |
| FGSM | 0.1502 | 0.1854 | 0.2400 | **0.4423** | 0.3184 |
| BIM | 0.1621 | 0.1733 | 0.1653 | 0.1520 | **0.2994** |
| PGD | 0.0605 | 0.0492 | 0.0674 | 0.0656 | **0.1492** |
| DeepFool | 0.3291 | 0.3138 | **0.6217** | 0.5556 | 0.4318 |
| JSMA | 0.0605 | 0.0611 | 0.0405 | **0.0680** | 0.0626 |
| C&W | 0.3004 | 0.2795 | **0.4813** | 0.4053 | 0.2435 |

#### Adversarial Training : PGD et non FGSM

La première version générait les exemples d'entraînement avec FGSM en un pas.
Résultat mesuré : +0.137 de F1 macro contre FGSM, mais +0.011 contre PGD et
+0.001 contre JSMA. La défense protégeait uniquement contre l'attaque exacte
sur laquelle elle s'entraînait — signature du masquage de gradient décrit par
**Madry et al. 2018**, la référence que l'article cite pour cette défense, et
qui recommande explicitement PGD avec plusieurs pas et départ aléatoire.

| Version | Gain moyen |
|---|---:|
| AT-FGSM | +0.0656 |
| **AT-PGD** | **+0.0986** |
| AT-PGD + LS | +0.1015 |

Le gain vient surtout de FGSM (+0.169), DeepFool (+0.035) et C&W (+0.043).
BIM régresse de 0.049 et PGD reste inchangé, donc le masquage de gradient
n'était pas la seule limite.

`eps=0.2` reprend la valeur FGSM de la Table 2, `alpha = 2.5*eps/steps` suit
la règle usuelle. L'article ne documente ni epsilon ni le nombre de pas de son
attaque d'entraînement.

**Trois criterions distincts** sont nécessaires dans ce script, et les
confondre serait un bug silencieux : lissé pour l'entraînement, **non lissé
pour la génération PGD** (lisser modifierait la direction du gradient
d'attaque, donc la nature des exemples), et non lissé pour l'évaluation.

#### L'effet du label smoothing est conditionnel

| Défense | Sans LS | Avec LS | Écart gain | Écart F1 clean |
|---|---:|---:|---:|---:|
| GA | +0.0789 | +0.1027 | +0.024 | **+0.125** |
| AT | +0.0986 | +0.1015 | +0.003 | −0.004 |

Le lissage aide massivement GA et pas du tout AT. L'explication est dans les
courbes d'entraînement : GA sans LS plafonnait à `train_acc` 0.875 avec un
meilleur epoch à **10 sur 100** — le modèle n'arrivait pas à apprendre, le
bruit à sigma=0.1 rendant la cible trop dure. Avec LS, le meilleur epoch passe
à **77 sur 100** et `train_acc` à 0.909. AT n'avait pas ce problème : il
convergeait déjà à l'epoch 93.

**Et LS seul ne défend pas** : +0.0001, soit rien. Combiné, il vaut +0.024 sur
GA et +0.003 sur AT. Le lissage n'est donc pas une défense, c'est un adjuvant
d'entraînement qui agit quand le modèle a du mal à converger.

L'article annonce un bénéfice général de la combinaison ; ce que nous mesurons
est un bénéfice **conditionnel**. C'est aussi le seul point où nos résultats
contredisent directement une affirmation du papier, qui donne à LS sa
meilleure accuracy individuelle (85.9 %).

#### Denoising Autoencoder : six versions

Le DAE est la défense qui a demandé le plus d'itérations. Chaque version a été
diagnostiquée par la mesure, et le cheminement est documenté parce qu'il
constitue une part du travail.

| Version | F1 clean | Gain attaques | Changement |
|---|---:|---:|---|
| v1 | 0.2081 | −0.036 | Bruit gaussien seul, 58 → ReLU(32) → 58 |
| v2 | 0.1577 | −0.073 | + FGSM eps=0.2 régénéré à la volée |
| v3 | 0.1820 | −0.053 | + profondeur, goulot **linéaire** |
| v4 | 0.8411 | 0.0000 | + résidu initialisé à l'identité, fraction propre |
| **v5** | 0.4498 | **+0.0737** | + 4 attaques réelles, reconstruction complète |

**v1 et v2 : mauvaise source de corruption.** Le DAE était entraîné sur du
bruit gaussien ou un FGSM approximatif régénéré à la volée. Or l'Algorithme 5
prescrit « Aggregated Adversarial examples of four attack categories (FGSM,
DEEPFOOL, BIM, JSMA) » et « Construct total samples by concatenating (adv
examples, real samples) ». Le bruit gaussien était une invention de notre
part.

**v3 : le ReLU du goulot détruisait la moitié de l'espace latent.** La mesure
sur le modèle entraîné montre **51.5 % d'activations latentes négatives** et
0 % annulées avec un goulot linéaire — donc avec le ReLU des v1/v2, la moitié
du latent était mise à zéro. Correction réelle mais insuffisante.

**v4 : le réseau n'avait jamais vu d'entrée propre.** Diagnostic décisif, en
passant des données non corrompues dans le DAE v3 :

| Mesure | Valeur |
|---|---:|
| MSE(x, DAE(x)) sur données propres | 0.007189 |
| soit, en part de la variance des données | **31.5 %** |
| Écart-type moyen : entrée → sortie | 0.0988 → 0.0747 |
| Features perdant plus de la moitié de leur écart-type | **29 / 58** |
| F1 macro : sans DAE → avec DAE | 0.8527 → 0.2109 |

`corrompre_batch` corrompait 100 % du batch dans les trois versions. Le réseau
avait donc appris « retire une corruption d'amplitude donnée » et l'appliquait
indistinctement, y compris là où il n'y avait rien à retirer. Aucune
modification d'architecture ne pouvait corriger cela, ce qui explique l'échec
identique des trois premières versions.

La v4 a réglé ce point (MSE(x, DAE(x)) = 0.000001, rapport d'écarts-types
1.000) mais **convergeait vers l'identité** : aucune epoch n'a dépassé la
référence du baseline seul, et le mécanisme de repli a retenu l'identité.

**v5 : les quatre attaques réelles et la reconstruction complète.** Un
nouveau script `08b_generate_train_attacks.py` génère les quatre attaques de
l'Algorithme 5 sur un sous-échantillon stratifié de 400 000 lignes du train —
les `X_adv` de l'étape 6 couvrent le test et ne peuvent pas servir sans fuite.
Amplitudes mesurées :

| Attaque | L∞ moyen | L2 moyen | Features touchées |
|---|---:|---:|---:|
| FGSM | 0.1020 | 0.6113 | 24.6 / 58 |
| BIM | 0.0765 | 0.3816 | 23.5 / 58 |
| DeepFool | 0.0261 | 0.0463 | 50.3 / 58 |
| **JSMA** | **0.9534** | **6.6445** | 53.3 / 58 |

JSMA a un L2 vingt fois supérieur à ce que les versions précédentes
apprenaient à retirer. La connexion résiduelle a aussi été supprimée : elle
court-circuitait la projection sur la variété apprise, qui est le mécanisme
même de purification.

Résultat : la défense fonctionne enfin. **+0.0737** de gain, cinq attaques sur
six progressent, et **PGD gagne +0.089 alors qu'il n'est pas dans
l'entraînement** — la purification transfère à une attaque jamais vue. C&W,
l'autre attaque absente, régresse de 0.057.

Mais le coût sur les données propres est de −0.39 : le DAE détruit encore de
l'information sur des entrées saines.

**v6, en file d'attente.** Le diagnostic de la v5 montre que la MSE brute est
dominée par JSMA. Amplitudes converties en MSE par feature :

| Groupe | MSE de corruption | Part de la loss | Gain en F1 macro |
|---|---:|---:|---:|
| **JSMA** | 0.761196 | **98.8 %** | **+0.002** |
| FGSM | 0.006443 | 0.8 % | +0.168 |
| BIM | 0.002511 | 0.3 % | +0.137 |
| DeepFool | 0.000037 | 0.0 % | +0.103 |

Un seul échantillon JSMA pèse autant que cent échantillons FGSM. Le DAE
consacrait donc la quasi-totalité de son effort à l'unique attaque qu'il ne
corrige pas. La v6 normalise la loss par groupe — chacun pèse 20 % — et
élargit le réseau à 40 de goulot avec deux couches cachées par côté. Les
résultats seront ajoutés ici.

### Étape 9 — Agrégation par ensemble

Les quatre défenses produisent leurs probabilités softmax sur les données
propres et chaque attaque. Trois méthodes d'agrégation sont comparées.

| Méthode | Gain attaques | F1 clean | Coût clean | Bilan net |
|---|---:|---:|---:|---:|
| **Weighted Average, poids égaux** | **+0.1103** | 0.7929 | −0.0482 | **+0.062** |
| Majority Voting | +0.0950 | 0.7842 | −0.0569 | +0.038 |
| Weighted Average optimisé | +0.0721 | 0.8392 | −0.0019 | +0.070 |

**F1 macro par attaque et par méthode** :

| Attaque | Baseline | Majority | WA égal | WA optimisé |
|---|---:|---:|---:|---:|
| Clean | 0.8411 | 0.7842 | 0.7929 | **0.8392** |
| FGSM | 0.1502 | 0.3258 | **0.3517** | 0.2306 |
| BIM | 0.1621 | 0.1878 | **0.1994** | 0.1534 |
| PGD | 0.0605 | 0.0757 | **0.0922** | 0.0629 |
| DeepFool | 0.3291 | 0.4566 | 0.5293 | **0.5454** |
| JSMA | 0.0605 | **0.0643** | 0.0631 | 0.0632 |
| C&W | 0.3004 | **0.5228** | 0.4889 | 0.4400 |

**Accuracies correspondantes** :

| Attaque | Baseline | Majority | WA égal | WA optimisé |
|---|---:|---:|---:|---:|
| Clean | 99.79 % | 98.95 % | 99.38 % | 99.77 % |
| FGSM | 86.75 % | 90.07 % | 90.22 % | 85.99 % |
| BIM | 87.42 % | 87.62 % | 88.38 % | 86.82 % |
| PGD | 82.10 % | 83.05 % | 83.60 % | 81.16 % |
| DeepFool | 86.92 % | 90.65 % | 93.35 % | 90.37 % |
| JSMA | 83.05 % | 83.12 % | 83.11 % | 83.11 % |
| C&W | 84.14 % | 93.54 % | 93.94 % | 90.66 % |

**Le résultat central de l'article est reproduit.** Le Weighted Average à
poids égaux atteint +0.1103, contre +0.1015 pour la meilleure défense
individuelle (AT+LS), pour un coût identique sur les données propres. L'union
fait mieux que chacun de ses membres.

#### Pourquoi la version optimisée fait moins bien

Nelder-Mead retient : **LS 0.364, AT 0.309, GA 0.288, DAE 0.039**. Il écarte
presque complètement le DAE.

C'est rationnel mais contre-productif ici, et cela découle d'un choix
méthodologique assumé : **les poids sont optimisés sur la validation propre,
pas sur les attaques du test**. L'article optimise pour maximiser la
performance sous attaque, mais le faire sur le test reviendrait à ajuster un
hyperparamètre sur le jeu d'évaluation — exactement le biais retiré de
l'étape 5. Or le DAE v5 est de loin le plus faible sur données propres
(F1 0.45 contre 0.79 pour AT), donc l'optimisation le pénalise alors que c'est
sous attaque qu'il apporte quelque chose.

Conséquence : la méthode optimisée préserve remarquablement les données
propres (F1 macro 0.8392, presque le baseline) mais perd le bénéfice du DAE.
Elle régresse même sur FGSM et BIM par rapport aux poids égaux.

**La fonction objectif est le F1 macro et non l'accuracy.** Optimiser
l'accuracy récompenserait un ensemble qui classe tout BENIGN, puisqu'elle est
plafonnée à 83.1 %.

**Déviation technique** : l'article utilise scikit-optimize (optimisation
bayésienne), nous utilisons Nelder-Mead de scipy — équivalent sur un problème
à 4 dimensions, sans dépendance externe à installer sur les clusters.

#### JSMA résiste à tout

C'est le résultat le plus robuste de cette étude.

| Configuration | F1 macro sous JSMA |
|---|---:|
| Baseline non défendu | 0.0605 |
| Label Smoothing | 0.0611 |
| Gaussian Augmentation + LS | 0.0405 |
| Adversarial Training + LS | 0.0680 |
| DAE v5 | 0.0626 |
| Ensemble Majority Voting | 0.0643 |
| Ensemble WA égal | 0.0631 |
| Ensemble WA optimisé | 0.0632 |

Cinq défenses, trois agrégations, quinze combinaisons : **rien ne dépasse
0.068**. Avec un L2 mesuré de 6.64 sur le train — soit une MSE par feature de
0.76, trente-trois fois la variance des données — la perturbation détruit
l'information avant qu'aucune défense n'intervienne. Le recall reste à 0.0000
sur les quatorze classes d'attaque et le recall BENIGN à 0.9993 : évasion
totale, systématiquement.

PGD suit le même schéma, avec un plafond à 0.0922.

---

## Correctifs méthodologiques du 3 septembre 2026

Une relecture du pipeline a mis au jour plusieurs écarts avec le protocole de
l'article. Ces écarts n'avaient pas été détectés parce que chaque script
redéfinissait ses propres constantes au lieu de lire `configs/config.yaml` —
deux sources de vérité qui ont divergé sans que ce soit visible.

**1. Absence de modèle substitut (l'écart le plus important).** Les six
attaques étaient générées directement sur le baseline, c'est-à-dire en white
box complet. L'article place l'attaquant en semi-white box via un substitut.
*Effet mesuré : DeepFool passe de 16.71 % à 86.92 %.*

**2. `clip_values` non défini sur le classifieur ART.** FGSM, BIM et PGD (via
`torchattacks`) étaient bornées dans [0,1] par un clamp interne ; DeepFool,
JSMA et C&W (via ART) ne l'étaient pas.

**3. `StandardScaler` au lieu de `MinMaxScaler`.** Avec des features
centrées-réduites, le clamp interne de `torchattacks` (`torch.clamp(x, 0, 1)`)
écrasait à zéro toutes les valeurs négatives de l'échantillon lui-même, et
seulement pour FGSM, BIM et PGD.

**4. Sélection du meilleur epoch sur le test set.** Le test servait à la fois
à choisir le modèle et à l'évaluer. *Constat après régénération : le biais
était négligeable — 99.69 % avant, 99.689 % après à nombre d'epochs égal. La
correction reste nécessaire pour la validité du protocole.*

**5. Paramètres d'attaque divergents de la Table 2.** JSMA utilisait
`theta=0.3` (Table 2 : 0.1) et C&W `max_iter=10` (Table 2 : 9). Cause racine :
`config.yaml` contenait les bonnes valeurs mais n'était lu par aucun script de
calcul.

**6. Doublon entre les scripts 08 et 09.** `09_generate_attacks_jsma_sample.py`
générait JSMA sur 30 000 échantillons et C&W sur le test complet, en white box
et sans `clip_values`. Ses sorties n'étaient consommées par aucun autre
script. Le script est désactivé.

> *Correction du 5 septembre : une version antérieure de ce README affirmait
> que le tableau de résultats mélangeait deux périmètres d'évaluation. C'est
> faux. Les logs confirment que les six attaques portaient bien sur les
> 831 864 échantillons du test complet.*

**7. `SMOTE_STRATEGY` en dur, et un bug de `k_neighbors`.** Les plafonds sont
maintenant dans la configuration. `k_neighbors` se calculait sur la classe la
plus petite de tout `y_train`, y compris des classes non concernées par SMOTE.

**8. Chiffres de référence du papier erronés dans `10_evaluate_and_plot_attacks.py`.**
`PAPER_RESULTS` contenait FGSM 0.859, BIM 0.810, PGD 0.8025 et JSMA 0.482 —
des valeurs issues d'un **mélange entre la Table 5** (accuracy sous attaque)
**et la Table 7** (performance des défenses). Les vraies références de la
Table 5 sont FGSM 54.5, BIM 45, PGD 46, DeepFool 53, JSMA 81, C&W 36.
L'erreur inversait la lecture de JSMA : comparé à 48.2 %, notre 83.05 %
semblait très éloigné ; comparé au vrai 81 %, il coïncidait. Cette confusion
figurait aussi dans les échanges antérieurs sur l'avancement du projet.

**9. Défenses héritées de l'ancien pipeline (7 septembre).** Les quatre
scripts sélectionnaient le meilleur epoch sur le test, lisaient les `y_*` avec
`pd.read_pickle` alors que l'étape 4 les écrit en tableaux numpy, avaient
leurs hyperparamètres en dur, et restreignaient les moyennes macro aux classes
présentes. Le module `src/defenses/common.py` factorise désormais le
chargement, l'évaluation et les métriques.

**10. Label smoothing absent de trois défenses sur quatre.** Corrigé
conformément aux Algorithmes 3 et 4 (voir étape 8).

### Garde-fou : empreintes de configuration

`src/utils/config.py` calcule une empreinte pour les données, le baseline, le
substitut et chaque attaque. Chaque artefact mis en cache porte l'empreinte
sous laquelle il a été produit et est invalidé si la configuration a changé
depuis — avec un message indiquant quelle clé diffère.

L'empreinte des données est écrite dans `data/processed/data_fingerprint.json`
et revalidée en tête des étapes 5, 6, 8 et 9. Sans elle, modifier `val_size`
sans relancer l'étape 4 produirait un checkpoint cohérent avec sa propre
empreinte mais entraîné sur des données périmées — des chiffres faux sous une
étiquette juste.

Politique par artefact : le baseline lève une erreur bloquante, le substitut
est réentraîné automatiquement, les `X_adv` sont régénérés.

Le mécanisme a fait son travail lors du passage de `scope: "sample"` à
`scope: "full"` : les six fichiers ont été détectés périmés et régénérés.

**Limite connue** : la plupart des scripts `.sh` n'ont pas de `set -e`, donc
Slurm rapporte `COMPLETED` avec un code 0 même quand le script Python plante.
Vérifier les fichiers produits, pas le statut Slurm. Les scripts `08b`, `14`,
`15` et `16` propagent correctement le code de sortie.

---

## Structure du projet

```
IDS-Adversarial-Defense/
├── src/
│   ├── data/              Chargement et preprocessing
│   ├── models/            dnn.py (baseline), substitute.py
│   ├── attacks/           Implémentations des attaques
│   ├── defenses/          common.py (module partagé des 4 défenses)
│   └── utils/             config.py (chargement + empreintes)
├── notebooks/             Exploration
├── scripts/               Scripts exécutables (numérotés par étape)
├── configs/               config.yaml — source unique de vérité
├── data/                  Non versionné, lien vers scratch
│   ├── raw/               CSV originaux
│   └── processed/         Données prétraitées + data_fingerprint.json
├── results/
│   ├── logs/              Logs d'exécution
│   ├── checkpoints/       baseline, substitut, 4 défenses
│   ├── attacks/           Exemples adversariaux du test
│   │   └── train/         Exemples du train, pour le DAE (08b)
│   └── figures/           Graphiques
└── tests/
```

---

## Installation

Python 3.12 sur Linux, ou Python 3.11 sur les serveurs Alliance Canada.

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
sbatch scripts/15_balanced_evaluation.sh

sbatch scripts/08b_generate_train_attacks.sh   # prerequis du DAE

sbatch scripts/11_defense_adversarial_training.sh
sbatch scripts/12_defense_label_smoothing.sh
sbatch scripts/13_defense_gaussian_augmentation.sh
sbatch scripts/14_defense_denoising_autoencoder.sh

sbatch scripts/16_ensemble_aggregation.sh
```

Les étapes 05, 06 et 08 se chaînent avec `--dependency=afterok` :

```bash
J05=$(sbatch --parsable scripts/05_split_and_prepare.sh)
J06=$(sbatch --parsable --dependency=afterok:$J05 scripts/06_train_baseline.sh)
J08=$(sbatch --parsable --dependency=afterok:$J06 scripts/08_generate_attacks.sh)
```

**Temps d'exécution mesurés** (narval, A100) :

| Étape | Durée |
|---|---:|
| 04 Feature selection | 52 s |
| 05 Split + SMOTE | 19 s |
| 06 Baseline (100 epochs) | 37 min |
| 08 Attaques (test complet) | 2 h 53 |
| 08b Attaques (train, 400k) | 45 min |
| 11 AT-PGD (100 epochs) | ~3 h |
| 12 LS, 13 GA (100 epochs) | ~45 min |
| 14 DAE (100 epochs) | 9 à 50 min |
| 16 Ensemble | < 1 min |

Certains `.sh` contiennent des directives `--account` et `--gres` spécifiques
au cluster. Sur narval les comptes sont `def-smoolak_cpu` et
`def-smoolak_gpu` (le compte sans suffixe n'existe pas) et les GPU sont des
A100 ; sur nibi l'alias sans suffixe est résolu automatiquement et les GPU
sont des H100. Ces différences ne sont pas versionnées.

---

## Attaques adversariales implémentées

Paramètres alignés sur la Table 2. Générées sur le **substitut**, évaluées par
transfert sur le baseline.

| Attaque | Référence | Type | Norme | Hyperparamètres |
|---|---|---|---|---|
| FGSM | Goodfellow et al., 2014 | Single-step | L∞ | eps=0.2 |
| BIM | Kurakin et al., 2016 | Iterative | L∞ | eps=0.3, alpha=0.01, 100 iter |
| PGD | Madry et al., 2017 | Iterative | L∞ | eps=0.3, alpha=0.01, 100 iter |
| DeepFool | Moosavi-Dezfooli et al., 2015 | Iterative | L2 | epsilon=1e-6 (overshoot), max_iter=100 |
| JSMA | Papernot et al., 2015 | Feature-based | L0 | theta=0.1, gamma=1.0, untargeted |
| C&W | Carlini & Wagner, 2016 | Optimization | L2 | max_iter=9, confidence=0.0, untargeted |

Pour C&W, les paramètres laissés implicites par l'article sont explicites dans
la configuration (`binary_search_steps=10`, `initial_const=0.01`,
`learning_rate=0.01`) plutôt que subis comme défauts de bibliothèque. Avec
`max_iter=9`, l'attaque reste probablement sous-itérée.

## Mécanismes de défense

| Script | Défense | Approche | Hyperparamètres |
|---|---|---|---|
| `11` | Adversarial Training | PGD à la volée, mix 50/50 | eps=0.2, 7 pas, alpha=0.071 |
| `12` | Label Smoothing | Cross-entropy adoucie | alpha=0.1 |
| `13` | Gaussian Augmentation | Bruit gaussien à l'entraînement | sigma=0.1 |
| `14` | Denoising Autoencoder | Autoencodeur 58→40→58 en amont | 4 attaques, clean_ratio=0.5 |

Le label smoothing (alpha=0.1) est appliqué aux quatre défenses,
conformément aux Algorithmes 3 et 4.

**Hyperparamètres non documentés par l'article** et donc choisis :
`sigma=0.1` pour GA, `eps=0.2` et 7 pas pour AT, `alpha=0.1` pour LS
(d'après Müller et al. 2019), `hidden_dim`, `clean_ratio` et `l1_reg` pour le
DAE, ainsi que la taille du sous-échantillon de `08b` (400 000).

---

## Résultats de référence (papier Awad et al., CIC-IDS 2017)

Deux tableaux distincts, à ne pas confondre — cette confusion a causé
l'erreur documentée au point 8 des correctifs.

**Table 5 — Accuracy du détecteur DNN sous attaque, sans défense** :

| Attaque | Accuracy |
|---|---:|
| Aucune (clean) | 98.11 % |
| FGSM | 54.50 % |
| BIM | 45.00 % |
| PGD | 46.00 % |
| DeepFool | 53.00 % |
| JSMA | 81.00 % |
| C&W | 36.00 % |

**Table 7 — Performance des défenses et de l'ensemble** :

| Configuration | Accuracy |
|---|---:|
| Label Smoothing (seul) | 85.90 % |
| Denoising Autoencoder (seul) | 84.80 % |
| Adversarial Training (seul) | 80.25 % |
| Gaussian Augmentation (seul) | 79.80 % |
| Ensemble simple (Majority Voting) | 84.35 % |
| **Ensemble optimisé (Majority Voting)** | **87.49 %** |

## Comparaison avec nos résultats

### Sur données propres

| Métrique | Nous | Papier | Écart |
|---|---:|---:|---:|
| **Accuracy** | **99.79 %** | 98.11 % | +1.68 |
| F1 macro | 84.11 % | Non détaillé | — |

### Sous attaque

Les accuracies ne sont pas directement comparables : notre test contient
83.1 % de BENIGN contre ~50 % pour l'article, ce qui fixe deux planchers
différents. Sur un test rééquilibré à 50/50, l'écart absolu moyen passe de
32.5 à 19.3 points, et PGD tombe à 3.4 points du chiffre du papier (étape 7).

### Défenses

Le classement diffère de celui de l'article :

| Défense | Notre rang (F1 macro) | Rang du papier (accuracy) |
|---|---|---|
| Adversarial Training | **1er** (+0.1015) | 3e (80.25 %) |
| Gaussian Augmentation | 2e (+0.1027 brut, mais −0.13 sur clean) | 4e (79.80 %) |
| Denoising Autoencoder | 3e (+0.0737, −0.39 sur clean) | 2e (84.80 %) |
| Label Smoothing | 4e (+0.0001) | **1er** (85.90 %) |

Le désaccord le plus net porte sur Label Smoothing, que l'article place
premier et qui ne défend pas du tout dans nos mesures — tout en améliorant
significativement GA lorsqu'il lui est combiné.

### Ensemble

La structure du résultat de l'article est reproduite : **l'ensemble dépasse
chaque défense prise séparément**. Le Weighted Average à poids égaux atteint
+0.1103 de gain moyen en F1 macro contre +0.1015 pour AT+LS.

---

## Limites connues

**Le plancher d'accuracy** rend la comparaison directe avec l'article
impossible sur cette métrique. Toutes nos conclusions reposent sur le F1
macro.

**Les poids de l'ensemble sont optimisés sur la validation propre** et non
sous attaque, pour ne pas ajuster un hyperparamètre sur le jeu d'évaluation.
C'est plus rigoureux que le protocole de l'article mais donne un ensemble
optimisé moins performant sous attaque.

**Le DAE est entraîné sur 400 000 lignes** du train et non sur les 1 819 198,
pour un coût de génération de 45 minutes au lieu d'environ 3 h 25. Le
sous-échantillon est stratifié et couvre les quinze classes.

**Trois classes n'ont aucune signification statistique** dans le test :
Heartbleed (4 échantillons), SQL Injection (7), Infiltration (12). Aucun
chiffre les concernant n'est interprétable, même sur le test complet.

**Les epsilon de la Table 2 sont élevés** relativement à l'échelle des
données : `eps=0.3` vaut 2.9 fois l'écart-type moyen des features (0.103). Ces
valeurs ont du sens pour des images mais rendent PGD et JSMA
quasi irrécupérables sur du trafic réseau normalisé.

---

## Pistes pour la suite

**Élargir le DAE.** La MSE plafonne à 0.0203 alors qu'une PCA à 32
composantes atteint 0.000004 sur la même dimension latente. La v6 en cours
teste un goulot à 40 avec deux couches cachées par côté.

**Explorer sigma pour GA.** La valeur de 0.1 n'est pas documentée par
l'article et le meilleur epoch était le 10 sur 100 sans label smoothing, ce
qui suggère qu'elle est trop élevée. Une courbe du compromis
robustesse/précision en fonction de sigma serait un résultat en soi.

**Augmenter le budget de C&W.** Avec `max_iter=9` de la Table 2 et les
défauts d'ART, l'attaque n'a pas le budget d'optimisation pour converger. Elle
est la moins destructrice des six chez nous et la plus destructrice dans
l'article.

**Retirer les six paires de features dupliquées.** Sur 58 features, 52 sont
réellement indépendantes. L'effet sur les attaques L∞, qui dépensent une
partie de leur budget sur des dimensions redondantes, mérite d'être mesuré.

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
- Madry et al. (2018). *Towards Deep Learning Models Resistant to Adversarial Attacks.* ICLR.
- Carlini & Wagner (2016). *Towards Evaluating the Robustness of Neural Networks.* IEEE S&P.
- Papernot et al. (2015). *The Limitations of Deep Learning in Adversarial Settings.* IEEE EuroS&P.
- Moosavi-Dezfooli et al. (2015). *DeepFool.* CVPR.
- Vincent et al. (2008). *Extracting and Composing Robust Features with Denoising Autoencoders.* ICML.
- Gu & Rigazio (2014). *Towards Deep Neural Network Architectures Robust to Adversarial Examples.* ICLR workshop.
- Zhang et al. (2017). *Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising.* IEEE TIP.
- Müller et al. (2019). *When Does Label Smoothing Help?* NeurIPS.
- Zantedeschi et al. (2017). *Efficient Defenses Against Adversarial Attacks.* AISec.
- Chawla et al. (2002). *SMOTE.* JAIR.
- Tsipras et al. (2019). *Robustness May Be at Odds with Accuracy.* ICLR.

---

## Licence

Projet académique. Utilisation à des fins de recherche uniquement.