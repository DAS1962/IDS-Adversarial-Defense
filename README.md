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

Le pipeline complet a été exécuté de bout en bout. Les exécutions de référence de chaque défense sont figées dans
`configs/config.yaml`, section `reference_runs`, avec leur justification.

---

## Résumé des résultats

Tous les chiffres viennent de `results/figures/defenses_summary.csv`, produit
par `scripts/17_plot_defenses.py` qui lit directement les fichiers de
résultats. Aucun n'est recopié à la main : deux valeurs fausses s'étaient
glissées dans une version antérieure de ce document par transcription.

**Baseline sur données propres** : accuracy 99.79 %, F1 macro 0.8411,
F1 pondéré 0.9979.

**Vulnérabilité en semi-white box** : les six attaques ramènent le F1 macro
entre 0.06 et 0.33. Les accuracies restent entre 82 et 88 %, artefact du
plancher fixé par les 83.1 % de trafic bénin du test — voir l'étape 7.

**Tableau de référence.** Le gain est la moyenne des écarts au baseline sur
les six attaques, le coût l'écart sur données propres, le bilan net leur
somme. Le rappel BENIGN minimal est le minimum sur les six attaques.

| Configuration | F1 macro | F1 pondéré | MCC binaire | Rappel BENIGN min | Gain | Coût | Bilan net |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.8411 | 0.9979 | 0.9936 | 0.8810 | — | — | — |
| **Ensemble parts égales** | 0.8188 | 0.9972 | 0.9922 | 0.9644 | **+0.1189** | −0.0224 | **+0.0965** |
| **Adversarial (PGD, 150)** | **0.8494** | 0.9972 | 0.9914 | 0.8834 | +0.0863 | **+0.0082** | +0.0945 |
| Ensemble optimisé | 0.8264 | 0.9977 | 0.9931 | 0.9531 | +0.1081 | −0.0148 | +0.0933 |
| Ensemble vote majoritaire | 0.8278 | 0.9943 | 0.9811 | **0.9810** | +0.0906 | −0.0133 | +0.0773 |
| Gaussienne (sigma 0.02) | 0.6813 | 0.9696 | 0.8901 | 0.9582 | +0.1084 | −0.1598 | −0.0513 |
| Lissage des étiquettes | 0.8359 | 0.9979 | 0.9936 | **0.6069** | +0.0001 | −0.0052 | −0.0051 |
| Autoencodeur (v5) | 0.4498 | 0.9300 | 0.7540 | 0.9176 | +0.0737 | −0.3914 | −0.3177 |

**Ce que ce tableau établit.**

La proposition centrale de l'article est vérifiée, mais de justesse :
l'ensemble à parts égales dépasse la meilleure défense individuelle de
+0.033 en gain brut. En bilan net l'écart tombe à **0.002**, indépartageable
sans mesure de variance.

**Aucune métrique ne suffit seule.** Le lissage est premier en F1 pondéré
(0.9979, identique au baseline) et dernier en rappel BENIGN minimal (0.6069,
soit 39 % du trafic légitime rejeté sous PGD). Une défense qui ne gagne rien
(+0.0001) paraît parfaite sous la métrique que publie l'article.

**Le MCC binaire nous place au-dessus de l'article.** Sa Table 9 donne 0.596
à 0.698 selon la défense ; nous obtenons 0.754 à 0.994. C'est la comparaison
la moins contestable dont nous disposions, le MCC étant robuste au
déséquilibre — mais nos valeurs portent sur un test à 83.1 % de bénin et les
leurs sur un test de composition non documentée.

**Deux attaques résistent à tout.** Dix-huit configurations testées, rien ne
dépasse 0.068 de F1 macro contre JSMA et 0.092 contre PGD.

**Les défenses sont monomaniaques.** Chacune tire l'essentiel de son gain
d'une seule attaque : l'adversarial 71 % de FGSM, la gaussienne 54 % de
DeepFool. Cela contredit le récit de synergie complémentaire du papier, dont
la Figure 7 montre au contraire des défenses uniformes sur les six attaques.

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
du test complet. Le module `src/defenses/common.py` factorise le chargement,
l'évaluation et les métriques pour que les quatre soient mesurées à
l'identique.

**Le lissage des étiquettes est appliqué aux quatre défenses**, comme le
prescrit l'article. Son Algorithme 3 dit « train the IDS classifier with the
adversarial augmented dataset and smooth train labels », son Algorithme 4
« with the Gaussian augmented dataset and smooth train labels », et la section
« Defense strategies » précise que les défenses sont améliorées
« particularly when trained with smoothed labels ». Le lissage n'est donc pas
une défense parallèle aux trois autres chez eux, c'est une couche appliquée
en dessous.

*Note : les Algorithmes 3 et 4 sont intervertis dans l'article — leurs titres
ne correspondent pas à leur contenu.*

#### Exécutions de référence

Plusieurs exécutions existent par défense : quatre pour l'adversarial, quatre
pour la gaussienne, six pour l'autoencodeur. Les scripts retenaient
initialement la plus récente, ce qui est arbitraire — `defense_ga_best.pth`
contenait encore sigma 0.1 alors que sigma 0.02 est meilleur sur les quatre
critères. Le choix est maintenant figé dans `configs/config.yaml`, section
`reference_runs`, avec sa justification.

| Défense | Version retenue | Motif |
|---|---|---|
| Lissage | alpha 0.1 | exécution unique |
| Adversarial | PGD en ligne, 7 pas, 150 passages | attaques d'évaluation non vues |
| Gaussienne | sigma 0.02 | domine sur les quatre critères |
| Autoencodeur | v5, 58-45-32-45-58 | seule des six à gain positif |

#### Adversarial training : deux protocoles, et un problème de mesure

L'Algorithme 3 prend en entrée « Previously Generated Adversarial examples of
four attack categories (FGSM, DEEPFOOL, BIM, JSMA) ». Les deux versions ont
été exécutées.

| | PGD en ligne | Quatre attaques (Algo 3) |
|---|---:|---:|
| Gain moyen | +0.0863 | **+0.3869** |
| F1 macro propre | **0.8494** | 0.8191 |
| Coût propre | **+0.0082** | −0.0220 |
| Rappel BENIGN min | **0.8834** | 0.5748 |
| F1 sous BIM | 0.1642 | **0.9651** |

**La version conforme obtient quatre fois plus, et c'est un artefact.** Ses
exemples d'entraînement (`08b`, sur le train) et d'évaluation (`08`, sur le
test) viennent du même substitut avec les mêmes paramètres : les lignes
diffèrent, la distribution des perturbations non. Le modèle a donc vu la forme
exacte de FGSM, BIM, DeepFool et JSMA.

Le signe est net : sous BIM elle atteint **0.9651, soit mieux que son F1 macro
sur données propres (0.8191)**. Un modèle qui classe mieux des données
attaquées que des données saines n'est pas robuste, il a mémorisé la forme de
l'attaque. Son rappel BENIGN tombe par ailleurs à 0.5748 sous C&W : elle
rejette 42 % du trafic légitime, ce que le gain moyen masque entièrement.

La version PGD en ligne s'entraîne contre des perturbations générées sur
elle-même, tandis que les six attaques d'évaluation viennent du substitut. Son
+0.0863 est donc intégralement mesuré sur des perturbations non vues. Elle est
retenue pour cette raison, et c'est aussi la seule configuration du projet à
**améliorer** le F1 macro sur données propres.

**À retenir malgré son écartement.** La version conforme porte PGD de 0.0605 à
**0.3121** et JSMA de 0.0605 à **0.1092**, sur deux attaques absentes de son
entraînement. Ce sont les seuls progrès réels du projet sur les deux attaques
que rien n'arrête.

L'article a la même structure de protocole. Si ses exemples d'entraînement et
d'évaluation partagent aussi leur distribution, ses chiffres pour ces quatre
attaques sont surestimés par construction — ce qui expliquerait la bande
étroite de 77.4 à 83.85 % de sa Table 6.

Le paramètre `defenses.AdversarialTraining.source` bascule entre les deux
protocoles : `precomputed` suit l'Algorithme 3, `online` génère à la volée.

Point d'implémentation : **trois fonctions de coût distinctes** sont
nécessaires dans ce script, et les confondre serait un bug silencieux — lissée
pour l'entraînement, **non lissée pour générer l'attaque** (lisser modifierait
la direction du gradient, donc la nature des exemples produits), non lissée
pour l'évaluation.

#### Augmentation gaussienne : balayage de sigma

L'article ne documente pas sigma. Trois valeurs ont été testées.

| sigma | Gain | F1 macro propre | Rappel BENIGN min | Bilan net |
|---:|---:|---:|---:|---:|
| **0.02** | **+0.1084** | **0.6813** | **0.9582** | **−0.0513** |
| 0.05 | +0.0948 | 0.6459 | 0.0119 | −0.1005 |
| 0.1 | +0.0844 | 0.5998 | 0.0184 | −0.1569 |

Sigma 0.02 domine sur les quatre critères. Et le rappel BENIGN minimal révèle
ce que les autres métriques ne montrent pas : **à 0.05 et 0.1, le modèle
rejette plus de 98 % du trafic légitime sous JSMA**. Il est alors inutilisable
en pratique, quel que soit son gain moyen.

Une observation qui tempère le mérite de cette défense : sous DeepFool, son F1
macro (0.6769) est **identique à son F1 sur données propres** (0.6813). La
perturbation de DeepFool a une MSE de 0.000194, inférieure à la variance du
bruit d'entraînement (0.0004 à sigma 0.02). L'attaque est donc simplement
invisible pour ce modèle — ce n'est pas de la défense, c'est un non-événement.
Or DeepFool fournit 54 % du gain de cette configuration.

Le plafonnement de l'apprentissage n'est pas dû au bruit : le meilleur passage
reste au 18ᵉ sur 100 à sigma 0.02, au 19ᵉ à sigma 0.1. La cause reste
inexpliquée.

#### Denoising autoencoder : six versions

C'est la partie qui a demandé le plus d'itérations. Chaque échec a été
diagnostiqué par la mesure.

| Version | F1 propre | Gain | Changement |
|---|---:|---:|---|
| v1 | 0.2081 | −0.036 | bruit gaussien seul |
| v2 | 0.1577 | −0.073 | + FGSM régénéré à la volée |
| v3 | 0.1820 | −0.053 | goulot rendu **linéaire** |
| v4 | 0.8411 | 0.0000 | + résidu et fraction propre |
| **v5** | 0.4498 | **+0.0737** | + 4 attaques réelles, reconstruction complète |
| v6 | 0.1726* | abandonnée | loss normalisée par groupe |

*\*F1 macro moyen sur validation propre et quatre attaques du train.*

**v3** : le ReLU du goulot annulait toute coordonnée négative. Mesure : 51.5 %
des activations latentes sont négatives, donc la moitié de l'espace était
détruite.

**v4** : diagnostic décisif. En passant des données saines dans le v3,
MSE(x, DAE(x)) = 0.00719, soit **31.5 % de la variance**, et 29 features sur
58 perdant plus de la moitié de leur écart-type. Le batch était corrompu à
100 %, donc le réseau n'avait jamais appris à laisser passer une entrée
propre. Corrigé, il converge alors vers l'identité : gain exactement nul.

**v5** : les quatre attaques réelles de `08b` et la reconstruction complète
sans résidu, qui court-circuitait la projection sur la variété apprise.
Premier gain positif. PGD y gagne +0.089 alors qu'il n'est pas dans
l'entraînement, ce qui montre un transfert réel ; C&W régresse de 0.057.

**v6, abandonnée.** La MSE brute était dominée par JSMA à 98.8 % du gradient
pour un gain de +0.002. La loss normalisée par groupe a créé le problème
symétrique : DeepFool est passé à 95.8 %. En reconstituant les MSE brutes, les
groupes propre, FGSM, BIM et DeepFool donnent tous 0.0037 — **le DAE a un
plancher de bruit propre, indépendant de son entrée**. La perturbation de
DeepFool (0.000194) est dix-neuf fois inférieure, rendant la cible normalisée
inatteignable.

Ce plancher définit une **fenêtre d'utilité étroite** : le DAE ne peut aider
que les attaques dont la perturbation dépasse son propre bruit sans excéder sa
capacité. FGSM (0.0128) et BIM (0.0096) y sont, DeepFool est dessous, JSMA
(0.806) très au-dessus.

**Réserve** : la v5 est entraînée sur les mêmes quatre attaques que l'AT
écarté plus haut, donc son gain sur FGSM, BIM, DeepFool et JSMA est
probablement surestimé pour la même raison. L'Algorithme 5 prescrit
explicitement ces quatre attaques, sans alternative.

### Étape 9 — Agrégation par ensemble

Les quatre défenses produisent leurs probabilités softmax sur les données
propres et chaque attaque. Trois méthodes sont comparées.

| Méthode | Gain | F1 macro propre | Rappel BENIGN min | Bilan net |
|---|---:|---:|---:|---:|
| **Parts égales** | **+0.1189** | 0.8188 | 0.9644 | **+0.0965** |
| Optimisé (Nelder-Mead) | +0.1081 | 0.8264 | 0.9531 | +0.0933 |
| Vote majoritaire | +0.0906 | 0.8278 | **0.9810** | +0.0773 |

**La proposition de l'article est vérifiée, de justesse.** La moyenne à parts
égales atteint +0.1189 contre +0.0863 pour la meilleure défense individuelle.
En bilan net l'écart tombe à 0.002 — indépartageable sans mesure de variance.

Correspondance avec l'article : son « soft voting », qui donne son meilleur
résultat (87.49 %), est la moyenne des probabilités, donc notre méthode à
parts égales. Cohérent avec nous, où c'est aussi le meilleur gain.

**Ce que l'ensemble apporte réellement ici.** Ce n'est pas la performance,
puisque l'adversarial seul est à 0.002 de lui. C'est la **protection contre
l'effondrement d'un membre** : le rappel BENIGN minimal passe de 0.6069 pour
le lissage seul à 0.9810 pour le vote majoritaire. Un membre qui rejette 39 %
du trafic légitime est corrigé par les trois autres.

C'est une conclusion différente de celle du papier — l'ensemble assure contre
les défaillances plutôt qu'il n'améliore la détection — mais elle est mesurée.

#### Sur les poids optimisés

Nelder-Mead retient lissage 0.290, gaussienne 0.263, adversarial 0.255,
autoencodeur 0.193. La gaussienne reçoit donc **plus de poids que
l'adversarial** alors que son F1 macro propre est de 0.6813 contre 0.8494.

C'est la conséquence d'un choix assumé : **les poids sont optimisés sur la
validation propre, pas sur les attaques du test**. L'article optimise pour la
performance sous attaque, mais le faire sur le jeu d'évaluation reviendrait à
y ajuster un hyperparamètre — exactement le biais retiré de l'étape 5.

**La fonction objectif est le F1 macro et non l'accuracy.** Optimiser
l'accuracy récompenserait un ensemble classant tout en BENIGN, puisqu'elle est
plafonnée à 83.1 %.

**Déviation technique** : l'article utilise scikit-optimize (optimisation
bayésienne), nous utilisons Nelder-Mead de scipy — équivalent sur un problème
à 4 dimensions, sans dépendance externe à installer sur les clusters.

#### JSMA et PGD résistent à tout

| Configuration | JSMA | PGD |
|---|---:|---:|
| Baseline | 0.0605 | 0.0605 |
| Lissage | 0.0618 | 0.0491 |
| Adversarial | 0.0628 | 0.0885 |
| Gaussienne | 0.0651 | 0.0636 |
| Autoencodeur | 0.0626 | **0.1492** |
| Ensemble vote | 0.0627 | 0.0630 |
| Ensemble parts égales | 0.0626 | 0.0733 |
| Ensemble optimisé | 0.0624 | 0.0625 |

**Rien ne dépasse 0.068 contre JSMA.** Sa perturbation a une MSE de 0.806, soit
trente-cinq fois la variance des données : l'information discriminante est
détruite avant que la moindre défense n'intervienne.

Une seule exception, hors référence : l'adversarial entraîné sur les quatre
attaques atteint 0.1092 sous JSMA et 0.3121 sous PGD. JSMA figure dans son
entraînement, PGD non — ce dernier gain est donc réel.

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

## Comparaison avec l'article

### L'écart principal : la sélection de features est inversée

C'est le point le plus structurant, et il est en amont de tout le reste.

L'article écrit, section *Preprocessing* : « To ensure that the functionality
of the network communication channel is preserved […] **the non-functional
features that do not matter the most are selected** for training and testing
purposes. In this context, we use the Extremely Randomized Trees (ERTs)
classifier **to select the most important (functional) features and exclude
them**. » La légende de sa Figure 3 confirme : *« The functional features to be
removed from the dataset. »*

**Ils gardent les 58 features les moins importantes et suppriment les 20 plus
importantes.** Leur Figure 3 liste les supprimées : `packet_length_variance`,
`max_packet_length`, `packet_length_mean`, `avg_bwd_segment_size`,
`bwd_packet_length_max`, `packet_length_std`, `average_packet_size`,
`destination_port`…

Notre top-10 conservé : Packet Length Variance (0.0618), Packet Length Std
(0.0598), Avg Bwd Segment Size (0.0565), Max Packet Length (0.0491), Bwd
Packet Length Max (0.0422), Average Packet Size (0.0397)…

**Nous avons gardé exactement ce qu'ils ont jeté.**

Leur logique : une attaque adversariale doit préserver la fonctionnalité du
trafic. On ne peut pas modifier arbitrairement la taille des paquets sans
casser la communication. L'IDS est donc entraîné uniquement sur des features
qu'un attaquant peut manipuler sans casser son attaque. C'est leur
contribution revendiquée : « Fulfil the constraint of Preserving the
functionality of traffic features during the adversarial generation by
Leveraging Extremely Randomized Trees for robust non-functional feature
selection. »

**Conséquences.** Notre baseline à 99.79 % contre leur 98.11 % s'explique en
partie par là : nous disposons des features discriminantes, eux non. Et
surtout, nos attaques et les leurs ne portent pas sur le même espace de
features, ce qui limite fortement la comparabilité des chiffres sous attaque.

Cet écart n'est pas corrigé sur cette branche. Il fait l'objet d'une
reproduction séparée.

### La métrique de l'article n'est pas le F1 macro

Sa Table 4 donne accuracy 98.11, recall 98.11, precision 98.11, F1 98.068.
Quatre valeurs quasi identiques : signature d'une moyenne pondérée ou micro,
pas macro. Le chiffre comparable au leur est donc **notre F1 pondéré de
0.9979**, pas notre F1 macro de 0.8411.

Notre analyse sur le F1 macro reste valide et utile, mais c'est un ajout de
notre part, pas une reproduction.

### La composition du jeu de test

L'article écrit : « From the training dataset, we randomly select 40,000
samples (10,000 for each attack) of which 20,000 represent regular traffic and
the remaining represent intrusions. From the testing data set we select 20,000
samples (5000 for each attack). »

Le 50/50 est explicite pour l'entraînement. **Pour le test, la proportion de
bénin n'est pas donnée.** Ces chiffres confirment en revanche que le protocole
porte sur **quatre attaques** (10 000 × 4 = 40 000), alors que la Table 5 en
évalue six.

Notre analyse du plancher (étape 7) reste donc valide : sans connaître leur
proportion de bénin, les accuracies ne sont pas comparables.

### Tableau de correspondance

| Élément | Article | Nous | Verdict |
|---|---|---|---|
| Dataset | CIC-IDS 2017 | idem | conforme |
| **Sélection de features** | **ERT, 58 moins importantes** | **RF, 58 plus importantes** | **inversé** |
| Équilibrage | sous-échantillonnage 50 %, avant split | SMOTE plafonné, après split | opposé |
| Doublons | non mentionné | 307 078 retirés | ajout |
| Baseline | 512-256-15, lr 0.01, 30 passages | idem, lr 0.001, 100 passages | ajusté |
| Substitut | 58-100-100-15, F1 0.98 | idem, F1 0.9826 | conforme |
| Attaques (Table 2) | 6 évaluées, 4 en défense | 6 évaluées, 4 en défense | conforme |
| Test adversarial | 20 000, composition inconnue | 831 864, 83.1 % bénin | divergent |
| Lissage sur les 4 défenses | oui | oui | conforme |
| DAE sur adv + réels | oui | oui (v5) | conforme |
| AT sur 4 attaques | oui | non (PGD en ligne) | écart assumé |
| Métrique publiée | pondérée ou micro | macro + pondérée + MCC | ajout |
| Optimisation des poids | non précisée | validation propre | plus strict |
| Black box | annoncé | non fait | non étayé chez eux |

### Résultats de référence de l'article

**Table 5 — accuracy du détecteur sous attaque, sans défense :**

| Attaque | Accuracy |
|---|---:|
| Aucune | 98.11 % |
| JSMA | 81.00 % |
| FGSM | 54.50 % |
| DeepFool | 53.00 % |
| PGD | 46.00 % |
| BIM | 45.00 % |
| C&W | 36.00 % |

**Table 7 — performance des défenses :**

| Configuration | Accuracy |
|---|---:|
| Label Smoothing | 85.90 % |
| Denoising Autoencoder | 84.80 % |
| Ensemble (majority voting) | 84.35 % |
| Adversarial Training | 80.25 % |
| Gaussian Augmentation | 79.80 % |
| **Ensemble optimisé (soft voting)** | **87.49 %** |

**Table 9 — MCC binaire :** LS 0.698, DAE 0.680, AT 0.608, GA 0.596.
Nous obtenons 0.994, 0.754, 0.991 et 0.890 respectivement.

### Désaccords de classement

| Défense | Notre rang (bilan net) | Rang de l'article (accuracy) |
|---|---|---|
| Adversarial | **1ᵉʳ** (+0.0945) | 4ᵉ (80.25 %) |
| Lissage | 3ᵉ (−0.0051) | **1ᵉʳ** (85.90 %) |
| Gaussienne | 4ᵉ (−0.0513) | 5ᵉ (79.80 %) |
| Autoencodeur | 5ᵉ (−0.3177) | 2ᵉ (84.80 %) |

Le désaccord le plus net porte sur le lissage, que l'article place premier et
qui ne défend pas du tout dans nos mesures (+0.0001), tout en rejetant 39 % du
trafic légitime sous PGD.

### Incohérences relevées dans l'article

Elles relativisent la fiabilité de ses chiffres et sont listées ici par souci
de complétude.

**Formule de précision erronée**, Eq. (7) : `Precision = TP/(TP+TN)`. Le
dénominateur correct est `TP+FP` ; la formule donnée est celle d'aucune
métrique standard.

**Trois valeurs qui ne concordent pas entre elles.** L'abstract annonce
87.34 % en majority voting, la Table 8 donne 87.49 %, la discussion 84.34 %.
La Table 4 donne 98.11 % et la Figure 5 98.105 %. Le texte cite « MCC values
of 0.608 and 0.640 » pour le DAE quand sa Table 9 affiche 0.680.

**Algorithmes 3 et 4 intervertis** : leurs titres ne correspondent pas à leur
contenu.

**Quatre attaques ou six ?** Les Algorithmes 2 à 5 en utilisent quatre, la
Table 5 en évalue six, la Table 10 n'en liste que quatre.

**MCC calculé en binaire** — « we reduce the classification task into binary
classification for simplicity » — alors que tout le reste est multiclasse.

**Black box annoncé, jamais rapporté** : « All the defense mechanisms are
evaluated in semi-white box and black box settings », mais aucun résultat ne
distingue les deux scénarios.

**Matériel** : Google Colab sur un i7 à 2.70 GHz avec 8 Go de RAM, ce qui
explique les sous-échantillons de 20 000 et 40 000 lignes sur un jeu de
2.8 millions.

## Limites connues

**Une seule exécution par configuration, donc aucune variance mesurée.** C'est
la limite principale. Les écarts sur lesquels reposent plusieurs conclusions
sont du même ordre que ce qu'une graine différente pourrait produire :
0.002 entre l'ensemble à parts égales et l'adversarial seul en bilan net,
0.007 entre l'adversarial avec et sans lissage. **Ces comparaisons-là ne sont
pas départageables en l'état.** Trois à cinq graines par configuration, soit
environ six heures de GPU, transformeraient « semble meilleur » en « est
meilleur ».

**Aucun attaquant adaptatif.** Les six jeux d'exemples adversariaux ont été
générés une fois, sur un substitut entraîné à imiter le baseline. Un attaquant
réel face à un IDS défendu entraînerait son substitut contre **le modèle
défendu**. Nos gains sont donc des bornes supérieures optimistes — critique
standard en robustesse adversariale (Carlini et al., *On Evaluating
Adversarial Robustness*, 2019). L'article a la même limite.

**Le trafic bénin est perturbé lui aussi.** Les `X_adv` contiennent les
831 864 lignes du test, dont les 691 369 BENIGN. Or un attaquant perturbe son
propre trafic d'attaque, pas le trafic légitime de la victime qu'il ne
contrôle pas. C'est ce qui produit l'effondrement de la gaussienne à
sigma 0.05 et 0.1, un scénario qui ne peut pas se produire en pratique. Une
réévaluation ne perturbant que les 140 495 lignes d'attaque ne demanderait
aucun calcul GPU.

**Sélection du modèle sur `val_acc` alors que le F1 macro est notre métrique.**
Incohérence interne relevée tardivement. Le cas de l'adversarial à 150
passages l'illustre : `val_acc` 0.9973 contre 0.9979 pour le baseline — plus
basse — alors que son F1 macro test est plus haut. La sélection travaille
contre l'objectif.

**Le gain moyen masque que chaque défense est monomaniaque.** L'adversarial
tire 71 % de son gain de FGSM, la gaussienne 54 % de DeepFool. Une moyenne sur
six valeurs dont quatre sont nulles n'est pas une mesure de robustesse.

**Trois classes n'ont aucune signification statistique** dans le test :
Heartbleed (4 échantillons), SQL Injection (7), Infiltration (12).

**Huit hyperparamètres sont nos choix** et non ceux de l'article : mise à
l'échelle, rééquilibrage, pas d'apprentissage, sigma, type d'attaque pour
l'adversarial, taille interne de l'autoencodeur, pondération de sa fonction de
coût, composition du test.

**L'autoencodeur n'a vu que 400 000 lignes** du train sur 1 819 198, pour un
coût de génération de 45 minutes au lieu d'environ 3 h 25.

**Les epsilon de la Table 2 sont élevés** relativement à l'échelle des
données : `eps=0.3` vaut 2.9 fois l'écart-type moyen des features (0.103). Ces
valeurs ont du sens pour des images, moins pour du trafic réseau normalisé, et
c'est une piste d'explication au fait que PGD et JSMA soient irrécupérables.

---

## Pistes pour la suite

**Reproduction fidèle du protocole de l'article.** La sélection de features
inversée est l'écart le plus structurant et n'est pas corrigée ici. Une
reproduction séparée reprendrait : features non fonctionnelles sélectionnées
par ERT, sous-échantillonnage à 50 % par classe avant découpage, test de
20 000 échantillons, quatre attaques, F1 pondéré comme métrique. Environ huit
heures de GPU pour tout le pipeline.

**Mesure de variance.** Trois à cinq graines sur les configurations retenues.
C'est la correction qui rendrait les comparaisons fines défendables.

**Attaque adaptative.** Régénérer les six attaques sur un substitut entraîné
contre le modèle défendu, puis réévaluer. Environ trois heures. Le résultat
est publiable quel qu'il soit : soit les défenses tiennent, soit on a mesuré
une limite que l'article ne rapporte pas.

**Évaluation à trafic bénin intact.** Ne perturber que les lignes d'attaque.
Aucun calcul GPU, corrige un défaut de modèle de menace.

**Sélection sur le F1 macro** plutôt que sur `val_acc`, et ajout du rappel
BENIGN minimal comme troisième critère de décision.

**Augmenter le budget de C&W.** Avec les 9 itérations de la Table 2, l'attaque
ne converge pas. Elle est la moins destructrice chez nous et la plus
destructrice dans l'article.

**Retirer les six paires de features dupliquées.** Sur 58 features, 52 sont
réellement indépendantes ; les attaques L∞ dépensent une partie de leur budget
sur des dimensions redondantes.

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