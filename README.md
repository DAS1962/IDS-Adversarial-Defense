# Reproduction fidèle — Awad et al. (2025)

**Branche `reproduction-fidele`.** Reproduction du protocole de l'article tel
qu'il est décrit, y compris quand il est discutable.

> **Awad, Z., Zakaria, M., & Hassan, R. (2025).** *An Enhanced Ensemble Defense
> Framework for Boosting Adversarial Robustness of Intrusion Detection Systems.*
> Scientific Reports, 15, 14177.
> [10.1038/s41598-025-94023-z](https://doi.org/10.1038/s41598-025-94023-z)

---

## Pourquoi cette branche existe

La branche `main` contient une implémentation **méthodologiquement plus
rigoureuse** que l'article : découpage avant toute transformation, SMOTE
plutôt que sous-échantillonnage, sélection des features les plus
discriminantes. Elle atteint 99,79 % d'accuracy sur données propres.

Mais elle s'écarte tellement du protocole d'origine que les chiffres sous
attaque ne sont plus comparables : 32,5 points d'écart moyen avec leur
Table 5, et encore 19,3 après rééquilibrage du test.

Cette branche-ci fait l'inverse. **On applique leur protocole à la lettre**,
même là où il pose problème, pour répondre à une question précise : leurs
résultats sont-ils reproductibles quand on suit exactement ce qu'ils
décrivent ?

**Réponse courte : oui, à 12,7 points près, et quatre attaques sur six sont
reproduites à moins de 3 points.**

---

## Ce qui sépare les deux branches

| | `main` | `reproduction-fidele` |
|---|---|---|
| **Sélection de features** | garde les 58 **plus** importantes (Random Forest) | garde les 58 **moins** importantes (ERT) |
| **Ordre des opérations** | découpage en premier | découpage en **dernier** |
| **Équilibrage** | SMOTE plafonné, sur le train | sous-échantillonnage 50 % par classe, avant découpage |
| **Doublons** | 307 078 retirés | conservés |
| **Test adversarial** | 831 864 lignes, 83,1 % bénin | 20 000 lignes, 50 % bénin |
| **Métrique principale** | F1 macro | accuracy et F1, comme eux |
| Baseline propre | 99,79 % | 96,80 % (fidèle) / 98,42 % (variante lr) |
| Écart moyen à leur Table 5 | 32,5 points | **12,7 points** |

### La sélection de features est inversée, et c'est voulu

C'est l'écart le plus structurant, et il vient d'une relecture attentive de
leur section *Preprocessing* :

> « To ensure that the functionality of the network communication channel is
> preserved […] **the non-functional features that do not matter the most are
> selected** for training and testing purposes. In this context, we use the
> Extremely Randomized Trees (ERTs) classifier **to select the most important
> (functional) features and exclude them**. »

La légende de leur Figure 3 le confirme : *« The functional features to be
removed from the dataset. »*

**Ils gardent les 58 features les moins importantes et suppriment les 20 plus
importantes.** Leur logique : une attaque adversariale doit préserver la
fonctionnalité du trafic réseau. On ne peut pas modifier arbitrairement la
taille des paquets sans casser la communication. L'IDS est donc entraîné
uniquement sur des features qu'un attaquant peut manipuler sans casser son
attaque.

Sur `main`, nous avions gardé exactement ce qu'ils ont jeté.

---

## Décisions prises face à ce que le papier ne documente pas

Six points ne sont pas précisés. Chaque choix est justifié plutôt que subi.

| Point | Ce que dit le papier | Notre décision |
|---|---|---|
| **Ordre des opérations** | Figure 2 : échelle, équilibrage, sélection **puis** découpage | reproduit tel quel, fuite d'information assumée |
| **Mise à l'échelle** | texte : « between 0,1 » ; Figure 2 : « robust scaling, standardization & normalization » | MinMax vers [0,1], seule lecture compatible avec les bornes des attaques |
| **Doublons** | rien | conservés — les retirer corrigerait un problème que leur ordre d'opérations recrée de toute façon |
| **Composition du test** | « 20,000 samples (5000 for each attack) », proportion de bénin non donnée | 50 % bénin, par symétrie avec l'entraînement où le 50/50 est explicite |
| **Alpha, sigma, taille du DAE** | rien | choisis sur la validation, jamais sur le test |
| **Gamma de JSMA** | rien | 1,0 — l'attaque n'est pas bridée |

---

## Pipeline complet

```bash
sbatch scripts/01_pipeline_donnees.sh                      # ~1 min  CPU
sbatch scripts/02_baseline.sh                              # ~6 min  lr 0.01
sbatch scripts/02_baseline_varlr.sh --suffixe _ep150       # ~27 min lr 0.001
sbatch scripts/03_attaques.sh                              # ~15 min substitut + 6 attaques
sbatch scripts/04_evaluation.sh --suffixe _ep150           # ~2 min

sbatch scripts/defense.sh 07_defense_ls  --suffixe _ep150  # ~31 min
sbatch scripts/defense.sh 08_defense_ga  --suffixe _ep150  # ~32 min
sbatch scripts/defense.sh 09_defense_at  --suffixe _ep150  # ~34 min
sbatch scripts/defense.sh 10_defense_dae --suffixe _ep150  # ~7 min

sbatch scripts/12_ensemble.sh --suffixe _ep150             # ~1 min

python scripts/05_tableau_comparaison.py
python scripts/06_figures.py
python scripts/11_tableau_defenses.py  --suffixe _ep150
python scripts/13_figures_defenses.py  --suffixe _ep150
```

Les quatre défenses tournent en parallèle, mais **après** l'évaluation : elles
comparent leurs gains au baseline qu'elle produit.

Le `--suffixe` sépare les séries. Sans lui, les sorties portent les noms de la
série à 100 passages ; avec `_ep150`, celles de la série longue. Rien n'est
écrasé.

Tous les hyperparamètres viennent de `configs/config.yaml`. Aucune valeur
n'est codée en dur dans les scripts.

Environ 2 h 15 de calcul pour tout le pipeline, dont 1 h 45 en parallèle.

## Étape 1 — Pipeline de données

Ordre de leur Figure 2 : nettoyage, encodage, échelle, équilibrage,
sélection, **découpage en dernier**.

Les trois opérations qui dépendent des données arrivent avant le découpage,
donc le jeu de test influence la normalisation, l'équilibrage et la sélection
de features. C'est une fuite d'information sur trois plans. C'est leur
protocole.

| Étape | Résultat |
|---|---|
| Chargement | 2 830 743 lignes, 79 colonnes |
| Nettoyage | 2 867 lignes retirées (Inf et NaN), 307 078 doublons **conservés** |
| Échelle | MinMax vers [0, 1], sur tout le jeu |
| Équilibrage | 50 % par classe, six classes sous 2 000 prises entières → 1 416 027 lignes |
| Sélection ERT | 20 features retirées, 58 conservées |
| Découpage | train 901 301 / val 47 437 / test 467 289 |

**Le test contient 80,2 % de trafic bénin**, contre 83,1 % sur `main`.
L'équilibrage à 50 % ne change presque rien au déséquilibre, puisque les
grosses classes d'attaques sont divisées par deux elles aussi.

### Résultat notable : l'ERT retrouve 18 des 20 features de leur Figure 3

À partir de nos données, sans avoir vu leur liste, l'ERT sélectionne
pratiquement les mêmes features à retirer — et dans un ordre voisin :
`PSH Flag Count` premier chez eux comme chez nous, `Destination Port`
deuxième.

| | Chez eux seulement | Chez nous seulement |
|---|---|---|
| Écarts | `idle_max`, `min_seg_size_forward` | `Fwd IAT Std`, `Fwd Packets/s` |

Les deux écarts portent sur des features de rang 16 à 20, à la frontière de
la coupure. **Leur méthode de sélection est donc reproductible.**

---

## Étape 2 — Baselines

Architecture de leur Table 3 : 58 → 512 → 256 → 15, ReLU, Adam, batch 128.
165 391 paramètres.

Trois entraînements, ne différant que par le learning rate et la durée.

| | lr 0.01, 30 ep | lr 0.001, 30 ep | **lr 0.001, 100 ep** |
|---|---:|---:|---:|
| Accuracy | 96,80 % | 97,24 % | **98,42 %** |
| F1 pondéré | 96,71 % | 97,20 % | **98,41 %** |
| **F1 macro** | 57,72 % | 69,47 % | **77,68 %** |
| Meilleur passage | 21 | 29 | 93 |
| Écart à leur 98,11 % | −1,31 | −0,87 | **+0,31** |

### Leur learning rate ne converge pas

Avec lr 0.01, `val_acc` atteint 0,9664 **dès le passage 3**, puis oscille
entre 0,943 et 0,970 pendant vingt-sept passages sans progresser. Il y a même
un décrochage au passage 10. Le modèle rebondit, il ne converge pas.

Le même comportement avait été observé sur `main`, et se reproduit sur le
substitut. Trois observations indépendantes.

**Vingt points de F1 macro perdus pour un seul hyperparamètre.** Et le gain
vient entièrement des classes rares :

| Classe | lr 0.01 | lr 0.001, 100 ep |
|---|---:|---:|
| Web Attack Brute Force | 2,2 % | **89,3 %** |
| Heartbleed | 0,0 % | **75,0 %** |
| Infiltration | 0,0 % | **66,7 %** |
| Bot | 35,6 % | 59,1 % |
| PortScan | 93,4 % | 99,9 % |

Quatre classes passent de zéro à détectables, pendant que l'accuracy ne bouge
que de 1,6 point. **Un détecteur peut afficher 96,8 % tout en étant aveugle à
quatre types d'attaques sur quatorze**, et aucune des deux métriques que
publie l'article ne le verrait.

Deux classes restent bloquées quel que soit le réglage : SSH-Patator à 48,4 %
et Web Attack XSS à 2,3 %. Leur signal était probablement porté par les
features retirées.


---

## Étape 3 — Substitut et attaques

L'article génère les exemples adversariaux sur un **modèle substitut**, pas
sur le détecteur. L'attaquant connaît l'architecture générale mais pas les
poids : c'est le scénario **semi-white box**, qu'ils jugent réaliste.

Substitut : 58 → 100 → 100 → 15, lr 0.01, 30 passages, batch 256.
17 515 paramètres.

**Son F1 pondéré atteint 0,9700, contre la cible de 0,98 annoncée.** Il
plafonne exactement comme le baseline avec le même learning rate : 0,9662 au
passage 10, 0,9700 au passage 30. Quatrième observation du même phénomène.

### Deux échantillons, comme dans leur section *Adversarial examples generation*

> « From the training dataset, we randomly select 40,000 samples (10,000 for
> each attack) of which 20,000 represent regular traffic and the remaining
> represent intrusions. From the testing data set we select 20,000 samples
> (5000 for each attack). »

| Échantillon | Source | Taille | Bénin | Usage |
|---|---|---:|---:|---|
| entraînement | train | 40 000 | 50 % | alimentera les défenses |
| évaluation | test | 20 000 | 50 % | mesure la vulnérabilité |

**Les deux jeux sont disjoints** — l'un vient du train, l'autre du test.
C'est ce qui évite le problème rencontré sur `main`, où une défense entraînée
sur les mêmes perturbations que celles de l'évaluation atteignait un F1 de
0,9651 sous BIM, supérieur à son F1 sur données propres : de la mémorisation
de distribution, pas de la robustesse.

Ces chiffres confirment aussi que **leur protocole porte sur quatre attaques**
(10 000 × 4 = 40 000), alors que leur Table 5 en évalue six.

### Amplitude des perturbations, paramètres de leur Table 2

| Attaque | L∞ moyen | L2 moyen | Features touchées | Durée |
|---|---:|---:|---:|---:|
| FGSM | 0,1681 | 0,9563 | 35,2 / 58 | < 1 min |
| BIM | 0,1780 | 0,8658 | 36,6 / 58 | < 1 min |
| PGD | 0,2954 | 1,2772 | 42,2 / 58 | < 1 min |
| DeepFool | 0,0078 | 0,0130 | 40,7 / 58 | 1,1 min |
| **JSMA** | **0,9254** | **6,4036** | 50,7 / 58 | 1,3 min |
| C&W | 0,0696 | 0,0860 | **10,5 / 58** | 1,8 min |

JSMA perturbe cinq cents fois plus que DeepFool. C&W ne touche que dix
features : avec les 9 itérations de leur Table 2, l'attaque reste très
localisée et ne converge pas.

Par rapport à `main`, FGSM et BIM perturbent bien plus fort (L2 de 0,61 à
0,96 pour FGSM). C'est l'effet des features non fonctionnelles : sans les
colonnes discriminantes, il faut pousser plus loin pour faire basculer une
décision.

---

## Étape 4 — Évaluation, comparaison à leur Table 5

Test de 20 000 lignes à 50 % de bénin, **donc plancher d'accuracy à 50 %**.
C'est la première configuration du projet directement comparable à leurs
chiffres sans correction après coup.

### Accuracy

| Attaque | Papier | Fidèle | Variante lr | F−P | V−P |
|---|---:|---:|---:|---:|---:|
| Clean | 98,11 % | 88,63 % | 94,16 % | −9,5 | −4,0 |
| FGSM | 54,5 % | 51,79 % | 47,86 % | −2,7 | −6,6 |
| **BIM** | 45,0 % | 51,23 % | **46,58 %** | +6,2 | **+1,6** |
| **PGD** | 46,0 % | 49,82 % | **48,20 %** | +3,8 | **+2,2** |
| **DeepFool** | 53,0 % | 68,17 % | **52,51 %** | +15,2 | **−0,5** |
| JSMA | 81,0 % | 50,09 % | 49,56 % | −30,9 | −31,4 |
| C&W | 36,0 % | 68,78 % | 70,03 % | +32,8 | +34,0 |

**Écart absolu moyen sur les six attaques : 15,3 points en fidèle, 12,7 en
variante.** Pour mémoire, 32,5 sur `main` et 19,3 après rééquilibrage du
test.

**Quatre attaques sur six sont reproduites à moins de 3 points.** DeepFool à
un demi-point, BIM à 1,6, PGD à 2,2, FGSM à 2,7.

### Les deux attaques qui divergent

**JSMA : 49,56 % contre leurs 81 %.** Avec un L2 de 6,40 et 50 features
touchées sur 58, notre JSMA est dévastateur. Le leur, à 81 % d'accuracy,
serait quasiment inoffensif. Soit ils l'ont bridé — leur `gamma` n'est pas
documenté et nous avons pris 1,0 —, soit leur implémentation avait un
problème. Avec `theta=0.1` et une attaque non bridée, il est mathématiquement
difficile que JSMA soit si peu destructeur.

**C&W : 70,03 % contre leurs 36 %.** Le nôtre ne touche que 10,5 features
avec un L2 de 0,085 : il est sous-itéré avec les 9 itérations de leur
Table 2. Le leur, à 36 %, est le plus destructeur des six — incompatible avec
9 itérations d'une attaque qui cherche la perturbation minimale.

### Leur métrique est probablement le F1 micro

Écarts absolus moyens avec leurs chiffres, six attaques :

| Métrique | Fidèle | Variante |
|---|---:|---:|
| **F1 micro** | **13,5** | 14,7 |
| Accuracy | 15,3 | 12,7 |
| F1 pondéré | 22,3 | 25,0 |
| F1 macro | 41,2 | 44,3 |

Le F1 micro est celui qui colle le mieux. Or **en multiclasse, le F1 micro
est égal à l'accuracy** — nos colonnes le vérifient, les deux valeurs sont
identiques partout. Leur Table 4 le confirme : 98,11 / 98,11 / 98,11 /
98,068, quatre valeurs quasi identiques.

**Ils publient donc deux fois la même information.** Leur « F1-score »
n'apporte rien à leur accuracy.

Leur precision et leur recall ne sont en revanche pas cohérents entre eux :
sous DeepFool ils donnent 67 % et 53 %, un écart de 14 points qu'aucun type
de moyenne ne reproduit. Leur Eq. 7 donne d'ailleurs une formule de precision
erronée, `TP/(TP+TN)` au lieu de `TP/(TP+FP)`.

### Le paradoxe robustesse/précision, mesuré

Sous DeepFool, la version fidèle donne 68,17 % et la variante 52,51 %. **Le
meilleur détecteur résiste moins bien.**

C'est logique : DeepFool cherche la perturbation minimale pour franchir la
frontière de décision. Un modèle mieux entraîné a des frontières plus nettes,
donc plus faciles à cibler précisément. C'est le résultat de Tsipras et al.
(2019), observé ici sur du trafic réseau.

### Rappel sur le trafic normal — pas publié par l'article

| Attaque | Fidèle | Variante |
|---|---:|---:|
| FGSM | 99,79 % | 91,94 % |
| BIM | 98,69 % | 89,38 % |
| PGD | 99,64 % | 96,38 % |
| DeepFool | 95,87 % | 94,10 % |
| JSMA | 99,96 % | 98,73 % |
| C&W | 91,34 % | 91,48 % |

Le modèle fidèle garde 99,79 % sous FGSM **parce qu'il ne détecte presque
rien** : il classe tout en bénin, donc son rappel sur cette classe est
excellent. C'est encore le plancher qui parle.

---

## Étape 5 — Les quatre défenses

Algorithmes 2 à 5. **Les titres des Algorithmes 3 et 4 sont intervertis dans
l'article** : le 3 est intitulé « Gaussian Augmentation » mais décrit
l'entraînement adversarial, le 4 l'inverse. On suit le contenu.

Le lissage est appliqué aux trois défenses d'entraînement, comme le
prescrivent les Algorithmes 2, 3 et 4 (« and smooth train labels »). Pas au
DAE, qui n'a pas d'étiquettes à lisser puisqu'il optimise une MSE de
reconstruction.

**Point clé du protocole.** Les 40 000 exemples adversariaux d'entraînement
viennent du **train**, les 20 000 d'évaluation du **test**. Les deux jeux
sont disjoints, donc aucune mémorisation n'est possible. C'est ce qui
manquait sur `main`, où une défense entraînée sur la même distribution que
l'évaluation atteignait un F1 de 0,9651 sous BIM, supérieur à son F1 sur
données propres.

### Résultats, 150 passages

| Défense | Gain | Coût propre | Bilan net | Rappel BENIGN min |
|---|---:|---:|---:|---:|
| **Adversarial** | **+0,5001** | −0,0483 | **+0,4517** | 0,9014 |
| Autoencodeur | +0,0511 | −0,4662 | −0,4151 | **0,9527** |
| Gaussienne (σ=0,01) | +0,0436 | −0,2637 | −0,2201 | 0,9297 |
| Lissage | +0,0161 | −0,0459 | −0,0298 | 0,7495 |

Le gain est la moyenne des écarts au baseline sur les six attaques, le coût
l'écart sur données propres, le bilan net leur somme.

### Le classement est l'inverse du leur

| Rang | Papier (Table 7) | Nous (bilan net) |
|---|---|---|
| 1 | Label Smoothing 85,90 % | **Adversarial** +0,4517 |
| 2 | Denoising Autoencoder 84,80 % | Lissage −0,0298 |
| 3 | Adversarial Training 80,25 % | Gaussienne −0,2201 |
| 4 | Gaussian Augmentation 79,80 % | Autoencodeur −0,4151 |

### Face à leur Table 7, en moyenne micro sur les sept jeux

| Défense | Papier | Nous | Écart |
|---|---:|---:|---:|
| **Adversarial** | 80,25 % | **85,75 %** | **+5,5** |
| Gaussienne | 79,80 % | 62,72 % | −17,1 |
| Autoencodeur | 84,80 % | 60,21 % | −24,6 |
| Lissage | 85,90 % | 57,19 % | −28,7 |

**L'entraînement adversarial est le seul point du projet où nous dépassons
leurs chiffres**, sur leur protocole et leur métrique.

### Face à leur Table 9, MCC binaire

| Défense | Papier | Nous, propre | Nous, pire attaque |
|---|---:|---:|---:|
| Adversarial | 0,608 | **0,878** | **0,280** |
| Lissage | 0,698 | 0,889 | **−0,105** |
| Autoencodeur | 0,680 | 0,417 | 0,106 |
| Gaussienne | 0,596 | 0,657 | 0,053 |

**Le MCC du lissage devient négatif sous FGSM.** Le modèle fait alors pire
qu'un tirage au sort pour distinguer attaque et trafic normal. Aucune des
métriques du papier ne le montrerait.

### Adversarial : le seul gain qui généralise

PGD et C&W ne figurent pas dans les quatre attaques d'entraînement. Leur gain
mesure donc une vraie généralisation, pas une reconnaissance de forme.

**Gain sur PGD et C&W seuls : +0,2735.** PGD passe de 0,0442 à 0,5912.

### Gaussienne : sigma doit être à l'échelle des données

L'écart-type moyen des 58 features non fonctionnelles vaut **0,0537**, contre
0,103 sur `main`. Le même sigma représente donc deux fois plus de bruit
relatif.

| sigma | Gain | F1 propre | Bilan net | Rappel BENIGN min |
|---:|---:|---:|---:|---:|
| 0,02 (37 % du signal) | −0,0015 | 0,4159 | −0,3830 | **0,0458** |
| **0,01 (19 % du signal)** | **+0,0436** | **0,5337** | **−0,2201** | **0,9297** |

À sigma 0,02, le modèle **rejette 95 % du trafic légitime sous JSMA**.
Diviser sigma par deux fait disparaître l'effondrement.

Mais le modèle plafonne dans les deux cas — meilleur passage au 14 puis au 19
sur 150. Le niveau de bruit n'est donc pas la cause du plafonnement.

Et son gain vient presque entièrement de DeepFool (+0,2334 sur +0,0436), où
son F1 sous attaque (0,5329) est **identique à son F1 propre** (0,5337) : la
perturbation de DeepFool est plus petite que le bruit d'entraînement, donc
l'attaque devient invisible. Ce n'est pas de la défense.

### Autoencodeur : une MSE basse ne suffit pas

Le DAE reconstruit presque parfaitement une entrée saine — MSE(x, DAE(x)) =
0,000029, soit **0,0 % de la variance**. Et pourtant le F1 macro chute de
0,7974 à 0,3313.

La raison : **12 features sur 58 perdent plus de la moitié de leur
écart-type**. Le DAE préserve l'énergie globale du signal mais écrase la
variabilité de douze colonnes, précisément celle qui distingue les classes.

Sa fonction de coût est par ailleurs dominée par JSMA, dont la corruption
vaut 0,781344 contre 0,000012 pour DeepFool — un rapport de 65 000. Il
consacre tout son budget à une attaque qu'il n'arrive pas à corriger.

## Étape 6 — Agrégation par ensemble

Le papier donne deux règles de fusion et leurs versions optimisées. Son « soft
voting » est la moyenne des probabilités à poids égaux, et c'est elle qui lui
donne son meilleur résultat.

| Méthode | Gain | Coût | Bilan net | Accuracy moy. | Papier |
|---|---:|---:|---:|---:|---:|
| Moyenne optimisée | +0,1292 | −0,0540 | **+0,0753** | 67,67 % | 86,11 % |
| Vote majoritaire | +0,0952 | −0,1793 | −0,0841 | 65,63 % | 84,35 % |
| Poids égaux | +0,0939 | −0,1794 | −0,0855 | 65,56 % | 87,49 % |

**Leur proposition centrale ne se vérifie pas ici.** L'adversarial seul
(+0,4517) fait six fois mieux que le meilleur ensemble (+0,0753).

La raison est mécanique : un ensemble ne dépasse ses membres que s'ils sont de
qualité comparable et se trompent différemment. Ici trois membres sur quatre
ne détectent presque rien, donc l'agrégation noie le seul qui fonctionne.
Sous BIM, l'adversarial atteint 0,9778 seul mais l'ensemble tombe à 0,1816 :
les trois autres votent contre lui et l'emportent.

**Sur `main`, où les quatre défenses étaient de niveau voisin, on obtenait le
résultat inverse.** La conclusion du papier dépend donc de la qualité relative
des membres, pas de l'agrégation en elle-même.

L'optimisation des poids le montre bien : elle retient lissage 0,4608 et
adversarial 0,4608, écartant la gaussienne à 0,0196. Mais elle optimise sur
la validation **propre**, où lissage et adversarial ont des F1 quasi
identiques (0,7515 et 0,7491). Elle ne peut pas voir que l'adversarial vaut
dix fois plus sous attaque.

*Écart à noter : le papier optimise l'**accuracy** par optimisation
bayésienne, 50 itérations et 100 points initiaux. Nous optimisons le F1 macro
par Nelder-Mead. Maximiser l'accuracy récompenserait un ensemble classant
tout en bénin, puisqu'elle est plafonnée par la proportion de trafic normal.*

**Un bénéfice réel malgré tout** : le rappel BENIGN minimal passe de 0,7495
pour le lissage seul à 0,9577 pour l'ensemble. L'agrégation protège contre
l'effondrement d'un membre, même si elle n'améliore pas la détection.

## Effet de la durée d'entraînement

Deux séries complètes, à 100 et 150 passages, tout le reste identique.

| | Bilan à 100 | Bilan à 150 | Écart |
|---|---:|---:|---:|
| Adversarial | +0,4524 | +0,4517 | **−0,0007** |
| Lissage | −0,0262 | −0,0298 | −0,0036 |
| Gaussienne | −0,3830 | −0,2201 | +0,1629 |
| Autoencodeur | −0,5585 | −0,4151 | +0,1434 |

**La durée ne change rien.** L'adversarial bouge de 0,0007, le lissage de
0,0036. Les deux écarts importants viennent d'ailleurs : la gaussienne du
sigma divisé par deux, l'autoencodeur d'un meilleur passage trouvé par hasard
dans un F1 de validation qui oscille sans converger.

Le baseline lui-même donne des chiffres **rigoureusement identiques** à 100 et
150 passages — accuracy 98,42 %, F1 macro 77,68 %, meilleur passage 93 dans
les deux cas. Il avait convergé.

## Incohérences relevées dans l'article

Listées par souci de complétude, elles relativisent la fiabilité de ses
chiffres.

**Formule de precision erronée**, Eq. (7) : `Precision = TP/(TP+TN)`. Le
dénominateur correct est `TP+FP`.

**Trois valeurs qui ne concordent pas.** L'abstract annonce 87,34 % en
majority voting, la Table 8 donne 87,49 %, la discussion 84,34 %. La Table 4
donne 98,11 % et la Figure 5 98,105 %. Le texte cite « MCC values of 0.608
and 0.640 » pour le DAE quand la Table 9 affiche 0,680.

**Algorithmes 3 et 4 intervertis** : leurs titres ne correspondent pas à leur
contenu.

**Quatre attaques ou six ?** Les Algorithmes 2 à 5 en utilisent quatre, la
Table 5 en évalue six, la Table 10 n'en liste que quatre.

**Black box annoncé, jamais rapporté** : « All the defense mechanisms are
evaluated in semi-white box and black box settings », mais aucun résultat ne
distingue les deux scénarios.

**Matériel** : Google Colab sur un i7 à 2,70 GHz avec 8 Go de RAM, ce qui
explique les sous-échantillons de 20 000 et 40 000 lignes sur un jeu de
2,8 millions.

---

## Limites de cette reproduction

**Une seule exécution par configuration.** Pas de variance mesurée, donc les
écarts inférieurs à quelques points ne sont pas départageables.

**Aucun attaquant adaptatif.** Les attaques sont générées une fois sur le
substitut. Un attaquant réel face à un IDS défendu entraînerait son substitut
contre le modèle défendu. L'article a la même limite.

**Le trafic bénin est perturbé lui aussi**, alors qu'un attaquant ne contrôle
que son propre trafic.

**Deux hyperparamètres ne sont pas reproductibles à l'identique** : le
`gamma` de JSMA et les paramètres internes de C&W, non documentés. Ce sont
précisément les deux attaques qui divergent.

---

## Fichiers produits

```
results/
├── checkpoints/
│   ├── baseline.pth                  lr 0.01, 30 passages
│   ├── baseline_varlr_ep150.pth      lr 0.001, 150 passages
│   ├── substitut.pth                 58-100-100-15
│   └── defense_{ls,ga,at,dae}_ep150.pth
├── attacks/
│   ├── X_{train,test}_clean.pkl      40 000 et 20 000 lignes, 50 % bénin
│   └── X_adv_{train,test}_*.pkl      6 attaques × 2 échantillons
├── logs/                             historiques et résultats de chaque run
└── figures/
    ├── baselines_courbes.png         les trois entraînements du détecteur
    ├── accuracy_trois_colonnes.png   papier / fidèle / variante
    ├── ecarts_papier.png             écart par attaque
    ├── f1_types_moyenne.png          macro / pondéré / micro
    ├── heatmap_metriques.png         toutes les métriques, deux baselines
    ├── rappel_benign.png             trafic normal, baselines
    ├── perturbations.png             amplitude des six attaques
    ├── defenses_f1_ep150.png         F1 macro par attaque, 4 défenses
    ├── defenses_gain_cout_ep150.png  le compromis de chaque défense
    ├── defenses_vs_papier_ep150.png  face à leur Table 7
    ├── defenses_mcc_ep150.png        face à leur Table 9
    ├── defenses_recall_ep150.png     rappel BENIGN, 4 défenses
    ├── defenses_heatmap_ep150.png    tout, défenses et ensemble
    ├── defenses_courbes_ep150.png    entraînement des 3 défenses
    ├── evaluation_baselines_ep150.csv
    ├── comparaison_papier.csv
    ├── defenses_completes_ep150.csv
    └── ensemble_complet_ep150.csv
```

**Aucun chiffre de ce document n'est recopié à la main.** Ils viennent tous
des CSV, produits par les scripts `04`, `11` et `12`.

## Environnement

Alliance Canada narval, Python 3.11, GPU A100 via SLURM. Environ 45 minutes
de calcul pour tout le pipeline.

Dépendances : PyTorch 2.5.1, scikit-learn, torchattacks 3.5.1,
adversarial-robustness-toolbox 1.18, imbalanced-learn.

## Références

- Awad et al. (2025). *Enhanced Ensemble Defense Framework.* Scientific Reports 15:14177.
- Goodfellow et al. (2014). *Explaining and Harnessing Adversarial Examples.* ICLR.
- Kurakin et al. (2016). *Adversarial Machine Learning at Scale.* ICLR.
- Madry et al. (2017). *Towards Deep Learning Models Resistant to Adversarial Attacks.* ICLR.
- Moosavi-Dezfooli et al. (2015). *DeepFool.* CVPR.
- Papernot et al. (2015). *The Limitations of Deep Learning in Adversarial Settings.* IEEE EuroS&P.
- Carlini & Wagner (2017). *Towards Evaluating the Robustness of Neural Networks.* IEEE S&P.
- Tsipras et al. (2019). *Robustness May Be at Odds with Accuracy.* ICLR.
- Geurts et al. (2006). *Extremely Randomized Trees.* Machine Learning.
