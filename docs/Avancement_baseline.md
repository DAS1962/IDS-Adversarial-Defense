

# Point d'avancement — DNN Baseline (Étape 5)

---



**Baseline DNN — Reproduction Awad et al. (2025)**
Point d'avancement stage IDS Adversarial Defense
[Date de la réunion]

---

## Résultats obtenus

- Accuracy globale : **90.69%**
- F1 macro : 47.13%
- F1 weighted : 93.98%
- Papier de référence : **98.11%**



---

## Comportement pendant l'entraînement

- Meilleur modèle atteint à **l'epoch 2**
- Dégradation progressive ensuite
- Accuracy passe de 90% (epoch 2) à 68% (epoch 30)

Le modèle "désapprend" après avoir bien commencé.

---

## Fidèle au papier

- Architecture : 58 → 512 → 256 → 15
- Optimizer : Adam
- Learning rate : 0.01
- Batch size : 128
- Epochs : 30
- SMOTE + StandardScaler + Random Forest features

---

## Décisions face aux ambiguïtés du papier

Le papier ne précise pas :

- Ordre des opérations (split, scale, SMOTE)
- Hyperparamètres SMOTE
- Suppression des doublons
- Random state
- Stratification du split
- Early stopping éventuel

---

## Impact de nos choix

- Suppression de 11% de doublons → moins de données
- SMOTE équilibrage total → train passe à 21M lignes
- 5M itérations d'optimizer sur 30 epochs

Ces choix expliquent probablement l'instabilité observée.

---

## Problème principal : instabilité

- lr=0.01 + 21M lignes = trop d'itérations trop agressives
- Le modèle diverge après epoch 2
- Comment le papier atteint 98% ? Mécanisme non documenté ?

---

## Problème secondaire : SMOTE

- Heartbleed : 7 exemples réels → 1.4M synthétiques
- Modèle prédit "Heartbleed" pour 32k faux positifs
- Littérature : minimum 100 exemples réels recommandé

---

## Solutions envisagées

1. Réduire le learning rate (0.001)
2. Ajouter early stopping
3. Learning rate scheduler
4. Repenser la stratégie SMOTE

---

## Slide 10 — Questions pour vous

1. Rester fidèle au papier ou optimiser ?
2. SMOTE : explorer ou documenter comme limite ?
3. Analyse critique comme angle du rapport ?
4. Comment traiter les classes ultra-rares ?

---

## Plan proposé

- Garder script actuel = "reproduction fidèle"
- Développer version "améliorée"
- Comparer les deux dans le rapport
- Continuer avec étape 6 (attaques adversariales)

---

## Livrables

- GitHub : DAS1962/IDS-Adversarial-Defense
- Étapes 1 à 5 complétées et documentées
- Modèle baseline sauvegardé
- Logs SLURM disponibles
