
"""
Sélection des 58 features les plus importantes avec Random Forest.

Étapes :
  1. Charger le dataset propre (issu du preprocessing)
  2. Séparer X (features) et y (labels)
  3. Entraîner un Random Forest sur le dataset complet (fidèle au papier)
  4. Extraire l'importance de chaque feature
  5. Garder les 58 features les plus importantes
  6. Sauvegarder le dataset réduit et la liste des features retenues

Correspond à l'étape 3 du framework Awad et al. (2025).

Note : entraînement sur les 2.5M lignes complètes pour reproduire fidèlement
la méthodologie du papier. Durée estimée : 20 à 40 minutes sur nibi.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier


# Chemins des dossiers
DATA_PROCESSED_DIR = Path("data/processed")
LOG_DIR = Path("results/logs")

# Paramètres de sélection
N_FEATURES_A_GARDER = 58        # Comme dans le papier Awad et al.
N_ARBRES = 100                  # Nombre d'arbres dans la forêt
RANDOM_STATE = 42               # Pour la reproductibilité


# Fonction 1 : charger le dataset propre
def load_clean_data(data_dir):
    """
    Charge le fichier pickle produit par le preprocessing.
    Retourne un DataFrame avec toutes les 79 colonnes (78 features + label).
    """
    print("Chargement du dataset propre...")
    df = pd.read_pickle(data_dir / "cicids2017_clean.pkl")
    print(f"  Shape : {df.shape[0]:,} lignes × {df.shape[1]} colonnes\n")
    return df


# Fonction 2 : séparer les features (X) et les labels (y)
def separate_features_and_labels(df, label_col="Label"):
    """
    Sépare le DataFrame en deux :
    - X : toutes les colonnes sauf le label (les features)
    - y : uniquement la colonne du label

    Convention en ML : X (majuscule) pour la matrice de features,
    y (minuscule) pour le vecteur cible.
    """
    print("Séparation features / labels...")
    X = df.drop(columns=[label_col])
    y = df[label_col]
    print(f"  Features (X) : {X.shape[1]} colonnes")
    print(f"  Labels (y)   : {len(y):,} valeurs\n")
    return X, y


# Fonction 3 : entraîner le Random Forest et extraire les importances
def compute_feature_importances(X, y, n_arbres, random_state):
    """
    Entraîne un Random Forest sur TOUT le dataset et extrait
    l'importance de chaque feature.

    L'importance est calculée à partir de la réduction moyenne d'impureté
    (Gini) apportée par chaque feature lors des splits dans les arbres.

    Paramètres :
    - n_jobs=-1 : utilise tous les cœurs CPU pour paralléliser
    - verbose=2 : affiche la progression arbre par arbre
    """
    print(f"Entraînement du Random Forest sur le dataset complet")
    print(f"  {X.shape[0]:,} lignes × {X.shape[1]} features")
    print(f"  {n_arbres} arbres")
    print(f"  Durée estimée : 20 à 40 minutes\n")

    debut = datetime.now()

    rf = RandomForestClassifier(
        n_estimators=n_arbres,
        random_state=random_state,
        n_jobs=-1,
        verbose=2,  # Affiche la progression des arbres
    )
    rf.fit(X, y)

    duree = datetime.now() - debut
    print(f"\n  Entraînement terminé en {duree}")

    # Créer une Series pandas triée par importance décroissante
    importances = pd.Series(
        rf.feature_importances_,
        index=X.columns,
    ).sort_values(ascending=False)

    return importances


# Fonction 4 : sélectionner les top N features
def select_top_features(importances, n_features):
    """
    Retourne les noms des n features les plus importantes.
    Affiche aussi les importances pour vérification.
    """
    print(f"\nSélection des {n_features} features les plus importantes...\n")

    top_features = importances.head(n_features).index.tolist()

    # Afficher le top 10 pour aperçu
    print("  Top 10 features :")
    for i, (feature, score) in enumerate(importances.head(10).items(), start=1):
        print(f"    {i:2d}. {feature:<40s} {score:.5f}")

    # Afficher les 5 features les moins importantes parmi celles retenues
    print(f"\n  Features 54-58 (les 5 dernières retenues) :")
    for i, (feature, score) in enumerate(
        importances.iloc[53:58].items(), start=54
    ):
        print(f"    {i:2d}. {feature:<40s} {score:.5f}")

    # Afficher les features rejetées avec les importances les plus faibles
    print(f"\n  5 features rejetées avec l'importance la plus faible :")
    for i, (feature, score) in enumerate(
        importances.iloc[-5:].items(), start=len(importances) - 4
    ):
        print(f"    {i:2d}. {feature:<40s} {score:.5f}")

    # Calculer combien d'importance cumulée représentent les 58 top
    importance_cumulee = importances.head(n_features).sum()
    print(f"\n  Importance cumulée des {n_features} top : {importance_cumulee:.4f}")
    print(f"  (sur un total de 1.0000)")

    print()
    return top_features


# Fonction 5 : réduire le dataset et sauvegarder
def save_reduced_dataset(df, top_features, output_dir, label_col="Label"):
    """
    Garde uniquement les features sélectionnées + le label,
    puis sauvegarde le dataset réduit et la liste des features.
    """
    print("Sauvegarde du dataset réduit...")

    # Créer le DataFrame avec uniquement les colonnes retenues
    cols_a_garder = top_features + [label_col]
    df_reduced = df[cols_a_garder]

    # Sauvegarder le dataset
    data_path = output_dir / "cicids2017_selected.pkl"
    df_reduced.to_pickle(data_path)
    taille_mb = data_path.stat().st_size / (1024 * 1024)
    print(f"  Données : {data_path} ({taille_mb:.1f} MB)")
    print(f"  Shape   : {df_reduced.shape[0]:,} lignes × {df_reduced.shape[1]} colonnes")

    # Sauvegarder la liste des features pour référence
    features_path = output_dir / "selected_features.pkl"
    joblib.dump(top_features, features_path)
    print(f"  Features : {features_path}\n")


# Fonction 6 : sauvegarder un rapport détaillé
def save_report(importances, top_features, log_dir):
    """
    Génère un fichier texte listant toutes les features avec leur importance,
    pour référence future et pour le rapport de stage.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = log_dir / f"feature_selection_{timestamp}.txt"

    with open(report_path, "w") as f:
        f.write("Sélection de features - Random Forest\n")
        f.write(f"Date : {datetime.now()}\n")
        f.write(f"Features gardées : {len(top_features)} / {len(importances)}\n")
        f.write(f"Méthode : Random Forest sur dataset complet (fidèle au papier)\n\n")

        f.write("Importance de toutes les features (triée) :\n")
        for i, (feature, score) in enumerate(importances.items(), start=1):
            marqueur = "GARDEE" if feature in top_features else "REJETEE"
            f.write(f"  {i:3d}. [{marqueur}] {feature:<40s} {score:.6f}\n")

    print(f"Rapport sauvegardé : {report_path}\n")


# Fonction principale
def main():
    print("=" * 60)
    print("Feature Selection avec Random Forest (méthodologie du papier)")
    print(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60 + "\n")

    df = load_clean_data(DATA_PROCESSED_DIR)
    X, y = separate_features_and_labels(df)
    importances = compute_feature_importances(X, y, N_ARBRES, RANDOM_STATE)
    top_features = select_top_features(importances, N_FEATURES_A_GARDER)
    save_reduced_dataset(df, top_features, DATA_PROCESSED_DIR)
    save_report(importances, top_features, LOG_DIR)

    print("=" * 60)
    print("Feature selection terminée")
    print(f"Dataset final : {len(top_features)} features + 1 label")
    print("=" * 60)


if __name__ == "__main__":
    main()