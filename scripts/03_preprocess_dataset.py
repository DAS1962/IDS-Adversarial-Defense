
"""
Preprocessing du dataset CIC-IDS 2017.

Nettoie les données pour préparer les étapes suivantes :
  1. Retire les espaces parasites dans les noms de colonnes
  2. Remplace les valeurs infinies par NaN puis supprime les lignes concernées
  3. Supprime les doublons
  4. Encode les labels textuels en entiers
  5. Sauvegarde le dataset propre

Note : la normalisation (StandardScaler) et l'équilibrage (SMOTE) seront
faits plus tard, APRES la séparation train/test, pour éviter le data leakage.

Correspond à l'étape 2 du framework Awad et al. (2025).
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import LabelEncoder


# Chemins des dossiers
DATA_RAW_DIR = Path("data/raw")
DATA_PROCESSED_DIR = Path("data/processed")


# Fonction 1 : charger et concaténer tous les fichiers CSV
def load_all_csv(data_dir):
    """
    Charge tous les fichiers CSV et les concatène en un seul DataFrame.
    Même logique que dans le script d'exploration, mais ici on veut
    modifier les données ensuite.
    """
    print("Chargement des fichiers CSV...")
    csv_files = sorted(data_dir.glob("*.csv"))

    dataframes = []
    for f in csv_files:
        df = pd.read_csv(f, low_memory=False)
        print(f"  {f.name} : {df.shape[0]:,} lignes")
        dataframes.append(df)

    df_all = pd.concat(dataframes, ignore_index=True)
    print(f"Total : {df_all.shape[0]:,} lignes, {df_all.shape[1]} colonnes\n")
    return df_all


# Fonction 2 : nettoyer les noms de colonnes
def clean_column_names(df):
    """
    Retire les espaces au début et à la fin des noms de colonnes.

    Problème observé : 65 colonnes commencent par un espace
    (ex: ' Destination Port' au lieu de 'Destination Port').
    Cause : CICFlowMeter utilise ', ' comme séparateur au lieu de ','.
    """
    print("Nettoyage des noms de colonnes...")

    # Compter les colonnes problématiques avant modification
    n_avant = sum(1 for c in df.columns if c != c.strip())

    # Appliquer strip() à tous les noms
    df.columns = df.columns.str.strip()

    print(f"  {n_avant} colonnes nettoyées (espaces retirés)\n")
    return df


# Fonction 3 : traiter les valeurs infinies et manquantes
def handle_missing_and_infinite(df):
    """
    Remplace les valeurs infinies par NaN, puis supprime les lignes concernées.

    Problème observé : environ 4400 lignes ont des Inf ou NaN dans les
    colonnes 'Flow Bytes/s' et 'Flow Packets/s' (division par zéro).
    Un DNN plante avec des NaN dans ses inputs (loss devient NaN).
    On perd 0.16% du dataset, négligeable.
    """
    print("Traitement des valeurs infinies et manquantes...")
    n_avant = len(df)

    # Étape 1 : convertir tous les Inf en NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Étape 2 : supprimer les lignes avec au moins un NaN
    df = df.dropna()

    n_apres = len(df)
    n_supprimees = n_avant - n_apres
    pct = 100 * n_supprimees / n_avant

    print(f"  Lignes avant  : {n_avant:,}")
    print(f"  Lignes après  : {n_apres:,}")
    print(f"  Supprimées    : {n_supprimees:,} ({pct:.3f}%)\n")

    return df


# Fonction 4 : supprimer les doublons
def remove_duplicates(df):
    """
    Supprime les lignes en double.

    Pourquoi c'est important : les doublons peuvent gonfler artificiellement
    certaines classes et créer du data leakage (une ligne identique dans
    train et test → le modèle "triche").
    """
    print("Suppression des doublons...")
    n_avant = len(df)

    df = df.drop_duplicates()

    n_apres = len(df)
    n_doublons = n_avant - n_apres
    pct = 100 * n_doublons / n_avant

    print(f"  Doublons supprimés : {n_doublons:,} ({pct:.3f}%)")
    print(f"  Lignes restantes   : {n_apres:,}\n")

    return df


# Fonction 5 : encoder les labels
def encode_labels(df, label_col):
    """
    Convertit les labels textuels en entiers.

    Un DNN travaille avec des nombres, pas des chaînes. LabelEncoder assigne
    un entier à chaque classe par ordre alphabétique :
    BENIGN → 0, Bot → 1, DDoS → 2, etc.

    On retourne aussi l'encodeur pour pouvoir décoder plus tard
    (utile quand on affiche les résultats aux utilisateurs).
    """
    print("Encodage des labels...")
    encoder = LabelEncoder()
    df[label_col] = encoder.fit_transform(df[label_col])

    # Afficher la correspondance
    print(f"  {len(encoder.classes_)} classes encodées :")
    for idx, nom in enumerate(encoder.classes_):
        print(f"    {idx:2d} -> {nom}")
    print()

    return df, encoder


# Fonction 6 : sauvegarder les données propres
def save_cleaned_data(df, encoder, output_dir):
    """
    Sauvegarde le DataFrame propre et l'encodeur.

    Format Pickle : beaucoup plus rapide que CSV pour lire/écrire (10x à 100x),
    et conserve les types de données exactement. Fichier binaire donc pas
    lisible dans un éditeur de texte, mais ce n'est pas grave puisque c'est
    pour la machine.
    """
    print("Sauvegarde des données propres...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sauvegarder le DataFrame
    data_path = output_dir / "cicids2017_clean.pkl"
    df.to_pickle(data_path)
    taille_mb = data_path.stat().st_size / (1024 * 1024)
    print(f"  Données : {data_path} ({taille_mb:.1f} MB)")

    # Sauvegarder l'encodeur pour usage futur
    encoder_path = output_dir / "label_encoder.pkl"
    joblib.dump(encoder, encoder_path)
    print(f"  Encoder : {encoder_path}\n")


# Fonction principale : enchaîne toutes les étapes
def main():
    print("=" * 60)
    print("Preprocessing du dataset CIC-IDS 2017")
    print(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60 + "\n")

    df = load_all_csv(DATA_RAW_DIR)
    df = clean_column_names(df)
    df = handle_missing_and_infinite(df)
    df = remove_duplicates(df)
    df, encoder = encode_labels(df, label_col="Label")
    save_cleaned_data(df, encoder, DATA_PROCESSED_DIR)

    print("=" * 60)
    print("Preprocessing terminé")
    print(f"Dataset final : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
    print("=" * 60)


if __name__ == "__main__":
    main()