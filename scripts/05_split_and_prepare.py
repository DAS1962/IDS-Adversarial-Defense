
"""
Split train/test + Normalisation + SMOTE.

Cette etape combine trois operations dans un ordre precis :
  1. Split stratifie train/test (67/33 comme le papier)
  2. Normalisation avec StandardScaler (fit sur train, transform sur les deux)
  3. SMOTE sur le train avec strategie custom (cap par classe)

La strategie SMOTE custom evite la sur-generation extreme sur les classes
ultra-rares (Heartbleed avec 7 exemples reels donnait 1.4M synthetiques
en version equilibrage total). Ici on plafonne les ratios d'expansion pour
un equilibre entre representation des classes rares et fidelite statistique.


Ordre crucial pour eviter le data leakage :
  - Normaliser AVANT le split leak les stats du test dans le train
  - SMOTE AVANT le split cree des exemples synthetiques du test dans le train
  - Toute transformation dependante des donnees s'apprend sur le train seul

Correspond a l'etape 4 du framework Awad et al. (2025).
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


DATA_PROCESSED_DIR = Path("data/processed")
LOG_DIR = Path("results/logs")

TEST_SIZE = 0.33
RANDOM_STATE = 42

# Strategie SMOTE : cap par classe pour eviter la sur-generation
# Cle = ID de classe (apres LabelEncoder), Valeur = nombre cible apres SMOTE
# Les classes non listees gardent leur nombre actuel
SMOTE_STRATEGY = {
    3: 50_000,    # DoS GoldenEye (etait 6,891)
    7: 50_000,    # FTP-Patator (3,974)
    6: 50_000,    # DoS slowloris (3,608)
    5: 50_000,    # DoS Slowhttptest (3,503)
    11: 30_000,   # SSH-Patator (2,157)
    1: 30_000,    # Bot (1,305)
    12: 10_000,   # Web Attack Brute Force (985)
    14: 5_000,    # Web Attack XSS (437)
    9: 1_000,     # Infiltration (24) - cap raisonnable
    13: 1_000,    # Web Attack SQL Injection (14)
    8: 500,       # Heartbleed (7) - cap tres bas car ultra-rare
}


def load_selected_data(data_dir):
    """Charge le dataset avec les 58 features selectionnees."""
    print("Chargement du dataset selectionne...")
    df = pd.read_pickle(data_dir / "cicids2017_selected.pkl")
    print(f"  Shape : {df.shape[0]:,} lignes x {df.shape[1]} colonnes\n")
    return df


def split_train_test(df, test_size, random_state, label_col="Label"):
    """Split stratifie 67/33."""
    print(f"Split train/test ({int((1-test_size)*100)}/{int(test_size*100)}, stratifie)...")

    X = df.drop(columns=[label_col])
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    print(f"  Train : {len(X_train):,} lignes ({100*len(X_train)/len(X):.1f}%)")
    print(f"  Test  : {len(X_test):,} lignes ({100*len(X_test)/len(X):.1f}%)\n")

    return X_train, X_test, y_train, y_test


def normalize_features(X_train, X_test):
    """StandardScaler fit sur train uniquement, transform sur les deux."""
    print("Normalisation des features (StandardScaler)...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    print(f"  Train : mean={X_train_scaled.mean().mean():.4f}, std={X_train_scaled.std().mean():.4f}")
    print(f"  Test  : mean={X_test_scaled.mean().mean():.4f}, std={X_test_scaled.std().mean():.4f}\n")

    return X_train_scaled, X_test_scaled, scaler


def apply_smote(X_train, y_train, random_state):
    """
    Applique SMOTE avec une strategie custom (cap par classe).

    Contrairement a l'equilibrage total qui monte toutes les classes au
    niveau de la majoritaire, on plafonne l'expansion pour eviter de creer
    des millions d'interpolations bruitees a partir de quelques exemples reels.
    """
    print("Application de SMOTE avec strategie custom...")

    print(f"  Distribution AVANT SMOTE :")
    counts_avant = y_train.value_counts().sort_index()
    for classe, count in counts_avant.items():
        cible = SMOTE_STRATEGY.get(classe, count)
        marqueur = f"-> {cible:,}" if cible != count else "(inchange)"
        print(f"    Classe {classe:2d} : {count:>10,}  {marqueur}")

    # Ajuster k_neighbors pour les tres petites classes
    # SMOTE necessite k_neighbors + 1 exemples minimum par classe
    n_min = counts_avant.min()
    k_neighbors = min(5, n_min - 1)

    print(f"\n  k_neighbors utilise : {k_neighbors}")
    print(f"  (classe la plus petite : {n_min} exemples)\n")

    smote = SMOTE(
        sampling_strategy=SMOTE_STRATEGY,
        random_state=random_state,
        k_neighbors=k_neighbors,
    )
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    print(f"  Distribution APRES SMOTE :")
    counts_apres = pd.Series(y_train_bal).value_counts().sort_index()
    for classe, count in counts_apres.items():
        print(f"    Classe {classe:2d} : {count:>10,}")
    print(f"    Total       : {len(y_train_bal):>10,}")

    n_synthetiques = len(y_train_bal) - len(y_train)
    print(f"\n  Exemples synthetiques ajoutes : {n_synthetiques:,}")
    print(f"  Ratio expansion global : {len(y_train_bal)/len(y_train):.2f}x\n")

    return X_train_bal, y_train_bal


def save_artifacts(X_train, X_test, y_train, y_test, scaler, output_dir):
    """Sauvegarde des fichiers."""
    print("Sauvegarde des artefacts...")
    output_dir.mkdir(parents=True, exist_ok=True)

    fichiers = {
        "X_train.pkl": X_train,
        "X_test.pkl": X_test,
        "y_train.pkl": y_train,
        "y_test.pkl": y_test,
    }

    for nom, obj in fichiers.items():
        path = output_dir / nom
        if isinstance(obj, pd.DataFrame) or isinstance(obj, pd.Series):
            obj.to_pickle(path)
        else:
            joblib.dump(obj, path)
        taille_mb = path.stat().st_size / (1024 * 1024)
        print(f"  {nom:20s} : {taille_mb:>7.1f} MB")

    scaler_path = output_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"  scaler.pkl           : {scaler_path.stat().st_size / 1024:.1f} KB\n")


def main():
    print("Split train/test + Normalisation + SMOTE (v3, strategie custom)")
    print(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    df = load_selected_data(DATA_PROCESSED_DIR)
    X_train, X_test, y_train, y_test = split_train_test(df, TEST_SIZE, RANDOM_STATE)
    X_train, X_test, scaler = normalize_features(X_train, X_test)
    X_train, y_train = apply_smote(X_train, y_train, RANDOM_STATE)
    save_artifacts(X_train, X_test, y_train, y_test, scaler, DATA_PROCESSED_DIR)

    print(f"Train final : {len(X_train):,} lignes x {X_train.shape[1]} features")
    print(f"Test final  : {len(X_test):,} lignes x {X_test.shape[1]} features")


if __name__ == "__main__":
    main()