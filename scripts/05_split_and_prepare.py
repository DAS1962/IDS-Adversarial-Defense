
"""
Split train/test + Normalisation + SMOTE.

Cette etape combine trois operations dans un ordre precis :
  1. Split stratifie train/test (67/33 comme le papier)
  2. Normalisation avec StandardScaler (fit sur train, transform sur les deux)
  3. SMOTE sur le train uniquement (equilibrage des classes)

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


# Chemins des dossiers
DATA_PROCESSED_DIR = Path("data/processed")
LOG_DIR = Path("results/logs")

# Parametres
TEST_SIZE = 0.33            # 33% pour le test, 67% pour le train (comme le papier)
RANDOM_STATE = 42           # Reproductibilite


# Fonction 1 : charger le dataset selectionne
def load_selected_data(data_dir):
    """
    Charge le dataset avec les 58 features selectionnees.
    Retourne un DataFrame avec 59 colonnes (58 features + Label).
    """
    print("Chargement du dataset selectionne...")
    df = pd.read_pickle(data_dir / "cicids2017_selected.pkl")
    print(f"  Shape : {df.shape[0]:,} lignes x {df.shape[1]} colonnes\n")
    return df


# Fonction 2 : split train/test stratifie
def split_train_test(df, test_size, random_state, label_col="Label"):
    """
    Separe le dataset en train et test, avec stratification.

    Stratification : preserve les proportions des classes.
    Sans elle, un split aleatoire pourrait mettre 0 Heartbleed
    dans le train (11 exemples au total).

    Ratio : 67% train, 33% test (comme le papier Awad et al.).
    """
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

    # Verifier que toutes les classes sont dans les deux splits
    classes_train = set(y_train.unique())
    classes_test = set(y_test.unique())
    print(f"  Classes dans train : {len(classes_train)}")
    print(f"  Classes dans test  : {len(classes_test)}")

    if classes_train != classes_test:
        manquantes = classes_train.symmetric_difference(classes_test)
        print(f"  ATTENTION : classes manquantes dans un split : {manquantes}")
    print()

    return X_train, X_test, y_train, y_test


# Fonction 3 : normalisation avec StandardScaler
def normalize_features(X_train, X_test):
    """
    Normalise les features avec StandardScaler.

    Le scaler apprend la moyenne et l'ecart-type UNIQUEMENT sur le train,
    puis les applique au train ET au test.

    Formule : x_norm = (x - mean) / std

    Apres normalisation, chaque feature a mean=0 et std=1 sur le train.
    Sur le test, mean et std seront proches mais pas exactement (normal).
    """
    print("Normalisation des features (StandardScaler)...")
    print(f"  Fit sur train, transform sur train ET test")

    scaler = StandardScaler()

    # fit_transform sur le train : apprend et applique
    X_train_scaled = scaler.fit_transform(X_train)

    # transform sur le test : applique seulement (pas de fit !)
    X_test_scaled = scaler.transform(X_test)

    # Conserver les noms de colonnes (perdu par StandardScaler qui retourne numpy)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    print(f"  Train apres scaling : mean~{X_train_scaled.mean().mean():.4f}, std~{X_train_scaled.std().mean():.4f}")
    print(f"  Test  apres scaling : mean~{X_test_scaled.mean().mean():.4f}, std~{X_test_scaled.std().mean():.4f}")
    print()

    return X_train_scaled, X_test_scaled, scaler


# Fonction 4 : SMOTE sur le train
def apply_smote(X_train, y_train, random_state):
    """
    Applique SMOTE sur le train pour equilibrer les classes.

    SMOTE (Synthetic Minority Over-sampling Technique) genere des exemples
    synthetiques par interpolation lineaire entre voisins de la meme classe.

    Par defaut, imblearn.SMOTE equilibre toutes les classes au niveau
    de la classe majoritaire (BENIGN dans notre cas).

    Attention : SMOTE ne s'applique JAMAIS sur le test.
    Le test doit refleter la realite operationnelle.
    """
    print("Application de SMOTE sur le train...")

    # Distribution avant SMOTE
    print(f"  Distribution AVANT SMOTE :")
    counts_avant = y_train.value_counts().sort_index()
    for classe, count in counts_avant.items():
        print(f"    Classe {classe:2d} : {count:>10,}")
    print(f"    Total       : {len(y_train):>10,}")

    # Application de SMOTE
    smote = SMOTE(random_state=random_state, n_jobs=-1)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    # Distribution apres SMOTE
    print(f"\n  Distribution APRES SMOTE :")
    counts_apres = pd.Series(y_train_bal).value_counts().sort_index()
    for classe, count in counts_apres.items():
        print(f"    Classe {classe:2d} : {count:>10,}")
    print(f"    Total       : {len(y_train_bal):>10,}")

    n_synthetiques = len(y_train_bal) - len(y_train)
    print(f"\n  Exemples synthetiques ajoutes : {n_synthetiques:,}")
    print()

    return X_train_bal, y_train_bal


# Fonction 5 : sauvegarder tous les artefacts
def save_artifacts(X_train, X_test, y_train, y_test, scaler, output_dir):
    """
    Sauvegarde les splits, le scaler et les labels.

    5 fichiers :
    - X_train.pkl : features d'entrainement (apres SMOTE et scaling)
    - X_test.pkl  : features de test (apres scaling, sans SMOTE)
    - y_train.pkl : labels d'entrainement (apres SMOTE)
    - y_test.pkl  : labels de test (sans SMOTE)
    - scaler.pkl  : le StandardScaler entraine (pour transformer d'autres donnees)
    """
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
            # Cas des arrays numpy retournes par SMOTE
            joblib.dump(obj, path)
        taille_mb = path.stat().st_size / (1024 * 1024)
        print(f"  {nom:20s} : {taille_mb:>7.1f} MB")

    # Sauvegarder le scaler avec joblib (recommande pour sklearn)
    scaler_path = output_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"  scaler.pkl           : {scaler_path.stat().st_size / 1024:.1f} KB")
    print()


# Fonction principale
def main():
    print("=" * 60)
    print("Split train/test + Normalisation + SMOTE")
    print(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60 + "\n")

    df = load_selected_data(DATA_PROCESSED_DIR)
    X_train, X_test, y_train, y_test = split_train_test(
        df, TEST_SIZE, RANDOM_STATE
    )
    X_train, X_test, scaler = normalize_features(X_train, X_test)
    X_train, y_train = apply_smote(X_train, y_train, RANDOM_STATE)
    save_artifacts(X_train, X_test, y_train, y_test, scaler, DATA_PROCESSED_DIR)

    print("=" * 60)
    print("Preparation des donnees terminee")
    print(f"Train final : {len(X_train):,} lignes x {X_train.shape[1]} features")
    print(f"Test final  : {len(X_test):,} lignes x {X_test.shape[1]} features")
    print("=" * 60)


if __name__ == "__main__":
    main()