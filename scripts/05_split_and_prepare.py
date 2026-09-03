
"""
Split train/val/test + Normalisation + SMOTE.

Cette etape combine trois operations dans un ordre precis :
  1. Split stratifie en trois : train / validation / test
  2. Normalisation avec MinMaxScaler (fit sur train, transform sur les trois)
  3. SMOTE sur le train avec strategie custom (cap par classe)

Version precedente vs celle-ci
-------------------------------
- Scaler : StandardScaler -> MinMaxScaler. L'article ramene les features
  dans [0,1] ("feature standardization (scaling) to limit the values
  between 0,1"). Avec StandardScaler, torchattacks ecrase a zero toutes les
  valeurs negatives via son clamp interne torch.clamp(x, 0, 1) : cela
  detruisait l'echantillon (pas seulement la perturbation) pour FGSM, BIM
  et PGD, et biaisait la comparaison avec DeepFool/JSMA/C&W qui passent par
  ART sans ce clamp. Voir configs/config.yaml pour la meme regle cote ART
  (clip_values).

- Un vrai split de validation apparait. Avant, seul train/test existaient
  et 06_train_baseline.py choisissait le meilleur epoch sur l'accuracy du
  TEST, ce qui biaise l'accuracy rapportee a la hausse (le test set servait
  a la fois a choisir le modele et a l'evaluer). Le split de validation
  (dataset.val_size dans configs/config.yaml) est reserve a la selection du
  modele et au scheduler ; le test n'est plus touche qu'une fois, pour le
  rapport final.

- Val et test sont explicitement bornes (np.clip) dans clip_values apres
  transform. MinMaxScaler.transform() ne garantit PAS que val/test tombent
  dans l'intervalle appris sur le train : les features de flux reseau
  (duree, taille de paquets) sont a queue tres lourde, un flux de test plus
  extreme que tout le train produit une valeur de 3, 20, ou plus - pas "un
  leger depassement". Sans ce clip, evaluate_baseline_on_clean (etape 6)
  evaluerait le baseline sur des entrees non bornees pendant que les six
  attaques (torchattacks clampe en interne, ART borne via clip_values)
  produisent des entrees bornees : la reference "propre" et les mesures
  "sous attaque" ne porteraient plus sur le meme domaine d'entree, ce qui
  est exactement le biais que MinMaxScaler visait a eliminer. Le nombre de
  valeurs debordantes est logge avant clipping pour que l'ampleur reelle
  soit visible, pas supposee "legere".

La strategie SMOTE custom evite la sur-generation extreme sur les classes
ultra-rares (Heartbleed avec 7 exemples reels donnait 1.4M synthetiques
en version equilibrage total). Ici on plafonne les ratios d'expansion pour
un equilibre entre representation des classes rares et fidelite statistique.
Ce plafonnage n'est pas documente dans l'article ; c'est un choix du projet,
justifie dans le README.

Ordre crucial pour eviter le data leakage :
  - Normaliser AVANT le split leak les stats du val/test dans le train
  - SMOTE AVANT le split cree des exemples synthetiques du val/test dans
    le train
  - Toute transformation dependante des donnees s'apprend sur le train seul

Correspond a l'etape 4 du framework Awad et al. (2025).
"""

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.config import load_config, write_data_fingerprint


def load_selected_data(data_dir):
    """Charge le dataset avec les 58 features selectionnees."""
    print("Chargement du dataset selectionne...")
    df = pd.read_pickle(data_dir / "cicids2017_selected.pkl")
    print(f"  Shape : {df.shape[0]:,} lignes x {df.shape[1]} colonnes\n")
    return df


def split_train_val_test(df, test_size, val_size, random_state, label_col="Label"):
    """
    Split stratifie en trois : train / validation / test.

    test_size et val_size sont des fractions du dataset ORIGINAL (pas l'un
    de l'autre). On splitte donc en deux temps :
      1. Retirer le test (test_size de l'original), stratifie.
      2. Retirer la validation du reste, en reconvertissant val_size
         (fraction de l'original) en fraction du reste :
         val_size / (1 - test_size).

    Repli sur le split de validation : CIC-IDS2017 contient des classes a
    une poignee d'exemples au total (Heartbleed : ~11). train_test_split
    avec stratify leve une ValueError si une classe se retrouve avec moins
    de 2 membres dans l'un des deux groupes produits. Le split du TEST (sur
    les ~2.5M lignes completes) n'y est pas expose en pratique ; celui de
    la VALIDATION, preleve sur le reste deja reduit par le split precedent,
    l'est. Si ca se produit, on se replie sur un split non stratifie pour
    cette seule etape plutot que de faire echouer tout le pipeline apres le
    feature selection : la stratification est desirable, pas une condition
    de correction du reste du pipeline.
    """
    print(
        f"Split train/val/test "
        f"({int((1 - test_size - val_size) * 100)}/{int(val_size * 100)}/"
        f"{int(test_size * 100)}, stratifie)..."
    )

    X = df.drop(columns=[label_col])
    y = df[label_col]

    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    val_fraction_of_rest = val_size / (1 - test_size)
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X_rest, y_rest,
            test_size=val_fraction_of_rest,
            stratify=y_rest,
            random_state=random_state,
        )
    except ValueError as erreur:
        print(
            f"  ATTENTION : split de validation stratifie impossible "
            f"({erreur}). Repli sur un split non stratifie pour cette etape "
            f"uniquement (une classe ultra-rare, probablement Heartbleed, "
            f"a trop peu d'exemples pour etre repartie dans les deux groupes)."
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_rest, y_rest,
            test_size=val_fraction_of_rest,
            stratify=None,
            random_state=random_state,
        )

    n_total = len(X)
    print(f"  Train : {len(X_train):,} lignes ({100*len(X_train)/n_total:.1f}%)")
    print(f"  Val   : {len(X_val):,} lignes ({100*len(X_val)/n_total:.1f}%)")
    print(f"  Test  : {len(X_test):,} lignes ({100*len(X_test)/n_total:.1f}%)\n")

    return X_train, X_val, X_test, y_train, y_val, y_test


def _clip_and_report(X_scaled, borne_min, borne_max, nom_split):
    """
    Borne un split dans [borne_min, borne_max] et rapporte l'ampleur reelle
    du depassement AVANT clipping, pour ne jamais supposer "leger" sans
    verifier.
    """
    hors_domaine = (X_scaled < borne_min) | (X_scaled > borne_max)
    n_hors_domaine = int(hors_domaine.values.sum())
    n_valeurs = X_scaled.size
    if n_hors_domaine > 0:
        depassement_max = max(
            float((X_scaled.values - borne_max)[X_scaled.values > borne_max].max(initial=0)),
            float((borne_min - X_scaled.values)[X_scaled.values < borne_min].max(initial=0)),
        )
        print(
            f"  {nom_split} : {n_hors_domaine:,}/{n_valeurs:,} valeurs "
            f"({100*n_hors_domaine/n_valeurs:.4f}%) hors de [{borne_min},{borne_max}] "
            f"avant clipping, depassement max = {depassement_max:.4f}"
        )
    else:
        print(f"  {nom_split} : aucune valeur hors de [{borne_min},{borne_max}]")
    return X_scaled.clip(lower=borne_min, upper=borne_max)


def normalize_features(X_train, X_val, X_test, clip_values):
    """
    MinMaxScaler fit sur train uniquement, transform sur les trois splits,
    puis clip explicite dans clip_values.

    MinMaxScaler.transform() ne garantit PAS que val/test restent dans
    l'intervalle appris sur le train : les features de flux reseau (duree,
    taille de paquets) sont a queue tres lourde, et un flux plus extreme que
    tout le train produit une valeur de plusieurs unites, pas un leger
    depassement. Sans ce clip, le test set servirait de reference "propre"
    hors du domaine [0,1] pendant que les attaques (torchattacks clampe en
    interne, ART borne via clip_values) y ramenent leurs sorties : la
    comparaison propre vs sous-attaque ne porterait plus sur le meme domaine
    d'entree. Le nombre de valeurs debordantes est logge avant clipping.
    """
    print(f"Normalisation des features (MinMaxScaler -> {clip_values})...")

    borne_min, borne_max = clip_values
    scaler = MinMaxScaler(feature_range=(borne_min, borne_max))
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    print("  Verification du domaine (avant clipping) :")
    X_train_scaled = _clip_and_report(X_train_scaled, borne_min, borne_max, "Train")
    X_val_scaled = _clip_and_report(X_val_scaled, borne_min, borne_max, "Val  ")
    X_test_scaled = _clip_and_report(X_test_scaled, borne_min, borne_max, "Test ")

    print(f"\n  Train (apres clip) : min={X_train_scaled.min().min():.4f}, max={X_train_scaled.max().max():.4f}")
    print(f"  Val   (apres clip) : min={X_val_scaled.min().min():.4f}, max={X_val_scaled.max().max():.4f}")
    print(f"  Test  (apres clip) : min={X_test_scaled.min().min():.4f}, max={X_test_scaled.max().max():.4f}\n")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def apply_smote(X_train, y_train, random_state, smote_strategy):
    """
    Applique SMOTE avec une strategie custom (cap par classe, depuis
    configs/config.yaml -> dataset.smote_strategy).

    Contrairement a l'equilibrage total qui monte toutes les classes au
    niveau de la majoritaire, on plafonne l'expansion pour eviter de creer
    des millions d'interpolations bruitees a partir de quelques exemples reels.

    smote_strategy est filtree sur les classes ayant au moins 2 exemples
    reels dans y_train (pas seulement presentes) avant d'etre passee a
    SMOTE. Necessaire depuis l'ajout du split de validation : le repli non
    stratifie de split_train_val_test (cas Heartbleed, quelques exemples au
    total) peut, par tirage aleatoire, laisser une classe ultra-rare avec
    zero ou un seul exemple dans le train. SMOTE exige au moins
    k_neighbors + 1 exemples reels par classe suréchantillonnée ; une classe
    a 1 exemple ferait chuter k_neighbors a 0 (invalide) si elle n'etait pas
    filtree ici.

    n_min (qui determine k_neighbors) est calcule UNIQUEMENT sur les
    classes retenues dans la strategie, pas sur toutes les classes de
    y_train : une classe rare non listee dans smote_strategy (donc non
    suréchantillonnée) n'a aucun rapport avec la recherche de voisins des
    classes qui LE SONT, et ne doit pas faire chuter k_neighbors pour elles.
    """
    print("Application de SMOTE avec strategie custom...")

    print(f"  Distribution AVANT SMOTE :")
    counts_avant = y_train.value_counts().sort_index()

    strategie = {}
    for classe, cible in smote_strategy.items():
        count_reel = int(counts_avant.get(classe, 0))
        if count_reel >= 2:
            strategie[classe] = cible
        elif count_reel == 0:
            print(
                f"    ATTENTION : classe {classe} absente de y_train (probablement "
                f"tombee entierement dans le val par le repli non stratifie) - "
                f"ignoree pour SMOTE."
            )
        else:
            print(
                f"    ATTENTION : classe {classe} presente avec seulement "
                f"{count_reel} exemple dans y_train (< 2 requis par SMOTE) - "
                f"ignoree pour SMOTE."
            )

    for classe, count in counts_avant.items():
        cible = strategie.get(classe, count)
        marqueur = f"-> {cible:,}" if cible != count else "(inchange)"
        print(f"    Classe {classe:2d} : {count:>10,}  {marqueur}")

    if not strategie:
        print(
            "\n  ATTENTION : aucune classe eligible pour SMOTE (toutes absentes ou "
            "a un seul exemple). Train renvoye inchange.\n"
        )
        return X_train, y_train

    # k_neighbors calcule uniquement sur les classes reellement
    # sur-echantillonnees (cf. docstring) : SMOTE necessite au moins
    # k_neighbors + 1 exemples reels par classe suréchantillonnée.
    n_min = counts_avant.reindex(list(strategie.keys())).min()
    k_neighbors = min(5, int(n_min) - 1)

    print(f"\n  k_neighbors utilise : {k_neighbors}")
    print(f"  (classe sur-echantillonnee la plus petite : {int(n_min)} exemples)\n")

    smote = SMOTE(
        sampling_strategy=strategie,
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


def save_artifacts(X_train, X_val, X_test, y_train, y_val, y_test, scaler, output_dir):
    """
    Sauvegarde des fichiers.

    X_* (DataFrames, conservent les noms de colonnes) via to_pickle ; les
    y_* systematiquement en numpy array via joblib.dump, quel que soit leur
    type d'entree (Series pandas issue du split, ou array numpy issu de
    SMOTE selon la version d'imbalanced-learn). Avant, le choix du format
    dependait du type runtime de l'objet (isinstance DataFrame/Series), ce
    qui marchait par tolerance de joblib.load a lire un fichier ecrit par
    pandas.to_pickle, pas par conception : 06 et 08 chargent tous les y_*
    avec joblib.load, donc l'ecriture doit etre uniforme cote y.
    """
    print("Sauvegarde des artefacts...")
    output_dir.mkdir(parents=True, exist_ok=True)

    fichiers_X = {"X_train.pkl": X_train, "X_val.pkl": X_val, "X_test.pkl": X_test}
    for nom, obj in fichiers_X.items():
        if not isinstance(obj, pd.DataFrame):
            # normalize_features() garantit un DataFrame aujourd'hui ; si ce
            # n'est plus le cas apres une modification future, echouer
            # bruyamment vaut mieux qu'ecrire un format que 06/08 ne
            # sauraient pas relire silencieusement de la meme facon.
            raise TypeError(
                f"{nom} : attendu un DataFrame pandas, recu {type(obj).__name__}."
            )
        path = output_dir / nom
        obj.to_pickle(path)
        taille_mb = path.stat().st_size / (1024 * 1024)
        print(f"  {nom:20s} : {taille_mb:>7.1f} MB")

    fichiers_y = {"y_train.pkl": y_train, "y_val.pkl": y_val, "y_test.pkl": y_test}
    for nom, obj in fichiers_y.items():
        array = obj.values if isinstance(obj, (pd.Series, pd.DataFrame)) else np.asarray(obj)
        path = output_dir / nom
        joblib.dump(array, path)
        taille_mb = path.stat().st_size / (1024 * 1024)
        print(f"  {nom:20s} : {taille_mb:>7.1f} MB")

    scaler_path = output_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"  scaler.pkl           : {scaler_path.stat().st_size / 1024:.1f} KB\n")


def main():
    print("Split train/val/test + Normalisation + SMOTE (v4 : MinMaxScaler + validation reelle)")
    print(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    cfg = load_config()
    print(cfg.resume())
    print()

    data_processed_dir = Path(cfg.paths["data_processed"])
    df = load_selected_data(data_processed_dir)
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        df,
        test_size=cfg.dataset["test_size"],
        val_size=cfg.dataset["val_size"],
        random_state=cfg.seed,
    )
    X_train, X_val, X_test, scaler = normalize_features(
        X_train, X_val, X_test, cfg.clip_values
    )
    X_train, y_train = apply_smote(X_train, y_train, cfg.seed, cfg.dataset["smote_strategy"])
    save_artifacts(
        X_train, X_val, X_test, y_train, y_val, y_test, scaler, data_processed_dir
    )

    fingerprint_path = write_data_fingerprint(cfg, data_processed_dir)
    print(f"Empreinte de configuration ecrite : {fingerprint_path} (hash={cfg.data_fingerprint()})")

    print(f"Train final : {len(X_train):,} lignes x {X_train.shape[1]} features")
    print(f"Val final   : {len(X_val):,} lignes x {X_val.shape[1]} features")
    print(f"Test final  : {len(X_test):,} lignes x {X_test.shape[1]} features")


if __name__ == "__main__":
    main()
