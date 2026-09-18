"""
Pipeline de donnees, ordre de la Figure 2 du papier.

Nettoyage -> encodage -> echelle -> equilibrage 50% -> selection ERT -> split.

Le split arrive en DERNIER. L'echelle, l'equilibrage et la selection voient
donc le test set : il y a fuite. C'est ce que fait le papier, on le reproduit.
Sur la branche main l'ordre est inverse et le split arrive en premier.
"""

import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.config import load_config


def charger_csv(dossier):
    fichiers = sorted(Path(dossier).glob("*.csv"))
    if not fichiers:
        raise RuntimeError(f"Aucun CSV dans {dossier}")

    print(f"Chargement de {len(fichiers)} fichiers...")
    morceaux = []
    for f in fichiers:
        df = pd.read_csv(f, low_memory=False)
        # 65 colonnes ont un espace parasite en debut de nom, a cause du
        # separateur ", " de CICFlowMeter.
        df.columns = df.columns.str.strip()
        morceaux.append(df)
        print(f"  {f.name:<50} {len(df):>9,}")

    df = pd.concat(morceaux, ignore_index=True)
    print(f"  total : {len(df):,} lignes x {df.shape[1]} colonnes\n")
    return df


def nettoyer(df, retirer_doublons):
    print("Nettoyage...")
    n0 = len(df)

    df = df.replace([np.inf, -np.inf], np.nan)
    n_inf = n0 - len(df.dropna())
    df = df.dropna()
    print(f"  Inf et NaN : {n_inf:,} lignes retirees")

    if retirer_doublons:
        avant = len(df)
        df = df.drop_duplicates()
        print(f"  doublons   : {avant - len(df):,} lignes retirees")
    else:
        n_dup = df.duplicated().sum()
        print(f"  doublons   : {n_dup:,} presents, conserves "
              f"(le papier n'en parle pas)")

    print(f"  reste {len(df):,} lignes\n")
    return df.reset_index(drop=True)


def encoder(df, col="Label"):
    print("Encodage des etiquettes...")
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    for i, nom in enumerate(le.classes_):
        print(f"  {i:2d} {str(nom)[:35]:<35} {(df[col] == i).sum():>9,}")
    print()
    return df, le


def mettre_echelle(df, bornes, col="Label"):
    """MinMax sur tout le jeu. Le papier le fait avant le split."""
    print(f"Mise a l'echelle MinMax vers {bornes}...")
    y = df[col]
    X = df.drop(columns=[col])

    scaler = MinMaxScaler(feature_range=bornes)
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    print(f"  min {X_scaled.min().min():.4f}, max {X_scaled.max().max():.4f}\n")
    df = X_scaled
    df[col] = y.values
    return df, scaler


def equilibrer(df, fraction, seuil, seed, col="Label"):
    """
    50% de chaque classe, sauf celles sous le seuil qui sont prises
    entierement. Vient de : "selecting 50% only of each class member except
    those containing less than 2000 instances they are taken entirely".
    """
    print(f"Equilibrage : {fraction:.0%} par classe, seuil {seuil}...")
    rng = np.random.default_rng(seed)
    garde = []

    for classe in sorted(df[col].unique()):
        idx = df.index[df[col] == classe].to_numpy()
        if len(idx) < seuil:
            garde.append(idx)
            print(f"  classe {classe:2d} : {len(idx):>9,} -> {len(idx):>9,} "
                  f"(entiere, sous le seuil)")
        else:
            n = int(len(idx) * fraction)
            garde.append(rng.choice(idx, size=n, replace=False))
            print(f"  classe {classe:2d} : {len(idx):>9,} -> {n:>9,}")

    idx = np.sort(np.concatenate(garde))
    df = df.loc[idx].reset_index(drop=True)
    print(f"  total : {len(df):,} lignes\n")
    return df


def normaliser_nom(nom):
    """Nom de feature en minuscules avec des tirets bas, pour comparer."""
    return nom.lower().replace(" ", "_").replace(".", "").replace("/", "_")


def selectionner_features(df, cfg_fs, seed, col="Label"):
    """
    Retire les features les PLUS importantes et garde les autres.

    Le papier : "we use the Extremely Randomized Trees (ERTs) classifier to
    select the most important (functional) features and exclude them". Les
    features gardees sont donc les non fonctionnelles, celles qu'un attaquant
    peut perturber sans casser la communication.
    """
    mode = cfg_fs["mode"]
    n_remove = cfg_fs["n_remove"]
    X = df.drop(columns=[col])
    y = df[col]

    print(f"Selection des features, mode {mode}...")

    if mode == "ert":
        ert = ExtraTreesClassifier(
            n_estimators=cfg_fs["n_estimators"], random_state=seed, n_jobs=-1)
        ert.fit(X, y)
        importances = pd.Series(ert.feature_importances_, index=X.columns)
        importances = importances.sort_values(ascending=False)
        a_retirer = list(importances.index[:n_remove])

        print(f"  {n_remove} features les plus importantes, retirees :")
        for nom in a_retirer:
            print(f"    {nom:<40} {importances[nom]:.4f}")

        # Comparaison avec leur Figure 3, pour voir si l'ERT retrouve la
        # meme liste a partir de nos donnees.
        leur_liste = {normaliser_nom(n) for n in cfg_fs["figure3"]}
        notre_liste = {normaliser_nom(n) for n in a_retirer}
        commun = leur_liste & notre_liste
        print(f"\n  En commun avec leur Figure 3 : {len(commun)}/{n_remove}")
        if len(commun) < n_remove:
            print(f"  Chez eux seulement : {sorted(leur_liste - notre_liste)}")
            print(f"  Chez nous seulement : {sorted(notre_liste - leur_liste)}")

    elif mode == "figure3":
        # On cherche les vrais noms du CSV correspondant a leur graphique.
        table = {normaliser_nom(c): c for c in X.columns}
        a_retirer, absentes = [], []
        for nom in cfg_fs["figure3"]:
            vrai = table.get(normaliser_nom(nom))
            if vrai:
                a_retirer.append(vrai)
            else:
                absentes.append(nom)

        print(f"  {len(a_retirer)}/{len(cfg_fs['figure3'])} features trouvees")
        if absentes:
            raise RuntimeError(
                f"Features de la Figure 3 sans equivalent dans le CSV : "
                f"{absentes}. Verifier la transcription dans config.yaml."
            )
    else:
        raise ValueError(f"mode {mode!r} inconnu, attendu 'ert' ou 'figure3'")

    X = X.drop(columns=a_retirer)
    print(f"\n  reste {X.shape[1]} features\n")

    df = X.copy()
    df[col] = y.values
    return df, a_retirer


def decouper(df, test_size, val_size, seed, col="Label"):
    """Split en dernier, comme la Figure 2."""
    print(f"Split train/test ({1-test_size:.0%}/{test_size:.0%})...")
    X = df.drop(columns=[col])
    y = df[col]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)

    # Le papier utilise validation_split=0.33 de Keras, donc un morceau du
    # train. On en fait un vrai set pour choisir les hyperparametres qu'ils
    # ne documentent pas.
    effectifs = y_tr.value_counts()
    strat = y_tr if (effectifs >= 2).all() else None
    if strat is None:
        print("  (split val non stratifie, une classe a moins de 2 exemples)")

    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, test_size=val_size, stratify=strat, random_state=seed)

    for nom, part in (("train", X_tr), ("val", X_va), ("test", X_te)):
        print(f"  {nom:<6} {len(part):>9,} ({100*len(part)/len(X):.1f}%)")

    n_benin = int((y_te == 0).sum())
    print(f"\n  test : {n_benin:,} benin sur {len(y_te):,} "
          f"({100*n_benin/len(y_te):.1f}%)\n")

    return X_tr, X_va, X_te, y_tr, y_va, y_te


def main():
    print("=" * 70)
    print("Pipeline de donnees - reproduction fidele")
    print(f"Date : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 70 + "\n")

    cfg = load_config()
    print(cfg.resume())
    print()

    d = cfg.dataset
    sortie = Path(cfg.paths["processed"])
    sortie.mkdir(parents=True, exist_ok=True)

    df = charger_csv(cfg.paths["raw"])
    df = nettoyer(df, d["drop_duplicates"])
    df, le = encoder(df)
    df, scaler = mettre_echelle(df, cfg.clip_values)
    df = equilibrer(df, d["balance_fraction"], d["balance_min_instances"],
                    cfg.seed)
    df, retirees = selectionner_features(df, cfg.feature_selection, cfg.seed)

    X_tr, X_va, X_te, y_tr, y_va, y_te = decouper(
        df, d["test_size"], d["val_size"], cfg.seed)

    print("Sauvegarde...")
    for nom, obj in (("X_train", X_tr), ("X_val", X_va), ("X_test", X_te),
                     ("y_train", y_tr), ("y_val", y_va), ("y_test", y_te)):
        chemin = sortie / f"{nom}.pkl"
        joblib.dump(obj.to_numpy(), chemin)
        print(f"  {nom + '.pkl':<16} {chemin.stat().st_size/1e6:>7.1f} MB")

    joblib.dump(scaler, sortie / "scaler.pkl")
    joblib.dump(le, sortie / "label_encoder.pkl")
    joblib.dump(list(X_tr.columns), sortie / "features.pkl")
    joblib.dump(retirees, sortie / "features_retirees.pkl")

    print(f"\n{X_tr.shape[1]} features gardees, {len(retirees)} retirees")
    print("Termine")


if __name__ == "__main__":
    main()
