"""
Tableau complet des quatre defenses, toutes metriques.

Le papier publie :
  Table 6  accuracy par defense et par attaque
  Table 7  accuracy, precision, recall, F1 moyennes par defense
  Table 9  MCC binaire par defense

Il ne dit pas sur quoi portent les moyennes de sa Table 7 : sur les six
attaques, sur les quatre, ou sur donnees propres. On donne les trois.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

# Les .pkl contiennent des tenseurs enregistres sur GPU. Sur un noeud de
# connexion sans CUDA, joblib echoue au chargement. On force le CPU.
_torch_load = torch.load
torch.load = lambda *a, **k: _torch_load(*a, **{**k, "map_location": "cpu"})

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.config import load_config

ATTAQUES = ["FGSM", "BIM", "PGD", "DeepFool", "JSMA", "CW"]
ORDRE = ["Clean"] + ATTAQUES
DEFENSES = [("ls", "Label Smoothing"), ("ga", "Gaussian Augmentation"),
            ("at", "Adversarial Training"), ("dae", "Denoising Autoencoder")]

# Table 6 : accuracy par defense et par attaque
TABLE6 = {
    "ls":  {"DeepFool": 81.71, "BIM": 82.75, "JSMA": 81.25, "FGSM": 83.71,
            "PGD": 82.91, "CW": 80.40},
    "at":  {"DeepFool": 80.47, "BIM": 80.49, "JSMA": 81.47, "FGSM": 80.85,
            "PGD": 80.75, "CW": 78.30},
    "ga":  {"DeepFool": 80.21, "BIM": 80.22, "JSMA": 81.59, "FGSM": 80.23,
            "PGD": 80.66, "CW": 77.40},
    "dae": {"DeepFool": 81.71, "BIM": 82.87, "JSMA": 81.33, "FGSM": 83.85,
            "PGD": 82.89, "CW": 80.10},
}

# Table 7 : moyennes par defense
TABLE7 = {
    "ls":  {"acc": 85.90, "prec": 81.41, "rec": 87.23, "f1": 84.20},
    "ga":  {"acc": 79.80, "prec": 80.00, "rec": 79.68, "f1": 79.80},
    "at":  {"acc": 80.25, "prec": 80.75, "rec": 79.95, "f1": 78.70},
    "dae": {"acc": 84.80, "prec": 82.20, "rec": 81.87, "f1": 82.87},
}

# Table 9 : MCC binaire
TABLE9 = {"ls": 0.698, "dae": 0.680, "at": 0.608, "ga": 0.596}


def charger(log_dir):
    res = {}
    for court, _ in DEFENSES:
        fichiers = sorted(Path(log_dir).glob(f"defense_{court}_*.pkl"))
        if fichiers:
            res[court] = joblib.load(fichiers[-1])["resultats"]
    ev = sorted(Path(log_dir).glob("evaluation_*.pkl"))
    base = None
    if ev:
        d = joblib.load(ev[-1])
        cle = [k for k in d["resultats"] if "reduit" in k][0]
        base = d["resultats"][cle]
    return res, base


def main():
    cfg = load_config()
    res, base = charger(cfg.paths["logs"])
    if base is None:
        raise RuntimeError("Aucun evaluation_*.pkl")

    def v(court, attaque, col):
        return 100 * res[court][attaque][col] if court in res else float("nan")

    def b(attaque, col):
        return 100 * base[attaque][col]

    print("=" * 110)
    print("LES QUATRE DEFENSES - TOUTES LES METRIQUES")
    print("=" * 110)
    print("Baseline de reference : lr 0.001, 100 epochs")
    print("Test : 20 000 lignes a 50 % benin, plancher a 50 %")
    print()

    # --- 1. Accuracy par attaque, face a leur Table 6 ---
    print("-" * 110)
    print("1. ACCURACY PAR ATTAQUE — face a leur Table 6")
    print("-" * 110)
    print(f"{'Attaque':<10} {'Base':>7} | {'LS':>7} {'P':>7} | {'GA':>7} {'P':>7} "
          f"| {'AT':>7} {'P':>7} | {'DAE':>7} {'P':>7}")
    for a in ORDRE:
        ligne = f"{a:<10} {b(a,'accuracy'):>6.2f}% |"
        for court, _ in DEFENSES:
            p = TABLE6[court].get(a)
            ligne += (f" {v(court,a,'accuracy'):>6.2f}% "
                      f"{p:>6.1f}% |" if p else f" {v(court,a,'accuracy'):>6.2f}% "
                      f"{'—':>7} |")
        print(ligne)
    print()

    # --- 2. F1 macro par attaque ---
    print("-" * 110)
    print("2. F1 MACRO PAR ATTAQUE — le papier n'en publie pas")
    print("-" * 110)
    print(f"{'Attaque':<10} {'Base':>8} | {'LS':>8} {'GA':>8} {'AT':>8} {'DAE':>8}")
    for a in ORDRE:
        print(f"{a:<10} {b(a,'f1_macro'):>7.2f}% | "
              + " ".join(f"{v(c,a,'f1_macro'):>7.2f}%" for c, _ in DEFENSES))
    print()

    # --- 3. Les quatre metriques du papier, trois moyennes possibles ---
    print("-" * 110)
    print("3. METRIQUES MOYENNES — face a leur Table 7")
    print("-" * 110)
    print("Le papier ne dit pas sur quoi il moyenne. Trois lectures possibles.")
    print()
    for court, nom in DEFENSES:
        if court not in res:
            continue
        p = TABLE7[court]
        print(f"{nom}")
        print(f"  {'':22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print(f"  {'Papier (Table 7)':22} {p['acc']:>9.2f}% {p['prec']:>9.2f}% "
              f"{p['rec']:>9.2f}% {p['f1']:>9.2f}%")

        for lib, sous_ensemble in (("Nous, donnees propres", ["Clean"]),
                                   ("Nous, six attaques", ATTAQUES),
                                   ("Nous, les sept", ORDRE)):
            for typ in ("macro", "weighted", "micro"):
                acc = np.mean([v(court, a, "accuracy") for a in sous_ensemble])
                pr = np.mean([v(court, a, f"precision_{typ}") for a in sous_ensemble])
                rc = np.mean([v(court, a, f"recall_{typ}") for a in sous_ensemble])
                f1 = np.mean([v(court, a, f"f1_{typ}") for a in sous_ensemble])
                print(f"  {lib + ' (' + typ + ')':22} {acc:>9.2f}% {pr:>9.2f}% "
                      f"{rc:>9.2f}% {f1:>9.2f}%")
        print()

    # --- 4. MCC, face a leur Table 9 ---
    print("-" * 110)
    print("4. MCC BINAIRE — face a leur Table 9")
    print("-" * 110)
    print(f"{'Defense':<24} {'Papier':>8} {'Clean':>8} {'Min att.':>9} "
          f"{'Moy. att.':>10}")
    for court, nom in DEFENSES:
        if court not in res:
            continue
        vals = [res[court][a]["mcc_binaire"] for a in ATTAQUES]
        print(f"{nom:<24} {TABLE9[court]:>8.3f} "
              f"{res[court]['Clean']['mcc_binaire']:>8.3f} "
              f"{min(vals):>9.3f} {np.mean(vals):>10.3f}")
    print()

    # --- 5. Rappel BENIGN ---
    print("-" * 110)
    print("5. RAPPEL SUR LE TRAFIC NORMAL — pas publie par le papier")
    print("-" * 110)
    print(f"{'Attaque':<10} {'Base':>8} | {'LS':>8} {'GA':>8} {'AT':>8} {'DAE':>8}")
    for a in ORDRE:
        print(f"{a:<10} {b(a,'recall_benign'):>7.2f}% | "
              + " ".join(f"{v(c,a,'recall_benign'):>7.2f}%" for c, _ in DEFENSES))
    print()

    # --- 6. Bilan ---
    print("=" * 110)
    print("6. BILAN")
    print("=" * 110)
    print(f"{'Defense':<24} {'Gain':>9} {'Cout':>9} {'Bilan net':>11} "
          f"{'RecBEN min':>11}")
    lignes_bilan = []
    for court, nom in DEFENSES:
        if court not in res:
            continue
        gain = np.mean([res[court][a]["f1_macro"] - base[a]["f1_macro"]
                        for a in ATTAQUES])
        cout = res[court]["Clean"]["f1_macro"] - base["Clean"]["f1_macro"]
        rb = min(res[court][a]["recall_benign"] for a in ATTAQUES)
        print(f"{nom:<24} {gain:>+9.4f} {cout:>+9.4f} {gain+cout:>+11.4f} "
              f"{rb:>11.4f}")
        lignes_bilan.append((court, nom, gain, cout, rb))
    print()
    print("Rappel du papier, Table 7 : LS 85.90 %, DAE 84.80 %, AT 80.25 %,")
    print("GA 79.80 %. Leur classement place LS premier et AT quatrieme.")
    print("=" * 110)

    # --- CSV ---
    lignes = []
    for court, nom in DEFENSES:
        if court not in res:
            continue
        for a in ORDRE:
            d = {"defense": court, "nom": nom, "attaque": a,
                 "papier_accuracy": TABLE6[court].get(a, np.nan)}
            d.update({k: val for k, val in res[court][a].items()
                      if k not in ("confusion_matrix", "attaque")})
            lignes.append(d)
    for a in ORDRE:
        d = {"defense": "baseline", "nom": "Baseline lr reduit", "attaque": a,
             "papier_accuracy": np.nan}
        d.update({k: val for k, val in base[a].items()
                  if k not in ("confusion_matrix", "attaque")})
        lignes.append(d)

    sortie = Path(cfg.paths["figures"]) / "defenses_completes.csv"
    pd.DataFrame(lignes).to_csv(sortie, index=False, float_format="%.6f")
    print(f"\nCSV : {sortie}")


if __name__ == "__main__":
    main()
