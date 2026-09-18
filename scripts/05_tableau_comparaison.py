"""
Tableau comparatif : papier, baseline fidele, variante lr.

Le papier publie quatre metriques a ce stade : accuracy, precision, recall,
F1. Table 4 pour le clean, Table 5 sous attaque. Il ne dit jamais si ses
moyennes sont macro, ponderees ou micro ; ses quatre valeurs de la Table 4
(98.11 / 98.11 / 98.11 / 98.068) sont quasi identiques, ce qui signe du
pondere ou du micro. On affiche nos trois versions en face.

Le MCC n'arrive qu'a sa Table 9, pour les defenses. On le gardera pour
l'etape des defenses, pas ici.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.config import load_config

PAPIER = {
    "Clean":    {"acc": 98.11, "prec": 98.11, "rec": 98.11, "f1": 98.068},
    "FGSM":     {"acc": 54.5,  "prec": 66.0,  "rec": 54.0,  "f1": 59.0},
    "BIM":      {"acc": 45.0,  "prec": 54.0,  "rec": 45.0,  "f1": 52.0},
    "PGD":      {"acc": 46.0,  "prec": 57.0,  "rec": 46.0,  "f1": 54.0},
    "DeepFool": {"acc": 53.0,  "prec": 67.0,  "rec": 53.0,  "f1": 58.0},
    "JSMA":     {"acc": 81.0,  "prec": 65.0,  "rec": 81.0,  "f1": 72.0},
    "CW":       {"acc": 36.0,  "prec": 30.0,  "rec": 35.0,  "f1": 32.0},
}

ORDRE = ["Clean", "FGSM", "BIM", "PGD", "DeepFool", "JSMA", "CW"]


def main():
    cfg = load_config()
    df = pd.read_csv(Path(cfg.paths["figures"]) / "evaluation_baselines.csv")

    def v(modele, attaque, col):
        L = df[(df["modele"] == modele) & (df["attaque"] == attaque)]
        return 100 * L[col].iloc[0] if len(L) else float("nan")

    print("=" * 104)
    print("COMPARAISON PAPIER / REPRODUCTION FIDELE / VARIANTE LR")
    print("=" * 104)
    print("P = papier, Tables 4 et 5")
    print("F = reproduction fidele, lr 0.01, 30 epochs")
    print("V = variante, lr 0.001, 100 epochs, tout le reste identique")
    print()
    print("Test : 20 000 lignes a 50 % benin, plancher d'accuracy a 50 %.")
    print("Le papier ne precise pas le type de ses moyennes, on donne les trois.")
    print()

    print("-" * 104)
    print("1. ACCURACY")
    print("-" * 104)
    print(f"{'Attaque':<10} {'P':>9} {'F':>9} {'V':>9} {'F-P':>9} {'V-P':>9}")
    for a in ORDRE:
        p = PAPIER[a]["acc"]
        f, w = v("fidele", a, "accuracy"), v("varlr", a, "accuracy")
        print(f"{a:<10} {p:>8.1f}% {f:>8.2f}% {w:>8.2f}% "
              f"{f-p:>+9.1f} {w-p:>+9.1f}")
    print()

    for titre, base in (("2. PRECISION", "precision"),
                        ("3. RECALL", "recall"),
                        ("4. F1", "f1")):
        cle_papier = {"precision": "prec", "recall": "rec", "f1": "f1"}[base]
        print("-" * 104)
        print(f"{titre} — les trois types de moyenne")
        print("-" * 104)
        print(f"{'Attaque':<10} {'P':>8} | {'F mac':>7} {'F pond':>7} "
              f"{'F mic':>7} | {'V mac':>7} {'V pond':>7} {'V mic':>7}")
        for a in ORDRE:
            print(f"{a:<10} {PAPIER[a][cle_papier]:>7.1f}% | "
                  f"{v('fidele',a,base+'_macro'):>6.1f}% "
                  f"{v('fidele',a,base+'_weighted'):>6.1f}% "
                  f"{v('fidele',a,base+'_micro'):>6.1f}% | "
                  f"{v('varlr',a,base+'_macro'):>6.1f}% "
                  f"{v('varlr',a,base+'_weighted'):>6.1f}% "
                  f"{v('varlr',a,base+'_micro'):>6.1f}%")
        print()

    print("-" * 104)
    print("5. RAPPEL SUR LE TRAFIC NORMAL — pas publie par le papier")
    print("-" * 104)
    print("Montre si le modele se met a rejeter du trafic legitime, ce que")
    print("les moyennes cachent.")
    print(f"{'Attaque':<10} {'F':>9} {'V':>9}")
    for a in ORDRE:
        print(f"{a:<10} {v('fidele',a,'recall_benign'):>8.2f}% "
              f"{v('varlr',a,'recall_benign'):>8.2f}%")
    print()

    att = [a for a in ORDRE if a != "Clean"]
    print("=" * 104)
    print("ECARTS ABSOLUS MOYENS AVEC LE PAPIER, six attaques")
    print("=" * 104)
    for court, nom in (("fidele", "Fidele"), ("varlr", "Variante lr")):
        parts = []
        for col, lib, ref in (("accuracy", "accuracy", "acc"),
                              ("f1_weighted", "F1 pond", "f1"),
                              ("f1_micro", "F1 micro", "f1"),
                              ("f1_macro", "F1 macro", "f1")):
            e = sum(abs(v(court, a, col) - PAPIER[a][ref]) for a in att) / len(att)
            parts.append(f"{lib} {e:.1f}")
        print(f"  {nom:<14} " + " | ".join(parts) + "  (points)")
    print()
    print("Pour memoire, branche main : 32.5 points d'accuracy, 19.3 apres")
    print("reequilibrage du test a 50/50.")
    print("=" * 104)

    lignes = []
    for a in ORDRE:
        p = PAPIER[a]
        ligne = {"attaque": a,
                 "papier_accuracy": p["acc"] / 100,
                 "papier_precision": p["prec"] / 100,
                 "papier_recall": p["rec"] / 100,
                 "papier_f1": p["f1"] / 100}
        for court in ("fidele", "varlr"):
            for col in ("accuracy", "precision_macro", "precision_weighted",
                        "precision_micro", "recall_macro", "recall_weighted",
                        "recall_micro", "f1_macro", "f1_weighted", "f1_micro",
                        "recall_benign"):
                ligne[f"{court}_{col}"] = v(court, a, col) / 100
        lignes.append(ligne)

    sortie = Path(cfg.paths["figures"]) / "comparaison_papier.csv"
    pd.DataFrame(lignes).to_csv(sortie, index=False, float_format="%.6f")
    print(f"\nCSV : {sortie}")


if __name__ == "__main__":
    main()
