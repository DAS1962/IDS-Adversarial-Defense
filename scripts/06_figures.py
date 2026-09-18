"""Figures de la reproduction fidele."""

import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.config import load_config

ATTAQUES = ["FGSM", "BIM", "PGD", "DeepFool", "JSMA", "CW"]
ORDRE = ["Clean"] + ATTAQUES

PAPIER_ACC = {"Clean": 98.11, "FGSM": 54.5, "BIM": 45.0, "PGD": 46.0,
              "DeepFool": 53.0, "JSMA": 81.0, "CW": 36.0}

GRIS, BLEU, VERT, ROUGE = "#5A6C7A", "#1C7293", "#2E7D5B", "#B23A32"


def courbes(log_dir, sortie):
    """Courbes des trois baselines."""
    runs = []
    for motif, nom, couleur in (
            ("baseline_2*.pkl", "lr 0.01, 30 epochs", ROUGE),
            ("baseline_varlr_*.pkl", "lr 0.001", VERT)):
        for f in sorted(log_dir.glob(motif)):
            d = joblib.load(f)
            n = len(d["historique"]["val_acc"])
            runs.append((f"{nom} ({n} ep)", d["historique"],
                         d["meilleur_epoch"], couleur))

    if not runs:
        print("  pas d'historique trouve")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    styles = ["-", "--", ":"]
    for i, (nom, h, best, coul) in enumerate(runs):
        ep = range(1, len(h["val_acc"]) + 1)
        st = styles[i % len(styles)]
        ax1.plot(ep, h["val_acc"], st, color=coul, linewidth=1.6, label=nom)
        ax2.plot(ep, h["val_loss"], st, color=coul, linewidth=1.6, label=nom)
        ax1.axvline(best, color=coul, linestyle=":", alpha=0.35)

    ax1.set_xlabel("Passage sur les donnees")
    ax1.set_ylabel("Accuracy de validation")
    ax1.set_title("Accuracy de validation\n"
                  "Le lr 0.01 du papier plafonne des l'epoch 3", fontsize=10.5)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("Passage sur les donnees")
    ax2.set_ylabel("Perte de validation")
    ax2.set_title("Perte de validation", fontsize=10.5)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(sortie, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {sortie.name}")


def accuracy_trois(df, sortie):
    """Accuracy : papier, fidele, variante."""
    x = np.arange(len(ORDRE))
    larg = 0.26
    fig, ax = plt.subplots(figsize=(12.5, 5.5))

    p = [PAPIER_ACC[a] for a in ORDRE]
    f = [100 * df[(df.modele == "fidele") & (df.attaque == a)].accuracy.iloc[0]
         for a in ORDRE]
    v = [100 * df[(df.modele == "varlr") & (df.attaque == a)].accuracy.iloc[0]
         for a in ORDRE]

    ax.bar(x - larg, p, larg, label="Papier (Tables 4 et 5)", color=GRIS)
    ax.bar(x, f, larg, label="Fidele (lr 0.01, 30 ep)", color=ROUGE)
    ax.bar(x + larg, v, larg, label="Variante (lr 0.001, 100 ep)", color=VERT)

    ax.axhline(50, color="#16232E", linestyle="--", linewidth=1.2,
               label="plancher : 50 % de benin")
    ax.set_xticks(x)
    ax.set_xticklabels(ORDRE)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Accuracy : papier contre nos deux baselines\n"
                 "Test de 20 000 lignes a 50 % benin, comme le papier",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(sortie, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {sortie.name}")


def ecarts(df, sortie):
    """Ecart a l'accuracy du papier, par attaque."""
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(ATTAQUES))
    larg = 0.36

    ef = [100 * df[(df.modele == "fidele") & (df.attaque == a)].accuracy.iloc[0]
          - PAPIER_ACC[a] for a in ATTAQUES]
    ev = [100 * df[(df.modele == "varlr") & (df.attaque == a)].accuracy.iloc[0]
          - PAPIER_ACC[a] for a in ATTAQUES]

    b1 = ax.bar(x - larg/2, ef, larg, label="Fidele", color=ROUGE)
    b2 = ax.bar(x + larg/2, ev, larg, label="Variante lr", color=VERT)
    for barres, vals in ((b1, ef), (b2, ev)):
        for b, val in zip(barres, vals):
            dec = 1.5 if val > 0 else -3.5
            ax.text(b.get_x() + b.get_width()/2, val + dec, f"{val:+.1f}",
                    ha="center", fontsize=8.5)

    ax.axhline(0, color="#16232E", linewidth=1)
    ax.axhspan(-3, 3, color=VERT, alpha=0.10, label="a moins de 3 points")
    ax.set_xticks(x)
    ax.set_xticklabels(ATTAQUES)
    ax.set_ylabel("Ecart d'accuracy avec le papier (points)")
    ax.set_title("Ecart avec le papier, attaque par attaque\n"
                 "BIM, PGD et DeepFool sont reproduits a moins de 3 points",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(sortie, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {sortie.name}")


def metriques_comparees(df, sortie):
    """Les trois types de moyenne pour le F1, et ce que publie le papier."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(ORDRE))
    larg = 0.2

    for ax, modele, titre in zip(axes, ("fidele", "varlr"),
                                 ("Fidele (lr 0.01)", "Variante (lr 0.001)")):
        def col(c):
            return [100 * df[(df.modele == modele) & (df.attaque == a)][c].iloc[0]
                    for a in ORDRE]

        ax.bar(x - 1.5*larg, [PAPIER_ACC[a] if a == "Clean" else
                              {"FGSM":59,"BIM":52,"PGD":54,"DeepFool":58,
                               "JSMA":72,"CW":32}[a] for a in ORDRE],
               larg, label="F1 du papier", color=GRIS)
        ax.bar(x - 0.5*larg, col("f1_macro"), larg, label="F1 macro", color=ROUGE)
        ax.bar(x + 0.5*larg, col("f1_weighted"), larg, label="F1 pondere", color=BLEU)
        ax.bar(x + 1.5*larg, col("f1_micro"), larg, label="F1 micro", color=VERT)

        ax.set_xticks(x)
        ax.set_xticklabels(ORDRE, rotation=20)
        ax.set_ylabel("F1 (%)")
        ax.set_ylim(0, 105)
        ax.set_title(titre, fontsize=10.5)
        ax.legend(fontsize=8.5)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    fig.suptitle("Le papier ne dit pas quelle moyenne il utilise. Son F1 colle "
                 "au F1 micro,\nqui vaut l'accuracy en multiclasse : il publie "
                 "donc deux fois la meme chose.", fontsize=11)
    plt.tight_layout()
    plt.savefig(sortie, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {sortie.name}")


def heatmap(df, sortie):
    """Toutes les metriques, les deux modeles."""
    cols = [("accuracy", "Accuracy"), ("f1_macro", "F1 macro"),
            ("f1_weighted", "F1 pondere"), ("f1_micro", "F1 micro"),
            ("precision_macro", "Precision macro"),
            ("recall_macro", "Recall macro"),
            ("recall_benign", "Rappel BENIGN")]

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
    for ax, modele, titre in zip(axes, ("fidele", "varlr"),
                                 ("Fidele (lr 0.01, 30 ep)",
                                  "Variante (lr 0.001, 100 ep)")):
        mat = np.array([[df[(df.modele == modele) & (df.attaque == a)][c].iloc[0]
                         for a in ORDRE] for c, _ in cols])
        sns.heatmap(mat, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=1,
                    xticklabels=ORDRE, yticklabels=[lib for _, lib in cols],
                    ax=ax, cbar=False, linewidths=0.5, linecolor="white")
        ax.set_title(titre, fontsize=10.5)
        ax.tick_params(axis="x", rotation=25)
        ax.tick_params(axis="y", rotation=0)

    fig.suptitle("Toutes les metriques, les deux baselines", fontsize=12)
    plt.tight_layout()
    plt.savefig(sortie, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {sortie.name}")


def rappel_benign(df, sortie):
    """Rappel sur le trafic normal, que le papier ne publie pas."""
    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(ORDRE))
    larg = 0.36

    f = [100 * df[(df.modele == "fidele") & (df.attaque == a)].recall_benign.iloc[0]
         for a in ORDRE]
    v = [100 * df[(df.modele == "varlr") & (df.attaque == a)].recall_benign.iloc[0]
         for a in ORDRE]

    ax.bar(x - larg/2, f, larg, label="Fidele", color=ROUGE)
    ax.bar(x + larg/2, v, larg, label="Variante lr", color=VERT)
    ax.set_xticks(x)
    ax.set_xticklabels(ORDRE)
    ax.set_ylabel("Rappel sur le trafic normal (%)")
    ax.set_ylim(85, 101)
    ax.set_title("Rappel sur le trafic normal — pas publie par le papier\n"
                 "Le modele fidele garde 99.8 % sous FGSM parce qu'il classe "
                 "presque tout en BENIGN", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(sortie, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {sortie.name}")


def perturbations(atk_dir, sortie):
    """Amplitude des perturbations par attaque."""
    X = joblib.load(atk_dir / "X_test_clean.pkl")
    l2, linf, touchees = [], [], []
    for a in ATTAQUES:
        X_adv = joblib.load(atk_dir / f"X_adv_test_{a.lower()}.pkl")
        d = np.abs(X_adv - X)
        l2.append(np.linalg.norm(d, axis=1).mean())
        linf.append(d.max(axis=1).mean())
        touchees.append((d > 1e-6).sum(axis=1).mean())
        del X_adv

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, vals, titre, ylab in (
            (axes[0], l2, "Norme L2 moyenne", "L2"),
            (axes[1], linf, "Norme L-infini moyenne", "L-inf"),
            (axes[2], touchees, "Features modifiees", "sur 58")):
        ax.bar(ATTAQUES, vals, color=BLEU)
        for i, val in enumerate(vals):
            ax.text(i, val, f"{val:.2f}" if val < 10 else f"{val:.1f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_title(titre, fontsize=10.5)
        ax.set_ylabel(ylab)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    fig.suptitle("Amplitude des perturbations, echantillon test\n"
                 "JSMA perturbe cinquante fois plus que DeepFool", fontsize=11)
    plt.tight_layout()
    plt.savefig(sortie, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {sortie.name}")


def main():
    cfg = load_config()
    fig_dir = Path(cfg.paths["figures"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg.paths["logs"])
    atk_dir = Path(cfg.paths["attacks"])

    df = pd.read_csv(fig_dir / "evaluation_baselines.csv")

    print("Figures :")
    courbes(log_dir, fig_dir / "baselines_courbes.png")
    accuracy_trois(df, fig_dir / "accuracy_trois_colonnes.png")
    ecarts(df, fig_dir / "ecarts_papier.png")
    metriques_comparees(df, fig_dir / "f1_types_moyenne.png")
    heatmap(df, fig_dir / "heatmap_metriques.png")
    rappel_benign(df, fig_dir / "rappel_benign.png")
    perturbations(atk_dir, fig_dir / "perturbations.png")
    print(f"\nTout est dans {fig_dir}/")


if __name__ == "__main__":
    main()
