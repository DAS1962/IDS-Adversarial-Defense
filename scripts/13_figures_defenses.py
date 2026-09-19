"""Figures des quatre defenses et de l'ensemble."""

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

_l = torch.load
torch.load = lambda *a, **k: _l(*a, **{**k, "map_location": "cpu"})

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.config import load_config

ATTAQUES = ["FGSM", "BIM", "PGD", "DeepFool", "JSMA", "CW"]
ORDRE = ["Clean"] + ATTAQUES
DEFENSES = [("ls", "Lissage"), ("ga", "Gaussienne"),
            ("at", "Adversarial"), ("dae", "Autoencodeur")]

# Table 6 du papier
PAPIER6 = {"ls": 82.12, "ga": 80.05, "at": 80.39, "dae": 82.13}  # moyennes
PAPIER7 = {"ls": 85.90, "ga": 79.80, "at": 80.25, "dae": 84.80}

GRIS, BLEU, VERT, ORANGE, ROUGE = ("#5A6C7A", "#1C7293", "#2E7D5B",
                                   "#D98816", "#B23A32")
COUL = {"ls": BLEU, "ga": ORANGE, "at": VERT, "dae": ROUGE}


def charger(log_dir, suffixe):
    res, ens = {}, {}
    for court, _ in DEFENSES:
        f = sorted(Path(log_dir).glob(f"defense_{court}{suffixe}_*.pkl"))
        if f:
            res[court] = joblib.load(f[-1])
    ev = sorted(Path(log_dir).glob(f"evaluation{suffixe}_*.pkl"))
    d = joblib.load(ev[-1])
    base = d["resultats"][[k for k in d["resultats"] if "reduit" in k][0]]
    e = sorted(Path(log_dir).glob(f"ensemble{suffixe}_*.pkl"))
    if e:
        ens = joblib.load(e[-1])
    return res, base, ens


def f1_par_attaque(res, base, sortie):
    x = np.arange(len(ORDRE))
    larg = 0.16
    fig, ax = plt.subplots(figsize=(13, 5.5))

    ax.bar(x - 2*larg, [base[a]["f1_macro"] for a in ORDRE], larg,
           label="Sans defense", color=GRIS)
    for i, (court, nom) in enumerate(DEFENSES):
        if court not in res:
            continue
        r = res[court]["resultats"]
        ax.bar(x + (i-1)*larg, [r[a]["f1_macro"] for a in ORDRE], larg,
               label=nom, color=COUL[court])

    ax.set_xticks(x)
    ax.set_xticklabels(ORDRE)
    ax.set_ylabel("F1 macro")
    ax.set_title("F1 macro par attaque, quatre defenses\n"
                 "Seul l'entrainement adversarial defend reellement",
                 fontsize=11)
    ax.legend(fontsize=9, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(sortie, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {sortie.name}")


def gain_vs_cout(res, base, ens, sortie):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    pts = []

    for court, nom in DEFENSES:
        if court not in res:
            continue
        r = res[court]["resultats"]
        g = np.mean([r[a]["f1_macro"] - base[a]["f1_macro"] for a in ATTAQUES])
        c = r["Clean"]["f1_macro"] - base["Clean"]["f1_macro"]
        pts.append((c, g, nom, COUL[court], "o"))

    for nom, r in ens.get("resultats", {}).items():
        g = np.mean([r[a]["f1_macro"] - base[a]["f1_macro"] for a in ATTAQUES])
        c = r["Clean"]["f1_macro"] - base["Clean"]["f1_macro"]
        pts.append((c, g, "Ens. " + nom.split(",")[0].split()[0].lower(),
                    "#16232E", "D"))

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    mx = (max(xs) - min(xs)) * 0.25 + 0.05
    my = (max(ys) - min(ys)) * 0.2 + 0.03
    lim = [min(xs) - mx, max(xs) + mx * 1.8]

    ax.plot(lim, [-lim[0], -lim[1]], "--", color=ROUGE, linewidth=1.2,
            label="bilan net nul", zorder=1)
    ax.axhline(0, color="#C9D4DC", linewidth=0.8)
    ax.axvline(0, color="#C9D4DC", linewidth=0.8)

    for c, g, nom, coul, marq in pts:
        ax.scatter(c, g, s=170, marker=marq, color=coul,
                   edgecolor="white", linewidth=1.5, zorder=3)
        ax.annotate(nom, (c, g), textcoords="offset points",
                    xytext=(11, 5), fontsize=9)

    ax.set_xlim(lim)
    ax.set_ylim(min(ys) - my, max(ys) + my)
    ax.set_xlabel("Cout sur donnees propres (ecart de F1 macro au baseline)")
    ax.set_ylabel("Gain moyen sous attaque")
    ax.set_title("Le compromis de chaque defense\n"
                 "Au-dessus de la diagonale : le gain depasse le prix paye",
                 fontsize=11)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(sortie, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {sortie.name}")


def face_au_papier(res, sortie):
    """Accuracy micro sur les sept jeux, face a leur Table 7."""
    noms = [n for c, n in DEFENSES if c in res]
    nous = [100 * np.mean([res[c]["resultats"][a]["accuracy"] for a in ORDRE])
            for c, _ in DEFENSES if c in res]
    papier = [PAPIER7[c] for c, _ in DEFENSES if c in res]

    x = np.arange(len(noms))
    larg = 0.36
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - larg/2, papier, larg, label="Papier (Table 7)", color=GRIS)
    ax.bar(x + larg/2, nous, larg, label="Nous (micro, sept jeux)", color=VERT)

    for i, (p, n) in enumerate(zip(papier, nous)):
        ax.text(i - larg/2, p + 1, f"{p:.1f}", ha="center", fontsize=9)
        ax.text(i + larg/2, n + 1, f"{n:.1f}", ha="center", fontsize=9,
                fontweight="bold" if n > p else "normal")

    ax.set_xticks(x)
    ax.set_xticklabels(noms)
    ax.set_ylabel("Accuracy moyenne (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Face a leur Table 7\n"
                 "L'entrainement adversarial est la seule ou nous les depassons",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(sortie, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {sortie.name}")


def mcc(res, sortie):
    """MCC binaire, face a leur Table 9."""
    PAPIER9 = {"ls": 0.698, "ga": 0.596, "at": 0.608, "dae": 0.680}
    noms = [n for c, n in DEFENSES if c in res]
    x = np.arange(len(noms))
    larg = 0.26

    fig, ax = plt.subplots(figsize=(10.5, 5))
    p = [PAPIER9[c] for c, _ in DEFENSES if c in res]
    clean = [res[c]["resultats"]["Clean"]["mcc_binaire"]
             for c, _ in DEFENSES if c in res]
    mini = [min(res[c]["resultats"][a]["mcc_binaire"] for a in ATTAQUES)
            for c, _ in DEFENSES if c in res]

    ax.bar(x - larg, p, larg, label="Papier (Table 9)", color=GRIS)
    ax.bar(x, clean, larg, label="Nous, donnees propres", color=VERT)
    ax.bar(x + larg, mini, larg, label="Nous, pire attaque", color=ROUGE)
    ax.axhline(0, color="#16232E", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(noms)
    ax.set_ylabel("MCC binaire")
    ax.set_title("MCC binaire, face a leur Table 9\n"
                 "Deux valeurs negatives : le modele fait pire qu'un tirage au sort",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(sortie, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {sortie.name}")


def rappel_benign(res, base, sortie):
    x = np.arange(len(ORDRE))
    larg = 0.16
    fig, ax = plt.subplots(figsize=(13, 5))

    ax.bar(x - 2*larg, [base[a]["recall_benign"] for a in ORDRE], larg,
           label="Sans defense", color=GRIS)
    for i, (court, nom) in enumerate(DEFENSES):
        if court not in res:
            continue
        r = res[court]["resultats"]
        ax.bar(x + (i-1)*larg, [r[a]["recall_benign"] for a in ORDRE], larg,
               label=nom, color=COUL[court])

    ax.set_xticks(x)
    ax.set_xticklabels(ORDRE)
    ax.set_ylabel("Rappel sur le trafic normal")
    ax.set_ylim(0.7, 1.02)
    ax.set_title("Rappel sur le trafic normal — pas publie par le papier\n"
                 "Le lissage tombe a 0.75 sous FGSM : un quart du trafic "
                 "legitime rejete\nAxe tronque a 0.70", fontsize=11)
    ax.legend(fontsize=9, ncol=3, loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(sortie, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {sortie.name}")


def heatmap(res, base, ens, sortie):
    lignes = [("Sans defense", base)]
    lignes += [(n, res[c]["resultats"]) for c, n in DEFENSES if c in res]
    lignes += [("Ens. " + n.split(",")[0].split()[0].lower(), r)
               for n, r in ens.get("resultats", {}).items()]

    mat = np.array([[r[a]["f1_macro"] for a in ORDRE] for _, r in lignes])
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(lignes) + 2))
    sns.heatmap(mat, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=1,
                xticklabels=ORDRE, yticklabels=[n for n, _ in lignes],
                ax=ax, cbar_kws={"label": "F1 macro"},
                linewidths=0.5, linecolor="white")
    ax.set_title("F1 macro : chaque configuration face a chaque attaque",
                 fontsize=12, pad=12)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(sortie, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {sortie.name}")


def courbes(res, sortie):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    for court, nom in DEFENSES:
        if court not in res or court == "dae":
            continue
        h = res[court]["historique"]
        ep = range(1, len(h["val_acc"]) + 1)
        ax1.plot(ep, h["val_acc"], color=COUL[court], linewidth=1.6, label=nom)
        ax2.plot(ep, h["val_loss"], color=COUL[court], linewidth=1.6, label=nom)
        ax1.axvline(res[court]["meilleur_epoch"], color=COUL[court],
                    linestyle=":", alpha=0.4)

    ax1.set_xlabel("Passage sur les donnees")
    ax1.set_ylabel("Accuracy de validation")
    ax1.set_title("Accuracy de validation\n"
                  "La gaussienne plafonne des le passage 19", fontsize=10.5)
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


def main():
    parseur = argparse.ArgumentParser()
    parseur.add_argument("--suffixe", default="")
    args = parseur.parse_args()

    cfg = load_config()
    fig_dir = Path(cfg.paths["figures"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    res, base, ens = charger(cfg.paths["logs"], args.suffixe)
    sfx = args.suffixe

    print("Figures :")
    f1_par_attaque(res, base, fig_dir / f"defenses_f1{sfx}.png")
    gain_vs_cout(res, base, ens, fig_dir / f"defenses_gain_cout{sfx}.png")
    face_au_papier(res, fig_dir / f"defenses_vs_papier{sfx}.png")
    mcc(res, fig_dir / f"defenses_mcc{sfx}.png")
    rappel_benign(res, base, fig_dir / f"defenses_recall{sfx}.png")
    heatmap(res, base, ens, fig_dir / f"defenses_heatmap{sfx}.png")
    courbes(res, fig_dir / f"defenses_courbes{sfx}.png")
    print(f"\nTout est dans {fig_dir}/")


if __name__ == "__main__":
    main()
