"""
Figures des defenses et de l'agregation par ensemble, plus CSV recapitulatif.

Pourquoi ce script existe
--------------------------
Les chiffres des defenses ont ete recopies a la main depuis les journaux
d'execution vers le README, la presentation et le compte-rendu. Deux valeurs
fausses s'y sont glissees : AT+LS annonce a +0.1015 au lieu de +0.0917, et
GA+LS a +0.1027 au lieu de +0.0844. Les .pkl et les .out concordaient
pourtant parfaitement ; l'erreur venait uniquement de la transcription.

Ce script lit donc directement les .pkl produits par les scripts 11 a 14 et
16, recalcule tout, et ecrit results/figures/defenses_summary.csv. Ce fichier
devient la source unique : tout chiffre de defense figurant dans un document
doit en provenir, jamais d'une lecture de journal.

Selection des fichiers
----------------------
Plusieurs executions existent par defense (trois pour AT, deux pour GA, cinq
pour le DAE). Le script retient par defaut la plus recente, mais affiche
toutes les executions trouvees avec leurs hyperparametres, pour que le choix
soit verifiable plutot qu'implicite.

Figures produites
-----------------
  defenses_f1_macro_par_attaque.png   comparaison des 4 defenses au baseline
  defenses_gain_vs_cout.png           le compromis, en un coup d'oeil
  defenses_courbes.png                courbes d'entrainement des 3 defenses
                                      reentrainees
  defenses_heatmap.png                defenses x attaques, F1 macro
  defenses_recall_benign.png          degradation du trafic normal
  ensemble_trois_methodes.png         les 3 agregations par attaque
  ensemble_poids.png                  poids retenus par l'optimisation
  ensemble_vs_meilleure_defense.png   le resultat central de l'article
"""

import sys
import glob
import os
from datetime import datetime
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

# Palette : une couleur par defense, stable d'une figure a l'autre.
COULEURS = {
    "Baseline": "#5A6C7A",
    "LS": "#1C7293",
    "AT": "#2E7D5B",
    "GA": "#D98816",
    "DAE": "#B23A32",
    "Vote majoritaire": "#1C7293",
    "Parts égales": "#2E7D5B",
    "Optimisé": "#D98816",
}

NOMS_LONGS = {
    "LS": "Lissage des étiquettes",
    "AT": "Entraînement adversarial",
    "GA": "Augmentation gaussienne",
    "DAE": "Autoencodeur débruiteur",
}


def dernier(motif, log_dir):
    """Fichier le plus recent correspondant au motif, ou None."""
    fichiers = sorted(glob.glob(str(log_dir / motif)), key=os.path.getmtime)
    return Path(fichiers[-1]) if fichiers else None


def inventaire(log_dir):
    """
    Liste toutes les executions trouvees, avec leurs hyperparametres.

    Affiche systematiquement : le choix du fichier retenu doit etre visible
    et verifiable, pas implicite.
    """
    print("=" * 78)
    print("Executions trouvees (le fichier retenu est marque d'une fleche)")
    print("=" * 78)
    retenus = {}
    for cle, motif in [("LS", "defense_ls_*.pkl"), ("AT", "defense_at_*.pkl"),
                       ("GA", "defense_ga_*.pkl"), ("DAE", "defense_dae_*.pkl")]:
        fichiers = sorted(glob.glob(str(log_dir / motif)), key=os.path.getmtime)
        print(f"\n{NOMS_LONGS[cle]}")
        if not fichiers:
            print("   aucune execution trouvee")
            continue
        for f in fichiers:
            d = joblib.load(f)
            hp = d.get("hyperparameters", {})
            identite = d.get("best_epoch") == 0
            marque = "  -> " if f == fichiers[-1] else "     "
            if identite:
                marque = "  id "
            ls = hp.get("label_smoothing", "absent")
            detail = f"lissage={ls}"
            if "at_config" in hp:
                a = hp["at_config"]
                detail += f", attaque={a.get('attack')}, pas={a.get('steps', 1)}"
            if "gaussian_sigma" in hp:
                detail += f", sigma={hp['gaussian_sigma']}"
            if "architecture" in hp:
                detail += f", {hp['architecture']}"
            if identite:
                detail += "  [converge vers l'identite : sert de reference]"
            horo = datetime.fromtimestamp(os.path.getmtime(f))
            print(f"{marque}{Path(f).name}  ({horo:%d/%m %H:%M})  {detail}")
        retenus[cle] = Path(fichiers[-1])
    print()
    return retenus


def charger_defense(chemin):
    """Dictionnaire attaque -> metriques, depuis un .pkl de defense."""
    d = joblib.load(chemin)
    return {r["attack"]: r for r in d["results"]}, d


def construire_recap(baseline, baseline_acc, baseline_rb, defenses, ensemble):
    """
    Tableau recapitulatif : une ligne par configuration, avec le F1 macro par
    attaque, le gain moyen, le cout sur donnees propres et le bilan net.

    Le gain est la moyenne des ecarts au baseline sur les six attaques. Le
    cout est l'ecart au baseline sur les donnees propres. Le bilan net est
    leur somme : une defense doit etre jugee sur les deux plans.
    """
    lignes = []

    def ajouter(nom, categorie, res):
        f1 = {a: res[a]["f1_macro"] for a in ORDRE}
        gain = float(np.mean([f1[a] - baseline[a] for a in ATTAQUES]))
        cout = f1["Clean"] - baseline["Clean"]
        ligne = {"configuration": nom, "categorie": categorie}
        ligne.update({f"f1_macro_{a}": f1[a] for a in ORDRE})
        ligne.update({f"accuracy_{a}": res[a]["accuracy"] for a in ORDRE})
        ligne.update({f"recall_benign_{a}": res[a]["recall_benign"] for a in ORDRE})
        ligne["gain_attaques"] = gain
        ligne["cout_clean"] = cout
        ligne["bilan_net"] = gain + cout
        lignes.append(ligne)

    # Le baseline lui-meme, comme reference explicite.
    ref = {a: {"f1_macro": baseline[a], "accuracy": baseline_acc[a],
               "recall_benign": baseline_rb[a]} for a in ORDRE}
    ajouter("Baseline (sans défense)", "référence", ref)

    for cle, (res, _) in defenses.items():
        ajouter(NOMS_LONGS[cle], "défense", res)

    for nom, res in ensemble.items():
        ajouter(f"Ensemble - {nom}", "ensemble", res)

    return pd.DataFrame(lignes)


# ---------- Figures ----------

def fig_f1_par_attaque(recap, sortie):
    """Barres groupees : F1 macro par attaque, baseline contre les 4 defenses."""
    configs = ["Baseline (sans défense)"] + [NOMS_LONGS[c] for c in ["LS", "AT", "GA", "DAE"]]
    cles = ["Baseline", "LS", "AT", "GA", "DAE"]

    x = np.arange(len(ORDRE))
    largeur = 0.16
    fig, ax = plt.subplots(figsize=(13, 5.5))

    for i, (conf, cle) in enumerate(zip(configs, cles)):
        ligne = recap[recap["configuration"] == conf]
        if ligne.empty:
            continue
        vals = [ligne[f"f1_macro_{a}"].iloc[0] for a in ORDRE]
        ax.bar(x + (i - 2) * largeur, vals, largeur,
               label=conf, color=COULEURS[cle], edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(ORDRE)
    ax.set_ylabel("F1 macro")
    ax.set_title("F1 macro par attaque : baseline et quatre défenses\n"
                 "Le F1 macro pèse chaque classe également ; l'accuracy serait "
                 "plafonnée par les 83.1 % de trafic normal",
                 fontsize=11)
    ax.legend(fontsize=9, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(sortie, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {sortie.name}")


def fig_gain_vs_cout(recap, sortie):
    """
    Nuage gain / cout. Chaque defense est un point ; la diagonale marque le
    bilan net nul. Rend le compromis immediatement lisible.

    Les bornes suivent les donnees et non la diagonale, sinon les points se
    tassent dans un coin. Les etiquettes sont decalees individuellement : les
    configurations a faible cout se superposent sinon.
    """
    pts = recap[recap["categorie"].isin(["défense", "ensemble"])]

    x_min, x_max = float(pts["cout_clean"].min()), float(pts["cout_clean"].max())
    y_min, y_max = float(pts["gain_attaques"].min()), float(pts["gain_attaques"].max())
    marge_x = (x_max - x_min) * 0.18 + 0.02
    marge_y = (y_max - y_min) * 0.22 + 0.01
    lim = [x_min - marge_x, x_max + marge_x * 2.6]
    lim_y = [y_min - marge_y, y_max + marge_y]

    fig, ax = plt.subplots(figsize=(11.5, 6.5))

    ax.plot(lim, [-lim[0], -lim[1]], "--", color="#B23A32", linewidth=1.2,
            label="bilan net nul", zorder=1)
    ax.axhline(0, color="#C9D4DC", linewidth=0.8, zorder=1)
    ax.axvline(0, color="#C9D4DC", linewidth=0.8, zorder=1)

    # Les points a faible cout se tassent a droite. Les etiquettes sont
    # reparties sur des hauteurs differentes pour ne pas se recouvrir.
    decalages = {
        "Autoencodeur débruiteur":     (12, 8),
        "Augmentation gaussienne":     (12, -16),
        "Ensemble - Vote majoritaire": (-12, 8),
        "Ensemble - Parts égales":     (13, 3),
        "Entraînement adversarial":    (13, -5),
        "Ensemble - Optimisé":         (13, 2),
        "Lissage des étiquettes":      (13, 2),
    }

    for _, r in pts.iterrows():
        est_ens = r["categorie"] == "ensemble"
        couleur = "#5A6C7A"
        for cle, long in NOMS_LONGS.items():
            if r["configuration"] == long:
                couleur = COULEURS[cle]
        if est_ens:
            couleur = COULEURS.get(r["configuration"].replace("Ensemble - ", ""), "#5A6C7A")

        ax.scatter(r["cout_clean"], r["gain_attaques"],
                   s=185 if est_ens else 150,
                   marker="D" if est_ens else "o",
                   color=couleur, edgecolor="white", linewidth=1.5, zorder=3)

        dx, dy = decalages.get(r["configuration"], (10, 6))
        ax.annotate(r["configuration"].replace("Ensemble - ", "Ens. "),
                    (r["cout_clean"], r["gain_attaques"]),
                    textcoords="offset points", xytext=(dx, dy), fontsize=9,
                    ha="right" if dx < 0 else "left", zorder=4)

    ax.set_xlim(lim)
    ax.set_ylim(lim_y)
    ax.set_xlabel("Coût sur données propres (écart de F1 macro au baseline)")
    ax.set_ylabel("Gain moyen sous attaque (écart de F1 macro)")
    ax.set_title("Le compromis de chaque défense\n"
                 "Au-dessus de la diagonale rouge : le gain sous attaque dépasse "
                 "le prix payé en temps normal", fontsize=11)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(sortie, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {sortie.name}")


def fig_courbes(defenses, sortie):
    """
    Courbes d'entrainement des trois defenses reentrainees. Le DAE est exclu :
    son historique suit une MSE et des F1 par attaque, pas une accuracy de
    validation, donc il ne se superpose pas aux trois autres.
    """
    dispo = [(c, d) for c, (_, d) in defenses.items()
             if c != "DAE" and "val_acc" in d.get("history", {})]
    if not dispo:
        print("  (courbes ignorees : aucun historique d'accuracy)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    for cle, d in dispo:
        h = d["history"]
        ep = range(1, len(h["val_acc"]) + 1)
        ax1.plot(ep, h["val_acc"], color=COULEURS[cle], linewidth=1.6,
                 label=NOMS_LONGS[cle])
        ax2.plot(ep, h["val_loss"], color=COULEURS[cle], linewidth=1.6,
                 label=NOMS_LONGS[cle])
        best = d.get("best_epoch")
        if best:
            ax1.axvline(best, color=COULEURS[cle], linestyle=":", alpha=0.45)

    ax1.set_xlabel("Passage sur les données")
    ax1.set_ylabel("Accuracy de validation")
    ax1.set_title("Accuracy de validation\n(pointillés : meilleur passage retenu)",
                  fontsize=10.5)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("Passage sur les données")
    ax2.set_ylabel("Perte de validation")
    ax2.set_title("Perte de validation", fontsize=10.5)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(sortie, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {sortie.name}")


def fig_heatmap(recap, sortie):
    """Carte de chaleur configurations x attaques, en F1 macro."""
    lignes = recap[recap["categorie"].isin(["référence", "défense", "ensemble"])]
    mat = np.array([[r[f"f1_macro_{a}"] for a in ORDRE] for _, r in lignes.iterrows()])
    noms = [r["configuration"].replace("Ensemble - ", "Ensemble ")
            for _, r in lignes.iterrows()]

    fig, ax = plt.subplots(figsize=(11, 0.52 * len(noms) + 2.2))
    sns.heatmap(mat, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=0.9,
                xticklabels=ORDRE, yticklabels=noms, ax=ax,
                cbar_kws={"label": "F1 macro"}, linewidths=0.5, linecolor="white")
    ax.set_title("F1 macro : chaque configuration face à chaque attaque",
                 fontsize=12, pad=12)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(sortie, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {sortie.name}")


def fig_recall_benign(recap, sortie):
    """
    Rappel sur le trafic normal. Revele les defenses qui achetent leur
    robustesse en degradant le trafic legitime, ce que le gain moyen masque.
    """
    configs = ["Baseline (sans défense)"] + [NOMS_LONGS[c] for c in ["LS", "AT", "GA", "DAE"]]
    cles = ["Baseline", "LS", "AT", "GA", "DAE"]

    x = np.arange(len(ORDRE))
    largeur = 0.16
    fig, ax = plt.subplots(figsize=(13, 5))

    for i, (conf, cle) in enumerate(zip(configs, cles)):
        ligne = recap[recap["configuration"] == conf]
        if ligne.empty:
            continue
        vals = [ligne[f"recall_benign_{a}"].iloc[0] for a in ORDRE]
        ax.bar(x + (i - 2) * largeur, vals, largeur,
               label=conf, color=COULEURS[cle], edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(ORDRE)
    # Les rappels vivent tous entre 0.7 et 1 : un axe partant de zero
    # ecraserait les ecarts. Il est donc tronque, et la mention l'annonce.
    toutes = [ligne[f"recall_benign_{a}"].iloc[0]
              for conf in configs
              for ligne in [recap[recap["configuration"] == conf]]
              if not ligne.empty
              for a in ORDRE]
    bas = max(0.0, min(toutes) - 0.08)
    ax.set_ylabel("Rappel sur le trafic normal")
    ax.set_ylim(bas, 1.02)
    mention = f"\nAxe tronqué à partir de {bas:.2f} pour rendre les écarts lisibles" if bas > 0 else ""
    ax.set_title("Rappel sur le trafic normal\n"
                 "Une valeur basse signifie que la défense crée des faux positifs "
                 "sur du trafic légitime" + mention, fontsize=11)
    ax.legend(fontsize=9, ncol=3, loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(sortie, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {sortie.name}")


def fig_ensemble_methodes(recap, baseline, sortie):
    """Les trois methodes d'agregation, par attaque, avec le baseline."""
    methodes = ["Vote majoritaire", "Parts égales", "Optimisé"]
    x = np.arange(len(ORDRE))
    largeur = 0.2

    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.bar(x - 1.5 * largeur, [baseline[a] for a in ORDRE], largeur,
           label="Sans défense", color=COULEURS["Baseline"],
           edgecolor="white", linewidth=0.5)

    for i, m in enumerate(methodes):
        ligne = recap[recap["configuration"] == f"Ensemble - {m}"]
        if ligne.empty:
            continue
        vals = [ligne[f"f1_macro_{a}"].iloc[0] for a in ORDRE]
        ax.bar(x + (i - 0.5) * largeur, vals, largeur, label=m,
               color=COULEURS[m], edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(ORDRE)
    ax.set_ylabel("F1 macro")
    ax.set_title("Les trois méthodes d'agrégation par ensemble", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(sortie, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {sortie.name}")


def fig_poids(poids, sortie):
    """Poids retenus par l'optimisation, avec la reference des parts egales."""
    noms = list(poids.keys())
    vals = [poids[n] for n in noms]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    etiquettes = [NOMS_LONGS.get(n, n).replace(" ", "\n", 1) for n in noms]
    barres = ax.bar(etiquettes, vals,
                    color=[COULEURS.get(n, "#5A6C7A") for n in noms],
                    edgecolor="white", linewidth=0.8)
    for b, v in zip(barres, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008,
                f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")

    ax.axhline(0.25, color="#B23A32", linestyle="--", linewidth=1.3,
               label="parts égales (0.25)")
    ax.set_ylabel("Poids")
    ax.set_ylim(0, max(vals) * 1.28)
    ax.set_title("Poids retenus par l'optimisation\n"
                 "Optimisés sur la validation propre, où l'autoencodeur est le "
                 "plus faible : il est donc presque écarté", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(sortie, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {sortie.name}")


def fig_ensemble_vs_defense(recap, sortie):
    """
    Le resultat central de l'article : gain moyen de chaque configuration.
    Presente aussi le bilan net, ou la conclusion differe.
    """
    pts = recap[recap["categorie"].isin(["défense", "ensemble"])].copy()
    pts = pts.sort_values("gain_attaques")
    noms = [c.replace("Ensemble - ", "Ensemble ") for c in pts["configuration"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    couleurs = ["#2E7D5B" if c == "ensemble" else "#1C7293"
                for c in pts["categorie"]]
    ax1.barh(noms, pts["gain_attaques"], color=couleurs, edgecolor="white")
    for i, v in enumerate(pts["gain_attaques"]):
        ax1.text(v + 0.002, i, f"{v:+.4f}", va="center", fontsize=9)
    ax1.set_xlabel("Gain moyen en F1 macro sous attaque")
    ax1.set_title("Gain brut sous attaque\nL'ensemble dépasse chaque défense",
                  fontsize=10.5)
    ax1.grid(axis="x", alpha=0.3)
    ax1.set_axisbelow(True)
    ax1.set_xlim(0, float(pts["gain_attaques"].max()) * 1.25)

    pts2 = pts.sort_values("bilan_net")
    noms2 = [c.replace("Ensemble - ", "Ensemble ") for c in pts2["configuration"]]
    couleurs2 = ["#2E7D5B" if v > 0 else "#B23A32" for v in pts2["bilan_net"]]
    ax2.barh(noms2, pts2["bilan_net"], color=couleurs2, edgecolor="white")
    for i, v in enumerate(pts2["bilan_net"]):
        dec = 0.006 if v > 0 else -0.006
        ax2.text(v + dec, i, f"{v:+.4f}", va="center", fontsize=9,
                 ha="left" if v > 0 else "right")
    ax2.axvline(0, color="#16232E", linewidth=1)
    etendue = float(pts2["bilan_net"].max()) - float(pts2["bilan_net"].min())
    ax2.set_xlim(float(pts2["bilan_net"].min()) - etendue * 0.22,
                 float(pts2["bilan_net"].max()) + etendue * 0.18)
    ax2.set_xlabel("Bilan net (gain sous attaque + coût sur données propres)")
    ax2.set_title("Bilan net\nLa conclusion s'inverse : une défense seule fait mieux",
                  fontsize=10.5)
    ax2.grid(axis="x", alpha=0.3)
    ax2.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(sortie, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {sortie.name}")


def main():
    print("=" * 78)
    print("Figures des defenses et de l'ensemble")
    print(f"Date : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 78 + "\n")

    cfg = load_config()
    log_dir = Path(cfg.paths["logs"])
    fig_dir = Path(cfg.paths["figures"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    retenus = inventaire(log_dir)

    # Reference du baseline.
    #
    # attacks_results_*.pkl, produit par le script 08, ne contient que les six
    # attaques : sa liste de resultats n'a pas de ligne « Clean ».
    #
    # Le run v4 du DAE en fournit une meilleure. Aucune de ses epochs n'ayant
    # battu l'identite, le checkpoint retenu EST l'identite : ses sept lignes
    # sont donc exactement celles du baseline, mesurees dans les memes
    # conditions que les defenses, avec toutes les metriques dont
    # recall_benign. On le reconnait a best_epoch == 0.
    res_ref = None
    ref_nom = None

    for f in sorted(glob.glob(str(log_dir / "defense_dae_*.pkl")),
                    key=os.path.getmtime, reverse=True):
        d = joblib.load(f)
        if d.get("best_epoch") == 0:
            r = {x["attack"]: x for x in d["results"]}
            if "Clean" in r and "recall_benign" in r["Clean"]:
                res_ref, ref_nom = r, Path(f).name + " (DAE identite)"
                break

    if res_ref is None:
        # Repli : un fichier d'attaques contenant bien une ligne « Clean ».
        for f in sorted(glob.glob(str(log_dir / "attacks_results_*.pkl")),
                        key=os.path.getmtime, reverse=True):
            d = joblib.load(f)
            r = {x["attack"]: x for x in d.get("results", [])}
            if "Clean" in r:
                res_ref, ref_nom = r, Path(f).name
                break

    if res_ref is None:
        raise RuntimeError(
            "Aucune reference de baseline trouvee. Il faut soit un run de DAE "
            "ayant converge vers l'identite (best_epoch == 0), soit un "
            "attacks_results_*.pkl contenant une ligne « Clean »."
        )

    manquantes = [a for a in ORDRE if a not in res_ref]
    if manquantes:
        raise RuntimeError(f"{ref_nom} : lignes manquantes {manquantes}")

    baseline = {a: res_ref[a]["f1_macro"] for a in ORDRE}
    baseline_acc = {a: res_ref[a]["accuracy"] for a in ORDRE}
    baseline_rb = {a: res_ref[a].get("recall_benign", float("nan")) for a in ORDRE}
    print(f"Reference du baseline : {ref_nom}")
    print("  F1 macro : " + "  ".join(f"{a}={baseline[a]:.4f}" for a in ORDRE))
    print()

    defenses = {}
    for cle, chemin in retenus.items():
        defenses[cle] = charger_defense(chemin)

    chemin_ens = dernier("ensemble_results_*.pkl", log_dir)
    ensemble = {}
    poids = {}
    if chemin_ens:
        e = joblib.load(chemin_ens)
        correspondance = {
            "Vote majoritaire": "majority_voting",
            "Parts égales": "weighted_average_equal",
            "Optimisé": "weighted_average_optimized",
        }
        for nom, cle in correspondance.items():
            if cle in e:
                ensemble[nom] = {r["attack"]: r for r in e[cle]}
        poids = e.get("optimal_weights", {})
        print(f"Ensemble : {chemin_ens.name}")
    else:
        print("Aucun fichier d'ensemble trouve, figures d'ensemble ignorees.")
    print()

    recap = construire_recap(baseline, baseline_acc, baseline_rb, defenses, ensemble)

    csv = fig_dir / "defenses_summary.csv"
    recap.to_csv(csv, index=False, float_format="%.6f")

    print("=" * 78)
    print("Recapitulatif — source unique pour tout document")
    print("=" * 78)
    colonnes = ["configuration", "f1_macro_Clean", "gain_attaques",
                "cout_clean", "bilan_net"]
    affichage = recap[colonnes].copy()
    affichage.columns = ["Configuration", "F1 propre", "Gain attaques",
                         "Coût propre", "Bilan net"]
    print(affichage.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print("=" * 78)
    print(f"\nCSV : {csv}\n")

    print("Figures :")
    fig_f1_par_attaque(recap, fig_dir / "defenses_f1_macro_par_attaque.png")
    fig_gain_vs_cout(recap, fig_dir / "defenses_gain_vs_cout.png")
    fig_courbes(defenses, fig_dir / "defenses_courbes.png")
    fig_heatmap(recap, fig_dir / "defenses_heatmap.png")
    fig_recall_benign(recap, fig_dir / "defenses_recall_benign.png")
    if ensemble:
        fig_ensemble_methodes(recap, baseline, fig_dir / "ensemble_trois_methodes.png")
        fig_ensemble_vs_defense(recap, fig_dir / "ensemble_vs_meilleure_defense.png")
        if poids:
            fig_poids(poids, fig_dir / "ensemble_poids.png")

    print(f"\nTermine. Tout est dans {fig_dir}/")


if __name__ == "__main__":
    main()
