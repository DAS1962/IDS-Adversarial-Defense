"""
Defense par entrainement adversarial. Algorithme 3 du papier.

Attention : l'Algorithme 3 est INTITULE "Gaussian Augmentation Defense" mais
son contenu decrit l'entrainement adversarial. Son entree est "Previously
Generated Adversarial examples of four attack categories (FGSM, DEEPFOOL,
BIM, JSMA)" et son etape "generate the adversarial augmented dataset for
train, test samples". Les titres des Algorithmes 3 et 4 sont intervertis.

Point important : les exemples viennent de l'echantillon TRAIN (40 000
lignes, script 03), ceux de l'evaluation viennent du TEST (20 000 lignes).
Les deux jeux sont disjoints, donc le modele ne peut pas memoriser les
perturbations qui serviront a le tester.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.defenses.commun import (baseline_reference, charger_adv_train,
                                 entrainer, evaluer, modele_neuf, resume,
                                 sauvegarder)
from src.utils.commun import charger_donnees
from src.utils.config import load_config


def main():
    print("=" * 78)
    print("Defense AT - entrainement adversarial (Algorithme 3)")
    print(f"Date : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 78 + "\n")

    parseur = argparse.ArgumentParser()
    parseur.add_argument("--suffixe", default="",
                         help="ajoute au nom des sorties, pour ne rien ecraser")
    args = parseur.parse_args()

    cfg = load_config()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = list(range(cfg.dataset["num_classes"]))
    atk_dir = Path(cfg.paths["attacks"])

    X_tr, y_tr, X_va, y_va, _, _ = charger_donnees(cfg.paths["processed"])
    alpha = cfg.defenses["LabelSmoothing"]["alpha"]
    print(f"train {X_tr.shape}, alpha = {alpha}\n")

    print("Exemples adversariaux de l'echantillon train :")
    X_adv, y_adv = charger_adv_train(atk_dir)
    X_aug = np.concatenate([X_tr, X_adv])
    y_aug = np.concatenate([y_tr, y_adv])
    part = len(X_adv) / len(X_aug)
    print(f"    jeu augmente : {len(X_tr):,} propres + {len(X_adv):,} "
          f"adversariaux = {len(X_aug):,} ({part:.1%} adversarial)\n")
    del X_adv, y_adv

    print("Entrainement :")
    model = modele_neuf(cfg, device)
    hist, best_ep, best_acc, duree = entrainer(
        model, X_aug, y_aug, X_va, y_va, cfg, device, alpha)
    print()

    res = evaluer(model, cfg, device, labels, atk_dir)
    base, nom_base = baseline_reference(cfg.paths["logs"])
    print(f"Reference : {nom_base}\n")
    bilan = resume(res, base, "Adversarial Training (4 attaques)")

    # PGD et C&W ne sont pas dans les quatre attaques d'entrainement : leur
    # gain mesure une vraie generalisation, pas une reconnaissance.
    non_vues = [a for a in ("PGD", "CW")]
    g = np.mean([res[a]["f1_macro"] - base[a]["f1_macro"] for a in non_vues])
    print(f"\nGain sur PGD et C&W, absentes de l'entrainement : {g:+.4f}")

    sauvegarder(cfg, "at", model, res, hist, best_ep, best_acc,
                duree, bilan, args.suffixe)


if __name__ == "__main__":
    main()
