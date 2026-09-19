"""
Defense par lissage des etiquettes. Algorithme 2 du papier.

"generate a smoothed label vector for training and test labels, train the
IDS classifier with a smooth label vector"

y_ls = (1 - alpha) * y_hot + alpha / K, avec K = 15 classes. Le papier ne
documente pas alpha ; on prend 0.1, valeur usuelle depuis Muller et al. 2019
qu'il cite lui-meme.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.defenses.commun import (baseline_reference, entrainer, evaluer,
                                 modele_neuf, resume, sauvegarder)
from src.utils.commun import charger_donnees
from src.utils.config import load_config


def main():
    print("=" * 78)
    print("Defense LS - lissage des etiquettes (Algorithme 2)")
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

    X_tr, y_tr, X_va, y_va, _, _ = charger_donnees(cfg.paths["processed"])
    alpha = cfg.defenses["LabelSmoothing"]["alpha"]
    print(f"train {X_tr.shape}, alpha = {alpha}\n")

    print("Entrainement :")
    model = modele_neuf(cfg, device)
    hist, best_ep, best_acc, duree = entrainer(
        model, X_tr, y_tr, X_va, y_va, cfg, device, alpha)
    print()

    res = evaluer(model, cfg, device, labels, Path(cfg.paths["attacks"]))
    base, nom_base = baseline_reference(cfg.paths["logs"])
    print(f"Reference : {nom_base}\n")
    bilan = resume(res, base, "Label Smoothing")
    sauvegarder(cfg, "ls", model, res, hist, best_ep, best_acc,
                duree, bilan, args.suffixe)


if __name__ == "__main__":
    main()
