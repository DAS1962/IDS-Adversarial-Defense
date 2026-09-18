"""
Defense par augmentation gaussienne. Algorithme 4 du papier.

Attention : l'Algorithme 4 est INTITULE "Adversarial Training Defense" mais
son contenu decrit l'augmentation gaussienne ("generate a Gaussian augmented
dataset for training and test samples, train the IDS classifier with the
Gaussian augmented dataset and smooth train labels"). Les titres des
Algorithmes 3 et 4 sont intervertis dans l'article. On suit le contenu.

Le bruit est ajoute a chaque batch, puis borne dans le domaine des features.
Le papier ne documente pas sigma ; on prend 0.02, valeur retenue apres le
balayage 0.02 / 0.05 / 0.1 fait sur la branche main.
"""

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
    print("Defense GA - augmentation gaussienne (Algorithme 4)")
    print(f"Date : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 78 + "\n")

    cfg = load_config()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = list(range(cfg.dataset["num_classes"]))

    X_tr, y_tr, X_va, y_va, _, _ = charger_donnees(cfg.paths["processed"])
    alpha = cfg.defenses["LabelSmoothing"]["alpha"]
    sigma = cfg.defenses["GaussianAugmentation"]["sigma"]
    print(f"train {X_tr.shape}, sigma = {sigma}, alpha = {alpha}")
    print(f"ecart-type moyen des features : {X_tr.std(axis=0).mean():.4f}\n")

    print("Entrainement :")
    model = modele_neuf(cfg, device)
    hist, best_ep, best_acc, duree = entrainer(
        model, X_tr, y_tr, X_va, y_va, cfg, device, alpha, sigma=sigma)
    print()

    res = evaluer(model, cfg, device, labels, Path(cfg.paths["attacks"]))
    base, nom_base = baseline_reference(cfg.paths["logs"])
    print(f"Reference : {nom_base}\n")
    bilan = resume(res, base, f"Gaussian Augmentation (sigma={sigma})")
    sauvegarder(cfg, "ga", model, res, hist, best_ep, best_acc, duree, bilan)


if __name__ == "__main__":
    main()
