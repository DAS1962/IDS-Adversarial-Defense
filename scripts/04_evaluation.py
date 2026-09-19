"""
Evaluation des deux baselines sous les six attaques, toutes metriques.

Les memes X_adv passent dans les deux modeles, donc l'ecart vient du
detecteur et pas des attaques.

  baseline.pth        lr 0.01, 30 epochs   -> exactement le papier
  baseline_varlr.pth  lr 0.001, 100 epochs -> seul le lr change

Test : 20 000 lignes a 50 % benin, donc plancher a 50 %. Premiere fois qu'on
peut comparer directement a leur Table 5.

Metriques calculees : accuracy, precision/recall/F1 en macro, pondere et
micro, MCC multiclasse et binaire, TPR, FPR, TNR, FNR, rappel BENIGN.
Le papier ne publie que l'accuracy, precision, recall, F1 (Tables 4 et 5) et
le MCC binaire (Table 9).
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.dnn import DNN
from src.utils.commun import predire, metriques
from src.utils.config import load_config

ATTAQUES = ["FGSM", "BIM", "PGD", "DeepFool", "JSMA", "CW"]
ORDRE = ["Clean"] + ATTAQUES


def charger(cfg, fichier, device):
    chemin = Path(cfg.paths["checkpoints"]) / fichier
    if not chemin.exists():
        return None, None
    ck = torch.load(chemin, weights_only=False, map_location=device)
    model = DNN(input_dim=cfg.dataset["num_features"],
                hidden=tuple(cfg.model["hidden_layers"]),
                output_dim=cfg.dataset["num_classes"]).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, ck


def evaluer(model, cfg, device, labels, atk_dir):
    X = joblib.load(atk_dir / "X_test_clean.pkl")
    y = joblib.load(atk_dir / "y_test_clean.pkl")
    bs = cfg.evaluation["batch_size"]

    res = {"Clean": metriques(y, predire(model, X, device, bs), labels, "Clean")}
    for nom in ATTAQUES:
        X_adv = joblib.load(atk_dir / f"X_adv_test_{nom.lower()}.pkl")
        res[nom] = metriques(y, predire(model, X_adv, device, bs), labels, nom)
        del X_adv
    return res


def afficher(titre, res, info):
    print("=" * 100)
    print(f"{titre}  (epoch {info.get('epoch')}, "
          f"val_acc {info.get('val_acc', float('nan')):.4f})")
    print("=" * 100)
    print(f"{'Attaque':<10} {'Acc':>7} {'F1 mac':>7} {'F1 pond':>8} "
          f"{'F1 mic':>7} {'P mac':>7} {'R mac':>7} {'P pond':>7} {'R pond':>7} "
          f"{'MCC bin':>8} {'TPR':>6} {'FPR':>6} {'RecBEN':>7}")
    print("-" * 100)
    for a in ORDRE:
        m = res[a]
        print(f"{a:<10} {m['accuracy']:>7.4f} {m['f1_macro']:>7.4f} "
              f"{m['f1_weighted']:>8.4f} {m['f1_micro']:>7.4f} "
              f"{m['precision_macro']:>7.4f} {m['recall_macro']:>7.4f} "
              f"{m['precision_weighted']:>7.4f} {m['recall_weighted']:>7.4f} "
              f"{m['mcc_binaire']:>8.4f} {m['tpr']:>6.3f} {m['fpr']:>6.3f} "
              f"{m['recall_benign']:>7.4f}")
    print()


def main():
    print("=" * 100)
    print("Evaluation complete - reproduction fidele")
    print(f"Date : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 100 + "\n")

    parseur = argparse.ArgumentParser()
    parseur.add_argument("--suffixe", default="",
                         help="suffixe des checkpoints et des sorties")
    args = parseur.parse_args()

    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = list(range(cfg.dataset["num_classes"]))
    atk_dir = Path(cfg.paths["attacks"])

    y = joblib.load(atk_dir / "y_test_clean.pkl")
    frac = float((y == 0).mean())
    print(f"Test : {len(y):,} lignes, {int((y == 0).sum()):,} benin "
          f"({100*frac:.1f}%), plancher a {100*frac:.1f}%\n")

    # Le suffixe ne s'applique qu'a la variante : le baseline fidele reste
    # celui du papier, lr 0.01 et 30 epochs, il n'a pas de variante longue.
    n_ep = 150 if args.suffixe == "_ep150" else 100
    modeles = {
        "Baseline fidele (lr 0.01, 30 epochs)": "baseline.pth",
        f"Baseline lr reduit (lr 0.001, {n_ep} epochs)":
            f"baseline_varlr{args.suffixe}.pth",
    }

    tous, infos = {}, {}
    for titre, fichier in modeles.items():
        model, info = charger(cfg, fichier, device)
        if model is None:
            print(f"{fichier} introuvable\n")
            continue
        res = evaluer(model, cfg, device, labels, atk_dir)
        afficher(titre, res, info)
        tous[titre], infos[titre] = res, info

    lignes = []
    for titre, res in tous.items():
        court = "fidele" if "fidele" in titre else "varlr"
        for a in ORDRE:
            m = res[a]
            lignes.append({"modele": court, "titre": titre, "attaque": a,
                           **{k: v for k, v in m.items()
                              if k not in ("confusion_matrix", "attaque")}})

    fig_dir = Path(cfg.paths["figures"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    csv = fig_dir / f"evaluation_baselines{args.suffixe}.csv"
    pd.DataFrame(lignes).to_csv(csv, index=False, float_format="%.6f")
    print(f"CSV : {csv}")

    sortie = Path(cfg.paths["logs"]) / f"evaluation{args.suffixe}_{datetime.now():%Y%m%d_%H%M%S}.pkl"
    joblib.dump({"resultats": tous, "infos": infos,
                 "n_test": len(y), "frac_benin": frac}, sortie)
    print(f"Sauvegarde : {sortie}")


if __name__ == "__main__":
    main()
