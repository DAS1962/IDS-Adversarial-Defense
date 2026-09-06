"""
Evaluation du baseline sur un test set reequilibre 50/50 BENIGN / attaques.

Pourquoi ce script existe
--------------------------
Les six attaques evaluees sur le test complet donnent des accuracies entre
82 et 88%, alors que Awad et al. (Table 5) rapportent 36 a 81%. L'ecart
n'est pas necessairement un defaut d'implementation : le test set complet
contient 691 369 BENIGN sur 831 864 lignes, soit 83.1%. Une attaque
adversariale masque les intrusions mais ne degrade pas le trafic benin,
qui reste correctement classe. L'accuracy ne peut donc pas descendre sous
83.1% tant que BENIGN tient — c'est un plancher arithmetique, pas de la
robustesse.

L'article echantillonne differemment : sa section "Adversarial examples
generation" mentionne 20 000 echantillons de test dont la moitie de trafic
regulier. Si leur test est a ~50% BENIGN, leur plancher est a ~50%, ce qui
suffit a expliquer l'ecart.

Ce script teste cette hypothese sans rien regenerer : il reutilise les
X_adv_*.pkl deja produits par 08, selectionne un sous-ensemble d'indices
equilibre, et recalcule les metriques dessus. Si les accuracies tombent
dans la zone du papier, l'hypothese est confirmee.

Strategie d'echantillonnage
---------------------------
Toutes les lignes d'attaque sont conservees (140 495), et autant de BENIGN
sont tires au hasard parmi les 691 369 disponibles. Aucune classe d'attaque
n'est perdue : Heartbleed garde ses 4 exemples, SQL Injection ses 7.
Total : ~280 990 lignes, exactement 50% BENIGN.

Le sous-ensemble est defini par un vecteur d'indices applique de facon
identique a y_test et aux six X_adv, donc les six attaques sont evaluees
sur exactement les memes lignes.

Ce script est un complement d'analyse, pas une etape du pipeline : il ne
remplace pas 10_evaluate_and_plot_attacks.py, dont les chiffres sur le test
complet restent la reference.
"""

import sys
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.dnn import BaselineDNN
from src.utils.config import load_config, check_data_fingerprint


BATCH_SIZE = 512
BENIGN_CLASS = 0

ATTACK_FILES = [
    ("FGSM",     "X_adv_fgsm.pkl"),
    ("BIM",      "X_adv_bim.pkl"),
    ("PGD",      "X_adv_pgd.pkl"),
    ("DeepFool", "X_adv_deepfool.pkl"),
    ("JSMA",     "X_adv_jsma.pkl"),
    ("CW",       "X_adv_cw.pkl"),
]

# Table 5 de Awad et al. (2025), accuracy du detecteur sous attaque.
PAPER_RESULTS = {
    "Clean": 0.9811, "FGSM": 0.5450, "BIM": 0.4500, "PGD": 0.4600,
    "DeepFool": 0.5300, "JSMA": 0.8100, "CW": 0.3600,
}


def load_baseline_model(cfg, device, checkpoint_dir):
    """Charge le baseline avec les memes verifications bloquantes que 08 et 10."""
    print("Chargement du baseline...")
    model = BaselineDNN(
        input_dim=cfg.dataset["num_features"],
        hidden1=cfg.model["hidden_layers"][0],
        hidden2=cfg.model["hidden_layers"][1],
        output_dim=cfg.dataset["num_classes"],
    )
    checkpoint_path = checkpoint_dir / "baseline_best.pth"
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)

    if "val_acc" not in checkpoint:
        raise RuntimeError(
            f"{checkpoint_path} n'a pas de cle 'val_acc' : checkpoint du pipeline "
            f"precedent. Relancer scripts/06_train_baseline.py."
        )
    hash_attendu = cfg.baseline_fingerprint()
    if checkpoint.get("config_hash") != hash_attendu:
        raise RuntimeError(
            f"{checkpoint_path} entraine sous une autre configuration "
            f"({checkpoint.get('config_hash')!r} != {hash_attendu!r}). "
            f"Relancer scripts/06_train_baseline.py."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()
    print(f"  Epoch : {checkpoint['epoch']}  |  Val acc : {checkpoint['val_acc']:.4f}\n")
    return model


def build_balanced_indices(y_test, seed):
    """
    Indices d'un sous-ensemble a 50% BENIGN.

    Toutes les lignes d'attaque sont gardees ; autant de BENIGN sont tires
    au hasard. Les indices sont tries pour que l'ordre des lignes reste
    celui du test d'origine, ce qui n'a pas d'effet sur les metriques mais
    rend le sous-ensemble reproductible et inspectable.
    """
    idx_attaque = np.flatnonzero(y_test != BENIGN_CLASS)
    idx_benign = np.flatnonzero(y_test == BENIGN_CLASS)

    n_attaque = len(idx_attaque)
    if n_attaque > len(idx_benign):
        raise ValueError(
            f"{n_attaque} lignes d'attaque pour seulement {len(idx_benign)} BENIGN : "
            f"un equilibre 50/50 en gardant toutes les attaques est impossible."
        )

    rng = np.random.default_rng(seed)
    idx_benign_tires = rng.choice(idx_benign, size=n_attaque, replace=False)
    indices = np.sort(np.concatenate([idx_attaque, idx_benign_tires]))

    print("Construction du sous-ensemble equilibre :")
    print(f"  Lignes d'attaque conservees : {n_attaque:,} (toutes)")
    print(f"  BENIGN tires au hasard      : {len(idx_benign_tires):,} "
          f"sur {len(idx_benign):,} disponibles")
    print(f"  Total                       : {len(indices):,} lignes")
    print(f"  Proportion BENIGN           : {(y_test[indices] == BENIGN_CLASS).mean():.4f}\n")
    return indices


def predict(model, X, device):
    model.eval()
    n = len(X)
    preds = np.zeros(n, dtype=np.int64)
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            end = min(i + BATCH_SIZE, n)
            xb = torch.tensor(X[i:end], dtype=torch.float32).to(device)
            preds[i:end] = model(xb).argmax(dim=1).cpu().numpy()
    return preds


def compute_metrics(y_true, y_pred, name, all_labels):
    """Moyennes calculees sur les 15 classes, pas seulement celles presentes."""
    return {
        "attack": name,
        "n_samples": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, labels=all_labels, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, labels=all_labels, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, labels=all_labels, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, labels=all_labels, average="weighted", zero_division=0),
        "recall_benign": recall_score(y_true, y_pred, labels=[BENIGN_CLASS], average="macro", zero_division=0),
    }


def plot_comparison(df_full, df_bal, out_path):
    """
    Trois series : test complet, test equilibre, papier. Avec les deux
    planchers BENIGN correspondants, qui sont l'objet meme de la figure.
    """
    attaques = [r for r in df_bal["attack"] if r != "Clean"]
    labels = ["Clean"] + attaques
    x = np.arange(len(labels))

    serie_full = [df_full.loc[df_full["attack"] == n, "accuracy"].iloc[0] for n in labels]
    serie_bal = [df_bal.loc[df_bal["attack"] == n, "accuracy"].iloc[0] for n in labels]
    serie_paper = [PAPER_RESULTS.get(n, np.nan) for n in labels]

    width = 0.27
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - width, serie_full, width, label="Test complet (83.1% BENIGN)",
           color="#27ae60", edgecolor="black")
    ax.bar(x, serie_bal, width, label="Test equilibre (50% BENIGN)",
           color="#2980b9", edgecolor="black")
    ax.bar(x + width, serie_paper, width, label="Papier (Awad 2025, Table 5)",
           color="#8e44ad", edgecolor="black")

    ax.axhline(0.831, color="#27ae60", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(len(labels) - 0.5, 0.838, "plancher test complet",
            fontsize=8, color="#27ae60", ha="right")
    ax.axhline(0.50, color="#2980b9", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(len(labels) - 0.5, 0.507, "plancher test equilibre",
            fontsize=8, color="#2980b9", ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Effet de la composition du test set sur l'accuracy sous attaque\n"
                 "(memes exemples adversariaux, meme modele, seul le sous-ensemble evalue change)",
                 fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Sauvegarde : {out_path.name}")


def main():
    print("Evaluation sur test set reequilibre 50/50")
    print(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    cfg = load_config()
    data_dir = Path(cfg.paths["data_processed"])
    checkpoint_dir = Path(cfg.paths["checkpoints"])
    attacks_dir = Path(cfg.paths["attacks"])
    figures_dir = Path(cfg.paths["figures"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    check_data_fingerprint(cfg, data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}\n")

    model = load_baseline_model(cfg, device, checkpoint_dir)

    print("Chargement du test set...")
    X_test = pd.read_pickle(data_dir / "X_test.pkl")
    y_test = joblib.load(data_dir / "y_test.pkl")
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values.astype(np.float32)
    y_test = np.asarray(y_test)
    print(f"  X_test : {X_test.shape}\n")

    indices = build_balanced_indices(y_test, cfg.seed)
    y_bal = y_test[indices]
    all_labels = list(range(cfg.dataset["num_classes"]))

    print("Distribution du sous-ensemble equilibre :")
    label_encoder = joblib.load(data_dir / "label_encoder.pkl")
    for classe in all_labels:
        n = int((y_bal == classe).sum())
        print(f"  Classe {classe:2d} ({str(label_encoder.classes_[classe])[:28]:28s}) : {n:>7,}")
    print()

    lignes_full, lignes_bal = [], []

    print("Evaluation sur donnees propres...")
    pred_clean_full = predict(model, X_test, device)
    lignes_full.append(compute_metrics(y_test, pred_clean_full, "Clean", all_labels))
    lignes_bal.append(compute_metrics(y_bal, pred_clean_full[indices], "Clean", all_labels))
    print(f"  Test complet   : acc={lignes_full[0]['accuracy']:.4f}  f1_macro={lignes_full[0]['f1_macro']:.4f}")
    print(f"  Test equilibre : acc={lignes_bal[0]['accuracy']:.4f}  f1_macro={lignes_bal[0]['f1_macro']:.4f}\n")
    del pred_clean_full

    print("Evaluation des attaques :\n")
    for name, x_file in ATTACK_FILES:
        path = attacks_dir / x_file
        if not path.exists():
            print(f"  [SKIP] {name} : {x_file} introuvable")
            continue
        X_adv = joblib.load(path)
        if len(X_adv) != len(y_test):
            print(f"  [SKIP] {name} : {len(X_adv):,} lignes vs {len(y_test):,} attendues "
                  f"(genere sur un autre perimetre)")
            del X_adv
            continue

        preds = predict(model, X_adv, device)
        m_full = compute_metrics(y_test, preds, name, all_labels)
        m_bal = compute_metrics(y_bal, preds[indices], name, all_labels)
        lignes_full.append(m_full)
        lignes_bal.append(m_bal)

        print(f"  {name}")
        print(f"    Test complet   : acc={m_full['accuracy']:.4f}  f1_macro={m_full['f1_macro']:.4f}  "
              f"recall_BENIGN={m_full['recall_benign']:.4f}")
        print(f"    Test equilibre : acc={m_bal['accuracy']:.4f}  f1_macro={m_bal['f1_macro']:.4f}  "
              f"recall_BENIGN={m_bal['recall_benign']:.4f}")
        print(f"    Papier         : acc={PAPER_RESULTS.get(name, float('nan')):.4f}")
        del X_adv, preds

    if len(lignes_bal) <= 1:
        print("\nAucune attaque evaluee. Arret.")
        return

    df_full = pd.DataFrame(lignes_full)
    df_bal = pd.DataFrame(lignes_bal)

    recap = pd.DataFrame({
        "attack": df_bal["attack"],
        "acc_test_complet": df_full["accuracy"].values,
        "acc_test_equilibre": df_bal["accuracy"].values,
        "acc_papier": [PAPER_RESULTS.get(n, np.nan) for n in df_bal["attack"]],
        "f1_macro_complet": df_full["f1_macro"].values,
        "f1_macro_equilibre": df_bal["f1_macro"].values,
        "recall_benign_equilibre": df_bal["recall_benign"].values,
        "n_equilibre": df_bal["n_samples"].values,
    })
    recap["ecart_papier_complet"] = recap["acc_test_complet"] - recap["acc_papier"]
    recap["ecart_papier_equilibre"] = recap["acc_test_equilibre"] - recap["acc_papier"]

    print("\n" + "=" * 110)
    print("Recapitulatif")
    print("=" * 110)
    print(recap.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("=" * 110)

    ecart_avant = recap.loc[recap["attack"] != "Clean", "ecart_papier_complet"].abs().mean()
    ecart_apres = recap.loc[recap["attack"] != "Clean", "ecart_papier_equilibre"].abs().mean()
    print("\nEcart absolu moyen avec le papier (attaques seulement) :")
    print(f"  Test complet   : {ecart_avant:.4f} ({ecart_avant*100:.1f} points)")
    print(f"  Test equilibre : {ecart_apres:.4f} ({ecart_apres*100:.1f} points)")
    if ecart_apres < ecart_avant:
        print(f"  -> Rapprochement de {(ecart_avant - ecart_apres)*100:.1f} points : "
              f"la composition du test explique une part de l'ecart.")
    else:
        print("  -> Pas de rapprochement : la composition du test n'explique pas l'ecart, "
              "chercher ailleurs (force des perturbations, robustesse du baseline).")

    csv_path = figures_dir / "attacks_balanced_comparison.csv"
    recap.to_csv(csv_path, index=False)
    print(f"\nCSV : {csv_path}")

    print("\nGeneration de la figure...")
    plot_comparison(df_full, df_bal, figures_dir / "attacks_balanced_comparison.png")

    print("\nTermine.")


if __name__ == "__main__":
    main()
