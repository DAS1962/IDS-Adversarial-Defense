"""
Évaluation du baseline sur les 6 attaques adversariales (full test set),
et génération des figures comparatives complètes.

Charge automatiquement tous les fichiers X_adv_*.pkl présents dans
results/attacks/. Recalcule les métriques sans dépendre de fichiers
précalculés, donc reste valide même si 08 a tourné en plusieurs fois.

Génère :
  - CSV récapitulatif : results/figures/attacks_summary.csv
  - Bar plots individuels : accuracy, precision, recall, F1 (macro et weighted)
  - Chute d'accuracy vs baseline
  - Vues combinées 4 métriques (macro et weighted séparées)
  - Comparaisons macro vs weighted
  - Heatmaps F1, precision, recall par classe × attaque
  - Comparaison avec le papier, avec le plancher BENIGN
  - Matrices de confusion (une par attaque)
"""

import sys
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.dnn import BaselineDNN
from src.utils.config import load_config, check_data_fingerprint


BATCH_SIZE = 512

ATTACK_FILES = [
    ("FGSM",     "X_adv_fgsm.pkl"),
    ("BIM",      "X_adv_bim.pkl"),
    ("PGD",      "X_adv_pgd.pkl"),
    ("DeepFool", "X_adv_deepfool.pkl"),
    ("JSMA",     "X_adv_jsma.pkl"),
    ("CW",       "X_adv_cw.pkl"),
]

# Accuracy du detecteur DNN sous attaque, Table 5 de Awad et al. (2025),
# CIC-IDS 2017. Le texte de l'article confirme ces valeurs : "the accuracy
# drops to 54% under FGSM, 53% under DeepFool, 81% under JSMA, 45% under
# BIM, and 36% under C&W".
#
# ATTENTION : la version precedente de ce dictionnaire contenait des valeurs
# fausses (FGSM 0.859, BIM 0.810, PGD 0.8025, JSMA 0.482) issues d'un melange
# entre la Table 5 (attaques sur le baseline) et la Table 7 (performances des
# defenses individuelles : Label Smoothing 85.9, Adversarial Training 80.25,
# Gaussian Augmentation 79.8, DAE 84.8). L'erreur inversait notamment la
# lecture de JSMA, dont le vrai chiffre de reference est 81% et non 48.2%.
PAPER_RESULTS = {
    "Clean":    0.9811,
    "FGSM":     0.5450,
    "BIM":      0.4500,
    "PGD":      0.4600,
    "DeepFool": 0.5300,
    "JSMA":     0.8100,
    "CW":       0.3600,
}


def nettoyer_nom(nom):
    """Retire les caracteres de remplacement issus de l'encodage des CSV."""
    for mauvais in ("\ufffd", "\x96", "\u2013", "\u2014"):
        nom = nom.replace(mauvais, "-")
    return " ".join(nom.split())


def load_baseline_model(cfg, device, checkpoint_dir):
    """
    Charge le baseline, avec les memes verifications bloquantes que 08.

    Un checkpoint sans 'val_acc' vient du pipeline precedent (selection sur
    le test set) ; un config_hash different signifie qu'il a ete entraine
    sous une autre configuration. Dans les deux cas, l'utiliser produirait
    des figures fausses sous une etiquette juste.
    """
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
            f"precedent, incompatible. Relancer scripts/06_train_baseline.py."
        )

    hash_attendu = cfg.baseline_fingerprint()
    hash_checkpoint = checkpoint.get("config_hash")
    if hash_checkpoint != hash_attendu:
        raise RuntimeError(
            f"{checkpoint_path} entraine sous une autre configuration "
            f"(config_hash={hash_checkpoint!r}, attendu {hash_attendu!r}). "
            f"Relancer scripts/06_train_baseline.py."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()
    print(f"  Epoch : {checkpoint['epoch']}  |  Val acc : {checkpoint['val_acc']:.4f}")
    print(f"  config_hash : {hash_checkpoint} (verifie)\n")
    return model


def load_test_data(data_dir):
    print("Chargement du test set propre...")
    X_test = pd.read_pickle(data_dir / "X_test.pkl")
    y_test = joblib.load(data_dir / "y_test.pkl")
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values.astype(np.float32)
    if isinstance(y_test, (pd.Series, pd.DataFrame)):
        y_test = y_test.values
    print(f"  X_test : {X_test.shape}\n")
    return X_test, np.asarray(y_test)


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
    """
    Toutes les moyennes sont calculees sur les 15 classes (labels=all_labels),
    pas seulement sur celles presentes dans y_true. Sans ce parametre, sklearn
    restreint la moyenne macro aux classes vues, ce qui produit deux chiffres
    differents pour la meme grandeur selon l'appel.
    """
    return {
        "attack": name,
        "n_samples": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, labels=all_labels, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, labels=all_labels, average="weighted", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, labels=all_labels, average="macro", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, labels=all_labels, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, labels=all_labels, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, labels=all_labels, average="weighted", zero_division=0),
        "f1_per_class": f1_score(y_true, y_pred, average=None, labels=all_labels, zero_division=0),
        "precision_per_class": precision_score(y_true, y_pred, average=None, labels=all_labels, zero_division=0),
        "recall_per_class": recall_score(y_true, y_pred, average=None, labels=all_labels, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=all_labels),
    }


def plot_single_metric_bar(results, clean_metrics, metric_key, title, out_path,
                           benign_floor=None):
    names = ["Clean"] + [r["attack"] for r in results]
    values = [clean_metrics[metric_key]] + [r[metric_key] for r in results]
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ["#2ecc71"] + ["#e74c3c"] * len(results)
    bars = ax.bar(names, values, color=colors, edgecolor="black")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")

    if benign_floor is not None:
        ax.axhline(benign_floor, color="#c0392b", linestyle="--", linewidth=1.5,
                   label=f"Plancher BENIGN ({benign_floor:.1%})")
        ax.legend(loc="lower right", fontsize=9)

    ax.set_ylabel(title, fontsize=12)
    ax.set_title(f"Baseline : {title} sur donnees propres et sous attaques", fontsize=13)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Sauvegarde : {out_path.name}")


def plot_all_metrics_grouped(results, clean_metrics, average, out_path):
    names = ["Clean"] + [r["attack"] for r in results]
    metrics = ["accuracy", f"precision_{average}", f"recall_{average}", f"f1_{average}"]
    metric_labels = ["Accuracy", f"Precision {average}", f"Recall {average}", f"F1 {average}"]

    data = np.array([
        [clean_metrics[m] for m in metrics]
    ] + [
        [r[m] for m in metrics] for r in results
    ])

    x = np.arange(len(names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["#2ecc71", "#3498db", "#f39c12", "#9b59b6"]
    for i, (lab, col) in enumerate(zip(metric_labels, colors)):
        ax.bar(x + (i - 1.5) * width, data[:, i], width, label=lab, color=col, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Comparaison des 4 metriques ({average}) par attaque", fontsize=13)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Sauvegarde : {out_path.name}")


def plot_macro_vs_weighted(results, metric_name, out_path):
    names = [r["attack"] for r in results]
    macro = [r[f"{metric_name}_macro"] for r in results]
    weighted = [r[f"{metric_name}_weighted"] for r in results]
    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width / 2, macro, width, label=f"{metric_name} macro", color="#3498db", edgecolor="black")
    ax.bar(x + width / 2, weighted, width, label=f"{metric_name} weighted", color="#f39c12", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel(metric_name.capitalize(), fontsize=12)
    ax.set_title(f"{metric_name.capitalize()} macro vs weighted par attaque\n"
                 "(gros ecart = classes minoritaires effondrees)", fontsize=13)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Sauvegarde : {out_path.name}")


def plot_accuracy_drop(results, baseline_acc, out_path):
    names = [r["attack"] for r in results]
    drops = [(baseline_acc - r["accuracy"]) * 100 for r in results]
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(names, drops, color="#c0392b", edgecolor="black")
    for bar, d in zip(bars, drops):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"-{d:.1f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Chute d'accuracy (points de %)", fontsize=12)
    ax.set_title("Vulnerabilite du baseline : chute d'accuracy par attaque\n"
                 "(bornee par la proportion de BENIGN dans le test)", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Sauvegarde : {out_path.name}")


def plot_metric_heatmap(results, class_names, metric_key, title, out_path):
    data = np.array([r[metric_key] for r in results])
    names = [r["attack"] for r in results]
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        data, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
        xticklabels=class_names, yticklabels=names, cbar_kws={"label": title},
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel("Classe", fontsize=12)
    ax.set_ylabel("Attaque", fontsize=12)
    ax.set_title(f"{title} par classe et par attaque", fontsize=13)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Sauvegarde : {out_path.name}")


def plot_paper_comparison(results, our_baseline_acc, out_path, benign_floor=None):
    """
    Comparaison des accuracies avec la Table 5 de l'article.

    benign_floor : proportion de BENIGN dans le test evalue, tracee en ligne
    horizontale. L'accuracy ne peut pas descendre en dessous tant que les
    attaques ne degradent pas le trafic benin : une attaque qui masque 100%
    des intrusions atteint exactement ce plancher. Sans cette reference,
    l'ecart avec le papier se lit comme une faiblesse de l'implementation
    alors qu'il vient d'une composition de test differente.
    """
    attack_names = [r["attack"] for r in results]
    ours = [our_baseline_acc] + [r["accuracy"] for r in results]
    theirs = [PAPER_RESULTS["Clean"]] + [PAPER_RESULTS.get(n, np.nan) for n in attack_names]
    labels = ["Clean"] + attack_names
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, ours, width, label="Notre implementation",
           color="#27ae60", edgecolor="black")
    ax.bar(x + width / 2, theirs, width, label="Papier (Awad 2025, Table 5)",
           color="#8e44ad", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    if benign_floor is not None:
        ax.axhline(benign_floor, color="#c0392b", linestyle="--", linewidth=1.5,
                   label=f"Plancher BENIGN ({benign_floor:.1%} du test)")
        ax.text(len(labels) - 0.4, benign_floor + 0.015,
                "evasion totale = accuracy a ce niveau",
                fontsize=8, color="#c0392b", ha="right")

    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Comparaison avec le papier de reference\n"
                 "(accuracies non directement comparables : compositions de test differentes)",
                 fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Sauvegarde : {out_path.name}")


def plot_confusion_matrix(cm, class_names, name, out_path):
    supports = cm.sum(axis=1)
    row_labels = [f"{n} (n={s:,})" for n, s in zip(class_names, supports)]
    fig, ax = plt.subplots(figsize=(12, 9))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
        xticklabels=class_names, yticklabels=row_labels,
        cbar_kws={"label": "Proportion"}, ax=ax, linewidths=0.3,
    )
    ax.set_xlabel("Prediction", fontsize=12)
    ax.set_ylabel("Vraie classe", fontsize=12)
    ax.set_title(f"Matrice de confusion normalisee - {name}", fontsize=13)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Sauvegarde : {out_path.name}")


def main():
    print("Evaluation du baseline sur attaques adversariales")
    print(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    cfg = load_config()
    print(cfg.resume())
    print()

    data_dir = Path(cfg.paths["data_processed"])
    checkpoint_dir = Path(cfg.paths["checkpoints"])
    attacks_dir = Path(cfg.paths["attacks"])
    figures_dir = Path(cfg.paths["figures"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    check_data_fingerprint(cfg, data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}\n")

    model = load_baseline_model(cfg, device, checkpoint_dir)
    X_test, y_test = load_test_data(data_dir)

    label_encoder = joblib.load(data_dir / "label_encoder.pkl")
    class_names = [nettoyer_nom(str(c)) for c in label_encoder.classes_]
    all_labels = list(range(cfg.dataset["num_classes"]))

    benign_floor = float((y_test == 0).mean())
    print(f"Proportion de BENIGN dans le test : {benign_floor:.4f}")
    print("  (plancher theorique de l'accuracy sous evasion totale)\n")

    print("Recalcul des metriques du baseline sur donnees propres...")
    y_pred_clean = predict(model, X_test, device)
    clean_metrics = compute_metrics(y_test, y_pred_clean, "Clean", all_labels)
    print(f"  Accuracy    : {clean_metrics['accuracy']:.4f}")
    print(f"  Prec macro  : {clean_metrics['precision_macro']:.4f}")
    print(f"  Rec  macro  : {clean_metrics['recall_macro']:.4f}")
    print(f"  F1   macro  : {clean_metrics['f1_macro']:.4f}\n")

    results = []
    print("Evaluation des attaques :\n")
    for name, x_file in ATTACK_FILES:
        x_path = attacks_dir / x_file
        if not x_path.exists():
            print(f"  [SKIP] {name} : {x_file} introuvable")
            continue
        print(f"  -> {name} ({x_file})")
        X_adv = joblib.load(x_path)
        if len(X_adv) != len(y_test):
            print(f"     [WARN] taille {len(X_adv)} vs {len(y_test)} : ce fichier a ete")
            print(f"            genere sur un autre perimetre (evaluation.scope), skip")
            del X_adv
            continue
        y_pred = predict(model, X_adv, device)
        m = compute_metrics(y_test, y_pred, name, all_labels)
        print(f"     Acc: {m['accuracy']:.4f} | Prec: {m['precision_macro']:.4f}/{m['precision_weighted']:.4f} | "
              f"Rec: {m['recall_macro']:.4f}/{m['recall_weighted']:.4f} | "
              f"F1: {m['f1_macro']:.4f}/{m['f1_weighted']:.4f}")
        results.append(m)
        del X_adv

    if not results:
        print("\nAucune attaque evaluee. Arret.")
        return

    print("\nSauvegarde du CSV recapitulatif...")
    summary_rows = [{
        "attack": "Clean",
        "n_samples": clean_metrics["n_samples"],
        "accuracy": clean_metrics["accuracy"],
        "precision_macro": clean_metrics["precision_macro"],
        "precision_weighted": clean_metrics["precision_weighted"],
        "recall_macro": clean_metrics["recall_macro"],
        "recall_weighted": clean_metrics["recall_weighted"],
        "f1_macro": clean_metrics["f1_macro"],
        "f1_weighted": clean_metrics["f1_weighted"],
        "accuracy_drop_vs_clean": 0.0,
        "paper_accuracy": PAPER_RESULTS.get("Clean", np.nan),
    }]
    for r in results:
        summary_rows.append({
            "attack": r["attack"],
            "n_samples": r["n_samples"],
            "accuracy": r["accuracy"],
            "precision_macro": r["precision_macro"],
            "precision_weighted": r["precision_weighted"],
            "recall_macro": r["recall_macro"],
            "recall_weighted": r["recall_weighted"],
            "f1_macro": r["f1_macro"],
            "f1_weighted": r["f1_weighted"],
            "accuracy_drop_vs_clean": clean_metrics["accuracy"] - r["accuracy"],
            "paper_accuracy": PAPER_RESULTS.get(r["attack"], np.nan),
        })
    df = pd.DataFrame(summary_rows)
    csv_path = figures_dir / "attacks_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"  {csv_path}")
    print()
    print(df.to_string(index=False))
    print()

    print("Generation des figures...\n")

    print("  [Bar plots individuels]")
    plot_single_metric_bar(results, clean_metrics, "accuracy", "Accuracy",
                           figures_dir / "attacks_accuracy_bar.png",
                           benign_floor=benign_floor)
    plot_single_metric_bar(results, clean_metrics, "precision_macro", "Precision (macro)",
                           figures_dir / "attacks_precision_macro_bar.png")
    plot_single_metric_bar(results, clean_metrics, "precision_weighted", "Precision (weighted)",
                           figures_dir / "attacks_precision_weighted_bar.png")
    plot_single_metric_bar(results, clean_metrics, "recall_macro", "Recall (macro)",
                           figures_dir / "attacks_recall_macro_bar.png")
    plot_single_metric_bar(results, clean_metrics, "recall_weighted", "Recall (weighted)",
                           figures_dir / "attacks_recall_weighted_bar.png")
    plot_single_metric_bar(results, clean_metrics, "f1_macro", "F1 (macro)",
                           figures_dir / "attacks_f1_macro_bar.png")
    plot_single_metric_bar(results, clean_metrics, "f1_weighted", "F1 (weighted)",
                           figures_dir / "attacks_f1_weighted_bar.png")

    print("\n  [Vues combinees 4 metriques]")
    plot_all_metrics_grouped(results, clean_metrics, "macro",
                             figures_dir / "attacks_all_metrics_macro.png")
    plot_all_metrics_grouped(results, clean_metrics, "weighted",
                             figures_dir / "attacks_all_metrics_weighted.png")

    print("\n  [Macro vs weighted]")
    plot_macro_vs_weighted(results, "precision",
                           figures_dir / "attacks_precision_macro_vs_weighted.png")
    plot_macro_vs_weighted(results, "recall",
                           figures_dir / "attacks_recall_macro_vs_weighted.png")
    plot_macro_vs_weighted(results, "f1",
                           figures_dir / "attacks_f1_macro_vs_weighted.png")

    print("\n  [Chute d'accuracy et comparaison papier]")
    plot_accuracy_drop(results, clean_metrics["accuracy"],
                       figures_dir / "attacks_accuracy_drop.png")
    plot_paper_comparison(results, clean_metrics["accuracy"],
                          figures_dir / "attacks_paper_comparison.png",
                          benign_floor=benign_floor)

    print("\n  [Heatmaps par classe]")
    plot_metric_heatmap(results, class_names, "f1_per_class", "F1 score",
                        figures_dir / "attacks_f1_heatmap.png")
    plot_metric_heatmap(results, class_names, "precision_per_class", "Precision",
                        figures_dir / "attacks_precision_heatmap.png")
    plot_metric_heatmap(results, class_names, "recall_per_class", "Recall",
                        figures_dir / "attacks_recall_heatmap.png")

    print("\n  [Matrices de confusion]")
    plot_confusion_matrix(clean_metrics["confusion_matrix"], class_names,
                          "Clean (baseline)",
                          figures_dir / "cm_clean.png")
    for r in results:
        safe = r["attack"].lower().replace("&", "").replace(" ", "_")
        plot_confusion_matrix(r["confusion_matrix"], class_names, r["attack"],
                              figures_dir / f"cm_{safe}.png")

    print(f"\nTermine. Figures dans : {figures_dir}/")


if __name__ == "__main__":
    main()
