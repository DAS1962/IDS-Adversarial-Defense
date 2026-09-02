"""
Évaluation du baseline v4 sur les 6 attaques adversariales (full test set),
et génération des figures comparatives complètes (accuracy, precision, recall, F1).

Charge automatiquement tous les fichiers X_adv_*.pkl présents dans
results/attacks/. Recalcule les métriques sans dépendre de fichiers précalculés.

Génère :
  - CSV récapitulatif : results/figures/attacks_summary.csv
  - Bar plots individuels : accuracy, precision (macro/weighted), recall (macro/weighted), F1 (macro/weighted)
  - Chute d'accuracy vs baseline
  - Vues combinées 4 métriques (macro et weighted séparées)
  - Comparaisons macro vs weighted pour precision, recall, F1
  - Heatmaps F1, precision, recall par classe × attaque
  - Comparaison avec le papier
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


DATA_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("results/checkpoints")
ATTACKS_DIR = Path("results/attacks")
FIGURES_DIR = Path("results/figures")

BATCH_SIZE = 512
NUM_CLASSES = 15
INPUT_DIM = 58

ATTACK_FILES = [
    ("FGSM",     "X_adv_fgsm.pkl"),
    ("BIM",      "X_adv_bim.pkl"),
    ("PGD",      "X_adv_pgd.pkl"),
    ("DeepFool", "X_adv_deepfool.pkl"),
    ("JSMA",     "X_adv_jsma.pkl"),
    ("CW",       "X_adv_cw.pkl"),
]

PAPER_RESULTS = {
    "Clean":    0.9811,
    "FGSM":     0.8590,
    "BIM":      0.8100,
    "PGD":      0.8025,
    "DeepFool": 0.5340,
    "JSMA":     0.4820,
    "CW":       0.3600,
}


def load_baseline_model(device):
    print("Chargement du baseline v4...")
    model = BaselineDNN(
        input_dim=INPUT_DIM, hidden1=512, hidden2=256, output_dim=NUM_CLASSES
    )
    checkpoint = torch.load(
        CHECKPOINT_DIR / "baseline_best.pth",
        weights_only=False,
        map_location=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()
    print(f"  Epoch : {checkpoint['epoch']}  |  Test acc : {checkpoint['test_acc']:.4f}\n")
    return model, float(checkpoint["test_acc"])


def load_test_data():
    print("Chargement du test set propre...")
    X_test = pd.read_pickle(DATA_DIR / "X_test.pkl")
    y_test = pd.read_pickle(DATA_DIR / "y_test.pkl")
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values.astype(np.float32)
    if isinstance(y_test, pd.Series):
        y_test = y_test.values
    print(f"  X_test : {X_test.shape}\n")
    return X_test, y_test


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
    return {
        "attack": name,
        "n_samples": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_per_class": f1_score(y_true, y_pred, average=None, labels=all_labels, zero_division=0),
        "precision_per_class": precision_score(y_true, y_pred, average=None, labels=all_labels, zero_division=0),
        "recall_per_class": recall_score(y_true, y_pred, average=None, labels=all_labels, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=all_labels),
    }


def plot_single_metric_bar(results, clean_metrics, metric_key, title, out_path):
    names = ["Clean"] + [r["attack"] for r in results]
    values = [clean_metrics[metric_key]] + [r[metric_key] for r in results]
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ["#2ecc71"] + ["#e74c3c"] * len(results)
    bars = ax.bar(names, values, color=colors, edgecolor="black")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel(title, fontsize=12)
    ax.set_title(f"Baseline v4 : {title} sur données propres et sous attaques", fontsize=13)
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
    ax.set_title(f"Comparaison des 4 métriques ({average}) par attaque", fontsize=13)
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
                 "(gros écart = classes minoritaires effondrées)", fontsize=13)
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
    ax.set_title("Vulnérabilité du baseline v4 : chute d'accuracy par attaque", fontsize=13)
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


def plot_paper_comparison(results, our_baseline_acc, out_path):
    attack_names = [r["attack"] for r in results]
    ours = [our_baseline_acc] + [r["accuracy"] for r in results]
    theirs = [PAPER_RESULTS["Clean"]] + [PAPER_RESULTS.get(n, np.nan) for n in attack_names]
    labels = ["Clean"] + attack_names
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, ours, width, label="Notre implémentation",
           color="#27ae60", edgecolor="black")
    ax.bar(x + width / 2, theirs, width, label="Papier (Awad 2025)",
           color="#8e44ad", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Comparaison avec le papier de référence", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Sauvegarde : {out_path.name}")


def plot_confusion_matrix(cm, class_names, name, out_path):
    fig, ax = plt.subplots(figsize=(11, 9))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={"label": "Proportion"}, ax=ax, linewidths=0.3,
    )
    ax.set_xlabel("Prédiction", fontsize=12)
    ax.set_ylabel("Vraie classe", fontsize=12)
    ax.set_title(f"Matrice de confusion normalisée — {name}", fontsize=13)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Sauvegarde : {out_path.name}")


def main():
    print(f"Évaluation baseline v4 sur attaques adversariales")
    print(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}\n")

    model, baseline_acc = load_baseline_model(device)
    X_test, y_test = load_test_data()

    label_encoder = joblib.load(DATA_DIR / "label_encoder.pkl")
    class_names = [str(c) for c in label_encoder.classes_]
    all_labels = list(range(NUM_CLASSES))

    print("Recalcul métriques baseline sur données propres...")
    y_pred_clean = predict(model, X_test, device)
    clean_metrics = compute_metrics(y_test, y_pred_clean, "Clean", all_labels)
    print(f"  Accuracy    : {clean_metrics['accuracy']:.4f}")
    print(f"  Prec macro  : {clean_metrics['precision_macro']:.4f}")
    print(f"  Rec  macro  : {clean_metrics['recall_macro']:.4f}")
    print(f"  F1   macro  : {clean_metrics['f1_macro']:.4f}\n")

    results = []
    print("Évaluation des attaques :\n")
    for name, x_file in ATTACK_FILES:
        x_path = ATTACKS_DIR / x_file
        if not x_path.exists():
            print(f"  [SKIP] {name} : {x_file} introuvable")
            continue
        print(f"  -> {name} ({x_file})")
        X_adv = joblib.load(x_path)
        if len(X_adv) != len(y_test):
            print(f"     [WARN] taille mismatch {len(X_adv)} vs {len(y_test)}, skip")
            continue
        y_pred = predict(model, X_adv, device)
        m = compute_metrics(y_test, y_pred, name, all_labels)
        print(f"     Acc: {m['accuracy']:.4f} | Prec: {m['precision_macro']:.4f}/{m['precision_weighted']:.4f} | "
              f"Rec: {m['recall_macro']:.4f}/{m['recall_weighted']:.4f} | "
              f"F1: {m['f1_macro']:.4f}/{m['f1_weighted']:.4f}")
        results.append(m)
        del X_adv

    if not results:
        print("\nAucune attaque évaluée. Arrêt.")
        return

    print("\nSauvegarde du CSV récapitulatif...")
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
        })
    df = pd.DataFrame(summary_rows)
    csv_path = FIGURES_DIR / "attacks_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"  {csv_path}")
    print()
    print(df.to_string(index=False))
    print()

    print("Génération des figures...\n")

    print("  [Bar plots individuels]")
    plot_single_metric_bar(results, clean_metrics, "accuracy",
                           "Accuracy",
                           FIGURES_DIR / "attacks_accuracy_bar.png")
    plot_single_metric_bar(results, clean_metrics, "precision_macro",
                           "Precision (macro)",
                           FIGURES_DIR / "attacks_precision_macro_bar.png")
    plot_single_metric_bar(results, clean_metrics, "precision_weighted",
                           "Precision (weighted)",
                           FIGURES_DIR / "attacks_precision_weighted_bar.png")
    plot_single_metric_bar(results, clean_metrics, "recall_macro",
                           "Recall (macro)",
                           FIGURES_DIR / "attacks_recall_macro_bar.png")
    plot_single_metric_bar(results, clean_metrics, "recall_weighted",
                           "Recall (weighted)",
                           FIGURES_DIR / "attacks_recall_weighted_bar.png")
    plot_single_metric_bar(results, clean_metrics, "f1_macro",
                           "F1 (macro)",
                           FIGURES_DIR / "attacks_f1_macro_bar.png")
    plot_single_metric_bar(results, clean_metrics, "f1_weighted",
                           "F1 (weighted)",
                           FIGURES_DIR / "attacks_f1_weighted_bar.png")

    print("\n  [Vues combinees 4 metriques]")
    plot_all_metrics_grouped(results, clean_metrics, "macro",
                             FIGURES_DIR / "attacks_all_metrics_macro.png")
    plot_all_metrics_grouped(results, clean_metrics, "weighted",
                             FIGURES_DIR / "attacks_all_metrics_weighted.png")

    print("\n  [Macro vs weighted]")
    plot_macro_vs_weighted(results, "precision",
                           FIGURES_DIR / "attacks_precision_macro_vs_weighted.png")
    plot_macro_vs_weighted(results, "recall",
                           FIGURES_DIR / "attacks_recall_macro_vs_weighted.png")
    plot_macro_vs_weighted(results, "f1",
                           FIGURES_DIR / "attacks_f1_macro_vs_weighted.png")

    print("\n  [Chute d'accuracy et comparaison papier]")
    plot_accuracy_drop(results, clean_metrics["accuracy"],
                       FIGURES_DIR / "attacks_accuracy_drop.png")
    plot_paper_comparison(results, clean_metrics["accuracy"],
                          FIGURES_DIR / "attacks_paper_comparison.png")

    print("\n  [Heatmaps par classe]")
    plot_metric_heatmap(results, class_names, "f1_per_class", "F1 score",
                        FIGURES_DIR / "attacks_f1_heatmap.png")
    plot_metric_heatmap(results, class_names, "precision_per_class", "Precision",
                        FIGURES_DIR / "attacks_precision_heatmap.png")
    plot_metric_heatmap(results, class_names, "recall_per_class", "Recall",
                        FIGURES_DIR / "attacks_recall_heatmap.png")

    print("\n  [Matrices de confusion]")
    plot_confusion_matrix(clean_metrics["confusion_matrix"], class_names,
                          "Clean (baseline v4)",
                          FIGURES_DIR / "cm_clean.png")
    for r in results:
        safe = r["attack"].lower().replace("&", "").replace(" ", "_")
        plot_confusion_matrix(r["confusion_matrix"], class_names,
                              r["attack"],
                              FIGURES_DIR / f"cm_{safe}.png")

    print(f"\nTermine. Figures dans : {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
