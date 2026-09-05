
"""
Generation des graphiques a partir des resultats du baseline.

Produit 5 figures :
  1. Learning curves (loss + accuracy train/test)
  2. Evolution du learning rate
  3. Metriques finales (Accuracy, Precision, Recall, F1) - format papier
  4. F1 par classe (bar chart)
  5. Matrice de confusion (heatmap)

Utilise le fichier .pkl le plus recent dans results/logs/.
"""

import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score


LOG_DIR = Path("results/logs")
FIGURES_DIR = Path("results/figures")
DATA_DIR = Path("data/processed")


def load_latest_results():
    """Charge le fichier baseline_training_*.pkl le plus recent."""
    files = sorted(LOG_DIR.glob("baseline_training_*.pkl"))
    if not files:
        raise FileNotFoundError("Aucun fichier baseline_training_*.pkl trouve")
    latest = files[-1]
    print(f"Chargement de : {latest.name}")
    data = joblib.load(latest)
    return data["history"], data["results"], "final_" + latest.stem


def plot_learning_curves(history, output_path):
    """Loss et accuracy train/test en fonction des epochs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train Loss", marker='o', markersize=4)
    ax1.plot(epochs, history["val_loss"], label="Validation Loss", marker='s', markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss evolution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], label="Train Accuracy", marker='o', markersize=4)
    ax2.plot(epochs, history["val_acc"], label="Validation Accuracy", marker='s', markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy evolution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Sauvegarde : {output_path.name}")


def plot_learning_rate(history, output_path):
    """Evolution du learning rate en fonction des epochs."""
    fig, ax = plt.subplots(figsize=(10, 5))

    epochs = range(1, len(history["learning_rate"]) + 1)
    ax.plot(epochs, history["learning_rate"], marker='o', markersize=6, color='purple')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning rate evolution (ReduceLROnPlateau)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Sauvegarde : {output_path.name}")


def plot_final_metrics(results, output_path):
    """Bar chart Accuracy/Precision/Recall/F1 comme dans le papier."""
    predictions = results["predictions"]
    labels = results["labels"]

    metrics = {
        "Accuracy": results["accuracy"] * 100,
        "Precision": precision_score(labels, predictions, average="weighted", zero_division=0) * 100,
        "Recall": recall_score(labels, predictions, average="weighted", zero_division=0) * 100,
        "F1-score": f1_score(labels, predictions, average="weighted", zero_division=0) * 100,
    }

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(metrics.keys(), metrics.values(), color=['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000'])

    for bar, val in zip(bars, metrics.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.3f}%", ha='center', fontsize=11, fontweight='bold')

    ax.set_ylabel("Values (%)")
    ax.set_title("DNN-based IDS Classifier Performance (adversarial-free)")
    ax.set_ylim(0, 105)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Sauvegarde : {output_path.name}")


def plot_f1_per_class(results, output_path):
    """F1 par classe pour identifier les classes problematiques."""
    from sklearn.metrics import f1_score

    predictions = results["predictions"]
    labels = results["labels"]
    label_encoder = joblib.load(DATA_DIR / "label_encoder.pkl")

    f1_scores = f1_score(labels, predictions, average=None, zero_division=0)
    class_names = [str(c) for c in label_encoder.classes_]

    # Trier par F1 decroissant pour lisibilite
    sorted_idx = np.argsort(f1_scores)[::-1]
    f1_sorted = f1_scores[sorted_idx] * 100
    names_sorted = [class_names[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#2E7D32' if f > 80 else '#F9A825' if f > 40 else '#C62828' for f in f1_sorted]
    bars = ax.bar(range(len(names_sorted)), f1_sorted, color=colors)

    ax.set_xticks(range(len(names_sorted)))
    ax.set_xticklabels(names_sorted, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel("F1-score (%)")
    ax.set_title("F1-score par classe")
    ax.set_ylim(0, 105)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Sauvegarde : {output_path.name}")


def plot_confusion_matrix(results, output_path):
    """Matrice de confusion en heatmap (normalisee par ligne)."""
    cm = results["confusion_matrix"]
    label_encoder = joblib.load(DATA_DIR / "label_encoder.pkl")
    class_names = [str(c) for c in label_encoder.classes_]

    # Normaliser par ligne pour lisibilite (proportion des predictions par vraie classe)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    supports = cm.sum(axis=1)
    row_labels = [f"{n} (n={s:,})" for n, s in zip(class_names, supports)]

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=row_labels,
        cbar_kws={'label': 'Proportion'},
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (normalized by row)")

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Sauvegarde : {output_path.name}")


def main():
    print("Generation des graphiques de resultats")
    print(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    history, results, base_name = load_latest_results()

    print("\nGeneration des figures :")
    plot_learning_curves(history, FIGURES_DIR / f"{base_name}_01_learning_curves.png")
    plot_learning_rate(history, FIGURES_DIR / f"{base_name}_02_learning_rate.png")
    plot_final_metrics(results, FIGURES_DIR / f"{base_name}_03_final_metrics.png")
    plot_f1_per_class(results, FIGURES_DIR / f"{base_name}_04_f1_per_class.png")
    plot_confusion_matrix(results, FIGURES_DIR / f"{base_name}_05_confusion_matrix.png")

    print(f"\nTous les graphiques sauvegardes dans : {FIGURES_DIR}/")


if __name__ == "__main__":
    main()