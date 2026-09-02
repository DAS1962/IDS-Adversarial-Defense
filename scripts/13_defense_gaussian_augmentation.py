"""
Défense par Gaussian Augmentation (GA).

Pendant l'entraînement, on ajoute du bruit gaussien N(0, sigma^2) aux
features d'entrée à chaque batch. Cela oblige le modèle à apprendre des
représentations robustes aux petites perturbations, ce qui améliore
sa résistance aux attaques adversariales.

Après l'entraînement, on évalue le modèle sur les données propres et sur
les 6 attaques adversariales.

Référence : Zantedeschi et al. 2017, Awad et al. 2025.
"""

import sys
import time
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.dnn import BaselineDNN


DATA_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("results/checkpoints")
LOG_DIR = Path("results/logs")
ATTACKS_DIR = Path("results/attacks")

# Hyperparamètres de l'entraînement
EPOCHS = 50
BATCH_SIZE = 128
LR_INIT = 0.001
LR_MIN = 1e-5
LR_PATIENCE = 5
LR_FACTOR = 0.5
RANDOM_STATE = 42

# Hyperparamètre du bruit gaussien (papier Awad 2025)
GAUSSIAN_SIGMA = 0.1

# Architecture
NUM_CLASSES = 15
INPUT_DIM = 58

# Fichiers X_adv pour l'évaluation finale
ATTACK_FILES = [
    ("FGSM", "X_adv_fgsm.pkl"),
    ("BIM", "X_adv_bim.pkl"),
    ("PGD", "X_adv_pgd.pkl"),
    ("DeepFool", "X_adv_deepfool.pkl"),
    ("JSMA", "X_adv_jsma.pkl"),
    ("CW", "X_adv_cw.pkl"),
]


def load_data():
    """Charge les données d'entraînement et de test."""
    print("Chargement des données...")
    X_train = pd.read_pickle(DATA_DIR / "X_train.pkl")
    y_train = pd.read_pickle(DATA_DIR / "y_train.pkl")
    X_test = pd.read_pickle(DATA_DIR / "X_test.pkl")
    y_test = pd.read_pickle(DATA_DIR / "y_test.pkl")

    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values.astype(np.float32)
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values.astype(np.float32)
    if isinstance(y_train, pd.Series):
        y_train = y_train.values
    if isinstance(y_test, pd.Series):
        y_test = y_test.values

    print(f"  X_train : {X_train.shape}")
    print(f"  X_test  : {X_test.shape}")
    print()
    return X_train, y_train, X_test, y_test


def make_loaders(X_train, y_train, X_test, y_test):
    """Crée les DataLoaders."""
    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long(),
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test).long(),
    )
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    return train_loader, test_loader


def add_gaussian_noise(x, sigma):
    """Ajoute du bruit gaussien N(0, sigma^2) à un tenseur."""
    noise = torch.randn_like(x) * sigma
    return x + noise


def train_one_epoch(model, loader, optimizer, criterion, device, sigma):
    """Entraîne le modèle une epoch avec bruit gaussien sur les inputs."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # On ajoute du bruit gaussien aux inputs
        x_noisy = add_gaussian_noise(x, sigma)

        optimizer.zero_grad()
        logits = model(x_noisy)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * y.size(0)
        total_correct += (preds == y).sum().item()
        total_seen += y.size(0)

    return total_loss / total_seen, total_correct / total_seen


def evaluate(model, loader, criterion, device):
    """Évalue le modèle sur un DataLoader (sans bruit)."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_seen += y.size(0)

    return total_loss / total_seen, total_correct / total_seen


def predict_array(model, X, device):
    """Prédit sur un array numpy en batchs."""
    model.eval()
    n = len(X)
    preds = np.zeros(n, dtype=np.int64)

    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            end = min(i + BATCH_SIZE, n)
            xb = torch.tensor(X[i:end], dtype=torch.float32).to(device)
            preds[i:end] = model(xb).argmax(dim=1).cpu().numpy()

    return preds


def compute_metrics(y_true, y_pred, name):
    """Calcule les métriques standards."""
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    prec_wght = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    rec_wght = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_wght = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    return {
        "attack": name,
        "accuracy": acc,
        "precision_macro": prec_macro,
        "precision_weighted": prec_wght,
        "recall_macro": rec_macro,
        "recall_weighted": rec_wght,
        "f1_macro": f1_macro,
        "f1_weighted": f1_wght,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES))),
    }


def print_metrics(m):
    """Affiche les métriques d'une évaluation."""
    print(f"  Accuracy               : {m['accuracy']:.4f}")
    print(f"  Precision (macro)      : {m['precision_macro']:.4f}")
    print(f"  Precision (weighted)   : {m['precision_weighted']:.4f}")
    print(f"  Recall (macro)         : {m['recall_macro']:.4f}")
    print(f"  Recall (weighted)      : {m['recall_weighted']:.4f}")
    print(f"  F1 (macro)             : {m['f1_macro']:.4f}")
    print(f"  F1 (weighted)          : {m['f1_weighted']:.4f}")


def print_summary(results):
    """Affiche le tableau récapitulatif final."""
    print("\n" + "=" * 100)
    print("Résumé - Défense Gaussian Augmentation")
    print("=" * 100)
    header = (
        f"{'Attaque':<12} {'Accuracy':>10} {'Prec. macro':>12} "
        f"{'Rec. macro':>11} {'F1 macro':>10} {'F1 wght':>10}"
    )
    print(header)
    print("-" * 100)
    for r in results:
        print(
            f"{r['attack']:<12} {r['accuracy']:>10.4f} "
            f"{r['precision_macro']:>12.4f} {r['recall_macro']:>11.4f} "
            f"{r['f1_macro']:>10.4f} {r['f1_weighted']:>10.4f}"
        )
    print("=" * 100)


def main():
    print("=" * 70)
    print("Défense par Gaussian Augmentation (GA)")
    print(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

    # Reproductibilité
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print()

    # Chargement des données
    X_train, y_train, X_test, y_test = load_data()
    train_loader, test_loader = make_loaders(X_train, y_train, X_test, y_test)
    print(f"Batches train : {len(train_loader):,}")
    print(f"Batches test  : {len(test_loader):,}")
    print()

    # Modèle
    print("Création du modèle...")
    model = BaselineDNN(input_dim=INPUT_DIM, hidden1=512, hidden2=256, output_dim=NUM_CLASSES)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Paramètres : {n_params:,}")
    print()

    # Optimizer et scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_INIT)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=LR_FACTOR,
        patience=LR_PATIENCE, min_lr=LR_MIN,
    )

    # Entraînement
    print("Début de l'entraînement avec Gaussian Augmentation")
    print(f"  Epochs            : {EPOCHS}")
    print(f"  Learning rate init: {LR_INIT}")
    print(f"  Bruit sigma       : {GAUSSIAN_SIGMA}")
    print(f"  Batch size        : {BATCH_SIZE}")
    print()

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [], "lr": []}
    best_acc = 0.0
    best_epoch = -1
    ckpt_path = CHECKPOINT_DIR / "defense_ga_best.pth"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    start_train = time.time()

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, GAUSSIAN_SIGMA,
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step(test_acc)
        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["lr"].append(lr_now)

        print(
            f"Epoch {epoch:2d}/{EPOCHS} | lr={lr_now:.6f} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} | "
            f"time={elapsed:.1f}s",
            flush=True,
        )

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "test_acc": test_acc,
            }, ckpt_path)
            print(f"           -> Nouveau meilleur modèle sauvegardé ({test_acc:.4f})", flush=True)

    train_time_min = (time.time() - start_train) / 60
    print()
    print(f"Entraînement terminé en {train_time_min:.1f} min")
    print(f"Meilleur epoch : {best_epoch} | test_acc = {best_acc:.4f}")
    print()

    # Chargement du meilleur modèle
    print("Chargement du meilleur modèle pour évaluation...")
    checkpoint = torch.load(ckpt_path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Évaluation sur données propres et attaques
    print()
    print("=" * 70)
    print("Évaluation sur données propres et attaques adversariales")
    print("=" * 70)

    all_results = []

    print("\n--- Données propres ---")
    y_pred_clean = predict_array(model, X_test, device)
    clean_metrics = compute_metrics(y_test, y_pred_clean, "Clean")
    print_metrics(clean_metrics)
    all_results.append(clean_metrics)

    for name, x_file in ATTACK_FILES:
        x_path = ATTACKS_DIR / x_file
        if not x_path.exists():
            print(f"\n--- {name} ---")
            print(f"  Fichier {x_file} non trouvé, skip")
            continue

        print(f"\n--- {name} ---")
        X_adv = joblib.load(x_path)
        y_pred = predict_array(model, X_adv, device)
        m = compute_metrics(y_test, y_pred, name)
        print_metrics(m)
        all_results.append(m)
        del X_adv

    print_summary(all_results)

    # Sauvegarde des résultats
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = LOG_DIR / f"defense_ga_{timestamp}.pkl"
    joblib.dump({
        "results": all_results,
        "history": history,
        "hyperparameters": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr_init": LR_INIT,
            "gaussian_sigma": GAUSSIAN_SIGMA,
            "seed": RANDOM_STATE,
        },
        "best_epoch": best_epoch,
        "training_time_min": train_time_min,
    }, out_path)
    print(f"\nRésultats sauvegardés : {out_path}")
    print("\nDéfense Gaussian Augmentation terminée")


if __name__ == "__main__":
    main()
