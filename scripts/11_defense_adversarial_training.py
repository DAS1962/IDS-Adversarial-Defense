
"""
Défense par Adversarial Training (AT).

Entraîne un DNN sur un mélange 50/50 d'exemples propres et d'exemples adversariaux
générés à la volée avec FGSM. Architecture et hyperparamètres identiques au
baseline v4 pour permettre une comparaison directe.
"""

from __future__ import annotations

import json
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
EPOCHS = 50
BATCH_SIZE = 128
LR_INIT = 1e-3
LR_MIN = 1e-5
LR_PATIENCE = 5
LR_FACTOR = 0.5
FGSM_EPSILON = 0.05
ADV_RATIO = 0.5

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "results" / "checkpoints"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
LOG_DIR = PROJECT_ROOT / "results" / "logs"

for d in (CHECKPOINT_DIR, FIGURES_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)


class BaselineDNN(nn.Module):
    """Architecture identique au baseline v4."""

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    with open(DATA_DIR / "X_train.pkl", "rb") as f:
        X_train = pickle.load(f)
    with open(DATA_DIR / "y_train.pkl", "rb") as f:
        y_train = pickle.load(f)
    with open(DATA_DIR / "X_test.pkl", "rb") as f:
        X_test = pickle.load(f)
    with open(DATA_DIR / "y_test.pkl", "rb") as f:
        y_test = pickle.load(f)

    if hasattr(X_train, "values"):
        X_train = X_train.values.astype(np.float32)
    if hasattr(X_test, "values"):
        X_test = X_test.values.astype(np.float32)
    if hasattr(y_train, "values"):
        y_train = y_train.values
    if hasattr(y_test, "values"):
        y_test = y_test.values

    num_classes = int(np.max(np.concatenate([y_train, y_test])) + 1)
    return X_train, y_train, X_test, y_test, num_classes


def make_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[DataLoader, DataLoader]:
    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long(),
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test).long(),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    return train_loader, test_loader


def fgsm_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    criterion: nn.Module,
) -> torch.Tensor:
    """Génère des exemples adversariaux avec FGSM (une étape)."""
    x_adv = x.clone().detach().requires_grad_(True)
    logits = model(x_adv)
    loss = criterion(logits, y)
    grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
    x_adv = x_adv.detach() + epsilon * grad.sign()
    return x_adv.detach()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epsilon: float,
    adv_ratio: float,
) -> tuple[float, float]:
    """Entraîne une epoch en mélangeant exemples propres et adversariaux."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        n = x.size(0)
        n_adv = int(n * adv_ratio)

        if n_adv > 0:
            idx = torch.randperm(n, device=device)
            idx_adv = idx[:n_adv]
            idx_clean = idx[n_adv:]

            model.eval()
            x_adv = fgsm_attack(model, x[idx_adv], y[idx_adv], epsilon, criterion)
            model.train()

            x_batch = torch.cat([x[idx_clean], x_adv], dim=0)
            y_batch = torch.cat([y[idx_clean], y[idx_adv]], dim=0)
        else:
            x_batch = x
            y_batch = y

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * y_batch.size(0)
        total_correct += (preds == y_batch).sum().item()
        total_seen += y_batch.size(0)

    return total_loss / total_seen, total_correct / total_seen


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_seen += y.size(0)
    return total_loss / total_seen, total_correct / total_seen


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    preds = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        preds.append(model(x).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def main() -> None:
    set_seed(SEED)
    start = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"Défense par Adversarial Training (AT)")
    print(f"Date : {ts}")
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print()

    print("Chargement des données...")
    X_train, y_train, X_test, y_test, num_classes = load_data()
    print(f"  X_train : {X_train.shape}")
    print(f"  X_test  : {X_test.shape}")
    print(f"  Classes : {num_classes}")
    print()

    print("Création des DataLoaders...")
    train_loader, test_loader = make_loaders(X_train, y_train, X_test, y_test)
    print(f"  Train : {len(train_loader):,} batches")
    print(f"  Test  : {len(test_loader):,} batches")
    print()

    print("Création du modèle...")
    model = BaselineDNN(X_train.shape[1], num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Paramètres : {n_params:,}")
    print()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_INIT)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=LR_MIN,
    )

    print("Début de l'entraînement adversarial")
    print(f"  Epochs            : {EPOCHS}")
    print(f"  Learning rate init: {LR_INIT}")
    print(f"  FGSM epsilon      : {FGSM_EPSILON}")
    print(f"  Ratio adversarial : {ADV_RATIO:.0%}")
    print(f"  Batch size        : {BATCH_SIZE}")
    print()

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "lr": [],
    }
    best_acc = 0.0
    best_epoch = -1
    ckpt_path = CHECKPOINT_DIR / "defense_at_best.pth"

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, FGSM_EPSILON, ADV_RATIO
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
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "test_acc": test_acc,
                    "input_dim": X_train.shape[1],
                    "num_classes": num_classes,
                },
                ckpt_path,
            )
            print(f"           -> Nouveau meilleur modèle sauvegardé ({test_acc:.4f})", flush=True)

    print()
    print(f"Entraînement terminé en {(time.time() - start) / 60:.1f} min")
    print(f"Meilleur epoch : {best_epoch} | test_acc = {best_acc:.4f}")
    print()

    print("Évaluation finale sur le test set...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    y_pred = predict(model, test_loader, device)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"  Accuracy    : {acc:.4f}")
    print(f"  F1 macro    : {f1_macro:.4f}")
    print(f"  F1 weighted : {f1_weighted:.4f}")
    print()

    print("Rapport détaillé :")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "defense": "adversarial_training",
        "date": ts,
        "hyperparameters": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr_init": LR_INIT,
            "lr_min": LR_MIN,
            "lr_patience": LR_PATIENCE,
            "lr_factor": LR_FACTOR,
            "fgsm_epsilon": FGSM_EPSILON,
            "adv_ratio": ADV_RATIO,
            "seed": SEED,
        },
        "best_epoch": best_epoch,
        "test_accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "training_time_min": (time.time() - start) / 60,
        "history": history,
    }
    metrics_path = LOG_DIR / "defense_at_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Métriques sauvegardées : {metrics_path}")

    cm_path = CHECKPOINT_DIR / "defense_at_confusion_matrix.npy"
    np.save(cm_path, cm)
    print(f"Matrice de confusion   : {cm_path}")


if __name__ == "__main__":
    main()