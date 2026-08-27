
"""
Défense par Label Smoothing (LS).

Reproduction du mécanisme de défense LS du papier Awad et al. (2025).

Principe :
  Au lieu d'utiliser des labels one-hot durs (0, 0, 1, 0, ...), on adoucit
  les labels : la vraie classe reçoit (1 - alpha) et les autres classes se
  partagent alpha uniformément. Cela réduit la sur-confiance du modèle et
  améliore sa robustesse face aux attaques adversariales.

Hyperparamètres :
  - alpha (smoothing) : 0.1
  - Reste identique au baseline : lr=0.001, scheduler, 50 epochs

Correspond à l'étape 8 du framework Awad et al. (2025).
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
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Ajouter src/ au path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.dnn import BaselineDNN, count_parameters


DATA_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("results/checkpoints")
LOG_DIR = Path("results/logs")
ATTACKS_DIR = Path("results/attacks")

# Hyperparamètres du baseline (identiques)
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
RANDOM_STATE = 42

# Paramètres du scheduler
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5
SCHEDULER_MIN_LR = 1e-5

# Paramètre spécifique à Label Smoothing
LABEL_SMOOTHING_ALPHA = 0.1

NUM_WORKERS = 4
NUM_CLASSES = 15
INPUT_DIM = 58


class IDSDataset(Dataset):
    """Dataset PyTorch pour les données IDS."""

    def __init__(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data(data_dir):
    """Charge X_train, X_test, y_train, y_test."""
    print("Chargement des données...")
    X_train = pd.read_pickle(data_dir / "X_train.pkl")
    X_test = pd.read_pickle(data_dir / "X_test.pkl")
    y_train = joblib.load(data_dir / "y_train.pkl")
    y_test = pd.read_pickle(data_dir / "y_test.pkl")

    if isinstance(y_test, pd.Series):
        y_test = y_test.values

    print(f"  X_train : {X_train.shape}")
    print(f"  X_test  : {X_test.shape}")
    print(f"  Classes : {len(np.unique(y_train))}")
    print()

    return X_train, X_test, y_train, y_test


def create_dataloaders(X_train, y_train, X_test, y_test, batch_size, num_workers):
    """Crée les DataLoaders PyTorch."""
    print("Création des DataLoaders...")

    train_dataset = IDSDataset(X_train, y_train)
    test_dataset = IDSDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"  Train : {len(train_loader):,} batches")
    print(f"  Test  : {len(test_loader):,} batches\n")

    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Une epoch d'entraînement."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_X, batch_y in loader:
        batch_X = batch_X.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_X.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(batch_y).sum().item()
        total += batch_y.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    """Évaluation sur un dataset."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item() * batch_X.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_y.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    return (
        total_loss / total,
        correct / total,
        np.array(all_preds),
        np.array(all_labels),
    )


def train_model(model, train_loader, test_loader, num_epochs, learning_rate, device):
    """Boucle principale d'entraînement avec Label Smoothing."""
    print("Début de l'entraînement avec Label Smoothing")
    print(f"  Device            : {device}")
    print(f"  Epochs            : {num_epochs}")
    print(f"  Learning rate init: {learning_rate}")
    print(f"  Label smoothing α : {LABEL_SMOOTHING_ALPHA}")
    print(f"  Batch size        : {train_loader.batch_size}")
    print()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Label Smoothing intégré dans CrossEntropyLoss (PyTorch 1.10+)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING_ALPHA)

    scheduler = ReduceLROnPlateau(
        optimizer, mode='min',
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=SCHEDULER_MIN_LR,
    )

    history = {
        "train_loss": [], "train_acc": [],
        "test_loss": [], "test_acc": [],
        "learning_rate": [], "epoch_time": [],
    }

    best_test_acc = 0.0
    best_epoch = 0

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)

        epoch_time = time.time() - start_time

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["learning_rate"].append(current_lr)
        history["epoch_time"].append(epoch_time)

        print(
            f"Epoch {epoch:2d}/{num_epochs} | "
            f"lr={current_lr:.6f} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} | "
            f"time={epoch_time:.1f}s"
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "test_acc": test_acc,
                "test_loss": test_loss,
                "learning_rate": current_lr,
                "defense": "label_smoothing",
                "alpha": LABEL_SMOOTHING_ALPHA,
            }, CHECKPOINT_DIR / "defense_label_smoothing_best.pth")
            print(f"           -> Nouveau meilleur modèle sauvegardé ({test_acc:.4f})")

        scheduler.step(test_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < current_lr:
            print(f"           -> Learning rate reduit : {current_lr:.6f} -> {new_lr:.6f}")

    print()
    print(f"Meilleur modèle : epoch {best_epoch}, test_acc = {best_test_acc:.4f}")
    print()

    return history


def evaluate_on_clean_and_attacks(model, X_test, y_test, device):
    """
    Évalue le modèle sur données propres et sur toutes les attaques disponibles.
    """
    print("=" * 70)
    print("Évaluation sur données propres et attaques adversariales")
    print("=" * 70)

    results = []

    # 1. Évaluation sur données propres
    print("\n--- Données propres ---")
    result = evaluate_on_data(model, X_test, y_test, device, "Clean")
    results.append(result)

    # 2. Évaluation sur chaque attaque disponible
    attack_files = {
        "FGSM": "X_adv_fgsm.pkl",
        "BIM": "X_adv_bim.pkl",
        "PGD": "X_adv_pgd.pkl",
        "DeepFool": "X_adv_deepfool.pkl",
        "JSMA": "X_adv_jsma_sample20k.pkl",
        "CW": "X_adv_cw.pkl",
    }

    for name, filename in attack_files.items():
        path = ATTACKS_DIR / filename
        if not path.exists():
            print(f"\n--- {name} ---")
            print(f"  Fichier {filename} non trouvé, skip")
            continue

        print(f"\n--- {name} ---")
        X_adv = joblib.load(path)

        # JSMA a un échantillon différent, il faut charger les labels correspondants
        if name == "JSMA":
            y_labels_path = ATTACKS_DIR / "y_jsma_sample20k.pkl"
            if y_labels_path.exists():
                y_adv = joblib.load(y_labels_path)
            else:
                y_adv = y_test
        else:
            y_adv = y_test

        result = evaluate_on_data(model, X_adv, y_adv, device, name)
        results.append(result)
        del X_adv

    return results


def evaluate_on_data(model, X, y, device, name):
    """Évalue le modèle sur un dataset spécifique."""
    model.eval()

    if isinstance(X, pd.DataFrame):
        X = X.values.astype(np.float32)

    n_samples = len(X)
    predictions = np.zeros(n_samples, dtype=np.int64)

    with torch.no_grad():
        for i in range(0, n_samples, BATCH_SIZE):
            end = min(i + BATCH_SIZE, n_samples)
            X_batch = torch.tensor(X[i:end], dtype=torch.float32).to(device)
            outputs = model(X_batch)
            _, preds = outputs.max(1)
            predictions[i:end] = preds.cpu().numpy()

    acc = accuracy_score(y, predictions)
    precision_macro = precision_score(y, predictions, average="macro", zero_division=0)
    precision_weighted = precision_score(y, predictions, average="weighted", zero_division=0)
    recall_macro = recall_score(y, predictions, average="macro", zero_division=0)
    recall_weighted = recall_score(y, predictions, average="weighted", zero_division=0)
    f1_macro = f1_score(y, predictions, average="macro", zero_division=0)
    f1_weighted = f1_score(y, predictions, average="weighted", zero_division=0)

    print(f"  Accuracy   : {acc:.4f}")
    print(f"  F1 macro   : {f1_macro:.4f}")
    print(f"  F1 weighted: {f1_weighted:.4f}")

    return {
        "attack": name,
        "accuracy": acc,
        "precision_macro": precision_macro,
        "precision_weighted": precision_weighted,
        "recall_macro": recall_macro,
        "recall_weighted": recall_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "predictions": predictions,
    }


def print_summary_table(results):
    """Affiche le tableau récapitulatif."""
    print("\n" + "=" * 90)
    print("Résumé - Défense Label Smoothing")
    print("=" * 90)
    print(f"{'Attaque':<12} {'Accuracy':>10} {'Prec. macro':>12} {'Rec. macro':>11} {'F1 macro':>10} {'F1 wght':>10}")
    print("-" * 90)

    for r in results:
        print(
            f"{r['attack']:<12} "
            f"{r['accuracy']:>10.4f} "
            f"{r['precision_macro']:>12.4f} "
            f"{r['recall_macro']:>11.4f} "
            f"{r['f1_macro']:>10.4f} "
            f"{r['f1_weighted']:>10.4f}"
        )
    print("=" * 90)


def main():
    print("=" * 70)
    print("Défense par Label Smoothing (LS)")
    print(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

    # Reproductibilité
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print()

    # Charger données
    X_train, X_test, y_train, y_test = load_data(DATA_DIR)
    train_loader, test_loader = create_dataloaders(
        X_train, y_train, X_test, y_test, BATCH_SIZE, NUM_WORKERS
    )

    # Créer modèle
    print("Création du modèle...")
    model = BaselineDNN(input_dim=INPUT_DIM, hidden1=512, hidden2=256, output_dim=NUM_CLASSES)
    model = model.to(device)
    print(f"  Paramètres : {count_parameters(model):,}\n")

    # Entraîner avec Label Smoothing
    history = train_model(model, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, device)

    # Charger le meilleur modèle
    print("Chargement du meilleur modèle pour évaluation...")
    checkpoint = torch.load(
        CHECKPOINT_DIR / "defense_label_smoothing_best.pth",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Meilleur epoch : {checkpoint['epoch']}")
    print(f"  Test accuracy  : {checkpoint['test_acc']:.4f}")
    print()

    # Évaluer sur clean + toutes les attaques
    results = evaluate_on_clean_and_attacks(model, X_test, y_test, device)

    # Résumé
    print_summary_table(results)

    # Sauvegarder les résultats
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    joblib.dump({
        "history": history,
        "results": results,
        "defense": "label_smoothing",
        "alpha": LABEL_SMOOTHING_ALPHA,
    }, LOG_DIR / f"defense_label_smoothing_{timestamp}.pkl")

    print(f"\nRésultats sauvegardés : {LOG_DIR / f'defense_label_smoothing_{timestamp}.pkl'}")
    print("\nDéfense Label Smoothing terminée")


if __name__ == "__main__":
    main()