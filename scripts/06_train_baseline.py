
"""
Entraînement du DNN baseline sur CIC-IDS 2017.

Reproduction des hyperparamètres du papier Awad et al. (2025) avec ajout
d'un learning rate scheduler pour stabiliser l'entraînement :
  - Architecture : 58 -> 512 -> 256 -> 15
  - Optimizer   : Adam avec learning_rate initial = 0.01
  - Scheduler   : ReduceLROnPlateau (divise le lr par 10 si test_loss stagne)
  - Loss        : CrossEntropyLoss
  - Batch size  : 128
  - Epochs      : 30

Le modèle est entraîné sur le train (après SMOTE) et évalué sur le test
(distribution naturelle, sans SMOTE) à chaque epoch. Le meilleur modèle
(selon accuracy test) est sauvegardé.

Correspond à l'étape 5 du framework Awad et al. (2025).
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
    f1_score,
    classification_report,
    confusion_matrix,
)

# Ajouter src/ au path pour importer notre modèle
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.dnn import BaselineDNN, count_parameters


# Chemins des dossiers
DATA_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("results/checkpoints")
LOG_DIR = Path("results/logs")

# Hyperparamètres
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 30
RANDOM_STATE = 42

# Paramètres du learning rate scheduler
SCHEDULER_FACTOR = 0.5          # reductions plus douces
SCHEDULER_PATIENCE = 5          # plus tolerant
SCHEDULER_MIN_LR = 1e-5         #ne pas descendre en dessous

# Configuration technique
NUM_WORKERS = 4  # Chargement parallèle des batches


# Dataset PyTorch pour les données IDS
class IDSDataset(Dataset):
    """
    Convertit les DataFrames pandas en tensors PyTorch à la volée.
    Les données sont stockées en RAM (déjà chargées), le dataset ne fait
    que les indexer et convertir.
    """

    def __init__(self, X, y):
        # Conversion en tensors PyTorch (float32 pour features, long pour labels)
        # float32 est suffisant pour DNN (plus rapide que float64)
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


# Fonction 1 : charger les données
def load_data(data_dir):
    """
    Charge les fichiers pickle produits par l'étape 4.
    Retourne 4 objets : X_train, X_test, y_train, y_test.
    """
    print("Chargement des données...")

    X_train = pd.read_pickle(data_dir / "X_train.pkl")
    X_test = pd.read_pickle(data_dir / "X_test.pkl")
    y_train = joblib.load(data_dir / "y_train.pkl")
    y_test = pd.read_pickle(data_dir / "y_test.pkl")

    # y_train peut être un numpy array (issu de SMOTE), on le laisse tel quel
    # y_test est une Series pandas, on le convertit
    if isinstance(y_test, pd.Series):
        y_test = y_test.values

    print(f"  X_train : {X_train.shape} ({X_train.memory_usage(deep=True).sum() / 1e9:.2f} GB)")
    print(f"  X_test  : {X_test.shape}")
    print(f"  y_train : {len(y_train):,} labels")
    print(f"  y_test  : {len(y_test):,} labels")
    print(f"  Classes : {len(np.unique(y_train))}")
    print()

    return X_train, X_test, y_train, y_test


# Fonction 2 : créer les DataLoaders
def create_dataloaders(X_train, y_train, X_test, y_test, batch_size, num_workers):
    """
    Crée les DataLoaders PyTorch pour train et test.

    Le train est shuffled à chaque epoch (bonne pratique).
    Le test ne l'est pas (pour reproductibilité de l'évaluation).
    """
    print("Création des DataLoaders...")

    train_dataset = IDSDataset(X_train, y_train)
    test_dataset = IDSDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # Accélère le transfert vers GPU
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"  Train : {len(train_loader):,} batches de {batch_size}")
    print(f"  Test  : {len(test_loader):,} batches de {batch_size}")
    print()

    return train_loader, test_loader


# Fonction 3 : boucle d'entraînement pour une epoch
def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Effectue une passe complète sur les données d'entraînement.
    Retourne la loss moyenne et l'accuracy sur cette epoch.
    """
    model.train()  # Mode entraînement (active dropout, batchnorm, etc.)

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_X, batch_y in loader:
        # Déplacer les données sur le device (GPU si dispo)
        batch_X = batch_X.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        # Reset des gradients (sinon ils s'accumulent)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_X)

        # Calcul de la loss
        loss = criterion(outputs, batch_y)

        # Backward pass (calcul des gradients)
        loss.backward()

        # Mise à jour des poids
        optimizer.step()

        # Statistiques
        total_loss += loss.item() * batch_X.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(batch_y).sum().item()
        total += batch_y.size(0)

    return total_loss / total, correct / total


# Fonction 4 : évaluation sur le test
def evaluate(model, loader, criterion, device):
    """
    Évalue le modèle sur un dataset (typiquement le test).
    Retourne loss moyenne, accuracy, et les prédictions complètes
    pour analyse détaillée (matrice de confusion, F1).
    """
    model.eval()  # Mode évaluation (désactive dropout, batchnorm, etc.)

    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    # torch.no_grad() désactive le calcul de gradients (rapide, moins de RAM)
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

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    return (
        total_loss / total,
        correct / total,
        np.array(all_predictions),
        np.array(all_labels),
    )


# Fonction 5 : boucle principale d'entraînement
def train_model(model, train_loader, test_loader, num_epochs, learning_rate, device):
    """
    Boucle d'entraînement complète sur num_epochs.

    Utilise ReduceLROnPlateau pour ajuster automatiquement le learning rate
    quand la test_loss stagne : divise le lr par 10 après 2 epochs sans
    amélioration. Ceci stabilise l'entraînement quand lr=0.01 devient trop
    élevé pour continuer à converger.

    Sauvegarde le meilleur modèle basé sur l'accuracy test.
    Retourne l'historique des métriques pour analyse.
    """
    print("Début de l'entraînement")
    print(f"  Device            : {device}")
    print(f"  Epochs            : {num_epochs}")
    print(f"  Learning rate init: {learning_rate}")
    print(f"  Scheduler         : ReduceLROnPlateau (factor={SCHEDULER_FACTOR}, patience={SCHEDULER_PATIENCE})")
    print(f"  Batch size        : {train_loader.batch_size}")
    print()

    # Optimizer et loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Learning rate scheduler : réduit le lr si test_loss stagne
    # mode='min' : on surveille une métrique à minimiser (la loss)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=SCHEDULER_MIN_LR, 
    )

    # Historique pour analyse
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "learning_rate": [],
        "epoch_time": [],
    }

    best_test_acc = 0.0
    best_epoch = 0

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        # Récupérer le lr courant avant l'entraînement de l'epoch
        current_lr = optimizer.param_groups[0]['lr']

        # Entraînement
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Évaluation
        test_loss, test_acc, _, _ = evaluate(
            model, test_loader, criterion, device
        )

        epoch_time = time.time() - start_time

        # Enregistrer dans l'historique
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["learning_rate"].append(current_lr)
        history["epoch_time"].append(epoch_time)

        # Log avec le lr courant
        print(
            f"Epoch {epoch:2d}/{num_epochs} | "
            f"lr={current_lr:.6f} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} | "
            f"time={epoch_time:.1f}s"
        )

        # Sauvegarder le meilleur modèle
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "test_acc": test_acc,
                    "test_loss": test_loss,
                    "learning_rate": current_lr,
                },
                CHECKPOINT_DIR / "baseline_best.pth",
            )
            print(f"           -> Nouveau meilleur modèle sauvegardé ({test_acc:.4f})")

        # Ajuster le lr en fonction de la test_loss
        # Doit être appelé APRES l'évaluation, avec la métrique surveillée
        scheduler.step(test_loss)

        # Détecter et logger si le lr a été réduit
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < current_lr:
            print(f"           -> Learning rate reduit : {current_lr:.6f} -> {new_lr:.6f}")

    print()
    print(f"Meilleur modèle : epoch {best_epoch}, test_acc = {best_test_acc:.4f}")
    print()

    return history


# Fonction 6 : évaluation finale détaillée
def final_evaluation(model, test_loader, device, label_encoder=None):
    """
    Évaluation finale complète sur le test :
    - Accuracy globale
    - F1 macro (prend en compte le déséquilibre)
    - Classification report par classe
    - Matrice de confusion
    """
    print("Évaluation finale sur le test set")

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, predictions, labels = evaluate(
        model, test_loader, criterion, device
    )

    print(f"\nAccuracy globale : {test_acc:.4f}")
    print(f"F1 macro         : {f1_score(labels, predictions, average='macro'):.4f}")
    print(f"F1 weighted      : {f1_score(labels, predictions, average='weighted'):.4f}")

    # Classification report par classe
    print("\nRapport par classe :")
    if label_encoder is not None:
        target_names = [str(c) for c in label_encoder.classes_]
    else:
        target_names = None

    print(classification_report(
        labels, predictions,
        target_names=target_names,
        digits=4,
        zero_division=0,
    ))

    # Matrice de confusion
    cm = confusion_matrix(labels, predictions)
    print("\nMatrice de confusion (lignes=vrai, colonnes=prédit) :")
    print(cm)

    return {
        "accuracy": test_acc,
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "f1_weighted": f1_score(labels, predictions, average="weighted"),
        "confusion_matrix": cm,
        "predictions": predictions,
        "labels": labels,
    }


# Fonction principale
def main():
    print("Baseline DNN — CIC-IDS 2017")
    print(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Reproductibilité
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)

    # Device (GPU si disponible)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device utilisé : {device}")
    if device.type == "cuda":
        print(f"GPU            : {torch.cuda.get_device_name(0)}")
        print(f"CUDA version   : {torch.version.cuda}")
    print()

    # Charger les données
    X_train, X_test, y_train, y_test = load_data(DATA_DIR)

    # Créer les DataLoaders
    train_loader, test_loader = create_dataloaders(
        X_train, y_train, X_test, y_test, BATCH_SIZE, NUM_WORKERS
    )

    # Créer le modèle
    print("Création du modèle...")
    model = BaselineDNN(input_dim=58, hidden1=512, hidden2=256, output_dim=15)
    model = model.to(device)
    print(f"  Paramètres : {count_parameters(model):,}")
    print()

    # Entraîner
    history = train_model(
        model, train_loader, test_loader, NUM_EPOCHS, LEARNING_RATE, device
    )

    # Charger le meilleur modèle pour l'évaluation finale
    print("Chargement du meilleur modèle pour évaluation finale...")
    checkpoint = torch.load(
        CHECKPOINT_DIR / "baseline_best.pth", weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Meilleur epoch : {checkpoint['epoch']}")
    print(f"  Test accuracy  : {checkpoint['test_acc']:.4f}")
    print(f"  Learning rate  : {checkpoint['learning_rate']:.6f}")
    print()

    # Chargement du label encoder pour les noms de classes
    label_encoder = joblib.load(DATA_DIR / "label_encoder.pkl")

    # Évaluation finale détaillée
    results = final_evaluation(model, test_loader, device, label_encoder)

    # Sauvegarder l'historique et les résultats
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    joblib.dump(
        {"history": history, "results": results},
        LOG_DIR / f"baseline_training_{timestamp}.pkl",
    )
    print(f"Historique sauvegardé : {LOG_DIR / f'baseline_training_{timestamp}.pkl'}")

    print()
    print("Baseline DNN terminé")


if __name__ == "__main__":
    main()