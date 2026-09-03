
"""
Entraînement du DNN baseline sur CIC-IDS 2017.

Hyperparamètres pilotés par configs/config.yaml (section training), pas de
valeur en dur ici. Deux déviations documentées par rapport à la Table 3 de
l'article :
  - learning_rate 0.01 -> 0.001 : lr=0.01 produisait une instabilité claire
    dès la 2e epoch (voir README, itérations v1/v2).
  - epochs 30 -> 50 : le modèle progressait encore à l'epoch 30.

Sélection du modèle : sur le split de VALIDATION (config.dataset.val_size),
jamais sur le test. La version précédente sélectionnait le meilleur epoch
sur best_test_acc, ce qui biaise l'accuracy rapportée à la hausse — le test
set servait alors à la fois à choisir le modèle et à l'évaluer. Le test
n'est évalué qu'UNE SEULE FOIS, après l'entraînement complet, avec le
modèle déjà figé (final_evaluation). Il n'est pas évalué à chaque epoch,
même à titre informatif : afficher test_acc pendant l'entraînement
réintroduirait une fuite par l'humain qui observe le chiffre et ajuste en
conséquence — tout l'intérêt du split de validation est de ne toucher au
test qu'une fois. C'est aussi plus rapide (une passe de moins par epoch
sur ~832k échantillons).

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
    f1_score,
    classification_report,
    confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.dnn import BaselineDNN, count_parameters
from src.utils.config import load_config, check_data_fingerprint


class IDSDataset(Dataset):
    """
    Convertit les DataFrames pandas en tensors PyTorch à la volée.
    Les données sont stockées en RAM (déjà chargées), le dataset ne fait
    que les indexer et convertir.
    """

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
    """
    Charge les fichiers pickle produits par l'étape 4 (script 05) :
    train / validation / test.

    X_* : DataFrames pandas (to_pickle côté 05). y_* : arrays numpy
    (joblib.dump côté 05, quel que soit leur type d'origine) — chargement
    uniforme avec joblib.load, sans conversion conditionnelle a posteriori.
    """
    print("Chargement des données...")

    X_train = pd.read_pickle(data_dir / "X_train.pkl")
    X_val = pd.read_pickle(data_dir / "X_val.pkl")
    X_test = pd.read_pickle(data_dir / "X_test.pkl")
    y_train = joblib.load(data_dir / "y_train.pkl")
    y_val = joblib.load(data_dir / "y_val.pkl")
    y_test = joblib.load(data_dir / "y_test.pkl")

    print(f"  X_train : {X_train.shape} ({X_train.memory_usage(deep=True).sum() / 1e9:.2f} GB)")
    print(f"  X_val   : {X_val.shape}")
    print(f"  X_test  : {X_test.shape}")
    print(f"  y_train : {len(y_train):,} labels")
    print(f"  y_val   : {len(y_val):,} labels")
    print(f"  y_test  : {len(y_test):,} labels")
    print(f"  Classes : {len(np.unique(y_train))}")
    print()

    return X_train, X_val, X_test, y_train, y_val, y_test


def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size, num_workers):
    """
    Crée les DataLoaders PyTorch pour train, validation et test.

    Seul le train est shuffled à chaque epoch. Validation et test ne le
    sont pas, pour reproductibilité de l'évaluation.
    """
    print("Création des DataLoaders...")

    train_dataset = IDSDataset(X_train, y_train)
    val_dataset = IDSDataset(X_val, y_val)
    test_dataset = IDSDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"  Train : {len(train_loader):,} batches de {batch_size}")
    print(f"  Val   : {len(val_loader):,} batches de {batch_size}")
    print(f"  Test  : {len(test_loader):,} batches de {batch_size}")
    print()

    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Une passe complète sur les données d'entraînement."""
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
    """
    Évalue le modèle sur un dataset (validation ou test).
    Retourne loss moyenne, accuracy, et les prédictions complètes.
    """
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
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

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    return (
        total_loss / total,
        correct / total,
        np.array(all_predictions),
        np.array(all_labels),
    )


def train_model(model, train_loader, val_loader, train_cfg, device, checkpoint_dir,
                 config_hash, config_snapshot):
    """
    Boucle d'entraînement complète.

    config_hash (cfg.baseline_fingerprint()) est embarqué dans chaque
    checkpoint sauvegardé, ainsi que config_snapshot (cfg.baseline_fingerprint_data(),
    le dict qui a produit ce hash). 08_generate_attacks.py revalide le hash
    avant d'utiliser ce checkpoint comme cible d'évaluation, et affiche un
    diff clé par clé à partir du snapshot en cas de désaccord — deux hashes
    opaques ne disent pas QUOI a changé. baseline_best.pth est versionné
    dans git (négation explicite dans .gitignore) et peut donc arriver sur
    le cluster via un git pull sans qu'aucun entraînement n'ait eu lieu
    localement, ou avoir été entraîné sous une configuration différente
    (ancien scaler, anciens hyperparamètres).

    Le scheduler et la sélection du meilleur modèle se basent sur la
    VALIDATION (val_loss / val_acc). Le test set n'est pas passé à cette
    fonction : il n'est évalué nulle part avant final_evaluation, pas même
    pour affichage informatif. Regarder test_acc epoch par epoch reviendrait
    à surveiller le test pendant l'entraînement, ce qui pousse a ajuster
    des choix (epochs, lr, patience) en fonction de ce qu'on y voit — la
    fuite par l'humain que le split de validation existe pour eviter.
    """
    num_epochs = train_cfg["epochs"]
    learning_rate = train_cfg["learning_rate"]
    sched_cfg = train_cfg["scheduler"]

    print("Début de l'entraînement")
    print(f"  Device            : {device}")
    print(f"  Epochs            : {num_epochs}")
    print(f"  Learning rate init: {learning_rate}")
    print(
        f"  Scheduler         : ReduceLROnPlateau sur {sched_cfg['monitor']} "
        f"(factor={sched_cfg['factor']}, patience={sched_cfg['patience']})"
    )
    print(f"  Batch size        : {train_loader.batch_size}")
    print(f"  Sélection modèle  : {train_cfg['model_selection_metric']} (validation)")
    print()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=sched_cfg["factor"],
        patience=sched_cfg["patience"],
        min_lr=sched_cfg["min_lr"],
    )

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "learning_rate": [], "epoch_time": [],
    }

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        epoch_time = time.time() - start_time

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["learning_rate"].append(current_lr)
        history["epoch_time"].append(epoch_time)

        print(
            f"Epoch {epoch:2d}/{num_epochs} | lr={current_lr:.6f} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"time={epoch_time:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "learning_rate": current_lr,
                    "config_hash": config_hash,
                    "config_snapshot": config_snapshot,
                },
                checkpoint_dir / "baseline_best.pth",
            )
            print(f"           -> Nouveau meilleur modèle sauvegardé (val_acc={val_acc:.4f})")

        # Le scheduler surveille la métrique de VALIDATION, pas le test.
        scheduler.step(val_loss)

        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < current_lr:
            print(f"           -> Learning rate reduit : {current_lr:.6f} -> {new_lr:.6f}")

    print()
    print(f"Meilleur modèle : epoch {best_epoch}, val_acc = {best_val_acc:.4f}")
    print()

    return history


def final_evaluation(model, test_loader, device, label_encoder=None):
    """
    Évaluation finale sur le test, une seule fois, avec le modèle déjà
    figé par la sélection sur validation.
    """
    print("Évaluation finale sur le test set (jamais utilisé pour choisir le modèle)")

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, predictions, labels = evaluate(model, test_loader, criterion, device)

    print(f"\nAccuracy globale : {test_acc:.4f}")
    print(f"F1 macro         : {f1_score(labels, predictions, average='macro'):.4f}")
    print(f"F1 weighted      : {f1_score(labels, predictions, average='weighted'):.4f}")

    print("\nRapport par classe :")
    target_names = [str(c) for c in label_encoder.classes_] if label_encoder is not None else None

    print(classification_report(
        labels, predictions,
        target_names=target_names,
        digits=4,
        zero_division=0,
    ))

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


def main():
    print("Baseline DNN — CIC-IDS 2017")
    print(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    cfg = load_config()
    print(cfg.resume())
    print()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device utilisé : {device}")
    if device.type == "cuda":
        print(f"GPU            : {torch.cuda.get_device_name(0)}")
        print(f"CUDA version   : {torch.version.cuda}")
    print()

    data_dir = Path(cfg.paths["data_processed"])
    checkpoint_dir = Path(cfg.paths["checkpoints"])
    log_dir = Path(cfg.paths["logs"])

    # Verifie que data/processed/ correspond a la configuration courante,
    # AVANT tout chargement. Sans ca, un changement de dataset.val_size (par
    # exemple) sans relancer 05 passerait inapercu : ce script embarquerait
    # une empreinte "a jour" dans le checkpoint tout en entrainant sur des
    # donnees generees sous l'ancienne valeur.
    check_data_fingerprint(cfg, data_dir)

    X_train, X_val, X_test, y_train, y_val, y_test = load_data(data_dir)

    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        cfg.training["batch_size"], cfg.env["num_workers"],
    )

    print("Création du modèle...")
    model = BaselineDNN(
        input_dim=cfg.dataset["num_features"],
        hidden1=cfg.model["hidden_layers"][0],
        hidden2=cfg.model["hidden_layers"][1],
        output_dim=cfg.dataset["num_classes"],
    )
    model = model.to(device)
    print(f"  Paramètres : {count_parameters(model):,}")
    print()

    history = train_model(
        model, train_loader, val_loader, cfg.training, device, checkpoint_dir,
        config_hash=cfg.baseline_fingerprint(),
        config_snapshot=cfg.baseline_fingerprint_data(),
    )

    print("Chargement du meilleur modèle (sélectionné sur validation) pour évaluation finale...")
    checkpoint = torch.load(checkpoint_dir / "baseline_best.pth", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Meilleur epoch : {checkpoint['epoch']}")
    print(f"  Val accuracy   : {checkpoint['val_acc']:.4f}")
    print(f"  Learning rate  : {checkpoint['learning_rate']:.6f}")
    print()

    label_encoder = joblib.load(data_dir / "label_encoder.pkl")

    results = final_evaluation(model, test_loader, device, label_encoder)

    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    joblib.dump(
        {"history": history, "results": results},
        log_dir / f"baseline_training_{timestamp}.pkl",
    )
    print(f"Historique sauvegardé : {log_dir / f'baseline_training_{timestamp}.pkl'}")

    print()
    print("Baseline DNN terminé")


if __name__ == "__main__":
    main()
