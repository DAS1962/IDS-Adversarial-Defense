"""
Défense par Denoising Autoencoder (DAE).

On entraîne un autoencodeur à reconstruire les données propres à partir
de versions bruitées (x + N(0, sigma^2)). Au moment de la prédiction,
on passe d'abord chaque input à travers le DAE pour le "purifier", puis
on le classifie avec le baseline v4 déjà entraîné.

Le classifieur (baseline) n'est pas ré-entraîné : seul le DAE est nouveau.

Architecture DAE : 58 -> 32 -> 58 (bottleneck de 32 features).

Référence : Vincent et al. 2008, Gu & Rigazio 2014, Awad et al. 2025.
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

# Hyperparamètres de l'entraînement du DAE
EPOCHS = 50
BATCH_SIZE = 128
LR_INIT = 0.001
LR_MIN = 1e-5
LR_PATIENCE = 5
LR_FACTOR = 0.5
RANDOM_STATE = 42

# Bruit ajouté aux inputs du DAE pendant l'entraînement
DAE_NOISE_SIGMA = 0.1

# Régularisation L1 sur les activations du bottleneck
L1_LAMBDA = 1e-5

# Architecture
NUM_CLASSES = 15
INPUT_DIM = 58
BOTTLENECK_DIM = 32

# Fichiers X_adv pour l'évaluation finale
ATTACK_FILES = [
    ("FGSM", "X_adv_fgsm.pkl"),
    ("BIM", "X_adv_bim.pkl"),
    ("PGD", "X_adv_pgd.pkl"),
    ("DeepFool", "X_adv_deepfool.pkl"),
    ("JSMA", "X_adv_jsma.pkl"),
    ("CW", "X_adv_cw.pkl"),
]


class DenoisingAutoencoder(nn.Module):
    """
    Autoencodeur simple pour débruiter les features.

    Encoder : 58 -> 32 (compression)
    Decoder : 32 -> 58 (reconstruction)
    """

    def __init__(self, input_dim=58, bottleneck_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


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


def make_dae_loaders(X_train, X_test):
    """
    Crée les DataLoaders pour l'entraînement du DAE.

    Le DAE est en apprentissage non-supervisé : on lui donne x en entrée
    et il doit reconstruire x en sortie. Pas besoin des labels y.
    """
    train_ds = TensorDataset(torch.from_numpy(X_train).float())
    test_ds = TensorDataset(torch.from_numpy(X_test).float())

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
    """Ajoute du bruit gaussien N(0, sigma^2)."""
    noise = torch.randn_like(x) * sigma
    return x + noise


def train_dae_epoch(dae, loader, optimizer, criterion, device, sigma, l1_lambda):
    """
    Entraîne le DAE une epoch.

    Pour chaque batch :
      1. On bruite les inputs : x_noisy = x + noise
      2. Le DAE tente de reconstruire x propre à partir de x_noisy
      3. Loss = MSE(x_reconstruit, x_propre) + L1(activations bottleneck)
    """
    dae.train()
    total_loss = 0.0
    total_mse = 0.0
    total_seen = 0

    for (x,) in loader:
        x = x.to(device, non_blocking=True)

        # On bruite les inputs mais on veut reconstruire x propre
        x_noisy = add_gaussian_noise(x, sigma)

        optimizer.zero_grad()
        x_reconstructed, encoded = dae(x_noisy)

        # Loss principale : MSE entre reconstruction et x propre
        mse_loss = criterion(x_reconstructed, x)

        # Régularisation L1 sur le bottleneck (sparsité)
        l1_loss = l1_lambda * torch.abs(encoded).sum()

        total = mse_loss + l1_loss
        total.backward()
        optimizer.step()

        total_loss += total.item() * x.size(0)
        total_mse += mse_loss.item() * x.size(0)
        total_seen += x.size(0)

    return total_loss / total_seen, total_mse / total_seen


def evaluate_dae(dae, loader, criterion, device):
    """Évalue le DAE (sans bruit) : mesure la reconstruction sur x propre."""
    dae.eval()
    total_mse = 0.0
    total_seen = 0

    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device, non_blocking=True)
            x_reconstructed, _ = dae(x)
            mse = criterion(x_reconstructed, x)
            total_mse += mse.item() * x.size(0)
            total_seen += x.size(0)

    return total_mse / total_seen


def load_baseline_classifier(device):
    """Charge le baseline v4 déjà entraîné."""
    print("Chargement du baseline v4...")
    model = BaselineDNN(input_dim=INPUT_DIM, hidden1=512, hidden2=256, output_dim=NUM_CLASSES)
    checkpoint = torch.load(
        CHECKPOINT_DIR / "baseline_best.pth",
        weights_only=False, map_location=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()
    print(f"  Meilleur epoch : {checkpoint['epoch']}")
    print(f"  Test accuracy  : {checkpoint['test_acc']:.4f}")
    print()
    return model


def predict_through_dae(dae, classifier, X, device):
    """
    Prédit en passant d'abord par le DAE puis par le classifier.

    Pipeline : x -> DAE -> x_clean -> classifier -> prédiction
    """
    dae.eval()
    classifier.eval()
    n = len(X)
    preds = np.zeros(n, dtype=np.int64)

    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            end = min(i + BATCH_SIZE, n)
            xb = torch.tensor(X[i:end], dtype=torch.float32).to(device)

            # 1. Purification via le DAE
            x_clean, _ = dae(xb)

            # 2. Classification via le baseline
            logits = classifier(x_clean)
            preds[i:end] = logits.argmax(dim=1).cpu().numpy()

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
    print("Résumé - Défense Denoising Autoencoder")
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
    print("Défense par Denoising Autoencoder (DAE)")
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
    train_loader, test_loader = make_dae_loaders(X_train, X_test)
    print(f"Batches train : {len(train_loader):,}")
    print(f"Batches test  : {len(test_loader):,}")
    print()

    # Modèle DAE
    print("Création du Denoising Autoencoder...")
    dae = DenoisingAutoencoder(input_dim=INPUT_DIM, bottleneck_dim=BOTTLENECK_DIM)
    dae = dae.to(device)
    n_params = sum(p.numel() for p in dae.parameters())
    print(f"  Architecture : {INPUT_DIM} -> {BOTTLENECK_DIM} -> {INPUT_DIM}")
    print(f"  Paramètres   : {n_params:,}")
    print()

    # Chargement du baseline pour évaluation
    classifier = load_baseline_classifier(device)

    # Optimizer et scheduler
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(dae.parameters(), lr=LR_INIT)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR,
        patience=LR_PATIENCE, min_lr=LR_MIN,
    )

    # Entraînement du DAE
    print("Début de l'entraînement du Denoising Autoencoder")
    print(f"  Epochs            : {EPOCHS}")
    print(f"  Learning rate init: {LR_INIT}")
    print(f"  Bruit sigma       : {DAE_NOISE_SIGMA}")
    print(f"  L1 lambda         : {L1_LAMBDA}")
    print(f"  Batch size        : {BATCH_SIZE}")
    print()

    history = {"train_loss": [], "train_mse": [], "test_mse": [], "lr": []}
    best_mse = float("inf")
    best_epoch = -1
    ckpt_path = CHECKPOINT_DIR / "defense_dae_best.pth"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    start_train = time.time()

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_mse = train_dae_epoch(
            dae, train_loader, optimizer, criterion, device,
            DAE_NOISE_SIGMA, L1_LAMBDA,
        )
        test_mse = evaluate_dae(dae, test_loader, criterion, device)
        scheduler.step(test_mse)
        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_mse"].append(train_mse)
        history["test_mse"].append(test_mse)
        history["lr"].append(lr_now)

        print(
            f"Epoch {epoch:2d}/{EPOCHS} | lr={lr_now:.6f} | "
            f"train_loss={train_loss:.6f} train_mse={train_mse:.6f} | "
            f"test_mse={test_mse:.6f} | time={elapsed:.1f}s",
            flush=True,
        )

        if test_mse < best_mse:
            best_mse = test_mse
            best_epoch = epoch
            torch.save({
                "model_state_dict": dae.state_dict(),
                "epoch": epoch,
                "test_mse": test_mse,
            }, ckpt_path)
            print(f"           -> Nouveau meilleur DAE sauvegardé (test_mse={test_mse:.6f})", flush=True)

    train_time_min = (time.time() - start_train) / 60
    print()
    print(f"Entraînement terminé en {train_time_min:.1f} min")
    print(f"Meilleur epoch : {best_epoch} | test_mse = {best_mse:.6f}")
    print()

    # Chargement du meilleur DAE
    print("Chargement du meilleur DAE pour évaluation...")
    checkpoint = torch.load(ckpt_path, weights_only=False, map_location=device)
    dae.load_state_dict(checkpoint["model_state_dict"])
    dae.eval()

    # Évaluation : x -> DAE -> classifier
    print()
    print("=" * 70)
    print("Évaluation sur données propres et attaques adversariales")
    print("Pipeline : x -> DAE (purification) -> Baseline v4 -> prédiction")
    print("=" * 70)

    all_results = []

    print("\n--- Données propres ---")
    y_pred_clean = predict_through_dae(dae, classifier, X_test, device)
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
        y_pred = predict_through_dae(dae, classifier, X_adv, device)
        m = compute_metrics(y_test, y_pred, name)
        print_metrics(m)
        all_results.append(m)
        del X_adv

    print_summary(all_results)

    # Sauvegarde des résultats
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = LOG_DIR / f"defense_dae_{timestamp}.pkl"
    joblib.dump({
        "results": all_results,
        "history": history,
        "hyperparameters": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr_init": LR_INIT,
            "dae_noise_sigma": DAE_NOISE_SIGMA,
            "l1_lambda": L1_LAMBDA,
            "bottleneck_dim": BOTTLENECK_DIM,
            "seed": RANDOM_STATE,
        },
        "best_epoch": best_epoch,
        "best_test_mse": best_mse,
        "training_time_min": train_time_min,
    }, out_path)
    print(f"\nRésultats sauvegardés : {out_path}")
    print("\nDéfense Denoising Autoencoder terminée")


if __name__ == "__main__":
    main()
