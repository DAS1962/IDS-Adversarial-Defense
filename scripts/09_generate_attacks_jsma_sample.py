"""
Génération JSMA (échantillon stratifié) et C&W (full test set).

Complémentaire à 08_generate_attacks.py. Génère :
  - JSMA sur un échantillon stratifié (30 000 exemples par défaut)
  - C&W sur le test set complet

Attaques configurées en mode untargeted (y=None) : le but est de faire
prédire n'importe quelle classe incorrecte au modèle, pas une classe cible
précise. Passer les vrais labels rendrait l'attaque triviale puisque le
modèle prédit déjà ces classes correctement.

Note technique : patch np.product pour compatibilité NumPy 2.x avec ART 1.18.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

if not hasattr(np, "product"):
    np.product = np.prod

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from art.attacks.evasion import CarliniL2Method, SaliencyMapMethod
from art.estimators.classification import PyTorchClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.dnn import BaselineDNN


DATA_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("results/checkpoints")
LOG_DIR = Path("results/logs")
ATTACKS_DIR = Path("results/attacks")

BATCH_SIZE = 512
NUM_CLASSES = 15
INPUT_DIM = 58
RANDOM_STATE = 42
JSMA_SAMPLE_SIZE = 30_000

ATTACK_CONFIGS = {
    "JSMA": {"theta": 0.3, "gamma": 0.15},
    "CW": {"max_iter": 10, "confidence": 0.0},
}


def load_baseline_model(device):
    print("Chargement du baseline v4...")
    model = BaselineDNN(input_dim=INPUT_DIM, hidden1=512, hidden2=256, output_dim=NUM_CLASSES)
    checkpoint = torch.load(
        CHECKPOINT_DIR / "baseline_best.pth",
        weights_only=False,
        map_location=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"  Meilleur epoch : {checkpoint['epoch']}")
    print(f"  Test accuracy  : {checkpoint['test_acc']:.4f}\n")
    return model


def load_test_data():
    print("Chargement du test set...")
    X_test = pd.read_pickle(DATA_DIR / "X_test.pkl")
    y_test = pd.read_pickle(DATA_DIR / "y_test.pkl")
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values.astype(np.float32)
    if isinstance(y_test, pd.Series):
        y_test = y_test.values
    print(f"  X_test : {X_test.shape}")
    print(f"  y_test : {len(y_test):,} labels\n")
    return X_test, y_test


def create_art_classifier(model, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(INPUT_DIM,),
        nb_classes=NUM_CLASSES,
        device_type="gpu" if device.type == "cuda" else "cpu",
    )


def create_stratified_sample(X, y, sample_size, random_state):
    print(f"Création d'un échantillon stratifié de {sample_size:,} exemples...")
    _, X_sample, _, y_sample = train_test_split(
        X, y,
        test_size=sample_size,
        stratify=y,
        random_state=random_state,
    )
    print(f"  Échantillon : {len(X_sample):,} lignes")
    print(f"  Classes présentes : {len(np.unique(y_sample))}\n")
    print("  Distribution par classe dans l'echantillon :")
    unique, counts = np.unique(y_sample, return_counts=True)
    for cls, count in zip(unique, counts):
        pct = 100 * count / len(y_sample)
        print(f"    Classe {cls:2d} : {count:>6,} ({pct:.2f}%)")
    print()
    return X_sample.astype(np.float32), y_sample


def generate_art_attack(attack, X, y, name):
    """Génère les adversariaux. y=None pour attaque untargeted."""
    print(f"Génération {name}...")
    debut = time.time()
    X_np = X.astype(np.float32)
    if y is None:
        X_adv = attack.generate(x=X_np)
    else:
        X_adv = attack.generate(x=X_np, y=y)
    duree = time.time() - debut
    print(f"  {name} termine en {duree:.1f}s ({duree/60:.1f} min)\n")
    return X_adv


def evaluate_attack(model, X_adv, y_true, device, name):
    print(f"Évaluation sur {name}...")
    model.eval()
    n_samples = len(X_adv)
    predictions = np.zeros(n_samples, dtype=np.int64)
    with torch.no_grad():
        for i in range(0, n_samples, BATCH_SIZE):
            end = min(i + BATCH_SIZE, n_samples)
            X_batch = torch.tensor(X_adv[i:end], dtype=torch.float32).to(device)
            outputs = model(X_batch)
            _, preds = outputs.max(1)
            predictions[i:end] = preds.cpu().numpy()

    acc = accuracy_score(y_true, predictions)
    precision_macro = precision_score(y_true, predictions, average="macro", zero_division=0)
    precision_weighted = precision_score(y_true, predictions, average="weighted", zero_division=0)
    recall_macro = recall_score(y_true, predictions, average="macro", zero_division=0)
    recall_weighted = recall_score(y_true, predictions, average="weighted", zero_division=0)
    f1_macro = f1_score(y_true, predictions, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, predictions, average="weighted", zero_division=0)

    print(f"  Accuracy               : {acc:.4f}")
    print(f"  Precision (macro)      : {precision_macro:.4f}")
    print(f"  Precision (weighted)   : {precision_weighted:.4f}")
    print(f"  Recall (macro)         : {recall_macro:.4f}")
    print(f"  Recall (weighted)      : {recall_weighted:.4f}")
    print(f"  F1 (macro)             : {f1_macro:.4f}")
    print(f"  F1 (weighted)          : {f1_weighted:.4f}")

    label_encoder = joblib.load(DATA_DIR / "label_encoder.pkl")
    all_labels = list(range(NUM_CLASSES))
    target_names = [str(c) for c in label_encoder.classes_]

    print(f"\n  Rapport par classe pour {name} :")
    print(classification_report(
        y_true, predictions,
        labels=all_labels,
        target_names=target_names,
        digits=4,
        zero_division=0,
    ))

    cm = confusion_matrix(y_true, predictions, labels=all_labels)

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
        "confusion_matrix": cm,
    }


def main():
    print("Génération JSMA (échantillon) et C&W (full test set)")
    print(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}\n")

    model = load_baseline_model(device)
    X_test, y_test = load_test_data()
    art_classifier = create_art_classifier(model, device)

    all_results = []
    ATTACKS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Partie 1 : JSMA sur échantillon stratifié {JSMA_SAMPLE_SIZE:,}")
    print("=" * 70 + "\n")

    X_jsma_sample, y_jsma_sample = create_stratified_sample(
        X_test, y_test, JSMA_SAMPLE_SIZE, RANDOM_STATE
    )

    attack = SaliencyMapMethod(
        classifier=art_classifier,
        theta=ATTACK_CONFIGS["JSMA"]["theta"],
        gamma=ATTACK_CONFIGS["JSMA"]["gamma"],
        batch_size=BATCH_SIZE,
    )
    X_adv_jsma = generate_art_attack(attack, X_jsma_sample, None, "JSMA")

    joblib.dump(X_adv_jsma, ATTACKS_DIR / "X_adv_jsma_sample30k.pkl")
    joblib.dump(y_jsma_sample, ATTACKS_DIR / "y_jsma_sample30k.pkl")
    print(f"  Sauvegarde : X_adv_jsma_sample30k.pkl")
    print(f"  Sauvegarde : y_jsma_sample30k.pkl\n")

    result = evaluate_attack(model, X_adv_jsma, y_jsma_sample, device, "JSMA (30k sample)")
    all_results.append(result)
    del X_adv_jsma

    print("=" * 70)
    print("Partie 2 : C&W sur test set complet")
    print("=" * 70 + "\n")

    attack = CarliniL2Method(
        classifier=art_classifier,
        max_iter=ATTACK_CONFIGS["CW"]["max_iter"],
        confidence=ATTACK_CONFIGS["CW"]["confidence"],
        batch_size=BATCH_SIZE,
    )
    X_adv_cw = generate_art_attack(attack, X_test, None, "C&W")
    joblib.dump(X_adv_cw, ATTACKS_DIR / "X_adv_cw.pkl")
    taille_mb = (ATTACKS_DIR / "X_adv_cw.pkl").stat().st_size / (1024 * 1024)
    print(f"  Sauvegarde : X_adv_cw.pkl ({taille_mb:.1f} MB)\n")

    result = evaluate_attack(model, X_adv_cw, y_test, device, "C&W")
    all_results.append(result)
    del X_adv_cw

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    joblib.dump(
        {
            "results": all_results,
            "attack_configs": ATTACK_CONFIGS,
            "jsma_sample_size": JSMA_SAMPLE_SIZE,
        },
        LOG_DIR / f"jsma_cw_results_{timestamp}.pkl",
    )
    print(f"\nRésultats sauvegardés : {LOG_DIR / f'jsma_cw_results_{timestamp}.pkl'}")
    print("\nGénération terminée")


if __name__ == "__main__":
    main()
