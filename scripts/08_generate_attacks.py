"""
Génération des 6 attaques adversariales sur le baseline v4 (test set complet).

Attaques implémentées (référence : Awad et al. 2025) :
  - FGSM (Goodfellow 2014)     : single-step, L∞, eps=0.2
  - BIM (Kurakin 2016)         : iterative, L∞, eps=0.3, alpha=0.01, 100 iter
  - PGD (Madry 2017)           : iterative + random init, L∞, eps=0.3, 100 iter
  - DeepFool (Moosavi 2015)    : iterative, L2, 100 iter
  - JSMA (Papernot 2015)       : feature-based, L0, theta=0.3, gamma=0.15, untargeted
  - C&W (Carlini & Wagner 2016): optimization-based, L2, 10 iter, untargeted

JSMA et C&W sont configurés en mode untargeted (y=None) : passer les vrais
labels rendrait l'attaque triviale puisque le modèle prédit déjà correctement.

Pour chaque attaque :
  1. Génération sur tout le test set (skip si X_adv_*.pkl existe déjà)
  2. Sauvegarde des exemples adversariaux en pickle
  3. Évaluation baseline (accuracy, precision, recall, F1)
  4. Sauvegarde des métriques agrégées

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

import torchattacks

from art.attacks.evasion import CarliniL2Method, DeepFool, SaliencyMapMethod
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

ATTACK_CONFIGS = {
    "FGSM": {"eps": 0.2},
    "BIM": {"eps": 0.3, "alpha": 0.01, "steps": 100},
    "PGD": {"eps": 0.3, "alpha": 0.01, "steps": 100},
    "DeepFool": {"max_iter": 100, "epsilon": 1e-6},
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
    model = model.to(device).eval()
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


def load_or_none(name):
    """Charge un X_adv déjà généré, retourne None si absent."""
    path = ATTACKS_DIR / f"X_adv_{name.lower()}.pkl"
    if path.exists():
        print(f"  Fichier {path.name} déjà présent, chargement...")
        X_adv = joblib.load(path)
        taille_mb = path.stat().st_size / (1024 * 1024)
        print(f"  Chargé : {X_adv.shape} ({taille_mb:.1f} MB)\n")
        return X_adv
    return None


def generate_torchattacks_batch(attack, X_batch, y_batch, device):
    x_tensor = torch.tensor(X_batch, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_batch, dtype=torch.long).to(device)
    adv = attack(x_tensor, y_tensor)
    return adv.cpu().numpy()


def generate_torchattacks(attack, X, y, device, name):
    """Génère les adversariaux avec torchattacks (FGSM, BIM, PGD), par batchs."""
    print(f"Génération {name}...")
    debut = time.time()
    n_samples = len(X)
    X_adv = np.zeros_like(X)
    n_batches = (n_samples + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(n_batches):
        start = i * BATCH_SIZE
        end = min(start + BATCH_SIZE, n_samples)
        X_adv[start:end] = generate_torchattacks_batch(
            attack, X[start:end], y[start:end], device
        )
        if (i + 1) % 100 == 0 or (i + 1) == n_batches:
            pct = 100 * (i + 1) / n_batches
            elapsed = time.time() - debut
            print(f"  Batch {i+1}/{n_batches} ({pct:.1f}%) - {elapsed:.1f}s", flush=True)

    duree = time.time() - debut
    print(f"  {name} termine en {duree:.1f}s ({duree/60:.1f} min)\n")
    return X_adv


def generate_art_attack(attack, X, y, name):
    """Génère les adversariaux avec ART. y=None pour untargeted."""
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


def save_adversarial_examples(X_adv, name, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"X_adv_{name.lower()}.pkl"
    joblib.dump(X_adv, path)
    taille_mb = path.stat().st_size / (1024 * 1024)
    print(f"  Sauvegardé : {path.name} ({taille_mb:.1f} MB)\n")


def print_summary_table(all_results, baseline_metrics):
    print("\n" + "=" * 100)
    print("Résumé complet des attaques adversariales")
    print("=" * 100)
    header = (
        f"{'Attaque':<12} {'Accuracy':>10} {'Prec. macro':>12} {'Prec. wght':>12} "
        f"{'Rec. macro':>11} {'Rec. wght':>11} {'F1 macro':>10} {'F1 wght':>10}"
    )
    print(header)
    print("-" * 100)
    print(
        f"{'BENIGN':<12} {baseline_metrics['accuracy']:>10.4f} "
        f"{baseline_metrics['precision_macro']:>12.4f} {baseline_metrics['precision_weighted']:>12.4f} "
        f"{baseline_metrics['recall_macro']:>11.4f} {baseline_metrics['recall_weighted']:>11.4f} "
        f"{baseline_metrics['f1_macro']:>10.4f} {baseline_metrics['f1_weighted']:>10.4f}  (baseline)"
    )
    print("-" * 100)
    for r in all_results:
        print(
            f"{r['attack']:<12} {r['accuracy']:>10.4f} "
            f"{r['precision_macro']:>12.4f} {r['precision_weighted']:>12.4f} "
            f"{r['recall_macro']:>11.4f} {r['recall_weighted']:>11.4f} "
            f"{r['f1_macro']:>10.4f} {r['f1_weighted']:>10.4f}"
        )
    print("=" * 100)


def evaluate_baseline_on_clean(model, X_test, y_test, device):
    print("Évaluation du baseline sur données propres (référence)...")
    result = evaluate_attack(model, X_test, y_test, device, "BENIGN (clean)")
    return {
        "accuracy": result["accuracy"],
        "precision_macro": result["precision_macro"],
        "precision_weighted": result["precision_weighted"],
        "recall_macro": result["recall_macro"],
        "recall_weighted": result["recall_weighted"],
        "f1_macro": result["f1_macro"],
        "f1_weighted": result["f1_weighted"],
    }


def main():
    print("Génération des attaques adversariales sur le baseline v4 (full test set)")
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

    baseline_metrics = evaluate_baseline_on_clean(model, X_test, y_test, device)
    all_results = []

    print("=" * 70)
    print("1. FGSM")
    print("=" * 70 + "\n")
    X_adv_fgsm = load_or_none("FGSM")
    if X_adv_fgsm is None:
        attack = torchattacks.FGSM(model, eps=ATTACK_CONFIGS["FGSM"]["eps"])
        X_adv_fgsm = generate_torchattacks(attack, X_test, y_test, device, "FGSM")
        save_adversarial_examples(X_adv_fgsm, "FGSM", ATTACKS_DIR)
    result = evaluate_attack(model, X_adv_fgsm, y_test, device, "FGSM")
    all_results.append(result)
    del X_adv_fgsm

    print("=" * 70)
    print("2. BIM")
    print("=" * 70 + "\n")
    X_adv_bim = load_or_none("BIM")
    if X_adv_bim is None:
        attack = torchattacks.BIM(
            model,
            eps=ATTACK_CONFIGS["BIM"]["eps"],
            alpha=ATTACK_CONFIGS["BIM"]["alpha"],
            steps=ATTACK_CONFIGS["BIM"]["steps"],
        )
        X_adv_bim = generate_torchattacks(attack, X_test, y_test, device, "BIM")
        save_adversarial_examples(X_adv_bim, "BIM", ATTACKS_DIR)
    result = evaluate_attack(model, X_adv_bim, y_test, device, "BIM")
    all_results.append(result)
    del X_adv_bim

    print("=" * 70)
    print("3. PGD")
    print("=" * 70 + "\n")
    X_adv_pgd = load_or_none("PGD")
    if X_adv_pgd is None:
        attack = torchattacks.PGD(
            model,
            eps=ATTACK_CONFIGS["PGD"]["eps"],
            alpha=ATTACK_CONFIGS["PGD"]["alpha"],
            steps=ATTACK_CONFIGS["PGD"]["steps"],
        )
        X_adv_pgd = generate_torchattacks(attack, X_test, y_test, device, "PGD")
        save_adversarial_examples(X_adv_pgd, "PGD", ATTACKS_DIR)
    result = evaluate_attack(model, X_adv_pgd, y_test, device, "PGD")
    all_results.append(result)
    del X_adv_pgd

    print("=" * 70)
    print("4. DeepFool")
    print("=" * 70 + "\n")
    X_adv_df = load_or_none("DeepFool")
    if X_adv_df is None:
        attack = DeepFool(
            classifier=art_classifier,
            max_iter=ATTACK_CONFIGS["DeepFool"]["max_iter"],
            epsilon=ATTACK_CONFIGS["DeepFool"]["epsilon"],
            batch_size=BATCH_SIZE,
        )
        X_adv_df = generate_art_attack(attack, X_test, y_test, "DeepFool")
        save_adversarial_examples(X_adv_df, "DeepFool", ATTACKS_DIR)
    result = evaluate_attack(model, X_adv_df, y_test, device, "DeepFool")
    all_results.append(result)
    del X_adv_df

    print("=" * 70)
    print("5. JSMA (untargeted, full test set)")
    print("=" * 70 + "\n")
    X_adv_jsma = load_or_none("JSMA")
    if X_adv_jsma is None:
        attack = SaliencyMapMethod(
            classifier=art_classifier,
            theta=ATTACK_CONFIGS["JSMA"]["theta"],
            gamma=ATTACK_CONFIGS["JSMA"]["gamma"],
            batch_size=BATCH_SIZE,
        )
        X_adv_jsma = generate_art_attack(attack, X_test, None, "JSMA")
        save_adversarial_examples(X_adv_jsma, "JSMA", ATTACKS_DIR)
    result = evaluate_attack(model, X_adv_jsma, y_test, device, "JSMA")
    all_results.append(result)
    del X_adv_jsma

    print("=" * 70)
    print("6. C&W (untargeted, full test set)")
    print("=" * 70 + "\n")
    X_adv_cw = load_or_none("CW")
    if X_adv_cw is None:
        attack = CarliniL2Method(
            classifier=art_classifier,
            max_iter=ATTACK_CONFIGS["CW"]["max_iter"],
            confidence=ATTACK_CONFIGS["CW"]["confidence"],
            batch_size=BATCH_SIZE,
        )
        X_adv_cw = generate_art_attack(attack, X_test, None, "CW")
        save_adversarial_examples(X_adv_cw, "CW", ATTACKS_DIR)
    result = evaluate_attack(model, X_adv_cw, y_test, device, "CW")
    all_results.append(result)
    del X_adv_cw

    print_summary_table(all_results, baseline_metrics)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    joblib.dump(
        {
            "results": all_results,
            "baseline_metrics": baseline_metrics,
            "attack_configs": ATTACK_CONFIGS,
        },
        LOG_DIR / f"attacks_results_{timestamp}.pkl",
    )
    print(f"\nRésultats sauvegardés : {LOG_DIR / f'attacks_results_{timestamp}.pkl'}")
    print("\nGénération et évaluation des attaques terminées")


if __name__ == "__main__":
    main()
