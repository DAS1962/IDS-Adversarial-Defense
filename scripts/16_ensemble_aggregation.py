"""
Agrégation ensemble des 4 défenses (LS, AT, GA, DAE).

Le principe : chaque défense donne ses prédictions sur les données propres
et sur chaque attaque. On combine ces prédictions de 3 façons différentes
pour obtenir un ensemble plus robuste que chaque défense prise séparément.

Méthodes d'agrégation :
  1. Majority Voting : chaque modèle vote pour une classe, la classe
     majoritaire est retenue.
  2. Weighted Average simple : moyenne des probabilités softmax avec
     poids égaux (1/4 pour chaque défense).
  3. Weighted Average optimisé : les poids sont optimisés par recherche
     bayésienne pour maximiser l'accuracy sur les attaques.

Référence : Awad et al. 2025.
"""

import sys
import time
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from scipy.optimize import minimize

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

DEFENSE_CHECKPOINTS = {
    "LS":  "defense_ls_best.pth",
    "AT":  "defense_at_best.pth",
    "GA":  "defense_ga_best.pth",
    "DAE": "defense_dae_best.pth",
}

ATTACK_FILES = [
    ("FGSM", "X_adv_fgsm.pkl"),
    ("BIM", "X_adv_bim.pkl"),
    ("PGD", "X_adv_pgd.pkl"),
    ("DeepFool", "X_adv_deepfool.pkl"),
    ("JSMA", "X_adv_jsma.pkl"),
    ("CW", "X_adv_cw.pkl"),
]


class DenoisingAutoencoder(nn.Module):
    """Autoencodeur pour la défense DAE (doit correspondre au script 14)."""

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
    """Charge X_test et y_test."""
    print("Chargement des données...")
    X_test = pd.read_pickle(DATA_DIR / "X_test.pkl")
    y_test = pd.read_pickle(DATA_DIR / "y_test.pkl")

    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values.astype(np.float32)
    if isinstance(y_test, pd.Series):
        y_test = y_test.values

    print(f"  X_test : {X_test.shape}")
    print()
    return X_test, y_test


def load_defense_model(name, checkpoint_file, device):
    """Charge un modèle de défense entraîné."""
    ckpt_path = CHECKPOINT_DIR / checkpoint_file
    if not ckpt_path.exists():
        return None

    if name == "DAE":
        model = DenoisingAutoencoder(input_dim=INPUT_DIM, bottleneck_dim=32)
    else:
        model = BaselineDNN(input_dim=INPUT_DIM, hidden1=512, hidden2=256, output_dim=NUM_CLASSES)

    checkpoint = torch.load(ckpt_path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()
    return model


def load_baseline(device):
    """Charge le baseline v4 (nécessaire pour le DAE)."""
    model = BaselineDNN(input_dim=INPUT_DIM, hidden1=512, hidden2=256, output_dim=NUM_CLASSES)
    checkpoint = torch.load(
        CHECKPOINT_DIR / "baseline_best.pth",
        weights_only=False, map_location=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()
    return model


def get_probabilities(model, X, device, dae=None, baseline=None):
    """
    Retourne les probabilités softmax d'un modèle sur un array numpy.

    Pour le DAE, le pipeline est : x -> DAE -> baseline -> softmax
    Pour les autres défenses : x -> model -> softmax
    """
    model.eval()
    n = len(X)
    probs = np.zeros((n, NUM_CLASSES), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            end = min(i + BATCH_SIZE, n)
            xb = torch.tensor(X[i:end], dtype=torch.float32).to(device)

            if dae is not None and baseline is not None:
                x_clean, _ = dae(xb)
                logits = baseline(x_clean)
            else:
                logits = model(xb)

            batch_probs = F.softmax(logits, dim=1)
            probs[i:end] = batch_probs.cpu().numpy()

    return probs


def majority_voting(predictions_list):
    """
    Agrège les prédictions par vote majoritaire.

    predictions_list : liste de tableaux (n_samples,) contenant les
                       prédictions de chaque modèle.
    """
    stacked = np.stack(predictions_list, axis=0)
    n_samples = stacked.shape[1]
    result = np.zeros(n_samples, dtype=np.int64)

    for i in range(n_samples):
        counts = np.bincount(stacked[:, i], minlength=NUM_CLASSES)
        result[i] = counts.argmax()

    return result


def weighted_average(probs_list, weights):
    """
    Agrège les probabilités par moyenne pondérée puis argmax.

    probs_list : liste de tableaux (n_samples, num_classes) de probabilités.
    weights    : liste de poids (un par modèle), somme = 1.
    """
    weighted = np.zeros_like(probs_list[0])
    for probs, w in zip(probs_list, weights):
        weighted += w * probs
    return weighted.argmax(axis=1)


def optimize_weights(probs_list_by_attack, y_true_by_attack, n_models):
    """
    Optimise les poids de l'ensemble Weighted Average.

    On maximise la somme des accuracies sur toutes les attaques
    (l'accuracy sur clean est un bonus).

    Contrainte : les poids sont positifs et somment à 1.
    """
    def neg_score(weights):
        weights = np.abs(weights)
        weights = weights / weights.sum()
        total_acc = 0.0
        for attack_name, probs_list in probs_list_by_attack.items():
            y_true = y_true_by_attack[attack_name]
            y_pred = weighted_average(probs_list, weights)
            total_acc += accuracy_score(y_true, y_pred)
        return -total_acc

    initial_weights = np.ones(n_models) / n_models
    result = minimize(
        neg_score, initial_weights,
        method="Nelder-Mead",
        options={"maxiter": 500, "xatol": 1e-4, "fatol": 1e-4},
    )

    final_weights = np.abs(result.x)
    final_weights = final_weights / final_weights.sum()
    return final_weights, -result.fun


def compute_metrics(y_true, y_pred, name):
    """Calcule les métriques standards."""
    return {
        "attack": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
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


def print_summary(all_results, method_name):
    """Affiche le tableau récapitulatif d'une méthode d'agrégation."""
    print("\n" + "=" * 100)
    print(f"Résumé - Ensemble {method_name}")
    print("=" * 100)
    header = (
        f"{'Attaque':<12} {'Accuracy':>10} {'Prec. macro':>12} "
        f"{'Rec. macro':>11} {'F1 macro':>10} {'F1 wght':>10}"
    )
    print(header)
    print("-" * 100)
    for r in all_results:
        print(
            f"{r['attack']:<12} {r['accuracy']:>10.4f} "
            f"{r['precision_macro']:>12.4f} {r['recall_macro']:>11.4f} "
            f"{r['f1_macro']:>10.4f} {r['f1_weighted']:>10.4f}"
        )
    print("=" * 100)


def print_final_comparison(results_by_method):
    """Compare les 3 méthodes d'ensemble côte à côte."""
    print("\n" + "=" * 120)
    print("Comparaison finale des 3 méthodes d'ensemble (Accuracy)")
    print("=" * 120)

    attacks = [r["attack"] for r in results_by_method["Majority Voting"]]
    header = f"{'Attaque':<12}"
    for method in results_by_method.keys():
        header += f" | {method:>25}"
    print(header)
    print("-" * 120)

    for i, attack in enumerate(attacks):
        row = f"{attack:<12}"
        for method, results in results_by_method.items():
            row += f" | {results[i]['accuracy']:>25.4f}"
        print(row)
    print("=" * 120)


def main():
    print("=" * 70)
    print("Ensemble Aggregation des 4 défenses")
    print(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print()

    X_test, y_test = load_data()

    print("Chargement des 4 modèles de défense...")
    baseline = load_baseline(device)
    defense_models = {}
    for name, ckpt_file in DEFENSE_CHECKPOINTS.items():
        model = load_defense_model(name, ckpt_file, device)
        if model is None:
            print(f"  [ERREUR] {name} : checkpoint {ckpt_file} introuvable")
            print(f"  Le script ne peut pas continuer sans les 4 défenses entraînées.")
            return
        defense_models[name] = model
        print(f"  {name:<4} : {ckpt_file} chargé")
    print()

    defense_names = list(DEFENSE_CHECKPOINTS.keys())
    n_models = len(defense_names)

    print("Calcul des probabilités de chaque défense sur les données propres et attaques...")
    print()

    probs_by_attack = {}
    y_true_by_attack = {"Clean": y_test}

    print("  -> Clean")
    probs_clean = []
    for name in defense_names:
        model = defense_models[name]
        if name == "DAE":
            probs = get_probabilities(model, X_test, device, dae=model, baseline=baseline)
        else:
            probs = get_probabilities(model, X_test, device)
        probs_clean.append(probs)
    probs_by_attack["Clean"] = probs_clean

    for attack_name, x_file in ATTACK_FILES:
        x_path = ATTACKS_DIR / x_file
        if not x_path.exists():
            print(f"  [SKIP] {attack_name} : {x_file} introuvable")
            continue

        print(f"  -> {attack_name}")
        X_adv = joblib.load(x_path)
        probs_list = []
        for name in defense_names:
            model = defense_models[name]
            if name == "DAE":
                probs = get_probabilities(model, X_adv, device, dae=model, baseline=baseline)
            else:
                probs = get_probabilities(model, X_adv, device)
            probs_list.append(probs)
        probs_by_attack[attack_name] = probs_list
        y_true_by_attack[attack_name] = y_test
        del X_adv

    print()

    attack_names_ordered = list(probs_by_attack.keys())

    print("=" * 70)
    print("Méthode 1 : Majority Voting")
    print("=" * 70)
    results_mv = []
    for attack_name in attack_names_ordered:
        probs_list = probs_by_attack[attack_name]
        preds_list = [p.argmax(axis=1) for p in probs_list]
        y_pred = majority_voting(preds_list)
        m = compute_metrics(y_true_by_attack[attack_name], y_pred, attack_name)
        print(f"\n--- {attack_name} ---")
        print_metrics(m)
        results_mv.append(m)
    print_summary(results_mv, "Majority Voting")

    print()
    print("=" * 70)
    print("Méthode 2 : Weighted Average (poids égaux 1/4)")
    print("=" * 70)
    equal_weights = np.ones(n_models) / n_models
    print(f"\nPoids utilisés : {dict(zip(defense_names, equal_weights))}")

    results_wa = []
    for attack_name in attack_names_ordered:
        probs_list = probs_by_attack[attack_name]
        y_pred = weighted_average(probs_list, equal_weights)
        m = compute_metrics(y_true_by_attack[attack_name], y_pred, attack_name)
        print(f"\n--- {attack_name} ---")
        print_metrics(m)
        results_wa.append(m)
    print_summary(results_wa, "Weighted Average (égal)")

    print()
    print("=" * 70)
    print("Méthode 3 : Weighted Average optimisé (Nelder-Mead)")
    print("=" * 70)
    print("\nOptimisation des poids sur les attaques uniquement...")

    attack_only_probs = {k: v for k, v in probs_by_attack.items() if k != "Clean"}
    attack_only_labels = {k: v for k, v in y_true_by_attack.items() if k != "Clean"}

    start_opt = time.time()
    optimal_weights, best_score = optimize_weights(
        attack_only_probs, attack_only_labels, n_models,
    )
    opt_time = time.time() - start_opt

    print(f"  Temps d'optimisation : {opt_time:.1f}s")
    print(f"  Poids optimaux :")
    for name, w in zip(defense_names, optimal_weights):
        print(f"    {name:<4} : {w:.4f}")
    print(f"  Somme accuracies attaques : {best_score:.4f}")

    results_wa_opt = []
    for attack_name in attack_names_ordered:
        probs_list = probs_by_attack[attack_name]
        y_pred = weighted_average(probs_list, optimal_weights)
        m = compute_metrics(y_true_by_attack[attack_name], y_pred, attack_name)
        print(f"\n--- {attack_name} ---")
        print_metrics(m)
        results_wa_opt.append(m)
    print_summary(results_wa_opt, "Weighted Average optimisé")

    results_by_method = {
        "Majority Voting": results_mv,
        "Weighted Average (égal)": results_wa,
        "Weighted Average optimisé": results_wa_opt,
    }
    print_final_comparison(results_by_method)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = LOG_DIR / f"ensemble_results_{timestamp}.pkl"
    joblib.dump({
        "majority_voting": results_mv,
        "weighted_average_equal": results_wa,
        "weighted_average_optimized": results_wa_opt,
        "optimal_weights": dict(zip(defense_names, optimal_weights.tolist())),
        "defense_names": defense_names,
        "attack_names": attack_names_ordered,
    }, out_path)
    print(f"\nRésultats sauvegardés : {out_path}")
    print("\nEnsemble Aggregation terminée")


if __name__ == "__main__":
    main()
