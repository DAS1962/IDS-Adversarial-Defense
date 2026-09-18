"""Fonctions partagees par les scripts d'entrainement et d'evaluation."""

import joblib
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)


def charger_donnees(dossier):
    """Les six tableaux, deja en numpy depuis le pipeline."""
    d = {}
    for nom in ("X_train", "X_val", "X_test", "y_train", "y_val", "y_test"):
        d[nom] = joblib.load(f"{dossier}/{nom}.pkl")
    return (d["X_train"].astype(np.float32), d["y_train"].astype(np.int64),
            d["X_val"].astype(np.float32), d["y_val"].astype(np.int64),
            d["X_test"].astype(np.float32), d["y_test"].astype(np.int64))


@torch.no_grad()
def predire(model, X, device, batch_size=512):
    model.eval()
    preds = np.zeros(len(X), dtype=np.int64)
    for i in range(0, len(X), batch_size):
        fin = min(i + batch_size, len(X))
        xb = torch.tensor(X[i:fin], dtype=torch.float32).to(device)
        preds[i:fin] = model(xb).argmax(dim=1).cpu().numpy()
    return preds


def mcc_multiclasse(cm):
    """MCC multiclasse depuis la matrice de confusion (Gorodkin 2004)."""
    cm = np.asarray(cm, dtype=np.float64)
    n = cm.sum()
    if n == 0:
        return float("nan")
    t, p = cm.sum(axis=1), cm.sum(axis=0)
    num = np.trace(cm) * n - float(t @ p)
    den = np.sqrt(max(n**2 - float(p @ p), 0)) * np.sqrt(max(n**2 - float(t @ t), 0))
    return float(num / den) if den > 0 else 0.0


def mcc_binaire(cm):
    """
    MCC binaire attaque contre normal. C'est ce que donne la Table 9 du
    papier : "we reduce the classification task into binary classification
    for simplicity". Classe 0 = BENIGN = negatif.
    """
    cm = np.asarray(cm, dtype=np.float64)
    tn = cm[0, 0]
    fp = cm[0].sum() - tn
    fn = cm[:, 0].sum() - tn
    tp = cm.sum() - tn - fp - fn
    num = tp * tn - fp * fn
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float(num / den) if den > 0 else 0.0


def taux_binaires(cm):
    """TPR, FPR, TNR, FNR en binaire attaque contre normal."""
    cm = np.asarray(cm, dtype=np.float64)
    tn = cm[0, 0]
    fp = cm[0].sum() - tn
    fn = cm[:, 0].sum() - tn
    tp = cm.sum() - tn - fp - fn
    return {
        "tpr": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "tnr": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else 0.0,
    }


def metriques(y_vrai, y_pred, labels, nom=""):
    """
    Les metriques du papier plus les notres.

    Le papier ne publie que l'accuracy et des moyennes ponderees (sa Table 4
    donne accuracy 98.11, recall 98.11, precision 98.11, F1 98.068 : quatre
    valeurs quasi identiques). On garde aussi les moyennes macro, qui
    comptent chaque classe pareil et sont plus honnetes sur un jeu aussi
    desequilibre.
    """
    cm = confusion_matrix(y_vrai, y_pred, labels=labels)
    # recall sur la classe 0 (BENIGN) : montre si la defense se met a rejeter
    # du trafic legitime, ce que le F1 moyen cache.
    rappel_benign = cm[0, 0] / cm[0].sum() if cm[0].sum() else 0.0

    return {
        "attaque": nom,
        "accuracy": accuracy_score(y_vrai, y_pred),
        "precision_macro": precision_score(y_vrai, y_pred, labels=labels,
                                           average="macro", zero_division=0),
        "recall_macro": recall_score(y_vrai, y_pred, labels=labels,
                                     average="macro", zero_division=0),
        "f1_macro": f1_score(y_vrai, y_pred, labels=labels,
                             average="macro", zero_division=0),
        "precision_weighted": precision_score(y_vrai, y_pred, labels=labels,
                                              average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_vrai, y_pred, labels=labels,
                                        average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_vrai, y_pred, labels=labels,
                                average="weighted", zero_division=0),
        "precision_micro": precision_score(y_vrai, y_pred, labels=labels,
                                           average="micro", zero_division=0),
        "recall_micro": recall_score(y_vrai, y_pred, labels=labels,
                                     average="micro", zero_division=0),
        "f1_micro": f1_score(y_vrai, y_pred, labels=labels,
                             average="micro", zero_division=0),
        "recall_benign": float(rappel_benign),
        "mcc_multi": mcc_multiclasse(cm),
        "mcc_binaire": mcc_binaire(cm),
        **taux_binaires(cm),
        "confusion_matrix": cm,
    }


def afficher(m):
    print(f"  Accuracy        : {m['accuracy']:.4f}")
    print(f"  Precision macro : {m['precision_macro']:.4f}")
    print(f"  Recall macro    : {m['recall_macro']:.4f}")
    print(f"  F1 macro        : {m['f1_macro']:.4f}")
    print(f"  F1 weighted     : {m['f1_weighted']:.4f}")
    print(f"  Recall BENIGN   : {m['recall_benign']:.4f}")
    print(f"  MCC binaire     : {m['mcc_binaire']:.4f}")
    print(f"  TPR / FPR       : {m['tpr']:.4f} / {m['fpr']:.4f}")
