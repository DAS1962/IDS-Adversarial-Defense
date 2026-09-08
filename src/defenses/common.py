"""
Fonctions partagees par les quatre scripts de defense (11 a 14) et par
l'agregation par ensemble (16).

Raison d'etre : les quatre defenses repetaient exactement le meme code de
chargement, d'evaluation et de calcul de metriques, avec les memes trois
defauts herites de l'ancien pipeline :

  1. Selection du meilleur epoch sur le TEST set (`scheduler.step(test_acc)`,
     `if test_acc > best_acc`). C'est le biais retire de 06_train_baseline.py :
     le test servait a la fois a choisir le modele et a l'evaluer. Les
     defenses utilisent desormais X_val / y_val.
  2. Lecture des y_* avec pd.read_pickle alors que 05 les ecrit en tableaux
     numpy via joblib.dump.
  3. Hyperparametres en dur, sans lecture de configs/config.yaml ni
     verification des empreintes de configuration.

Toutes les moyennes macro sont calculees sur les 15 classes du label_encoder
(labels=all_labels), pas seulement sur celles presentes dans y_true.
"""

import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.models.dnn import BaselineDNN


BENIGN_CLASS = 0

ATTACK_FILES = [
    ("FGSM",     "X_adv_fgsm.pkl"),
    ("BIM",      "X_adv_bim.pkl"),
    ("PGD",      "X_adv_pgd.pkl"),
    ("DeepFool", "X_adv_deepfool.pkl"),
    ("JSMA",     "X_adv_jsma.pkl"),
    ("CW",       "X_adv_cw.pkl"),
]

# Baseline v5 sous attaque, semi-white box, test complet (831 864).
# Reference a battre pour chaque defense. Le F1 macro est la metrique qui
# compte : les accuracies sont plafonnees par la proportion de BENIGN dans
# le test (83.1%), donc une defense qui remonte l'accuracy sans ameliorer
# le F1 macro n'a rien defendu.
BASELINE_UNDEFENDED = {
    "Clean":    {"accuracy": 0.9979, "f1_macro": 0.8411},
    "FGSM":     {"accuracy": 0.8675, "f1_macro": 0.1502},
    "BIM":      {"accuracy": 0.8742, "f1_macro": 0.1621},
    "PGD":      {"accuracy": 0.8210, "f1_macro": 0.0605},
    "DeepFool": {"accuracy": 0.8692, "f1_macro": 0.3291},
    "JSMA":     {"accuracy": 0.8305, "f1_macro": 0.0605},
    "CW":       {"accuracy": 0.8414, "f1_macro": 0.3004},
}


def nettoyer_nom(nom):
    """Retire les caracteres de remplacement issus de l'encodage des CSV."""
    for mauvais in ("\ufffd", "\x96", "\u2013", "\u2014"):
        nom = nom.replace(mauvais, "-")
    return " ".join(nom.split())


def load_splits(data_dir, avec_test=True):
    """
    Charge train / val / test.

    X_* : DataFrames pandas ecrits par to_pickle, convertis en float32.
    y_* : tableaux numpy ecrits par joblib.dump — chargement uniforme,
    contrairement a l'ancienne version qui lisait les y_* avec
    pd.read_pickle et cassait depuis la reecriture de 05.
    """
    print("Chargement des donnees...")

    def _charger(nom):
        X = pd.read_pickle(data_dir / f"X_{nom}.pkl")
        y = joblib.load(data_dir / f"y_{nom}.pkl")
        if isinstance(X, pd.DataFrame):
            X = X.values.astype(np.float32)
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values
        return X, np.asarray(y)

    X_train, y_train = _charger("train")
    X_val, y_val = _charger("val")
    print(f"  X_train : {X_train.shape}")
    print(f"  X_val   : {X_val.shape}")

    if not avec_test:
        print()
        return X_train, y_train, X_val, y_val, None, None

    X_test, y_test = _charger("test")
    print(f"  X_test  : {X_test.shape}")
    print()
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_baseline_model(cfg, device, checkpoint_dir):
    """
    Charge le baseline v5, avec les memes verifications bloquantes que 08,
    10 et 15.

    Utilise par la defense DAE (le classifieur n'est pas reentraine) et par
    l'agregation par ensemble.
    """
    print("Chargement du baseline...")
    model = BaselineDNN(
        input_dim=cfg.dataset["num_features"],
        hidden1=cfg.model["hidden_layers"][0],
        hidden2=cfg.model["hidden_layers"][1],
        output_dim=cfg.dataset["num_classes"],
    )
    chemin = checkpoint_dir / "baseline_best.pth"
    checkpoint = torch.load(chemin, weights_only=False, map_location=device)

    if "val_acc" not in checkpoint:
        raise RuntimeError(
            f"{chemin} n'a pas de cle 'val_acc' : checkpoint du pipeline "
            f"precedent (selection sur le test). Relancer 06_train_baseline.py."
        )
    attendu = cfg.baseline_fingerprint()
    if checkpoint.get("config_hash") != attendu:
        raise RuntimeError(
            f"{chemin} entraine sous une autre configuration "
            f"({checkpoint.get('config_hash')!r} != {attendu!r}). "
            f"Relancer 06_train_baseline.py."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()
    print(f"  Epoch : {checkpoint['epoch']}  |  Val acc : {checkpoint['val_acc']:.4f}\n")
    return model


def evaluate_loader(model, loader, criterion, device):
    """Loss et accuracy sur un DataLoader."""
    model.eval()
    total_loss, total_correct, total_seen = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            total_loss += criterion(logits, y).item() * y.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_seen += y.size(0)
    return total_loss / total_seen, total_correct / total_seen


def predict_array(model, X, device, batch_size, dae=None, classifier=None):
    """
    Predictions sur un tableau numpy, par batchs.

    Si dae et classifier sont fournis, le pipeline devient
    x -> DAE (purification) -> classifier, utilise par la defense DAE.
    """
    n = len(X)
    preds = np.zeros(n, dtype=np.int64)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            fin = min(i + batch_size, n)
            xb = torch.tensor(X[i:fin], dtype=torch.float32).to(device)
            if dae is not None and classifier is not None:
                x_pur, _ = dae(xb)
                logits = classifier(x_pur)
            else:
                logits = model(xb)
            preds[i:fin] = logits.argmax(dim=1).cpu().numpy()
    return preds


def compute_metrics(y_true, y_pred, name, all_labels):
    """Metriques, moyennes calculees sur les 15 classes."""
    return {
        "attack": name,
        "n_samples": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, labels=all_labels, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, labels=all_labels, average="weighted", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, labels=all_labels, average="macro", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, labels=all_labels, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, labels=all_labels, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, labels=all_labels, average="weighted", zero_division=0),
        "recall_benign": recall_score(y_true, y_pred, labels=[BENIGN_CLASS], average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=all_labels),
    }


def print_metrics(m):
    print(f"  Accuracy             : {m['accuracy']:.4f}")
    print(f"  Precision (macro)    : {m['precision_macro']:.4f}")
    print(f"  Recall (macro)       : {m['recall_macro']:.4f}")
    print(f"  F1 (macro)           : {m['f1_macro']:.4f}")
    print(f"  F1 (weighted)        : {m['f1_weighted']:.4f}")
    print(f"  Recall BENIGN        : {m['recall_benign']:.4f}")


def evaluate_on_attacks(model, X_test, y_test, device, batch_size, attacks_dir,
                        all_labels, dae=None, classifier=None):
    """
    Evalue un modele defendu sur les donnees propres puis sur les six
    attaques, en verifiant que chaque X_adv correspond bien au perimetre
    du test charge.
    """
    resultats = []

    print("\n--- Donnees propres ---")
    preds = predict_array(model, X_test, device, batch_size, dae=dae, classifier=classifier)
    m = compute_metrics(y_test, preds, "Clean", all_labels)
    print_metrics(m)
    resultats.append(m)

    for nom, fichier in ATTACK_FILES:
        chemin = attacks_dir / fichier
        print(f"\n--- {nom} ---")
        if not chemin.exists():
            print(f"  {fichier} introuvable, ignore")
            continue
        X_adv = joblib.load(chemin)
        if len(X_adv) != len(y_test):
            print(f"  {len(X_adv):,} lignes contre {len(y_test):,} attendues : "
                  f"genere sur un autre perimetre (evaluation.scope), ignore")
            del X_adv
            continue
        preds = predict_array(model, X_adv, device, batch_size, dae=dae, classifier=classifier)
        m = compute_metrics(y_test, preds, nom, all_labels)
        print_metrics(m)
        resultats.append(m)
        del X_adv

    return resultats


def print_summary(resultats, nom_defense):
    """
    Tableau recapitulatif, avec comparaison au baseline non defendu.

    La colonne du F1 macro est celle qui compte : une defense peut remonter
    l'accuracy sans rien defendre, puisque le plancher de 83.1% correspond
    a une evasion totale ou seul BENIGN reste bien classe.
    """
    print("\n" + "=" * 104)
    print(f"Resume - Defense {nom_defense} (vs baseline non defendu)")
    print("=" * 104)
    print(f"{'Attaque':<12} {'Accuracy':>10} {'(base)':>9} {'Delta':>8}"
          f" | {'F1 macro':>10} {'(base)':>9} {'Delta':>8} | {'Rec.BENIGN':>11}")
    print("-" * 104)
    for r in resultats:
        base = BASELINE_UNDEFENDED.get(r["attack"])
        if base is None:
            print(f"{r['attack']:<12} {r['accuracy']:>10.4f} {'—':>9} {'—':>8}"
                  f" | {r['f1_macro']:>10.4f} {'—':>9} {'—':>8} | {r['recall_benign']:>11.4f}")
            continue
        d_acc = r["accuracy"] - base["accuracy"]
        d_f1 = r["f1_macro"] - base["f1_macro"]
        print(f"{r['attack']:<12} {r['accuracy']:>10.4f} {base['accuracy']:>9.4f} {d_acc:>+8.4f}"
              f" | {r['f1_macro']:>10.4f} {base['f1_macro']:>9.4f} {d_f1:>+8.4f}"
              f" | {r['recall_benign']:>11.4f}")
    print("=" * 104)

    attaques = [r for r in resultats if r["attack"] != "Clean"]
    if attaques:
        gains = [r["f1_macro"] - BASELINE_UNDEFENDED[r["attack"]]["f1_macro"]
                 for r in attaques if r["attack"] in BASELINE_UNDEFENDED]
        if gains:
            print(f"\nGain moyen en F1 macro sur les six attaques : {np.mean(gains):+.4f}")
            if np.mean(gains) <= 0:
                print("  -> La defense ne protege pas. Verifier les hyperparametres.")
