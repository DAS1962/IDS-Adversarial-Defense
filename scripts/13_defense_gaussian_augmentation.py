"""
Defense par Gaussian Augmentation (GA).

Du bruit gaussien N(0, sigma^2) est ajoute aux features d'entree a chaque
batch pendant l'entrainement. Le modele apprend ainsi des representations
invariantes aux petites perturbations.

Reference : Zantedeschi et al. 2017 ; Awad et al. 2025 (Algorithme 3).

Label smoothing applique a l'entrainement
------------------------------------------
L'article n'utilise pas le label smoothing comme une defense parallele aux
trois autres, mais comme une couche appliquee en dessous. Trois passages le
prescrivent explicitement.

Algorithme 3 : "train the IDS classifier with the adversarial augmented
dataset and smooth train labels".

Algorithme 4 : "train the IDS classifier with the Gaussian augmented
dataset and smooth train labels".

Section "Defense strategies" : "Notably, the application of these defense
mechanisms, particularly when trained with smoothed labels, results in a
significant improvement in the classifier's accuracy and resilience".

La version precedente de ce script utilisait nn.CrossEntropyLoss() sans
lissage, ce qui s'ecartait du protocole. alpha est lu depuis
defenses.LabelSmoothing.alpha, donc les deux defenses partagent la meme
valeur, coherent avec l'idee d'une couche commune.

Trois criterions distincts sont necessaires :
  - criterion_train : avec lissage, pour la descente de gradient
  - criterion_eval  : sans lissage, pour que la loss de validation mesure
                      la performance reelle et non la loss adoucie

Note sur le domaine : les features sont dans [0,1] (MinMaxScaler). Le bruit
n'est PAS borne apres ajout, contrairement aux attaques adversariales.
C'est volontaire : une augmentation de donnees n'a pas a respecter la
contrainte de fonctionnalite du trafic, elle sert uniquement a regulariser
l'entrainement.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.dnn import BaselineDNN
from src.utils.config import load_config, check_data_fingerprint
from src.defenses.common import (
    load_splits,
    evaluate_loader,
    evaluate_on_attacks,
    print_summary,
)


def train_one_epoch(model, loader, optimizer, criterion_train, device, sigma):
    """
    Une epoch avec bruit gaussien sur les inputs.

    criterion_train porte le label smoothing : c'est la loss optimisee.
    """
    model.train()
    total_loss, total_correct, total_seen = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x_bruite = x + torch.randn_like(x) * sigma

        optimizer.zero_grad()
        logits = model(x_bruite)
        loss = criterion_train(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_seen += y.size(0)
    return total_loss / total_seen, total_correct / total_seen


def main():
    print("=" * 70)
    print("Defense par Gaussian Augmentation (GA) + Label Smoothing")
    print(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")

    cfg = load_config()
    print(cfg.resume())
    print()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    data_dir = Path(cfg.paths["data_processed"])
    checkpoint_dir = Path(cfg.paths["checkpoints"])
    log_dir = Path(cfg.paths["logs"])
    attacks_dir = Path(cfg.paths["attacks"])

    check_data_fingerprint(cfg, data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print()

    X_train, y_train, X_val, y_val, X_test, y_test = load_splits(data_dir)

    batch_size = cfg.training["batch_size"]
    eval_batch = cfg.evaluation["batch_size"]
    epochs = cfg.training["epochs"]
    lr_init = cfg.training["learning_rate"]
    sched = cfg.training["scheduler"]
    sigma = cfg.defenses["GaussianAugmentation"]["sigma"]
    # Partage avec la defense LS : l'article traite le lissage comme une
    # couche commune aux defenses, pas comme un parametre propre a chacune.
    alpha = cfg.defenses["LabelSmoothing"]["alpha"]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float(),
                      torch.from_numpy(y_train).long()),
        batch_size=batch_size, shuffle=True,
        num_workers=cfg.env["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val).float(),
                      torch.from_numpy(y_val).long()),
        batch_size=batch_size, shuffle=False,
        num_workers=cfg.env["num_workers"], pin_memory=True,
    )

    model = BaselineDNN(
        input_dim=cfg.dataset["num_features"],
        hidden1=cfg.model["hidden_layers"][0],
        hidden2=cfg.model["hidden_layers"][1],
        output_dim=cfg.dataset["num_classes"],
    ).to(device)
    print(f"Modele : {sum(p.numel() for p in model.parameters()):,} parametres\n")

    criterion_train = nn.CrossEntropyLoss(label_smoothing=alpha)
    criterion_eval = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=sched["factor"],
        patience=sched["patience"], min_lr=sched["min_lr"],
    )

    print("Entrainement avec Gaussian Augmentation")
    print(f"  Epochs             : {epochs}")
    print(f"  Learning rate init : {lr_init}")
    print(f"  Bruit sigma        : {sigma}")
    print(f"  Label smoothing    : {alpha} (Algorithme 3 de l'article)")
    print(f"  Batch size         : {batch_size}")
    print("  Selection modele   : val_acc (validation, jamais le test)\n")

    history = {"train_loss": [], "train_acc": [], "val_loss": [],
                "val_acc": [], "lr": []}
    best_val_acc, best_epoch = 0.0, -1
    ckpt_path = checkpoint_dir / "defense_ga_best.pth"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    debut = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        lr_now = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion_train, device, sigma,
        )
        # Evaluation SANS bruit et SANS lissage : on mesure la performance
        # reelle, pas la tache d'entrainement.
        val_loss, val_acc = evaluate_loader(model, val_loader,
                                            criterion_eval, device)
        scheduler.step(val_loss)

        for cle, val in zip(history, (train_loss, train_acc, val_loss,
                                      val_acc, lr_now)):
            history[cle].append(val)

        print(f"Epoch {epoch:3d}/{epochs} | lr={lr_now:.6f} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"{time.time()-t0:.1f}s", flush=True)

        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch, "val_acc": val_acc, "val_loss": val_loss,
                "config_hash": cfg.baseline_fingerprint(),
                "defense": "GaussianAugmentation",
                "sigma": sigma, "label_smoothing": alpha,
            }, ckpt_path)
            print(f"            -> meilleur modele sauvegarde "
                  f"(val_acc={val_acc:.4f})", flush=True)

    duree = (time.time() - debut) / 60
    print(f"\nEntrainement termine en {duree:.1f} min")
    print(f"Meilleur epoch : {best_epoch} | val_acc = {best_val_acc:.4f}\n")

    checkpoint = torch.load(ckpt_path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("=" * 70)
    print("Evaluation sur donnees propres et attaques adversariales")
    print("=" * 70)

    all_labels = list(range(cfg.dataset["num_classes"]))
    resultats = evaluate_on_attacks(model, X_test, y_test, device, eval_batch,
                                    attacks_dir, all_labels)
    print_summary(resultats, "Gaussian Augmentation + LS")

    log_dir.mkdir(parents=True, exist_ok=True)
    horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")
    sortie = log_dir / f"defense_ga_{horodatage}.pkl"
    joblib.dump({
        "results": resultats, "history": history,
        "hyperparameters": {
            "epochs": epochs, "batch_size": batch_size, "lr_init": lr_init,
            "gaussian_sigma": sigma, "label_smoothing": alpha,
            "seed": cfg.seed,
        },
        "best_epoch": best_epoch, "best_val_acc": best_val_acc,
        "training_time_min": duree,
        "config_hash": cfg.baseline_fingerprint(),
    }, sortie)
    print(f"\nResultats sauvegardes : {sortie}")
    print("\nDefense Gaussian Augmentation terminee")


if __name__ == "__main__":
    main()
