"""
Defense par Label Smoothing (LS).

Le modele est entraine avec une cross-entropy adoucie (label_smoothing=alpha)
au lieu de la cross-entropy classique. Cela reduit la sur-confiance du modele
et lisse les frontieres de decision, ce qui le rend en principe moins
sensible aux perturbations adversariales.

Reference : Muller, Kornblith & Hinton 2019 ; Awad et al. 2025 (Algorithme 2).

Changements par rapport a la version precedente
------------------------------------------------
- Selection du meilleur epoch sur le set de VALIDATION, plus sur le test.
  L'ancienne version faisait `scheduler.step(test_acc)` et
  `if test_acc > best_acc`, donc le test servait a la fois a choisir le
  modele et a l'evaluer — le biais retire de 06_train_baseline.py.
- Hyperparametres lus depuis configs/config.yaml, plus en dur.
- Verification de l'empreinte des donnees avant tout calcul.
- y_* charges avec joblib.load (05 les ecrit en tableaux numpy).
- Moyennes macro sur les 15 classes.
- Checkpoint nomme defense_ls_best.pth, conforme a ce qu'attend le script 16
  (l'ancien s'appelait defense_label_smoothing_best.pth).
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
    evaluate_loader,
    evaluate_on_attacks,
    print_summary,
)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_seen = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_seen += y.size(0)
    return total_loss / total_seen, total_correct / total_seen


def main():
    print("=" * 70)
    print("Defense par Label Smoothing (LS)")
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

    from src.defenses.common import load_splits
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits(data_dir)

    batch_size = cfg.training["batch_size"]
    eval_batch = cfg.evaluation["batch_size"]
    epochs = cfg.training["epochs"]
    lr_init = cfg.training["learning_rate"]
    sched = cfg.training["scheduler"]
    alpha = cfg.defenses["LabelSmoothing"]["alpha"]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()),
        batch_size=batch_size, shuffle=True, num_workers=cfg.env["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long()),
        batch_size=batch_size, shuffle=False, num_workers=cfg.env["num_workers"], pin_memory=True,
    )

    model = BaselineDNN(
        input_dim=cfg.dataset["num_features"],
        hidden1=cfg.model["hidden_layers"][0],
        hidden2=cfg.model["hidden_layers"][1],
        output_dim=cfg.dataset["num_classes"],
    ).to(device)
    print(f"Modele : {sum(p.numel() for p in model.parameters()):,} parametres\n")

    criterion = nn.CrossEntropyLoss(label_smoothing=alpha)
    # Le criterion d'evaluation n'applique PAS le lissage : la loss de
    # validation doit mesurer la performance reelle, pas la loss adoucie.
    criterion_eval = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=sched["factor"],
        patience=sched["patience"], min_lr=sched["min_lr"],
    )

    print("Entrainement avec Label Smoothing")
    print(f"  Epochs             : {epochs}")
    print(f"  Learning rate init : {lr_init}")
    print(f"  Label smoothing    : {alpha}")
    print(f"  Batch size         : {batch_size}")
    print(f"  Selection modele   : val_acc (validation, jamais le test)\n")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_acc, best_epoch = 0.0, -1
    ckpt_path = checkpoint_dir / "defense_ls_best.pth"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    debut = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        lr_now = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate_loader(model, val_loader, criterion_eval, device)
        scheduler.step(val_loss)

        for cle, val in zip(history, (train_loss, train_acc, val_loss, val_acc, lr_now)):
            history[cle].append(val)

        print(f"Epoch {epoch:3d}/{epochs} | lr={lr_now:.6f} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"{time.time()-t0:.1f}s", flush=True)

        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "config_hash": cfg.baseline_fingerprint(),
                "defense": "LabelSmoothing",
                "alpha": alpha,
            }, ckpt_path)
            print(f"            -> meilleur modele sauvegarde (val_acc={val_acc:.4f})", flush=True)

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
    print_summary(resultats, "Label Smoothing")

    log_dir.mkdir(parents=True, exist_ok=True)
    horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")
    sortie = log_dir / f"defense_ls_{horodatage}.pkl"
    joblib.dump({
        "results": resultats,
        "history": history,
        "hyperparameters": {
            "epochs": epochs, "batch_size": batch_size, "lr_init": lr_init,
            "label_smoothing": alpha, "seed": cfg.seed,
        },
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "training_time_min": duree,
        "config_hash": cfg.baseline_fingerprint(),
    }, sortie)
    print(f"\nResultats sauvegardes : {sortie}")
    print("\nDefense Label Smoothing terminee")


if __name__ == "__main__":
    main()
