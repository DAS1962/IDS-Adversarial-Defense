"""
Entrainement du detecteur IDS, Table 3 du papier.

Adam, lr 0.01, 30 epochs, batch 128. On garde ces valeurs telles quelles
meme si sur main le lr 0.01 rendait l'entrainement instable : ici on
reproduit, on n'ameliore pas.

Le papier utilise validation_split=0.33 de Keras, donc un morceau du train
decoupe au moment du fit. On a un vrai set de validation, cree par le
pipeline, et on selectionne le meilleur epoch dessus. Le test n'est touche
qu'une fois, a la fin.
"""

import argparse
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
from src.models.dnn import DNN
from src.utils.commun import charger_donnees, predire, metriques, afficher
from src.utils.config import load_config


def evaluer(model, loader, criterion, device):
    model.eval()
    perte, correct, vus = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            perte += criterion(out, y).item() * y.size(0)
            correct += (out.argmax(dim=1) == y).sum().item()
            vus += y.size(0)
    return perte / vus, correct / vus


def main():
    parseur = argparse.ArgumentParser()
    parseur.add_argument("--variante-lr", action="store_true",
                         help="lr reduit + scheduler au lieu du lr 0.01 du papier")
    args = parseur.parse_args()

    print("=" * 70)
    print("Baseline - reproduction fidele")
    print(f"Date : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 70 + "\n")

    cfg = load_config()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print()

    X_tr, y_tr, X_va, y_va, X_te, y_te = charger_donnees(cfg.paths["processed"])
    print(f"train {X_tr.shape}  val {X_va.shape}  test {X_te.shape}\n")

    m = cfg.model
    model = DNN(input_dim=cfg.dataset["num_features"],
                hidden=tuple(m["hidden_layers"]),
                output_dim=cfg.dataset["num_classes"]).to(device)
    print(f"Modele : {sum(p.numel() for p in model.parameters()):,} parametres")
    print(f"  {m['hidden_layers']}, lr {m['learning_rate']}, "
          f"{m['epochs']} epochs, batch {m['batch_size']}\n")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=m["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
        batch_size=m["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    var = m.get("variante_lr", {})
    if args.variante_lr:
        lr = var["learning_rate"]
        epochs = var.get("epochs", m["epochs"])
        suffixe = "_varlr"
        print(f"Variante : lr {lr}, {epochs} epochs, "
              f"scheduler {var['scheduler']}\n")
    else:
        lr = m["learning_rate"]
        epochs = m["epochs"]
        suffixe = ""

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if args.variante_lr:
        sc = var["scheduler"]
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=sc["factor"],
            patience=sc["patience"], min_lr=sc["min_lr"])

    ckpt = Path(cfg.paths["checkpoints"]) / f"baseline{suffixe}.pth"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    historique = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    meilleur_acc, meilleur_epoch = 0.0, -1
    debut = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        perte, correct, vus = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            perte += loss.item() * y.size(0)
            correct += (out.argmax(dim=1) == y).sum().item()
            vus += y.size(0)

        train_loss, train_acc = perte / vus, correct / vus
        val_loss, val_acc = evaluer(model, val_loader, criterion, device)
        lr_courant = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step(val_loss)

        for cle, val in zip(historique, (train_loss, train_acc, val_loss, val_acc)):
            historique[cle].append(val)

        print(f"Epoch {epoch:3d}/{epochs} | lr={lr_courant:.6f} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"{time.time()-t0:.1f}s", flush=True)

        if val_acc > meilleur_acc:
            meilleur_acc, meilleur_epoch = val_acc, epoch
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch, "val_acc": val_acc}, ckpt)
            print(f"         -> sauvegarde (val_acc={val_acc:.4f})", flush=True)

    duree = (time.time() - debut) / 60
    print(f"\nTermine en {duree:.1f} min")
    print(f"Meilleur epoch : {meilleur_epoch}, val_acc {meilleur_acc:.4f}\n")

    model.load_state_dict(torch.load(ckpt, weights_only=False,
                                     map_location=device)["model_state_dict"])

    print("=" * 70)
    print("Evaluation sur le test")
    print("=" * 70)
    labels = list(range(cfg.dataset["num_classes"]))
    pred = predire(model, X_te, device, cfg.evaluation["batch_size"])
    res = metriques(y_te, pred, labels, "Clean")
    afficher(res)

    # Le papier donne 98.11 % d'accuracy (Table 4), et son F1 de 98.068 %
    # est une moyenne ponderee, pas macro.
    print(f"\n  Papier (Table 4) : accuracy 98.11 %, F1 98.068 %")
    print(f"  Ecart accuracy   : {100*res['accuracy'] - 98.11:+.2f} points")

    print("\nPar classe :")
    le = joblib.load(Path(cfg.paths["processed"]) / "label_encoder.pkl")
    cm = res["confusion_matrix"]
    for i in labels:
        support = cm[i].sum()
        if support == 0:
            continue
        rappel = cm[i, i] / support
        nom = str(le.classes_[i]).replace("\ufffd", "-")[:32]
        print(f"  {i:2d} {nom:<32} rappel {rappel:.4f}  support {support:>8,}")

    log_dir = Path(cfg.paths["logs"])
    log_dir.mkdir(parents=True, exist_ok=True)
    sortie = log_dir / f"baseline{suffixe}_{datetime.now():%Y%m%d_%H%M%S}.pkl"
    joblib.dump({"resultats": res, "historique": historique,
                 "meilleur_epoch": meilleur_epoch, "duree_min": duree,
                 "config": {"model": m, "seed": cfg.seed, "lr": lr,
                           "epochs": epochs,
                           "variante_lr": args.variante_lr}}, sortie)
    print(f"\nSauvegarde : {sortie}")


if __name__ == "__main__":
    main()
