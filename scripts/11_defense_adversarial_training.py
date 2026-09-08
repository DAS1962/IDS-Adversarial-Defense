"""
Defense par Adversarial Training (AT).

Le modele est entraine sur un melange clean / adversarial, les exemples
adversariaux etant generes a la volee sur le modele en cours
d'entrainement.

Reference : Madry et al. 2018 ; Awad et al. 2025 (Algorithme 4).

Pourquoi PGD et non FGSM
-------------------------
La premiere version generait les exemples d'entrainement avec FGSM en un
seul pas. Resultat mesure : +0.137 de F1 macro contre FGSM, mais +0.011
contre PGD et +0.001 contre JSMA. La defense protegeait uniquement contre
l'attaque exacte sur laquelle elle s'entrainait, signature du masquage de
gradient decrit par Madry et al. 2018 - la reference que l'article cite
pour cette defense, et qui recommande explicitement PGD avec plusieurs pas
et depart aleatoire.

Le passage a PGD a porte le gain moyen de +0.0656 a +0.0986. Le gain vient
surtout de FGSM (+0.169), DeepFool (+0.035) et C&W (+0.043) ; BIM regresse
de 0.049 et PGD reste inchange, donc le masquage de gradient n'etait pas la
seule limite.

L'article ne documente ni epsilon ni le nombre de pas de son attaque
d'entrainement (l'Algorithme 4 mentionne seulement "adversarial examples").
Les valeurs viennent de configs/config.yaml, avec eps=0.2 reprenant la
Table 2 et alpha = 2.5*eps/steps suivant la regle usuelle.

Label smoothing applique a l'entrainement
------------------------------------------
L'article n'utilise pas le label smoothing comme une defense parallele aux
trois autres, mais comme une couche appliquee en dessous. L'Algorithme 4
prescrit "train the IDS classifier with the Gaussian augmented dataset and
smooth train labels", et la section "Defense strategies" precise que les
defenses sont ameliorees "particularly when trained with smoothed labels".

La version precedente utilisait nn.CrossEntropyLoss() sans lissage, ce qui
s'ecartait du protocole.

TROIS criterions distincts sont necessaires ici, et les confondre serait un
bug silencieux :

  criterion_train  avec lissage. C'est la loss optimisee par la descente
                   de gradient.

  criterion_attack SANS lissage. Il sert a generer les exemples PGD. Lisser
                   les labels pendant la generation modifierait la
                   direction du gradient d'attaque, donc la nature meme des
                   exemples adversariaux produits : on ne s'entrainerait
                   plus contre PGD mais contre une attaque differente et
                   non documentee.

  criterion_eval   SANS lissage. Pour que la loss de validation mesure la
                   performance reelle et non la loss adoucie, ce qui
                   affecterait aussi le scheduler.

Note sur le protocole
---------------------
Les exemples d'entrainement sont generes en WHITE BOX sur le modele
lui-meme. Ce n'est pas contradictoire avec le protocole semi-white box de
08_generate_attacks.py : la defense s'entraine contre le pire cas,
l'evaluation mesure un attaquant realiste qui ne connait pas les poids.

Les exemples sont bornes dans clip_values apres chaque pas, coherent avec
le domaine produit par MinMaxScaler.
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


def pgd_attack(model, x, y, criterion_attack, eps, alpha, steps, clip_values,
               random_start=True):
    """
    PGD : plusieurs pas de gradient projetes dans la boule L-infini de
    rayon eps autour de x, puis bornes dans le domaine des features.

    criterion_attack doit etre SANS label smoothing (voir l'en-tete du
    module) : le gradient d'attaque doit pointer vers la vraie classe, pas
    vers une distribution adoucie.

    Le depart aleatoire evite que l'attaque parte toujours du meme point,
    ce qui rendrait l'entrainement exploitable par le modele.
    """
    borne_min, borne_max = clip_values
    x_orig = x.detach()

    if random_start:
        x_adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
        x_adv = torch.clamp(x_adv, borne_min, borne_max).detach()
    else:
        x_adv = x_orig.clone().detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = criterion_attack(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        # Projection dans la boule L-infini autour de l'original
        x_adv = x_orig + torch.clamp(x_adv - x_orig, -eps, eps)
        # Puis dans le domaine des features
        x_adv = torch.clamp(x_adv, borne_min, borne_max).detach()

    return x_adv


def fgsm_attack(model, x, y, criterion_attack, eps, clip_values):
    """FGSM en un pas, conserve pour comparaison via defenses.attack."""
    borne_min, borne_max = clip_values
    x_adv = x.clone().detach().requires_grad_(True)
    loss = criterion_attack(model(x_adv), y)
    grad = torch.autograd.grad(loss, x_adv)[0]
    x_adv = x_adv.detach() + eps * grad.sign()
    return torch.clamp(x_adv, borne_min, borne_max).detach()


def train_one_epoch(model, loader, optimizer, criterion_train,
                    criterion_attack, device, at_cfg, clip_values):
    """
    Une epoch sur un melange clean / adversarial.

    criterion_train porte le lissage et sert a la descente de gradient.
    criterion_attack n'en porte pas et sert a generer les exemples.
    """
    model.train()
    total_loss, total_correct, total_seen = 0.0, 0, 0

    methode = at_cfg["attack"].upper()
    eps = at_cfg["eps"]
    ratio = at_cfg["ratio"]

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        n = x.size(0)
        n_adv = int(n * ratio)

        if n_adv > 0:
            idx = torch.randperm(n, device=device)
            idx_adv, idx_clean = idx[:n_adv], idx[n_adv:]

            # eval() pendant la generation : desactive dropout et batchnorm
            # pour que le gradient corresponde au modele en inference.
            model.eval()
            if methode == "PGD":
                x_adv = pgd_attack(
                    model, x[idx_adv], y[idx_adv], criterion_attack,
                    eps=eps, alpha=at_cfg["alpha"], steps=at_cfg["steps"],
                    clip_values=clip_values,
                    random_start=at_cfg.get("random_start", True),
                )
            elif methode == "FGSM":
                x_adv = fgsm_attack(model, x[idx_adv], y[idx_adv],
                                    criterion_attack, eps, clip_values)
            else:
                raise ValueError(
                    f"defenses.AdversarialTraining.attack vaut {methode!r}, "
                    f"attendu 'PGD' ou 'FGSM'."
                )
            model.train()

            x_batch = torch.cat([x[idx_clean], x_adv], dim=0)
            y_batch = torch.cat([y[idx_clean], y[idx_adv]], dim=0)
        else:
            x_batch, y_batch = x, y

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion_train(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y_batch.size(0)
        total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total_seen += y_batch.size(0)

    return total_loss / total_seen, total_correct / total_seen


def main():
    print("=" * 70)
    print("Defense par Adversarial Training (AT) + Label Smoothing")
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
    at_cfg = cfg.defenses["AdversarialTraining"]
    clip_values = cfg.clip_values
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
    criterion_attack = nn.CrossEntropyLoss()
    criterion_eval = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=sched["factor"],
        patience=sched["patience"], min_lr=sched["min_lr"],
    )

    print("Entrainement adversarial")
    print(f"  Epochs             : {epochs}")
    print(f"  Learning rate init : {lr_init}")
    print(f"  Attaque            : {at_cfg['attack']}")
    print(f"  Epsilon            : {at_cfg['eps']}")
    if at_cfg["attack"].upper() == "PGD":
        print(f"  Pas                : {at_cfg['steps']} x alpha={at_cfg['alpha']}")
        print(f"  Depart aleatoire   : {at_cfg.get('random_start', True)}")
    print(f"  Ratio adversarial  : {at_cfg['ratio']:.0%}")
    print(f"  Label smoothing    : {alpha} (Algorithme 4 de l'article)")
    print(f"    entrainement : lisse | generation d'attaque : non lisse")
    print(f"  Bornes             : {clip_values}")
    print(f"  Batch size         : {batch_size}")
    print("  Selection modele   : val_acc (validation, jamais le test)\n")

    history = {"train_loss": [], "train_acc": [], "val_loss": [],
                "val_acc": [], "lr": []}
    best_val_acc, best_epoch = 0.0, -1
    ckpt_path = checkpoint_dir / "defense_at_best.pth"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    debut = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        lr_now = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion_train,
            criterion_attack, device, at_cfg, clip_values,
        )
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
                "defense": "AdversarialTraining",
                "at_config": dict(at_cfg), "label_smoothing": alpha,
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
    print_summary(resultats, f"Adversarial Training ({at_cfg['attack']}) + LS")

    log_dir.mkdir(parents=True, exist_ok=True)
    horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")
    sortie = log_dir / f"defense_at_{horodatage}.pkl"
    joblib.dump({
        "results": resultats, "history": history,
        "hyperparameters": {
            "epochs": epochs, "batch_size": batch_size, "lr_init": lr_init,
            "at_config": dict(at_cfg), "label_smoothing": alpha,
            "clip_values": list(clip_values), "seed": cfg.seed,
        },
        "best_epoch": best_epoch, "best_val_acc": best_val_acc,
        "training_time_min": duree,
        "config_hash": cfg.baseline_fingerprint(),
    }, sortie)
    print(f"\nResultats sauvegardes : {sortie}")
    print("\nDefense Adversarial Training terminee")


if __name__ == "__main__":
    main()
