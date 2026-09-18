"""
Code partage par les quatre defenses.

Les trois defenses d'entrainement (LS, GA, AT) utilisent la meme architecture,
le meme optimiseur et la meme evaluation. Seule change la facon de construire
le batch. Le mettre ici evite que les scripts derivent les uns des autres.
"""

import time
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.dnn import DNN
from src.utils.commun import predire, metriques

ATTAQUES = ["FGSM", "BIM", "PGD", "DeepFool", "JSMA", "CW"]
# Les quatre attaques des Algorithmes 3 et 5
QUATRE = ["fgsm", "bim", "deepfool", "jsma"]


def modele_neuf(cfg, device):
    return DNN(input_dim=cfg.dataset["num_features"],
               hidden=tuple(cfg.model["hidden_layers"]),
               output_dim=cfg.dataset["num_classes"]).to(device)


def charger_adv_train(atk_dir):
    """
    Les quatre attaques generees sur l'echantillon TRAIN par le script 03.

    Les etiquettes sont celles des echantillons d'origine : les attaques sont
    untargeted, elles changent la prediction du modele mais pas la vraie
    classe du trafic.
    """
    X = joblib.load(atk_dir / "X_train_clean.pkl")
    y = joblib.load(atk_dir / "y_train_clean.pkl")

    blocs_x, blocs_y = [], []
    for nom in QUATRE:
        chemin = atk_dir / f"X_adv_train_{nom}.pkl"
        if not chemin.exists():
            raise RuntimeError(f"{chemin} manquant. Lancer scripts/03_attaques.sh")
        X_adv = joblib.load(chemin)
        l2 = np.linalg.norm(X_adv - X, axis=1).mean()
        print(f"    {nom:<9} {len(X_adv):,} exemples, L2 moyen {l2:.4f}")
        blocs_x.append(X_adv)
        blocs_y.append(y)

    return np.concatenate(blocs_x), np.concatenate(blocs_y)


def entrainer(model, X, y, X_val, y_val, cfg, device, alpha, sigma=None):
    """
    Entrainement commun aux trois defenses.

    alpha : lissage des etiquettes. Applique aux trois, comme le prescrivent
            les Algorithmes 3 et 4 ("and smooth train labels").
    sigma : si donne, bruit gaussien ajoute a chaque batch. Sert a GA.

    Deux fonctions de cout : lissee pour l'entrainement, NON lissee pour la
    validation. Sinon la val_loss mesure la loss adoucie et le scheduler suit
    la mauvaise quantite.
    """
    m = cfg.model
    var = m.get("variante_lr", {})
    lr = var["learning_rate"]
    epochs = var.get("epochs", m["epochs"])
    bas, haut = cfg.clip_values

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=m["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=m["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=alpha)
    criterion_eval = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    sc = var["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=sc["factor"], patience=sc["patience"],
        min_lr=sc["min_lr"])

    historique = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    meilleur, meilleur_ep, etat = 0.0, -1, None
    debut = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        perte, correct, vus = 0.0, 0, 0
        for x, yb in loader:
            x = x.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if sigma:
                x = torch.clamp(x + torch.randn_like(x) * sigma, bas, haut)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            perte += loss.item() * yb.size(0)
            correct += (out.argmax(dim=1) == yb).sum().item()
            vus += yb.size(0)

        model.eval()
        vperte, vcorrect, vvus = 0.0, 0, 0
        with torch.no_grad():
            for x, yb in val_loader:
                x, yb = x.to(device), yb.to(device)
                out = model(x)
                vperte += criterion_eval(out, yb).item() * yb.size(0)
                vcorrect += (out.argmax(dim=1) == yb).sum().item()
                vvus += yb.size(0)

        tl, ta = perte / vus, correct / vus
        vl, va = vperte / vvus, vcorrect / vvus
        scheduler.step(vl)
        for cle, val in zip(historique, (tl, ta, vl, va)):
            historique[cle].append(val)

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(f"    epoch {epoch:3d}/{epochs} | "
                  f"lr={optimizer.param_groups[0]['lr']:.6f} | "
                  f"train_acc={ta:.4f} val_acc={va:.4f}", flush=True)

        if va > meilleur:
            meilleur, meilleur_ep = va, epoch
            etat = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(etat)
    duree = (time.time() - debut) / 60
    print(f"    termine en {duree:.1f} min, meilleur epoch {meilleur_ep} "
          f"(val_acc {meilleur:.4f})")
    return historique, meilleur_ep, meilleur, duree


@torch.no_grad()
def nettoyer(dae, X, device, batch_size, clip_values):
    """Passe l'entree dans le DAE avant le detecteur. Sert au script 10."""
    dae.eval()
    bas, haut = clip_values
    sortie = np.zeros_like(X)
    for i in range(0, len(X), batch_size):
        fin = min(i + batch_size, len(X))
        xb = torch.tensor(X[i:fin], dtype=torch.float32).to(device)
        sortie[i:fin] = dae(xb).clamp(bas, haut).cpu().numpy()
    return sortie


def evaluer(model, cfg, device, labels, atk_dir, dae=None):
    """Clean plus les six attaques, sur l'echantillon test."""
    X = joblib.load(atk_dir / "X_test_clean.pkl")
    y = joblib.load(atk_dir / "y_test_clean.pkl")
    bs = cfg.evaluation["batch_size"]

    def pred(entree):
        if dae is not None:
            entree = nettoyer(dae, entree, device, bs, cfg.clip_values)
        return predire(model, entree, device, bs)

    res = {"Clean": metriques(y, pred(X), labels, "Clean")}
    for nom in ATTAQUES:
        X_adv = joblib.load(atk_dir / f"X_adv_test_{nom.lower()}.pkl")
        res[nom] = metriques(y, pred(X_adv), labels, nom)
        del X_adv
    return res


def baseline_reference(log_dir, quel="varlr"):
    """Resultats du baseline, pour calculer les gains."""
    fichiers = sorted(Path(log_dir).glob("evaluation_*.pkl"))
    if not fichiers:
        raise RuntimeError("Aucun evaluation_*.pkl. Lancer scripts/04_evaluation.sh")
    ref = joblib.load(fichiers[-1])
    cle = [k for k in ref["resultats"] if ("fidele" in k) == (quel == "fidele")][0]
    return ref["resultats"][cle], cle


def resume(res, base, nom_defense):
    """Tableau de comparaison au baseline, identique pour les quatre."""
    print("=" * 92)
    print(f"Resume - {nom_defense}")
    print("=" * 92)
    print(f"{'Attaque':<10} {'Accuracy':>9} {'base':>9} {'delta':>8} | "
          f"{'F1 macro':>9} {'base':>9} {'delta':>8} | {'MCC bin':>8} {'RecBEN':>8}")
    print("-" * 92)

    gains = []
    for nom in ["Clean"] + ATTAQUES:
        m, b = res[nom], base[nom]
        d_acc = m["accuracy"] - b["accuracy"]
        d_f1 = m["f1_macro"] - b["f1_macro"]
        if nom != "Clean":
            gains.append(d_f1)
        print(f"{nom:<10} {m['accuracy']:>9.4f} {b['accuracy']:>9.4f} "
              f"{d_acc:>+8.4f} | {m['f1_macro']:>9.4f} {b['f1_macro']:>9.4f} "
              f"{d_f1:>+8.4f} | {m['mcc_binaire']:>8.4f} {m['recall_benign']:>8.4f}")

    print("-" * 92)
    gain = float(np.mean(gains))
    cout = res["Clean"]["f1_macro"] - base["Clean"]["f1_macro"]
    rb_min = min(res[a]["recall_benign"] for a in ATTAQUES)
    print(f"Gain moyen en F1 macro, six attaques : {gain:+.4f}")
    print(f"Cout sur donnees propres             : {cout:+.4f}")
    print(f"Bilan net                            : {gain + cout:+.4f}")
    print(f"Rappel BENIGN minimal                : {rb_min:.4f}")
    print(f"MCC binaire sur donnees propres      : {res['Clean']['mcc_binaire']:.4f}")
    return {"gain": gain, "cout": cout, "bilan": gain + cout, "rb_min": rb_min}


def sauvegarder(cfg, nom, model, res, hist, best_ep, best_acc, duree, bilan):
    ck = Path(cfg.paths["checkpoints"]) / f"defense_{nom}.pth"
    ck.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "epoch": best_ep,
                "val_acc": best_acc, "defense": nom}, ck)

    from datetime import datetime
    sortie = (Path(cfg.paths["logs"]) /
              f"defense_{nom}_{datetime.now():%Y%m%d_%H%M%S}.pkl")
    joblib.dump({"resultats": res, "historique": hist, "defense": nom,
                 "meilleur_epoch": best_ep, "duree_min": duree,
                 "bilan": bilan, "config": dict(cfg.defenses)}, sortie)
    print(f"\nCheckpoint : {ck}")
    print(f"Resultats  : {sortie}")
