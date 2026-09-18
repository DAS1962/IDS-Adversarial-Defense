"""
Substitut + generation des six attaques.

Le papier entraine un substitut (58 -> 100 -> 100 -> 15) et genere les
attaques dessus, puis les transfere sur le detecteur. L'attaquant connait
l'architecture generale mais pas les poids : semi-white box.

Deux echantillons, comme dans la section "Adversarial examples generation" :
  - 40 000 du TRAIN (10 000 par attaque, 20 000 benin) -> servira aux defenses
  - 20 000 du TEST (5 000 par attaque) -> sert a l'evaluation

Ils disent 50 % benin pour l'entrainement. Pour le test ils ne le disent pas,
on prend 50 % par symetrie (voir evaluation.benign_fraction).

Les six attaques sont generees sur les deux echantillons, avec les parametres
de la Table 2.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

# ART 1.18 appelle np.product, supprime dans NumPy 2.x. A patcher avant
# l'import d'ART.
if not hasattr(np, "product"):
    np.product = np.prod

import torch
import torch.nn as nn
import torchattacks
from art.attacks.evasion import CarliniL2Method, DeepFool, SaliencyMapMethod
from art.estimators.classification import PyTorchClassifier
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.dnn import Substitut
from src.utils.commun import charger_donnees, predire, metriques
from src.utils.config import load_config

ATTAQUES = ["FGSM", "BIM", "PGD", "DeepFool", "JSMA", "CW"]


def entrainer_substitut(cfg, X_tr, y_tr, X_va, y_va, device):
    """lr 0.01, 30 epochs, batch 256, comme le papier. Cible F1 0.98."""
    s = cfg.substitute
    model = Substitut(input_dim=cfg.dataset["num_features"],
                      hidden=tuple(s["hidden_layers"]),
                      output_dim=cfg.dataset["num_classes"]).to(device)
    print(f"Substitut : {sum(p.numel() for p in model.parameters()):,} parametres")
    print(f"  {s['hidden_layers']}, lr {s['learning_rate']}, "
          f"{s['epochs']} epochs, batch {s['batch_size']}")

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=s["batch_size"], shuffle=True, num_workers=4, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=s["learning_rate"])
    labels = list(range(cfg.dataset["num_classes"]))

    for epoch in range(1, s["epochs"] + 1):
        model.train()
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()

        if epoch % 10 == 0 or epoch == s["epochs"]:
            m = metriques(y_va, predire(model, X_va, device), labels)
            print(f"  epoch {epoch:2d} : F1 pondere val {m['f1_weighted']:.4f}")

    m = metriques(y_va, predire(model, X_va, device), labels)
    f1 = m["f1_weighted"]
    print(f"\n  F1 pondere final : {f1:.4f} (cible du papier : {s['f1_cible']})")
    if f1 < s["f1_cible"] - 0.02:
        print(f"  Sous la cible de plus de 2 points. On continue quand meme,")
        print(f"  l'ecart est une information sur leur protocole.")
    return model, f1


def tirer_echantillon(X, y, n_total, fraction_benin, seed, nom):
    """
    Echantillon a proportion de benin fixee.

    Le papier : "40,000 samples [...] of which 20,000 represent regular
    traffic and the remaining represent intrusions". Donc moitie benin,
    moitie attaques, ces dernieres tirees a parts egales entre classes.
    """
    rng = np.random.default_rng(seed)
    n_benin = int(n_total * fraction_benin)
    n_attaque = n_total - n_benin

    idx_benin = np.where(y == 0)[0]
    idx_attaque = np.where(y != 0)[0]

    if len(idx_benin) < n_benin:
        raise RuntimeError(f"{nom} : {len(idx_benin)} benin disponibles, "
                           f"{n_benin} demandes")

    pris = [rng.choice(idx_benin, size=n_benin, replace=False)]

    # Les classes d'attaque sont tirees a parts egales, dans la limite de ce
    # qui existe. Les classes rares sont prises entieres.
    classes = sorted(set(y[idx_attaque]))
    par_classe = n_attaque // len(classes)
    reste = []
    for c in classes:
        idx_c = np.where(y == c)[0]
        n = min(par_classe, len(idx_c))
        pris.append(rng.choice(idx_c, size=n, replace=False))
        if len(idx_c) > n:
            reste.append(np.setdiff1d(idx_c, pris[-1]))

    # Completer si les classes rares n'ont pas fourni leur quota
    manque = n_total - sum(len(p) for p in pris)
    if manque > 0 and reste:
        dispo = np.concatenate(reste)
        pris.append(rng.choice(dispo, size=min(manque, len(dispo)),
                               replace=False))

    idx = np.sort(np.concatenate(pris))
    n_b = int((y[idx] == 0).sum())
    print(f"  {nom} : {len(idx):,} lignes, {n_b:,} benin "
          f"({100*n_b/len(idx):.1f}%)")
    return idx


def generer(nom, model, art_clf, X, y, cfg, device, batch_size):
    """Une attaque, parametres de la Table 2."""
    p = cfg.attacks[nom]
    n = len(X)
    X_adv = np.zeros_like(X)
    t0 = time.time()

    if nom == "FGSM":
        atk = torchattacks.FGSM(model, eps=p["eps"])
    elif nom == "BIM":
        atk = torchattacks.BIM(model, eps=p["eps"], alpha=p["alpha"],
                               steps=p["steps"])
    elif nom == "PGD":
        atk = torchattacks.PGD(model, eps=p["eps"], alpha=p["alpha"],
                               steps=p["steps"])
    elif nom == "DeepFool":
        atk = DeepFool(classifier=art_clf, epsilon=p["epsilon"],
                       max_iter=p["max_iter"], batch_size=batch_size,
                       verbose=False)
    elif nom == "JSMA":
        atk = SaliencyMapMethod(classifier=art_clf, theta=p["theta"],
                                gamma=p["gamma"], batch_size=batch_size,
                                verbose=False)
    elif nom == "CW":
        atk = CarliniL2Method(classifier=art_clf, max_iter=p["max_iter"],
                              confidence=p["confidence"],
                              binary_search_steps=p["binary_search_steps"],
                              initial_const=p["initial_const"],
                              learning_rate=p["learning_rate"],
                              batch_size=batch_size, verbose=False)
    else:
        raise ValueError(nom)

    for i in range(0, n, batch_size):
        fin = min(i + batch_size, n)
        if nom in ("FGSM", "BIM", "PGD"):
            xb = torch.tensor(X[i:fin], dtype=torch.float32).to(device)
            yb = torch.tensor(y[i:fin], dtype=torch.long).to(device)
            X_adv[i:fin] = atk(xb, yb).detach().cpu().numpy()
        else:
            # y=None : untargeted. Passer les vrais labels comme cible rend
            # l'attaque triviale, le modele les predit deja.
            X_adv[i:fin] = atk.generate(x=X[i:fin], y=None)

    bas, haut = cfg.clip_values
    X_adv = np.clip(X_adv, bas, haut)
    delta = np.abs(X_adv - X)
    print(f"  {nom:<9} {(time.time()-t0)/60:5.1f} min | "
          f"L-inf {delta.max(axis=1).mean():.4f} | "
          f"L2 {np.linalg.norm(delta, axis=1).mean():.4f} | "
          f"features touchees {(delta > 1e-6).sum(axis=1).mean():.1f}/58")
    return X_adv


def main():
    print("=" * 70)
    print("Substitut et attaques - reproduction fidele")
    print(f"Date : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 70 + "\n")

    cfg = load_config()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}\n")

    X_tr, y_tr, X_va, y_va, X_te, y_te = charger_donnees(cfg.paths["processed"])
    atk_dir = Path(cfg.paths["attacks"])
    atk_dir.mkdir(parents=True, exist_ok=True)

    substitut, f1_sub = entrainer_substitut(cfg, X_tr, y_tr, X_va, y_va, device)
    substitut.eval()
    torch.save({"model_state_dict": substitut.state_dict(), "f1_weighted": f1_sub},
               Path(cfg.paths["checkpoints"]) / "substitut.pth")
    print()

    art_clf = PyTorchClassifier(
        model=substitut, loss=nn.CrossEntropyLoss(),
        input_shape=(cfg.dataset["num_features"],),
        nb_classes=cfg.dataset["num_classes"],
        clip_values=cfg.clip_values,
        device_type="gpu" if device.type == "cuda" else "cpu")

    e = cfg.evaluation
    print("Echantillons :")
    # 40 000 du train pour les defenses, 20 000 du test pour l'evaluation
    idx_tr = tirer_echantillon(X_tr, y_tr, 40000, 0.5, cfg.seed, "train")
    idx_te = tirer_echantillon(X_te, y_te, e["n_samples"],
                               e["benign_fraction"], cfg.seed, "test")
    print()

    for nom_ech, X, y, idx in (("train", X_tr, y_tr, idx_tr),
                               ("test", X_te, y_te, idx_te)):
        X_ech = np.ascontiguousarray(X[idx])
        y_ech = np.ascontiguousarray(y[idx])
        joblib.dump(X_ech, atk_dir / f"X_{nom_ech}_clean.pkl")
        joblib.dump(y_ech, atk_dir / f"y_{nom_ech}_clean.pkl")

        print(f"Attaques sur l'echantillon {nom_ech} ({len(X_ech):,} lignes) :")
        for nom in ATTAQUES:
            chemin = atk_dir / f"X_adv_{nom_ech}_{nom.lower()}.pkl"
            if chemin.exists():
                print(f"  {nom:<9} deja genere")
                continue
            X_adv = generer(nom, substitut, art_clf, X_ech, y_ech, cfg,
                            device, e["batch_size"])
            joblib.dump(X_adv, chemin)
            del X_adv
        print()

    print("Termine")


if __name__ == "__main__":
    main()
