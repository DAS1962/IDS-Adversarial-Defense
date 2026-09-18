"""
Defense par autoencodeur debruiteur. Algorithme 5 du papier.

"the DAE is trained on a combination of clean and adversarial samples,
enabling it to learn robust representations that effectively filter out
adversarial noise. Once trained, the DAE acts as a denoiser, processing
perturbed test samples and producing denoised outputs."

Contrairement aux trois autres, le detecteur n'est PAS reentraine. Le DAE
nettoie l'entree, puis le detecteur d'origine classe la version nettoyee.

Deux points que le papier ne documente pas :
  - la taille du goulot : on prend 32
  - la proportion de clean dans le melange : on prend 50 %

Le melange clean/adversarial est essentiel. Sur la branche main, une version
entrainee uniquement sur du corrompu detruisait 31.5 % de l'information d'une
entree SAINE : le reseau n'avait jamais appris a laisser passer une entree
propre, donc il la "corrigeait" aussi.
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
from src.defenses.commun import (ATTAQUES, QUATRE, baseline_reference,
                                 evaluer, nettoyer, resume)
from src.models.dnn import DAE, DNN
from src.utils.commun import charger_donnees, metriques, predire
from src.utils.config import load_config


def charger_detecteur(cfg, device, quel="varlr"):
    """Le detecteur d'origine, que le DAE protege sans le modifier."""
    fichier = "baseline_varlr.pth" if quel == "varlr" else "baseline.pth"
    ck = torch.load(Path(cfg.paths["checkpoints"]) / fichier,
                    weights_only=False, map_location=device)
    model = DNN(input_dim=cfg.dataset["num_features"],
                hidden=tuple(cfg.model["hidden_layers"]),
                output_dim=cfg.dataset["num_classes"]).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model


def construire_paires(atk_dir, clean_ratio):
    """
    Paires (entree, cible) : le DAE apprend a retrouver l'echantillon propre
    a partir de sa version attaquee.

    Une fraction du jeu est composee de paires (propre, propre) pour que le
    reseau apprenne aussi a ne rien faire quand l'entree est deja saine.
    """
    X_clean = joblib.load(atk_dir / "X_train_clean.pkl")
    n = len(X_clean)

    entrees, cibles = [], []
    print("  Paires d'entrainement :")
    for nom in QUATRE:
        X_adv = joblib.load(atk_dir / f"X_adv_train_{nom}.pkl")
        mse = ((X_adv - X_clean) ** 2).mean()
        entrees.append(X_adv)
        cibles.append(X_clean)
        print(f"    {nom:<9} {n:,} paires, MSE de corruption {mse:.6f}")

    n_clean = int(n * len(QUATRE) * clean_ratio / (1 - clean_ratio))
    n_clean = min(n_clean, n * len(QUATRE))
    rng = np.random.default_rng(42)
    idx = rng.choice(n, size=min(n_clean, n), replace=False)
    entrees.append(X_clean[idx])
    cibles.append(X_clean[idx])
    print(f"    {'propre':<9} {len(idx):,} paires, aucune corruption")

    X_in = np.concatenate(entrees)
    X_out = np.concatenate(cibles)
    print(f"    total {len(X_in):,} paires, "
          f"{100*len(idx)/len(X_in):.1f}% propres")
    print(f"    variance des donnees : {X_clean.var():.6f}")
    return X_in, X_out


def entrainer_dae(dae, X_in, X_out, cfg, device, detecteur, X_val, y_val,
                  labels):
    """
    Entrainement du DAE. La loss est la MSE de reconstruction plus une
    penalite L1 sur le latent, qui rend les activations creuses.

    Le modele retenu est celui qui maximise le F1 macro du detecteur sur la
    validation propre. Optimiser la MSE seule choisirait un reseau qui
    reconstruit bien mais ne sert pas la classification.
    """
    d = cfg.defenses["DenoisingAutoencoder"]
    var = cfg.model.get("variante_lr", {})
    lr = var["learning_rate"]
    epochs = var.get("epochs", cfg.model["epochs"])
    l1 = d["l1_reg"]
    bas, haut = cfg.clip_values

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_in), torch.from_numpy(X_out)),
        batch_size=cfg.model["batch_size"], shuffle=True, num_workers=4,
        pin_memory=True)

    optimizer = torch.optim.Adam(dae.parameters(), lr=lr)
    mse = nn.MSELoss()
    bs = cfg.evaluation["batch_size"]

    # Reference : le detecteur seul sur la validation propre. Si le DAE fait
    # moins bien que ca, il nuit plus qu'il n'aide.
    f1_ref = metriques(y_val, predire(detecteur, X_val, device, bs),
                       labels)["f1_macro"]
    print(f"    reference sans DAE, F1 macro validation : {f1_ref:.4f}")

    historique = {"mse": [], "l1": [], "f1_val": []}
    meilleur, meilleur_ep, etat = -1.0, -1, None
    debut = time.time()

    for epoch in range(1, epochs + 1):
        dae.train()
        somme_mse, somme_l1, vus = 0.0, 0.0, 0
        for xin, xout in loader:
            xin = xin.to(device, non_blocking=True)
            xout = xout.to(device, non_blocking=True)
            optimizer.zero_grad()
            latent = dae.encodeur(xin)
            sortie = dae.decodeur(latent)
            perte_mse = mse(sortie, xout)
            perte_l1 = l1 * latent.abs().mean()
            (perte_mse + perte_l1).backward()
            optimizer.step()
            somme_mse += perte_mse.item() * xin.size(0)
            somme_l1 += perte_l1.item() * xin.size(0)
            vus += xin.size(0)

        X_net = nettoyer(dae, X_val, device, bs, (bas, haut))
        f1 = metriques(y_val, predire(detecteur, X_net, device, bs),
                       labels)["f1_macro"]
        historique["mse"].append(somme_mse / vus)
        historique["l1"].append(somme_l1 / vus)
        historique["f1_val"].append(f1)

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(f"    epoch {epoch:3d}/{epochs} | mse={somme_mse/vus:.6f} "
                  f"| F1 val {f1:.4f} (ref {f1_ref:.4f})", flush=True)

        if f1 > meilleur:
            meilleur, meilleur_ep = f1, epoch
            etat = {k: v.detach().clone() for k, v in dae.state_dict().items()}

    dae.load_state_dict(etat)
    duree = (time.time() - debut) / 60
    print(f"    termine en {duree:.1f} min, meilleur epoch {meilleur_ep} "
          f"(F1 val {meilleur:.4f})")
    if meilleur < f1_ref:
        print(f"    ATTENTION : le DAE degrade la validation propre "
              f"({meilleur:.4f} < {f1_ref:.4f})")
    return historique, meilleur_ep, meilleur, duree


def diagnostic(dae, X, device, bs, clip_values):
    """
    Ce que le DAE fait a une entree SAINE. Mesure prise sur la branche main,
    ou la version 3 detruisait 31.5 % de la variance d'une entree propre.
    """
    X_net = nettoyer(dae, X, device, bs, clip_values)
    mse = ((X_net - X) ** 2).mean()
    var = X.var()
    perte_var = (X.std(axis=0) - X_net.std(axis=0)) / (X.std(axis=0) + 1e-9)
    print(f"  MSE(x, DAE(x)) sur donnees propres : {mse:.6f} "
          f"({100*mse/var:.1f} % de la variance)")
    print(f"  features perdant plus de la moitie de leur ecart-type : "
          f"{int((perte_var > 0.5).sum())}/{X.shape[1]}")


def main():
    print("=" * 78)
    print("Defense DAE - autoencodeur debruiteur (Algorithme 5)")
    print(f"Date : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 78 + "\n")

    cfg = load_config()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = list(range(cfg.dataset["num_classes"]))
    atk_dir = Path(cfg.paths["attacks"])

    _, _, X_va, y_va, _, _ = charger_donnees(cfg.paths["processed"])
    d = cfg.defenses["DenoisingAutoencoder"]

    detecteur = charger_detecteur(cfg, device)
    print("Detecteur charge, il n'est PAS reentraine\n")

    X_in, X_out = construire_paires(atk_dir, d["clean_ratio"])
    print()

    dae = DAE(input_dim=cfg.dataset["num_features"],
              bottleneck=d["hidden_dim"]).to(device)
    n_par = sum(p.numel() for p in dae.parameters())
    print(f"DAE : {dae.dims[0]}-{dae.dims[1]}-{dae.dims[2]}-{dae.dims[1]}-"
          f"{dae.dims[0]}, {n_par:,} parametres, L1 = {d['l1_reg']}\n")

    print("Entrainement :")
    hist, best_ep, best_f1, duree = entrainer_dae(
        dae, X_in, X_out, cfg, device, detecteur, X_va, y_va, labels)
    print()

    print("Diagnostic sur entree saine :")
    diagnostic(dae, X_va, device, cfg.evaluation["batch_size"], cfg.clip_values)
    print()

    res = evaluer(detecteur, cfg, device, labels, atk_dir, dae=dae)
    base, nom_base = baseline_reference(cfg.paths["logs"])
    print(f"Reference : {nom_base}\n")
    bilan = resume(res, base, "Denoising Autoencoder")

    ck = Path(cfg.paths["checkpoints"]) / "defense_dae.pth"
    torch.save({"model_state_dict": dae.state_dict(), "epoch": best_ep,
                "f1_val": best_f1, "dims": dae.dims}, ck)
    sortie = (Path(cfg.paths["logs"]) /
              f"defense_dae_{datetime.now():%Y%m%d_%H%M%S}.pkl")
    joblib.dump({"resultats": res, "historique": hist, "defense": "dae",
                 "meilleur_epoch": best_ep, "duree_min": duree,
                 "bilan": bilan, "config": dict(d)}, sortie)
    print(f"\nCheckpoint : {ck}")
    print(f"Resultats  : {sortie}")


if __name__ == "__main__":
    main()
