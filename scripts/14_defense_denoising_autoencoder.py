"""
Defense par Denoising Autoencoder (DAE) - version 6.

Un autoencodeur apprend a reconstruire l'echantillon propre depuis une
version adversariale. A l'inference, chaque entree passe d'abord par le DAE
pour etre purifiee, puis est classifiee par le baseline deja entraine. Le
classifieur n'est PAS reentraine.

Reference : Vincent et al. 2008 ; Gu & Rigazio 2014 ; Awad et al. 2025
(Algorithme 5).

Historique
----------
v1  bruit gaussien seul, 58 -> ReLU(32) -> 58.
    F1 clean 0.2081 | gain attaques -0.036
v2  + FGSM eps=0.2 regenere a la volee.
    F1 clean 0.1577 | gain -0.073
v3  + profondeur, goulot LINEAIRE sans ReLU (51.5% du latent est negatif,
    un ReLU en detruisait la moitie), adv_eps=0.05.
    F1 clean 0.1820 | gain -0.053
v4  + connexion residuelle initialisee a l'identite et fraction propre dans
    le batch (le reseau n'avait jamais vu d'entree non corrompue et
    sur-corrigeait : MSE 0.00719 entre x et DAE(x) sur donnees propres).
    F1 clean 0.8411 | gain 0.0000 : le DAE converge vers l'identite.
v5  + les quatre attaques REELLES generees sur le train par 08b, et
    reconstruction complete sans residu (le residu court-circuitait la
    projection sur la variete apprise, qui est le mecanisme de purification).
    F1 clean 0.4498 | gain +0.0737 : la defense fonctionne enfin, mais au
    prix de -0.39 sur les donnees propres.

Diagnostic de la v5
-------------------
Le DAE reduisait la MSE de 78.9% par rapport a un DAE inactif, donc il
apprenait correctement - la capacite n'etait pas la limite. Le probleme
etait la ponderation de la loss.

Amplitudes de corruption mesurees par 08b, converties en MSE par feature
(L2^2 / 58) :

    jsma      0.761196   soit 33.4x la variance des donnees (0.022818)
    fgsm      0.006443
    bim       0.002511
    deepfool  0.000037

Avec un batch a 50% propre et 12.5% par attaque, la MSE brute se
decomposait ainsi :

    jsma      98.8% de la loss   pour un gain en F1 macro de +0.002
    fgsm       0.8% de la loss   pour un gain de +0.168
    bim        0.3% de la loss   pour un gain de +0.137
    deepfool   0.0% de la loss   pour un gain de +0.103

Un seul echantillon JSMA pesait autant que cent echantillons FGSM. Le DAE
consacrait donc la quasi-totalite de son effort d'optimisation a l'unique
attaque qu'il ne parvient pas a corriger, et negligeait les trois qui
repondent. La projection tres agressive qui en resultait expliquait aussi
la chute de 0.39 sur les donnees propres.

Ce que fait cette version
--------------------------
1. LOSS NORMALISEE PAR GROUPE. L'erreur de chaque groupe (propre, fgsm,
   bim, deepfool, jsma) est divisee par l'amplitude de corruption de ce
   groupe, mesuree sur les donnees, puis les cinq pertes sont moyennees.
   Chaque groupe pese donc exactement 20% du gradient, quelle que soit son
   amplitude. La perte par groupe s'interprete comme la fraction de
   corruption non retiree, dans [0,1].

   Pour le groupe propre il n'y a pas de corruption : l'echelle est la
   variance des donnees, et la perte s'interprete comme la fraction de
   variance detruite par un pretraitement inutile.

   L'Algorithme 5 impose de concatener les quatre attaques aux echantillons
   reels, ce qui reste le cas. Il ne specifie pas la ponderation de
   l'esperance dans min E[||g(x + x_adv) - x||^2].

2. RESEAU PLUS LARGE : goulot 40 au lieu de 32, deux couches cachees par
   cote au lieu d'une. 58-52-46-40-46-52-58, environ 14 800 parametres
   contre 8 280. Test de l'hypothese d'une limite de capacite.

Le journal affiche la perte par groupe a chaque epoch, ce qui rend
directement visible si un groupe accapare encore l'optimisation.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.config import load_config, check_data_fingerprint
from src.defenses.common import (
    load_splits,
    load_baseline_model,
    evaluate_on_attacks,
    print_summary,
)


# Attaques de l'Algorithme 5, dans l'ordre de l'article. PGD et C&W sont
# volontairement absents : ils ne figurent pas dans sa liste. Cela permet de
# mesurer si la purification transfere a des attaques non vues - en v5,
# PGD gagnait +0.089 et C&W perdait 0.057.
ATTAQUES_ENTRAINEMENT = ["fgsm", "bim", "deepfool", "jsma"]

# Groupe 0 = echantillons propres, groupes 1..4 = les quatre attaques.
GROUPE_PROPRE = 0

# Variance moyenne des features de X_train, mesuree sur 300 000 lignes.
# Sert d'echelle de normalisation pour le groupe propre.
VARIANCE_DONNEES = 0.022818


class DenoisingAutoencoder(nn.Module):
    """
    Autoencodeur profond a goulot lineaire, reconstruction complete.

    Encodeur : input_dim -> h1 -> h2 -> bottleneck_dim
    Decodeur : bottleneck_dim -> h2 -> h1 -> input_dim, puis bornage.
    ReLU sur les couches intermediaires uniquement.

    Pas de connexion residuelle : la sortie est decodeur(encodeur(x)) et non
    x + correction. C'est ce qui donne au DAE son pouvoir de purification -
    l'entree est projetee sur la variete apprise, et ce qui n'appartient pas
    a cette variete disparait. La v4, residuelle, ne pouvait qu'ajouter une
    correction a une entree laissee intacte, et convergeait vers l'identite.

    h1 et h2 sont derives de input_dim et bottleneck_dim par interpolation
    lineaire, plutot que passes en parametres : 16_ensemble_aggregation.py
    reconstruit cette architecture pour charger le state_dict, et tout
    parametre supplementaire devrait y etre propage sous peine d'echec
    obscur au chargement. Les deriver garantit que les deux scripts restent
    d'accord.

    Le goulot n'a pas d'activation. La mesure sur v3 montre 51.5% du latent
    negatif ; un ReLU en detruirait la moitie.
    """

    def __init__(self, input_dim=58, bottleneck_dim=40, clip_values=(0.0, 1.0)):
        super().__init__()
        self.borne_min, self.borne_max = clip_values
        ecart = input_dim - bottleneck_dim
        h1 = input_dim - ecart // 3
        h2 = input_dim - 2 * ecart // 3

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, h2),
            nn.ReLU(),
            nn.Linear(h2, h1),
            nn.ReLU(),
            nn.Linear(h1, input_dim),
        )
        self.dims = (input_dim, h1, h2, bottleneck_dim)

    def forward(self, x):
        latent = self.encoder(x)
        sortie = torch.clamp(self.decoder(latent), self.borne_min, self.borne_max)
        return sortie, latent


def charger_attaques_train(attacks_dir):
    """
    Charge le sous-echantillon propre du train, les quatre jeux d'exemples
    adversariaux produits par 08b, et l'echelle de corruption de chacun.

    L'echelle est la MSE par feature qu'un DAE inactif subirait sur ce
    groupe. Elle sert a normaliser la loss : sans elle, JSMA representait
    98.8% du gradient.

    Verifie que les cinq tableaux ont la meme longueur : chaque ligne i de
    X_adv doit correspondre a la ligne i de X_clean, sinon l'appariement
    entree corrompue / cible propre est faux et le DAE apprend n'importe
    quoi.
    """
    train_dir = attacks_dir / "train"
    if not train_dir.exists():
        raise RuntimeError(
            f"{train_dir} introuvable. Lancer d'abord "
            f"scripts/08b_generate_train_attacks.py."
        )

    chemin_clean = train_dir / "X_train_clean_sub.pkl"
    chemin_y = train_dir / "y_train_clean_sub.pkl"
    for c in (chemin_clean, chemin_y):
        if not c.exists():
            raise RuntimeError(
                f"{c} introuvable. Relancer 08b_generate_train_attacks.py."
            )

    X_clean = np.ascontiguousarray(joblib.load(chemin_clean), dtype=np.float32)
    y_clean = np.asarray(joblib.load(chemin_y))
    n = len(X_clean)
    print(f"  sous-echantillon propre : {X_clean.shape}")

    attaques, echelles = {}, {}
    for nom in ATTAQUES_ENTRAINEMENT:
        chemin = train_dir / f"X_adv_train_{nom}.pkl"
        if not chemin.exists():
            raise RuntimeError(
                f"{chemin} introuvable. L'Algorithme 5 demande les quatre "
                f"attaques (FGSM, BIM, DeepFool, JSMA). Relancer "
                f"08b_generate_train_attacks.py."
            )
        X_adv = np.ascontiguousarray(joblib.load(chemin), dtype=np.float32)
        if len(X_adv) != n:
            raise RuntimeError(
                f"{chemin.name} a {len(X_adv):,} lignes contre {n:,} pour le "
                f"sous-echantillon propre. L'appariement corrompu/propre "
                f"serait faux. Relancer 08b_generate_train_attacks.py."
            )
        ecart = X_adv - X_clean
        mse_corruption = float((ecart ** 2).mean())
        echelles[nom] = mse_corruption
        delta = np.abs(ecart)
        print(f"  {nom:<9} : L2 moyen {np.linalg.norm(delta, axis=1).mean():.4f}, "
              f"MSE de corruption {mse_corruption:.6f} "
              f"({mse_corruption/VARIANCE_DONNEES:6.1f}x la variance)")
        attaques[nom] = X_adv

    echelles["clean"] = VARIANCE_DONNEES
    print(f"  {'clean':<9} : aucune corruption, echelle = variance des "
          f"donnees {VARIANCE_DONNEES:.6f}")
    print()
    return X_clean, y_clean, attaques, echelles


def construire_dataset(X_clean, attaques, clean_ratio, seed, verbose=False):
    """
    Construit le jeu d'entrainement : paires (entree, cible propre, groupe).

    Chaque echantillon apparait une fois, soit tel quel (groupe 0) avec
    probabilite clean_ratio, soit corrompu par l'une des quatre attaques
    tiree uniformement (groupes 1 a 4). Le groupe est necessaire pour
    normaliser la loss par amplitude de corruption.
    """
    n = len(X_clean)
    rng = np.random.default_rng(seed)
    noms = list(attaques.keys())

    est_propre = rng.random(n) < clean_ratio
    quelle_attaque = rng.integers(0, len(noms), size=n)

    X_entree = np.empty_like(X_clean)
    groupe = np.zeros(n, dtype=np.int64)
    X_entree[est_propre] = X_clean[est_propre]

    for k, nom in enumerate(noms):
        masque = (~est_propre) & (quelle_attaque == k)
        X_entree[masque] = attaques[nom][masque]
        groupe[masque] = k + 1

    if verbose:
        n_propre = int(est_propre.sum())
        print(f"  groupe 0 (propre)   : {n_propre:,} ({100*n_propre/n:.1f}%)")
        for k, nom in enumerate(noms):
            m = int((groupe == k + 1).sum())
            print(f"  groupe {k+1} ({nom:<8}) : {m:,} ({100*m/n:.1f}%)")
        print()

    return X_entree, groupe


def loss_normalisee(sortie, cible, groupe, echelles_tensor, n_groupes):
    """
    Moyenne des pertes par groupe, chacune normalisee par l'amplitude de
    corruption de son groupe.

    loss = moyenne_sur_les_groupes_presents( MSE(groupe) / echelle(groupe) )

    Chaque groupe pese donc 1/n_groupes du gradient quelle que soit son
    amplitude. Sans cette normalisation, JSMA (MSE de corruption 0.761,
    soit 33x la variance des donnees) representait 98.8% de la loss pour un
    gain en F1 macro de +0.002.

    Seuls les groupes presents dans le batch sont moyennes : avec un batch
    de 128 et cinq groupes, l'un peut etre absent, et le compter comme une
    perte nulle biaiserait la moyenne vers le bas.
    """
    err = ((sortie - cible) ** 2).mean(dim=1)          # MSE par echantillon
    pertes, presents = [], []

    for g in range(n_groupes):
        masque = (groupe == g)
        if not masque.any():
            continue
        pertes.append(err[masque].mean() / echelles_tensor[g])
        presents.append(g)

    return torch.stack(pertes).mean(), presents


@torch.no_grad()
def pertes_par_groupe(dae, X_entree, X_clean, groupe, echelles_tensor,
                      n_groupes, device, batch_size):
    """
    Perte normalisee de chaque groupe, evaluee sur un echantillon.

    Sert de diagnostic : rend visible si un groupe accapare encore
    l'optimisation, ou si le DAE progresse sur certains et pas sur
    d'autres.
    """
    dae.eval()
    sommes = np.zeros(n_groupes)
    effectifs = np.zeros(n_groupes, dtype=np.int64)

    for i in range(0, len(X_entree), batch_size):
        fin = min(i + batch_size, len(X_entree))
        xe = torch.tensor(X_entree[i:fin], dtype=torch.float32).to(device)
        xc = torch.tensor(X_clean[i:fin], dtype=torch.float32).to(device)
        g = torch.tensor(groupe[i:fin], dtype=torch.long).to(device)
        sortie, _ = dae(xe)
        err = ((sortie - xc) ** 2).mean(dim=1)
        for k in range(n_groupes):
            m = (g == k)
            if m.any():
                sommes[k] += float(err[m].sum())
                effectifs[k] += int(m.sum())

    resultat = {}
    for k in range(n_groupes):
        if effectifs[k] > 0:
            resultat[k] = (sommes[k] / effectifs[k]) / float(echelles_tensor[k])
    return resultat


@torch.no_grad()
def f1_pipeline(dae, classifier, X, y, device, batch_size, all_labels):
    """F1 macro du pipeline x -> DAE -> baseline."""
    dae.eval()
    classifier.eval()
    n = len(X)
    preds = np.zeros(n, dtype=np.int64)
    for i in range(0, n, batch_size):
        fin = min(i + batch_size, n)
        xb = torch.tensor(X[i:fin], dtype=torch.float32).to(device)
        x_pur, _ = dae(xb)
        preds[i:fin] = classifier(x_pur).argmax(dim=1).cpu().numpy()
    return f1_score(y, preds, labels=all_labels, average="macro",
                    zero_division=0)


@torch.no_grad()
def f1_direct(classifier, X, y, device, batch_size, all_labels):
    """F1 macro du baseline seul, sans purification."""
    classifier.eval()
    n = len(X)
    preds = np.zeros(n, dtype=np.int64)
    for i in range(0, n, batch_size):
        fin = min(i + batch_size, n)
        xb = torch.tensor(X[i:fin], dtype=torch.float32).to(device)
        preds[i:fin] = classifier(xb).argmax(dim=1).cpu().numpy()
    return f1_score(y, preds, labels=all_labels, average="macro",
                    zero_division=0)


def main():
    print("=" * 70)
    print("Defense par Denoising Autoencoder (DAE) - v6")
    print("Loss normalisee par groupe, reseau elargi")
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

    _, _, X_val, y_val, X_test, y_test = load_splits(data_dir)
    classifier = load_baseline_model(cfg, device, checkpoint_dir)

    print("Chargement des exemples adversariaux du train (08b)...")
    X_clean, y_clean, attaques, echelles = charger_attaques_train(attacks_dir)

    batch_size = cfg.training["batch_size"]
    eval_batch = cfg.evaluation["batch_size"]
    epochs = cfg.training["epochs"]
    lr_init = cfg.training["learning_rate"]
    sched = cfg.training["scheduler"]
    dae_cfg = dict(cfg.defenses["DenoisingAutoencoder"])
    bottleneck = dae_cfg["hidden_dim"]
    l1_lambda = dae_cfg["l1_reg"]
    clean_ratio = dae_cfg["clean_ratio"]
    normaliser = dae_cfg.get("loss_normalisee", True)
    clip_values = cfg.clip_values
    all_labels = list(range(cfg.dataset["num_classes"]))

    noms_groupes = ["clean"] + ATTAQUES_ENTRAINEMENT
    n_groupes = len(noms_groupes)
    echelles_tensor = torch.tensor(
        [echelles[n] for n in noms_groupes], dtype=torch.float32,
    ).to(device)

    dae = DenoisingAutoencoder(
        input_dim=cfg.dataset["num_features"],
        bottleneck_dim=bottleneck,
        clip_values=clip_values,
    ).to(device)
    e, h1, h2, b = dae.dims
    print("Denoising Autoencoder, reconstruction complete")
    print(f"  sortie = clamp(decodeur(encodeur(x)))  (pas de residu)")
    print(f"  Encodeur : {e} -> ReLU({h1}) -> ReLU({h2}) -> {b}")
    print(f"  Decodeur : {b} -> ReLU({h2}) -> ReLU({h1}) -> {e}")
    print(f"  Parametres : {sum(p.numel() for p in dae.parameters()):,} "
          f"(v5 : 8,280)\n")

    print("References du baseline seul (sans DAE) :")
    rng = np.random.default_rng(cfg.seed)
    idx_val = np.sort(rng.choice(len(X_val), size=min(60000, len(X_val)),
                                 replace=False))
    X_val_sub = np.ascontiguousarray(X_val[idx_val])
    y_val_sub = np.asarray(y_val)[idx_val]
    ref_clean_val = f1_direct(classifier, X_val_sub, y_val_sub, device,
                              eval_batch, all_labels)
    print(f"  validation propre        : F1 macro {ref_clean_val:.4f}")

    n_diag = min(40000, len(X_clean))
    rng = np.random.default_rng(cfg.seed + 1)
    idx_diag = np.sort(rng.choice(len(X_clean), size=n_diag, replace=False))
    y_diag = y_clean[idx_diag]
    X_diag = {nom: np.ascontiguousarray(attaques[nom][idx_diag])
              for nom in attaques}

    refs_attaque = {}
    for nom in ATTAQUES_ENTRAINEMENT:
        refs_attaque[nom] = f1_direct(classifier, X_diag[nom], y_diag, device,
                                      eval_batch, all_labels)
        print(f"  train attaque par {nom:<9}: F1 macro {refs_attaque[nom]:.4f}")
    print()

    optimizer = torch.optim.Adam(dae.parameters(), lr=lr_init)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=sched["factor"],
        patience=sched["patience"], min_lr=sched["min_lr"],
    )

    print("Entrainement du DAE")
    print(f"  Epochs             : {epochs}")
    print(f"  Learning rate init : {lr_init}")
    print(f"  Paires propres     : {clean_ratio:.0%}")
    print(f"  Attaques           : {', '.join(ATTAQUES_ENTRAINEMENT)}")
    print(f"  Loss               : "
          f"{'normalisee par groupe (chacun 20%)' if normaliser else 'MSE brute'}")
    print(f"  L1 lambda          : {l1_lambda}")
    print(f"  Batch size         : {batch_size}")
    print("  Selection modele   : F1 macro moyen (validation propre + "
          "4 attaques du train)\n")

    history = {"train_loss": [], "val_f1_clean": [], "f1_moyen": [], "lr": []}
    for nom in ATTAQUES_ENTRAINEMENT:
        history[f"f1_{nom}"] = []
    for nom in noms_groupes:
        history[f"perte_{nom}"] = []

    ref_moyenne = float(np.mean([ref_clean_val] + list(refs_attaque.values())))
    print(f"Reference a battre : F1 macro moyen {ref_moyenne:.4f}\n")

    best_moyenne, best_epoch = ref_moyenne, 0
    ckpt_path = checkpoint_dir / "defense_dae_best.pth"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    debut = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        lr_now = optimizer.param_groups[0]["lr"]

        if epoch == 1:
            print("Composition du dataset (reechantillonnee a chaque epoch) :")
        X_entree, groupe = construire_dataset(
            X_clean, attaques, clean_ratio, cfg.seed + epoch,
            verbose=(epoch == 1),
        )

        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_entree),
                          torch.from_numpy(X_clean),
                          torch.from_numpy(groupe)),
            batch_size=batch_size, shuffle=True,
            num_workers=cfg.env["num_workers"], pin_memory=True,
        )

        dae.train()
        total_loss, total_seen = 0.0, 0
        for x_corrompu, x_cible, g in loader:
            x_corrompu = x_corrompu.to(device, non_blocking=True)
            x_cible = x_cible.to(device, non_blocking=True)
            g = g.to(device, non_blocking=True)

            optimizer.zero_grad()
            sortie, latent = dae(x_corrompu)
            if normaliser:
                perte, _ = loss_normalisee(sortie, x_cible, g,
                                           echelles_tensor, n_groupes)
            else:
                perte = ((sortie - x_cible) ** 2).mean()
            l1 = torch.abs(latent).mean()
            (perte + l1_lambda * l1).backward()
            optimizer.step()

            n = x_cible.size(0)
            total_loss += perte.item() * n
            total_seen += n

        train_loss = total_loss / total_seen

        pertes_g = pertes_par_groupe(dae, X_entree[:60000], X_clean[:60000],
                                     groupe[:60000], echelles_tensor,
                                     n_groupes, device, eval_batch)

        f1_clean = f1_pipeline(dae, classifier, X_val_sub, y_val_sub, device,
                               eval_batch, all_labels)
        f1_par_attaque = {}
        for nom in ATTAQUES_ENTRAINEMENT:
            f1_par_attaque[nom] = f1_pipeline(dae, classifier, X_diag[nom],
                                              y_diag, device, eval_batch,
                                              all_labels)
        moyenne = float(np.mean([f1_clean] + list(f1_par_attaque.values())))
        scheduler.step(moyenne)

        history["train_loss"].append(train_loss)
        history["val_f1_clean"].append(f1_clean)
        history["f1_moyen"].append(moyenne)
        history["lr"].append(lr_now)
        for nom in ATTAQUES_ENTRAINEMENT:
            history[f"f1_{nom}"].append(f1_par_attaque[nom])
        for k, nom in enumerate(noms_groupes):
            history[f"perte_{nom}"].append(pertes_g.get(k, float("nan")))

        detail_f1 = " ".join(
            f"{nom[:4]}={f1_par_attaque[nom]:.3f}" for nom in ATTAQUES_ENTRAINEMENT
        )
        detail_perte = " ".join(
            f"{nom[:4]}={pertes_g.get(k, float('nan')):.3f}"
            for k, nom in enumerate(noms_groupes)
        )
        print(f"Epoch {epoch:3d}/{epochs} | lr={lr_now:.6f} | "
              f"loss={train_loss:.4f} | pertes[{detail_perte}] | "
              f"clean={f1_clean:.4f} | {detail_f1} | "
              f"moy={moyenne:.4f} (ref {ref_moyenne:.4f}) | "
              f"{time.time()-t0:.1f}s", flush=True)

        if moyenne > best_moyenne:
            best_moyenne, best_epoch = moyenne, epoch
            torch.save({
                "model_state_dict": dae.state_dict(),
                "epoch": epoch,
                "val_f1_macro": f1_clean,
                "f1_moyen": moyenne,
                "f1_par_attaque": dict(f1_par_attaque),
                "pertes_par_groupe": {noms_groupes[k]: v
                                      for k, v in pertes_g.items()},
                "echelles_corruption": echelles,
                "reference_baseline_seul": {
                    "clean": ref_clean_val, **refs_attaque,
                    "moyenne": ref_moyenne,
                },
                "config_hash": cfg.baseline_fingerprint(),
                "defense": "DenoisingAutoencoder",
                "dae_config": dae_cfg,
                "architecture": f"{e}-{h1}-{h2}-{b}-{h2}-{h1}-{e}",
                "residuel": False,
                "loss_normalisee": normaliser,
                "attaques_entrainement": ATTAQUES_ENTRAINEMENT,
                "clip_values": list(clip_values),
            }, ckpt_path)
            print(f"            -> meilleur DAE sauvegarde "
                  f"(moyenne={moyenne:.4f})", flush=True)

    duree = (time.time() - debut) / 60
    print(f"\nEntrainement termine en {duree:.1f} min")

    if best_epoch == 0:
        print("Aucune epoch n'a depasse le baseline seul.")
        if not ckpt_path.exists():
            print("Aucun defense_dae_best.pth disponible. Arret.")
            return
        print("Le checkpoint existant est conserve.")
    else:
        print(f"Meilleur epoch : {best_epoch} | F1 moyen {best_moyenne:.4f}")
        print(f"Reference baseline seul : {ref_moyenne:.4f}")
        print(f"Gain : {best_moyenne - ref_moyenne:+.4f}")
    print()

    checkpoint = torch.load(ckpt_path, weights_only=False, map_location=device)
    dae.load_state_dict(checkpoint["model_state_dict"])
    dae.eval()

    print("=" * 70)
    print("Evaluation sur le test : x -> DAE -> baseline")
    print("=" * 70)

    resultats = evaluate_on_attacks(
        dae, X_test, y_test, device, eval_batch, attacks_dir, all_labels,
        dae=dae, classifier=classifier,
    )
    print_summary(resultats, "Denoising Autoencoder v6")

    log_dir.mkdir(parents=True, exist_ok=True)
    horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")
    sortie = log_dir / f"defense_dae_{horodatage}.pkl"
    joblib.dump({
        "results": resultats, "history": history,
        "hyperparameters": {
            "epochs": epochs, "batch_size": batch_size, "lr_init": lr_init,
            "dae_config": dae_cfg, "clip_values": list(clip_values),
            "architecture": checkpoint["architecture"],
            "attaques_entrainement": ATTAQUES_ENTRAINEMENT,
            "loss_normalisee": normaliser, "residuel": False,
            "echelles_corruption": echelles, "seed": cfg.seed,
        },
        "best_epoch": best_epoch, "best_f1_moyen": best_moyenne,
        "reference_baseline_seul": {
            "clean": ref_clean_val, **refs_attaque, "moyenne": ref_moyenne,
        },
        "variance_donnees": VARIANCE_DONNEES,
        "training_time_min": duree,
        "config_hash": cfg.baseline_fingerprint(),
    }, sortie)
    print(f"\nResultats sauvegardes : {sortie}")
    print("\nDefense Denoising Autoencoder v6 terminee")


if __name__ == "__main__":
    main()
