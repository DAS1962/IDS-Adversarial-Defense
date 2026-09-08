"""
Generation des exemples adversariaux sur le TRAIN, pour l'entrainement du DAE.

Pourquoi ce script existe
--------------------------
L'Algorithme 5 de Awad et al. (2025) definit en entree du DAE :

    "Aggregated Adversarial examples (adv examples) of four attack
     categories (FGSM, DEEPFOOL, BIM, JSMA), trained IDS-classifier,
     training and testing real samples [...]"

et a l'etape 2 :

    "Construct total samples by concatenating (adv examples, real samples)"

Le DAE doit donc etre entraine sur de VRAIS exemples adversariaux issus de
quatre attaques nommees, concatenes aux echantillons reels. Les versions v1
a v4 de 14_defense_denoising_autoencoder.py regeneraient a la place un FGSM
approximatif a la volee (eps=0.05), puis etaient evaluees contre PGD a
eps=0.3, JSMA et C&W. Le DAE apprenait a retirer une perturbation six fois
plus petite et de nature differente de celles qu'il devait affronter : il
convergeait vers l'identite, strategie optimale sous ce mauvais appariement.

Les X_adv_*.pkl produits par 08_generate_attacks.py ne peuvent pas servir :
ils couvrent le test set. Les utiliser pour entrainer le DAE serait une
fuite directe du jeu d'evaluation.

Perimetre
---------
Sous-echantillon stratifie du train, taille lue depuis
defenses.DenoisingAutoencoder.train_adv_samples.

Compromis assume : les quatre attaques sur les 1 819 198 lignes du train
couteraient environ quatre heures de GPU (DeepFool ~1h40, JSMA ~2h,
extrapolees des mesures de 08 sur le test complet). 400 000 echantillons
stratifies couvrent les quinze classes et suffisent a apprendre une
fonction de purification.

Les quatre attaques de l'Algorithme 5 uniquement
------------------------------------------------
FGSM, BIM, DeepFool, JSMA. PGD et C&W sont volontairement absents : ils ne
figurent pas dans la liste de l'article. Cela permettra en plus de mesurer
si la purification transfere a des attaques que le DAE n'a jamais vues.

Empreintes
----------
L'empreinte utilisee ici n'est PAS cfg.attack_fingerprint(nom) : celle-ci
inclut evaluation_scope_data(), donc scope et sample_size, qui determinent
quelles lignes du TEST sont attaquees. Le perimetre du train est fixe par
train_adv_samples et par la seed. Utiliser l'empreinte d'evaluation
invaliderait ces fichiers a chaque changement de scope, sans raison.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

# ART 1.18 utilise np.product, supprime dans NumPy 2.x. Patch AVANT l'import.
if not hasattr(np, "product"):
    np.product = np.prod

import torch.nn as nn
import torchattacks
from art.attacks.evasion import DeepFool, SaliencyMapMethod
from art.estimators.classification import PyTorchClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.substitute import build_substitute_from_config
from src.utils.config import load_config, check_data_fingerprint, config_fingerprint


ATTAQUES_DAE = ["FGSM", "BIM", "DeepFool", "JSMA"]


def train_attack_fingerprint(cfg, nom):
    """
    Empreinte d'un fichier d'exemples adversariaux du TRAIN.

    Couvre les donnees sous-jacentes, le substitut qui genere l'attaque, les
    parametres de l'attaque, et le perimetre du sous-echantillon (taille et
    seed, qui determinent ensemble quelles lignes sont attaquees).

    N'inclut pas evaluation_scope : ce parametre gouverne le test, pas le
    train.
    """
    return config_fingerprint({
        "attack_name": nom,
        "data": cfg.data_fingerprint_data(),
        "substitute": cfg.substitute,
        "attack_params": cfg.attack_params(nom),
        "train_adv_samples": cfg.defenses["DenoisingAutoencoder"]["train_adv_samples"],
        "seed": cfg.seed,
    })


def load_substitute(cfg, device, checkpoint_dir):
    """Charge le substitut, avec verification bloquante de son empreinte."""
    chemin = checkpoint_dir / "substitute_best.pth"
    if not chemin.exists():
        raise RuntimeError(
            f"{chemin} introuvable. Lancer d'abord scripts/08_generate_attacks.py."
        )
    checkpoint = torch.load(chemin, weights_only=False, map_location=device)
    attendu = cfg.substitute_fingerprint()
    obtenu = checkpoint.get("config_hash")
    if obtenu != attendu:
        raise RuntimeError(
            f"{chemin} entraine sous une autre configuration "
            f"({obtenu!r} != {attendu!r}). Relancer 08_generate_attacks.py."
        )
    modele = build_substitute_from_config(cfg).to(device)
    modele.load_state_dict(checkpoint["model_state_dict"])
    modele.eval()
    print(f"Substitut charge : epoch {checkpoint.get('epoch')}, "
          f"F1 pondere val {checkpoint.get('f1_weighted_val', float('nan')):.4f}\n")
    return modele


def create_art_classifier(modele, cfg, device):
    """
    Enveloppe ART avec clip_values.

    Sans clip_values, ART ne borne pas ses perturbations et produit des
    exemples hors du domaine [0,1] des features normalisees.
    """
    return PyTorchClassifier(
        model=modele,
        loss=nn.CrossEntropyLoss(),
        input_shape=(cfg.dataset["num_features"],),
        nb_classes=cfg.dataset["num_classes"],
        clip_values=cfg.clip_values,
        device_type="gpu" if device.type == "cuda" else "cpu",
    )


def echantillon_stratifie(X, y, taille, seed):
    """
    Sous-echantillon stratifie du train.

    Repli sur un tirage aleatoire si une classe a moins de 2 exemples,
    plutot que d'echouer, avec avertissement explicite.
    """
    if taille >= len(X):
        print(f"  taille demandee ({taille:,}) >= train ({len(X):,}) : "
              f"tout le train est utilise")
        return np.arange(len(X))

    effectifs = pd.Series(y).value_counts()
    trop_rares = effectifs[effectifs < 2]
    try:
        if len(trop_rares) > 0:
            raise ValueError(f"classes a moins de 2 exemples : {list(trop_rares.index)}")
        indices, _ = train_test_split(
            np.arange(len(X)), train_size=taille, stratify=y, random_state=seed,
        )
    except ValueError as e:
        print(f"  stratification impossible ({e})")
        print("  repli sur un tirage aleatoire simple")
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(X), size=taille, replace=False)
    return np.sort(indices)


def generer_attaque(nom, modele, art_clf, X, y, cfg, device, batch_size):
    """
    Genere une attaque sur X, par batchs, avec les parametres de la Table 2.

    FGSM et BIM passent par torchattacks (bornage [0,1] interne, coherent
    avec MinMaxScaler). DeepFool et JSMA passent par ART.
    """
    params = cfg.attack_params(nom)
    n = len(X)
    X_adv = np.zeros_like(X, dtype=np.float32)
    print("  parametres : " + ", ".join(f"{k}={v}" for k, v in sorted(params.items())))
    debut = time.time()

    if nom in ("FGSM", "BIM"):
        if nom == "FGSM":
            attaque = torchattacks.FGSM(modele, eps=params["eps"])
        else:
            attaque = torchattacks.BIM(
                modele, eps=params["eps"], alpha=params["alpha"],
                steps=params["steps"],
            )
        for i in range(0, n, batch_size):
            fin = min(i + batch_size, n)
            xb = torch.tensor(X[i:fin], dtype=torch.float32).to(device)
            yb = torch.tensor(y[i:fin], dtype=torch.long).to(device)
            X_adv[i:fin] = attaque(xb, yb).detach().cpu().numpy()
            if (i // batch_size) % 200 == 0 and i > 0:
                ecoule = time.time() - debut
                print(f"    {i:,}/{n:,} ({100*i/n:.1f}%) - reste "
                      f"~{ecoule/i*(n-i)/60:.1f} min", flush=True)

    elif nom == "DeepFool":
        attaque = DeepFool(
            classifier=art_clf, epsilon=params["epsilon"],
            max_iter=params["max_iter"], batch_size=batch_size, verbose=False,
        )
        for i in range(0, n, batch_size):
            fin = min(i + batch_size, n)
            # y=None : untargeted, coherent avec l'hypothese de l'article.
            X_adv[i:fin] = attaque.generate(x=X[i:fin], y=None)
            if (i // batch_size) % 50 == 0 and i > 0:
                ecoule = time.time() - debut
                print(f"    {i:,}/{n:,} ({100*i/n:.1f}%) - reste "
                      f"~{ecoule/i*(n-i)/60:.1f} min", flush=True)

    elif nom == "JSMA":
        attaque = SaliencyMapMethod(
            classifier=art_clf, theta=params["theta"], gamma=params["gamma"],
            batch_size=batch_size, verbose=False,
        )
        for i in range(0, n, batch_size):
            fin = min(i + batch_size, n)
            X_adv[i:fin] = attaque.generate(x=X[i:fin], y=None)
            if (i // batch_size) % 20 == 0 and i > 0:
                ecoule = time.time() - debut
                print(f"    {i:,}/{n:,} ({100*i/n:.1f}%) - reste "
                      f"~{ecoule/i*(n-i)/60:.1f} min", flush=True)
    else:
        raise ValueError(f"attaque {nom!r} non prevue par ce script")

    duree = time.time() - debut
    borne_min, borne_max = cfg.clip_values
    hors = int(np.sum((X_adv < borne_min) | (X_adv > borne_max)))
    if hors:
        print(f"    {hors:,} valeurs hors [{borne_min}, {borne_max}] -> bornees")
    X_adv = np.clip(X_adv, borne_min, borne_max)

    # Amplitude reelle de la perturbation : dit ce que le DAE devra apprendre
    # a retirer, et permet de comparer aux attaques du test.
    delta = np.abs(X_adv - X)
    print(f"  duree : {duree/60:.1f} min")
    print(f"  perturbation : L-inf moyen {delta.max(axis=1).mean():.4f}, "
          f"L2 moyen {np.linalg.norm(delta, axis=1).mean():.4f}, "
          f"features touchees {(delta > 1e-6).sum(axis=1).mean():.1f}/58")
    return X_adv, duree


def main():
    print("=" * 70)
    print("Generation des exemples adversariaux sur le TRAIN (pour le DAE)")
    print(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")

    cfg = load_config()
    print(cfg.resume())
    print()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    data_dir = Path(cfg.paths["data_processed"])
    checkpoint_dir = Path(cfg.paths["checkpoints"])
    sortie_dir = Path(cfg.paths["attacks"]) / "train"
    sortie_dir.mkdir(parents=True, exist_ok=True)

    check_data_fingerprint(cfg, data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print()

    print("Chargement du train...")
    X_train = pd.read_pickle(data_dir / "X_train.pkl")
    y_train = joblib.load(data_dir / "y_train.pkl")
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values.astype(np.float32)
    y_train = np.asarray(y_train)
    print(f"  X_train : {X_train.shape}\n")

    taille = cfg.defenses["DenoisingAutoencoder"]["train_adv_samples"]
    print(f"Sous-echantillon stratifie de {taille:,} lignes...")
    indices = echantillon_stratifie(X_train, y_train, taille, cfg.seed)
    X_sub = np.ascontiguousarray(X_train[indices])
    y_sub = np.ascontiguousarray(y_train[indices])
    print(f"  retenu : {len(indices):,} lignes")

    label_encoder = joblib.load(data_dir / "label_encoder.pkl")
    print("  distribution :")
    for classe in range(cfg.dataset["num_classes"]):
        n = int((y_sub == classe).sum())
        print(f"    {classe:2d} {str(label_encoder.classes_[classe])[:28]:28s} : {n:>7,}")
    print()

    del X_train, y_train

    substitut = load_substitute(cfg, device, checkpoint_dir)
    art_clf = create_art_classifier(substitut, cfg, device)
    batch_size = cfg.evaluation["batch_size"]

    # Sauvegarde du sous-echantillon propre et de ses indices AVANT les
    # attaques : sans eux, impossible de reapparier chaque exemple
    # adversarial a sa cible propre lors de l'entrainement du DAE.
    joblib.dump(indices, sortie_dir / "indices_train_adv.pkl")
    joblib.dump(X_sub, sortie_dir / "X_train_clean_sub.pkl")
    joblib.dump(y_sub, sortie_dir / "y_train_clean_sub.pkl")
    print(f"Sous-echantillon propre sauvegarde dans {sortie_dir}/\n")

    durees = {}
    for nom in ATTAQUES_DAE:
        chemin = sortie_dir / f"X_adv_train_{nom.lower()}.pkl"
        chemin_hash = sortie_dir / f"X_adv_train_{nom.lower()}.hash"
        empreinte = train_attack_fingerprint(cfg, nom)

        if chemin.exists() and chemin_hash.exists():
            if chemin_hash.read_text().strip() == empreinte:
                X_adv = joblib.load(chemin)
                if len(X_adv) == len(X_sub):
                    print(f"--- {nom} : deja genere sous la meme configuration, "
                          f"conserve ---\n")
                    del X_adv
                    continue
            print(f"--- {nom} : artefact perime, regeneration ---")

        print(f"--- {nom} ---")
        X_adv, duree = generer_attaque(
            nom, substitut, art_clf, X_sub, y_sub, cfg, device, batch_size,
        )
        joblib.dump(X_adv, chemin)
        chemin_hash.write_text(empreinte)
        print(f"  sauvegarde : {chemin.name} "
              f"({chemin.stat().st_size / (1024*1024):.1f} MB)\n")
        durees[nom] = duree
        del X_adv

    print("=" * 70)
    print("Recapitulatif")
    print("=" * 70)
    for nom in ATTAQUES_DAE:
        chemin = sortie_dir / f"X_adv_train_{nom.lower()}.pkl"
        etat = "present" if chemin.exists() else "MANQUANT"
        d = durees.get(nom)
        print(f"  {nom:<10} : {etat:<9} "
              f"({f'{d/60:.1f} min' if d else 'depuis le cache'})")
    print(f"\nDossier : {sortie_dir}")
    print("\nEtape suivante : scripts/14_defense_denoising_autoencoder.py")


if __name__ == "__main__":
    main()
