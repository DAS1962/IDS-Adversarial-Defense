"""
Agregation par ensemble des quatre defenses (LS, AT, GA, DAE).

Chaque defense produit ses probabilites softmax sur les donnees propres et
sur chaque attaque. Trois methodes d'agregation sont comparees :

  1. Majority voting  : chaque modele vote, la classe majoritaire gagne.
  2. Weighted average : moyenne des probabilites, poids egaux (1/4).
  3. Weighted average optimise : poids ajustes par Nelder-Mead.

Reference : Awad et al. 2025 (Algorithme 1, section "Ensemble defense
optimization function").

Changements par rapport a la version precedente
------------------------------------------------
- Configuration centralisee, empreintes verifiees, y_* lus avec joblib.
- Verification du config_hash de chaque checkpoint de defense : un ensemble
  melangeant des defenses entrainees sous des configurations differentes
  produirait des chiffres incomparables entre eux.
- Les poids sont optimises sur le set de VALIDATION et non sur les attaques
  du test. L'ancienne version optimisait directement sur les accuracies du
  test, ce qui revenait a ajuster un hyperparametre sur le jeu d'evaluation.
  Voir optimize_weights pour le detail du compromis.
- La fonction objectif maximise le F1 MACRO et non l'accuracy : l'accuracy
  est plafonnee par la proportion de BENIGN (83.1% du test), donc l'optimiser
  pousse l'ensemble a tout classer BENIGN — l'inverse du but.
- Moyennes macro sur les 15 classes.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.dnn import BaselineDNN
from src.utils.config import load_config, check_data_fingerprint
from src.defenses.common import (
    ATTACK_FILES,
    BASELINE_UNDEFENDED,
    load_splits,
    load_baseline_model,
    compute_metrics,
    print_metrics,
)

import importlib.util


DEFENSE_CHECKPOINTS = {
    "LS":  "defense_ls_best.pth",
    "AT":  "defense_at_best.pth",
    "GA":  "defense_ga_best.pth",
    "DAE": "defense_dae_best.pth",
}


def build_dae(cfg, clip_values):
    """
    Reconstruit l'architecture du DAE.

    Importee depuis le script 14 plutot que redefinie ici : deux definitions
    de la meme architecture divergeraient tot ou tard, et un state_dict
    charge dans une architecture legerement differente echoue de facon
    obscure.
    """
    # import_module ne fonctionne pas ici : un nom de module Python ne peut
    # pas commencer par un chiffre. On charge donc le fichier par son chemin.
    chemin = Path(__file__).parent / "14_defense_denoising_autoencoder.py"
    spec = importlib.util.spec_from_file_location("dae_module", chemin)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dae_cfg = cfg.defenses["DenoisingAutoencoder"]
    return module.DenoisingAutoencoder(
        input_dim=cfg.dataset["num_features"],
        bottleneck_dim=dae_cfg["hidden_dim"],
        clip_values=clip_values,
    )


def load_defense(nom, fichier, cfg, device, checkpoint_dir):
    """
    Charge une defense entrainee, en verifiant son empreinte.

    Une defense entrainee sous une autre configuration ne doit pas entrer
    dans l'ensemble : ses probabilites porteraient sur un autre pretraitement
    ou une autre architecture, et l'agregation melangerait des choses
    incomparables.
    """
    chemin = checkpoint_dir / fichier
    if not chemin.exists():
        return None, f"{fichier} introuvable"

    checkpoint = torch.load(chemin, weights_only=False, map_location=device)

    attendu = cfg.baseline_fingerprint()
    obtenu = checkpoint.get("config_hash")
    if obtenu != attendu:
        return None, (f"config_hash {obtenu!r} != {attendu!r} : defense entrainee "
                      f"sous une autre configuration, relancer son script")

    if nom == "DAE":
        modele = build_dae(cfg, cfg.clip_values)
    else:
        modele = BaselineDNN(
            input_dim=cfg.dataset["num_features"],
            hidden1=cfg.model["hidden_layers"][0],
            hidden2=cfg.model["hidden_layers"][1],
            output_dim=cfg.dataset["num_classes"],
        )

    modele.load_state_dict(checkpoint["model_state_dict"])
    modele = modele.to(device).eval()
    # Chaque defense sauvegarde une metrique de selection differente :
    # val_acc pour LS/AT/GA (accuracy de validation), f1_moyen pour le DAE
    # (moyenne du F1 macro sur la validation propre et les quatre attaques).
    # Les versions anterieures du DAE utilisaient val_mse. On prend la
    # premiere disponible et on affiche laquelle, plutot que de formater un
    # None.
    for cle in ("val_acc", "f1_moyen", "val_f1_macro", "val_mse"):
        if checkpoint.get(cle) is not None:
            detail = f"{cle}={checkpoint[cle]:.4f}"
            break
    else:
        detail = "metrique de selection absente du checkpoint"
    return modele, f"epoch {checkpoint.get('epoch')}, {detail}"


@torch.no_grad()
def get_probabilities(modele, X, device, batch_size, nb_classes,
                      dae=None, classifier=None):
    """
    Probabilites softmax d'un modele sur un tableau numpy.

    Pour le DAE, le pipeline est x -> DAE -> baseline -> softmax : le
    parametre `modele` est alors ignore, seuls `dae` et `classifier` servent.
    """
    n = len(X)
    probs = np.zeros((n, nb_classes), dtype=np.float32)
    for i in range(0, n, batch_size):
        fin = min(i + batch_size, n)
        xb = torch.tensor(X[i:fin], dtype=torch.float32).to(device)
        if dae is not None and classifier is not None:
            x_pur, _ = dae(xb)
            logits = classifier(x_pur)
        else:
            logits = modele(xb)
        probs[i:fin] = F.softmax(logits, dim=1).cpu().numpy()
    return probs


def probabilites_toutes_defenses(defenses, X, device, batch_size, nb_classes,
                                 classifier):
    """Liste de matrices de probabilites, une par defense, dans l'ordre."""
    sorties = []
    for nom, modele in defenses.items():
        if nom == "DAE":
            probs = get_probabilities(modele, X, device, batch_size, nb_classes,
                                      dae=modele, classifier=classifier)
        else:
            probs = get_probabilities(modele, X, device, batch_size, nb_classes)
        sorties.append(probs)
    return sorties


def majority_voting(probs_list, nb_classes):
    """
    Vote majoritaire.

    En cas d'egalite, np.bincount().argmax() retient la classe d'indice le
    plus petit, donc BENIGN. C'est un choix conservateur mais qui biaise vers
    le non-detection : a garder en tete pour interpreter les resultats.
    """
    preds = np.stack([p.argmax(axis=1) for p in probs_list], axis=0)
    n = preds.shape[1]
    resultat = np.zeros(n, dtype=np.int64)
    for i in range(n):
        resultat[i] = np.bincount(preds[:, i], minlength=nb_classes).argmax()
    return resultat


def weighted_average(probs_list, poids):
    """Moyenne ponderee des probabilites, puis argmax."""
    accumule = np.zeros_like(probs_list[0])
    for probs, p in zip(probs_list, poids):
        accumule += p * probs
    return accumule.argmax(axis=1)


def optimize_weights(probs_val, y_val, all_labels, n_modeles):
    """
    Optimise les poids sur le set de VALIDATION propre.

    Deux choix a expliciter.

    D'abord le jeu de donnees. L'article optimise les poids pour maximiser
    la performance sous attaque, mais le faire sur le test set reviendrait a
    ajuster un hyperparametre sur le jeu d'evaluation — exactement le biais
    retire de 06_train_baseline.py. On optimise donc sur la validation
    propre. Consequence assumee : les poids ne sont pas specialises contre
    les attaques, ce qui rend l'ensemble optimise moins performant que dans
    l'article mais mesurable sans biais.

    Ensuite la metrique. On maximise le F1 macro et non l'accuracy :
    l'accuracy est plafonnee par la proportion de BENIGN, donc l'optimiser
    revient a recompenser un ensemble qui classe tout BENIGN.
    """
    def objectif(w):
        w = np.abs(w)
        somme = w.sum()
        if somme == 0:
            return 0.0
        w = w / somme
        y_pred = weighted_average(probs_val, w)
        return -f1_score(y_val, y_pred, labels=all_labels,
                         average="macro", zero_division=0)

    depart = np.ones(n_modeles) / n_modeles
    resultat = minimize(objectif, depart, method="Nelder-Mead",
                        options={"maxiter": 500, "xatol": 1e-4, "fatol": 1e-4})
    poids = np.abs(resultat.x)
    poids = poids / poids.sum()
    return poids, -resultat.fun


def print_summary_ensemble(resultats, methode):
    """Recapitulatif d'une methode, avec comparaison au baseline non defendu."""
    print("\n" + "=" * 104)
    print(f"Resume - Ensemble {methode} (vs baseline non defendu)")
    print("=" * 104)
    print(f"{'Attaque':<12} {'Accuracy':>10} {'(base)':>9} {'Delta':>8}"
          f" | {'F1 macro':>10} {'(base)':>9} {'Delta':>8} | {'Rec.BENIGN':>11}")
    print("-" * 104)
    for r in resultats:
        base = BASELINE_UNDEFENDED.get(r["attack"])
        if base is None:
            continue
        print(f"{r['attack']:<12} {r['accuracy']:>10.4f} {base['accuracy']:>9.4f} "
              f"{r['accuracy']-base['accuracy']:>+8.4f}"
              f" | {r['f1_macro']:>10.4f} {base['f1_macro']:>9.4f} "
              f"{r['f1_macro']-base['f1_macro']:>+8.4f}"
              f" | {r['recall_benign']:>11.4f}")
    print("=" * 104)

    attaques = [r for r in resultats if r["attack"] != "Clean"
                and r["attack"] in BASELINE_UNDEFENDED]
    if attaques:
        gains = [r["f1_macro"] - BASELINE_UNDEFENDED[r["attack"]]["f1_macro"]
                 for r in attaques]
        print(f"Gain moyen en F1 macro sur les attaques : {np.mean(gains):+.4f}")


def print_comparaison_finale(par_methode):
    """Les trois methodes cote a cote, en F1 macro."""
    print("\n" + "=" * 100)
    print("Comparaison des trois methodes d'agregation (F1 macro)")
    print("=" * 100)
    methodes = list(par_methode.keys())
    attaques = [r["attack"] for r in par_methode[methodes[0]]]

    entete = f"{'Attaque':<12} {'Baseline':>10}"
    for m in methodes:
        entete += f" {m[:22]:>23}"
    print(entete)
    print("-" * 100)
    for i, attaque in enumerate(attaques):
        base = BASELINE_UNDEFENDED.get(attaque, {}).get("f1_macro", float("nan"))
        ligne = f"{attaque:<12} {base:>10.4f}"
        for m in methodes:
            ligne += f" {par_methode[m][i]['f1_macro']:>23.4f}"
        print(ligne)
    print("=" * 100)


def main():
    print("=" * 70)
    print("Agregation par ensemble des quatre defenses")
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

    nb_classes = cfg.dataset["num_classes"]
    all_labels = list(range(nb_classes))
    batch_size = cfg.evaluation["batch_size"]

    print("Chargement des quatre defenses...")
    defenses = {}
    manquantes = []
    for nom, fichier in DEFENSE_CHECKPOINTS.items():
        modele, detail = load_defense(nom, fichier, cfg, device, checkpoint_dir)
        if modele is None:
            print(f"  {nom:<4} : ECHEC — {detail}")
            manquantes.append(nom)
        else:
            print(f"  {nom:<4} : {fichier} ({detail})")
            defenses[nom] = modele
    print()

    if manquantes:
        print(f"Defenses indisponibles : {', '.join(manquantes)}")
        print("L'ensemble a besoin des quatre. Lancer les scripts 11 a 14 d'abord.")
        return

    noms = list(defenses.keys())
    n_modeles = len(noms)

    print("Calcul des probabilites sur la validation (optimisation des poids)...")
    probs_val = probabilites_toutes_defenses(defenses, X_val, device, batch_size,
                                             nb_classes, classifier)
    print(f"  {len(probs_val)} matrices de {probs_val[0].shape}\n")

    print("Calcul des probabilites sur les donnees propres et les attaques...")
    probs_par_attaque = {}
    print("  -> Clean")
    probs_par_attaque["Clean"] = probabilites_toutes_defenses(
        defenses, X_test, device, batch_size, nb_classes, classifier)

    for nom_attaque, fichier in ATTACK_FILES:
        chemin = attacks_dir / fichier
        if not chemin.exists():
            print(f"  [SKIP] {nom_attaque} : {fichier} introuvable")
            continue
        X_adv = joblib.load(chemin)
        if len(X_adv) != len(y_test):
            print(f"  [SKIP] {nom_attaque} : {len(X_adv):,} lignes contre "
                  f"{len(y_test):,} attendues (autre perimetre)")
            del X_adv
            continue
        print(f"  -> {nom_attaque}")
        probs_par_attaque[nom_attaque] = probabilites_toutes_defenses(
            defenses, X_adv, device, batch_size, nb_classes, classifier)
        del X_adv
    print()

    ordre = list(probs_par_attaque.keys())

    print("=" * 70)
    print("Methode 1 : Majority Voting")
    print("=" * 70)
    resultats_mv = []
    for nom_attaque in ordre:
        y_pred = majority_voting(probs_par_attaque[nom_attaque], nb_classes)
        m = compute_metrics(y_test, y_pred, nom_attaque, all_labels)
        print(f"\n--- {nom_attaque} ---")
        print_metrics(m)
        resultats_mv.append(m)
    print_summary_ensemble(resultats_mv, "Majority Voting")

    print("\n" + "=" * 70)
    print("Methode 2 : Weighted Average, poids egaux")
    print("=" * 70)
    poids_egaux = np.ones(n_modeles) / n_modeles
    print(f"\nPoids : {dict(zip(noms, poids_egaux.round(4)))}")
    resultats_wa = []
    for nom_attaque in ordre:
        y_pred = weighted_average(probs_par_attaque[nom_attaque], poids_egaux)
        m = compute_metrics(y_test, y_pred, nom_attaque, all_labels)
        print(f"\n--- {nom_attaque} ---")
        print_metrics(m)
        resultats_wa.append(m)
    print_summary_ensemble(resultats_wa, "Weighted Average (egal)")

    print("\n" + "=" * 70)
    print("Methode 3 : Weighted Average optimise (Nelder-Mead)")
    print("=" * 70)
    print("\nOptimisation sur la validation propre, objectif = F1 macro.")
    t0 = time.time()
    poids_opt, meilleur_f1 = optimize_weights(probs_val, y_val, all_labels, n_modeles)
    print(f"  Duree : {time.time()-t0:.1f}s")
    print(f"  F1 macro atteint sur validation : {meilleur_f1:.4f}")
    print("  Poids optimaux :")
    for nom, p in zip(noms, poids_opt):
        print(f"    {nom:<4} : {p:.4f}")

    resultats_opt = []
    for nom_attaque in ordre:
        y_pred = weighted_average(probs_par_attaque[nom_attaque], poids_opt)
        m = compute_metrics(y_test, y_pred, nom_attaque, all_labels)
        print(f"\n--- {nom_attaque} ---")
        print_metrics(m)
        resultats_opt.append(m)
    print_summary_ensemble(resultats_opt, "Weighted Average optimise")

    par_methode = {
        "Majority Voting": resultats_mv,
        "Weighted Avg egal": resultats_wa,
        "Weighted Avg optimise": resultats_opt,
    }
    print_comparaison_finale(par_methode)

    log_dir.mkdir(parents=True, exist_ok=True)
    horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")
    sortie = log_dir / f"ensemble_results_{horodatage}.pkl"
    joblib.dump({
        "majority_voting": resultats_mv,
        "weighted_average_equal": resultats_wa,
        "weighted_average_optimized": resultats_opt,
        "optimal_weights": dict(zip(noms, poids_opt.tolist())),
        "defense_names": noms,
        "attack_names": ordre,
        "optimization": {
            "split": "validation",
            "objective": "f1_macro",
            "best_value": float(meilleur_f1),
        },
        "config_hash": cfg.baseline_fingerprint(),
    }, sortie)
    print(f"\nResultats sauvegardes : {sortie}")
    print("\nAgregation par ensemble terminee")


if __name__ == "__main__":
    main()
