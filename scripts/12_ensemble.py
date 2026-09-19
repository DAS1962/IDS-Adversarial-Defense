"""
Agregation par ensemble. Derniere etape du framework.

Le papier donne deux regles de fusion, plus leurs versions optimisees :

  y_final = mode(f1(x_adv), f2(x_adv), f3(x_adv), f4(x_tilde))   vote majoritaire
  y_final = w1*y1 + w2*y2 + w3*y3 + w4*y4                        moyenne ponderee

"the majority voting prediction is optimized with soft voting, and the
weighted average prediction is optimized with Bayesian optimization"

Le soft voting est la moyenne des probabilites a poids egaux : c'est donc la
moyenne ponderee avec w = 0.25 partout. C'est cette methode qui leur donne
leur meilleur resultat, 87.49 %.

Note sur le DAE : les trois premieres defenses recoivent l'entree attaquee
telle quelle, la quatrieme recoit sa version nettoyee. C'est ce que dit leur
formule, ou seul f4 prend x_tilde.

L'optimisation des poids se fait sur la VALIDATION PROPRE, pas sur le test.
Le papier ne dit pas sur quoi il optimise ; le faire sur le jeu d'evaluation
reviendrait a y ajuster un hyperparametre.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.defenses.commun import ATTAQUES, baseline_reference, nettoyer
from src.models.dnn import DAE, DNN
from src.utils.commun import charger_donnees, metriques
from src.utils.config import load_config

ORDRE = ["Clean"] + ATTAQUES
NOMS = {"ls": "LS", "ga": "GA", "at": "AT", "dae": "DAE"}


def charger_defenses(cfg, device, suffixe=""):
    """Les trois detecteurs reentraines, plus le DAE et le detecteur d'origine."""
    ck_dir = Path(cfg.paths["checkpoints"])
    modeles = {}

    for court in ("ls", "ga", "at"):
        chemin = ck_dir / f"defense_{court}{suffixe}.pth"
        if not chemin.exists():
            raise RuntimeError(f"{chemin} manquant")
        ck = torch.load(chemin, weights_only=False, map_location=device)
        m = DNN(input_dim=cfg.dataset["num_features"],
                hidden=tuple(cfg.model["hidden_layers"]),
                output_dim=cfg.dataset["num_classes"]).to(device)
        m.load_state_dict(ck["model_state_dict"])
        m.eval()
        modeles[court] = m
        print(f"  {NOMS[court]:<4} defense_{court}{suffixe}.pth (epoch {ck['epoch']})")

    # Le DAE ne remplace pas le detecteur, il le precede.
    ck = torch.load(ck_dir / f"defense_dae{suffixe}.pth", weights_only=False,
                    map_location=device)
    d = cfg.defenses["DenoisingAutoencoder"]
    dae = DAE(input_dim=cfg.dataset["num_features"],
              bottleneck=d["hidden_dim"]).to(device)
    dae.load_state_dict(ck["model_state_dict"])
    dae.eval()

    ck = torch.load(ck_dir / f"baseline_varlr{suffixe}.pth", weights_only=False,
                    map_location=device)
    detecteur = DNN(input_dim=cfg.dataset["num_features"],
                    hidden=tuple(cfg.model["hidden_layers"]),
                    output_dim=cfg.dataset["num_classes"]).to(device)
    detecteur.load_state_dict(ck["model_state_dict"])
    detecteur.eval()
    print(f"  DAE  defense_dae{suffixe}.pth + baseline_varlr{suffixe}.pth (epoch {ck['epoch']})")

    return modeles, dae, detecteur


@torch.no_grad()
def probas(model, X, device, batch_size):
    model.eval()
    sortie = np.zeros((len(X), 15), dtype=np.float32)
    for i in range(0, len(X), batch_size):
        fin = min(i + batch_size, len(X))
        xb = torch.tensor(X[i:fin], dtype=torch.float32).to(device)
        sortie[i:fin] = F.softmax(model(xb), dim=1).cpu().numpy()
    return sortie


def toutes_probas(modeles, dae, detecteur, X, cfg, device):
    """Les quatre jeux de probabilites pour une entree donnee."""
    bs = cfg.evaluation["batch_size"]
    p = {c: probas(m, X, device, bs) for c, m in modeles.items()}
    X_net = nettoyer(dae, X, device, bs, cfg.clip_values)
    p["dae"] = probas(detecteur, X_net, device, bs)
    return p


def vote_majoritaire(p, labels):
    """
    y_final = mode(...). En cas d'egalite, np.bincount retient le plus petit
    indice, ce qui favorise BENIGN. On departage par la somme des
    probabilites, plus juste.
    """
    preds = np.stack([p[c].argmax(axis=1) for c in ("ls", "ga", "at", "dae")])
    somme = sum(p[c] for c in ("ls", "ga", "at", "dae"))
    sortie = np.zeros(preds.shape[1], dtype=np.int64)
    for i in range(preds.shape[1]):
        comptes = np.bincount(preds[:, i], minlength=len(labels))
        maxi = comptes.max()
        candidats = np.where(comptes == maxi)[0]
        sortie[i] = (candidats[0] if len(candidats) == 1
                     else candidats[np.argmax(somme[i, candidats])])
    return sortie


def moyenne_ponderee(p, poids):
    agrege = sum(poids[c] * p[c] for c in ("ls", "ga", "at", "dae"))
    return agrege.argmax(axis=1)


def optimiser_poids(p_val, y_val, labels):
    """
    Nelder-Mead sur la validation PROPRE, objectif F1 macro.

    Le papier utilise l'optimisation bayesienne. Nelder-Mead est equivalent
    sur un probleme a quatre dimensions et evite une dependance de plus.

    L'objectif est le F1 macro et non l'accuracy : optimiser l'accuracy
    recompenserait un ensemble classant tout en BENIGN, puisqu'elle est
    plafonnee par la proportion de trafic normal.
    """
    cles = ("ls", "ga", "at", "dae")

    def objectif(w):
        w = np.abs(w)
        w = w / w.sum() if w.sum() > 0 else np.ones(4) / 4
        pred = sum(wi * p_val[c] for wi, c in zip(w, cles)).argmax(axis=1)
        return -metriques(y_val, pred, labels)["f1_macro"]

    res = minimize(objectif, np.ones(4) / 4, method="Nelder-Mead",
                   options={"maxiter": 300, "xatol": 1e-4, "fatol": 1e-5})
    w = np.abs(res.x)
    w = w / w.sum()
    return {c: float(wi) for c, wi in zip(cles, w)}, -res.fun


def main():
    print("=" * 92)
    print("Agregation par ensemble - reproduction fidele")
    print(f"Date : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 92 + "\n")

    parseur = argparse.ArgumentParser()
    parseur.add_argument("--suffixe", default="",
                         help="suffixe des checkpoints et des sorties")
    args = parseur.parse_args()

    cfg = load_config()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = list(range(cfg.dataset["num_classes"]))
    atk_dir = Path(cfg.paths["attacks"])

    print("Chargement des quatre defenses :")
    modeles, dae, detecteur = charger_defenses(cfg, device, args.suffixe)
    print()

    _, _, X_va, y_va, _, _ = charger_donnees(cfg.paths["processed"])
    print("Optimisation des poids sur la validation propre...")
    p_val = toutes_probas(modeles, dae, detecteur, X_va, cfg, device)
    poids_opt, f1_val = optimiser_poids(p_val, y_va, labels)
    print(f"  F1 macro atteint : {f1_val:.4f}")
    for c, w in poids_opt.items():
        print(f"    {NOMS[c]:<4} {w:.4f}")
    del p_val
    print()

    X = joblib.load(atk_dir / "X_test_clean.pkl")
    y = joblib.load(atk_dir / "y_test_clean.pkl")
    poids_egaux = {c: 0.25 for c in ("ls", "ga", "at", "dae")}

    methodes = {
        "Vote majoritaire": None,
        "Moyenne, poids egaux": poids_egaux,
        "Moyenne, poids optimises": poids_opt,
    }
    resultats = {nom: {} for nom in methodes}

    print("Evaluation sur l'echantillon test :")
    for attaque in ORDRE:
        X_att = (X if attaque == "Clean"
                 else joblib.load(atk_dir / f"X_adv_test_{attaque.lower()}.pkl"))
        p = toutes_probas(modeles, dae, detecteur, X_att, cfg, device)

        for nom, poids in methodes.items():
            pred = (vote_majoritaire(p, labels) if poids is None
                    else moyenne_ponderee(p, poids))
            resultats[nom][attaque] = metriques(y, pred, labels, attaque)

        print(f"  {attaque} fait", flush=True)
        del p
        if attaque != "Clean":
            del X_att
    print()

    base, nom_base = baseline_reference(cfg.paths["logs"])
    print(f"Reference : {nom_base}\n")

    bilans = {}
    for nom, res in resultats.items():
        print("=" * 92)
        print(f"{nom}")
        print("=" * 92)
        print(f"{'Attaque':<10} {'Accuracy':>9} {'base':>9} {'delta':>8} | "
              f"{'F1 macro':>9} {'base':>9} {'delta':>8} | {'MCC bin':>8} "
              f"{'RecBEN':>8}")
        print("-" * 92)
        gains = []
        for a in ORDRE:
            m, b = res[a], base[a]
            d_f1 = m["f1_macro"] - b["f1_macro"]
            if a != "Clean":
                gains.append(d_f1)
            print(f"{a:<10} {m['accuracy']:>9.4f} {b['accuracy']:>9.4f} "
                  f"{m['accuracy']-b['accuracy']:>+8.4f} | "
                  f"{m['f1_macro']:>9.4f} {b['f1_macro']:>9.4f} {d_f1:>+8.4f} | "
                  f"{m['mcc_binaire']:>8.4f} {m['recall_benign']:>8.4f}")
        print("-" * 92)
        gain = float(np.mean(gains))
        cout = res["Clean"]["f1_macro"] - base["Clean"]["f1_macro"]
        rb = min(res[a]["recall_benign"] for a in ATTAQUES)
        print(f"Gain moyen : {gain:+.4f} | Cout propre : {cout:+.4f} | "
              f"Bilan net : {gain+cout:+.4f} | RecBEN min : {rb:.4f}")
        # Micro sur les sept jeux : c'est la lecture qui colle a leur Table 7
        micro = np.mean([res[a]["accuracy"] for a in ORDRE])
        print(f"Accuracy moyenne sur les sept jeux : {100*micro:.2f}%")
        print()
        bilans[nom] = {"gain": gain, "cout": cout, "bilan": gain + cout,
                       "rb_min": rb, "micro": micro}

    print("=" * 92)
    print("Comparaison au papier")
    print("=" * 92)
    print(f"{'Methode':<28} {'Nous':>9} {'Papier':>9}")
    print(f"{'Vote majoritaire':<28} "
          f"{100*bilans['Vote majoritaire']['micro']:>8.2f}% {84.35:>8.2f}%")
    print(f"{'Soft voting / poids egaux':<28} "
          f"{100*bilans['Moyenne, poids egaux']['micro']:>8.2f}% {87.49:>8.2f}%")
    print(f"{'Moyenne ponderee':<28} "
          f"{100*bilans['Moyenne, poids optimises']['micro']:>8.2f}% {84.45:>8.2f}%")
    print(f"{'Moyenne ponderee optimisee':<28} {'—':>9} {86.11:>8.2f}%")
    print()
    # Meilleure defense seule, lue depuis les resultats plutot que codee
    meilleure_def, nom_meilleure = -9.9, "?"
    for court in ("ls", "ga", "at", "dae"):
        fs = sorted(Path(cfg.paths["logs"]).glob(f"defense_{court}{args.suffixe}_*.pkl"))
        if fs:
            b = joblib.load(fs[-1])["bilan"]["bilan"]
            if b > meilleure_def:
                meilleure_def, nom_meilleure = b, NOMS[court]
    print(f"Meilleure defense seule ({nom_meilleure}), bilan net : {meilleure_def:+.4f}")
    for nom, b in bilans.items():
        marque = "  <-- depasse AT" if b["bilan"] > meilleure_def else ""
        print(f"  {nom:<28} {b['bilan']:>+8.4f}{marque}")
    print("=" * 92)

    sortie = (Path(cfg.paths["logs"]) /
              f"ensemble{args.suffixe}_{datetime.now():%Y%m%d_%H%M%S}.pkl")
    joblib.dump({"resultats": resultats, "poids_optimises": poids_opt,
                 "f1_val_optimisation": f1_val, "bilans": bilans}, sortie)

    lignes = []
    for nom, res in resultats.items():
        for a in ORDRE:
            d = {"methode": nom, "attaque": a}
            d.update({k: v for k, v in res[a].items()
                      if k not in ("confusion_matrix", "attaque")})
            lignes.append(d)
    csv = Path(cfg.paths["figures"]) / f"ensemble_complet{args.suffixe}.csv"
    pd.DataFrame(lignes).to_csv(csv, index=False, float_format="%.6f")
    print(f"\nResultats : {sortie}")
    print(f"CSV       : {csv}")


if __name__ == "__main__":
    main()
