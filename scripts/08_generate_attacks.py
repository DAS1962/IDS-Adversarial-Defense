"""
Génération des 6 attaques adversariales, évaluées par transfert sur le
baseline v4 (Awad et al. 2025).

Protocole semi-white box (fidèle à l'article)
----------------------------------------------
Les exemples adversariaux sont générés sur un modèle SUBSTITUT (58 -> 100 ->
100 -> 15, cf. src/models/substitute.py et configs/config.yaml), pas sur le
baseline. L'attaquant connaît l'architecture générale mais pas les poids ni
les hyperparamètres du baseline : c'est la définition du semi-white box que
l'article revendique. La version précédente de ce script passait directement
`torchattacks.FGSM(model, ...)` et `PyTorchClassifier(model=model)` avec le
baseline lui-même : l'attaquant disposait donc des vrais gradients du modèle
attaqué (white box complet), ce qui rend les résultats structurellement plus
destructeurs et incomparables à l'article.

Le substitut est entraîné une fois (si aucun checkpoint n'existe), puis
réutilisé pour générer les six attaques. Les exemples générés sur le
substitut sont ensuite évalués sur le BASELINE (transférabilité) : c'est la
métrique qui compte, celle qui mesure l'impact réel de l'attaque sur le
système qu'on cherche à protéger.

clip_values
-----------
Le classifieur ART reçoit clip_values=(0,1) depuis configs/config.yaml.
Sans cela, DeepFool/JSMA/C&W (ART) ne bornaient pas leurs perturbations
dans le domaine des features normalisées, alors que FGSM/BIM/PGD
(torchattacks) sont bornées dans [0,1] par un clamp interne à la
bibliothèque. Les deux ne sont cohérentes que si le scaler amont (05) est
bien un MinMaxScaler produisant des features dans [0,1] — c'est validé au
chargement de la config (src/utils/config.py).

Périmètre d'évaluation
-----------------------
configs/config.yaml -> evaluation.scope ("full" ou "sample") s'applique aux
six attaques de façon identique. Avant, JSMA tournait sur un échantillon de
30 000 (script 09) pendant que les cinq autres tournaient sur les 831 864
échantillons du test set complet, sans que ce soit signalé : un tableau de
résultats mélangeant deux tailles d'échantillon n'est pas comparable.
Ce script remplace entièrement 09_generate_attacks_jsma_sample.py (voir
l'en-tête de ce dernier, désormais désactivé).

Note technique : patch np.product pour compatibilité NumPy 2.x avec ART 1.18.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

if not hasattr(np, "product"):
    np.product = np.prod

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

import torchattacks

from art.attacks.evasion import CarliniL2Method, DeepFool, SaliencyMapMethod
from art.estimators.classification import PyTorchClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.dnn import BaselineDNN
from src.models.substitute import build_substitute_from_config, count_parameters
from src.utils.config import (
    load_config,
    check_data_fingerprint,
    config_fingerprint,
    describe_config_diff,
    write_fingerprint_file,
    read_fingerprint_file,
)


def load_baseline_model(cfg, device, checkpoint_dir):
    """
    Charge le baseline v4 : c'est la CIBLE, jamais la source des attaques.

    Deux verifications avant utilisation, toutes deux bloquantes (une erreur
    ici doit arreter le job, pas s'afficher comme si de rien n'etait) :

    1. Le checkpoint doit contenir 'val_acc'. Un checkpoint qui ne l'a pas
       vient du pipeline PRECEDENT (selection sur best_test_acc, avant le
       split de validation) : afficher un chiffre de repli sous une
       etiquette differente masquerait ce changement de format.
    2. Son 'config_hash' doit correspondre a cfg.baseline_fingerprint().
       baseline_best.pth est versionne dans git (negation explicite dans
       .gitignore) : un git pull peut l'apporter sur le cluster sans qu'un
       entrainement ait eu lieu localement, ou il peut avoir ete produit
       sous un scaler/des hyperparametres differents de la config actuelle.
       Contrairement au substitut, on ne peut pas reentrainer le baseline
       automatiquement ici (c'est le role de 06, pas de ce script) : la
       seule reponse correcte est d'arreter avec un message explicite.
    """
    print("Chargement du baseline v4 (cible de l'évaluation)...")
    model = BaselineDNN(
        input_dim=cfg.dataset["num_features"],
        hidden1=cfg.model["hidden_layers"][0],
        hidden2=cfg.model["hidden_layers"][1],
        output_dim=cfg.dataset["num_classes"],
    )
    checkpoint_path = checkpoint_dir / "baseline_best.pth"
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)

    if "val_acc" not in checkpoint:
        raise RuntimeError(
            f"{checkpoint_path} n'a pas de cle 'val_acc' : c'est un checkpoint du "
            f"pipeline precedent (selection sur le test, avant le split de "
            f"validation), incompatible avec la configuration actuelle. "
            f"Relancer scripts/06_train_baseline.py pour regenerer ce checkpoint."
        )

    hash_attendu = cfg.baseline_fingerprint()
    hash_checkpoint = checkpoint.get("config_hash")
    if hash_checkpoint != hash_attendu:
        diff = describe_config_diff(
            checkpoint.get("config_snapshot", {}), cfg.baseline_fingerprint_data()
        )
        raise RuntimeError(
            f"{checkpoint_path} a ete entraine sous une configuration differente "
            f"de celle chargee actuellement (config_hash={hash_checkpoint!r}, "
            f"attendu {hash_attendu!r}). L'utiliser tel quel produirait des "
            f"resultats faux sous une etiquette qui affirme le contraire.\n"
            f"Cles qui different :\n{diff}\n"
            f"Relancer scripts/06_train_baseline.py."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()
    print(f"  Meilleur epoch : {checkpoint['epoch']}")
    print(f"  Val accuracy   : {checkpoint['val_acc']:.4f}")
    print(f"  config_hash    : {hash_checkpoint} (verifie)\n")
    return model


def load_train_data(data_dir):
    """Train post-SMOTE : mêmes données que celles utilisées pour le baseline."""
    print("Chargement du train set (entraînement du substitut)...")
    X_train = pd.read_pickle(data_dir / "X_train.pkl")
    y_train = joblib.load(data_dir / "y_train.pkl")
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values.astype(np.float32)
    print(f"  X_train : {X_train.shape}\n")
    return X_train, y_train


def load_val_data(data_dir):
    """
    Split de validation : sert de gate F1 pour le substitut, jamais le test.

    Reutiliser X_test/y_test pour cette decision reproduirait exactement le
    biais qu'on vient de retirer de 06_train_baseline.py : une decision (ici,
    lancer ou non les six attaques) prise en regardant le test. Le val
    existe precisement pour ce genre de decision intermediaire.
    """
    print("Chargement du set de validation (gate F1 du substitut)...")
    X_val = pd.read_pickle(data_dir / "X_val.pkl")
    y_val = joblib.load(data_dir / "y_val.pkl")
    if isinstance(X_val, pd.DataFrame):
        X_val = X_val.values.astype(np.float32)
    print(f"  X_val : {X_val.shape}\n")
    return X_val, y_val


def load_test_data(data_dir):
    print("Chargement du test set...")
    X_test = pd.read_pickle(data_dir / "X_test.pkl")
    y_test = joblib.load(data_dir / "y_test.pkl")
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values.astype(np.float32)
    print(f"  X_test : {X_test.shape}")
    print(f"  y_test : {len(y_test):,} labels\n")
    return X_test, y_test


def select_evaluation_scope(X_test, y_test, cfg):
    """
    Applique configs/config.yaml -> evaluation.scope aux SIX attaques de
    façon identique. "sample" tire un unique échantillon stratifié partagé
    par toutes les attaques, pour que le tableau de résultats reste
    comparable colonne par colonne.
    """
    scope = cfg.evaluation["scope"]
    if scope == "full":
        print(f"Périmètre d'évaluation : test set complet ({len(X_test):,} échantillons)\n")
        return X_test, y_test

    sample_size = cfg.evaluation["sample_size"]
    print(f"Périmètre d'évaluation : échantillon stratifié partagé ({sample_size:,} échantillons)")
    print("  (les six attaques utilisent ce même échantillon)\n")
    _, X_sample, _, y_sample = train_test_split(
        X_test, y_test,
        test_size=sample_size,
        stratify=y_test,
        random_state=cfg.seed,
    )
    return X_sample.astype(np.float32), y_sample


def predict_in_batches(model, X, device, batch_size):
    """
    Prédictions par batchs, jamais un seul tensor pour tout X.

    Pousser le test set complet (831 864 x 58, plus les activations
    intermediaires) en un seul tensor peut passer sur un H100 80 Go mais
    echoue sur n'importe quel autre GPU. Reutilise le meme decoupage que
    evaluate_attack, pour un cout memoire constant quelle que soit la
    taille de X.
    """
    model.eval()
    n_samples = len(X)
    predictions = np.zeros(n_samples, dtype=np.int64)
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            X_batch = torch.tensor(X[i:end], dtype=torch.float32).to(device)
            predictions[i:end] = model(X_batch).argmax(1).cpu().numpy()
    return predictions


def train_substitute(cfg, X_train, y_train, X_val, y_val, device, checkpoint_dir, batch_size):
    """
    Entraîne le modèle substitut sur les mêmes données que le baseline.

    Validé sur X_val/y_val, jamais sur le test : reproduire ici la décision
    "regarder le test pour décider quoi faire ensuite" recréerait exactement
    le biais qu'on vient de retirer de 06_train_baseline.py. Le test set
    reste intact jusqu'à evaluate_baseline_on_clean.

    L'article rapporte un F1 moyen de 0.98 sur données propres pour le
    substitut ("We obtain the same detection ability as the IDS baseline
    classifier on the clean data samples"). Comparé en F1 PONDÉRÉ, pas
    macro : sur ce dataset à 15 classes très déséquilibrées, le baseline
    lui-même fait ~99.7% de F1 pondéré contre ~80% de F1 macro (voir
    README) — un F1 macro de 0.98 serait hors de portée d'un réseau de
    17k paramètres, donc le 0.98 de l'article correspond presque
    certainement à une moyenne pondérée. C'est un critère de validation
    avant de générer les attaques, pas un critère bloquant : un substitut
    qui n'atteint pas ce F1 reste un substitut réaliste (l'attaquant n'a
    pas forcément un modèle parfait), mais l'écart doit être visible dans
    les logs plutôt que silencieux.
    """
    print("Entraînement du modèle substitut...")
    sub_cfg = cfg.substitute
    model = build_substitute_from_config(cfg).to(device)
    print(f"  Architecture : {sub_cfg['hidden_layers']}, {count_parameters(model):,} paramètres")

    optimizer = torch.optim.Adam(model.parameters(), lr=sub_cfg["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=sub_cfg["batch_size"], shuffle=True)

    model.train()
    for epoch in range(1, sub_cfg["epochs"] + 1):
        debut = time.time()
        total_loss, correct, total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
            correct += outputs.argmax(1).eq(y_batch).sum().item()
            total += y_batch.size(0)
        print(
            f"  Epoch {epoch:2d}/{sub_cfg['epochs']} | "
            f"loss={total_loss/total:.4f} acc={correct/total:.4f} | "
            f"{time.time()-debut:.1f}s"
        )

    preds = predict_in_batches(model, X_val, device, batch_size)
    f1 = f1_score(y_val, preds, average="weighted", zero_division=0)
    print(f"\n  F1 pondéré du substitut sur validation (clean) : {f1:.4f} (cible article : {sub_cfg['f1_cible']})")
    if f1 < sub_cfg["f1_cible"] - 0.03:
        print(
            "  ATTENTION : le substitut est nettement en dessous du F1 rapporté par "
            "l'article. Les attaques transféreront probablement moins bien vers le "
            "baseline ; documenter cet écart plutôt que de l'ignorer."
        )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    substitute_checkpoint = checkpoint_dir / "substitute_best.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "f1_weighted_val": f1,
            "config_hash": cfg.substitute_fingerprint(),
            "config_snapshot": cfg.substitute_fingerprint_data(),
        },
        substitute_checkpoint,
    )
    print(f"  Substitut sauvegardé : {substitute_checkpoint}\n")
    return model


def load_or_train_substitute(cfg, X_train, y_train, X_val, y_val, device, checkpoint_dir, batch_size):
    """
    Charge le substitut en cache, ou l'entraîne s'il est absent OU périmé.

    Contrairement au baseline (checkpoint_hash invalide -> erreur, car
    reentrainer le baseline est le role de 06, pas de ce script), un
    substitut perime declenche un reentrainement automatique ici : c'est un
    cache bon marche (quelques minutes), pas un artefact couteux a
    reproduire manuellement.
    """
    substitute_checkpoint = checkpoint_dir / "substitute_best.pth"
    hash_attendu = cfg.substitute_fingerprint()
    if substitute_checkpoint.exists():
        checkpoint = torch.load(substitute_checkpoint, weights_only=False, map_location=device)
        if checkpoint.get("config_hash") == hash_attendu:
            print(f"Substitut déjà entraîné (config inchangée), chargement de {substitute_checkpoint}...")
            model = build_substitute_from_config(cfg).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            print(f"  F1 pondéré (val) au moment de l'entraînement : {checkpoint['f1_weighted_val']:.4f}\n")
            return model
        diff = describe_config_diff(
            checkpoint.get("config_snapshot", {}), cfg.substitute_fingerprint_data()
        )
        print(
            f"Substitut en cache mais config_hash perime "
            f"({checkpoint.get('config_hash')!r} != {hash_attendu!r}) : reentrainement.\n"
            f"  Cles qui different :\n{diff}\n"
        )
    return train_substitute(cfg, X_train, y_train, X_val, y_val, device, checkpoint_dir, batch_size)


def create_art_classifier(model, cfg, device):
    """
    Classifieur ART enveloppant le SUBSTITUT (source des attaques DeepFool,
    JSMA, C&W). clip_values borne le domaine à celui produit par le scaler
    (configs/config.yaml -> dataset.clip_values) ; sans ça, ART ne borne
    rien et peut produire des exemples hors du domaine des features
    normalisées.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(cfg.dataset["num_features"],),
        nb_classes=cfg.dataset["num_classes"],
        clip_values=cfg.clip_values,
        device_type="gpu" if device.type == "cuda" else "cpu",
    )


def _fingerprint_path(pkl_path):
    """Fichier sidecar JSON (hash + config) sous laquelle un X_adv a été généré."""
    return pkl_path.with_suffix(".fingerprint.json")


def load_or_none(name, attacks_dir, fingerprint_data):
    """
    Charge un X_adv déjà généré, retourne None si absent OU périmé.

    Le contenu du .pkl reste un array numpy brut (pas un dict avec
    metadonnees) : 10_evaluate_and_plot_attacks.py et les scripts de
    defense (11-14) le chargent avec joblib.load(path) en attendant un
    array directement. L'empreinte vit donc dans un fichier sidecar JSON
    plutôt que dans le pickle lui-même, pour ne rien casser en aval.

    Sans cette vérification, un X_adv_*.pkl produit par l'ancien pipeline
    (white box, sans clip_values, StandardScaler) resterait sur le cluster
    et serait chargé tel quel au prochain lancement : le script sauterait
    la génération, et le tableau de résultats afficherait "semi-white box,
    substitut, bornes [0,1]" au-dessus de chiffres qui n'ont rien à voir.
    """
    path = attacks_dir / f"X_adv_{name.lower()}.pkl"
    if not path.exists():
        return None

    sidecar = read_fingerprint_file(_fingerprint_path(path))
    hash_attendu = config_fingerprint(fingerprint_data)
    if sidecar is None or sidecar.get("hash") != hash_attendu:
        print(
            f"  Fichier {path.name} présent mais périmé ou sans empreinte de config "
            f"connue : régénération plutôt que chargement."
        )
        if sidecar is not None:
            diff = describe_config_diff(sidecar.get("config", {}), fingerprint_data)
            print(f"    Cles qui different :\n{diff}")
        return None

    print(f"  Fichier {path.name} déjà présent (config inchangée), chargement...")
    X_adv = joblib.load(path)
    taille_mb = path.stat().st_size / (1024 * 1024)
    print(f"  Chargé : {X_adv.shape} ({taille_mb:.1f} MB)\n")
    return X_adv


def generate_torchattacks_batch(attack, X_batch, y_batch, device):
    x_tensor = torch.tensor(X_batch, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_batch, dtype=torch.long).to(device)
    adv = attack(x_tensor, y_tensor)
    return adv.cpu().numpy()


def generate_torchattacks(attack, X, y, device, name, batch_size):
    """Génère les adversariaux avec torchattacks (FGSM, BIM, PGD), par batchs."""
    print(f"Génération {name} (sur le substitut)...")
    debut = time.time()
    n_samples = len(X)
    X_adv = np.zeros_like(X)
    n_batches = (n_samples + batch_size - 1) // batch_size

    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, n_samples)
        X_adv[start:end] = generate_torchattacks_batch(attack, X[start:end], y[start:end], device)
        if (i + 1) % 100 == 0 or (i + 1) == n_batches:
            pct = 100 * (i + 1) / n_batches
            elapsed = time.time() - debut
            print(f"  Batch {i+1}/{n_batches} ({pct:.1f}%) - {elapsed:.1f}s", flush=True)

    duree = time.time() - debut
    print(f"  {name} terminé en {duree:.1f}s ({duree/60:.1f} min)\n")
    return X_adv


def generate_art_attack(attack, X, y, name):
    """Génère les adversariaux avec ART (sur le substitut). y=None pour untargeted."""
    print(f"Génération {name} (sur le substitut)...")
    debut = time.time()
    X_np = X.astype(np.float32)
    X_adv = attack.generate(x=X_np) if y is None else attack.generate(x=X_np, y=y)
    duree = time.time() - debut
    print(f"  {name} terminé en {duree:.1f}s ({duree/60:.1f} min)\n")
    return X_adv


def warn_if_slow(attack, X, y, name, n_total, sample_n=5000, seuil_heures=8.0, random_state=42):
    """
    Estime la durée totale de generate() a partir d'un echantillon
    STRATIFIE, AVANT de lancer l'attaque en grandeur reelle.

    Raison d'etre : le README documente que JSMA avec les parametres fideles
    a l'article (theta=0.1, gamma=1.0, Table 2) prenait environ 12 jours sur
    le test set complet avec le baseline (512-256), ce qui avait motive le
    detour vers theta=0.3, gamma=0.15. configs/config.yaml est revenu aux
    valeurs de la Table 2. Le substitut (100-100, 17k parametres) est plus
    petit, mais le cout de JSMA depend surtout du nombre de features et de
    classes, pas seulement de la largeur des couches cachees : rien ne
    garantit que le probleme de duree soit resolu. Mieux vaut le decouvrir
    sur un echantillon de quelques secondes que sur plusieurs jours de job
    SLURM.

    L'echantillon est stratifie et pas les n premieres lignes de X : le
    cout de JSMA depend du nombre d'iterations avant bascule, donc de la
    classe visee, et le test set est trie/regroupe par nature (issu d'un
    split stratifie mais pas mélangé en sortie). Prendre les 1000 premieres
    lignes brutes donnerait une estimation dominee par une poignee de
    classes, bruitee au point de ne rien dire d'utile.

    Le tirage stratifie lui-meme peut echouer : stratifier 831 864 lignes
    vers 5 000 avec une classe a quelques membres (Heartbleed) est
    precisement le regime ou StratifiedShuffleSplit devient capricieux,
    encore plus sous evaluation.scope="sample" ou la marge est plus mince.
    Repli sur un tirage aleatoire simple si la stratification echoue :
    cette fonction est purement informative, un echantillon moins
    representatif reste largement suffisant pour une estimation de duree.

    N'interrompt rien : imprime une estimation et une recommandation.
    La decision (attendre, ou repasser evaluation.scope a "sample" dans
    configs/config.yaml) revient a l'utilisateur.
    """
    n_echantillon = min(sample_n, n_total)
    if n_echantillon < n_total:
        try:
            _, X_echantillon, _, _ = train_test_split(
                X, y, test_size=n_echantillon, stratify=y, random_state=random_state,
            )
        except ValueError as erreur:
            print(
                f"  Echantillonnage stratifie impossible pour l'estimation "
                f"({erreur}) : repli sur un tirage aleatoire simple."
            )
            _, X_echantillon = train_test_split(
                X, test_size=n_echantillon, random_state=random_state,
            )
    else:
        X_echantillon = X
    X_echantillon = X_echantillon.astype(np.float32)

    print(f"Estimation de duree pour {name} sur {len(X_echantillon):,} echantillons stratifies...")
    debut = time.time()
    try:
        attack.generate(x=X_echantillon)
    except Exception as erreur:
        # Purement informatif : un echec ici (cas limite sur l'echantillon
        # d'estimation) ne doit jamais empecher la generation reelle qui suit.
        print(f"  Estimation impossible ({erreur}), poursuite sans estimation.\n")
        return
    duree_echantillon = time.time() - debut
    n_echantillon = len(X_echantillon)

    duree_par_exemple = duree_echantillon / n_echantillon
    duree_totale_estimee = duree_par_exemple * n_total
    heures = duree_totale_estimee / 3600

    print(f"  {duree_echantillon:.1f}s pour {n_echantillon:,} exemples "
          f"-> estimation pour {n_total:,} exemples : {heures:.1f}h")

    if heures > seuil_heures:
        print(
            f"  ATTENTION : duree estimee ({heures:.1f}h) au-dela du seuil "
            f"informatif ({seuil_heures}h). Options : augmenter le temps alloue "
            f"au job SLURM en consequence, ou repasser evaluation.scope a "
            f"'sample' dans configs/config.yaml pour partager un echantillon "
            f"stratifie entre les six attaques (voir select_evaluation_scope)."
        )
    print()


def evaluate_attack(baseline_model, X_adv, y_true, device, name, batch_size, data_dir):
    """
    Évalue la TRANSFÉRABILITÉ : les exemples générés sur le substitut sont
    ici passés au baseline. C'est cette évaluation, pas l'accuracy du
    substitut, qui mesure la vulnérabilité réelle de l'IDS.
    """
    print(f"Évaluation sur {name} (transfert vers le baseline)...")
    baseline_model.eval()
    n_samples = len(X_adv)
    predictions = np.zeros(n_samples, dtype=np.int64)

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            X_batch = torch.tensor(X_adv[i:end], dtype=torch.float32).to(device)
            outputs = baseline_model(X_batch)
            _, preds = outputs.max(1)
            predictions[i:end] = preds.cpu().numpy()

    acc = accuracy_score(y_true, predictions)
    precision_macro = precision_score(y_true, predictions, average="macro", zero_division=0)
    precision_weighted = precision_score(y_true, predictions, average="weighted", zero_division=0)
    recall_macro = recall_score(y_true, predictions, average="macro", zero_division=0)
    recall_weighted = recall_score(y_true, predictions, average="weighted", zero_division=0)
    f1_macro = f1_score(y_true, predictions, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, predictions, average="weighted", zero_division=0)

    print(f"  Accuracy               : {acc:.4f}")
    print(f"  Precision (macro)      : {precision_macro:.4f}")
    print(f"  Precision (weighted)   : {precision_weighted:.4f}")
    print(f"  Recall (macro)         : {recall_macro:.4f}")
    print(f"  Recall (weighted)      : {recall_weighted:.4f}")
    print(f"  F1 (macro)             : {f1_macro:.4f}")
    print(f"  F1 (weighted)          : {f1_weighted:.4f}")

    label_encoder = joblib.load(data_dir / "label_encoder.pkl")
    all_labels = list(range(len(label_encoder.classes_)))
    target_names = [str(c) for c in label_encoder.classes_]

    print(f"\n  Rapport par classe pour {name} :")
    print(classification_report(
        y_true, predictions,
        labels=all_labels,
        target_names=target_names,
        digits=4,
        zero_division=0,
    ))

    cm = confusion_matrix(y_true, predictions, labels=all_labels)

    return {
        "attack": name,
        "accuracy": acc,
        "precision_macro": precision_macro,
        "precision_weighted": precision_weighted,
        "recall_macro": recall_macro,
        "recall_weighted": recall_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "predictions": predictions,
        "confusion_matrix": cm,
    }


def save_adversarial_examples(X_adv, name, output_dir, fingerprint_data):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"X_adv_{name.lower()}.pkl"
    joblib.dump(X_adv, path)
    write_fingerprint_file(_fingerprint_path(path), fingerprint_data)
    taille_mb = path.stat().st_size / (1024 * 1024)
    print(f"  Sauvegardé : {path.name} ({taille_mb:.1f} MB)\n")


def print_summary_table(all_results, baseline_metrics):
    print("\n" + "=" * 100)
    print("Résumé complet des attaques adversariales (transfert substitut -> baseline)")
    print("=" * 100)
    header = (
        f"{'Attaque':<12} {'Accuracy':>10} {'Prec. macro':>12} {'Prec. wght':>12} "
        f"{'Rec. macro':>11} {'Rec. wght':>11} {'F1 macro':>10} {'F1 wght':>10}"
    )
    print(header)
    print("-" * 100)
    print(
        f"{'BENIGN':<12} {baseline_metrics['accuracy']:>10.4f} "
        f"{baseline_metrics['precision_macro']:>12.4f} {baseline_metrics['precision_weighted']:>12.4f} "
        f"{baseline_metrics['recall_macro']:>11.4f} {baseline_metrics['recall_weighted']:>11.4f} "
        f"{baseline_metrics['f1_macro']:>10.4f} {baseline_metrics['f1_weighted']:>10.4f}  (baseline)"
    )
    print("-" * 100)
    for r in all_results:
        print(
            f"{r['attack']:<12} {r['accuracy']:>10.4f} "
            f"{r['precision_macro']:>12.4f} {r['precision_weighted']:>12.4f} "
            f"{r['recall_macro']:>11.4f} {r['recall_weighted']:>11.4f} "
            f"{r['f1_macro']:>10.4f} {r['f1_weighted']:>10.4f}"
        )
    print("=" * 100)


def evaluate_baseline_on_clean(model, X_test, y_test, device, batch_size, data_dir):
    print("Évaluation du baseline sur données propres (référence)...")
    result = evaluate_attack(model, X_test, y_test, device, "BENIGN (clean)", batch_size, data_dir)
    return {k: result[k] for k in (
        "accuracy", "precision_macro", "precision_weighted",
        "recall_macro", "recall_weighted", "f1_macro", "f1_weighted",
    )}


def main():
    print("Génération des attaques adversariales (semi-white box, substitut -> baseline)")
    print(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    cfg = load_config()
    print(cfg.resume())
    print()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}\n")

    batch_size = cfg.evaluation["batch_size"]

    data_dir = Path(cfg.paths["data_processed"])
    checkpoint_dir = Path(cfg.paths["checkpoints"])
    log_dir = Path(cfg.paths["logs"])
    attacks_dir = Path(cfg.paths["attacks"])

    # Verifie que data/processed/ correspond a la configuration courante,
    # AVANT tout chargement (meme raison qu'en 06 : baseline/substitut/
    # attaque sont tous calcules depuis la config en vigueur, pas depuis
    # ce qui est reellement sur le disque).
    check_data_fingerprint(cfg, data_dir)

    baseline_model = load_baseline_model(cfg, device, checkpoint_dir)
    X_train, y_train = load_train_data(data_dir)
    X_val, y_val = load_val_data(data_dir)
    X_test_full, y_test_full = load_test_data(data_dir)
    X_test, y_test = select_evaluation_scope(X_test_full, y_test_full, cfg)

    substitute_model = load_or_train_substitute(
        cfg, X_train, y_train, X_val, y_val, device, checkpoint_dir, batch_size
    )
    art_classifier = create_art_classifier(substitute_model, cfg, device)

    baseline_metrics = evaluate_baseline_on_clean(
        baseline_model, X_test, y_test, device, batch_size, data_dir
    )
    all_results = []

    print("=" * 70)
    print("1. FGSM")
    print("=" * 70 + "\n")
    params = cfg.attack_params("FGSM")
    fgsm_fp = cfg.attack_fingerprint_data("FGSM")
    X_adv_fgsm = load_or_none("FGSM", attacks_dir, fgsm_fp)
    if X_adv_fgsm is None:
        attack = torchattacks.FGSM(substitute_model, eps=params["eps"])
        X_adv_fgsm = generate_torchattacks(attack, X_test, y_test, device, "FGSM", batch_size)
        save_adversarial_examples(X_adv_fgsm, "FGSM", attacks_dir, fgsm_fp)
    result = evaluate_attack(baseline_model, X_adv_fgsm, y_test, device, "FGSM", batch_size, data_dir)
    all_results.append(result)
    del X_adv_fgsm

    print("=" * 70)
    print("2. BIM")
    print("=" * 70 + "\n")
    params = cfg.attack_params("BIM")
    bim_fp = cfg.attack_fingerprint_data("BIM")
    X_adv_bim = load_or_none("BIM", attacks_dir, bim_fp)
    if X_adv_bim is None:
        attack = torchattacks.BIM(
            substitute_model, eps=params["eps"], alpha=params["alpha"], steps=params["steps"],
        )
        X_adv_bim = generate_torchattacks(attack, X_test, y_test, device, "BIM", batch_size)
        save_adversarial_examples(X_adv_bim, "BIM", attacks_dir, bim_fp)
    result = evaluate_attack(baseline_model, X_adv_bim, y_test, device, "BIM", batch_size, data_dir)
    all_results.append(result)
    del X_adv_bim

    print("=" * 70)
    print("3. PGD")
    print("=" * 70 + "\n")
    params = cfg.attack_params("PGD")
    pgd_fp = cfg.attack_fingerprint_data("PGD")
    X_adv_pgd = load_or_none("PGD", attacks_dir, pgd_fp)
    if X_adv_pgd is None:
        attack = torchattacks.PGD(
            substitute_model, eps=params["eps"], alpha=params["alpha"], steps=params["steps"],
        )
        X_adv_pgd = generate_torchattacks(attack, X_test, y_test, device, "PGD", batch_size)
        save_adversarial_examples(X_adv_pgd, "PGD", attacks_dir, pgd_fp)
    result = evaluate_attack(baseline_model, X_adv_pgd, y_test, device, "PGD", batch_size, data_dir)
    all_results.append(result)
    del X_adv_pgd

    print("=" * 70)
    print("4. DeepFool")
    print("=" * 70 + "\n")
    params = cfg.attack_params("DeepFool")
    deepfool_fp = cfg.attack_fingerprint_data("DeepFool")
    X_adv_df = load_or_none("DeepFool", attacks_dir, deepfool_fp)
    if X_adv_df is None:
        attack = DeepFool(
            classifier=art_classifier,
            max_iter=params["max_iter"],
            epsilon=params["epsilon"],
            batch_size=batch_size,
        )
        X_adv_df = generate_art_attack(attack, X_test, y_test, "DeepFool")
        save_adversarial_examples(X_adv_df, "DeepFool", attacks_dir, deepfool_fp)
    result = evaluate_attack(baseline_model, X_adv_df, y_test, device, "DeepFool", batch_size, data_dir)
    all_results.append(result)
    del X_adv_df

    print("=" * 70)
    print("5. JSMA (untargeted)")
    print("=" * 70 + "\n")
    params = cfg.attack_params("JSMA")
    jsma_fp = cfg.attack_fingerprint_data("JSMA")
    X_adv_jsma = load_or_none("JSMA", attacks_dir, jsma_fp)
    if X_adv_jsma is None:
        attack = SaliencyMapMethod(
            classifier=art_classifier,
            theta=params["theta"],
            gamma=params["gamma"],
            batch_size=batch_size,
        )
        warn_if_slow(attack, X_test, y_test, "JSMA", n_total=len(X_test), random_state=cfg.seed)
        X_adv_jsma = generate_art_attack(attack, X_test, None, "JSMA")
        save_adversarial_examples(X_adv_jsma, "JSMA", attacks_dir, jsma_fp)
    result = evaluate_attack(baseline_model, X_adv_jsma, y_test, device, "JSMA", batch_size, data_dir)
    all_results.append(result)
    del X_adv_jsma

    print("=" * 70)
    print("6. C&W (untargeted)")
    print("=" * 70 + "\n")
    params = cfg.attack_params("CW")
    cw_fp = cfg.attack_fingerprint_data("CW")
    X_adv_cw = load_or_none("CW", attacks_dir, cw_fp)
    if X_adv_cw is None:
        attack = CarliniL2Method(
            classifier=art_classifier,
            max_iter=params["max_iter"],
            confidence=params["confidence"],
            binary_search_steps=params["binary_search_steps"],
            initial_const=params["initial_const"],
            learning_rate=params["learning_rate"],
            batch_size=batch_size,
        )
        X_adv_cw = generate_art_attack(attack, X_test, None, "CW")
        save_adversarial_examples(X_adv_cw, "CW", attacks_dir, cw_fp)
    result = evaluate_attack(baseline_model, X_adv_cw, y_test, device, "CW", batch_size, data_dir)
    all_results.append(result)
    del X_adv_cw

    print_summary_table(all_results, baseline_metrics)

    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    joblib.dump(
        {
            "results": all_results,
            "baseline_metrics": baseline_metrics,
            "attack_configs": {name: cfg.attack_params(name) for name in cfg.attacks},
            "evaluation_scope": cfg.evaluation["scope"],
        },
        log_dir / f"attacks_results_{timestamp}.pkl",
    )
    print(f"\nRésultats sauvegardés : {log_dir / f'attacks_results_{timestamp}.pkl'}")
    print("\nGénération et évaluation des attaques terminées")


if __name__ == "__main__":
    main()
