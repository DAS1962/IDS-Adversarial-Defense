
"""
Chargement centralise de la configuration.

Raison d'etre : avant cette version, chaque script redefinissait ses propres
constantes (ATTACK_CONFIGS dans 08 et 09, TEST_SIZE dans 05, etc.) alors que
configs/config.yaml contenait deja ces valeurs. Le YAML n'etait lu que par
00_test_environment.py. Les deux sources ont diverge sans que ce soit visible :
JSMA tournait avec theta=0.3 alors que le YAML disait 0.1, et C&W avec
max_iter=10 alors que le YAML disait 9.

Regle a partir de maintenant : aucun hyperparametre en dur dans scripts/.
Tout passe par load_config().

Usage typique :

    from src.utils.config import load_config

    cfg = load_config()
    eps = cfg.attacks["FGSM"]["eps"]
    clip = cfg.clip_values          # (0.0, 1.0)
"""

import hashlib
import json
from pathlib import Path

import yaml


DEFAULT_CONFIG_PATH = Path("configs/config.yaml")


def config_fingerprint(relevant: dict) -> str:
    """
    Hash court et stable d'un sous-ensemble de configuration.

    Sert a invalider un artefact mis en cache (checkpoint, exemples
    adversariaux) quand la configuration dont il depend a change entre deux
    executions. Sans ca, un script qui "saute la generation si le fichier
    existe deja" chargerait silencieusement un artefact produit sous une
    configuration perimee (ancien scaler, anciens parametres d'attaque,
    ancien format de checkpoint) et produirait des resultats faux avec un
    en-tete qui affirme le contraire.
    """
    serialise = json.dumps(relevant, sort_keys=True, default=str)
    return hashlib.sha256(serialise.encode("utf-8")).hexdigest()[:16]


def _flatten(d: dict, prefix: str = "") -> dict:
    """Aplati un dict imbrique en {"a.b.c": valeur} pour un diff cle par cle."""
    plat = {}
    for k, v in d.items():
        cle = k if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            plat.update(_flatten(v, cle))
        else:
            plat[cle] = v
    return plat


def describe_config_diff(stored: dict, current: dict) -> str:
    """
    Rapport lisible des cles qui different entre deux configurations.

    Sans ca, un desaccord de hash n'affiche que deux chaines opaques : on
    sait qu'il y a un ecart mais pas lequel, et il faut relancer a l'aveugle
    en esperant que ca corrige le probleme.
    """
    stored_plat = _flatten(stored or {})
    current_plat = _flatten(current or {})
    toutes_cles = sorted(set(stored_plat) | set(current_plat))
    lignes = [
        f"  {cle} : {stored_plat.get(cle, '<absent>')!r} -> {current_plat.get(cle, '<absent>')!r}"
        for cle in toutes_cles
        if stored_plat.get(cle, "<absent>") != current_plat.get(cle, "<absent>")
    ]
    return "\n".join(lignes) if lignes else "  (aucune cle differente detectee - verifier l'ordre ou un type non stable)"


def write_fingerprint_file(path: Path, fingerprint_data: dict) -> None:
    """Ecrit {"hash":..., "config":...} en JSON, format partage par tous les sidecars d'empreinte."""
    payload = {
        "hash": config_fingerprint(fingerprint_data),
        "config": fingerprint_data,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str, sort_keys=True)


def read_fingerprint_file(path: Path) -> dict | None:
    """Lit un sidecar ecrit par write_fingerprint_file, ou None si absent/illisible."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


class Config:
    """
    Acces par attribut aux sections du YAML, avec validation au chargement.

    On expose les sections telles quelles (dicts) plutot que de creer une
    classe par section : le YAML est la reference, dupliquer sa structure en
    Python recreerait exactement le probleme de double source qu'on corrige.
    """

    #  Parametres attendus pour chaque attaque, d'apres la Table 2 de
    #  Awad et al. (2025). Sert a detecter un YAML incomplet AVANT de lancer
    #  plusieurs heures de calcul sur le cluster.
    REQUIRED_ATTACK_KEYS = {
        "FGSM": {"eps"},
        "BIM": {"eps", "alpha", "steps"},
        "PGD": {"eps", "alpha", "steps"},
        "DeepFool": {"epsilon", "max_iter"},
        "JSMA": {"theta", "gamma"},
        "CW": {"max_iter", "confidence", "binary_search_steps",
               "initial_const", "learning_rate"},
    }

    def __init__(self, raw: dict, source: Path):
        self._raw = raw
        self.source = source

        self.env = raw["env"]
        self.paths = raw["paths"]
        self.dataset = raw["dataset"]
        self.model = raw["model"]
        self.substitute = raw["substitute"]
        self.training = raw["training"]
        self.attacks = raw["attacks"]
        self.evaluation = raw["evaluation"]
        self.defenses = raw["defenses"]
        self.ensemble = raw["ensemble"]

        self._validate()

    @property
    def clip_values(self):
        """
        Bornes du domaine des features, sous la forme attendue par ART.

        Critique : les features sont ramenees dans [0,1] par MinMaxScaler
        (cf. article, section Preprocessing). ART ne borne rien si on ne lui
        passe pas clip_values, et torchattacks borne dans [0,1] en dur. Les
        deux bibliotheques ne sont coherentes que si le scaler est bien un
        MinMaxScaler. Ne pas changer l'un sans l'autre.
        """
        bornes = self.dataset["clip_values"]
        return (float(bornes[0]), float(bornes[1]))

    @property
    def seed(self):
        return int(self.env["random_seed"])

    def data_fingerprint_data(self) -> dict:
        """
        Sous-ensemble de configuration qui rend data/processed/ (X_train,
        X_val, X_test, produits par 05_split_and_prepare.py) valide ou perime.

        Fondation des trois autres empreintes (baseline, substitut, attaque) :
        toutes en dependent, puisque toutes consomment ces fichiers en aval.
        Inclut dataset.smote_strategy, qui est le plus gros hyperparametre du
        projet hors de l'article (les 11 plafonds par classe) et qui, avant
        cet ajout, pouvait changer sans qu'aucune empreinte ne bouge.
        """
        return {
            "scaler": self.dataset["scaler"],
            "clip_values": self.dataset["clip_values"],
            "test_size": self.dataset["test_size"],
            "val_size": self.dataset["val_size"],
            "seed": self.seed,
            "smote_strategy": self.dataset["smote_strategy"],
        }

    def data_fingerprint(self) -> str:
        return config_fingerprint(self.data_fingerprint_data())

    def baseline_fingerprint_data(self) -> dict:
        return {
            "data": self.data_fingerprint_data(),
            "model": self.model,
            "training": self.training,
        }

    def baseline_fingerprint(self) -> str:
        """
        Empreinte de tout ce qui rend un checkpoint baseline valide ou perime.

        06_train_baseline.py l'ecrit dans chaque checkpoint sauvegarde ;
        08_generate_attacks.py la revalide avant d'utiliser le baseline
        comme cible d'evaluation. baseline_best.pth est versionne dans git
        (negation explicite dans .gitignore) : un git pull peut l'apporter
        sur le cluster sans qu'aucun entrainement n'ait eu lieu localement.
        Sans cette empreinte, un tel checkpoint serait charge sans erreur
        meme s'il a ete produit sous un scaler ou des hyperparametres
        differents de la configuration courante.

        Cette empreinte seule ne suffit PAS a garantir que le checkpoint
        correspond aux donnees reellement sur le disque : elle est calculee
        depuis la configuration EN VIGUEUR au moment de l'appel, pas depuis
        ce que 05 a effectivement ecrit. Voir check_data_fingerprint(), qui
        ferme ce trou en verifiant data/processed/data_fingerprint.json
        independamment.
        """
        return config_fingerprint(self.baseline_fingerprint_data())

    def substitute_fingerprint_data(self) -> dict:
        return {
            "data": self.data_fingerprint_data(),
            "substitute": self.substitute,
        }

    def substitute_fingerprint(self) -> str:
        """
        Empreinte de ce qui rend un checkpoint substitut valide ou perime.

        Le substitut est un cache bon marche a regenerer (quelques minutes) :
        contrairement au baseline, un substitut perime declenche un
        reentrainement automatique plutot qu'une erreur bloquante.
        """
        return config_fingerprint(self.substitute_fingerprint_data())

    def evaluation_scope_data(self) -> dict:
        """
        Sous-ensemble de evaluation.* qui change QUELLES lignes sont
        attaquees. batch_size est exclu expres : c'est un levier purement
        calculatoire (vitesse, memoire), il ne change pas la sortie de
        FGSM/BIM/PGD/DeepFool/JSMA/C&W. L'inclure dans l'empreinte des
        attaques invaliderait les six X_adv (potentiellement des jours de
        JSMA/C&W) pour un simple ajustement de taille de batch au GPU
        disponible.
        """
        return {
            "scope": self.evaluation["scope"],
            "sample_size": self.evaluation["sample_size"],
        }

    def attack_fingerprint_data(self, name: str) -> dict:
        return {
            "attack_name": name,
            "data": self.data_fingerprint_data(),
            "substitute": self.substitute,
            "evaluation_scope": self.evaluation_scope_data(),
            "attack_params": self.attack_params(name),
        }

    def attack_fingerprint(self, name: str) -> str:
        """
        Empreinte de ce qui rend un fichier X_adv_<name>.pkl valide ou perime.

        Couvre tout ce qui influence la GENERATION de cette attaque : les
        donnees sous-jacentes (data_fingerprint_data, donc scaler/clip_values/
        test_size/val_size/smote_strategy), l'architecture et les
        hyperparametres du substitut (source des attaques), le perimetre
        d'evaluation SANS son batch_size (scope/sample_size determinent
        quelles lignes sont attaquees ; batch_size ne change que la vitesse),
        et les parametres propres a cette attaque. Un fichier X_adv existant
        dont l'empreinte ne correspond plus est traite comme absent :
        regenere, pas charge tel quel.
        """
        return config_fingerprint(self.attack_fingerprint_data(name))

    def attack_params(self, name: str) -> dict:
        """
        Retourne les parametres d'une attaque, en echouant explicitement si
        elle est absente plutot qu'en retombant sur des valeurs par defaut.
        """
        if name not in self.attacks:
            connues = ", ".join(sorted(self.attacks))
            raise KeyError(
                f"Attaque '{name}' absente de {self.source}. Attaques definies : {connues}"
            )
        return dict(self.attacks[name])

    def _validate(self):
        """Verifie la coherence du YAML avant tout calcul."""
        erreurs = []

        if self.dataset["scaler"] != "MinMaxScaler":
            erreurs.append(
                f"dataset.scaler vaut '{self.dataset['scaler']}' mais les bornes "
                f"clip_values et le clamp interne de torchattacks supposent "
                f"MinMaxScaler. Avec StandardScaler les valeurs negatives sont "
                f"ecrasees a zero par torchattacks."
            )

        bornes = self.dataset["clip_values"]
        if len(bornes) != 2 or bornes[0] >= bornes[1]:
            erreurs.append(f"dataset.clip_values invalide : {bornes}")

        for attaque, requises in self.REQUIRED_ATTACK_KEYS.items():
            if attaque not in self.attacks:
                erreurs.append(f"attacks.{attaque} manquante")
                continue
            manquantes = requises - set(self.attacks[attaque])
            if manquantes:
                erreurs.append(
                    f"attacks.{attaque} : cles manquantes {sorted(manquantes)}"
                )

        if self.substitute["input_dim"] != self.dataset["num_features"]:
            erreurs.append(
                f"substitute.input_dim ({self.substitute['input_dim']}) != "
                f"dataset.num_features ({self.dataset['num_features']})"
            )

        if self.substitute["output_dim"] != self.dataset["num_classes"]:
            erreurs.append(
                f"substitute.output_dim ({self.substitute['output_dim']}) != "
                f"dataset.num_classes ({self.dataset['num_classes']})"
            )

        fraction_test = self.dataset["test_size"]
        fraction_val = self.dataset["val_size"]
        if fraction_test + fraction_val >= 1.0:
            erreurs.append(
                f"dataset.test_size ({fraction_test}) + dataset.val_size ({fraction_val}) "
                f">= 1.0 : il ne resterait aucune donnee pour l'entrainement."
            )

        if self.evaluation["scope"] not in ("full", "sample"):
            erreurs.append(
                f"evaluation.scope vaut '{self.evaluation['scope']}', attendu 'full' ou 'sample'."
            )

        smote_strategy = self.dataset.get("smote_strategy")
        if not isinstance(smote_strategy, dict) or not smote_strategy:
            erreurs.append(
                "dataset.smote_strategy manquante ou vide : doit lister au moins "
                "un plafond par classe (cle = id de classe, valeur = effectif cible)."
            )
        elif any(not isinstance(v, int) or v <= 0 for v in smote_strategy.values()):
            erreurs.append(
                f"dataset.smote_strategy : toutes les valeurs doivent etre des "
                f"entiers positifs, recu {smote_strategy}."
            )

        if erreurs:
            details = "\n  - ".join(erreurs)
            raise ValueError(f"Configuration invalide ({self.source}) :\n  - {details}")

    def resume(self) -> str:
        """Resume lisible, a logger en tete de chaque script."""
        lignes = [
            f"Configuration      : {self.source}",
            f"Seed               : {self.seed}",
            f"Scaler             : {self.dataset['scaler']} -> clip_values {self.clip_values}",
            f"Features / classes : {self.dataset['num_features']} / {self.dataset['num_classes']}",
            f"Split (train/val/test) fractions : "
            f"{1 - self.dataset['test_size'] - self.dataset['val_size']:.3f} / "
            f"{self.dataset['val_size']:.3f} / {self.dataset['test_size']:.3f}",
            f"Perimetre d'evaluation des attaques : {self.evaluation['scope']}",
            f"Baseline           : {self.model['hidden_layers']}",
            f"Substitut          : {self.substitute['hidden_layers']} "
            f"(lr={self.substitute['learning_rate']}, "
            f"epochs={self.substitute['epochs']}, "
            f"batch={self.substitute['batch_size']})",
        ]
        for nom in sorted(self.attacks):
            params = ", ".join(f"{k}={v}" for k, v in sorted(self.attacks[nom].items()))
            lignes.append(f"  {nom:<9} : {params}")
        return "\n".join(lignes)


def write_data_fingerprint(cfg: Config, data_dir: Path) -> Path:
    """
    Ecrit data/processed/data_fingerprint.json a la fin de 05_split_and_prepare.py.

    C'est la piece qui manquait au mecanisme d'empreintes : baseline/
    substitut/attaque sont calcules depuis la configuration EN VIGUEUR au
    moment de l'appel, jamais depuis ce qui a ete effectivement ecrit sur
    le disque. Sans ce fichier, le scenario suivant passe inapercu :
    lancer 05 avec dataset.val_size=0.05, changer la valeur a 0.10 dans le
    YAML SANS relancer 05, puis lancer 06 - qui embarquerait une empreinte
    coherente avec 0.10 dans le checkpoint, alors que les donnees sur le
    disque ont ete generees sous 0.05. Le checkpoint et son empreinte
    seraient coherents entre eux, faux par rapport aux donnees reelles.
    """
    path = Path(data_dir) / "data_fingerprint.json"
    write_fingerprint_file(path, cfg.data_fingerprint_data())
    return path


def check_data_fingerprint(cfg: Config, data_dir: Path) -> None:
    """
    Verifie que data/processed/ correspond a la configuration courante.

    A appeler en tout debut de main() dans 06 et 08, avant toute autre
    utilisation des donnees. Complementaire aux empreintes baseline/
    substitut/attaque (qui verifient qu'UN ARTEFACT correspond a la config
    courante) : celle-ci verifie que LES DONNEES SOURCE elles-memes y
    correspondent encore, ce qu'aucune des trois autres ne peut detecter
    puisqu'elles sont toutes calculees depuis la meme configuration en
    vigueur, pas depuis un etat verifie du disque.
    """
    path = Path(data_dir) / "data_fingerprint.json"
    stocke = read_fingerprint_file(path)
    if stocke is None:
        raise RuntimeError(
            f"{path} introuvable ou illisible. Lancer scripts/05_split_and_prepare.py "
            f"avant d'entrainer un modele ou de generer des attaques."
        )

    hash_attendu = cfg.data_fingerprint()
    if stocke.get("hash") != hash_attendu:
        diff = describe_config_diff(stocke.get("config", {}), cfg.data_fingerprint_data())
        raise RuntimeError(
            f"{path} ne correspond plus a la configuration courante "
            f"(hash {stocke.get('hash')!r} != {hash_attendu!r}).\n"
            f"Cles qui different :\n{diff}\n"
            f"Relancer scripts/05_split_and_prepare.py pour regenerer les donnees "
            f"sous la configuration actuelle."
        )


def load_config(path=None) -> Config:
    """
    Charge et valide la configuration.

    Args:
        path: chemin vers le YAML. Par defaut configs/config.yaml, resolu
              depuis le repertoire courant (les scripts sont lances depuis
              la racine du projet).
    """
    chemin = Path(path) if path is not None else DEFAULT_CONFIG_PATH

    if not chemin.exists():
        raise FileNotFoundError(
            f"Configuration introuvable : {chemin.resolve()}. "
            f"Les scripts doivent etre lances depuis la racine du projet."
        )

    with open(chemin, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return Config(raw, chemin)


if __name__ == "__main__":
    cfg = load_config()
    print(cfg.resume())