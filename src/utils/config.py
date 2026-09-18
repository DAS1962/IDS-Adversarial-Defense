"""Chargement de la config."""

from pathlib import Path
import yaml


class Config:
    def __init__(self, raw):
        self.raw = raw
        for cle in ("paths", "dataset", "feature_selection", "model",
                    "substitute", "attacks", "evaluation", "defenses"):
            setattr(self, cle, raw[cle])
        self.seed = raw["seed"]

    @property
    def clip_values(self):
        a, b = self.dataset["clip_values"]
        return (float(a), float(b))

    def resume(self):
        d, f, e = self.dataset, self.feature_selection, self.evaluation
        return "\n".join([
            f"Seed               : {self.seed}",
            f"Doublons retires   : {d['drop_duplicates']}",
            f"Scaler             : {d['scaler']} -> {self.clip_values}",
            f"Equilibrage        : {d['balance_fraction']:.0%} par classe, "
            f"seuil {d['balance_min_instances']}",
            f"Selection features : mode {f['mode']}, retire {f['n_remove']}",
            f"Split test         : {d['test_size']}",
            f"Evaluation         : {e['scope']} ({e['n_samples']} echantillons, "
            f"{e['benign_fraction']:.0%} benin)",
        ])


def load_config(chemin="configs/config.yaml"):
    p = Path(chemin)
    if not p.exists():
        raise FileNotFoundError(
            f"{p.resolve()} introuvable. Lancer depuis la racine du projet."
        )
    return Config(yaml.safe_load(p.read_text(encoding="utf-8")))


if __name__ == "__main__":
    print(load_config().resume())
