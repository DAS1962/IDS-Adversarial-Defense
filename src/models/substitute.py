
"""
Modele substitut pour la generation des exemples adversariaux.

Pourquoi ce fichier existe
--------------------------
La version precedente du pipeline generait les six attaques directement sur
le baseline (`torchattacks.FGSM(model, ...)` et `PyTorchClassifier(model=model)`
recevaient tous deux BaselineDNN). L'attaquant disposait donc des vrais
gradients du modele attaque : c'est du white box complet.

L'article place l'attaquant en semi-white box. Section "Adversarial examples
generation" :

    "we consider as a substitute model a fully connected feed-forward neural
     network classifier with an input layer of dimension 58 [...] followed by
     two layers of 100 neurons each. The output layer contains 15 neurons
     [...] learning rate of 0.01 within 30 training epochs, 256 mini-batch
     size, and categorical cross-entropy loss function. We obtain the same
     detection ability as the IDS baseline classifier on the clean data
     samples, with an average F1 score of 0.98."

L'attaquant connait donc l'architecture generale mais pas les parametres du
baseline. Les exemples adversariaux se transferent du substitut vers le
baseline, ce qui est nettement moins destructeur que le white box et rend
les chiffres comparables a ceux de l'article.

Architecture : 58 -> 100 -> 100 -> 15

Comme pour BaselineDNN, pas de Softmax explicite : CrossEntropyLoss applique
LogSoftmax en interne. En ajouter un ici provoquerait une double application.
"""

import torch
import torch.nn as nn


class SubstituteDNN(nn.Module):
    """
    Reseau substitut, volontairement plus petit que le baseline.

    - Couche 1 : 58 -> 100, ReLU
    - Couche 2 : 100 -> 100, ReLU
    - Couche 3 : 100 -> 15 (logits)

    Environ 17 000 parametres, contre 165 000 pour le baseline. L'ecart est
    voulu : le substitut n'est pas une copie du baseline, c'est le modele
    approximatif dont dispose un attaquant realiste.
    """

    def __init__(self, input_dim=58, hidden_layers=(100, 100), output_dim=15):
        super().__init__()

        couches = []
        dim_precedente = input_dim
        for dim_cachee in hidden_layers:
            couches.append(nn.Linear(dim_precedente, dim_cachee))
            couches.append(nn.ReLU())
            dim_precedente = dim_cachee
        couches.append(nn.Linear(dim_precedente, output_dim))

        self.network = nn.Sequential(*couches)

    def forward(self, x):
        """
        Args:
            x: Tensor de shape (batch_size, input_dim)

        Returns:
            Tensor de shape (batch_size, output_dim) : logits, sans softmax.
        """
        return self.network(x)


def build_substitute_from_config(cfg):
    """
    Instancie le substitut depuis la configuration, sans valeur en dur.

    Args:
        cfg: objet Config issu de src.utils.config.load_config()

    Returns:
        SubstituteDNN
    """
    return SubstituteDNN(
        input_dim=cfg.substitute["input_dim"],
        hidden_layers=tuple(cfg.substitute["hidden_layers"]),
        output_dim=cfg.substitute["output_dim"],
    )


def count_parameters(model):
    """Compte les parametres entrainables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = SubstituteDNN()
    print(f"Architecture :\n{model}\n")
    print(f"Nombre de parametres : {count_parameters(model):,}")

    batch_test = torch.randn(256, 58)
    sortie = model(batch_test)
    print(f"\nInput shape  : {tuple(batch_test.shape)}")
    print(f"Output shape : {tuple(sortie.shape)}")