
"""
Architecture du DNN baseline.

Reproduction fidèle du modèle décrit dans Awad et al. (2025) :
  Input (58) -> Dense(512) + ReLU -> Dense(256) + ReLU -> Dense(15) + Softmax

Le Softmax est implicite : on utilise CrossEntropyLoss de PyTorch qui
applique LogSoftmax en interne. Ne pas ajouter de Softmax explicite dans
le modèle (double application = bug classique).
"""

import torch
import torch.nn as nn


class BaselineDNN(nn.Module):
    """
    Deep Neural Network baseline pour la classification d'intrusions.

    Architecture simple à deux couches cachées, comme dans le papier :
    - Couche 1 : 58 -> 512 avec activation ReLU
    - Couche 2 : 512 -> 256 avec activation ReLU
    - Couche 3 : 256 -> 15 (logits, pas de softmax explicite)

    Total de paramètres : environ 165 000, très léger.
    """

    def __init__(self, input_dim=58, hidden1=512, hidden2=256, output_dim=15):
        super().__init__()

        # Définition des couches en tant que Sequential pour la lisibilité
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, output_dim),
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Tensor de shape (batch_size, 58)

        Returns:
            Tensor de shape (batch_size, 15) contenant les logits.
            Note : pas de softmax, il est appliqué par CrossEntropyLoss.
        """
        return self.network(x)


def count_parameters(model):
    """Compte le nombre de paramètres entraînables du modèle."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test rapide du modèle
    model = BaselineDNN()
    print(f"Architecture :\n{model}\n")
    print(f"Nombre de paramètres : {count_parameters(model):,}")

    # Test avec un batch factice
    batch_test = torch.randn(128, 58)
    output = model(batch_test)
    print(f"\nInput shape  : {batch_test.shape}")
    print(f"Output shape : {output.shape}")