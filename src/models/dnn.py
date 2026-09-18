"""Modeles du papier : le detecteur (Table 3) et le substitut."""

import torch.nn as nn


class DNN(nn.Module):
    """
    Detecteur IDS. Table 3 : 58 -> 512 -> 256 -> 15, ReLU sur les couches
    cachees, SoftMax en sortie.

    Pas de SoftMax explicite : CrossEntropyLoss l'applique en interne. En
    ajouter un serait une double application.
    """

    def __init__(self, input_dim=58, hidden=(512, 256), output_dim=15):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class Substitut(nn.Module):
    """
    Le modele de l'attaquant. Le papier : "a fully connected feed-forward
    neural network classifier with an input layer of dimension 58 [...]
    followed by two layers of 100 neurons each".

    Il sert a generer les attaques, qui sont ensuite transferees sur le
    detecteur. C'est ce qui place l'attaquant en semi-white box : il connait
    l'architecture generale mais pas les poids.
    """

    def __init__(self, input_dim=58, hidden=(100, 100), output_dim=15):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, output_dim),
        )

    def forward(self, x):
        return self.net(x)
