import torch
import torch.nn as nn
from typing import List, Dict, Any, Callable

class FlexibleCNNModel(nn.Module):
    """
    Una CNN la cui architettura è definita strato per strato
    tramite una lista di configurazione. Offre il massimo controllo.
    """

    def __init__(self, features: nn.Sequential, classifier: nn.Sequential):
        """
        Inizializza la rete.

        Args:
            cfg (dict): configurazione del modello
            num_classes (int): Numeri di classi.
            num_channels (Int): Numeri di canali dell'immagine in input.
            img_size (int): Dimensione dell'immagine di input.
        """
        super().__init__()

        self.features: nn.Sequential = features
        self.classifier: nn.Sequential = classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
