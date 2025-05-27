"""
Modulo per una rete neurale convoluzionale semplice per la classificazione di immagini.

Questo modulo definisce una CNN compatta con un singolo blocco convoluzionale seguito
da un classificatore fully-connected.
"""

import torch
import torch.nn as nn

class SmallCNN(nn.Module):
    """
    Rete neurale convoluzionale semplice per la classificazione di immagini.

    Architettura della rete:
        - Strato convoluzionale: 1 blocco di Conv2D + ReLU seguito da MaxPool2D
        - Strato denso: Flatten + singolo strato nascosto con ReLU + strato di output

    La rete è progettata per processare immagini di dimensione fissa e produrre
    predizioni per un numero specificato di classi.

    Attributes:
        features (nn.Sequential): Blocco di estrazione delle feature convoluzionali
        classifier (nn.Sequential): Blocco di classificazione fully-connected

    Args:
        num_classes (int): Numero di classi di output per la classificazione
        num_channels (int): Numero di canali dell'immagine di input (es. 3 per RGB, 1 per grayscale)
        img_size (int): Dimensione dell'immagine quadrata di input (altezza = larghezza)

    Example:
        >>> model = SmallCNN(num_classes=10, num_channels=3, img_size=64)
        >>> input_tensor = torch.randn(32, 3, 64, 64)  # batch di 32 immagini RGB 64x64
        >>> output = model(input_tensor)
        >>> print(output.shape)  # torch.Size([32, 10])
    """

    def __init__(self, num_classes: int, num_channels: int, img_size: int):
        """
        Inizializza la rete SmallCNN.

        Args:
            num_classes (int): Numero di classi per la classificazione
            num_channels (int): Numero di canali dell'immagine di input
            img_size (int): Dimensione laterale dell'immagine quadrata di input
        """
        super().__init__()

        print(f"Numero di classi: {num_classes}")
        print(f"Numero di canali: {num_channels}")
        print(f"Dimensione dell'Immagine: {img_size}x{img_size}")

        self._num_pooling: int = 1
        self._final_img_size: int = img_size // pow(2, self._num_pooling)

        self.features: nn.Sequential = nn.Sequential(
            # (channel, height, width)
            self.convolutional(num_channels, 32),
            nn.MaxPool2d(kernel_size=2),
            # immagine input 64x64 -> finale da 32x32
        )

        self.classifier: nn.Sequential = nn.Sequential(
            nn.Flatten(),
            # dimensione rispetto al numero di feature_map * height * width in input
            # 256 neuroni in 1 singola hidden layer
            nn.Linear(32 * self._final_img_size * self._final_img_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Esegue il forward pass attraverso la rete.

        Args:
            x (torch.Tensor): Tensore di input con shape (batch_size, num_channels, height, width)

        Returns:
            torch.Tensor: Logits di output con shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x

    def convolutional(
        self,
        input_channel: int,
        output_feature_map: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> nn.Sequential:
        """
        Crea un blocco convoluzionale composto da Conv2D + ReLU.

        Args:
            input_channel (int): Numero di canali in input
            output_feature_map (int): Numero di feature map in output (numero di filtri)
            kernel_size (int, optional): Dimensione del kernel convoluzionale. Default: 3
            stride (int, optional): Passo di scorrimento del kernel. Default: 1
            padding (int, optional): Padding di 0 applicato ai bordi dell'input. Default: 1

        Returns:
            nn.Sequential: Blocco sequenziale contenente Conv2D e ReLU

        Note:
            Con kernel_size=3, stride=1 e padding=1, le dimensioni spaziali
            dell'output rimangono invariate rispetto all'input.
        """
        return nn.Sequential(
            nn.Conv2d(
                in_channels=input_channel,
                out_channels=output_feature_map,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.ReLU(),
        )
