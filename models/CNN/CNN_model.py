import torch
import torch.nn as nn
from typing import List, Dict, Any

CONV_LAYER: str = "convolutional_layer"
FC: str = "fully_connected_layer"
LAYER_TYPE: str = "layer_type"
OUTPUT_CHANNELS: str = "output_channels"

# conv2d
KERNEL_SIZE: str = "kernel_size"
STRIDE: str = "stride"
PADDING: str = "padding"

# relu
INPLACE: str = "inplace"

DROPOUT: str = "dropout"

class FlexibleCNNModel(nn.Module):
    """
    Una CNN la cui architettura è definita strato per strato
    tramite una lista di configurazione. Offre il massimo controllo.
    """

    def __init__(self, cfg: dict, num_classes: int, num_channels: int, img_size: int):
        """
        Inizializza la rete.

        Args:
            cfg (dict): configurazione del modello
            num_classes (int): Numeri di classi.
            num_channels (Int): Numeri di canali dell'immagine in input.
            img_size (int): Dimensione dell'immagine di input.
        """
        super().__init__()

        self.cfg = cfg

        self.num_classes: int = num_classes
        self.num_channels: int = num_channels
        self.img_size: int = img_size

        self._final_img_size: int = img_size

        self.features: nn.Sequential = self._create_conv_layer(
            self.num_classes, self.cfg[CONV_LAYER]
        )

        self.classifier: nn.Sequential = self._create_fully_connected_layer(
            self.output_channel, self.cfg[FC]
        )

    def _create_conv_layer(
        self, current_channels: int, blocks: list[list[dict[str, Any]]]
    ) -> nn.Sequential:
        layers_list: list = []

        for block in blocks:
            for layer in block:
                layer_type: str = layer[LAYER_TYPE].lower()

                if layer_type == "conv_2d":
                    layer_constructor = nn.Conv2d(
                        in_channels=current_channels,
                        out_channels=layer[OUTPUT_CHANNELS],
                        kernel_size=layer.get(KERNEL_SIZE, 3),
                        stride=layer.get(STRIDE, 1),
                        padding=layer.get(PADDING, 1),
                    )

                    current_channels = layer[OUTPUT_CHANNELS]
                    self.output_channel = current_channels

                    layers_list.append(layer_constructor)

                elif layer_type == "batch_norm_2d":
                    layer_type == nn.BatchNorm2d(self.output_channel)

                    layers_list.append(layer_constructor)

                elif layer_type == "relu":
                    layer_constructor = nn.ReLU(inplace=layer.get(INPLACE, False))

                    layers_list.append(layer_constructor)

                elif layer_type == "max_pool_2d":
                    layer_constructor = nn.MaxPool2d(kernel_size=layer[KERNEL_SIZE])

                    self._final_img_size = self._final_img_size // layer[KERNEL_SIZE]

                    layers_list.append(layer_constructor)

                elif layer_type == "adaptive_avg_pool_2d":
                    layer_constructor = nn.AdaptiveAvgPool2d((1, 1))

                    self._final_img_size = 1

                    layers_list.append(layer_constructor)

        print(layers_list)

        return nn.Sequential(*layers_list)

    def _create_fully_connected_layer(
        self, current_channels: int, blocks: list[list[dict[str, Any]]]
    ) -> nn.Sequential:
        layers_list: list = []

        for block in blocks:
            for layer in block:
                layer_type: str = layer[LAYER_TYPE].lower()

                if layer_type == "flatten":
                    layer_constructor = nn.Flatten()

                    layers_list.append(layer_constructor)

                elif layer_type == "linear":
                    layer_constructor = nn.Linear(
                        current_channels * self._final_img_size * self._final_img_size,
                        layer.get(OUTPUT_CHANNELS, self.num_classes),
                    )

                    current_channels = layer.get(OUTPUT_CHANNELS, self.num_classes)

                    if self._final_img_size != 1:
                        self._final_img_size = 1

                    layers_list.append(layer_constructor)

                elif layer_type == "relu":
                    layer_constructor = nn.ReLU(inplace=layer.get(INPLACE, False))

                    layers_list.append(layer_constructor)

                elif layer_type == "drouput":
                    layer_constructor = nn.Dropout(layer.get(DROPOUT, 0.5))

                    layers_list.append(layer_constructor)

        print(layers_list)

        return nn.Sequential(*layers_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
