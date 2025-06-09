import torch
import torch.nn as nn
from typing import List, Dict, Any, Callable

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

        self.input_channels: int = 0
        self.output_channel: int = 0

        self.features: nn.Sequential = self.create_layers(
            self.cfg[CONV_LAYER], self.num_channels
        )

        self.classifier: nn.Sequential = self.create_layers(
            self.cfg[FC], self.output_channel
        )

    def create_layers(
        self, blocks: list[list[dict[str, Any]]], num_channels: int
    ) -> nn.Sequential:
        self.current_channels = num_channels

        layers_list: list = []

        for block in blocks:
            for layer in block:
                layer_type: str = layer[LAYER_TYPE].lower()

                # restituisce la funzione che creerà il layer
                layer_constructor_fun = LAYERS.get(layer_type)

                # creare il layer
                layer_constructor = layer_constructor_fun(self)(layer)

                layers_list.append(layer_constructor)

        print(layers_list)

        return nn.Sequential(*layers_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

    def conv_2d(self, layer: dict[str, any]) -> nn.Sequential:
        layer_constructor = nn.Conv2d(
            in_channels=self.current_channels,
            out_channels=layer[OUTPUT_CHANNELS],
            kernel_size=layer.get(KERNEL_SIZE, 3),
            stride=layer.get(STRIDE, 1),
            padding=layer.get(PADDING, 1),
        )

        self.current_channels = layer[OUTPUT_CHANNELS]
        self.output_channel = self.current_channels

        return layer_constructor

    def batch_norm_2d(self, layer: dict[str, any]) -> nn.Sequential:
        layer_constructor = nn.BatchNorm2d(self.output_channel)

        return layer_constructor

    def relu(self, layer: dict[str, any]) -> nn.Sequential:
        layer_constructor = nn.ReLU(inplace=layer.get(INPLACE, False))

        return layer_constructor

    def max_pool_2d(self, layer: dict[str, any]) -> nn.Sequential:
        layer_constructor = nn.MaxPool2d(kernel_size=layer[KERNEL_SIZE])

        self._final_img_size = self._final_img_size // layer[KERNEL_SIZE]

        return layer_constructor

    def adaptive_avg_pool_2d(self, layer: dict[str, any]) -> nn.Sequential:
        layer_constructor = nn.AdaptiveAvgPool2d((1, 1))

        self._final_img_size = 1

        return layer_constructor

    def flatten(self, layer: dict[str, any]) -> nn.Sequential:
        layer_constructor = nn.Flatten()

        return layer_constructor

    def linear(self, layer: dict[str, any]) -> nn.Sequential:
        layer_constructor = nn.Linear(
            self.current_channels * self._final_img_size * self._final_img_size,
            layer.get(OUTPUT_CHANNELS, self.num_classes),
        )

        self.current_channels = layer.get(OUTPUT_CHANNELS, self.num_classes)

        if self._final_img_size != 1:
            self._final_img_size = 1

        return layer_constructor

    def dropout(self, layer: dict[str, any]) -> nn.Sequential:
        layer_constructor = nn.Dropout(layer.get(DROPOUT, 0.5))

        return layer_constructor


LayerFunction = Callable[[FlexibleCNNModel, Dict[str, Any]], Any]

LAYERS: Dict[str, LayerFunction] = {
    "conv_2d": lambda instance: instance.conv_2d,
    "batch_norm_2d": lambda instance: instance.batch_norm_2d,
    "relu": lambda instance: instance.relu,
    "max_pool_2d": lambda instance: instance.max_pool_2d,
    "adaptive_avg_pool_2d": lambda instance: instance.adaptive_avg_pool_2d,
    "flatten": lambda instance: instance.flatten,
    "linear": lambda instance: instance.linear,
    "dropout": lambda instance: instance.dropout,
}
