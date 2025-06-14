import torch
import torch.optim as optim
from torch.optim import (
    AdamW, Adam
)

from typing import Tuple

class Optimizer():
    def __init__(self, name: str, parameters, **kwargs) -> None:
        self.name = name.lower()
        self.parameters = parameters
        self.kwargs = kwargs

    def get_optimizer(self) -> torch.optim.Optimizer:
        if self.name == "none":
            return None
        elif self.name == "adam":
            return Adam(self.parameters, **self.kwargs)
        else:
            raise ValueError(f"Unsupported scheduler name: '{self.name}'")