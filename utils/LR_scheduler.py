import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, ExponentialLR, LambdaLR,
    CosineAnnealingLR, CosineAnnealingWarmRestarts,
    ReduceLROnPlateau, CyclicLR, OneCycleLR
)

from typing import Tuple


class LR_scheduler():
    def __init__(self, name: str, optim: torch.optim.Optimizer, **kwargs) -> None:
        self.name = name.lower()
        self.optim = optim
        self.kwargs = kwargs

    def get_scheduler(self) -> Tuple[torch.optim.lr_scheduler.LRScheduler, str]:
        if self.name == "none":
            return None, "none"
        elif self.name == "steplr":
            return torch.optim.lr_scheduler.StepLR(self.optim, **self.kwargs), "epoch"
        else:
            raise ValueError(f"Unsupported scheduler name: '{self.name}'")