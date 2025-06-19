import torch
import torch.nn as nn
from dataloader import CNNDataLoader as DataLoader

from .CNN_engine import CNNEngine

from pathlib import Path

from models import get_model

from utils import ColoredPrint, ConfigKeys, save_test_info as sti

from torch.utils.data import DataLoader as DL

import json

# ------------------------------------------------------------------
# CLASSE DI INFERENCE CNN
# ------------------------------------------------------------------
class TesterCNN:
    def __init__(self, cfg: dict):
        # config
        self.cfg: dict = cfg

        # cuda
        if self.cfg[ConfigKeys.GENERAL_SETTINGS][ConfigKeys.DEVICE] == ConfigKeys.CUDA and torch.cuda.is_available():
            self.device: str = ConfigKeys.CUDA
        else:
            self.device: str = ConfigKeys.CPU

        # data loader
        self.dataloader = DataLoader(self.cfg).createDatasetTest()
        self.test_dl: DL = self.dataloader

        # modello salvato
        self.checkpoint_model_name = (
            Path(self.cfg[ConfigKeys.MODEL][ConfigKeys.PRETRAINED][ConfigKeys.PATH])
            / self.cfg[ConfigKeys.MODEL][ConfigKeys.PRETRAINED][ConfigKeys.FOLDER]
            / ConfigKeys.MODEL
            / self.cfg[ConfigKeys.MODEL][ConfigKeys.PRETRAINED][ConfigKeys.NAME]
        )
        self.checkpoint_model = torch.load(
            self.checkpoint_model_name, map_location=self.device
        )

        # num classes
        self.classes: dict[str, int] = self.checkpoint_model[ConfigKeys.MODEL_CONFIG][ConfigKeys.CLASS_TO_IDX]
        self.num_classes: int = len(self.classes)

        # model name, channels, img_size and num_params
        self.model_name: str = self.cfg[ConfigKeys.MODEL][ConfigKeys.BACKBONE]
        self.num_channels: int = self.checkpoint_model[ConfigKeys.MODEL_CONFIG][ConfigKeys.NUM_CHANNELS]
        self.image_size: int = self.checkpoint_model[ConfigKeys.MODEL_CONFIG][ConfigKeys.IMG_SIZE]
        self.num_params: int = self.checkpoint_model[ConfigKeys.MODEL_CONFIG][ConfigKeys.NUM_PARAMS]
        self.model_config = self.checkpoint_model[ConfigKeys.CUSTOM_MODEL]

        # modello CNN
        self.model: nn.Module = get_model(
            name=self.model_name,
            num_classes=self.num_classes,
            num_channels=self.num_channels,
            img_size=self.image_size,
            model_cfg=self.model_config,
        )

        # load weight
        self.model.load_state_dict(self.checkpoint_model[ConfigKeys.MODEL_STATE])
        self.model.to(self.device).eval()

        # loss function
        self.loss = nn.CrossEntropyLoss()

        # Best accuracy
        self.best_acc: int = 0

        # epoche
        self.epochs: int = self.cfg[ConfigKeys.TRAIN][ConfigKeys.EPOCHS]

        # training engine
        self.training_engine = CNNEngine(
            self.model, self.loss, None, None, None, self.device, self.classes
        )

        # Save model
        self.checkpoint = self.cfg[ConfigKeys.CHECKPOINT][ConfigKeys.CHECKPOINT_DIR]

    def inference(self):
        ColoredPrint.green("\nINIZIO LOOP DI INFERENZA\n" + "-" * 20)

        test_loss, test_accuracy = self.training_engine.exec_epoch(self.test_dl, ConfigKeys.TEST)

        # Mostra loss e accuracy
        print(f"Test loss: {test_loss:.3f} | Test acc: {test_accuracy:.2f}%")

        self.save_test_info(test_loss, test_accuracy)

        ColoredPrint.green("\nFINE LOOP DI INFERENZA\n" + "-" * 20)

    def save_test_info(self, loss: float, accuracy: float) -> None:
        path: Path = (
            Path(self.cfg[ConfigKeys.MODEL][ConfigKeys.PRETRAINED][ConfigKeys.PATH])
            / self.cfg[ConfigKeys.MODEL][ConfigKeys.PRETRAINED][ConfigKeys.FOLDER]
            / "general_info_training.json"
        )

        sti(path, loss, accuracy)

