import torch
import torch.nn as nn
from dataloader import CNNDataLoader as DataLoader

from .CNN_engine import CNNEngine

from pathlib import Path

from models import get_model

from utils import ColoredPrint, ConfigKeys

from torch.utils.data import DataLoader as DL

# ------------------------------------------------------------------
# PARAMETRI DA PERSONALIZZARE
# ------------------------------------------------------------------
GENERAL_SETTINGS: str = "general_settings"
DEVICE: str = "device"
CUDA: str = "cuda"
CPU: str = "cpu"

DATA_SETTINGS: str = "data_settings"
DIRECTORY: str = "directory"
GENERAL_DIR: str = "general_dir"
DATASET_DIR: str = "dataset_dir"

DATASET: str = "dataset"
TRAIN: str = "train"
TRAIN_DIR: str = "train_dir_aug"
VAL: str = "val"
VAL_DIR: str = "val_dir"
TEST: str = "test"
TEST_DIR: str = "test_dir"

IMG_SIZE: str = "img_size"
BATCH_SIZE: str = "batch_size"
NUM_WORKERS: str = "num_workers"
NUM_CHANNELS: str = "num_channels"

MODEL: str = "model"
BACKBONE: str = "backbone"
PRETRAINED: str = "pretrained"
PATH: str = "path"
NAME: str = "name"

EPOCHS: str = "epochs"
OPTIMIZER: str = "optimizer"
LR: str = "lr"

CHECKPOINT: str = "checkpoint"
CHECKPOINT_DIR: str = "dir"

MODEL_CONFIG: str = "model_config"
CLASS_TO_IDX: str = "class_to_idx"
NUM_PARAMS: str = "num_params"
MODEL_NAME: str = "model_name"

MODEL_STATE: str = "model_state"


# ------------------------------------------------------------------
# CLASSE DI INFERENCE CNN
# ------------------------------------------------------------------
class TesterCNN:
    def __init__(self, cfg: dict):
        # config
        self.cfg: dict = cfg

        # cuda
        if self.cfg[GENERAL_SETTINGS][DEVICE] == CUDA and torch.cuda.is_available():
            self.device: str = CUDA
        else:
            self.device: str = CPU

        # data loader
        self.dataloader = DataLoader(self.cfg).createDatasetTest()
        self.test_dl: DL = self.dataloader

        # modello salvato
        self.checkpoint_model_name = (
            Path(self.cfg[MODEL][PRETRAINED][PATH])
            / self.cfg[MODEL][BACKBONE]
            / self.cfg[MODEL][PRETRAINED][NAME]
        )
        self.checkpoint_model = torch.load(
            self.checkpoint_model_name, map_location=self.device
        )

        # num classes
        self.num_classes: int = len(self.checkpoint_model[MODEL_CONFIG][CLASS_TO_IDX])

        # model name, channels, img_size and num_params
        # self.model_name: str = self.checkpoint_model[MODEL_CONFIG][MODEL_NAME] if self.checkpoint_model[MODEL_CONFIG][MODEL_NAME] else self.cfg[MODEL][BACKBONE]
        self.model_name: str = self.cfg[MODEL][BACKBONE]
        self.num_channels: int = self.checkpoint_model[MODEL_CONFIG][NUM_CHANNELS]
        self.image_size: int = self.checkpoint_model[MODEL_CONFIG][IMG_SIZE]
        self.num_params: int = self.checkpoint_model[MODEL_CONFIG][NUM_PARAMS]

        # modello CNN
        self.model: nn.Module = get_model(
            name=self.model_name,
            num_classes=self.num_classes,
            num_channels=self.num_channels,
            img_size=self.image_size,
        )

        # load weight
        self.model.load_state_dict(self.checkpoint_model[MODEL_STATE])
        self.model.to(self.device).eval()

        # loss function
        self.loss = nn.CrossEntropyLoss()

        # Best accuracy
        self.best_acc: int = 0

        # epoche
        self.epochs: int = self.cfg[TRAIN][EPOCHS]

        # training engine
        self.training_engine = CNNEngine(self.model, self.loss, None, self.device)

        # Save model
        self.checkpoint = self.cfg[CHECKPOINT][CHECKPOINT_DIR]

    def inference(self):
        ColoredPrint.green("\nINIZIO LOOP DI INFERENZA\n" + "-" * 20)

        test_loss, test_accuracy = self.training_engine.exec_epoch(self.test_dl, TEST)

        # Mostra loss e accuracy
        print(f"Test loss: {test_loss:.3f} | Test acc: {test_accuracy:.2f}%")

        ColoredPrint.green("\nFINE LOOP DI INFERENZA\n" + "-" * 20)
