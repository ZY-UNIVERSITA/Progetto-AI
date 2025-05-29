import os
from torchvision.transforms import v2 as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch
from typing import Tuple

# ------------------------------------------------------------------
# PARAMETRI DA PERSONALIZZARE
# ------------------------------------------------------------------
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


EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ------------------------------------------------------------------
# CLASSE DI DATA LOADER
# ------------------------------------------------------------------
class CNNDataLoader:
    def __init__(self, cfg: dict) -> None:
        self.cfg: dict = cfg

        self.general_dir: str = cfg[DATA_SETTINGS][DIRECTORY][GENERAL_DIR]

        self.dataset: str = os.path.join(
            self.general_dir, cfg[DATA_SETTINGS][DIRECTORY][DATASET_DIR]
        )

        self.train_dir: str = os.path.join(
            self.dataset, cfg[DATA_SETTINGS][DATASET][TRAIN_DIR]
        )

        self.val_dir: str = os.path.join(
            self.dataset, cfg[DATA_SETTINGS][DATASET][VAL_DIR]
        )

        self.test_dir: str = os.path.join(
            self.dataset, cfg[DATA_SETTINGS][DATASET][TEST_DIR]
        )

        self.image_size: int = cfg[DATA_SETTINGS][IMG_SIZE]
        self.batch_size: int = cfg[DATA_SETTINGS][BATCH_SIZE]
        self.num_workers: int = cfg[DATA_SETTINGS][NUM_WORKERS]
        self.num_channels: int = cfg[DATA_SETTINGS][NUM_CHANNELS]

        self.transform_list: list = []
        self.transform()
        self.compose = transforms.Compose(self.transform_list)

    # Trasformazioni
    def transform(self) -> None:
        # Resize
        resize = transforms.Resize((self.image_size, self.image_size))
        self.transform_list.append(resize)

        # Trasforma in scala di grigi
        if self.num_channels == 1:
            transforms.Grayscale(num_output_channels=self.num_channels)

        # Trasforma in un tensore di pytorch
        tensor = transforms.ToImage()
        self.transform_list.append(tensor)
        tensor_float = transforms.ToDtype(torch.float32, scale=True)
        self.transform_list.append(tensor_float)

        # Normalizza i dati
        normalize = transforms.Normalize(
            [0.5] * self.num_channels, [0.5] * self.num_channels
        )
        self.transform_list.append(normalize)

    def createDataset(
        self,
    ) -> Tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
        # Carica i dati
        train_ds = datasets.ImageFolder(self.train_dir, self.compose)
        val_ds = datasets.ImageFolder(self.val_dir, self.compose)
        test_ds = datasets.ImageFolder(self.test_dir, self.compose)

        # Crea il data loader
        train_dl = DataLoader(
            train_ds, self.batch_size, shuffle=True, num_workers=self.num_workers
        )
        val_dl = DataLoader(
            val_ds, self.batch_size, shuffle=False, num_workers=self.num_workers
        )
        test_dl = DataLoader(
            test_ds, self.batch_size, shuffle=False, num_workers=self.num_workers
        )

        return train_dl, val_dl, test_dl, train_ds.class_to_idx
