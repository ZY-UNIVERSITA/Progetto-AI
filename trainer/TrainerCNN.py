import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import CNNDataLoader

from .CNN_engine import CNNEngine

from datetime import datetime

from pathlib import Path

from models import get_model

from utils import ColoredPrint

import time

from utils import EarlyStopping

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
MODEL_NAME: str = "backbone"

EPOCHS: str = "epochs"
OPTIMIZER: str = "optimizer"
LR: str = "lr"

CHECKPOINT: str = "checkpoint"
CHECKPOINT_DIR: str = "dir"

# ------------------------------------------------------------------
# CLASSE DI TRAINER CNN
# ------------------------------------------------------------------
class TrainerCNN:
    def __init__(self, cfg: dict):
        # config
        self.cfg: dict = cfg

        # cuda
        if self.cfg[GENERAL_SETTINGS][DEVICE] == CUDA and torch.cuda.is_available():
            self.device: str = CUDA
        else:
            self.device: str = CPU

        # data loader
        self.dataloader: CNNDataLoader = CNNDataLoader(self.cfg).createDatasetTraining()
        self.train_dl: DataLoader = self.dataloader[0]
        self.val_dl: DataLoader = self.dataloader[1]
        self.class_to_idx: dict[str, int] = self.dataloader[2]

        # num classes
        self.num_classes: int = len(self.class_to_idx)

        # model name, channels, img_size and 
        self.model_name: str = self.cfg[MODEL][MODEL_NAME]
        self.num_channels: int = self.cfg[DATA_SETTINGS][NUM_CHANNELS]
        self.image_size: int = self.cfg[DATA_SETTINGS][IMG_SIZE]

        # get model
        self.model: nn.Module = get_model(
            name=self.model_name,
            num_classes=self.num_classes,
            num_channels=self.num_channels,
            img_size=self.image_size
        ).to(self.device)

        # training parameters
        # optimizer
        self.lr = self.cfg[TRAIN][OPTIMIZER][LR]
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # loss function
        self.loss = nn.CrossEntropyLoss()

        # Best accuracy
        self.best_acc: int = 0

        # epoche
        self.epochs: int = self.cfg[TRAIN][EPOCHS]

        # training engine
        self.training_engine = CNNEngine(self.model, self.loss, self.optimizer, self.device)

        # Save model
        self.checkpoint = self.cfg[CHECKPOINT][CHECKPOINT_DIR]


        self.model_saved: bool = False
        self.early_stopping: EarlyStopping = EarlyStopping(self.cfg, self.save_model)


    def loggingInfo(self):
        ColoredPrint.green("\nINIZIO LOGGING TRAINING\n" + "-" * 20)

        ColoredPrint.purple(f"Device: {self.device}")

        print(f"Numero di classi trovate: {self.num_classes}")
        if self.num_classes < 2:
            print(
                "Il numero di classi è inferiore a 2: c'è una sola classe. La classificazione multiclasse richiede almeno 2 classi."
            )

        print(f"Il modello scelto è: {self.model_name}") 
        print(f"Il numero di canali è: {self.num_channels}") 
        print(f"La grandezza dell'immagine è: {self.image_size}x{self.image_size}") 

        ColoredPrint.green("\nFINE LOGGING TRAINING\n" + "-" * 20)

    def train(self) -> None:
        ColoredPrint.green("\nINIZIO LOOP DI TRAINING\n" + "-" * 20)

        ColoredPrint.green(f"Il training è stato impostato per {self.epochs} epoche...")

        for epoch in range(self.epochs):
            print(f"\nEpoca {epoch+1} di {self.epochs}")

            # Tempo di inizio epoca
            start_time: float = time.time()
            
            # Fai il training e la validazione
            train_loss, train_accuracy = self.training_engine.exec_epoch(self.train_dl, TRAIN)
            val_loss, val_accuracy = self.training_engine.exec_epoch(self.val_dl, VAL)

            # Mostra loss e accuracy
            print(f"Train loss: {train_loss:.3f} | Train acc: {train_accuracy:.2f}%")
            print(f"Val loss: {val_loss:.3f} | Val acc: {val_accuracy:.2f}%")

            # salva il modello se l'accuratezza del dataset di validazione è migliorata
            if val_accuracy > self.best_acc:
                self.model_saved = True
                self.best_acc = val_accuracy
                self.save_model(val_accuracy, val_loss, epoch)
                
            # Valuta se stoppare l'addestramento
            self.early_stopping.calculateWhenStop(epoch, val_accuracy, val_loss, self.model_saved)

            
            self.model_saved = False

            # Tempo di fine epoca
            end_time: float = time.time() 
            epoch_duration = end_time - start_time
            ColoredPrint.blue(f"\nL'addestramento è durato {epoch_duration}.")

            # calcola l'early stop in base all'accuratezza/loss
            if self.early_stopping.stop:
                break
            
        ColoredPrint.green("\nFINE LOOP DI TRAINING\n" + "-" * 20)
        
    def count_parameters(self):
        return sum(params.numel() for params in self.model.parameters())
    
    def save_model(self, val_accuracy: float, val_loss: float, epoch: int) -> None:
        best_model = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "metrics": {
                "accuracy": val_accuracy,
                "loss": val_loss,
            },
            "model_config": {
                "model_name": self.model_name,
                "class_to_idx": self.class_to_idx,
                "img_size": self.image_size,
                "num_channels": self.num_channels,
                "num_params": self.count_parameters()
            }
        }

        checkpoint_path = Path(self.checkpoint) / self.model_name

        checkpoint_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        file_name = f"{self.model_name}_{timestamp}.pth"

        torch.save(best_model, checkpoint_path / file_name)

        ColoredPrint.blue(f"\nIl modello: {file_name} è stato salvato.\n")





            



