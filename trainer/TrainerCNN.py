# Import standard library modules
import json
import os
import time
import datetime
from pathlib import Path
import shutil

# Import third-party libraries
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

# Import local modules
from dataloader import CNNDataLoader
from .CNN_engine import CNNEngine
from models import get_model
from utils import EarlyStopping, ConfigKeys, LR_scheduler, Optimizer, ColoredPrint

import matplotlib

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

CHECKPOINT: str = "checkpoint"
CHECKPOINT_DIR: str = "dir"

matplotlib.rcParams["font.family"] = [
    "Noto Sans CJK JP",
    "DejaVu Sans",  # fallback generico
]


# ------------------------------------------------------------------
# CLASSE DI TRAINER CNN
# ------------------------------------------------------------------
class TrainerCNN:
    def __init__(
        self, cfg: dict, model_config: dict = None, restart_training: bool = False
    ):
        # config
        self.cfg: dict = cfg
        self.model_config: dict = model_config

        # cuda
        if (
            self.cfg[ConfigKeys.GENERAL_SETTINGS][ConfigKeys.DEVICE] == ConfigKeys.CUDA
            and torch.cuda.is_available()
        ):
            self.device: str = ConfigKeys.CUDA
        else:
            self.device: str = ConfigKeys.CPU

        # data loader
        self.dataloader: CNNDataLoader = CNNDataLoader(self.cfg)
        dataset = self.dataloader.createDatasetTraining()
        self.train_dl: DataLoader = dataset[0]
        self.val_dl: DataLoader = dataset[1]
        self.class_to_idx: dict[str, int] = dataset[2]

        # num classes
        self.num_classes: int = len(self.class_to_idx)

        # model name, channels, img_size and
        self.model_name: str = self.cfg[ConfigKeys.MODEL][ConfigKeys.MODEL_NAME]
        self.num_channels: int = self.cfg[ConfigKeys.DATA_SETTINGS][
            ConfigKeys.NUM_CHANNELS
        ]
        self.image_size: int = self.cfg[ConfigKeys.DATA_SETTINGS][ConfigKeys.IMG_SIZE]
        self.batch_size: int = self.cfg[ConfigKeys.DATA_SETTINGS][ConfigKeys.BATCH_SIZE]

        # get model
        # get model from previous training
        if restart_training:
            checkpoint_model_name = (
                Path(self.cfg[ConfigKeys.MODEL]["pretrained"]["path"])
                / self.cfg[ConfigKeys.MODEL]["backbone"]
                / self.cfg[ConfigKeys.MODEL]["pretrained"]["name"]
            )
            checkpoint = torch.load(checkpoint_model_name, map_location=self.device)

            if self.model_name == "custom":
                self.model_config = checkpoint["custom_model"]

            self.model: nn.Module = get_model(
                name=self.model_name,
                num_classes=self.num_classes,
                num_channels=self.num_channels,
                img_size=self.image_size,
                model_cfg=self.model_config,
            )

            self.model.load_state_dict(checkpoint["model_state"])
            self.model.to(self.device)

        # get new model to train
        else:
            self.model: nn.Module = get_model(
                name=self.model_name,
                num_classes=self.num_classes,
                num_channels=self.num_channels,
                img_size=self.image_size,
                model_cfg=self.model_config,
            ).to(self.device)

        # training parameters
        # optimizer
        self.optimizer_kwargs = self.cfg[ConfigKeys.TRAIN][ConfigKeys.OPTIMIZER][
            ConfigKeys.OPTIMIZER_ARGS
        ]
        self.lr = self.optimizer_kwargs[ConfigKeys.LR]
        self.optimizer_name = self.cfg[ConfigKeys.TRAIN][ConfigKeys.OPTIMIZER][
            ConfigKeys.OPTIMIZER_TYPE
        ]
        self.optimizer_class: Optimizer = Optimizer(
            self.optimizer_name, self.model.parameters(), **self.optimizer_kwargs
        )
        self.optimizer = self.optimizer_class.get_optimizer()

        # restart training with old optmizer
        if restart_training:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        # lr scheduler
        # get previous training optimizer
        if restart_training:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])

        # start new optimizer
        else:
            self.scheduler_name: str = self.cfg[ConfigKeys.TRAIN][ConfigKeys.SCHEDULER][
                ConfigKeys.SCHEDULER_TYPE
            ]
            self.scheduler_kwargs: dict[str, any] = self.cfg[ConfigKeys.TRAIN][
                ConfigKeys.SCHEDULER
            ][ConfigKeys.SCHEDULER_ARGS]

            # lr warmup scheduler
            self.normal_scheduler: bool = not self.cfg[ConfigKeys.TRAIN][
                ConfigKeys.OPTIMIZER
            ][ConfigKeys.WARMUP][ConfigKeys.USE_WARMUP]

            if not self.normal_scheduler:
                self.warmup_scheduler: int = self.cfg[ConfigKeys.TRAIN][
                    ConfigKeys.OPTIMIZER
                ]["warmup"]
                self.warmup_scheduler_epochs: int = self.warmup_scheduler["epochs"]
                self.warmup_scheduler_lr: int = self.warmup_scheduler["lr"]
                self.warmup_scheduler_step_size: int = self.warmup_scheduler[
                    "step_size"
                ]
                self.warmup_scheduler_gamma: float = (
                    self.warmup_scheduler_lr / self.lr
                ) ** (1 / self.warmup_scheduler_epochs)

                self.scheduler_warmup_kwargs = {
                    "step_size": self.warmup_scheduler_step_size,
                    "gamma": self.warmup_scheduler_gamma,
                }
                self.scheduler_class: LR_scheduler = LR_scheduler(
                    self.scheduler_name, self.optimizer, **self.scheduler_warmup_kwargs
                )
            else:
                self.scheduler_class: LR_scheduler = LR_scheduler(
                    self.scheduler_name, self.optimizer, **self.scheduler_kwargs
                )

            self.scheduler, self.scheduler_type = self.scheduler_class.get_scheduler()

        # loss function
        self.loss = nn.CrossEntropyLoss()

        # Best accuracy e loss
        if restart_training:
            self.best_acc: float = checkpoint["metrics"]["accuracy"]
            self.best_loss: float = checkpoint["metrics"]["loss"]
        else:
            self.best_acc: float = float("-inf")
            self.best_loss: float = float("inf")

        # epoche
        if restart_training:
            self.start_epoch: int = checkpoint["epoch"]
        else:
            self.start_epoch: int = 0

        self.epochs: int = self.cfg[ConfigKeys.TRAIN][ConfigKeys.EPOCHS]
        self.last_epoch: int = 0

        # training engine
        self.training_engine = CNNEngine(
            self.model,
            self.loss,
            self.optimizer,
            self.scheduler,
            self.scheduler_type,
            self.device,
            self.class_to_idx,
        )

        self.model_saved: bool = False
        self.early_stopping: EarlyStopping = EarlyStopping(self.cfg, self.save_model)

        # saving options
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        checkpoint_main_dir = self.cfg[ConfigKeys.CHECKPOINT][ConfigKeys.CHECKPOINT_DIR]
        model_numeration = f"{self.model_name}_{timestamp}"
        self.main_save_path = Path(checkpoint_main_dir) / model_numeration
        self.main_save_path.mkdir(parents=True, exist_ok=True)

        self.checkpoint_path = self.main_save_path / ConfigKeys.MODEL.value
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)

        self.tensorboard_path = self.main_save_path / "log"

        self.save_model_config()

        # Tensorboard
        self.writer = SummaryWriter(self.tensorboard_path)

        # batch di immagini di prova
        images_batch, _ = next(iter(self.train_dl))
        img_grid = make_grid(images_batch, normalize=True)
        self.writer.add_image("Immagini di training", img_grid)

        # grafo del modello
        self.writer.add_graph(self.model, images_batch)

        # writer
        self.training_engine.add_writer(self.writer)

    def loggingInfo(self):
        ColoredPrint.blue("\nINIZIO LOGGING TRAINING\n" + "-" * 20)

        ColoredPrint.purple(
            f"Il training viene eseguto sul device: {self.device.upper()}."
        )

        ColoredPrint.purple(f"Il modello scelto è: {self.model_name}.")

        ColoredPrint.purple(
            f"Lo scheduler usato è: {self.scheduler_name} con i seguenti parametri {self.scheduler_kwargs}."
        )

        ColoredPrint.purple(
            f"La grandezza dell'immagine è: {self.image_size}x{self.image_size}."
        )

        ColoredPrint.purple(f"Numero di classi trovate: {self.num_classes}.")
        if self.num_classes < 2:
            ColoredPrint.purple(
                "Il numero di classi è inferiore a 2: c'è una sola classe. La classificazione multiclasse richiede almeno 2 classi."
            )

        ColoredPrint.purple(f"Il numero di canali è: {self.num_channels}.")

        ColoredPrint.purple(f"Il modello verrà addestrato su: {self.epochs} epoche.")

        ColoredPrint.blue("\nFINE LOGGING TRAINING\n" + "-" * 20)

    def train(self) -> None:
        ColoredPrint.purple("\nINIZIO LOOP DI TRAINING\n" + "-" * 20)

        ColoredPrint.purple(
            f"Il training è stato impostato per {self.epochs} epoche..."
        )

        for epoch in range(self.start_epoch, self.epochs):
            ColoredPrint.purple(f"\nEpoca {epoch+1} di {self.epochs}")

            # Tempo di inizio epoca
            start_time: float = time.time()

            ColoredPrint.cyan("\nINIZIO ELABORAZIONE DEL MODELLO\n")

            # Fai il training e la validazione
            train_loss, train_accuracy = self.training_engine.exec_epoch(
                self.train_dl, TRAIN
            )
            val_loss, val_accuracy = self.training_engine.exec_epoch(self.val_dl, VAL)

            ColoredPrint.cyan("\nFINE ELABORAZIONE DEL MODELLO\n")

            if not self.normal_scheduler and epoch >= self.warmup_scheduler_epochs:
                self.scheduler_class: LR_scheduler = LR_scheduler(
                    self.scheduler_name, self.optimizer, **self.scheduler_kwargs
                )
                self.scheduler, self.scheduler_type = (
                    self.scheduler_class.get_scheduler()
                )

                self.normal_scheduler = True
            else:
                if self.scheduler_type == "epoch":
                    self.scheduler.step()

            # Mostra loss e accuracy
            print(f"Train loss: {train_loss:.3f} | Train acc: {train_accuracy:.2f}%")
            print(f"Val loss: {val_loss:.3f} | Val acc: {val_accuracy:.2f}%")

            # salva il modello se l'accuratezza del dataset di validazione è migliorata
            if val_accuracy > self.best_acc:
                self.model_saved = True
                self.best_acc = val_accuracy
                self.save_model(val_accuracy, val_loss, epoch)

            if val_loss < self.best_loss:
                self.best_loss = val_loss

            # Valuta se stoppare l'addestramento
            self.early_stopping.calculateWhenStop(
                epoch, val_accuracy, val_loss, self.model_saved
            )

            self.model_saved = False

            # Tempo di fine epoca
            end_time: float = time.time()
            epoch_duration: float = end_time - start_time
            hours, remainder = divmod(epoch_duration, 3600)  # Ore e resto
            minutes, remainder = divmod(remainder, 60)  # Minuti e resto
            seconds = remainder  # I secondi possono essere decimali
            epoch_duration = f"{int(hours)}h:{int(minutes)}m:{int(seconds)}s"
            ColoredPrint.blue(f"\nL'addestramento è durato {epoch_duration}.")

            # Tensorboard
            self.writer.add_scalar("train/loss", train_loss, epoch)
            self.writer.add_scalar("train/accuracy", train_accuracy, epoch)
            self.writer.add_scalar("validation/loss", val_loss, epoch)
            self.writer.add_scalar("validation/accuracy", val_accuracy, epoch)

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("Hyperparameters/learning_rate", current_lr, epoch)

            print(current_lr)

            self.save_training_info(
                val_accuracy,
                val_loss,
                train_accuracy,
                train_loss,
                self.file_name,
                self.count_parameters(),
                epoch,
                epoch_duration,
                current_lr,
                self.main_save_path,
            )

            self.last_epoch = epoch + 1

            # calcola l'early stop in base all'accuratezza/loss
            if self.early_stopping.stop:
                break

        # Tensorboard: dati finali
        self.writer.add_hparams(
            {
                "num_epochs": self.last_epoch,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
            },
            {
                "metric/val_best_loss": self.best_loss,
                "metric/val_best_accuracy": self.best_acc,
            },
        )

        self.writer.flush()
        self.writer.close()

        ColoredPrint.purple("\nFINE LOOP DI TRAINING\n" + "-" * 20)

    def count_parameters(self):
        return sum(params.numel() for params in self.model.parameters())

    def save_model(self, val_accuracy: float, val_loss: float, epoch: int) -> None:
        best_model = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "seed": self.cfg[ConfigKeys.GENERAL_SETTINGS][ConfigKeys.SEED],
            "metrics": {
                "accuracy": val_accuracy,
                "loss": val_loss,
            },
            "model_config": {
                "model_name": self.model_name,
                "class_to_idx": self.class_to_idx,
                "img_size": self.image_size,
                "num_channels": self.num_channels,
                "num_params": self.count_parameters(),
            },
            "general_config": self.cfg,
        }

        if self.model_name == "custom":
            best_model["custom_model"] = self.model_config

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.file_name = f"{self.model_name}_{timestamp}.pth"

        shutil.rmtree(self.checkpoint_path)
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)

        torch.save(best_model, self.checkpoint_path / self.file_name)

        ColoredPrint.green(f"\nIl modello: {self.file_name} è stato salvato.\n")

    def save_training_info(
        self,
        val_accuracy: float,
        val_loss: float,
        train_accuracy: float,
        train_loss: float,
        file_name: str,
        num_parameters: int,
        epoch: int,
        epoch_duration: str,
        current_lr: float,
        filepath: Path,
    ):

        # Dati da salvare
        data_to_save = {
            "model": {
                "model_name": self.model_name,
                "model_save_namme": file_name,
                "num_parameters": num_parameters,
            },
            "general_configurations": {
                "num_channels": self.num_channels,
                "image_size": self.image_size,
                "epochs": self.epochs,
                "batch_size": self.batch_size
            },
            "optimizer": {"name": self.optimizer_name, "learning_rate": current_lr},
            "scheduler": {
                "scheduler_name": self.scheduler_name,
                "scheduler_kwargs": self.scheduler_kwargs,
            },
            "training_info": {
                "epoch": f"{epoch+1}/{self.epochs}",
                "epoch_duration": epoch_duration,
            },
            "model_valutation": {
                "val": {"accuracy": val_accuracy, "loss": val_loss},
                "train": {"accuracy": train_accuracy, "loss": train_loss},
            },
        }

        # Se il file esiste, carica i dati esistenti
        complete_path: Path = filepath / "training.json"
        if complete_path.exists():
            with complete_path.open("r") as f:
                existing_data = json.load(f)
        else:
            existing_data = []

        # Aggiungi i nuovi dati
        existing_data.append(data_to_save)

        # Salva tutto nel file (sovrascrive)
        with complete_path.open("w") as f:
            json.dump(existing_data, f, indent=4)

        ColoredPrint.purple(f"Training info salvati.")

    def save_model_config(self):
        path = self.main_save_path / "model_config.json"
        with path.open("w") as f:
            json.dump(self.model_config, f, indent=4)
