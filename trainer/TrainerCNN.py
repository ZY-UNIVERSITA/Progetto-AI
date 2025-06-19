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
from utils import (
    EarlyStopping,
    ConfigKeys,
    LR_scheduler,
    Optimizer,
    ColoredPrint,
    save_training_info,
    save_epoch_info,
    generic_save_json_dict,
    save_testing_model,
    TrainingInfo,
    GeneralTrainingInfo,
)

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
    def __init__(self, cfg: dict, model_config: dict = None):
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

        # train option
        self.train_option: dict[str, any] = self.cfg[ConfigKeys.MODEL][
            ConfigKeys.TRAIN_OPTION
        ]
        self.transfer_learning: bool = self.train_option.get(
            ConfigKeys.TRANSFER_LEARNING, False
        )
        self.restart_training: bool = self.train_option.get(
            ConfigKeys.RESTART_TRAIN, False
        )

        # get model
        # get model from previous training
        if self.restart_training or self.transfer_learning:
            checkpoint_model_name = (
                Path(self.cfg[ConfigKeys.MODEL][ConfigKeys.PRETRAINED][ConfigKeys.PATH])
                / self.cfg[ConfigKeys.MODEL][ConfigKeys.PRETRAINED][ConfigKeys.FOLDER]
                / ConfigKeys.MODEL
                / self.cfg[ConfigKeys.MODEL][ConfigKeys.PRETRAINED][ConfigKeys.NAME]
            )
            checkpoint = torch.load(checkpoint_model_name, map_location=self.device)

            if self.model_name == "custom":
                self.model_config = checkpoint[ConfigKeys.CUSTOM_MODEL]

            loaded_classes = len(checkpoint["model_config"]["class_to_idx"])

            self.model: nn.Module = get_model(
                name=self.model_name,
                num_classes=loaded_classes,
                num_channels=self.num_channels,
                img_size=self.image_size,
                model_cfg=self.model_config,
            )

            self.model.load_state_dict(checkpoint["model_state"])

            print(self.model.features)
            print(self.model.classifier)

            if self.transfer_learning and loaded_classes != self.num_classes:
                last_layer: nn.Linear = self.model.classifier[-1]
                self.model.classifier[-1] = nn.Linear(
                    last_layer.in_features, self.num_classes
                )

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
        if self.restart_training:
            self.optimizer.load_state_dict(checkpoint[ConfigKeys.OPTIMIZER_STATE])

        # lr scheduler
        # start new optimizer
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
            # dict of scheduler
            self.warmup_scheduler: dict[str, any] = self.cfg[ConfigKeys.TRAIN][ConfigKeys.OPTIMIZER][ConfigKeys.WARMUP]

            # numbers of warmup epochs
            self.warmup_scheduler_epochs: int = self.warmup_scheduler[ConfigKeys.EPOCHS]
            
            # final lr 
            self.warmup_scheduler_lr: int = self.warmup_scheduler[ConfigKeys.LR]

            # numbers of epoch between an update of lr
            self.warmup_scheduler_step_size: int = self.warmup_scheduler[   ConfigKeys.STEP_SIZE]

            # calculate gamma value of scheduler
            self.warmup_scheduler_gamma: float = (
                self.warmup_scheduler_lr / self.lr
            ) ** (1 / self.warmup_scheduler_epochs)

            # create dict of scheduler args
            self.scheduler_warmup_kwargs = {
                ConfigKeys.STEP_SIZE: self.warmup_scheduler_step_size,
                ConfigKeys.GAMMA: self.warmup_scheduler_gamma,
            }

            # create lr scheduler
            self.scheduler_class: LR_scheduler = LR_scheduler(
                self.scheduler_name, self.optimizer, **self.scheduler_warmup_kwargs
            )
        else:
            self.scheduler_class: LR_scheduler = LR_scheduler(
                self.scheduler_name, self.optimizer, **self.scheduler_kwargs
            )

        self.scheduler, self.scheduler_type = self.scheduler_class.get_scheduler()

        # get previous training scheduler
        if self.restart_training:
            self.scheduler.load_state_dict(checkpoint[ConfigKeys.SCHEDULER_STATE])

        # loss function
        self.loss = nn.CrossEntropyLoss()

        # Best accuracy e loss
        if self.restart_training:
            self.best_acc: float = checkpoint["metrics"]["accuracy"]
            self.best_loss: float = checkpoint["metrics"]["loss"]
        else:
            self.best_acc: float = float("-inf")
            self.best_loss: float = float("inf")

        # epoche
        if self.restart_training:
            self.start_epoch: int = checkpoint[ConfigKeys.EPOCH]
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

        if self.restart_training:
            self.model_numeration = self.cfg[ConfigKeys.MODEL][ConfigKeys.PRETRAINED][
                ConfigKeys.FOLDER
            ]
        else:
            self.model_numeration = f"{self.model_name}_{timestamp}"

        self.main_save_path = Path(checkpoint_main_dir) / self.model_numeration
        self.main_save_path.mkdir(parents=True, exist_ok=True)

        self.checkpoint_path = self.main_save_path / ConfigKeys.MODEL.value
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)

        self.tensorboard_path = self.main_save_path / "log"

        self.save_model_config()
        self.save_config()

        # Tensorboard
        self.writer = SummaryWriter(self.tensorboard_path)

        # batch di immagini di prova
        images_batch, _ = next(iter(self.train_dl))
        images_batch = images_batch.to(self.device)
        img_grid = make_grid(images_batch, normalize=True)
        self.writer.add_image("Immagini di training", img_grid)

        # grafo del modello
        self.writer.add_graph(self.model, images_batch)

        # writer
        self.training_engine.add_writer(self.writer)

    def loggingInfo(self):
        ColoredPrint.blue("-" * 20 + "\nINIZIO LOGGING TRAINING\n")

        ColoredPrint.purple(
            f"Il training viene eseguto sul device: {self.device.upper()}."
        )

        ColoredPrint.purple(f"Il modello scelto è: {self.model_name}.")

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

        ColoredPrint.blue("\nFINE LOGGING TRAINING\n")

    def train(self) -> None:
        ColoredPrint.purple("\nINIZIO LOOP DI TRAINING\n")

        ColoredPrint.purple(f"Il training è stato impostato per {self.epochs} epoche.")

        for epoch in range(self.start_epoch, self.epochs):
            ColoredPrint.purple(f"\nEpoca {epoch+1} di {self.epochs}")

            current_lr = self.optimizer.param_groups[0][ConfigKeys.LR]

            ColoredPrint.purple(f"Learning rate di {current_lr}")

            # Tempo di inizio epoca
            start_time: float = time.time()

            ColoredPrint.cyan("\nINIZIO ELABORAZIONE DEL MODELLO\n")

            # Fai il training e la validazione
            train_loss, train_accuracy = self.training_engine.exec_epoch(
                self.train_dl, TRAIN
            )
            val_loss, val_accuracy = self.training_engine.exec_epoch(self.val_dl, VAL)

            ColoredPrint.cyan("\nFINE ELABORAZIONE DEL MODELLO\n")

            # Mostra loss e accuracy
            print(f"Train loss: {train_loss:.3f} | Train acc: {train_accuracy:.2f}%")
            print(f"Val loss: {val_loss:.3f} | Val acc: {val_accuracy:.2f}%")

            # salva il modello se l'accuratezza del dataset di validazione è migliorata
            if val_accuracy > self.best_acc:
                self.model_saved = True
                self.best_acc = val_accuracy
                self.save_model(val_accuracy, val_loss, epoch)

                self.best_loss = val_loss
                self.best_train_acc = train_accuracy
                self.best_train_loss = train_loss

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
            # eliminazione dei decimali tramite conversione in intero
            epoch_duration = f"{int(hours)}h:{int(minutes)}m:{int(seconds)}s"
            ColoredPrint.blue(f"\nL'addestramento è durato {epoch_duration}.")

            # Tensorboard
            self.writer.add_scalar("train/loss", train_loss, epoch)
            self.writer.add_scalar("train/accuracy", train_accuracy, epoch)
            self.writer.add_scalar("validation/loss", val_loss, epoch)
            self.writer.add_scalar("validation/accuracy", val_accuracy, epoch)

            self.writer.add_scalar("Hyperparameters/learning_rate", current_lr, epoch)

            self.save_training_information(
                val_accuracy,
                val_loss,
                train_accuracy,
                train_loss,
                epoch,
                epoch_duration,
                current_lr,
            )

            self.last_epoch = epoch + 1

            # se il warmup scheduler ha terminato, inietta lo scheduler normale
            if not self.normal_scheduler and epoch >= self.warmup_scheduler_epochs:
                self.scheduler_class: LR_scheduler = LR_scheduler(
                    self.scheduler_name, self.optimizer, **self.scheduler_kwargs
                )
                self.scheduler, self.scheduler_type = (
                    self.scheduler_class.get_scheduler()
                )

                self.normal_scheduler = True
            else:
                if self.scheduler_type == ConfigKeys.EPOCH:
                    self.scheduler.step()

            # istogramma dei pesi, dei bias e dei gradienti
            for name, module in self.model.named_modules():
                # Controlla se il modulo ha parametri (esclude ReLU, MaxPool, Flatten che non hanno pesi/bias)
                if isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
                    # Ottieni il tipo di layer come stringa (es. linear oppure Conv2d)
                    layer_type = type(module).__name__

                    # Itera sui parametri per esempio weight e bias
                    for param_name, param in module.named_parameters():
                        # 'param_name' è il nome del parametro come 'weight' o 'bias'
                        # 'name' è il nome del layer nel modello (es. 'features' o 'classifier')

                        # Crea un nome del tipo: nome_blocco (conv oppurer FC) + nome_layer + nome_parametro
                        tag_weights = f"{name}/{layer_type}/{param_name}"
                        self.writer.add_histogram(
                            tag_weights, param.data, global_step=epoch
                        )

                        # Se esistono dei gradienti fai il logging
                        if param.grad is not None:
                            tag_gradients = (
                                f"{name}/{layer_type}/gradients/{param_name}"
                            )
                            self.writer.add_histogram(
                                tag_gradients, param.grad.data, global_step=epoch
                            )

            # calcola l'early stop in base all'accuratezza/loss
            if self.early_stopping.stop:
                break

        # Tensorboard: dati finali
        self.writer.add_hparams(
            {
                "num_epochs": self.last_epoch,
                "learning_rate": self.optimizer.param_groups[0][ConfigKeys.LR],
            },
            {
                "metric/val_best_loss": self.best_loss,
                "metric/val_best_accuracy": self.best_acc,
            },
        )

        self.writer.flush()
        self.writer.close()

        self.save_training_general_info(
            self.best_acc, self.best_loss, self.best_train_acc, self.best_train_loss
        )

        self.save_testing_model()

        ColoredPrint.purple("\nFINE LOOP DI TRAINING\n" + "-" * 20)

    def count_parameters(self):
        return sum(params.numel() for params in self.model.parameters())

    def save_model(self, val_accuracy: float, val_loss: float, epoch: int) -> None:
        best_model = {
            ConfigKeys.EPOCH: epoch,
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
            best_model[ConfigKeys.CUSTOM_MODEL] = self.model_config

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.file_name = f"{self.model_name}_{timestamp}.pth"

        shutil.rmtree(self.checkpoint_path)
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)

        torch.save(best_model, self.checkpoint_path / self.file_name)

        ColoredPrint.green(f"\nIl modello: {self.file_name} è stato salvato.\n")

    def save_training_information(
        self,
        val_accuracy: float,
        val_loss: float,
        train_accuracy: float,
        train_loss: float,
        epoch: int,
        epoch_duration: str,
        current_lr: float,
    ):
        trainingInfo = TrainingInfo(
            file_name=self.file_name,
            epoch=epoch,
            epochs=self.epochs,
            epoch_duration=epoch_duration,
            current_lr=current_lr,
            val_accuracy=val_accuracy,
            val_loss=val_loss,
            train_accuracy=train_accuracy,
            train_loss=train_loss,
        )
        save_epoch_info(trainingInfo=trainingInfo, filepath=self.main_save_path)

    def save_training_general_info(
        self,
        val_accuracy: float,
        val_loss: float,
        train_accuracy: float,
        train_loss: float,
    ):

        generalTrainingInfo = GeneralTrainingInfo(
            model_name=self.model_name,
            file_name=self.file_name,
            num_parameters=self.count_parameters(),
            num_channels=self.num_channels,
            image_size=self.image_size,
            batch_size=self.batch_size,
            last_epoch=self.last_epoch,
            epochs=self.epochs,
            val_accuracy=val_accuracy,
            val_loss=val_loss,
            train_accuracy=train_accuracy,
            train_loss=train_loss,
            optimizer_name=self.optimizer_name,
            initial_lr=self.lr,
            scheduler_name=self.scheduler_name,
            scheduler_kwargs=self.scheduler_kwargs,
            warmup_kwargs=(
                {
                    ConfigKeys.EPOCH: self.warmup_scheduler_epochs,
                    ConfigKeys.LR: self.warmup_scheduler_lr,
                    ConfigKeys.STEP_SIZE: self.warmup_scheduler_step_size,
                }
                if self.cfg[ConfigKeys.TRAIN][ConfigKeys.OPTIMIZER][ConfigKeys.WARMUP][
                    ConfigKeys.USE_WARMUP
                ]
                else None
            ),
            early_stop_kwargs=(
                self.cfg[ConfigKeys.TRAIN][ConfigKeys.EARLY_STOPPING][
                    ConfigKeys.EARLY_STOPPING_KWARGS
                ]
                if self.cfg[ConfigKeys.TRAIN][ConfigKeys.EARLY_STOPPING][
                    ConfigKeys.USE_EARLY_STOPPING
                ]
                else None
            ),
        )

        save_training_info(
            trainingInfo=generalTrainingInfo, filepath=self.main_save_path
        )

    def save_model_config(self):
        path = self.main_save_path / "model_config.json"

        generic_save_json_dict(path, self.model_config)


    def save_config(self):
        path = self.main_save_path / "config.json"

        generic_save_json_dict(path, self.cfg)

    def save_testing_model(self):
        save_testing_model(self.file_name, self.main_save_path.name) 
