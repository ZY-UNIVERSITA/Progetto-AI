import json
import os

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingInfo:
    
    file_name: str
    epoch: int
    epochs: int
    epoch_duration: str
    current_lr: float
    val_accuracy: float
    val_loss: float
    train_accuracy: float
    train_loss: float


@dataclass
class GeneralTrainingInfo:
    model_name: str
    file_name: str
    num_parameters: int
    num_channels: int
    image_size: int
    batch_size: int

    last_epoch: int
    epochs: int

    val_accuracy: float
    val_loss: float
    train_accuracy: float
    train_loss: float

    optimizer_name: str
    initial_lr: float

    warmup_kwargs: dict[str, any] = None

    scheduler_name: str = None
    scheduler_kwargs: dict = None

    early_stop_kwargs: dict[str, any] = None


def save_training_info(trainingInfo: GeneralTrainingInfo, filepath: Path) -> None:

    data_to_save = {
        "model": {
            "model_name": trainingInfo.model_name,
            "model_save_namme": trainingInfo.file_name,
            "num_parameters": trainingInfo.num_parameters,
        },
        "general_configurations": {
            "num_channels": trainingInfo.num_channels,
            "image_size": trainingInfo.image_size,
            "epochs": trainingInfo.epochs,
            "batch_size": trainingInfo.batch_size,
        },
        "training_info": {
            "last_epoch": trainingInfo.last_epoch,
            "total_epoch": trainingInfo.epochs,
        },
        "model_evaluation": {
            "val": {
                "accuracy": trainingInfo.val_accuracy,
                "loss": trainingInfo.val_loss,
            },
            "train": {
                "accuracy": trainingInfo.train_accuracy,
                "loss": trainingInfo.train_loss,
            },
        },
        "optimizer": {
            "name": trainingInfo.optimizer_name,
            "initial_lr": trainingInfo.initial_lr,
        },
    }

    if trainingInfo.warmup_kwargs is not None:
        data_to_save["warmup"] = trainingInfo.warmup_kwargs
    else:
        data_to_save["warmup"] = {"use_warmup": False}

    if trainingInfo.scheduler_name is not None:
        data_to_save["scheduler"] = {
            "scheduler_name": trainingInfo.scheduler_name,
            "scheduler_kwargs": trainingInfo.scheduler_kwargs,
        }
    else:
        data_to_save["scheduler"] = {"use_scheduler": False}

    if trainingInfo.early_stop_kwargs is not None:
        data_to_save["use_early_stopping"] = trainingInfo.early_stop_kwargs
    else:
        data_to_save["early_stop"] = {"use_early_stopping": False}

    complete_path: Path = filepath / "general_info_training.json"
    # if complete_path.exists():
    #     with complete_path.open("r") as f:
    #         existing_data = json.load(f)
    # else:
    #     existing_data = []

    # existing_data.append(data_to_save)

    with complete_path.open("w") as f:
        json.dump(data_to_save, f, indent=4)

    print("Training info salvati.")


def save_epoch_info(trainingInfo: TrainingInfo, filepath: Path) -> None:
    data_to_save = {
        "model": {
            "model_save_namme": trainingInfo.file_name,
        },
        "training_info": {
            "epoch": f"{trainingInfo.epoch+1}/{trainingInfo.epochs}",
            "epoch_duration": trainingInfo.epoch_duration,
            
            "learning_rate": trainingInfo.current_lr,
        },
        "model_valutation": {
            "val": {
                "accuracy": trainingInfo.val_accuracy,
                "loss": trainingInfo.val_loss,
            },
            "train": {
                "accuracy": trainingInfo.train_accuracy,
                "loss": trainingInfo.train_loss,
            },
        },
    }

    complete_path: Path = filepath / "training.json"
    if complete_path.exists():
        with complete_path.open("r") as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    existing_data.append(data_to_save)

    with complete_path.open("w") as f:
        json.dump(existing_data, f, indent=4)

    print("Training info salvati.")
