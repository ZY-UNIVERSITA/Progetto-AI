from enum import Enum

class ConfigKeys(str, Enum):
    # General settings
    GENERAL_SETTINGS = "general_settings"
    DEVICE = "device"
    CUDA = "cuda"
    CPU = "cpu"
    SEED: str = "seed"

    # Data settings
    DATA_SETTINGS = "data_settings"
    DIRECTORY = "directory"
    GENERAL_DIR = "general_dir"
    DATASET_DIR = "dataset_dir"

    # Dataset
    DATASET = "dataset"
    TRAIN = "train"
    TRAIN_DIR = "train_dir_aug"
    VAL = "val"
    VAL_DIR = "val_dir"
    TEST = "test"
    TEST_DIR = "test_dir"

    # Data parameters
    IMG_SIZE = "img_size"
    BATCH_SIZE = "batch_size"
    NUM_WORKERS = "num_workers"
    NUM_CHANNELS = "num_channels"

    # Model settings
    MODEL = "model"
    MODEL_NAME = "backbone"
    PRETRAINED: str = "pretrained"
    NAME: str = "name"
    FOLDER: str = "folder"

    # Train option
    TRAIN_OPTION: str = "train_option"
    TRANSFER_LEARNING: str = "transfer_learning"
    RESTART_TRAIN: str = "restart_train"

    # Training settings
    EPOCHS = "epochs"
    OPTIMIZER = "optimizer"

    # Optimizer
    OPTIMIZER_TYPE: str = "type"
    OPTIMIZER_ARGS: str = "optimizer_args"
    LR = "lr"

    # Checkpoint settings
    CHECKPOINT = "checkpoint"
    CHECKPOINT_DIR = "dir"

    # Early stopping settings
    USE_EARLY_STOPPING: str = "use_early_stopping"
    EARLY_STOPPING_KWARGS: str = "early_stopping_kwargs"
    EARLY_STOPPING = "early_stopping"
    FIRST_EPOCH = "first_epoch"
    EVALUATION_BETWEEN_EPOCH = "evaluation_between_epoch"
    PATIENCE = "patience"
    IMPROVEMENT_RATE = "improvement_rate"
    MONITOR = "monitor"
    MODE = "mode"

    # Scheduler
    SCHEDULER: str = "scheduler"
    SCHEDULER_ARGS: str = "scheduler_args"
    SCHEDULER_TYPE: str = "type"
    WARMUP: str = "warmup"
    USE_WARMUP: str = "use_warmup"
