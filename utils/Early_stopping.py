from collections.abc import Callable
import numpy as np

from utils import ConfigKeys


class EarlyStopping:
    def __init__(
        self,
        cfg: dict,
        save_model_fun: Callable[[float, float, int], None],
    ) -> None:
        # config
        self.cfg = cfg

        # total epochs
        self.epochs: int = self.cfg[ConfigKeys.TRAIN][ConfigKeys.EPOCHS]

        # Early stop
        early_stopping_cfg: dict = self.cfg[ConfigKeys.TRAIN][ConfigKeys.EARLY_STOPPING]

        self.first_epoch: int = early_stopping_cfg[ConfigKeys.FIRST_EPOCH]
        self.eval_epoch: int = 0
        self.evaluation_between_epoch: bool = early_stopping_cfg[
            ConfigKeys.EVALUATION_BETWEEN_EPOCH
        ]
        self.patience: int = early_stopping_cfg[ConfigKeys.PATIENCE]
        self.improvement_rate: float = early_stopping_cfg[ConfigKeys.IMPROVEMENT_RATE]
        self.monitor: str = early_stopping_cfg[ConfigKeys.MONITOR]
        self.mode: str = early_stopping_cfg[ConfigKeys.MODE]

        self.save_model = save_model_fun

        self.best_metric = np.inf if self.mode == "min" else -np.inf
        self.counter = 0
        self.stop = False

    def calculateWhenStop(
        self, epoch: int, val_accuracy: float, val_loss: float, model_saved: bool
    ) -> None:
        self.counter += 1
        epoch += 1

        if epoch >= self.first_epoch or epoch == self.epochs:
            self.eval_epoch += 1

            if self.eval_epoch >= self.evaluation_between_epoch or epoch == self.epochs:
                self.eval_epoch = 0

                self.calculateStop(
                    epoch=epoch,
                    val_accuracy=val_accuracy,
                    val_loss=val_loss,
                    model_saved=model_saved,
                )
                
                if epoch == self.epochs:
                    self.stop = True

    def calculateStop(
        self, epoch: int, val_accuracy: float, val_loss: float, model_saved: bool
    ) -> None:
        if self.mode == "min":
            improved = val_loss < (self.best_metric - self.improvement_rate)
        # mode == 'max'
        else: 
            improved = val_accuracy > (self.best_metric + self.improvement_rate)

        if improved:
            self.best_metric = val_loss if self.mode == "min" else val_accuracy
            self.counter = 0

            if not model_saved:
                self.save_model(val_accuracy, val_loss, epoch)

        else:
            if self.counter >= self.patience:
                self.stop = True
