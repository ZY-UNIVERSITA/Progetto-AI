from collections.abc import Callable
import numpy as np

from utils.config_enum import ConfigKeys


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
        early_stopping_cfg: dict = self.cfg[ConfigKeys.TRAIN][ConfigKeys.EARLY_STOPPING][ConfigKeys.EARLY_STOPPING_KWARGS]

        self.first_epoch: int = early_stopping_cfg[ConfigKeys.FIRST_EPOCH]
        # numero di epoche passate tra 2 early_stop
        self.eval_epoch: int = 0
        self.evaluation_between_epoch: bool = early_stopping_cfg[
            ConfigKeys.EVALUATION_BETWEEN_EPOCH
        ]
        self.patience: int = early_stopping_cfg[ConfigKeys.PATIENCE]
        self.improvement_rate: float = early_stopping_cfg[ConfigKeys.IMPROVEMENT_RATE]
        self.monitor: str = early_stopping_cfg[ConfigKeys.MONITOR]
        self.mode: str = early_stopping_cfg[ConfigKeys.MODE]

        self.save_model = save_model_fun

        # se:
        # loss -> -inf - più piccolo è meglio è
        # acc -> +inf - più grande è meglio è
        self.best_metric = np.inf if self.mode == "min" else -np.inf
        
        # conta il numero di epoche passate per capire quando terminare il training quando arriva al livello definito d alla pazienza
        self.counter = 0

        # fermare il training
        self.stop = False

    def calculateWhenStop(
        self, epoch: int, val_accuracy: float, val_loss: float, model_saved: bool
    ) -> None:
        # aumenta il contatore per tenere il numero di epoche prima di terminare il training una volta arrivato al limite della pazienza
        self.counter += 1

        # Aggiugni 1 all'epoca perchè esso parte da index = 0 a contare mentre nel json viene inserito da index = 1
        epoch += 1

        # esegue la prova di early stop se l'epoca di partenza è almeno quello definito nel json oppure è l'ultima epoca
        if epoch >= self.first_epoch or epoch == self.epochs:
            # aumenta il numero di epoche tra 2 validazioni di early_stop
            self.eval_epoch += 1

            # se il numero di epoche passate tra 2 epoche di validazione dell'early_stop è passato oppure è l'ultima epoca allora prova a capire se eseguire l'early stop
            if self.eval_epoch >= self.evaluation_between_epoch or epoch == self.epochs:
                # resetta il contatore tra 2 early_stop
                self.eval_epoch = 0

                # calcola se eseguire l'early stop
                self.calculateStop(
                    epoch=epoch,
                    val_accuracy=val_accuracy,
                    val_loss=val_loss,
                    model_saved=model_saved,
                )

                # se è l'ultima epoca, ferma il training
                if epoch == self.epochs:
                    self.stop = True

    def calculateStop(
        self, epoch: int, val_accuracy: float, val_loss: float, model_saved: bool
    ) -> None:
        # se la modalità è min allora usa il loss altrimenti usare l'accuracy
        if self.mode == "min":
            improved = val_loss < (self.best_metric - self.improvement_rate)
        # mode == 'max'
        else:
            improved = val_accuracy > (self.best_metric + self.improvement_rate)

        # se è migliorato allora azzerra il numero di epochhe prima di fermarsi
        if improved:
            self.best_metric = val_loss if self.mode == "min" else val_accuracy
            self.counter = 0

            # salva il modello in questo caso
            if not model_saved:
                self.save_model(val_accuracy, val_loss, epoch)

        # se il numero di epoche senza miglioramenti è uguale a quello definito, allora termina il training
        else:
            if self.counter >= self.patience:
                self.stop = True
