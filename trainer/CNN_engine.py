import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from typing import Tuple

from tqdm import tqdm

from utils import ColoredPrint, Metrics

from torch.optim.lr_scheduler import LRScheduler

from torch.utils.tensorboard import SummaryWriter

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


from sklearn.metrics import confusion_matrix

TRAIN: str = "train"
VAL: str = "val"
TEST: str = "test"


class CNNEngine:
    def __init__(
        self,
        model: nn.Module,
        loss,
        optimizer,
        scheduler: LRScheduler,
        scheduler_type: str,
        device: str,
        classes: dict[str, int],
    ):
        self.model: nn.Module = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_type = scheduler_type
        self.device = device
        self.classes = classes

    def add_writer(self, writer):
        self.writer: SummaryWriter = writer

    def exec_epoch(self, dataLoader: DataLoader, tqdm_desc: str) -> Tuple[float, float]:
        # Variabili per calcolare loss e accuracy
        total_loss: float = 0.0
        correct: int = 0
        total: int = 0

        train: bool = tqdm_desc == TRAIN

        # Training mode:
        # train: dropout e batch normalization
        # val/test: senza dropout e batch normalization
        if train:
            self.model.train()
        else:
            self.model.eval()

        # Barra di progresso del training/val/test
        progress_bar: tqdm = tqdm(dataLoader, desc=tqdm_desc, disable=False)

        real_y = []
        pred_y = []

        # Carica i dati per batch
        for images, labels in progress_bar:

            # Spostamento delle immagini e etichette sul device di elaborazione (CPU o GPU)
            images, labels = images.to(self.device), labels.to(self.device)

            if train:
                # Azzera i gradienti ad ogni batch in quanto pytoch fa retropropagazione dopo ogni batch.
                # Non si vuole accumulare gradienti vecchi
                self.optimizer.zero_grad()

            # Attiva il calcolo dei gradienti se è in modalità di train
            # Disabilitato se è in test/val/inferenza
            with torch.set_grad_enabled(train):
                # Calcola le predizioni per tutte le immagini
                outputs = self.model(images)

                # Calcolare la perdita per tutte le immagini
                loss = self.loss(outputs, labels)

                # In training, calcola i gradienti e fai la retropropagazione per aggiornare i pesi
                if train:
                    # Calcolo dei gradienti e backpropagation
                    loss.backward()

                    # Aggiornamento dei pesi
                    self.optimizer.step()

                    # lr scheduler
                    if self.scheduler_type == "batch":
                        self.scheduler.step()

            # Aggiorna la perdita totale su tutte le immagini del batch
            total_loss += loss.item() * images.size(0)

            # Ottiene le predizioni finali scegliendo la classe con probabilità massima
            preds = outputs.argmax(dim=1)

            # Conta il numero di predizioni corrette
            correct += preds.eq(labels).sum().item()

            # Aggiorna il numero totale di esempi processati
            total += labels.size(0)

            # Aggiorna la barra di avanzamento con le metriche correnti
            progress_bar.set_postfix(
                loss=total_loss / total,
                acc=100 * correct / total,
            )

            real_y = real_y + labels.tolist()
            pred_y = pred_y + preds.tolist()

        # mt = Metrics(self.classes, real_y, pred_y)

        # mt.report()

        # mt.compute_confusion_matrix()

        # matrix = mt.confusion_matrix

        if not train:
            # trasforma gli indici in stringhe tramite il dizionario
            index_to_label = {v: k for k, v in self.classes.items()}

            # crea una lista di stringhe
            labels = list(index_to_label.keys())
        
            # crea la matrice di confusione
            cm = confusion_matrix(real_y, pred_y, labels=labels)

            # Crea una heatmap con le label 
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Greens",
                xticklabels=[index_to_label[i] for i in labels],
                yticklabels=[index_to_label[i] for i in labels],
            )

            # impostazione delle label degli assi
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title("Confusion Matrix")

            # rotazione delle etichette dell'asse x per una migliore leggibilità
            plt.xticks(rotation=0)
            plt.yticks(rotation=0)

            # salva un immagine della matrice
            plt.savefig("confusion_matrix.png", dpi=300)

            # Aggiunta della figura a TensorBoard
            self.writer.add_figure("Matrice di confusione finale", fig, global_step=0)

            plt.close()

        # calcola la loss e l'accuracy
        avg_loss: float = total_loss / total
        avg_acc: float = 100 * correct / total

        return avg_loss, avg_acc
