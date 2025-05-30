import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from typing import Tuple

from tqdm import tqdm

TRAIN: str = "train"
VAL: str = "val"
TEST: str = "test"

class TrainingEngine:
    def __init__(self, model: nn.Module, loss, optimizer, device):
        self.model: nn.Module = model
        self.loss = loss
        self.optimizer = optimizer
        self.device = device

    def exec_epoch(self, dataLoader: DataLoader, tqdm_desc: str) -> Tuple[float, float]:
        # Variabili per calcolare loss e accuracy
        total_loss: float = 0.0
        correct: int = 0
        total: int = 0

        train: bool = (tqdm_desc == TRAIN)

        # Training mode:
        # train: dropout e batch normalization
        # val/test: senza dropout e batch normalization
        if train:
            self.model.train()
        else:
            self.model.eval()

        # Barra di progresso del training/val/test
        progress_bar: tqdm = tqdm(dataLoader, desc=tqdm_desc)

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

                # Calcolare la perdita per tute le immagini
                loss = self.loss(outputs, labels)

                # In training, calcola i gradienti e fai la retropropagazione per aggiornare i pesi
                if train:
                    # Calcolo dei gradienti
                    loss.backward()
                    # Aggiornamento dei pesi
                    self.optimizer.step()

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

        # calcola la loss e l'accuracy
        avg_loss: float = total_loss / total
        avg_acc: float = 100 * correct / total

        return avg_loss, avg_acc
