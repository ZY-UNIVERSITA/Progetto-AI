from .TrainerCNN import TrainerCNN
from .TesterCNN import TesterCNN
from .CNN_engine import CNNEngine

# Interfaccia pubblica di quello esportabile
__all__ = ["TrainerCNN", "CNNEngine", "TesterCNN"]

# Codice di inizializzazione
# print("Il package dei trainer è stato importato!")