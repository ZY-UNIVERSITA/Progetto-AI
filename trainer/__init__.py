from .TrainerCNN import TrainerCNN
from .InferenceCNN import InferenceCNN
from .CNN_engine import CNNEngine

# Interfaccia pubblica di quello esportabile
__all__ = ["TrainerCNN", "CNNEngine", "InferenceCNN"]

# Codice di inizializzazione
print("Il package dei trainer è stato importato!")