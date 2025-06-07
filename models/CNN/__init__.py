from .v00_cnn_semplice import SmallCNN as CNN_v0
from .v01_cnn_semplice import SmallCNN as CNN_v1
from .v03_cnn_semplice import SmallCNN as CNN_v3
from .CNN_model import FlexibleCNNModel as CNNModel

__all__ = ["CNN_v0", "CNN_v1", "CNN_v3", "CNNModel"]
