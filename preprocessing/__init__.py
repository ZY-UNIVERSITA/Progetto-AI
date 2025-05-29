# Importazioni dirette
from ._01_image_crop import ImageCrop
from ._02_dataset_split import DatasetSplit
from ._03_data_augmentation import DataAugmentation

# Interfaccia pubblica di quello esportabile
__all__ = ["ImageCrop", "DatasetSplit", "DataAugmentation"]

# Codice di inizializzazione
print("Il package di pre-processing è stato importato!")

