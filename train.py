import argparse
from pathlib import Path

from utils.read_file import load_json
from preprocessing._01_image_crop import ImageCrop
from preprocessing._02_dataset_split import DatasetSplit
from preprocessing._03_data_augmentation import DataAugmentation

class Train:
    def __init__(self, cfg: str):
        self.cfg: dict = load_json(cfg)
    
        self.crop_image: ImageCrop = ImageCrop(self.cfg)
        self.split_dataset: DatasetSplit = DatasetSplit(self.cfg)
        self.data_augmentation: DataAugmentation = DataAugmentation(self.cfg)
        
    def cropImage(self):
        self.crop_image.cropImage()

    def slitDataset(self):
        self.split_dataset.splitDataset()

    def dataAugmentation(self):
        self.data_augmentation.augmentation()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="config.json")
    args = ap.parse_args()

    train = Train(Path(args.cfg))
    train.cropImage()
    train.slitDataset()
    train.dataAugmentation()
