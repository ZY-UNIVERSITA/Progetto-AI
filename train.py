import argparse
from pathlib import Path

from utils.read_file import load_json
from preprocessing import ImageCrop, DatasetSplit, DataAugmentation
from trainer import TrainerCNN, InferenceCNN

class Train():
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

    def loadTrainer(self):
        self.trainer = TrainerCNN(self.cfg)

    def trainLog(self):
        self.trainer.loggingInfo()

    def train(self):
        self.trainer.train()

    def loadInference(self):
        self.inferenceClass = InferenceCNN(self.cfg)

    def inference(self):
        self.inferenceClass.inference()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="config.json")
    args = ap.parse_args()

    train = Train(Path(args.cfg))
    # train.cropImage()
    # train.slitDataset()
    # train.dataAugmentation()

    # train.loadTrainer()
    # train.trainLog()
    # train.train()

    train.loadInference()
    train.inference()
