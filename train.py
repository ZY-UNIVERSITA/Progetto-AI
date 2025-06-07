import argparse
from pathlib import Path

from utils import load_JSON
from preprocessing import ImageCrop, DatasetSplit, DataAugmentation
from trainer import TrainerCNN, TesterCNN

class Train():
    def __init__(self, cfg: str, model_config: str = None):
        self.cfg: dict = load_JSON(cfg)
        self.mode_config = load_JSON(model_config)
    
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
        self.trainer = TrainerCNN(self.cfg, self.mode_config)

    def trainLog(self):
        self.trainer.loggingInfo()

    def train(self):
        self.trainer.train()

    def loadInference(self):
        self.inferenceClass = TesterCNN(self.cfg)

    def inference(self):
        self.inferenceClass.inference()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="config.json")
    ap.add_argument("--cfg_model", required=False, help="model_config.json")
    args = ap.parse_args()

    train = Train(Path(args.cfg), Path(args.cfg_model))
    train.cropImage()
    train.slitDataset()
    train.dataAugmentation()

    train.loadTrainer()
    train.trainLog()
    train.train()

    train.loadInference()
    train.inference()
