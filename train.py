import argparse
from pathlib import Path

from utils import load_JSON
from preprocessing import ImageCrop, DatasetSplit, DataAugmentation
from trainer import TrainerCNN, TesterCNN

from utils import config_checker

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

class Test():
    def __init__(self, cfg: str):
        self.cfg: dict = load_JSON(cfg)

    def loadInference(self):
        self.inferenceClass = TesterCNN(self.cfg)

    def inference(self):
        self.inferenceClass.inference()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, help="train/test")
    ap.add_argument("--cfg", required=True, help="config.json")
    ap.add_argument("--cfg_model", required=False, help="model_config.json")
    args = ap.parse_args()

    if config_checker(args.cfg, "config_schema/config_schema.json") and config_checker(args.cfg_model, "config_schema/model_config_schema.json"):
        if args.mode == "train":
            train = Train(Path(args.cfg), Path(args.cfg_model))
            
            train.cropImage()
            train.slitDataset()
            train.dataAugmentation()

            train.loadTrainer()
            train.trainLog()
            train.train()
        else:
            test = Test(Path(args.cfg))

            test.loadInference()
            test.inference()
