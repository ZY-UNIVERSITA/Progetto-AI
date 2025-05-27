import argparse
from pathlib import Path

from utils.read_file import load_json
from preprocessing._01_crop import CropImage

class Train:
    def __init__(self, cfg: str):
        self.cfg = load_json(cfg)
    
        self.crop_image = CropImage(self.cfg)
        
    def cropImage(self):
        self.crop_image.cropImage()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="config.json")
    args = ap.parse_args()

    train = Train(Path(args.cfg))
    train.cropImage()
