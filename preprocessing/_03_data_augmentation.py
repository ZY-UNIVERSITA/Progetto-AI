import os
from torchvision.transforms import v2 as transforms
from pathlib import Path
import torch

from torchvision.utils import save_image

from utils.load_image import imread_unicode

# ------------------------------------------------------------------
# PARAMETRI DA PERSONALIZZARE
# ------------------------------------------------------------------
GENERAL_SETTINGS: str = "general_settings"
DATA_AUG: str = "data_aug"
SEED: str = "seed"

DATA_SETTINGS: str = "data_settings"
DIRECTORY: str = "directory"
GENERAL_DIR: str = "general_dir"
DATASET_DIR: str = "dataset_dir"

DATASET: str = "dataset"
TRAIN_DIR: str = "train_dir"
TRAIN: str = "train"
TRAIN_DIR_AUG: str = "train_dir_aug"

IMG_SIZE: str = "img_size"

DATA_AUGMENTATION: str = "data_augmentation"
RANDOM_AFFINE: str = "random_affine"
RANDOM_PERSPECTIVE: str = "random_perspective"
RANDOM_ELASTIC: str = "random_elastic"
RANDOM_COLOR_JITTER: str = "random_color_jitter"
RANDOM_SHARPNESS: str = "random_sharpness"
RANDOM_APPLY: str = "random_apply"
RANDOM_CONTRAST: str = "random_contrast"
RANDOM_EQUALIZE: str = "random_equalize"
RANDOM_ERASING: str = "random_erasing"
COPIES: str = "copies"

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ------------------------------------------------------------------
# CLASSE DI DATA AUGMENTATION
# ------------------------------------------------------------------
class DataAugmentation:
    def __init__(self, cfg: dict) -> None:
        self.cfg: dict = cfg

        self.toAug: bool = self.cfg[GENERAL_SETTINGS][DATA_AUG]

        self.seed: int = self.cfg[GENERAL_SETTINGS][SEED]
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        self.general_dir: str = self.cfg[DATA_SETTINGS][DIRECTORY][GENERAL_DIR]

        self.input: str = os.path.join(
            self.general_dir,
            self.cfg[DATA_SETTINGS][DIRECTORY][DATASET_DIR],
            self.cfg[DATA_SETTINGS][DATASET][TRAIN_DIR],
        )

        self.output: str = os.path.join(
            self.general_dir,
            self.cfg[DATA_SETTINGS][DIRECTORY][DATASET_DIR],
            self.cfg[DATA_SETTINGS][DATASET][TRAIN_DIR_AUG],
        )

        self.copies: int = self.cfg[DATA_AUGMENTATION][COPIES]

        self.img_size: int = self.cfg[DATA_SETTINGS][IMG_SIZE]
        self.randomAffine: bool = self.cfg[DATA_AUGMENTATION][RANDOM_AFFINE]
        self.randomProspective: bool = self.cfg[DATA_AUGMENTATION][RANDOM_PERSPECTIVE]
        self.randomElastic: bool = self.cfg[DATA_AUGMENTATION][RANDOM_ELASTIC]
        self.randomColorJitter: bool = self.cfg[DATA_AUGMENTATION][RANDOM_COLOR_JITTER]
        self.randomSharpness: bool = self.cfg[DATA_AUGMENTATION][RANDOM_SHARPNESS]
        self.randomApply: bool = self.cfg[DATA_AUGMENTATION][RANDOM_APPLY]
        self.randomContrast: bool = self.cfg[DATA_AUGMENTATION][RANDOM_CONTRAST]
        self.randomEqualize: bool = self.cfg[DATA_AUGMENTATION][RANDOM_EQUALIZE]
        self.randomErasing: bool = self.cfg[DATA_AUGMENTATION][RANDOM_ERASING]

        self.transform_list: list = []

    def createTransformationPipeline(self) -> None:
        # Convert image to tensor
        tensor = transforms.ToImage()
        self.transform_list.append(tensor)
        tensor_float = transforms.ToDtype(torch.float32, scale=True)
        self.transform_list.append(tensor_float)

        # Resize images
        resize = transforms.Resize((self.img_size, self.img_size))
        self.transform_list.append(resize)


        # Apporta modifiche alla geometria dell'immagine, come rotazione, traslazione, scala e inclinazione
        if self.randomAffine:
            affine = transforms.RandomApply(
                [
                    transforms.RandomAffine(
                        degrees=10,
                        translate=(0.05, 0.05),
                        scale=(0.9, 1.1),
                        shear=(-5, 5, -5, 5),
                        # interpolation=transforms.InterpolationMode.NEAREST,
                        fill=255,
                    )
                ],
                p=1,
            )

            self.transform_list.append(affine)

        # Deforma l'immagine simulando una prospettiva diversa
        if self.randomProspective:
            perspective = transforms.RandomPerspective(
                distortion_scale=0.2, p=0.4, fill=255
            )

            self.transform_list.append(perspective)

        # Applica una trasformazione elastica (water-like-effect) per distorcere l'immagine
        if self.randomElastic:
            elastic = transforms.RandomApply(
                [
                    transforms.ElasticTransform(
                        alpha=30.0, sigma=4.0, fill=255
                    )
                ],
                p=0.3,
            )

            self.transform_list.append(elastic)

        # Modifica la luminosità e il contrasto dell'immagine
        if self.randomColorJitter:
            color_jitter = transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                    )
                ],
                p=0.3,
            )

            self.transform_list.append(color_jitter)

        # Regola la nitidezza dell'immagine
        if self.randomSharpness:
            sharpness = transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3)

            self.transform_list.append(sharpness)

        # Applica della sfocatura
        if self.randomApply:
            gaussianBlur = transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3)], p=0.3
            )

            self.transform_list.append(gaussianBlur)

        # Modifica il contrasto
        if self.randomContrast:
            contrast = transforms.RandomAutocontrast(p=0.2)

            self.transform_list.append(contrast)

        # Bilancia la distribuzione delle intensità dei pixel
        if self.randomEqualize:
            equalize = transforms.RandomEqualize(p=0.2)

            self.transform_list.append(equalize)

        # Aggiunge del rumore, con parti tagliate
        if self.randomErasing:
            erasing = transforms.RandomErasing(
                scale=(0.01, 0.04),
                ratio=(0.5, 2.0),
                value="random",
                p=0.2,
                inplace=False,
            )

            self.transform_list.append(erasing)

    def process_image(self, image: Path, output_dir: str, base_name: str) -> None:
        # Legge l'immagine
        orig_image = imread_unicode(str(image))

        # Crea delle copie e applica data augmentation e salva l'immagine
        for i in range(self.copies):
            aug_image: torch.Tensor = self.compose(orig_image)

            save_image(aug_image, str(output_dir / f"{base_name}_aug_{i}.jpg"))

    def augmentation(self) -> bool:
        if not self.toAug:
            return False

        print("\nINIZIO DATA AUGMENTATION\n" + "-" * 20)

        self.createTransformationPipeline()
        self.compose = transforms.Compose(self.transform_list)

        input: Path = Path(self.input)

        for class_dir in input.iterdir():
            print(f"\nInizio data augmentation per il carattere {class_dir.name}")

            images: list[Path] = [path for path in class_dir.iterdir()]

            # Nome della directory a seconda del carattere
            dir_name: str = class_dir.name

            # Crea la directory per il char
            output_dir = Path(self.output) / dir_name
            output_dir.mkdir(parents=True, exist_ok=True)

            # for image in images:
            for image in images:
                base_name: str = image.name.split(".")[0]
                self.process_image(image, output_dir, base_name)

            print("Fine data augmentation.\n")

        print("\nFINE DATA AUGMENTATION\n" + "-" * 20)

        return True
