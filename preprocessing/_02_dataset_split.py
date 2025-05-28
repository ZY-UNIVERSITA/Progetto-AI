import os
from pathlib import Path
import random
import shutil

# ------------------------------------------------------------------
# PARAMETRI DA PERSONALIZZARE
# ------------------------------------------------------------------
GENERAL_SETTINGS: str = "general_settings"
SPLIT: str = "split"
SEED: str = "seed"

DATA_SETTINGS: str = "data_settings"
DIRECTORY: str = "directory"
GENERAL_DIR: str = "general_dir"
INPUT_DIR: str = "output_dir"
OUTPUT_DIR: str = "dataset_dir"

DATASET: str = "dataset"
TRAIN_DIR: str = "train_dir"
VAL_DIR: str = "val_dir"
TEST_DIR: str = "test_dir"
SPLIT_VALUE: str = "split_value"
TRAIN: str = "train"
VAL: str = "val"
TEST: str = "test"

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ------------------------------------------------------------------
# CLASSE DI SPLIT
# ------------------------------------------------------------------
class DatasetSplit:
    def __init__(self, cfg: dict):
        self.cfg: dict = cfg

        self.toSplit: bool = cfg[GENERAL_SETTINGS][SPLIT]

        self.seed: int = cfg[GENERAL_SETTINGS][SEED]

        self.general_dir: str = cfg[DATA_SETTINGS][DIRECTORY][GENERAL_DIR]

        self.input: str = os.path.join(
            self.general_dir, cfg[DATA_SETTINGS][DIRECTORY][INPUT_DIR]
        )
        self.output: str = os.path.join(
            self.general_dir, cfg[DATA_SETTINGS][DIRECTORY][OUTPUT_DIR]
        )

        self.train: str = os.path.join(
            self.output, cfg[DATA_SETTINGS][DATASET][TRAIN_DIR]
        )
        self.val: str = os.path.join(self.output, cfg[DATA_SETTINGS][DATASET][VAL_DIR])
        self.test: str = os.path.join(
            self.output, cfg[DATA_SETTINGS][DATASET][TEST_DIR]
        )

        self.train_value: float = cfg[DATA_SETTINGS][DATASET][SPLIT_VALUE][TRAIN]
        self.val_value: float = cfg[DATA_SETTINGS][DATASET][SPLIT_VALUE][VAL]
        self.test_value: float = cfg[DATA_SETTINGS][DATASET][SPLIT_VALUE][TEST]


    def split(self, images: list[Path]) -> list[list[Path]]:
        random.shuffle(images)
        len_images: int = len(images)
        len_train: int = int(len_images * self.train_value)
        len_val: int = int(len_images * self.val_value)
        len_test: int = len_images - len_train - len_val

        # Se il numero di elementi non è sufficiente, prende solo il primo elemento
        train: list[Path] = images[:len_train] or images[:1]

        # Se la lista risultante è vuota, ruba un elemento dalla lista precedente in questo modo c'è almeno un elemento in ogni lista
        val: list[Path] = images[len_train : len_train + len_val] or train[-1:]
        test: list[Path] = images[len_train + len_val :] or val[-1:]

        return train, val, test

    def splitDataset(self):
        print("\nINIZIO SPLIT\n" + "-"*20)

        random.seed(self.seed)

        # Crea i path di input e output
        src_root: Path = Path(self.input)
        dst_root: Path = Path(self.output)

        # Crea la cartella di output se non esiste, comprese le cartelle di livello superiore se non esistono
        dst_root.mkdir(parents=True, exist_ok=True)

        # Loop tra le sottocartelle in input
        for sub_dir in src_root.iterdir():

            # Salta tutto quello che non è una cartella
            if not sub_dir.is_dir():
                continue

            # Loop di ricerca di tutti le immagini nella sottocartella
            images: list[Path] = [
                p for p in sub_dir.iterdir() if p.suffix.lower() in EXTS
            ]

            # Salta le sottocartelle che non presentano immagini dopo il loop
            if not images:
                continue

            # Suddividi randomicamente le immagini in 3 liste di immagini
            train, val, test = self.split(images)

            # Associa i nomi dei set alle immagini, creando 3 sottocartelle con associati le immagini suddivisi in cartelle
            for set_name, list_images in zip((TRAIN, VAL, TEST), (train, val, test)):
                # Crea la cartella di output completa
                output_dir: Path = dst_root / set_name / sub_dir.name
                output_dir.mkdir(parents=True, exist_ok=True)

                for image in list_images:
                    # copy2 mantiene anche i metadati che potrebbero essere importanti se definisci precedentemente
                    shutil.copy2(image, output_dir / image.name)

            print(f"Elaborato: {sub_dir.name}: {len(train)}/{len(val)}/{len(test)} (train/val/test)")

        print("Split concluso. Le immagini sono state salvate in:", dst_root.resolve())

        print("\nFINE SPLIT\n" + "-"*20)

