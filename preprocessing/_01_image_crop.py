import sys
from pathlib import Path
import cv2
import numpy as np
from utils.read_file import load_json
import os

# ------------------------------------------------------------------
# PARAMETRI DA PERSONALIZZARE
# ------------------------------------------------------------------
GENERAL_SETTINGS: str = "general_settings"
CROP: str = "crop"

DATA_SETTINGS: str = "data_settings"
DIRECTORY: str = "directory"
GENERAL_DIR: str = "general_dir"
INPUT_DIR: str = "input_dir"
OUTPUT_DIR: str = "output_dir"
GREY: str = "grey"

# taglio (in pixel) sull’immagine grande
CROP_TOP = 70  # pixel da togliere in alto
CROP_BOTTOM = 20 # pixe da togliere da sotto
CROP_RIGHT = 91  # pixel da togliere a destra

# griglia (righe, colonne) e spazio interno
ROWS, COLS = 11, 9
INNER_GAP = 0  # pixel fra le celle

# estensioni accettate
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ------------------------------------------------------------------
# CLASSE DI CROP
# ------------------------------------------------------------------
class ImageCrop:
    def __init__(self, cfg: dict):
        self.cfg: dict = cfg

        self.toCrop: bool = cfg[GENERAL_SETTINGS][CROP]

        self.grey: bool = cfg[DATA_SETTINGS][GREY]

        self.general_dir: str = cfg[DATA_SETTINGS][DIRECTORY][GENERAL_DIR]

        self.input: str = os.path.join(self.general_dir, cfg[DATA_SETTINGS][DIRECTORY][INPUT_DIR])
        self.output: str = os.path.join(self.general_dir, cfg[DATA_SETTINGS][DIRECTORY][OUTPUT_DIR])

    def imread_unicode(
        self, path: str, flags: int = cv2.IMREAD_GRAYSCALE
    ) -> cv2.typing.MatLike:
        """
        Prova a leggere il file con imread -> semplice e veloce.\n
        Se fallisce, usa imdecode che funziona sempre.
        """    
    
        # prova lettura con imgread
        try:
            path.encode("ascii")  
            img = cv2.imread(path, flags)

            if img is not None:
                return img
        except UnicodeEncodeError as e:
            print("Errore nella lettura del file per incompatabilità di caratteri non ASCII. Caricamento dei file in altra modalità.")
        
        # Se fallisce, come accade solitamente con caratteri non ASCII
        # usa lettura binaria con imdecode
        try:
            with open(path, "rb") as f:
                byte_data: bytes = f.read()

            data = np.frombuffer(byte_data, dtype=np.uint8)
            img = cv2.imdecode(data, flags)

            return img
        except Exception as e:
            print(f"Impossibile leggere il file {path}: errore {e}")
            return None

    def save_cell(self, cell_img, dest_path: str) -> bool:
        """
        Salva una cella su disco.\n
        • Tenta cv2.imwrite\n
        • Se fallisce (Unicode path), usa cv2.imencode + open('wb') come file binario\n
        Restituisce True se il file esiste alla fine.
        """
        ok = cv2.imwrite(dest_path, cell_img)
        if not ok:
            ok_buf, buf = cv2.imencode(".jpg", cell_img)
            if ok_buf:
                with open(dest_path, "wb") as f:
                    buf.tofile(f)
        return Path(dest_path).exists()

    def process_image(self, img_path: Path, label: str, start_counter: int) -> int:
        """
        Esegue tutto il flusso su UNA singola immagine.\n
        1. Carica con imread_unicode\n
        2. Esegue il crop (top/right)\n
        3. Calcola dimensioni celle\n
        4. Estrae e salva le 99 celle
        """

        # Preprocessing
        # label = img_path.stem                    # es: '年' da '年.jpg'
        out_dir: Path = Path(self.output) / label  # output/年

        # Crea cartella se non esiste
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) Caricamento immagine
        img = self.imread_unicode(
            str(img_path), cv2.IMREAD_COLOR if self.grey else cv2.IMREAD_GRAYSCALE
        )

        if img is None:
            print(f"Impossibile leggere l'immagine {img_path}")
            return start_counter

        # 2) Crop regione
        h, w = img.shape[:2]
        region = img[CROP_TOP:h-CROP_BOTTOM, 0 : w - CROP_RIGHT]  # coordinate y  # coordinate x

        # 3) Dimensione di ogni cella
        cell_h = (region.shape[0] - INNER_GAP * (ROWS - 1)) // ROWS
        cell_w = (region.shape[1] - INNER_GAP * (COLS - 1)) // COLS
        if cell_h <= 0 or cell_w <= 0:
            print(
                f"Errore nel taglio dell'immagine {img_path}, controlla i parametri di crop o la griglia."
            )
            return start_counter

        # 4) Estrazione e salvataggio celle
        counter: int = start_counter
        for r in range(ROWS):
            for c in range(COLS):
                y0 = r * (cell_h + INNER_GAP)
                x0 = c * (cell_w + INNER_GAP)

                cell = region[y0 : y0 + cell_h, x0 : x0 + cell_w]
                dest_file: str = os.path.join(out_dir, f"{counter}.jpg")
                self.save_cell(cell, dest_file)
                counter += 1

        print(
            f"-> Completata l'immagine: {img_path.name}: salvate {counter - start_counter} celle in {out_dir}"
        )
        return counter

    def cropImage(self) -> bool:
        
        if not self.toCrop:
            return
        
        print("INIZIO CROP\n" + "-"*20)

        input_dir: Path = Path(self.input)

        # Verifica che la cartella di input esista
        if not input_dir.exists():
            print(
                f"Cartella '{self.input}' non trovata. Inserisci una cartella con immagini."
            )
            return False

        # Ottieni tutte le sottocartelle (= classi, cioè caratteri come 年, 火, etc.)
        subfolders = [p for p in input_dir.iterdir() if p.is_dir()]

        # Contatore globale di immagini processate
        total_imgs = 0  

        # Itera ogni sottocartella (es. input/年, input/火, ...)
        for class_dir in subfolders:
            # Il nome della cartella è l'etichetta (es. "年")
            label = class_dir.name

            # Lista di immagini valide nella sottocartella corrente
            image_files = [
                p
                for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in EXTS
            ]

            # Se non ci sono immagini nella sottocartella, salta e avvisa
            if not image_files:
                print(f" Nessuna immagine valida in '{class_dir}'.")
                continue

            # Output visivo: quante immagini trovate per quella classe
            print(f"\nClasse '{label}': {len(image_files)} immagine/i trovate")

            counter = 1
            # Processa ogni immagine della cartella
            for img_path in image_files:
                counter = self.process_image(img_path, label, counter)
                total_imgs += 1  # aggiorna il totale

        # Termina e restituisce il risultato finale
        print(f"Elaborate {total_imgs} immagini\n")
        
        print("FINE CROP\n" + "-"*20)

        return True
