import cv2
import numpy as np


def imread_unicode(path: str, flags: int = cv2.IMREAD_GRAYSCALE) -> cv2.typing.MatLike:
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
        print(
            "Errore nella lettura del file per incompatabilità di caratteri non ASCII. Caricamento dei file in altra modalità."
        )

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
