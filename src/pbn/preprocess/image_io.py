from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()  # lets Pillow open .HEIC files from iPhone / Photos app


def load_rgb(path: Path) -> np.ndarray:
    """Return the image as a (height, width, 3) uint8 array."""
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)  # apply phone rotation stored in metadata
        return np.asarray(img.convert("RGB"))


def save_rgb(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr).save(path)
