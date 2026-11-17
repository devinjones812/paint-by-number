import numpy as np
from PIL import Image


def downsample(img: np.ndarray, long_side_px: int) -> np.ndarray:
    """Resize so the long side is `long_side_px`, keeping aspect ratio. Never upsamples."""
    h, w = img.shape[:2]
    scale = long_side_px / max(h, w)
    if scale >= 1:
        return img
    new_size = (round(w * scale), round(h * scale))  # Pillow wants (width, height)
    return np.asarray(Image.fromarray(img).resize(new_size, Image.Resampling.LANCZOS))
