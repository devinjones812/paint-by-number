import numpy as np
from scipy.ndimage import median_filter


def blur(img: np.ndarray, size: int) -> np.ndarray:
    """Median filter each color channel over a `size` x `size` window. `size` 1 = no blur."""
    if size % 2 == 0:
        raise ValueError(f"blur size must be odd so the window has a center pixel, got {size}")
    if size == 1:
        return img
    return median_filter(img, size=(size, size, 1))  # 1 = do not mix R, G, B channels
