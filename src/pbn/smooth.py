import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def smooth(labels: np.ndarray, out_size: tuple[int, int], sigma_px: float) -> np.ndarray:
    """Upscale the label map to `out_size` (width, height) and round off staircase borders.

    Each color's 0/1 mask is blurred; each pixel takes the color with the highest blurred value.
    Every pixel keeps exactly one label, so neighboring regions never get gaps or overlaps.
    """
    # NEAREST copies label values; other filters would invent in-between labels like 3.5
    big = np.asarray(Image.fromarray(labels.astype(np.uint8)).resize(out_size, Image.Resampling.NEAREST))

    best_value = np.full(big.shape, -1.0, dtype=np.float32)
    best_label = np.zeros(big.shape, dtype=labels.dtype)
    for c in np.unique(big):
        blurred = gaussian_filter((big == c).astype(np.float32), sigma_px)  # "how much c is around me"
        better = blurred > best_value
        best_value[better] = blurred[better]
        best_label[better] = c
    return best_label
