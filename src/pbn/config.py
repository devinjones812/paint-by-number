from dataclasses import dataclass
from typing import Literal

ColorSpace = Literal["rgb", "lab"]


@dataclass(frozen=True)
class Config:
    work_long_side_px: int = 300  # downsample target
    blur_size: int = 3  # median filter size; 1 = no blur
    k: int = 15
    color_space: ColorSpace = "lab"
    seed: int = 0
    palette_saturation: float = 1.3  # >1 = more intense paint colors; 1 = as in the photo
    palette_contrast: float = 1.15  # >1 = lights lighter, darks darker
    min_area_mm2: float = 30  # smallest region you will paint
    smooth_sigma_mm: float = 1.4  # border smoothing; higher = rounder, but small regions melt
    page_mm: tuple[float, float] = (215.9, 279.4)  # US Letter (8.5 x 11 in)
    margin_mm: float = 10
    dpi: int = 300
    min_font_pt: float = 6  # smallest number on the page; thinner regions are merged away
    max_font_pt: float = 14
    repeat_number_mm: float = 50  # large regions get another number every ~50 mm

    def page_mm_for(self, h: int, w: int) -> tuple[float, float]:
        """(width, height) of the page in mm. The page turns to landscape for a landscape image."""
        short, long = sorted(self.page_mm)
        return (short, long) if h >= w else (long, short)

    def mm_per_px(self, h: int, w: int) -> float:
        """Paper size of one pixel when an (h, w) image fills the printable area."""
        page_w, page_h = self.page_mm_for(h, w)
        return min((page_w - 2 * self.margin_mm) / w, (page_h - 2 * self.margin_mm) / h)
