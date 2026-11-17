from functools import cache
from math import ceil

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from pbn.config import Config

FONT_PATH = "/System/Library/Fonts/Avenir Next.ttc"
FONT_FACES = {"regular": 7, "medium": 5, "demibold": 2}  # face index inside the .ttc file
LINE_GRAY = 140  # light enough to hide under paint
NUMBER_GRAY = 110
TEXT_GRAY = (90, 90, 90)
FRAME_GRAY = (170, 170, 170)
FONT_PER_RADIUS = 1.2  # a 2-digit number is about 1.1 x font size wide, so it fits inside the circle


@cache
def font(size_px: int, weight: str = "regular") -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(FONT_PATH, size_px, index=FONT_FACES[weight])
    except OSError:
        return ImageFont.load_default(size_px)


def borders(labels: np.ndarray) -> np.ndarray:
    """True where a pixel's label differs from its right or bottom neighbor, thickened to ~0.25 mm."""
    edge = np.zeros(labels.shape, dtype=bool)
    edge[:, :-1] |= labels[:, 1:] != labels[:, :-1]
    edge[:-1, :] |= labels[1:, :] != labels[:-1, :]
    return ndimage.binary_dilation(edge)


def number_points(
    labels: np.ndarray, repeat_spacing_px: float, repeat_min_radius_px: float
) -> list[tuple[int, int, int, float]]:
    """(x, y, color index, radius) for each number to draw.

    The first number of a region goes at the point deepest inside it. Large regions get more
    numbers, each at least `repeat_spacing_px` from the others, wherever there is room for a
    circle of radius `repeat_min_radius_px`.
    """
    points = []
    for c in np.unique(labels):
        regions, _ = ndimage.label(labels == c)
        for region_id, (rows, cols) in enumerate(ndimage.find_objects(regions), start=1):
            mask = np.pad(regions[rows, cols] == region_id, 1)  # pad so the crop edge counts as an edge
            dist = ndimage.distance_transform_edt(mask)  # distance from each pixel to the nearest edge
            yy, xx = np.ogrid[: dist.shape[0], : dist.shape[1]]
            placed = 0
            while True:
                y, x = np.unravel_index(dist.argmax(), dist.shape)
                radius = float(dist[y, x])
                if placed and radius < repeat_min_radius_px:
                    break
                points.append((cols.start + x - 1, rows.start + y - 1, int(c), radius))
                placed += 1
                dist[(yy - y) ** 2 + (xx - x) ** 2 < repeat_spacing_px**2] = 0  # keep the next number away
    return points


def blank_page(labels: np.ndarray, cfg: Config, portrait: bool = False) -> Image.Image:
    """White page, turned to match the image orientation (or always portrait)."""
    page_w_mm, page_h_mm = sorted(cfg.page_mm) if portrait else cfg.page_mm_for(*labels.shape)
    px_per_mm = cfg.dpi / 25.4
    return Image.new("RGB", (round(page_w_mm * px_per_mm), round(page_h_mm * px_per_mm)), "white")


def outline_page(labels: np.ndarray, cfg: Config) -> tuple[Image.Image, int]:
    """Borders + numbers, centered on the page. Also return how many numbers did not fit."""
    page = blank_page(labels, cfg)
    h, w = labels.shape
    x0, y0 = (page.width - w) // 2, (page.height - h) // 2

    lines = np.where(borders(labels), LINE_GRAY, 255).astype(np.uint8)
    page.paste(Image.fromarray(lines).convert("RGB"), (x0, y0))

    draw = ImageDraw.Draw(page)
    draw.rectangle((x0, y0, x0 + w - 1, y0 + h - 1), outline=(LINE_GRAY,) * 3, width=3)  # frame
    px_per_pt = cfg.dpi / 72
    cramped = 0
    repeat_spacing_px = cfg.repeat_number_mm * cfg.dpi / 25.4
    repeat_min_radius_px = cfg.max_font_pt * px_per_pt / FONT_PER_RADIUS  # room for a full-size number
    for x, y, c, radius in number_points(labels, repeat_spacing_px, repeat_min_radius_px):
        size = radius * FONT_PER_RADIUS
        if size < cfg.min_font_pt * px_per_pt:
            cramped += 1
        size = round(np.clip(size, cfg.min_font_pt * px_per_pt, cfg.max_font_pt * px_per_pt))
        draw.text((x0 + x, y0 + y), str(c + 1), fill=(NUMBER_GRAY,) * 3, font=font(size), anchor="mm")
    return page, cramped


def legend_page(labels: np.ndarray, palette: np.ndarray, cfg: Config) -> Image.Image:
    """Centered color key (swatch, number, share of the picture) in columns, then a framed preview."""
    page = blank_page(labels, cfg, portrait=True)  # the key needs the height, even for landscape photos
    draw = ImageDraw.Draw(page)
    mm = cfg.dpi / 25.4
    margin = round(cfg.margin_mm * mm)
    center_x = page.width // 2

    y = margin + round(4 * mm)
    draw.text((center_x, y), "Color Key", fill="black", font=font(round(9 * mm), "demibold"), anchor="mt")
    y += round(14 * mm)
    draw.multiline_text(
        (center_x, y),
        "Paint each area on the outline page with the color of its number.\n"
        "The % is how much of the picture that color covers.",
        fill=TEXT_GRAY,
        font=font(round(3.8 * mm)),
        spacing=round(1.5 * mm),
        align="center",
        anchor="ma",
    )
    y += round(17 * mm)

    # key grid: fixed-width cells, the whole grid centered on the page
    columns, cell_w, row_h, swatch = 3, round(52 * mm), round(23 * mm), round(17 * mm)
    rows = ceil(len(palette) / columns)
    grid_left = center_x - columns * cell_w // 2 + round(6 * mm)
    share = np.bincount(labels.ravel(), minlength=len(palette)) / labels.size
    for i, rgb in enumerate(palette):
        col, row = divmod(i, rows)  # fill each column top to bottom
        x, top = grid_left + col * cell_w, y + row * row_h
        draw.rounded_rectangle(
            (x, top, x + swatch, top + swatch), radius=round(1.5 * mm),
            fill=tuple(int(v) for v in rgb), outline=FRAME_GRAY, width=3,
        )
        text_x = x + swatch + round(5 * mm)
        draw.text((text_x, top + round(6.5 * mm)), str(i + 1), fill="black", font=font(round(7.5 * mm), "demibold"), anchor="lm")
        draw.text((text_x, top + round(13.5 * mm)), f"{100 * share[i]:.0f}%", fill=TEXT_GRAY, font=font(round(3.8 * mm)), anchor="lm")
    y += rows * row_h + round(6 * mm)

    draw.text((center_x, y), "Preview", fill=TEXT_GRAY, font=font(round(4.2 * mm), "medium"), anchor="mt")
    y += round(8 * mm)
    preview = Image.fromarray(palette[labels])
    preview.thumbnail((page.width - 2 * margin, page.height - y - margin), Image.Resampling.LANCZOS)
    px = center_x - preview.width // 2
    page.paste(preview, (px, y))
    pad = round(1.5 * mm)
    draw.rectangle((px - pad, y - pad, px + preview.width + pad, y + preview.height + pad), outline=FRAME_GRAY, width=3)
    return page


def render_pages(labels: np.ndarray, palette: np.ndarray, cfg: Config) -> tuple[list[Image.Image], int]:
    """Return [outline page, legend page] and the number of regions too small for the minimum font."""
    outline, cramped = outline_page(labels, cfg)
    return [outline, legend_page(labels, palette, cfg)], cramped
