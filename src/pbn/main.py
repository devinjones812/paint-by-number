import argparse
from pathlib import Path

import numpy as np

from pbn.config import Config
from pbn.kmeans import boost_palette, quantize
from pbn.preprocess import blur, downsample, load_rgb, save_rgb
from pbn.regions import count_regions, merge_small_regions
from pbn.render import FONT_PER_RADIUS, render_pages
from pbn.smooth import smooth

PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".heif"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Photo to paint-by-number.")
    parser.add_argument("path", type=Path, help="a photo (jpg, png, heic), or a folder of photos")
    args = parser.parse_args()

    if args.path.is_dir():
        photos = sorted(p for p in args.path.iterdir() if p.suffix.lower() in PHOTO_SUFFIXES)
        if not photos:
            raise SystemExit(f"no photos in {args.path}")
    else:
        photos = [args.path]

    cfg = Config()
    for n, photo in enumerate(photos, start=1):
        print(f"[{n}/{len(photos)}] {photo.name}")
        run(photo, cfg)


def run(photo: Path, cfg: Config) -> None:
    out = Path("outputs") / photo.stem
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.txt").write_text(str(cfg))

    img = load_rgb(photo)
    print(f"loaded      shape={img.shape}")
    save_rgb(out / "00_input.png", img)

    img = downsample(img, cfg.work_long_side_px)
    print(f"downsampled shape={img.shape}")
    save_rgb(out / "01_downsampled.png", img)

    img = blur(img, cfg.blur_size)
    save_rgb(out / "02_blurred.png", img)

    labels, palette = quantize(img, cfg.k, cfg.color_space, cfg.seed)
    palette = boost_palette(palette, cfg.palette_saturation, cfg.palette_contrast)
    print(f"quantized   labels={labels.shape} palette={palette.shape}")
    for i, (rgb, count) in enumerate(zip(palette, np.bincount(labels.ravel(), minlength=len(palette)))):
        print(f"  color {i + 1:2d}  rgb={tuple(int(c) for c in rgb)}  {100 * count / labels.size:5.1f}%")
    save_rgb(out / "03_quantized.png", palette[labels])

    mm_per_px = cfg.mm_per_px(*labels.shape)
    min_area_px = cfg.min_area_mm2 / mm_per_px**2
    before = count_regions(labels)
    labels = merge_small_regions(labels, min_area_px)
    print(f"merged      {before} -> {count_regions(labels)} regions "
          f"(min area {cfg.min_area_mm2} mm² = {min_area_px:.0f} px at {mm_per_px:.2f} mm/px)")
    save_rgb(out / "04_merged.png", palette[labels])

    px_per_mm = cfg.dpi / 25.4
    h, w = labels.shape
    print_size = (round(w * mm_per_px * px_per_mm), round(h * mm_per_px * px_per_mm))  # (width, height)
    labels = smooth(labels, print_size, cfg.smooth_sigma_mm * px_per_mm)
    before = count_regions(labels)
    min_radius_px = cfg.min_font_pt * cfg.dpi / 72 / FONT_PER_RADIUS  # radius that fits the smallest number
    labels = merge_small_regions(labels, cfg.min_area_mm2 * px_per_mm**2, min_radius_px)
    print(f"smoothed    {labels.shape[::-1]} px at {cfg.dpi} dpi, {before} -> {count_regions(labels)} regions")
    save_rgb(out / "05_smoothed.png", palette[labels])

    pages, cramped = render_pages(labels, palette, cfg)
    pages[0].save(out / "06_outline.png")
    pages[1].save(out / "07_legend.png")
    pages[0].save(out / "paint_by_number.pdf", save_all=True, append_images=pages[1:], resolution=cfg.dpi)
    if cramped:
        print(f"warning: {cramped} regions are too small for a readable number")

    print(f"saved to {out}/  (print paint_by_number.pdf at actual size)")
