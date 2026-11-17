import numpy as np
from scipy import ndimage


def count_regions(labels: np.ndarray) -> int:
    """Number of connected regions (4-connected groups of same-label pixels)."""
    return sum(ndimage.label(labels == c)[1] for c in np.unique(labels))


def find_small_regions(
    labels: np.ndarray, min_area_px: float, min_radius_px: float
) -> list[tuple[int, int, tuple[slice, slice], np.ndarray]]:
    """(area, label, box, mask) for each region that is too small or too thin.

    `box` is the region's bounding box plus 1 pixel; `mask` marks the region inside the box.
    """
    small = []
    for c in np.unique(labels):
        regions, _ = ndimage.label(labels == c)  # 0 = not color c, 1..n = region id
        sizes = np.bincount(regions.ravel())
        for region_id, (rows, cols) in enumerate(ndimage.find_objects(regions), start=1):
            if sizes[region_id] >= min_area_px and min_radius_px == 0:
                continue
            box = (
                slice(max(rows.start - 1, 0), rows.stop + 1),
                slice(max(cols.start - 1, 0), cols.stop + 1),
            )
            mask = regions[box] == region_id
            if sizes[region_id] >= min_area_px:
                radius = ndimage.distance_transform_edt(np.pad(mask, 1)).max()
                if radius >= min_radius_px:
                    continue
            small.append((int(sizes[region_id]), int(c), box, mask))
    return small


def merge_small_regions(
    labels: np.ndarray, min_area_px: float, min_radius_px: float = 0, max_passes: int = 10
) -> np.ndarray:
    """Give each region that is too small the most common label around it, smallest first.

    Too small = area < `min_area_px`, or too thin: the largest circle inside it has a radius
    < `min_radius_px`. Repeats until no small region is left, because a merge can leave a
    neighbor still too small.
    """
    labels = labels.copy()
    for _ in range(max_passes):
        small = find_small_regions(labels, min_area_px, min_radius_px)
        if not small:
            break
        small.sort(key=lambda r: (r[0], r[2][0].start, r[2][1].start))  # ties: top-left first, not by color
        for _, c, box, mask in small:
            crop = labels[box]  # a view: writing to it writes to `labels`
            mask = mask & (crop == c)  # an earlier merge in this pass may have recolored part of it
            if not mask.any():
                continue
            ring = ndimage.binary_dilation(mask) & ~mask  # pixels just outside the region
            if ring.any():  # empty only if the region is the whole image
                crop[mask] = most_common_by_position(crop[ring])
    return labels


def most_common_by_position(values: np.ndarray) -> int:
    """Most common value. A tie goes to the value seen first, so the result never depends on label numbers."""
    counts = np.bincount(values)
    is_top = counts[values] == counts.max()
    return int(values[is_top.argmax()])
