import numpy as np
from scipy import ndimage


def count_regions(labels: np.ndarray) -> int:
    """Number of connected regions (4-connected groups of same-label pixels)."""
    return sum(ndimage.label(labels == c)[1] for c in np.unique(labels))


def merge_small_regions(
    labels: np.ndarray, min_area_px: float, min_radius_px: float = 0, max_passes: int = 10
) -> np.ndarray:
    """Give each region that is too small the most common label around it.

    Too small = area < `min_area_px`, or too thin: the largest circle inside it has a radius
    < `min_radius_px`. Repeats until no small region is left, because a merge can leave a
    neighbor still too small.
    """
    labels = labels.copy()
    for _ in range(max_passes):
        changed = False
        for c in np.unique(labels):
            regions, n = ndimage.label(labels == c)  # 0 = not color c, 1..n = region id
            sizes = np.bincount(regions.ravel())
            boxes = ndimage.find_objects(regions)  # bounding box of each region
            for region_id in range(1, n + 1):
                if sizes[region_id] >= min_area_px and min_radius_px == 0:
                    continue
                # crop to the region's box plus 1 pixel, so the work is small
                rows, cols = boxes[region_id - 1]
                box = (
                    slice(max(rows.start - 1, 0), rows.stop + 1),
                    slice(max(cols.start - 1, 0), cols.stop + 1),
                )
                mask = regions[box] == region_id
                if sizes[region_id] >= min_area_px:
                    radius = ndimage.distance_transform_edt(np.pad(mask, 1)).max()
                    if radius >= min_radius_px:
                        continue
                ring = ndimage.binary_dilation(mask) & ~mask  # pixels just outside the region
                if not ring.any():
                    continue  # the region is the whole image
                crop = labels[box]  # a view: writing to it writes to `labels`
                crop[mask] = np.bincount(crop[ring]).argmax()
                changed = True
        if not changed:
            break
    return labels
