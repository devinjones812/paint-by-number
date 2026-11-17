from dataclasses import dataclass
from typing import Literal

import numpy as np
from skimage.color import lab2rgb, rgb2lab

from pbn.config import ColorSpace

Init = Literal["random", "kmeans++"]

@dataclass
class KMeansResult:
    centers: np.ndarray # (k, 3)
    labels: np.ndarray # (N,) index of each point's center
    inertia: float # sum of squared distances from each point to its center
    center_history: list[np.ndarray] # centers at  the start of each iteration
    inertia_history: list[float] # inertia at each iteration


def squared_distances(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """ (N, 3) points, (k, 3) centers -> (N, k) squared distance from every point to every center"""
    diff = points[:, None, :] - centers[None, :, :] # (N, k, 3)
    return (diff**2).sum(axis=2)


def init_random(points: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Use k different random points as the starting centers"""
    return points[rng.choice(len(points), size=k, replace=False)].copy()


def init_kmeans_plus_plus(points:np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """ Pick centers one by one, each far from the ones already picked (on average)"""
    centers = [points[rng.integers(len(points))]]
    for _ in range(k - 1):
        nearest_d2 = squared_distances(points, np.array(centers)).min(axis=1) # (N,)
        probs = nearest_d2 / nearest_d2.sum()
        centers.append(points[rng.choice(len(points), p=probs)])
    return np.array(centers)


def update_centers(points: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Move each center to the mean of its points. An empty cluster takes the worst-fit point"""
    new_centers = centers.copy()
    own_d2 = ((points - centers[labels]) ** 2).sum(axis=1) # each point's distance to its center
    for j in range(len(centers)):
        members = points[labels == j]
        if len(members) > 0:
            new_centers[j] = members.mean(axis=0)
        else:
            worst = own_d2.argmax()
            new_centers[j] = points[worst]
            own_d2[worst] = 0 # a second empty cluster must take a different point
    return new_centers


def kmeans(
    points: np.ndarray,
    k: int,
    seed: int = 0,
    init: Init = "kmeans++",
    max_iter: int = 100,
    tol: float = 1e-3,
) -> KMeansResult:
    """Cluster (N, 3) points into k groups. Stops when no center moves more than `tol`."""
    rng = np.random.default_rng(seed)
    centers = init_kmeans_plus_plus(points, k, rng) if init == "kmeans++" else init_random(points, k, rng)
    center_history, inertia_history = [], []

    for _ in range(max_iter):
        d2 = squared_distances(points, centers)
        labels = d2.argmin(axis=1)  # assign step
        center_history.append(centers)
        inertia_history.append(float(d2.min(axis=1).sum()))

        new_centers = update_centers(points, labels, centers)  # update step
        max_shift = np.sqrt(((new_centers - centers) ** 2).sum(axis=1)).max()
        centers = new_centers
        if max_shift < tol:
            break

    d2 = squared_distances(points, centers)
    return KMeansResult(
        centers=centers,
        labels=d2.argmin(axis=1),
        inertia=float(d2.min(axis=1).sum()),
        center_history=center_history,
        inertia_history=inertia_history,
    )


def rgb_to_space(img: np.ndarray, color_space: ColorSpace) -> np.ndarray:
    """uint8 RGB (..., 3) -> float colors in the chosen space."""
    if color_space == "lab":
        return rgb2lab(img / 255)  # L: 0..100, a and b: about -128..127
    return img.astype(float)


def space_to_rgb(colors: np.ndarray, color_space: ColorSpace) -> np.ndarray:
    """Float colors (..., 3) in the chosen space -> uint8 RGB."""
    if color_space == "lab":
        colors = lab2rgb(colors) * 255
    return np.clip(np.round(colors), 0, 255).astype(np.uint8)


def quantize(
    img: np.ndarray,
    k: int,
    color_space: ColorSpace,
    seed: int,
    sample_size: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce the image to k colors. Return labels (h, w) and palette (k, 3) uint8 RGB."""
    h, w, _ = img.shape
    pixels = rgb_to_space(img, color_space).reshape(-1, 3)  # (h*w, 3)

    rng = np.random.default_rng(seed)
    sample = pixels[rng.choice(len(pixels), size=min(sample_size, len(pixels)), replace=False)]
    result = kmeans(sample, k, seed=seed)

    labels = squared_distances(pixels, result.centers).argmin(axis=1).reshape(h, w)
    palette = space_to_rgb(result.centers, color_space)
    return sort_palette(labels, palette)


def boost_palette(palette: np.ndarray, saturation: float, contrast: float) -> np.ndarray:
    """Make colors more intense (chroma x `saturation`) and spread lightness away from mid-gray (x `contrast`)."""
    lightness, a, b = rgb_to_space(palette[None], "lab")[0].T
    lightness = np.clip(50 + (lightness - 50) * contrast, 0, 100)
    boosted = np.stack([lightness, a * saturation, b * saturation], axis=1)
    return space_to_rgb(boosted, "lab")  # colors outside the printable RGB range are clipped


def sort_palette(labels: np.ndarray, palette: np.ndarray, gray_chroma: float = 15) -> tuple[np.ndarray, np.ndarray]:
    """Renumber colors so similar ones sit together in the key: grays light to dark, then colors by hue."""
    lightness, a, b = rgb_to_space(palette[None], "lab")[0].T
    chroma = np.hypot(a, b)  # 0 = pure gray
    hue = np.degrees(np.arctan2(b, a)) % 360
    keys = [(0, 0, -l) if c < gray_chroma else (1, h, -l) for l, c, h in zip(lightness, chroma, hue)]
    order = sorted(range(len(palette)), key=lambda i: keys[i])  # order[new] = old
    new_index = np.empty(len(order), dtype=int)
    new_index[order] = np.arange(len(order))  # new_index[old] = new
    return new_index[labels], palette[order]
