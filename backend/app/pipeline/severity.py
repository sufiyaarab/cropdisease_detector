"""Severity estimate: how much of the leaf has spots.

Severity = spot pixels / leaf pixels x 100, the same measure plant scientists
rate by eye with standard area diagrams.

TIER 1 PLACEHOLDER. This is a colour count, not a trained model:
  - "leaf" pixels are green tissue plus the spots found on it
  - "spot" pixels are brown, tan, yellow or very dark patches that sit on green
    tissue (patches with little green around them, such as soil, are ignored)
It is an estimate shown with the word "Estimate". It rounds UP when unsure
(a diagnosed disease is never shown as "none"), because missing a problem costs
the gardener more than a false alarm. Tier 2 can replace it with the leaf detector.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np
from PIL import Image

GRID = 96  # the photo is reduced to at most 96 x 96 cells before counting
MIN_SPOT_CELLS = 2  # smaller patches are treated as noise
MIN_GREEN_AROUND = 0.30  # share of green cells in the ring around a patch
MAX_SPOTS = 12  # boxes returned for drawing circles on the photo

# Leaf share with spots (%) -> level. Shared with frontend/src/lib/severity.ts.
LEVELS = (("mild", 5.0), ("moderate", 25.0))


@dataclass
class SpotBox:
    """A found spot as a box in 0..1 photo coordinates (x, y = top left)."""

    x: float
    y: float
    w: float
    h: float
    cells: int


@dataclass
class Estimate:
    leaf_pct: float
    spots: list[SpotBox]


def level_for(leaf_pct: float, has_disease: bool) -> str:
    """Level from the spotted share. A diagnosed problem is at least "mild"."""
    if not has_disease and leaf_pct < LEVELS[0][1]:
        return "none"
    for name, upper in LEVELS:
        if leaf_pct < upper:
            return name
    return "severe"


def _masks(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    green = (g > r + 5) & (g > b + 10) & (g > 40)
    brightest = np.maximum(np.maximum(r, g), b)
    colourful = brightest - np.minimum(np.minimum(r, g), b) > 25  # greys and shadows are not
    brown_or_yellow = (r >= g - 8) & (r > b + 15) & (brightest > 40) & (brightest < 245) & colourful
    very_dark_brown = (brightest < 70) & (brightest >= 35) & (r > g + 8) & (r > b + 15)  # not shadows
    return green, (brown_or_yellow | very_dark_brown) & ~green


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """4-connected patches of True cells."""
    seen = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    found = []
    for y0, x0 in zip(*np.nonzero(mask), strict=True):
        if seen[y0, x0]:
            continue
        seen[y0, x0] = True
        queue, cells = deque([(y0, x0)]), []
        while queue:
            y, x = queue.popleft()
            cells.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((ny, nx))
        found.append(cells)
    return found


def estimate(image: Image.Image) -> Estimate:
    small = image.convert("RGB")
    small.thumbnail((GRID, GRID))
    rgb = np.asarray(small)
    green, spotty = _masks(rgb)
    h, w = green.shape

    kept: list[SpotBox] = []
    spot_cells = 0
    for cells in _components(spotty):
        if len(cells) < MIN_SPOT_CELLS:
            continue
        ys = [c[0] for c in cells]
        xs = [c[1] for c in cells]
        y1, y2, x1, x2 = min(ys), max(ys), min(xs), max(xs)
        ry1, ry2, rx1, rx2 = max(0, y1 - 2), min(h, y2 + 3), max(0, x1 - 2), min(w, x2 + 3)
        ring = np.ones((ry2 - ry1, rx2 - rx1), dtype=bool)
        ring[y1 - ry1 : y2 - ry1 + 1, x1 - rx1 : x2 - rx1 + 1] = False
        if ring.sum() == 0 or green[ry1:ry2, rx1:rx2][ring].mean() < MIN_GREEN_AROUND:
            continue  # not sitting on a leaf: background, soil, a pot
        spot_cells += len(cells)
        kept.append(SpotBox(x1 / w, y1 / h, (x2 - x1 + 1) / w, (y2 - y1 + 1) / h, len(cells)))

    leaf_cells = int(green.sum()) + spot_cells
    pct = 100.0 * spot_cells / leaf_cells if leaf_cells else 0.0
    kept.sort(key=lambda s: s.cells, reverse=True)
    return Estimate(round(pct, 1), kept[:MAX_SPOTS])
