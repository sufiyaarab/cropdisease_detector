"""Turns validated photos into a diagnosis: model, leaf tiles, averaging, thresholds.

Decision rule (same as ml/src/metrics.py, where the numbers in the model's metrics.json come from):
  - top class is a disease and its confidence >= threshold (0.70)          -> diagnosed
  - top class is healthy and its confidence >= the crop's healthy threshold -> diagnosed healthy
  - otherwise                                                               -> not sure
The healthy threshold is higher than 0.70 because calling a sick leaf healthy costs more than a false alarm.

Photos with several leaves: the whole photo is shrunk to 224 x 224, so one small sick leaf can disappear.
With tiles on, the photo is also checked as overlapping close-ups (a grid of half-size tiles). A tile only counts
when the plant head sees the chosen crop in it, and it only replaces the whole-photo answer when it shows a
disease with confidence >= threshold and more disease than the whole photo.
"""
from dataclasses import dataclass

import numpy as np
from PIL import Image

from app.pipeline.classes import CLASSES, PLANT_CLASSES
from app.pipeline.model import Classifier
from app.schemas import Candidate

MIN_TILE_PX = 160  # smaller tiles are too blurry once resized to 224


@dataclass
class Diagnosis:
    status: str  # "diagnosed" or "not_sure"
    top: Candidate
    candidates: list[Candidate]
    used_tiles: bool = False  # True when a close-up tile, not the whole photo, decided at least one photo


def healthy_index(crop: str) -> int:
    return next(i for i, c in enumerate(CLASSES[crop]) if c.category == "healthy")


def tiles(image: Image.Image, grid: int) -> list[Image.Image]:
    """grid x grid overlapping tiles, each half the photo's width and height."""
    w, h = image.size
    tw, th = w // 2, h // 2
    if grid < 2 or min(tw, th) < MIN_TILE_PX:
        return []
    xs = np.linspace(0, w - tw, grid).astype(int)
    ys = np.linspace(0, h - th, grid).astype(int)
    return [image.crop((x, y, x + tw, y + th)) for y in ys for x in xs]


def photo_probs(image: Image.Image, crop: str, model: Classifier, threshold: float,
                grid: int, min_plant: float) -> tuple[np.ndarray, bool]:
    """Probabilities for one photo, and whether a tile decided it."""
    whole = model.predict(image, crop)
    both = getattr(model, "predict_with_plant", None)
    if both is None or grid < 2:  # the stub has no plant head, so it never uses tiles
        return whole, False
    h, plant_i = healthy_index(crop), PLANT_CLASSES.index(crop)
    best, best_sick = whole, 1.0 - whole[h]
    for tile in tiles(image, grid):
        p, plant = both(tile, crop)
        if plant[plant_i] < min_plant:
            continue  # soil, sky, stem, or another plant
        top = int(np.argmax(p))
        if top != h and p[top] >= threshold and 1.0 - p[h] > best_sick:
            best, best_sick = p, 1.0 - p[h]
    return best, best is not whole


def diagnose(images: list[Image.Image], crop: str, model: Classifier, threshold: float,
             tile_grid: int = 0, tile_min_plant: float = 0.5) -> Diagnosis:
    per_photo = [photo_probs(img, crop, model, threshold, tile_grid, tile_min_plant) for img in images]
    # With 2 or 3 photos of the same plant, average the probabilities.
    probs = np.mean([p for p, _ in per_photo], axis=0)
    order = np.argsort(probs)[::-1][:3]
    candidates = [
        Candidate(
            label=CLASSES[crop][i].label,
            display_name=CLASSES[crop][i].display_name,
            category=CLASSES[crop][i].category,
            confidence=round(float(probs[i]), 4),
        )
        for i in order
    ]
    top = candidates[0]
    need = threshold
    if top.category == "healthy":
        need = max(threshold, getattr(model, "healthy_threshold", {}).get(crop, threshold))
    status = "diagnosed" if top.confidence >= need else "not_sure"
    return Diagnosis(status, top, candidates, used_tiles=any(t for _, t in per_photo))
