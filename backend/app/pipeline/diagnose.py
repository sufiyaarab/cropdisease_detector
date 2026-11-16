"""Turns validated photos into a diagnosis: model, averaging, threshold."""
from dataclasses import dataclass

import numpy as np
from PIL import Image

from app.pipeline.classes import CLASSES
from app.pipeline.model import Classifier
from app.schemas import Candidate


@dataclass
class Diagnosis:
    status: str  # "diagnosed" or "not_sure"
    top: Candidate
    candidates: list[Candidate]


def diagnose(images: list[Image.Image], crop: str, model: Classifier, threshold: float) -> Diagnosis:
    # With 2 or 3 photos of the same plant, average the probabilities.
    probs = np.mean([model.predict(img, crop) for img in images], axis=0)
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
    status = "diagnosed" if top.confidence >= threshold else "not_sure"
    return Diagnosis(status, top, candidates)
