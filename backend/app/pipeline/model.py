"""The model interface and a stub that stands in for the real classifier.

Anything that implements `Classifier` can be plugged into the pipeline. The real
EfficientNet-B0 loader (weights from Hugging Face Hub) replaces `StubClassifier`
without touching the API or the frontend.
"""
import hashlib
from typing import Protocol

import numpy as np
from PIL import Image

from app.pipeline.classes import CLASSES


class Classifier(Protocol):
    version: str

    def predict(self, image: Image.Image, crop: str) -> np.ndarray:
        """Return calibrated probabilities, one per class in CLASSES[crop] order."""
        ...


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class StubClassifier:
    """Fake model: the same photo always gets the same answer.

    It hashes the pixels to pick a class, so different photos give different
    results and some fall under the "not sure" threshold. It knows nothing
    about plants. Never show its output to real users as a diagnosis.
    """

    def __init__(self, version: str = "stub-0.1", temperature: float = 1.0):
        self.version = version
        self.temperature = temperature

    def predict(self, image: Image.Image, crop: str) -> np.ndarray:
        small = image.resize((32, 32))
        seed = int.from_bytes(hashlib.sha256(small.tobytes()).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        n = len(CLASSES[crop])
        logits = rng.normal(0.0, 1.0, n)
        logits[rng.integers(n)] += 3.2
        return softmax(logits, self.temperature)
