"""The real classifier: loads a trained model folder made by ml/src/train.py.

The folder holds model.tflite plus classes.json, calibration.json, preprocess.json
and metrics.json (contracts/model_io.md). Set MODEL_DIR to that folder and
USE_STUB_MODEL=false to use it.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from app.pipeline.classes import CLASSES, PLANT_CLASSES
from app.pipeline.model import softmax


class ModelContractError(RuntimeError):
    """The model folder does not match the class lists the app was built for."""


class TFLiteClassifier:
    def __init__(self, model_dir: str | Path, version: str | None = None):
        from ai_edge_litert.interpreter import Interpreter  # imported here so the stub needs no extra package

        model_dir = Path(model_dir)
        classes = json.loads((model_dir / "classes.json").read_text())
        if classes.get("plant") != PLANT_CLASSES:
            raise ModelContractError(
                f"{model_dir}: plant classes are {classes.get('plant')}, the app expects {PLANT_CLASSES}"
            )
        for crop, items in CLASSES.items():
            expected = [c.label for c in items]
            got = classes.get("disease", {}).get(crop)
            if got != expected:
                raise ModelContractError(f"{model_dir}: classes for '{crop}' are {got}, the app expects {expected}")
        self.temperature: dict[str, float] = json.loads((model_dir / "calibration.json").read_text())["temperature"]
        pre = json.loads((model_dir / "preprocess.json").read_text())
        self.size = tuple(pre["input_size"])
        metrics_file = model_dir / "metrics.json"
        self.version = version or (
            json.loads(metrics_file.read_text()).get("version") if metrics_file.exists() else model_dir.name
        )
        self._interpreter = Interpreter(model_path=str(model_dir / "model.tflite"), num_threads=2)
        self._runner = self._interpreter.get_signature_runner()
        self._input_name = next(iter(self._runner.get_input_details()))

    def logits(self, image: Image.Image) -> dict[str, np.ndarray]:
        x = np.asarray(image.convert("RGB").resize(self.size, Image.BILINEAR), dtype=np.float32)[None]
        out = self._runner(**{self._input_name: x})
        return {crop: np.asarray(v[0], dtype=np.float64) for crop, v in out.items()}

    def predict(self, image: Image.Image, crop: str) -> np.ndarray:
        """Disease probabilities for the given crop, in CLASSES[crop] order."""
        if crop not in CLASSES:
            raise ValueError(f"Unsupported crop: {crop}")
        return softmax(self.logits(image)[crop], self.temperature[crop])

    def predict_plant(self, image: Image.Image) -> np.ndarray:
        """Which plant this is, in PLANT_CLASSES order (tomato, apple, corn, other_plant).

        Not used by the app yet. Planned use: ask the user to confirm the crop when this
        disagrees with their choice or is under the confidence threshold.
        """
        return softmax(self.logits(image)["plant"], self.temperature["plant"])
