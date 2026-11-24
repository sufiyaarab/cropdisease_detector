"""Checks for the trained-model loader. The full check needs a model folder:
    TEST_MODEL_DIR=path/to/model pytest tests/test_tflite_model.py
"""
import json
import os

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("ai_edge_litert")

from app.pipeline.classes import CLASSES, PLANT_CLASSES  # noqa: E402
from app.pipeline.tflite_model import ModelContractError, TFLiteClassifier  # noqa: E402


def test_rejects_model_with_wrong_class_order(tmp_path):
    disease = {crop: [c.label for c in items] for crop, items in CLASSES.items()}
    disease["tomato"] = list(reversed(disease["tomato"]))
    classes = {"plant": PLANT_CLASSES, "disease": disease}
    (tmp_path / "classes.json").write_text(json.dumps(classes))
    with pytest.raises(ModelContractError):
        TFLiteClassifier(tmp_path)


@pytest.mark.skipif(not os.getenv("TEST_MODEL_DIR"), reason="set TEST_MODEL_DIR to a trained model folder")
def test_real_model_returns_probabilities_in_contract_order():
    model = TFLiteClassifier(os.environ["TEST_MODEL_DIR"])
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (300, 400, 3), dtype=np.uint8))
    for crop, items in CLASSES.items():
        p = model.predict(img, crop)
        assert p.shape == (len(items),)
        assert abs(p.sum() - 1) < 1e-5
    plant = model.predict_plant(img)
    assert plant.shape == (len(PLANT_CLASSES),)
    assert abs(plant.sum() - 1) < 1e-5


def test_rejects_model_without_plant_head(tmp_path):
    disease = {crop: [c.label for c in items] for crop, items in CLASSES.items()}
    (tmp_path / "classes.json").write_text(json.dumps({"disease": disease}))
    with pytest.raises(ModelContractError):
        TFLiteClassifier(tmp_path)
