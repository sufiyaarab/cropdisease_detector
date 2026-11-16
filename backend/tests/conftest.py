import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import deps
from app.main import app
from app.store import Store


def make_photo(kind: str = "leaf", size: int = 400, fmt: str = "JPEG", seed: int = 0) -> bytes:
    """Synthetic test photos. 'leaf' is green with sharp texture, so it passes every check."""
    rng = np.random.default_rng(seed)
    if kind == "leaf":
        arr = np.zeros((size, size, 3), np.uint8)
        arr[..., 0] = rng.integers(30, 90, (size, size))
        arr[..., 1] = rng.integers(110, 200, (size, size))
        arr[..., 2] = rng.integers(20, 70, (size, size))
    elif kind == "blurry":
        arr = np.full((size, size, 3), (60, 150, 50), np.uint8)
    elif kind == "dark":
        arr = rng.integers(0, 20, (size, size, 3)).astype(np.uint8)
    elif kind == "bright":
        arr = rng.integers(240, 256, (size, size, 3)).astype(np.uint8)
    elif kind == "not_plant":
        arr = rng.integers(60, 200, (size, size, 1)).astype(np.uint8).repeat(3, axis=2)
    else:
        raise ValueError(kind)
    out = io.BytesIO()
    Image.fromarray(arr).save(out, format=fmt, quality=95)
    return out.getvalue()


class FixedModel:
    """Test model that returns whatever probabilities the test sets."""

    version = "test-model"

    def __init__(self):
        self.probs = None

    def predict(self, image, crop):
        from app.pipeline.classes import CLASSES

        if self.probs is not None:
            return np.array(self.probs, dtype=float)
        n = len(CLASSES[crop])
        p = np.full(n, 0.1 / (n - 1))
        p[1] = 0.9
        return p


@pytest.fixture
def model():
    return FixedModel()


@pytest.fixture
def client(model, monkeypatch):
    fresh = Store()
    monkeypatch.setattr(deps, "store", fresh)
    app.dependency_overrides[deps.get_store] = lambda: fresh
    app.dependency_overrides[deps.get_model] = lambda: model
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def auth(client):
    res = client.post("/auth/signup", json={"email": "gardener@example.com", "password": "correct-horse"})
    return {"Authorization": f"Bearer {res.json()['token']}"}


def scan(client, auth, kind="leaf", crop="tomato", plant_id=None, seed=0):
    data = {"crop": crop}
    if plant_id:
        data["plant_id"] = plant_id
    files = [("photos", ("leaf.jpg", make_photo(kind, seed=seed), "image/jpeg"))]
    return client.post("/scans", data=data, files=files, headers=auth)
