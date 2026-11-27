import numpy as np
import pytest
from PIL import Image

from app.pipeline.classes import CLASSES
from app.pipeline.severity import estimate, level_for
from app.treatment.visuals import get_visuals
from tests.conftest import scan

GREEN = (60, 150, 45)
BROWN = (110, 70, 35)
GREY = (150, 150, 150)


def leaf_with_spot(spot_fraction: float, size: int = 200) -> Image.Image:
    """A green square leaf with one brown square spot covering spot_fraction of it."""
    arr = np.zeros((size, size, 3), np.uint8)
    arr[:] = GREEN
    side = int(round(size * spot_fraction**0.5))
    start = (size - side) // 2
    arr[start : start + side, start : start + side] = BROWN
    return Image.fromarray(arr)


@pytest.mark.parametrize("fraction", [0.04, 0.10, 0.30])
def test_estimate_measures_the_spotted_share(fraction):
    found = estimate(leaf_with_spot(fraction))
    assert found.leaf_pct == pytest.approx(fraction * 100, abs=2.5)
    assert len(found.spots) == 1
    box = found.spots[0]
    assert box.x == pytest.approx(0.5 - fraction**0.5 / 2, abs=0.03)


def test_clean_leaf_has_no_spots():
    found = estimate(leaf_with_spot(0.0))
    assert found.leaf_pct == 0
    assert found.spots == []


def test_brown_background_is_not_counted_as_a_spot():
    # Left half grey table, right half leaf, brown "soil" strip on the grey side.
    arr = np.zeros((200, 200, 3), np.uint8)
    arr[:, :100] = GREY
    arr[:, 100:] = GREEN
    arr[80:120, 10:50] = BROWN
    found = estimate(Image.fromarray(arr))
    assert found.leaf_pct == 0


@pytest.mark.parametrize(
    "pct, disease, level",
    [(0, False, "none"), (4.9, False, "none"), (0, True, "mild"), (4.9, True, "mild"),
     (5, True, "moderate"), (25, True, "severe"), (24.9, True, "moderate"), (8, False, "moderate")],
)
def test_levels_round_up(pct, disease, level):
    assert level_for(pct, disease) == level


def test_every_label_has_pictures_content():
    for items in CLASSES.values():
        for item in items:
            visuals = get_visuals(item.label)
            assert visuals is not None, item.label
            assert len(visuals.examples) == 3
            assert len(visuals.stages) == (0 if item.category == "healthy" else 3)
            assert visuals.look.leaf == item.label.split("_")[0]


def test_diagnosed_scan_has_severity_and_pictures(client, auth):
    body = scan(client, auth).json()
    assert body["status"] == "diagnosed"
    assert body["visuals"]["label"] == body["top"]["label"]
    assert body["severity"]["level"] in {"mild", "moderate", "severe"}  # the test model says disease
    assert body["severity"]["method"] == "colour_estimate"


def test_not_sure_scan_has_no_severity(client, auth, model):
    model.probs = [0.3, 0.25, 0.2, 0.15, 0.1]
    body = scan(client, auth).json()
    assert body["status"] == "not_sure"
    assert body["severity"] is None and body["visuals"] is None


class LooksAtTopLeft:
    """Test model: confident in class 1 unless the top-left corner is covered."""

    version = "test-attention"

    def predict(self, image, crop):
        arr = np.asarray(image)
        corner_is_brown = arr[:20, :20, 0].mean() > arr[:20, :20, 1].mean()
        return np.array([0.1, 0.9, 0, 0, 0]) if corner_is_brown else np.array([0.8, 0.2, 0, 0, 0])


def test_occlusion_map_finds_what_the_model_used():
    from app.pipeline.explain import occlusion_map

    arr = np.zeros((224, 224, 3), np.uint8)
    arr[:] = GREEN
    arr[:32, :32] = BROWN
    grid = occlusion_map(LooksAtTopLeft(), Image.fromarray(arr), "tomato", 1)
    assert len(grid) == 7 and all(len(r) == 7 for r in grid)
    assert grid[0][0] == 1.0
    assert sum(map(sum, grid)) == 1.0  # nothing else mattered


def test_disease_scan_has_attention_but_healthy_does_not(client, auth, model):
    body = scan(client, auth).json()
    assert body["attention"]["method"] == "occlusion"
    assert len(body["attention"]["grid"]) == 7
    model.probs = [0.9, 0.025, 0.025, 0.025, 0.025]
    assert scan(client, auth, seed=1).json()["attention"] is None
