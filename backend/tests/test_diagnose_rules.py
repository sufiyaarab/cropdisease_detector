"""The healthy threshold and the leaf-tile check in diagnose()."""
import numpy as np
from PIL import Image

from app.pipeline.classes import PLANT_CLASSES
from app.pipeline.diagnose import diagnose, tiles

TOMATO = PLANT_CLASSES.index("tomato")


class FakeModel:
    """Whole photo: healthy at `healthy_p`. Tiles: the top-left tile shows early blight; the rest are healthy.
    Tiles whose top-left pixel is not green are 'not a tomato' for the plant head."""

    version = "fake"

    def __init__(self, healthy_p=0.80, healthy_threshold=None, sick_tile_plant=0.9):
        self.healthy_p = healthy_p
        self.healthy_threshold = healthy_threshold or {}
        self.sick_tile_plant = sick_tile_plant

    def predict(self, image, crop):
        rest = (1 - self.healthy_p) / 4
        return np.array([self.healthy_p, rest, rest, rest, rest])

    def predict_with_plant(self, image, crop):
        plant = np.zeros(len(PLANT_CLASSES))
        if image.getpixel((0, 0))[0] > 200:  # red marker = the sick leaf
            plant[TOMATO] = self.sick_tile_plant
            plant[-1] = 1 - self.sick_tile_plant
            return np.array([0.05, 0.85, 0.04, 0.03, 0.03]), plant
        plant[TOMATO] = 1.0
        return np.array([0.95, 0.02, 0.01, 0.01, 0.01]), plant


def photo(size=800):
    img = Image.new("RGB", (size, size), (40, 160, 40))
    img.paste((230, 30, 30), (0, 0, 50, 50))
    return img


def test_healthy_needs_the_higher_healthy_threshold():
    model = FakeModel(healthy_p=0.80, healthy_threshold={"tomato": 0.90})
    assert diagnose([photo()], "tomato", model, 0.70).status == "not_sure"
    model.healthy_threshold = {"tomato": 0.75}
    found = diagnose([photo()], "tomato", model, 0.70)
    assert found.status == "diagnosed" and found.top.label == "tomato_healthy"


def test_old_models_without_healthy_threshold_use_the_normal_one():
    found = diagnose([photo()], "tomato", FakeModel(healthy_p=0.75), 0.70)
    assert found.status == "diagnosed" and found.top.label == "tomato_healthy"


def test_a_sick_leaf_tile_overrides_a_healthy_whole_photo():
    found = diagnose([photo()], "tomato", FakeModel(healthy_p=0.95), 0.70, tile_grid=3)
    assert found.top.label == "tomato_early_blight" and found.status == "diagnosed" and found.used_tiles


def test_tiles_off_keeps_the_whole_photo_answer():
    found = diagnose([photo()], "tomato", FakeModel(healthy_p=0.95), 0.70, tile_grid=0)
    assert found.top.label == "tomato_healthy" and not found.used_tiles


def test_tiles_without_the_crop_are_ignored():
    found = diagnose([photo()], "tomato", FakeModel(healthy_p=0.95, sick_tile_plant=0.3), 0.70, tile_grid=3)
    assert found.top.label == "tomato_healthy" and not found.used_tiles


def test_small_photos_get_no_tiles():
    assert tiles(Image.new("RGB", (300, 300)), 3) == []
    assert len(tiles(Image.new("RGB", (800, 600)), 3)) == 9
