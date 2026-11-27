"""Settings read from environment variables, with safe defaults for local work."""
import os
from dataclasses import dataclass, field


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    allowed_origins: list[str] = field(
        default_factory=lambda: os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
    )
    # Below this calibrated confidence the app says "not sure" (Report 2, final concept).
    confidence_threshold: float = _float("CONFIDENCE_THRESHOLD", 0.70)
    # Photos with several leaves: also check overlapping close-up tiles, so a small sick leaf isn't lost when the
    # whole photo is shrunk to 224 px. Needs the trained model (it uses the plant head to skip tiles without the crop).
    tile_scan: bool = os.getenv("TILE_SCAN", "true").lower() == "true"
    tile_grid: int = _int("TILE_GRID", 3)  # 3 = 3 x 3 tiles, each half the photo's width and height
    tile_min_plant: float = _float("TILE_MIN_PLANT", 0.50)  # a tile counts only if the plant head sees the crop
    model_version: str = os.getenv("MODEL_VERSION", "stub-0.1")
    # The trained model: a folder made by ml/src/train.py (model.tflite + classes/calibration/preprocess json).
    use_stub_model: bool = os.getenv("USE_STUB_MODEL", "true").lower() == "true"
    model_dir: str = os.getenv("MODEL_DIR", "")
    # Local development only: creates a ready-made account at start-up. Set to false in production.
    seed_demo_user: bool = os.getenv("SEED_DEMO_USER", "true").lower() == "true"
    # "Where the model looked" heatmap on disease results: 49 extra model runs per scan.
    show_attention: bool = os.getenv("SHOW_ATTENTION", "true").lower() == "true"

    # Photo validation limits. Tune these with real field photos.
    max_photos: int = _int("MAX_PHOTOS", 3)
    max_upload_bytes: int = _int("MAX_UPLOAD_BYTES", 8 * 1024 * 1024)
    min_side_px: int = _int("MIN_SIDE_PX", 224)
    min_sharpness: float = _float("MIN_SHARPNESS", 40.0)
    min_brightness: float = _float("MIN_BRIGHTNESS", 40.0)
    max_brightness: float = _float("MAX_BRIGHTNESS", 225.0)
    min_plant_ratio: float = _float("MIN_PLANT_RATIO", 0.12)


settings = Settings()
