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
