"""Photo validation: the checks that run before any model sees a photo.

Order matters. Cheap checks run first, and each failure comes back with a
plain-language message and a tip so the gardener knows how to retake the photo.
"""
import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import Settings

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

# code -> (message, tip). The frontend shows both, so keep them friendly.
ISSUES: dict[str, tuple[str, str]] = {
    "empty_file": ("The photo file is empty.", "Try taking the photo again."),
    "file_too_large": ("The photo file is too large.", "Use the in-app camera, which shrinks photos for you."),
    "unsupported_type": ("This file type is not supported.", "Upload a JPEG, PNG or WebP photo."),
    "unreadable_image": ("We could not open this photo.", "The file may be damaged. Take a new photo."),
    "too_small": ("The photo is too small to analyse.", "Move closer or use a higher camera resolution."),
    "too_blurry": ("The photo looks blurry.", "Hold the phone steady and tap the leaf to focus."),
    "too_dark": ("The photo is too dark.", "Move to brighter light or turn so the leaf faces the light."),
    "too_bright": ("The photo is washed out.", "Avoid direct sun or flash on the leaf."),
    "not_a_plant": (
        "We could not find a leaf in this photo.",
        "Fill most of the frame with one leaf against a plain background.",
    ),
}


@dataclass
class Issue:
    code: str
    message: str
    tip: str


@dataclass
class Checked:
    """A photo that passed (image set) or failed (issues set)."""

    image: Image.Image | None
    issues: list[Issue]

    @property
    def ok(self) -> bool:
        return not self.issues


def _issue(code: str) -> Issue:
    message, tip = ISSUES[code]
    return Issue(code, message, tip)


def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian: low values mean few sharp edges, so blur."""
    g = gray.astype(np.float32)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def plant_ratio(rgb: np.ndarray) -> float:
    """Share of pixels that look like plant tissue (green, yellowing or brown-green).

    PLACEHOLDER crop gate. It only rejects photos with almost no plant colour.
    Tier 1 replaces it with a trained crop gate; Tier 2 adds the YOLOv8n leaf detector.
    """
    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    greenish = (g > r - 10) & (g > b + 12) & (g > 45)
    return float(greenish.mean())


def check_photo(data: bytes, settings: Settings) -> Checked:
    if not data:
        return Checked(None, [_issue("empty_file")])
    if len(data) > settings.max_upload_bytes:
        return Checked(None, [_issue("file_too_large")])

    try:
        with Image.open(io.BytesIO(data)) as probe:
            if probe.format not in ALLOWED_FORMATS:
                return Checked(None, [_issue("unsupported_type")])
            probe.verify()  # catches truncated or corrupt files
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError):
        return Checked(None, [_issue("unreadable_image")])

    if min(image.size) < settings.min_side_px:
        return Checked(None, [_issue("too_small")])

    # Quality checks run on a small copy so they are fast and size-independent.
    small = image.copy()
    small.thumbnail((512, 512))
    rgb = np.asarray(small)
    gray = np.asarray(small.convert("L"))

    issues: list[Issue] = []
    brightness = float(gray.mean())
    if brightness < settings.min_brightness:
        issues.append(_issue("too_dark"))
    elif brightness > settings.max_brightness:
        issues.append(_issue("too_bright"))
    if sharpness(gray) < settings.min_sharpness:
        issues.append(_issue("too_blurry"))
    if not issues and plant_ratio(rgb) < settings.min_plant_ratio:
        issues.append(_issue("not_a_plant"))

    return Checked(None if issues else image, issues)


def to_stored_jpeg(image: Image.Image, max_side: int = 1280) -> bytes:
    """Re-encode for storage. Re-encoding drops EXIF, so no GPS is ever kept."""
    copy = image.copy()
    copy.thumbnail((max_side, max_side))
    out = io.BytesIO()
    copy.save(out, format="JPEG", quality=85)
    return out.getvalue()
