"""Where the model looked: an occlusion heatmap.

How it works: cover one square of the photo at a time with a plain patch and see how
much the model's confidence in its answer drops. Squares whose cover makes the
answer much less likely are the parts the model relied on.

Why not Grad-CAM (yet): Grad-CAM needs the model's last feature maps and gradients,
and the exported model.tflite only gives the final scores. Occlusion works with any
model behind the `Classifier` interface, the stub included (the stub's map means
nothing). For a real Grad-CAM (for this model it equals a class activation map,
because the heads are global average pooling + one dense layer), the ML export has to
add a `features` output; see docs/disease-visuals.md.

This is NOT the severity: it shows what drove the answer, not how much leaf is damaged.
"""
import numpy as np
from PIL import Image

from app.pipeline.model import Classifier

GRID = 7  # 7 x 7 squares = 49 extra model runs per scan
WORK_SIZE = 224


def occlusion_map(
    model: Classifier, image: Image.Image, crop: str, class_index: int, grid: int = GRID
) -> list[list[float]]:
    """Rows of grid x grid values from 0 (did not matter) to 1 (mattered most)."""
    base_img = image.convert("RGB").resize((WORK_SIZE, WORK_SIZE), Image.BILINEAR)
    base = np.asarray(base_img).copy()
    fill = base.reshape(-1, 3).mean(axis=0).astype(np.uint8)
    start = float(model.predict(base_img, crop)[class_index])

    edges = np.linspace(0, WORK_SIZE, grid + 1).round().astype(int)
    drops = np.zeros((grid, grid))
    for row in range(grid):
        for col in range(grid):
            covered = base.copy()
            covered[edges[row] : edges[row + 1], edges[col] : edges[col + 1]] = fill
            prob = float(model.predict(Image.fromarray(covered), crop)[class_index])
            drops[row, col] = max(0.0, start - prob)

    top = drops.max()
    if top <= 1e-6:
        return [[0.0] * grid for _ in range(grid)]
    return [[round(float(v), 3) for v in r] for r in drops / top]
