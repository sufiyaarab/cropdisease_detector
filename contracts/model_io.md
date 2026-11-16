# Model interface

This is the agreement between the ML code (`ml/`) and the backend (`backend/app/pipeline/`).
The backend is written against it using a stub, so the real model can be swapped in later.

## What the backend calls

```python
class Classifier(Protocol):
    version: str
    def predict(self, image: PIL.Image.Image, crop: str) -> np.ndarray: ...
```

- **image**: RGB photo that already passed validation. The classifier does its own resize (224 x 224) and normalisation, using the same settings as training.
- **crop**: `"tomato"`, `"apple"` or `"corn"`. It picks the crop head.
- **returns**: probabilities that sum to 1, one per class, **after temperature scaling**, in the exact order of `CLASSES[crop]` in `backend/app/pipeline/classes.py`.

## What ML publishes with every model version (Hugging Face Hub)

| File | Contents |
| --- | --- |
| weights | EfficientNet-B0 with the three crop heads |
| `classes.json` | Class labels per crop, same order as the output |
| `calibration.json` | Temperature value per crop head |
| `preprocess.json` | Input size, mean and standard deviation |
| `metrics.json` | Macro-F1 and per-class scores on the test set |

## Rules

1. Changing a label or the order of a class list is a contract change. Update `classes.py`, the treatment data and this file in one pull request.
2. The backend marks a result "not sure" when the top probability is under 0.70 (`CONFIDENCE_THRESHOLD`).
3. Tier 2: the YOLOv8n leaf detector crops the photo **before** `predict`. This interface does not change.
