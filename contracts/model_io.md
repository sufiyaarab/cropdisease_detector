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

The trained model (`TFLiteClassifier`) also has `predict_plant(image)`: probabilities for which plant is in the photo,
in `PLANT_CLASSES` order (`tomato`, `apple`, `corn`, `other_plant`). The app doesn't use it yet; the plan is to ask the
user to confirm the crop when it disagrees with their choice or is under 0.70, and to show no diagnosis for `other_plant`.

## Model outputs

One shared EfficientNet-B0 backbone, four heads, each giving logits:

| Output | Classes | Trained on |
| --- | --- | --- |
| `plant` | tomato, apple, corn, other_plant | every photo (other_plant = PlantVillage crops we don't support, plus any Roboflow leaves mapped to it) |
| `tomato`, `apple`, `corn` | `CLASSES[crop]` | photos of that crop only |

Probabilities = softmax(logits / T), with T per head from `calibration.json`.

## What ML publishes with every model version (Hugging Face Hub)

| File | Contents |
| --- | --- |
| `model.tflite` | EfficientNet-B0 with the plant head and three crop heads (outputs named plant / tomato / apple / corn, logits). Made by `ml/src/train.py`, loaded by `backend/app/pipeline/tflite_model.py` |
| `classes.json` | `plant` list and `disease` lists per crop, same order as the outputs |
| `calibration.json` | `temperature` per head (plant, tomato, apple, corn); `healthy_threshold` per crop (optional, v0.2+) |
| `preprocess.json` | Input size, mean and standard deviation |
| `metrics.json` | Per head: precision, recall, F1, F2 per class, missed-disease and false-alarm rates, on field photos and per source |

## Rules

1. Changing a label or the order of a class list (disease or plant) is a contract change. Update `classes.py`, `ml/configs/classes.json`, the treatment data and this file in one pull request. Adding a supported crop means: new disease list, insert it in `plant` before `other_plant`, retrain.
2. The backend marks a result "not sure" when the top probability is under 0.70 (`CONFIDENCE_THRESHOLD`), or when
   the top class is healthy and its probability is under the crop's `healthy_threshold` (a missed disease costs more
   than a false alarm). Model folders without `healthy_threshold` use 0.70 for healthy too.
3. Photos with several leaves: with `TILE_SCAN=true` the backend also runs the model on overlapping half-size tiles
   (`predict_with_plant`, one model run gives both the crop head and the plant head) and keeps a tile's answer only
   when the plant head sees the crop and the tile shows a disease at ≥ 0.70.
4. Tier 2: the YOLOv8n leaf detector crops the photo **before** `predict`. This interface does not change.
