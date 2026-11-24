# ML: train the crop disease model and plug it into the MVP

Read `contracts/model_io.md` first. It says exactly what the backend expects.
Model files and datasets never go in git (see `.gitignore`).

## What's here

| Path | What it does |
| --- | --- |
| `configs/classes.json` | Plant classes (tomato, apple, corn, other_plant) and the 13 disease labels (tomato 5, apple 4, corn 4). Must match `backend/app/pipeline/classes.py`, same order. |
| `configs/roboflow_datasets.json` | Which Roboflow Universe datasets to use and how their class names map to ours. |
| `src/prepare_plantvillage.py` | Copies PlantVillage lab photos into `data/<label>/`; other crops (pepper, potato, grape, ...) go to `data/other_plant/`, 250 per folder. |
| `src/download_roboflow.py` | Downloads Roboflow datasets into `data/<label>/` (detection boxes are cut out as leaf photos). |
| `src/train.py` | Trains EfficientNet-B0 with a plant head plus one disease head per crop, fits a temperature per head, scores the test set, exports the model folder. |
| `notebooks/train_colab.ipynb` | Does all of the above on a Colab GPU. |

The model folder that comes out (`model_v0.1/`):
`model.tflite` (about 8 MB, used by the backend), `model.keras` (to keep training), `classes.json`,
`calibration.json`, `preprocess.json`, `metrics.json`, `splits.json` (which photo went to train/val/test).

## The model

One EfficientNet-B0 backbone (ImageNet weights) shared by four small heads (Dropout 0.3 + Dense):
`plant` (tomato / apple / corn / other_plant) and one disease head per crop. So it's a 2-in-1: which plant, and
which disease on it. Every photo trains the plant head; a photo only trains its own crop's disease head.

## How it trains

1. **Heads only** (always): the ImageNet backbone is frozen, its features are cached, and the four heads are trained (Adam 1e-3, class-weighted). Fast even on a laptop CPU (about 10 minutes for PlantVillage).
2. **Fine-tune** (`--fine-tune-epochs 8`, GPU): the top 60 backbone layers are trained too (Adam 1e-4, BatchNorm frozen), on augmented photos. This is what helps with field photos. Use Colab for this.
3. **Calibrate**: a temperature per head is fitted on the validation set, so a confidence of 0.70 means roughly 70% right. The app says "not sure" below 0.70.
4. **Score**: accuracy, macro-F1, per-class F1, confusion matrix and calibration error on the test set. Roboflow photos (file names starting `rf_`) are also scored on their own as `field_photos_only`. **That is the number to report.** PlantVillage-only scores will be very high and say little about real use.

## Option A: Colab (recommended, free GPU)

1. Push this folder to the team repo.
2. Fill `configs/roboflow_datasets.json`: for each dataset, copy workspace, project and version from its URL
   `https://universe.roboflow.com/<workspace>/<project>/dataset/<version>`, set `format` (`folder` for classification,
   `yolov8` for object detection), set `enabled: true`. Push.
3. Open `notebooks/train_colab.ipynb`:
   - **VS Code (Colab extension):** open the file, Select Kernel > Colab > New Colab Server > GPU. The code runs on Colab's
     machine, so the notebook clones the repo there (push first).
   - **Browser:** colab.research.google.com > File > Open notebook > GitHub. Runtime > Change runtime type > T4 GPU.
4. Keys: in the browser, add `ROBOFLOW_API_KEY`, `HF_TOKEN` (a Hugging Face write token) and, for a private repo,
   `GITHUB_TOKEN` in the 🔑 Secrets panel. In VS Code the cells ask you to paste them instead. They are never saved.
5. Set `REPO` and `HF_MODEL_REPO` in the first cell and run the cells in order. After `--list-classes`, check the
   mapping; fix the JSON and push if a class is missing.
6. The upload cell puts the model folder in your private Hugging Face repo, tagged `v0.1`. Get it on your PC with
   `huggingface-cli download <HF_MODEL_REPO> --revision v0.1 --local-dir backend/models/v0.1`
   (browser users can also use the last cell to download a zip).

## Option B: your own computer (CPU, Windows)

```powershell
cd base-code
python -m venv .venv-ml
.venv-ml\Scripts\activate
pip install -r ml\requirements.txt

# PlantVillage (about 0.9 GB: our crops plus the others for other_plant)
git clone --depth 1 --filter=blob:none --sparse https://github.com/spMohanty/PlantVillage-Dataset ..\PlantVillage-Dataset
cd ..\PlantVillage-Dataset
git sparse-checkout set --no-cone "/raw/color/*"
cd ..\base-code
python ml\src\prepare_plantvillage.py ..\PlantVillage-Dataset\raw\color data

# Roboflow (the key only lives in this terminal)
$env:ROBOFLOW_API_KEY = "your-key"
python ml\src\download_roboflow.py --list-classes
python ml\src\download_roboflow.py --out data

# Train (heads only on CPU)
python ml\src\train.py --data data --out ml\artifacts\v0.1 --version v0.1
```

TensorFlow has no GPU support on native Windows, so this is CPU only; use Colab for fine-tuning.

## Plug the model into the backend

1. Unzip so the files sit in `backend/models/v0.1/` (`backend/models/v0.1/model.tflite`, `classes.json`, ...).
2. In `backend/.env` set:
   ```
   USE_STUB_MODEL=false
   MODEL_DIR=models/v0.1
   ```
3. `pip install -r requirements.txt` (adds `ai-edge-litert`), then start with
   `uvicorn app.main:app --reload --env-file .env`.
4. http://localhost:8000/health should show `"model_version": "v0.1"`. Scans now use the real model.
5. Optional check (PowerShell): `$env:TEST_MODEL_DIR = "models\v0.1"; pytest -q tests/test_tflite_model.py`.

The backend refuses to start if the model's `classes.json` doesn't match `classes.py`, so a wrong class order can't slip through.

Later (deployment): publish the model folder to Hugging Face Hub and have the server download it at start (`HF_MODEL_REPO`).

## Known limits of v0

- PlantVillage photos are single leaves on a plain background, so the model will be much less accurate on field photos until Roboflow field data is added and fine-tuned on.
- PlantVillage has several photos of the same leaf, so a random split leaks a little and flatters the test scores.
- Corn gray leaf spot (513) and apple cedar rust (275) have the fewest photos; class weights help, more data helps more.
- `other_plant` covers other crops' leaves, but there is still no "not a leaf at all" or "other disease on a supported crop" class. The photo gates and the 0.70 threshold have to catch those.
- The app doesn't use the plant head yet (`predict_plant` is there for when it does).
