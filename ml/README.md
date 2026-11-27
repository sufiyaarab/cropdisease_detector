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
| `src/prepare_plant_pathology.py` | Adds Plant Pathology 2021 apple orchard photos (Kaggle) as `pp2021_*.jpg`, shrunk and capped per class. |
| `src/train.py` | Trains EfficientNet-B0 with a plant head plus one disease head per crop (field-weighted focal loss with a missed-disease penalty), fits a temperature and a healthy threshold, scores the test set, exports the model folder. |
| `src/metrics.py` | The app's decision rule, per-class precision / recall / F1 / F2, missed-disease and false-alarm rates, healthy-threshold tuning. |
| `src/compare_datasets.py` | Compares datasets and loss settings on the same test photos (frozen backbone, CPU is fine). Runs are in `configs/experiments.json`. |
| `src/eval_scenes.py` | Multi-leaf test: runs the backend's real `diagnose()` on whole field photos, with and without leaf tiles. |
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

## The training objective (v0.2)

Per head and per photo i with true class y and p = softmax(logits):

    L_i = w_i * [ (1 - p_y)^γ * (-log p_y)  +  β * [y is a disease] * (-log(1 - p_healthy)) ]
    w_i = class weight of y * (λ if photo i is a field photo else 1)

| Flag | Symbol | Default | What it does |
| --- | --- | --- | --- |
| `--field-weight` | λ | 3 | Field photos count λ times as much as PlantVillage lab photos (the lab-to-field gap). |
| `--miss-penalty` | β | 1 | Extra loss when a diseased leaf leans "healthy" (a miss costs more than a false alarm). 0 = off. |
| `--focal-gamma` | γ | 2 | Focal loss: photos the model already gets right count less, hard field photos count more. 0 = plain cross-entropy. |
| `--max-missed` | α | 0.05 | After training, the healthy threshold τ_h is the smallest value (≥ 0.70) that keeps missed diseases ≤ α on validation field photos. |
| `--sources` | | all | Train only on some sources, e.g. `--sources plantvillage fieldplant`. The test set always keeps every source. |

The app's rule (backend `diagnose.py`, same code path as `metrics.py`): a disease is reported when its confidence
is ≥ 0.70; "healthy" only when its confidence is ≥ τ_h (stored per crop in `calibration.json`); anything else is
"not sure". `metrics.json` now has, per head: `app_rule_field_photos` (the number to report: macro precision,
macro recall, F1, F2, missed-disease rate, false-alarm rate, not-sure share, per-class precision and recall),
`app_rule_by_source` (one score per dataset), `argmax_*` (the old plain scores, for comparison) and the τ_h sweep.

## Compare datasets and settings first (CPU, your PC)

```powershell
python ml\src\compare_datasets.py --data data --out ml\artifacts\compare
```

The first run caches the backbone features (about 30 to 60 minutes on a CPU for all the data); later runs take minutes.
It trains every run in `configs/experiments.json`: part A changes only the data (PlantVillage, + FieldPlant,
+ PlantDoc, + Plant Pathology 2021, field only), part B changes only the loss (λ, β, γ). It writes
`results.md` with two tables for the report: per-crop field precision / recall / missed / false alarm, and
cross-dataset scores (each source on its own). Pick the run with the highest **selection score** (macro-F2 on the
shared field validation photos), not the best test column, then train that setting with fine-tuning on Colab.
Add or edit runs in the JSON; `--only A1 B4` runs just those.

## Photos with several leaves

The whole photo is shrunk to 224 x 224, so one small sick leaf can disappear. The backend now also checks a 3 x 3
grid of overlapping half-size tiles (`TILE_SCAN=true` in `backend/.env`). A tile only counts when the plant head
sees the chosen crop in it, and it only replaces the whole-photo answer when it shows a disease at ≥ 0.70. To measure
it, keep whole photos when downloading and score them:

```powershell
python ml\src\download_roboflow.py --out data --scenes-out scenes
cd backend
python ..\ml\src\eval_scenes.py --model-dir models\v0.2 --scenes ..\scenes --out ..\scenes_eval.json
```

It prints the missed-disease, found and false-alarm rates for whole photos versus with tiles, using only photos in
the model's test split. The Tier 2 leaf detector (YOLOv8n) is the full fix; tiles are the no-retraining stand-in.

## Option A: Colab (recommended, free GPU)

1. Push this folder to the team repo.
2. Fill `configs/roboflow_datasets.json`: for each dataset, copy workspace, project and version from its URL
   `https://universe.roboflow.com/<workspace>/<project>/dataset/<version>`, set `format` (`coco` or `yolov8` for object
   detection, `folder` for classification), set `enabled: true`. Push. FieldPlant and PlantDoc entries are already there.

   For detection exports, each mapped leaf box is cropped into its own photo. Roboflow's augmented copies of a photo are
   merged back to one, and every leaf from the same photo stays in the same train/val/test split, so the test score
   isn't inflated by near-copies.
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

# Optional: Plant Pathology 2021 apple photos (Kaggle, about 15 GB download, accept the competition rules first)
kaggle competitions download -c plant-pathology-2021-fgvc8
Expand-Archive plant-pathology-2021-fgvc8.zip ..\plant-pathology-2021
python ml\src\prepare_plant_pathology.py ..\plant-pathology-2021 data

# Compare datasets and loss settings (see above), then train (heads only on CPU)
python ml\src\compare_datasets.py --data data --out ml\artifacts\compare
python ml\src\train.py --data data --out ml\artifacts\v0.2 --version v0.2 --field-weight 3 --miss-penalty 1 --focal-gamma 2
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
