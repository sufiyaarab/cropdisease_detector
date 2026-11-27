"""Add Plant Pathology 2021 (apple orchard photos, Cornell / FGVC8) to the training layout data/<label>/*.jpg.

Get it from Kaggle (accept the competition rules first, on the competition's page):
    pip install kaggle          # put your kaggle.json API token in %USERPROFILE%\\.kaggle\\
    kaggle competitions download -c plant-pathology-2021-fgvc8
    unzip plant-pathology-2021-fgvc8.zip -d plant-pathology-2021     (or right-click > Extract All on Windows)
It is large (about 15 GB of full-size photos), so this script shrinks each photo and caps each class.

Then:
    python ml/src/prepare_plant_pathology.py plant-pathology-2021 data [--max-per-class 1500]

Only photos with exactly one label are used (a photo with several diseases has no single right answer):
    healthy -> apple_healthy, scab -> apple_scab, rust -> apple_cedar_rust,
    frog_eye_leaf_spot -> apple_black_rot (frogeye leaf spot is the leaf stage of black rot)
"complex" and "powdery_mildew" are skipped: the app has no class for them.
Files are named pp2021_<id>.jpg, so train.py treats them as field photos and reports them as source "pp2021".
"""
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image

MAPPING = {
    "healthy": "apple_healthy",
    "scab": "apple_scab",
    "rust": "apple_cedar_rust",
    "frog_eye_leaf_spot": "apple_black_rot",
}


def main(src: Path, dst: Path, max_per_class: int, max_side: int) -> None:
    csv_path = next(src.rglob("train.csv"), None)
    if csv_path is None:
        raise SystemExit(f"No train.csv under {src}. Unzip the Kaggle download there first.")
    images = csv_path.parent / "train_images"
    by_label = defaultdict(list)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            labels = row["labels"].split()
            if len(labels) == 1 and labels[0] in MAPPING:
                by_label[MAPPING[labels[0]]].append(row["image"])
    rng = random.Random(42)
    for label, names in sorted(by_label.items()):
        rng.shuffle(names)
        out = dst / label
        out.mkdir(parents=True, exist_ok=True)
        done = 0
        for name in names[:max_per_class]:
            target = out / f"pp2021_{Path(name).stem}.jpg"
            if not target.exists():
                img = Image.open(images / name).convert("RGB")
                img.thumbnail((max_side, max_side))
                img.save(target, quality=92)
            done += 1
        print(f"  {label:28s} {done:5d}  (of {len(names)} single-label photos)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, help="folder with train.csv and train_images/")
    ap.add_argument("dst", type=Path, help="the data/ folder used by train.py")
    ap.add_argument("--max-per-class", type=int, default=1500, help="cap so apple doesn't swamp the other crops")
    ap.add_argument("--max-side", type=int, default=512, help="shrink photos to this size (the model uses 224)")
    a = ap.parse_args()
    main(a.src, a.dst, a.max_per_class, a.max_side)
