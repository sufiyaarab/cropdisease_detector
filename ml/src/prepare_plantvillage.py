"""Copy PlantVillage into the training layout data/<label>/*.jpg.

Our 13 disease classes come from the tomato, apple and corn folders. Every other
PlantVillage crop (pepper, potato, grape, strawberry, ...) goes into data/other_plant/,
capped per folder, so the plant head learns to say "not a plant we support".

Get the images first (about 0.9 GB, colour photos only):
    git clone --depth 1 --filter=blob:none --sparse https://github.com/spMohanty/PlantVillage-Dataset
    cd PlantVillage-Dataset && git sparse-checkout set --no-cone "/raw/color/*"

Then:
    python ml/src/prepare_plantvillage.py PlantVillage-Dataset/raw/color data/ [--other-per-folder 250]
"""
import argparse
import random
import shutil
from pathlib import Path

# PlantVillage folder -> our contract label (configs/classes.json)
MAPPING = {
    "Tomato___healthy": "tomato_healthy",
    "Tomato___Early_blight": "tomato_early_blight",
    "Tomato___Late_blight": "tomato_late_blight",
    "Tomato___Septoria_leaf_spot": "tomato_septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "tomato_spider_mites",
    "Apple___healthy": "apple_healthy",
    "Apple___Apple_scab": "apple_scab",
    "Apple___Cedar_apple_rust": "apple_cedar_rust",
    "Apple___Black_rot": "apple_black_rot",
    "Corn_(maize)___healthy": "corn_healthy",
    "Corn_(maize)___Northern_Leaf_Blight": "corn_northern_leaf_blight",
    "Corn_(maize)___Common_rust_": "corn_common_rust",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "corn_gray_leaf_spot",
}


IMG_EXT = {".jpg", ".jpeg", ".png"}
# A crop with any entry in MAPPING is supported, so none of its folders go to other_plant.
# Adding a crop = add its folders to MAPPING; it leaves other_plant automatically.
SUPPORTED_PREFIXES = tuple(sorted({folder.split("___")[0] + "___" for folder in MAPPING}))


def main(src: Path, dst: Path, other_per_folder: int) -> None:
    for folder, label in MAPPING.items():
        files = sorted(p for p in (src / folder).glob("*") if p.suffix.lower() in IMG_EXT)
        out = dst / label
        out.mkdir(parents=True, exist_ok=True)
        for p in files:
            shutil.copy2(p, out / f"pv_{p.name}")
        print(f"{label:28s} {len(files):5d} images")


    # Unsupported crops -> other_plant, a random sample per folder so no single crop dominates.
    rng = random.Random(42)
    out = dst / "other_plant"
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    for folder in sorted(d for d in src.iterdir() if d.is_dir() and not d.name.startswith(SUPPORTED_PREFIXES)):
        files = sorted(p for p in folder.glob("*") if p.suffix.lower() in IMG_EXT)
        rng.shuffle(files)
        tag = folder.name.split("___")[0].split("_")[0].lower()
        for p in files[:other_per_folder]:
            shutil.copy2(p, out / f"pv_{tag}_{p.name}")
        total += min(len(files), other_per_folder)
        print(f"  other_plant <- {folder.name:45s} {min(len(files), other_per_folder):5d}")
    print(f"{'other_plant':28s} {total:5d} images")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--other-per-folder", type=int, default=250)
    a = ap.parse_args()
    main(a.src, a.dst, a.other_per_folder)
