"""Download Roboflow Universe datasets and add them to the training layout data/<label>/*.jpg.

Your API key never goes in a file. Set it in the terminal first:
    Windows PowerShell:  $env:ROBOFLOW_API_KEY = "your-key"
    macOS / Linux:       export ROBOFLOW_API_KEY=your-key

Then:
    python ml/src/download_roboflow.py --list-classes     # print each dataset's class names
    python ml/src/download_roboflow.py --out data/        # download and add the images

Datasets and class mappings are set in ml/configs/roboflow_datasets.json.
"""
import argparse
import json
import os
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_BOX_PX = 64     # skip leaf boxes smaller than this (too blurry to learn from)
BOX_PADDING = 0.10  # keep a little context around each leaf box


def our_labels() -> set[str]:
    """Disease labels plus other_plant (photos of plants we don't support, for the plant head)."""
    classes = json.loads((ROOT / "configs/classes.json").read_text())
    return {lab for labels in classes["disease"].values() for lab in labels} | {"other_plant"}


def download(entry: dict, cache: Path) -> Path:
    from roboflow import Roboflow

    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise SystemExit("Set ROBOFLOW_API_KEY first (see the top of this file).")
    rf = Roboflow(api_key=key)
    project = rf.workspace(entry["workspace"]).project(entry["project"])
    location = cache / f"{entry['workspace']}__{entry['project']}__v{entry['version']}__{entry['format']}"
    if not location.exists():
        project.version(entry["version"]).download(entry["format"], location=str(location))
    return location


def yolo_names(location: Path) -> list[str]:
    import yaml

    names = yaml.safe_load((location / "data.yaml").read_text())["names"]
    return list(names.values()) if isinstance(names, dict) else list(names)


def dataset_classes(entry: dict, location: Path) -> list[str]:
    if entry["format"] == "folder":
        return sorted({p.name for split in location.iterdir() if split.is_dir() for p in split.iterdir() if p.is_dir()})
    return yolo_names(location)


def add_folder(entry, location, out, tag, counts):
    for split in [d for d in location.iterdir() if d.is_dir()]:
        for cls_dir in [d for d in split.iterdir() if d.is_dir()]:
            label = entry["class_map"].get(cls_dir.name)
            if not label:
                continue
            for img in cls_dir.iterdir():
                if img.suffix.lower() in IMG_EXT:
                    dst = out / label / f"rf_{tag}_{split.name}_{img.stem}.jpg"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    Image.open(img).convert("RGB").save(dst, quality=95)
                    counts[label] = counts.get(label, 0) + 1


def add_yolo(entry, location, out, tag, counts):
    names = yolo_names(location)
    for labels_dir in location.glob("*/labels"):
        images_dir = labels_dir.parent / "images"
        for lbl in labels_dir.glob("*.txt"):
            img_path = next((images_dir / (lbl.stem + e) for e in IMG_EXT if (images_dir / (lbl.stem + e)).exists()), None)
            if img_path is None:
                continue
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            for n, line in enumerate(lbl.read_text().splitlines()):
                parts = line.split()
                if len(parts) != 5:
                    continue  # polygon / segmentation rows are skipped
                label = entry["class_map"].get(names[int(parts[0])])
                if not label:
                    continue
                cx, cy, bw, bh = (float(v) for v in parts[1:])
                bw, bh = bw * (1 + 2 * BOX_PADDING), bh * (1 + 2 * BOX_PADDING)
                box = (max(0, (cx - bw / 2) * w), max(0, (cy - bh / 2) * h),
                       min(w, (cx + bw / 2) * w), min(h, (cy + bh / 2) * h))
                if min(box[2] - box[0], box[3] - box[1]) < MIN_BOX_PX:
                    continue
                dst = out / label / f"rf_{tag}_{labels_dir.parent.name}_{lbl.stem}_{n}.jpg"
                dst.parent.mkdir(parents=True, exist_ok=True)
                img.crop(box).save(dst, quality=95)
                counts[label] = counts.get(label, 0) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--cache", type=Path, default=Path("roboflow_downloads"))
    ap.add_argument("--list-classes", action="store_true")
    args = ap.parse_args()

    cfg = json.loads((ROOT / "configs/roboflow_datasets.json").read_text())
    valid = our_labels()
    for entry in [d for d in cfg["datasets"] if d.get("enabled")]:
        tag = entry["project"]
        location = download(entry, args.cache)
        if args.list_classes:
            print(f"\n{entry['workspace']}/{tag} v{entry['version']} classes:")
            for c in dataset_classes(entry, location):
                print(f"  {c!r:45s} -> {entry['class_map'].get(c, '(skipped)')}")
            continue
        bad = {v for v in entry["class_map"].values() if v not in valid}
        if bad:
            raise SystemExit(f"{tag}: class_map uses labels the app doesn't have: {sorted(bad)}")
        counts: dict[str, int] = {}
        (add_folder if entry["format"] == "folder" else add_yolo)(entry, location, args.out, tag, counts)
        print(f"\n{tag}: added")
        for label, n in sorted(counts.items()):
            print(f"  {label:28s} {n:5d}")


if __name__ == "__main__":
    main()
