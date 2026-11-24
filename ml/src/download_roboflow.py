"""Download Roboflow Universe datasets and add them to the training layout data/<label>/*.jpg.

Your API key never goes in a file. Set it in the terminal first:
    Windows PowerShell:  $env:ROBOFLOW_API_KEY = "your-key"
    macOS / Linux:       export ROBOFLOW_API_KEY=your-key

Then:
    python ml/src/download_roboflow.py --list-classes     # print each dataset's class names
    python ml/src/download_roboflow.py --out data/        # download and add the images

Datasets and class mappings are set in ml/configs/roboflow_datasets.json.
Formats: "coco" or "yolov8" (object detection: each leaf box is cropped into its own image)
and "folder" (classification). Roboflow's augmented copies are collapsed to one per photo.
"""
import argparse
import json
import os
import re
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
    if not isinstance(entry["version"], int):
        raise SystemExit(f"{entry['project']}: set \"version\" to the dataset version number in roboflow_datasets.json")
    project = rf.workspace(entry["workspace"]).project(entry["project"])
    location = cache / f"{entry['workspace']}__{entry['project']}__v{entry['version']}__{entry['format']}"
    if not location.exists():
        project.version(entry["version"]).download(entry["format"], location=str(location))
    return location


def photo_key(filename: str) -> str:
    """Roboflow names exports "<original>_jpg.rf.<hash>.jpg"; augmented copies share <original>.

    Returns <original>, cleaned so it is safe inside our file names (no "__", which we use as a separator).
    """
    stem = Path(filename).name.split(".rf.")[0]
    stem = re.sub(r"[^A-Za-z0-9_-]", "-", Path(stem).stem if "." in stem else stem)
    return re.sub(r"_{2,}", "_", stem).strip("_") or "photo"


def yolo_names(location: Path) -> list[str]:
    import yaml

    names = yaml.safe_load((location / "data.yaml").read_text())["names"]
    return list(names.values()) if isinstance(names, dict) else list(names)


def coco_files(location: Path) -> list[Path]:
    return sorted(location.glob("*/_annotations.coco.json"))


def dataset_classes(entry: dict, location: Path) -> list[str]:
    fmt = entry["format"]
    if fmt == "folder":
        return sorted({p.name for split in location.iterdir() if split.is_dir() for p in split.iterdir() if p.is_dir()})
    if fmt.startswith("coco"):
        return sorted({c["name"] for f in coco_files(location) for c in json.loads(f.read_text())["categories"]})
    return yolo_names(location)


# Each reader yields (image_path, [(class_name, box_or_None), ...]); box = (x0, y0, x1, y1) in pixels.
def read_folder(location):
    for split in [d for d in location.iterdir() if d.is_dir()]:
        for cls_dir in [d for d in split.iterdir() if d.is_dir()]:
            for img in sorted(cls_dir.iterdir()):
                if img.suffix.lower() in IMG_EXT:
                    yield img, [(cls_dir.name, None)]


def read_yolo(location):
    names = yolo_names(location)
    for labels_dir in sorted(location.glob("*/labels")):
        images_dir = labels_dir.parent / "images"
        for lbl in sorted(labels_dir.glob("*.txt")):
            img_path = next((images_dir / (lbl.stem + e) for e in IMG_EXT if (images_dir / (lbl.stem + e)).exists()), None)
            if img_path is None:
                continue
            w, h = Image.open(img_path).size
            objs = []
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue  # polygon / segmentation rows are skipped
                cx, cy, bw, bh = (float(v) for v in parts[1:])
                objs.append((names[int(parts[0])], ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)))
            yield img_path, objs


def read_coco(location):
    for ann_file in coco_files(location):
        coco = json.loads(ann_file.read_text())
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        by_image: dict[int, list] = {}
        for a in coco["annotations"]:
            x, y, bw, bh = a["bbox"]
            by_image.setdefault(a["image_id"], []).append((cats[a["category_id"]], (x, y, x + bw, y + bh)))
        for im in sorted(coco["images"], key=lambda i: i["file_name"]):
            path = ann_file.parent / im["file_name"]
            if path.exists() and im["id"] in by_image:
                yield path, by_image[im["id"]]


READERS = {"folder": read_folder, "yolov8": read_yolo, "yolov5pytorch": read_yolo, "coco": read_coco}


def add_dataset(entry, location, out, tag, counts):
    """Crop each mapped leaf box into data/<label>/rf_<tag>__<photo>__<n>.jpg.

    Roboflow's augmented copies are collapsed to one per original photo, and the photo name stays in the
    file name so train.py keeps every leaf of a photo in the same split (no train/test leakage).
    """
    fmt = "coco" if entry["format"].startswith("coco") else entry["format"]
    seen, dupes = set(), 0
    for img_path, objs in READERS[fmt](location):
        key = photo_key(img_path.name)
        if key in seen:
            dupes += 1
            continue
        seen.add(key)
        img = None
        for n, (cls, box) in enumerate(objs):
            label = entry["class_map"].get(cls)
            if not label:
                continue
            if img is None:
                img = Image.open(img_path).convert("RGB")
            w, h = img.size
            if box is not None:
                pad_x, pad_y = (box[2] - box[0]) * BOX_PADDING, (box[3] - box[1]) * BOX_PADDING
                box = (max(0, box[0] - pad_x), max(0, box[1] - pad_y), min(w, box[2] + pad_x), min(h, box[3] + pad_y))
                if min(box[2] - box[0], box[3] - box[1]) < MIN_BOX_PX:
                    continue
            crop = img.crop(box) if box is not None else img
            dst = out / label / f"rf_{tag}__{key}__{n}.jpg"
            dst.parent.mkdir(parents=True, exist_ok=True)
            crop.save(dst, quality=95)
            counts[label] = counts.get(label, 0) + 1
    return len(seen), dupes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--cache", type=Path, default=Path("roboflow_downloads"))
    ap.add_argument("--list-classes", action="store_true")
    args = ap.parse_args()

    cfg = json.loads((ROOT / "configs/roboflow_datasets.json").read_text())
    valid = our_labels()
    for entry in [d for d in cfg["datasets"] if d.get("enabled")]:
        tag = re.sub(r"[^A-Za-z0-9-]", "-", entry["project"])
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
        photos, dupes = add_dataset(entry, location, args.out, tag, counts)
        print(f"\n{tag}: {photos} photos used, {dupes} augmented copies skipped. Leaf images added:")
        for label, n in sorted(counts.items()):
            print(f"  {label:28s} {n:5d}")


if __name__ == "__main__":
    main()
