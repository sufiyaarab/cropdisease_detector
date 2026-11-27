"""Multi-leaf test: run the backend's real diagnose() on whole field photos, with and without leaf tiles.

The training and test photos are single-leaf crops, but users upload photos with several leaves. This scores the
model the way the app uses it, on whole Roboflow photos saved by
    python ml/src/download_roboflow.py --out data --scenes-out scenes
Only photos in the model's test split are used (same photo groups as splits.json), so nothing was trained on.

A photo counts as diseased when any of its leaf boxes has a disease label.
  missed       = a diseased photo answered "healthy" (confident)
  false alarm  = a photo with only healthy leaves not answered "healthy"
  found        = a diseased photo answered with one of the diseases on it (confident)

Run it from the backend's virtual environment (it needs the backend packages, not TensorFlow):
    cd backend
    python ../ml/src/eval_scenes.py --model-dir models/v0.2 --scenes ../scenes
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

BACKEND = Path(__file__).resolve().parents[2] / "backend"
sys.path.insert(0, str(BACKEND))

from app.pipeline.diagnose import diagnose
from app.pipeline.tflite_model import TFLiteClassifier


def crop_of(labels):
    crops = {lab.split("_", 1)[0] for lab in labels if lab != "other_plant"}
    return crops.pop() if len(crops) == 1 else None


def score(rows):
    sick = [r for r in rows if r["sick"]]
    well = [r for r in rows if not r["sick"]]

    def share(items, test):
        return round(sum(map(test, items)) / len(items), 4) if items else None

    return {
        "photos": len(rows), "diseased": len(sick), "healthy_only": len(well),
        "missed_disease_rate": share(sick, lambda r: r["status"] == "diagnosed" and r["category"] == "healthy"),
        "disease_found_rate": share(sick, lambda r: r["status"] == "diagnosed" and r["top"] in r["labels"]),
        "false_alarm_rate": share(well, lambda r: not (r["status"] == "diagnosed" and r["category"] == "healthy")),
        "not_sure_share": share(rows, lambda r: r["status"] == "not_sure"),
        "tile_decided_share": share(rows, lambda r: r["used_tiles"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--scenes", type=Path, required=True)
    ap.add_argument("--splits", type=Path, default=None, help="the model's splits.json (default: in --model-dir)")
    ap.add_argument("--all", action="store_true", help="use every photo, not only the test split (inflated)")
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--grid", type=int, default=3)
    ap.add_argument("--min-plant", type=float, default=0.50)
    ap.add_argument("--out", type=Path, default=None, help="write the results as JSON here")
    args = ap.parse_args()

    model = TFLiteClassifier(args.model_dir)
    photos = json.loads((args.scenes / "scenes.json").read_text())
    if not args.all:
        splits = json.loads((args.splits or args.model_dir / "splits.json").read_text())
        test_groups = {Path(p).stem.rsplit("__", 1)[0] for p, _ in splits["test"] if Path(p).name.startswith("rf_")}
        photos = {k: v for k, v in photos.items() if Path(k).stem in test_groups}
    print(f"{len(photos)} whole photos ({'all' if args.all else 'test split only'})", flush=True)

    results = {"whole_photo": [], "with_tiles": []}
    for name, info in sorted(photos.items()):
        crop = crop_of(info["labels"])
        if crop is None:
            continue  # several crops in one photo, or only other plants
        img = Image.open(args.scenes / name).convert("RGB")
        sick = any(not lab.endswith("_healthy") for lab in info["labels"] if lab.startswith(crop))
        for mode, grid in (("whole_photo", 0), ("with_tiles", args.grid)):
            d = diagnose([img], crop, model, args.threshold, tile_grid=grid, tile_min_plant=args.min_plant)
            results[mode].append({"photo": name, "crop": crop, "labels": info["labels"], "leaves": info["leaves"],
                                  "sick": sick, "status": d.status, "top": d.top.label,
                                  "category": d.top.category, "used_tiles": d.used_tiles})

    report = {}
    for mode, rows in results.items():
        report[mode] = {"all": score(rows)}
        for crop in sorted({r["crop"] for r in rows}):
            report[mode][crop] = score([r for r in rows if r["crop"] == crop])
        report[mode]["several_leaves"] = score([r for r in rows if r["leaves"] > 1])
    print(f"\n{'':16s}{'group':16s}{'photos':>7s}{'missed':>9s}{'found':>8s}{'false alarm':>13s}{'not sure':>10s}")
    for mode, groups in report.items():
        for g, s in groups.items():
            fmt = lambda v: f"{v:.3f}" if v is not None else "  -  "
            print(f"{mode:16s}{g:16s}{s['photos']:7d}{fmt(s['missed_disease_rate']):>9s}{fmt(s['disease_found_rate']):>8s}"
                  f"{fmt(s['false_alarm_rate']):>13s}{fmt(s['not_sure_share']):>10s}")
    print("\nTop answers with tiles:", Counter(r["top"] for r in results["with_tiles"]).most_common(6))
    if args.out:
        args.out.write_text(json.dumps({"summary": report, "photos": results}, indent=1))
        print("saved", args.out)


if __name__ == "__main__":
    main()
