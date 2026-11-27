"""Compare datasets and loss settings on the same test photos (heads only, frozen backbone: fine on a CPU).

Every run in ml/configs/experiments.json trains the four heads on cached EfficientNet-B0 features, using only
its own sources for training and validation, then:
  1. fits the temperature and the healthy threshold tau_h on its own validation photos,
  2. is scored for SELECTION on one shared set: the field photos of the validation split, from every source
     (macro-F2 of the disease heads under the app's rule; F2 counts recall more than precision),
  3. is scored for REPORTING on the shared test split (every source), plus each source on its own
     (cross-dataset: e.g. a run without PlantDoc, scored on PlantDoc).
Pick the run by the selection score, never by the test score, so the test number you present stays honest.
Then train that setting with fine-tuning on Colab (train.py takes the same flags).

    python ml/src/compare_datasets.py --data data --out ml/artifacts/compare
Writes results.md (tables for the report), results.csv and results.json. Backbone features are cached in
--out/features.npz, so a second run with other settings takes minutes.
"""
import argparse
import csv
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from metrics import decide, evaluate_head, is_field, matches, scores, softmax, source_of
from train import (
    ROOT,
    SEED,
    class_weights,
    features,
    fit_head,
    label_and_weight,
    list_split,
    load_classes,
    targets,
)


def cached_features(args, split):
    import keras

    cache = args.out / "features.npz"
    tr, va, te = split["train"], split["val"], split["test"]
    key = json.dumps([[p for p, _ in s] for s in (tr, va, te)] + [args.aug_copies])
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if str(z["key"]) == key:
            print("[features] from cache", flush=True)
            return z["xtr"], z["xva"], z["xte"]
    t0 = time.time()
    backbone = keras.applications.EfficientNetB0(include_top=False, weights="imagenet", pooling="avg",
                                                 input_shape=(224, 224, 3))
    xtr = np.stack([features(backbone, tr)] + [features(backbone, tr, aug=True) for _ in range(args.aug_copies)])
    xva, xte = features(backbone, va), features(backbone, te)
    np.savez(cache, key=key, xtr=xtr, xva=xva, xte=xte)
    print(f"[features] {xtr.shape[0] * xtr.shape[1] + len(xva) + len(xte)} done ({time.time() - t0:.0f}s)", flush=True)
    return xtr, xva, xte


def run_one(run, heads, split, xtr, xva, xte, args):
    import keras

    keras.utils.set_random_seed(SEED)
    tr, va, te = split["train"], split["val"], split["test"]
    wanted = run["sources"]
    in_tr = np.array([wanted == "all" or matches(source_of(p), wanted) for p, _ in tr])
    in_va = np.array([wanted == "all" or matches(source_of(p), wanted) for p, _ in va])
    tr_r = [x for x, k in zip(tr, in_tr) if k]
    va_r = [x for x, k in zip(va, in_va) if k]
    cfg = SimpleNamespace(epochs=args.epochs, miss_penalty=run["miss_penalty"], focal_gamma=run["focal_gamma"],
                          lr_heads=run.get("lr_heads", 1e-3), weight_decay=run.get("weight_decay", 1e-4))
    field_va_all = np.array([is_field(p) for p, _ in va])
    src_te = np.array([source_of(p) for p, _ in te])
    out = {"run": run, "n_train": len(tr_r), "heads": {}}
    for head, labels in heads.items():
        cw = class_weights(tr_r, head, len(labels))
        ytr = np.tile(label_and_weight(tr_r, head, cw, run["field_weight"]), (1 + args.aug_copies, 1))
        yva = label_and_weight(va_r, head, cw, run["field_weight"])
        x = np.concatenate([xtr[c][in_tr] for c in range(xtr.shape[0])])
        mtr, mva = ytr[:, 0] >= 0, yva[:, 0] >= 0
        if len(set(ytr[mtr, 0].astype(int))) < 2:
            continue
        model = fit_head(x[mtr], ytr[mtr], xva[in_va][mva], yva[mva], head, len(labels), cfg)
        y_va, y_te, y_all = targets(va_r, head), targets(te, head), targets(va, head)
        m_va, m_te, m_all = y_va >= 0, y_te >= 0, y_all >= 0
        l_va = model.predict(xva[in_va][m_va], verbose=0)
        l_te = model.predict(xte[m_te], verbose=0)
        field_va = np.array([is_field(p) for p, _ in va_r])[m_va]
        t, tau, m = evaluate_head(head, labels, l_va, y_va[m_va], field_va, l_te, y_te[m_te], src_te[m_te],
                                  args.threshold, args.max_missed)
        # Selection score: the shared field-validation photos (all sources), same rule as the app.
        sel = m_all & field_va_all
        if sel.any():
            p_sel = softmax(model.predict(xva[sel], verbose=0) / t)
            m["selection_field_val"] = scores(y_all[sel], decide(p_sel, args.threshold, tau if head != "plant"
                                                                 else args.threshold), labels, head != "plant")
        out["heads"][head] = m
    return out


def fmt(v):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"


def summarise(results, heads, args):
    crops = [h for h in heads if h != "plant"]
    rows = []
    for r in results:
        row = {"run": r["run"]["name"], "lambda": r["run"]["field_weight"], "beta": r["run"]["miss_penalty"],
               "gamma": r["run"]["focal_gamma"], "wd": r["run"].get("weight_decay", 1e-4), "n_train": r["n_train"]}
        sel = [r["heads"][c]["selection_field_val"]["macro_f2"] for c in crops
               if c in r["heads"] and "selection_field_val" in r["heads"][c]]
        row["selection_val_field_macro_f2"] = round(float(np.mean(sel)), 4) if len(sel) == len(crops) else None
        for c in crops:
            m = r["heads"].get(c, {})
            f = m.get("app_rule_field_photos") or m.get("app_rule_all_photos") or {}
            row[f"{c}_P"], row[f"{c}_R"] = f.get("macro_precision"), f.get("macro_recall")
            row[f"{c}_F1"], row[f"{c}_missed"] = f.get("macro_f1"), f.get("missed_disease_rate")
            row[f"{c}_false_alarm"], row[f"{c}_tau_h"] = f.get("false_alarm_rate"), m.get("healthy_threshold")
        for s in sorted({s for r2 in results for c in crops for s in r2["heads"].get(c, {}).get("app_rule_by_source", {})}):
            vals = [r["heads"][c]["app_rule_by_source"][s]["macro_f1"] for c in crops
                    if s in r["heads"].get(c, {}).get("app_rule_by_source", {})]
            row[f"test_{s}_macro_f1"] = round(float(np.mean(vals)), 4) if vals else None
        rows.append(row)

    keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(args.out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)

    best = max((r for r in rows if r["selection_val_field_macro_f2"] is not None),
               key=lambda r: r["selection_val_field_macro_f2"], default=None)
    lines = ["# Dataset and loss comparison (frozen backbone, heads only)", "",
             f"Same test photos for every run. App rule: disease if p >= {args.threshold}; healthy only if "
             f"p_healthy >= tau_h, tuned on each run's validation photos for at most {args.max_missed:.0%} missed diseases.",
             "Selection score = mean macro-F2 of the disease heads on the shared field validation photos "
             "(choose by this, not by the test columns).", "",
             "## Field test photos, per crop (macro precision / recall / missed-disease rate / false-alarm rate)", "",
             "| Run | λ | β | γ | wd | Selection F2 | " + " | ".join(f"{c} P / R / missed / false alarm" for c in crops) + " |",
             "|" + "---|" * (6 + len(crops))]
    for r in rows:
        cells = [f"{fmt(r[f'{c}_P'])} / {fmt(r[f'{c}_R'])} / {fmt(r[f'{c}_missed'])} / {fmt(r[f'{c}_false_alarm'])}"
                 for c in crops]
        name = f"**{r['run']}**" if best is r else r["run"]
        lines.append(f"| {name} | {r['lambda']} | {r['beta']} | {r['gamma']} | {r['wd']:g} | {fmt(r['selection_val_field_macro_f2'])} | "
                     + " | ".join(cells) + " |")
    src_cols = [k for k in keys if k.startswith("test_")]
    lines += ["", "## Cross-dataset: test macro-F1 per source (mean of the three crops)", "",
              "| Run | " + " | ".join(k[5:-9] for k in src_cols) + " |", "|" + "---|" * (1 + len(src_cols))]
    for r in rows:
        lines.append(f"| {r['run']} | " + " | ".join(fmt(r.get(k)) for k in src_cols) + " |")
    if best:
        lines += ["", f"Best by selection score: **{best['run']}** (λ={best['lambda']}, β={best['beta']}, "
                      f"γ={best['gamma']}, weight decay={best['wd']:g}). Train it with fine-tuning: see ml/README.md."]
    (args.out / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("ml/artifacts/compare"))
    ap.add_argument("--experiments", type=Path, default=ROOT / "configs/experiments.json")
    ap.add_argument("--only", nargs="+", default=None, help="run only these run names (prefix match, e.g. A1 B4)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--aug-copies", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--max-missed", type=float, default=0.05)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    raw, heads = load_classes()
    split = list_split(args.data, raw)
    present = {source_of(p) for s in split.values() for p, _ in s}
    print("sources in data:", sorted(present), {k: len(v) for k, v in split.items()}, flush=True)
    xtr, xva, xte = cached_features(args, split)

    runs = json.loads(args.experiments.read_text())["runs"]
    if args.only:
        runs = [r for r in runs if any(r["name"].startswith(o) for o in args.only)]
    results = []
    for run in runs:
        if run["sources"] != "all":
            missing = [s for s in run["sources"] if not any(matches(p, [s]) for p in present)]
            if missing:
                print(f"[skip] {run['name']}: no photos from {missing}", flush=True)
                continue
        t0 = time.time()
        r = run_one(run, heads, split, xtr, xva, xte, args)
        results.append(r)
        sel = [r["heads"][c].get("selection_field_val", {}).get("macro_f2") for c in heads if c != "plant" and c in r["heads"]]
        print(f"[run] {run['name']}: selection F2 {np.mean([s for s in sel if s is not None] or [np.nan]):.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=1, default=float))
    summarise(results, heads, args)
    print("saved to", args.out)


if __name__ == "__main__":
    main()
