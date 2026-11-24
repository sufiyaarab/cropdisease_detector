"""Train the crop disease classifier (Tier 1 engine, Report 2 final concept).

Model: one EfficientNet-B0 backbone (ImageNet weights) shared by four heads:
  - plant:  which plant is this? tomato / apple / corn / other_plant
  - tomato, apple, corn: which disease, given the plant (configs/classes.json order)
Every head outputs logits; the backend applies softmax(logits / T) with T per head.

Stage 1 (fast, fine on CPU): the backbone stays frozen, we cache its features once
(plus augmented copies of the training images) and train only the small heads.
Stage 2 (optional, use a GPU such as Colab): --fine-tune-epochs N also trains the top
of the backbone, end to end on augmented photos.

Data layout (made by prepare_plantvillage.py / download_roboflow.py):
    data/<label>/*.jpg     e.g. data/tomato_late_blight/pv_0001.jpg, data/other_plant/pv_pepper_0001.jpg
Every disease photo also teaches the plant head; other_plant photos only teach the plant head.

Run:
    python ml/src/train.py --data data/ --out ml/artifacts/v0.1 --version v0.1                       # CPU
    python ml/src/train.py --data data/ --out ml/artifacts/v0.1 --version v0.1 --fine-tune-epochs 8  # GPU
Outputs (contracts/model_io.md): model.tflite, model.keras, classes.json,
calibration.json, preprocess.json, metrics.json, splits.json.
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras

IMG_SIZE = 224
SEED = 42
ROOT = Path(__file__).resolve().parents[1]
IMG_EXT = {".jpg", ".jpeg", ".png"}
OTHER = "other_plant"


def load_classes():
    raw = json.loads((ROOT / "configs/classes.json").read_text())
    heads = {"plant": raw["plant"], **raw["disease"]}
    assert raw["plant"][:-1] == list(raw["disease"]) and raw["plant"][-1] == OTHER, "plant list must be crops + other_plant"
    return raw, heads


def group_key(path: str) -> str:
    """Photos that must stay in the same split. Roboflow leaf crops are named rf_<dataset>__<photo>__<n>.jpg,
    so all leaves of one field photo share a key. Anything else is its own group."""
    name = Path(path).stem
    return name.rsplit("__", 1)[0] if name.startswith("rf_") and name.count("__") >= 2 else name


def split_of(path: str, val: float, test: float) -> str:
    # Deterministic: a photo lands in the same split every run and across label folders.
    h = int(hashlib.md5(group_key(path).encode()).hexdigest()[:8], 16) / 2**32
    return "test" if h < test else "val" if h < test + val else "train"


def list_split(data: Path, raw: dict, val=0.15, test=0.15):
    """About 70/15/15 per label folder, split by photo group (see group_key).
    Returns {split: [(path, {head: target or -1}), ...]}."""
    heads = ["plant", *raw["disease"]]
    folders = [(lab, crop, i) for crop, labels in raw["disease"].items() for i, lab in enumerate(labels)]
    folders.append((OTHER, None, None))
    out = {"train": [], "val": [], "test": []}
    for label, crop, idx in folders:
        files = sorted(str(p) for p in (data / label).glob("*") if p.suffix.lower() in IMG_EXT)
        if not files:
            raise SystemExit(f"No images for {label} in {data / label}")
        target = {h: -1 for h in heads}
        target["plant"] = raw["plant"].index(crop) if crop else raw["plant"].index(OTHER)
        if crop:
            target[crop] = idx
        for f in files:
            out[split_of(f, val, test)].append((f, target))
    rng = random.Random(SEED)
    for items in out.values():
        rng.shuffle(items)
    return out


def targets(items, head):
    return np.array([t[head] for _, t in items])


def load_image(path):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    # Same as the backend: plain resize to 224 x 224, pixel values 0-255 (the model rescales inside).
    return tf.image.resize(tf.cast(img, tf.float32), (IMG_SIZE, IMG_SIZE))


def augment(img):
    # Rough stand-in for field conditions: framing, lighting, orientation.
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    scale = tf.random.uniform([], 0.75, 1.0)
    side = tf.cast(scale * IMG_SIZE, tf.int32)
    img = tf.image.random_crop(img, (side, side, 3))
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.image.random_brightness(img, 40.0)
    img = tf.image.random_contrast(img, 0.7, 1.3)
    img = tf.image.random_saturation(img, 0.7, 1.3)
    return tf.clip_by_value(img, 0.0, 255.0)


def features(backbone, items, aug=False, batch=64):
    paths = [p for p, _ in items]  # items are (path, targets)
    ds = tf.data.Dataset.from_tensor_slices(paths).map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if aug:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return backbone.predict(ds, verbose=0)


def make_head(n_classes, name):
    inp = keras.Input((1280,))
    x = keras.layers.Dropout(0.3)(inp)
    out = keras.layers.Dense(n_classes, name=f"{name}_logits")(x)
    return keras.Model(inp, out, name=f"{name}_head")


def fit_temperature(logits, y):
    """Grid search for the temperature that minimises negative log-likelihood on validation."""
    best_t, best_nll = 1.0, np.inf
    for t in np.arange(0.25, 5.01, 0.05):
        z = logits / t
        z = z - z.max(1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(1, keepdims=True))
        nll = -logp[np.arange(len(y)), y].mean()
        if nll < best_nll:
            best_t, best_nll = float(t), float(nll)
    return round(best_t, 2)


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ece(probs, y, bins=10):
    conf, pred = probs.max(1), probs.argmax(1)
    total = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


def scores(y, pred, labels):
    per = {}
    for i, lab in enumerate(labels):
        tp = int(((pred == i) & (y == i)).sum())
        fp = int(((pred == i) & (y != i)).sum())
        fn = int(((pred != i) & (y == i)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        per[lab] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4), "support": int((y == i).sum())}
    cm = [[int(((y == a) & (pred == b)).sum()) for b in range(len(labels))] for a in range(len(labels))]
    return {
        "accuracy": round(float((pred == y).mean()), 4),
        "macro_f1": round(float(np.mean([v["f1"] for v in per.values()])), 4),
        "per_class": per,
        "confusion_matrix": {"labels": labels, "rows_true_cols_pred": cm},
    }


def image_dataset(items, heads, aug, batch, shuffle):
    """Mixed photos. Each photo carries a target per head; -1 means "not this head's photo" (ignored)."""
    paths = [p for p, _ in items]
    ys = {h: targets(items, h).astype(np.int32) for h in heads}
    ds = tf.data.Dataset.from_tensor_slices((paths, ys))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)

    def load(path, y):
        img = load_image(path)
        return (augment(img) if aug else img), y

    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE)


def all_logits(model, items, batch=64):
    ds = tf.data.Dataset.from_tensor_slices([p for p, _ in items]).map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    out = model.predict(ds.batch(batch).prefetch(tf.data.AUTOTUNE), verbose=0)
    return {h: np.asarray(v) for h, v in out.items()}


def export_tflite(model, out: Path, sample: str):
    """SavedModel -> TFLite (float16 weights, about 8 MB), then check it matches the Keras model."""
    import shutil
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    archive = keras.export.ExportArchive()
    archive.track(model)
    archive.add_endpoint("serving_default", lambda image: model(image, training=False),
                         input_signature=[tf.TensorSpec([1, IMG_SIZE, IMG_SIZE, 3], tf.float32, name="image")])
    archive.write_out(str(tmp / "saved_model"))
    conv = tf.lite.TFLiteConverter.from_saved_model(str(tmp / "saved_model"))
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    (out / "model.tflite").write_bytes(conv.convert())
    shutil.rmtree(tmp, ignore_errors=True)

    from ai_edge_litert.interpreter import Interpreter

    x = load_image(sample).numpy()[None]
    runner = Interpreter(model_path=str(out / "model.tflite")).get_signature_runner()
    lite = runner(image=x)
    ref = model.predict(x, verbose=0)
    worst = max(float(np.abs(np.asarray(lite[c]) - np.asarray(ref[c])).max()) for c in ref)
    if not np.isfinite(worst) or worst > 0.1:
        raise SystemExit(f"TFLite export does not match the Keras model (max logit difference {worst})")
    print(f"model.tflite OK ({(out / 'model.tflite').stat().st_size / 1e6:.1f} MB, max logit diff {worst:.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--version", default="v0.1")
    ap.add_argument("--epochs", type=int, default=40, help="head-only epochs (cached features, fast on CPU)")
    ap.add_argument("--aug-copies", type=int, default=1, help="augmented copies of each training image for the head stage")
    ap.add_argument("--fine-tune-epochs", type=int, default=0,
                    help="then train the top of the backbone too. Use a GPU (Colab): try 8")
    ap.add_argument("--unfreeze-layers", type=int, default=60, help="how many backbone layers to unfreeze when fine-tuning")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    keras.utils.set_random_seed(SEED)
    raw, heads = load_classes()
    split = list_split(args.data, raw)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "splits.json").write_text(json.dumps(split))
    print("GPUs:", tf.config.list_physical_devices("GPU") or "none (CPU)", flush=True)
    print({k: len(v) for k, v in split.items()}, "photos", flush=True)

    backbone = keras.applications.EfficientNetB0(include_top=False, weights="imagenet", pooling="avg",
                                                 input_shape=(IMG_SIZE, IMG_SIZE, 3))
    backbone.trainable = False

    # Stage 1: heads only, on backbone features cached once for every photo.
    t0 = time.time()
    tr, va = split["train"], split["val"]
    xtr = np.concatenate([features(backbone, tr)] + [features(backbone, tr, aug=True) for _ in range(args.aug_copies)])
    xva = features(backbone, va)
    print(f"[features] {len(xtr) + len(xva)} done ({time.time() - t0:.0f}s)", flush=True)
    trained = {}
    for head, labels in heads.items():
        ytr = np.tile(targets(tr, head), 1 + args.aug_copies)
        yva = targets(va, head)
        mtr, mva = ytr >= 0, yva >= 0
        counts = np.bincount(ytr[mtr], minlength=len(labels))
        class_weight = {i: float(mtr.sum() / (len(labels) * max(c, 1))) for i, c in enumerate(counts)}
        model = make_head(len(labels), head)
        model.compile(optimizer=keras.optimizers.Adam(1e-3),
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
        model.fit(xtr[mtr], ytr[mtr], validation_data=(xva[mva], yva[mva]), epochs=args.epochs, batch_size=128,
                  class_weight=class_weight, verbose=0,
                  callbacks=[keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)])
        trained[head] = model
        print(f"[heads] {head}: val acc {model.evaluate(xva[mva], yva[mva], verbose=0)[1]:.4f}", flush=True)

    # One model, four named outputs (logits).
    inp = keras.Input((IMG_SIZE, IMG_SIZE, 3), name="image")
    feat = backbone(inp, training=False)  # training=False keeps BatchNorm frozen, also while fine-tuning
    feat = keras.layers.Dropout(0.3)(feat)
    outs = {h: keras.layers.Dense(len(labels), name=h)(feat) for h, labels in heads.items()}
    full = keras.Model(inp, outs, name="crop_disease")
    for h in heads:
        full.get_layer(h).set_weights(trained[h].get_layer(f"{h}_logits").get_weights())

    # Stage 2 (optional, GPU): unfreeze the top of the backbone and train end to end at a low learning rate.
    if args.fine_tune_epochs > 0:
        backbone.trainable = True
        for layer in backbone.layers[:-args.unfreeze_layers]:
            layer.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = False
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=-1)
        full.compile(optimizer=keras.optimizers.Adam(1e-4), loss={h: loss for h in heads})
        full.fit(image_dataset(tr, heads, aug=True, batch=args.batch, shuffle=True),
                 validation_data=image_dataset(va, heads, aug=False, batch=args.batch, shuffle=False),
                 epochs=args.fine_tune_epochs, verbose=2,
                 callbacks=[keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)])
        backbone.trainable = False

    # Calibrate on validation, score on test, using the exact model we export.
    lva, lte = all_logits(full, va), all_logits(full, split["test"])
    field = np.array([Path(p).name.startswith("rf_") for p, _ in split["test"]])
    calibration, metrics = {}, {}
    for head, labels in heads.items():
        yva, yte = targets(va, head), targets(split["test"], head)
        mva, mte = yva >= 0, yte >= 0
        t = fit_temperature(lva[head][mva], yva[mva])
        calibration[head] = t
        y, logits = yte[mte], lte[head][mte]
        p = softmax(logits / t)
        m = scores(y, p.argmax(1), labels)
        m["ece_before_calibration"] = round(ece(softmax(logits), y), 4)
        m["ece_after_calibration"] = round(ece(p, y), 4)
        m["share_below_0.70_confidence"] = round(float((p.max(1) < 0.70).mean()), 4)
        m["n_train"], m["n_val"], m["n_test"] = int((targets(tr, head) >= 0).sum()), int(mva.sum()), int(mte.sum())
        # Field photos (Roboflow, file names start rf_) scored on their own: the number that matters for the MVP.
        f = field[mte]
        if f.any():
            fm = scores(y[f], p[f].argmax(1), labels)
            m["field_photos_only"] = {"n": int(f.sum()), "accuracy": fm["accuracy"], "macro_f1": fm["macro_f1"]}
        metrics[head] = m
        print(f"[test] {head}: acc {m['accuracy']} macro-F1 {m['macro_f1']} T={t}"
              + (f" | field photos macro-F1 {m['field_photos_only']['macro_f1']} (n={m['field_photos_only']['n']})"
                 if f.any() else ""), flush=True)

    full.save(args.out / "model.keras")
    export_tflite(full, args.out, sample=split["test"][0][0])

    sources = sorted({Path(p).name.split("_")[0] for s in split.values() for p, _ in s})
    (args.out / "classes.json").write_text(json.dumps({"plant": raw["plant"], "disease": raw["disease"]}, indent=2))
    (args.out / "calibration.json").write_text(json.dumps({"temperature": calibration}, indent=2))
    (args.out / "preprocess.json").write_text(json.dumps({
        "input_size": [IMG_SIZE, IMG_SIZE], "resize": "bilinear, no crop (squash to 224 x 224)",
        "color": "RGB", "input_range": [0, 255], "dtype": "float32",
        "mean": None, "std": None,
        "note": "EfficientNet-B0 in Keras rescales and normalises inside the model, so pass raw 0-255 pixels.",
        "outputs": "logits per head: plant / tomato / apple / corn; probabilities = softmax(logits / T[head])",
    }, indent=2))
    (args.out / "metrics.json").write_text(json.dumps({
        "version": args.version,
        "data_sources": {"pv": "PlantVillage (lab photos)", "rf": "Roboflow Universe (field photos)"},
        "sources_used": sources,
        "split": "about 70/15/15 by photo (all leaves of one field photo in the same split), see splits.json",
        "warning": "PlantVillage test images are lab photos like the training ones, so overall scores are much higher "
                   "than field accuracy. Trust field_photos_only when it is there.",
        "training": {"head_epochs": args.epochs, "fine_tune_epochs": args.fine_tune_epochs,
                     "unfreeze_layers": args.unfreeze_layers if args.fine_tune_epochs else 0},
        "per_head": metrics,
    }, indent=2))
    print("saved to", args.out)


if __name__ == "__main__":
    main()
