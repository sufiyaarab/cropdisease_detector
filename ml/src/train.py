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

Loss (per head, per photo i with true class y, probabilities p = softmax(logits)):
    L_i = w_i * [ (1 - p_y)^gamma * (-log p_y)  +  beta * [y is a disease] * (-log(1 - p_healthy)) ]
    w_i = class weight of y  *  (lambda if photo i is a field photo else 1)
  --field-weight lambda   field photos count lambda times as much as lab photos (default 3)
  --miss-penalty beta     extra loss when a diseased leaf leans "healthy" (default 1; 0 = off)
  --focal-gamma gamma     focal loss: down-weights photos the model already gets right (default 2; 0 = plain CE)
After training: temperature per head, then the healthy threshold tau_h (--max-missed, default 0.05) on validation.

Learning rate and overfitting:
  heads (stage 1):    AdamW at --lr-heads (1e-3); the rate halves when validation loss stalls for 3 epochs
                      (ReduceLROnPlateau, floor 1e-5); early stopping after 6 epochs without improvement.
  fine-tune (stage 2): AdamW, linear warmup from 0 to --lr-fine-tune (1e-4) over --warmup-epochs (1), then
                      cosine decay to 1% of it: lr_t = lr_max * (0.01 + 0.99 * 0.5 * (1 + cos(pi * t / T)));
                      early stopping after 2 epochs without improvement (best weights kept).
  --weight-decay (1e-4): AdamW's decoupled weight decay, w <- w - lr * wd * w each step, keeps weights small.
  Also in place: dropout 0.3 before every head, augmentation, BatchNorm frozen, only the top layers unfrozen.
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

from metrics import evaluate_head, is_field, matches, print_head, source_of

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


def list_split(data: Path, raw: dict, val=0.15, test=0.15, sources=None):
    """About 70/15/15 per label folder, split by photo group (see group_key).
    sources: only use these sources for train/val (test always keeps every source, so runs compare fairly).
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
            part = split_of(f, val, test)
            if part == "test" or sources is None or matches(source_of(f), sources):
                out[part].append((f, target))
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


def class_weights(items, head, n_classes):
    y = targets(items, head)
    y = y[y >= 0]
    counts = np.bincount(y, minlength=n_classes)
    return np.array([len(y) / (n_classes * max(c, 1)) for c in counts], dtype=np.float32)


def label_and_weight(items, head, cw, field_weight):
    """Targets as (label, weight) pairs, the format the loss below expects. label -1 = not this head's photo."""
    y = targets(items, head)
    w = np.where(y >= 0, cw[np.maximum(y, 0)], 0.0)
    w = w * np.array([field_weight if is_field(p) else 1.0 for p, _ in items])
    return np.stack([y, w], 1).astype(np.float32)


def make_loss(healthy_index, miss_penalty, focal_gamma):
    """Weighted focal cross-entropy plus a missed-disease penalty (see the top of this file).
    y_true is (batch, 2) = (label, weight); the result is the weighted mean over this head's photos."""

    def loss(y_true, logits):
        label = tf.cast(y_true[:, 0], tf.int32)
        w = y_true[:, 1] * tf.cast(label >= 0, tf.float32)
        safe = tf.maximum(label, 0)
        logp = tf.nn.log_softmax(logits)
        logp_y = tf.gather(logp, safe, axis=1, batch_dims=1)
        per = -logp_y
        if focal_gamma:
            per = per * tf.pow(1.0 - tf.exp(logp_y), focal_gamma)
        if healthy_index is not None and miss_penalty:
            # log(1 - p_healthy) = log sum of the disease probabilities, computed stably
            n = tf.shape(logits)[1]
            disease_mask = tf.one_hot(healthy_index, n, on_value=-1e9, off_value=0.0)
            log_sick = tf.reduce_logsumexp(logits + disease_mask, axis=1) - tf.reduce_logsumexp(logits, axis=1)
            per = per + miss_penalty * tf.cast(label != healthy_index, tf.float32) * (-log_sick)
        return tf.reduce_sum(per * w) / tf.maximum(tf.reduce_sum(w), 1e-6)

    return loss


def image_dataset(items, heads, cws, field_weight, aug, batch, shuffle):
    """Mixed photos. Each photo carries (label, weight) per head; label -1 = "not this head's photo" (ignored)."""
    paths = [p for p, _ in items]
    ys = {h: label_and_weight(items, h, cws[h], field_weight) for h in heads}
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


def fit_head(x, y, xv, yv, head, n_classes, args):
    """Stage 1 for one head, on cached backbone features. y / yv are (label, weight) pairs."""
    model = make_head(n_classes, head)
    model.compile(optimizer=keras.optimizers.AdamW(getattr(args, "lr_heads", 1e-3),
                                                   weight_decay=getattr(args, "weight_decay", 1e-4)),
                  loss=make_loss(None if head == "plant" else 0, args.miss_penalty, args.focal_gamma))
    model.fit(x, y, validation_data=(xv, yv), epochs=args.epochs, batch_size=128, verbose=0,
              callbacks=[keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-5),
                         keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)])
    return model


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
    ap.add_argument("--field-weight", type=float, default=3.0, help="lambda: weight of a field photo vs a lab photo")
    ap.add_argument("--miss-penalty", type=float, default=1.0, help="beta: extra loss when a diseased leaf leans healthy")
    ap.add_argument("--focal-gamma", type=float, default=2.0, help="gamma: focal loss (0 = plain cross-entropy)")
    ap.add_argument("--max-missed", type=float, default=0.05,
                    help="healthy threshold target: at most this share of diseased validation photos answered healthy")
    ap.add_argument("--threshold", type=float, default=0.70, help="the app's 'not sure' cut-off (CONFIDENCE_THRESHOLD)")
    ap.add_argument("--lr-heads", type=float, default=1e-3, help="stage 1 start rate (halves on plateaus)")
    ap.add_argument("--lr-fine-tune", type=float, default=1e-4, help="stage 2 peak rate (warmup, then cosine decay)")
    ap.add_argument("--warmup-epochs", type=float, default=1.0, help="stage 2 linear warmup length (0 = none)")
    ap.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW decoupled weight decay (0 = plain Adam)")
    ap.add_argument("--sources", nargs="+", default=None,
                    help="train only on these sources, e.g. plantvillage fieldplant plantdoc pp2021 (test keeps all)")
    args = ap.parse_args()

    keras.utils.set_random_seed(SEED)
    raw, heads = load_classes()
    split = list_split(args.data, raw, sources=args.sources)
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
    cws = {h: class_weights(tr, h, len(labels)) for h, labels in heads.items()}
    trained = {}
    for head, labels in heads.items():
        ytr = np.tile(label_and_weight(tr, head, cws[head], args.field_weight), (1 + args.aug_copies, 1))
        yva = label_and_weight(va, head, cws[head], args.field_weight)
        mtr, mva = ytr[:, 0] >= 0, yva[:, 0] >= 0
        trained[head] = fit_head(xtr[mtr], ytr[mtr], xva[mva], yva[mva], head, len(labels), args)
        acc = (trained[head].predict(xva[mva], verbose=0).argmax(1) == yva[mva, 0]).mean()
        print(f"[heads] {head}: val acc {acc:.4f}", flush=True)

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
        losses = {h: make_loss(None if h == "plant" else 0, args.miss_penalty, args.focal_gamma) for h in heads}
        steps = int(np.ceil(len(tr) / args.batch))
        warmup = min(int(args.warmup_epochs * steps), steps * args.fine_tune_epochs - 1)
        schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.0 if warmup else args.lr_fine_tune, warmup_target=args.lr_fine_tune if warmup else None,
            warmup_steps=warmup, decay_steps=max(steps * args.fine_tune_epochs - warmup, 1), alpha=0.01)
        full.compile(optimizer=keras.optimizers.AdamW(schedule, weight_decay=args.weight_decay), loss=losses)
        kw = dict(heads=heads, cws=cws, field_weight=args.field_weight, batch=args.batch)
        full.fit(image_dataset(tr, aug=True, shuffle=True, **kw),
                 validation_data=image_dataset(va, aug=False, shuffle=False, **kw),
                 epochs=args.fine_tune_epochs, verbose=2,
                 callbacks=[keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)])
        backbone.trainable = False

    # Calibrate on validation, score on test, using the exact model we export.
    lva, lte = all_logits(full, va), all_logits(full, split["test"])
    field_va = np.array([is_field(p) for p, _ in va])
    src_te = np.array([source_of(p) for p, _ in split["test"]])
    calibration, healthy_thr, metrics = {}, {}, {}
    for head, labels in heads.items():
        yva, yte = targets(va, head), targets(split["test"], head)
        mva, mte = yva >= 0, yte >= 0
        t, tau, m = evaluate_head(head, labels, lva[head][mva], yva[mva], field_va[mva],
                                  lte[head][mte], yte[mte], src_te[mte], args.threshold, args.max_missed)
        calibration[head] = t
        if tau is not None:
            healthy_thr[head] = tau
        m["n_train"], m["n_val"], m["n_test"] = int((targets(tr, head) >= 0).sum()), int(mva.sum()), int(mte.sum())
        metrics[head] = m
        print_head(head, m)

    full.save(args.out / "model.keras")
    export_tflite(full, args.out, sample=split["test"][0][0])

    sources = sorted({source_of(p) for s in split.values() for p, _ in s})
    (args.out / "classes.json").write_text(json.dumps({"plant": raw["plant"], "disease": raw["disease"]}, indent=2))
    (args.out / "calibration.json").write_text(json.dumps({"temperature": calibration, "healthy_threshold": healthy_thr,
                                                         "confidence_threshold": args.threshold}, indent=2))
    (args.out / "preprocess.json").write_text(json.dumps({
        "input_size": [IMG_SIZE, IMG_SIZE], "resize": "bilinear, no crop (squash to 224 x 224)",
        "color": "RGB", "input_range": [0, 255], "dtype": "float32",
        "mean": None, "std": None,
        "note": "EfficientNet-B0 in Keras rescales and normalises inside the model, so pass raw 0-255 pixels.",
        "outputs": "logits per head: plant / tomato / apple / corn; probabilities = softmax(logits / T[head])",
    }, indent=2))
    (args.out / "metrics.json").write_text(json.dumps({
        "version": args.version,
        "data_sources": {"plantvillage": "PlantVillage (lab photos)", "pp2021": "Plant Pathology 2021 (orchard photos)",
                         "other": "Roboflow Universe projects (field photos)"},
        "sources_used": sources,
        "split": "about 70/15/15 by photo (all leaves of one field photo in the same split), see splits.json",
        "warning": "PlantVillage test images are lab photos like the training ones, so overall scores are much higher "
                   "than field accuracy. Report app_rule_field_photos: precision, recall and missed-disease rate.",
        "decision_rule": "disease if p_top >= confidence_threshold; healthy only if p_healthy >= healthy_threshold; "
                         "otherwise not sure",
        "training": {"head_epochs": args.epochs, "fine_tune_epochs": args.fine_tune_epochs,
                     "lr_heads": args.lr_heads, "lr_fine_tune": args.lr_fine_tune,
                     "lr_schedule": "heads: ReduceLROnPlateau(0.5, 3); fine-tune: linear warmup + cosine decay",
                     "warmup_epochs": args.warmup_epochs, "weight_decay": args.weight_decay,
                     "field_weight": args.field_weight, "miss_penalty": args.miss_penalty,
                     "focal_gamma": args.focal_gamma, "max_missed": args.max_missed, "train_sources": args.sources,
                     "unfreeze_layers": args.unfreeze_layers if args.fine_tune_epochs else 0},
        "per_head": metrics,
    }, indent=2))
    print("saved to", args.out)


if __name__ == "__main__":
    main()
