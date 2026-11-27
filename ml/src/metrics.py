"""Scoring, calibration and the healthy threshold (numpy only, shared by train.py and compare_datasets.py).

The decision rule the app uses (backend/app/pipeline/diagnose.py), with p = softmax(logits / T):
  - top class is a disease and p_top >= 0.70           -> that disease
  - top class is healthy and p_healthy >= tau_h         -> healthy
  - anything else                                       -> "not sure" (ranked candidates + expert referral)
tau_h >= 0.70 is chosen on validation photos so that at most `max_missed` of diseased photos get a
confident "healthy" answer. A "not sure" is not a miss: the user is sent to a closer look or an expert.

Words used below, per crop:
  missed disease = a diseased leaf answered "healthy"         (false negative for "sick")
  false alarm    = a healthy leaf NOT answered "healthy"      (false positive for "sick", includes "not sure")
"""
from pathlib import Path

import numpy as np

NOT_SURE = -1
HEALTHY = 0  # index of <crop>_healthy in every disease head (configs/classes.json)


# ---------- photo sources ----------
def source_of(path: str) -> str:
    """plantvillage, pp2021, or the Roboflow project tag (fieldplant, plantdoc-dxufs, ...), from the file name."""
    name = Path(path).name
    if name.startswith("pv_"):
        return "plantvillage"
    if name.startswith("rf_"):
        return name[3:].split("__", 1)[0]
    return name.split("_", 1)[0]


def is_field(path: str) -> bool:
    """Everything that is not a PlantVillage lab photo counts as a field photo."""
    return source_of(path) != "plantvillage"


def matches(source: str, wanted) -> bool:
    """--sources fieldplant also matches the tag fieldplant; plantdoc matches plantdoc-dxufs."""
    return any(source == w or source.startswith(w) for w in wanted)


# ---------- calibration ----------
def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(logits, y):
    """Grid search for the temperature that minimises negative log-likelihood on validation."""
    best_t, best_nll = 1.0, np.inf
    for t in np.arange(0.25, 5.01, 0.05):
        logp = np.log(softmax(logits / t) + 1e-12)
        nll = -logp[np.arange(len(y)), y].mean()
        if nll < best_nll:
            best_t, best_nll = float(t), float(nll)
    return round(best_t, 2)


def ece(probs, y, bins=10):
    conf, pred = probs.max(1), probs.argmax(1)
    total = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


# ---------- decision rule ----------
def decide(probs, threshold=0.70, healthy_threshold=None):
    """Final answer per photo: class index, or NOT_SURE. healthy_threshold=None means plain argmax (no rule)."""
    pred, conf = probs.argmax(1), probs.max(1)
    if healthy_threshold is None:
        return pred
    need = np.where(pred == HEALTHY, max(threshold, healthy_threshold), threshold)
    return np.where(conf >= need, pred, NOT_SURE)


def missed_rate(y, final):
    sick = y != HEALTHY
    return float((final[sick] == HEALTHY).mean()) if sick.any() else float("nan")


def false_alarm_rate(y, final):
    ok = y == HEALTHY
    return float((final[ok] != HEALTHY).mean()) if ok.any() else float("nan")


def tune_healthy_threshold(probs, y, threshold=0.70, max_missed=0.05):
    """Smallest tau_h in [threshold, 0.99] with missed-disease rate <= max_missed (fewest false alarms that
    still meets the target). Returns (tau_h, sweep) where sweep lists missed / false alarm for every tau."""
    sweep = []
    for t in np.round(np.arange(threshold, 0.995, 0.01), 2):
        final = decide(probs, threshold, t)
        sweep.append({"tau_h": float(t), "missed_disease": round(missed_rate(y, final), 4),
                      "false_alarm": round(false_alarm_rate(y, final), 4)})
    ok = [s for s in sweep if s["missed_disease"] <= max_missed]
    return (ok[0]["tau_h"] if ok else sweep[-1]["tau_h"]), sweep


# ---------- scores ----------
def f_beta(p, r, beta):
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r) if p + r else 0.0


def scores(y, final, labels, crop_head=True):
    """Per-class precision / recall / F1 / F2 and their macro averages. NOT_SURE answers count against recall
    (the class was not found) but not against precision (no class was claimed)."""
    per = {}
    for i, lab in enumerate(labels):
        tp = int(((final == i) & (y == i)).sum())
        fp = int(((final == i) & (y != i)).sum())
        fn = int(((final != i) & (y == i)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        per[lab] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f_beta(p, r, 1), 4),
                    "f2": round(f_beta(p, r, 2), 4), "support": int((y == i).sum())}
    present = [v for v in per.values() if v["support"]] or list(per.values())
    out = {
        "n": len(y),
        "accuracy": round(float((final == y).mean()), 4) if len(y) else 0.0,
        "macro_precision": round(float(np.mean([v["precision"] for v in present])), 4),
        "macro_recall": round(float(np.mean([v["recall"] for v in present])), 4),
        "macro_f1": round(float(np.mean([v["f1"] for v in present])), 4),
        "macro_f2": round(float(np.mean([v["f2"] for v in present])), 4),
        "not_sure_share": round(float((final == NOT_SURE).mean()), 4) if len(y) else 0.0,
        "per_class": per,
    }
    if crop_head:
        out["missed_disease_rate"] = round(missed_rate(y, final), 4)
        out["false_alarm_rate"] = round(false_alarm_rate(y, final), 4)
    return out


def confusion(y, final, labels):
    """Rows = true class, columns = predicted class, last column = not sure."""
    cols = list(range(len(labels))) + [NOT_SURE]
    return {"labels": labels + ["not_sure"],
            "rows_true_cols_pred": [[int(((y == a) & (final == b)).sum()) for b in cols] for a in range(len(labels))]}


def evaluate_head(head, labels, lva, yva, field_va, lte, yte, src_te, threshold=0.70, max_missed=0.05):
    """Calibrate on validation, pick tau_h on validation, then score the test photos.

    lva/lte: logits, yva/yte: true index, field_va: bool per val photo, src_te: source name per test photo.
    Returns (temperature, tau_h or None, metrics dict)."""
    crop_head = head != "plant"
    t = fit_temperature(lva, yva)
    pva, pte = softmax(lva / t), softmax(lte / t)
    tau, sweep = None, None
    if crop_head:
        # Tune on field validation photos when there are enough diseased ones; else on all validation photos.
        use = field_va if ((yva != HEALTHY) & field_va).sum() >= 20 else np.ones_like(field_va)
        tau, sweep = tune_healthy_threshold(pva[use], yva[use], threshold, max_missed)
        sweep = {"tuned_on": "field validation photos" if use is field_va else "all validation photos",
                 "rows": sweep}
    plain = decide(pte, threshold, None)
    final = decide(pte, threshold, tau if crop_head else threshold)
    field_te = np.array([s != "plantvillage" for s in src_te], dtype=bool)
    m = {
        "temperature": t,
        "healthy_threshold": tau,
        "ece_before_calibration": round(ece(softmax(lte), yte), 4),
        "ece_after_calibration": round(ece(pte, yte), 4),
        "argmax_all_photos": scores(yte, plain, labels, crop_head),
        "app_rule_all_photos": scores(yte, final, labels, crop_head),
        "app_rule_confusion": confusion(yte, final, labels),
        "healthy_threshold_sweep_val": sweep,
    }
    if field_te.any():
        m["argmax_field_photos"] = scores(yte[field_te], plain[field_te], labels, crop_head)
        m["app_rule_field_photos"] = scores(yte[field_te], final[field_te], labels, crop_head)
        m["app_rule_field_confusion"] = confusion(yte[field_te], final[field_te], labels)
    m["app_rule_by_source"] = {s: scores(yte[src_te == s], final[src_te == s], labels, crop_head)
                               for s in sorted(set(src_te.tolist()))}
    return t, tau, m


def print_head(head, m):
    key = "app_rule_field_photos" if "app_rule_field_photos" in m else "app_rule_all_photos"
    s = m[key]
    extra = (f" missed {s['missed_disease_rate']:.3f} false-alarm {s['false_alarm_rate']:.3f} tau_h {m['healthy_threshold']}"
             if "missed_disease_rate" in s else "")
    print(f"[test] {head} ({key}, n={s['n']}): P {s['macro_precision']:.3f} R {s['macro_recall']:.3f} "
          f"F1 {s['macro_f1']:.3f} F2 {s['macro_f2']:.3f} not-sure {s['not_sure_share']:.3f}{extra} T={m['temperature']}")
    for lab, v in s["per_class"].items():
        print(f"         {lab:28s} P {v['precision']:.3f}  R {v['recall']:.3f}  F1 {v['f1']:.3f}  n={v['support']}")
