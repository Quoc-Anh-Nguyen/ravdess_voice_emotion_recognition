"""Average out-of-fold (OOF) softmax probabilities from one or more
train_ssl.py runs — the highest-leverage way past the single-model plateau.

Each train_ssl.py / train_rcnn.py run writes  reports/<kind>/oof_probs_<tag>.npz
holding the full (N, 8) speaker-independent OOF probability matrix.  Because
GroupKFold is DETERMINISTIC (no shuffle), every run uses the identical test
partition, so the matrices align row-by-row and can be averaged directly —
including the from-scratch RCNN baseline alongside the SSL backbones.

Why combine DIFFERENT backbones (wavlm-large + wav2vec2-large + hubert-large):
their pretraining objectives and data differ, so their mistakes are partly
DECORRELATED.  Averaging cancels independent errors — exactly where a single
model is unsure (e.g. the sad <-> calm low-arousal pair).  This typically buys
more than another seed of the same backbone.

Usage:
  python scripts/ensemble_ssl.py                  # average ALL oof_probs_*.npz
  python scripts/ensemble_ssl.py A.npz B.npz      # average a specific subset
  python scripts/ensemble_ssl.py --weights 2,1 A.npz B.npz   # weighted average
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, recall_score)

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "reports", "ssl")
EMOTION_ORDER = ["neutral", "calm", "happy", "sad",
                 "angry", "fearful", "disgust", "surprised"]
N_CLS = len(EMOTION_ORDER)


def parse_args(argv):
    weights, files = None, []
    i = 0
    while i < len(argv):
        if argv[i] in ("--weights", "-w"):
            weights = [float(x) for x in argv[i + 1].split(",")]
            i += 2
        else:
            files.append(argv[i])
            i += 1
    if not files:
        # scan every reports/<kind>/ pool (ssl backbones + rcnn baseline)
        files = sorted(glob.glob(os.path.join(ROOT, "reports", "*",
                                              "oof_probs_*.npz")))
    return files, weights


def metric_pack(gts, preds):
    return {
        "acc":         float(accuracy_score(gts, preds)),
        "uar":         float(recall_score(gts, preds, average="macro", zero_division=0)),
        "macro_f1":    float(f1_score(gts, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(gts, preds, average="weighted", zero_division=0)),
    }


def text_confusion(cm):
    head = "          " + " ".join(f"{e[:4]:>5}" for e in EMOTION_ORDER)
    rows = [f"{e[:9]:>9} " + " ".join(f"{cm[i, j]:5d}" for j in range(N_CLS))
            for i, e in enumerate(EMOTION_ORDER)]
    return "\n".join([head] + rows)


def main():
    files, weights = parse_args(sys.argv[1:])
    if not files:
        print("No oof_probs_*.npz in reports/ssl/.  Run train_ssl.py first "
              "(it now writes one per run).")
        return 1

    runs = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        runs.append({"file": os.path.basename(f), "model": str(d["model"]),
                     "probs": d["probs"], "labels": d["labels"],
                     "filled": d["filled"], "macro_f1": float(d["macro_f1"])})

    base = runs[0]
    for r in runs[1:]:                       # splits must match to align rows
        if not np.array_equal(r["labels"], base["labels"]):
            print(f"ERROR: {r['file']} has a different label vector — "
                  "incompatible CV split (different --folds?).")
            return 1
        if not np.array_equal(r["filled"], base["filled"]):
            print(f"ERROR: {r['file']} has a different filled mask.")
            return 1

    if weights is not None and len(weights) != len(runs):
        print(f"ERROR: --weights has {len(weights)} values but {len(runs)} runs.")
        return 1
    w = np.ones(len(runs)) if weights is None else np.array(weights, dtype=float)
    w = w / w.sum()

    mask  = base["filled"]
    gts   = base["labels"][mask]
    stack = np.stack([r["probs"] for r in runs], axis=0)        # (R, N, C)
    ens   = np.tensordot(w, stack, axes=(0, 0))[mask]           # weighted mean
    preds = ens.argmax(-1)
    m     = metric_pack(gts, preds)
    best  = max(r["macro_f1"] for r in runs)

    print("=" * 66)
    print(f"ENSEMBLE of {len(runs)} run(s)   (n={int(mask.sum())})")
    print("-" * 66)
    for r, wi in zip(runs, w):
        print(f"  w={wi:.2f}  single_macroF1={r['macro_f1']:.4f}  "
              f"{r['model']:<26} {r['file']}")
    print("-" * 66)
    print(f"  ENSEMBLE  acc={m['acc']:.4f}  uar={m['uar']:.4f}  "
          f"macroF1={m['macro_f1']:.4f}  weightedF1={m['weighted_f1']:.4f}")
    print(f"  best single = {best:.4f}    ensemble gain = {m['macro_f1'] - best:+.4f}")

    pcf = f1_score(gts, preds, average=None,
                   labels=list(range(N_CLS)), zero_division=0)
    print("--- per-class F1 ---")
    print("  " + "  ".join(f"{e}={f:.3f}" for e, f in zip(EMOTION_ORDER, pcf)))
    cm = confusion_matrix(gts, preds, labels=list(range(N_CLS)))
    print("--- confusion matrix (rows=true, cols=pred) ---")
    print(text_confusion(cm))
    print("=" * 66)

    summary = {
        "members": [{"model": r["model"], "file": r["file"],
                     "macro_f1": r["macro_f1"], "weight": float(wi)}
                    for r, wi in zip(runs, w)],
        "ensemble": m,
        "best_single_macro_f1": best,
        "per_class_f1": {e: round(float(f), 4)
                         for e, f in zip(EMOTION_ORDER, pcf)},
    }
    out = os.path.join(OUT_DIR, "ensemble_summary.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(os.path.join(OUT_DIR, "ensemble_report.txt"), "w") as fh:
        fh.write(classification_report(gts, preds, labels=list(range(N_CLS)),
                                       target_names=EMOTION_ORDER, digits=3,
                                       zero_division=0)
                 + "\n\n" + text_confusion(cm) + "\n")
    print(f"saved {os.path.relpath(out, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
