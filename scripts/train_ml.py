"""Classical-ML baselines for RAVDESS speech emotion recognition.

Key methodological point for SER: evaluation must be SPEAKER-INDEPENDENT.
We therefore use GroupKFold grouped by actor id so that no actor appears in
both train and test. We also report a (leaky) random-split number to quantify
how much speaker overlap inflates accuracy — a common pitfall in SER papers.

Outputs:
  reports/ml_results.csv               per-model speaker-independent CV scores
  reports/ml_classification_report.txt best model per-class precision/recall/F1
  figures/15_confusion_matrix.png      best model, aggregated over CV folds
  figures/16_split_comparison.png      speaker-independent vs random split
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from sklearn.model_selection import (GroupKFold, StratifiedKFold,
                                     cross_val_predict, cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_CSV = os.path.join(ROOT, "data", "features_tabular.csv")
FIG_DIR = os.path.join(ROOT, "figures")
REP_DIR = os.path.join(ROOT, "reports")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(REP_DIR, exist_ok=True)

EMOTION_ORDER = ["neutral", "calm", "happy", "sad",
                 "angry", "fearful", "disgust", "surprised"]
N_SPLITS = 5
LABEL_COLS = ["filename", "actor", "gender", "emotion", "emotion_code",
              "intensity", "statement", "repetition", "duration_sec"]


def build_models():
    return {
        "SVM (RBF)": Pipeline([("sc", StandardScaler()),
                               ("clf", SVC(C=10, gamma="scale", kernel="rbf"))]),
        "LogReg": Pipeline([("sc", StandardScaler()),
                            ("clf", LogisticRegression(max_iter=2000, C=1.0))]),
        "RandomForest": Pipeline([("sc", StandardScaler()),
                                  ("clf", RandomForestClassifier(
                                      n_estimators=400, random_state=0, n_jobs=-1))]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=FEAT_CSV,
                    help="Feature CSV (label cols + numeric features).  Swap in "
                         "data/features_emotion2vec.csv for the SSL-embedding probe.")
    ap.add_argument("--out_prefix", default="",
                    help="Prefix for output files, e.g. 'emotion2vec_' so the "
                         "emotion2vec run does not overwrite the baseline.")
    args = ap.parse_args()

    df = pd.read_csv(args.features)
    X = df.drop(columns=[c for c in LABEL_COLS if c in df.columns])
    X = X.select_dtypes("number").fillna(0.0).values
    y = df["emotion"].values
    groups = df["actor"].values
    print(f"features={os.path.relpath(args.features, ROOT)}  X={X.shape}  "
          f"classes={len(np.unique(y))}  actors={len(np.unique(groups))}")

    gkf = GroupKFold(n_splits=N_SPLITS)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)

    rows = []
    for name, model in build_models().items():
        acc_si = cross_val_score(model, X, y, groups=groups, cv=gkf, scoring="accuracy")
        f1_si = cross_val_score(model, X, y, groups=groups, cv=gkf, scoring="f1_macro")
        acc_rand = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
        rows.append({
            "model": name,
            "acc_speaker_indep": round(acc_si.mean(), 4),
            "acc_si_std": round(acc_si.std(), 4),
            "f1_macro_speaker_indep": round(f1_si.mean(), 4),
            "acc_random_split": round(acc_rand.mean(), 4),
            "leakage_gap": round(acc_rand.mean() - acc_si.mean(), 4),
        })
        print(f"  {name:14s} SI acc={acc_si.mean():.3f}±{acc_si.std():.3f} "
              f"| SI f1={f1_si.mean():.3f} | random acc={acc_rand.mean():.3f}")

    res = pd.DataFrame(rows).sort_values("f1_macro_speaker_indep", ascending=False)
    res_csv = f"{args.out_prefix}ml_results.csv"
    res.to_csv(os.path.join(REP_DIR, res_csv), index=False)
    print(f"\nsaved reports/{res_csv}\n{res.to_string(index=False)}")

    # best model -> aggregated speaker-independent confusion matrix + report
    best_name = res.iloc[0]["model"]
    best = build_models()[best_name]
    y_pred = cross_val_predict(best, X, y, groups=groups, cv=gkf)
    acc = accuracy_score(y, y_pred)
    f1m = f1_score(y, y_pred, average="macro")

    report = classification_report(y, y_pred, labels=EMOTION_ORDER, digits=3)
    with open(os.path.join(REP_DIR,
                           f"{args.out_prefix}ml_classification_report.txt"), "w") as fh:
        fh.write(f"Best model: {best_name}\n")
        fh.write(f"Speaker-independent accuracy: {acc:.4f}  macro-F1: {f1m:.4f}\n\n")
        fh.write(report)
    print(f"\nBest model: {best_name} | acc={acc:.3f} | macro-F1={f1m:.3f}")

    fig, ax = plt.subplots(figsize=(10, 9))
    ConfusionMatrixDisplay.from_predictions(
        y, y_pred, labels=EMOTION_ORDER, normalize="true",
        cmap="Blues", xticks_rotation=45, values_format=".2f", ax=ax, colorbar=False)
    ax.set_title(f"{best_name} — speaker-independent confusion matrix\n"
                 f"acc={acc:.3f}, macro-F1={f1m:.3f}")
    fig.tight_layout()
    cm_png = f"{args.out_prefix}15_confusion_matrix.png"
    fig.savefig(os.path.join(FIG_DIR, cm_png), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figures/{cm_png}")

    # split-strategy comparison bar chart
    fig, ax = plt.subplots(figsize=(11, 6))
    melt = res.melt(id_vars="model",
                    value_vars=["acc_speaker_indep", "acc_random_split"],
                    var_name="split", value_name="accuracy")
    melt["split"] = melt["split"].map({"acc_speaker_indep": "speaker-independent",
                                       "acc_random_split": "random (leaky)"})
    sns.barplot(data=melt, x="model", y="accuracy", hue="split", palette="Set1", ax=ax)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f")
    ax.set_title("Speaker-independent vs random split (overestimation gap)")
    ax.set_ylim(0, 1); ax.set_xlabel("")
    fig.tight_layout()
    sc_png = f"{args.out_prefix}16_split_comparison.png"
    fig.savefig(os.path.join(FIG_DIR, sc_png), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figures/{sc_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
