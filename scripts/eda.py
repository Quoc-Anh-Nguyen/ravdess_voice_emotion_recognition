"""Exploratory Data Analysis for the RAVDESS speech emotion dataset.

Produces thesis-ready figures + summary tables in figures/ and reports/:

  Dataset-level : emotion / gender / intensity / per-actor counts, duration stats
  Signal-level  : waveforms, STFT & mel spectrograms, MFCC heatmaps (per emotion)
  Prosody       : F0 / RMS / ZCR distributions across emotions
  Feature-space : correlation heatmap, PCA, t-SNE, UMAP projections

Run after extract_tabular_features.py.
"""
from __future__ import annotations

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="talk")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META_CSV = os.path.join(ROOT, "ravdess_metadata.csv")
FEAT_CSV = os.path.join(ROOT, "data", "features_tabular.csv")
FIG_DIR = os.path.join(ROOT, "figures")
REP_DIR = os.path.join(ROOT, "reports")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(REP_DIR, exist_ok=True)

SR = 16_000
N_FFT, HOP = 1024, 256
EMOTION_ORDER = ["neutral", "calm", "happy", "sad",
                 "angry", "fearful", "disgust", "surprised"]
PALETTE = dict(zip(EMOTION_ORDER, sns.color_palette("tab10", 8)))


def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figures/{name}")


# ----------------------------------------------------------- dataset level ---
def dataset_level(meta: pd.DataFrame) -> None:
    print("[1/4] dataset-level EDA")

    # emotion distribution
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.countplot(data=meta, x="emotion", order=EMOTION_ORDER,
                  hue="emotion", palette=PALETTE, legend=False, ax=ax)
    for c in ax.containers:
        ax.bar_label(c)
    ax.set_title("Emotion distribution (n=%d)" % len(meta))
    ax.set_xlabel(""); ax.set_ylabel("clips")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    save(fig, "01_emotion_distribution.png")

    # gender + intensity
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.countplot(data=meta, x="gender", hue="gender", palette="Set2",
                  legend=False, ax=axes[0])
    for c in axes[0].containers:
        axes[0].bar_label(c)
    axes[0].set_title("Gender balance"); axes[0].set_xlabel("")
    sns.countplot(data=meta, x="emotion", hue="intensity", order=EMOTION_ORDER,
                  palette="rocket", ax=axes[1])
    axes[1].set_title("Emotional intensity per emotion")
    axes[1].set_xlabel(""); axes[1].set_ylabel("clips")
    plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right")
    save(fig, "02_gender_intensity.png")

    # utterances per actor (colored by gender)
    fig, ax = plt.subplots(figsize=(13, 6))
    counts = meta.groupby(["actor", "gender"]).size().reset_index(name="n")
    sns.barplot(data=counts, x="actor", y="n", hue="gender",
                palette="Set2", dodge=False, ax=ax)
    ax.set_title("Utterances per actor"); ax.set_xlabel("actor id")
    save(fig, "03_utterances_per_actor.png")

    # duration: histogram + boxplot by emotion
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.histplot(meta["duration_sec"], bins=40, kde=True, ax=axes[0], color="steelblue")
    axes[0].axvline(meta["duration_sec"].mean(), color="red", ls="--",
                    label=f"mean={meta['duration_sec'].mean():.2f}s")
    axes[0].axvline(meta["duration_sec"].median(), color="green", ls="--",
                    label=f"median={meta['duration_sec'].median():.2f}s")
    axes[0].legend(); axes[0].set_title("Clip duration (after trim)")
    axes[0].set_xlabel("seconds")
    sns.boxplot(data=meta, x="emotion", y="duration_sec", order=EMOTION_ORDER,
                hue="emotion", palette=PALETTE, legend=False, ax=axes[1])
    axes[1].set_title("Duration by emotion"); axes[1].set_xlabel("")
    plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right")
    save(fig, "04_duration_stats.png")

    # write a stats table
    dur = meta["duration_sec"]
    stats = pd.DataFrame({
        "metric": ["clips", "actors", "emotions", "dur_mean", "dur_median",
                   "dur_std", "dur_min", "dur_max"],
        "value": [len(meta), meta.actor.nunique(), meta.emotion.nunique(),
                  round(dur.mean(), 3), round(dur.median(), 3), round(dur.std(), 3),
                  round(dur.min(), 3), round(dur.max(), 3)],
    })
    stats.to_csv(os.path.join(REP_DIR, "dataset_summary.csv"), index=False)
    print("  saved reports/dataset_summary.csv")


# ------------------------------------------------------------- signal level ---
def _rep_sample(meta: pd.DataFrame, emotion: str) -> str:
    """Pick a comparable clip per emotion (actor 1, statement 1, normal, rep 1)."""
    sub = meta[(meta.emotion == emotion) & (meta.actor == 1)]
    for q in [(1, "normal", 1), (1, "normal", 2), (1, "strong", 1)]:
        m = sub[(sub.statement == q[0]) & (sub.intensity == q[1]) &
                (sub.repetition == q[2])]
        if len(m):
            return os.path.join(ROOT, m.iloc[0]["processed_path"])
    return os.path.join(ROOT, sub.iloc[0]["processed_path"])


def signal_level(meta: pd.DataFrame) -> None:
    print("[2/4] signal-level EDA")
    samples = {e: _rep_sample(meta, e) for e in EMOTION_ORDER}
    audio = {e: librosa.load(p, sr=SR)[0] for e, p in samples.items()}

    # waveforms grid
    fig, axes = plt.subplots(4, 2, figsize=(16, 14))
    for ax, e in zip(axes.ravel(), EMOTION_ORDER):
        librosa.display.waveshow(audio[e], sr=SR, ax=ax, color=PALETTE[e])
        ax.set_title(e); ax.set_xlabel("")
    fig.suptitle("Representative waveforms by emotion (actor 1)", y=1.01)
    fig.tight_layout()
    save(fig, "05_waveforms.png")

    # STFT spectrograms grid
    fig, axes = plt.subplots(4, 2, figsize=(16, 14))
    for ax, e in zip(axes.ravel(), EMOTION_ORDER):
        S = librosa.amplitude_to_db(np.abs(librosa.stft(audio[e], n_fft=N_FFT,
                                    hop_length=HOP)), ref=np.max)
        img = librosa.display.specshow(S, sr=SR, hop_length=HOP, x_axis="time",
                                       y_axis="hz", ax=ax, cmap="magma")
        ax.set_title(e); ax.set_xlabel("")
    fig.colorbar(img, ax=axes, format="%+2.0f dB", shrink=0.6)
    fig.suptitle("STFT spectrograms by emotion", y=1.01)
    save(fig, "06_stft_spectrograms.png")

    # mel spectrograms grid
    fig, axes = plt.subplots(4, 2, figsize=(16, 14))
    for ax, e in zip(axes.ravel(), EMOTION_ORDER):
        M = librosa.power_to_db(librosa.feature.melspectrogram(
            y=audio[e], sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=128), ref=np.max)
        img = librosa.display.specshow(M, sr=SR, hop_length=HOP, x_axis="time",
                                       y_axis="mel", ax=ax, cmap="magma")
        ax.set_title(e); ax.set_xlabel("")
    fig.colorbar(img, ax=axes, format="%+2.0f dB", shrink=0.6)
    fig.suptitle("Mel spectrograms by emotion", y=1.01)
    save(fig, "07_mel_spectrograms.png")

    # MFCC heatmaps for contrastive pairs
    pairs = [("angry", "calm"), ("happy", "sad"), ("fearful", "neutral")]
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    for row, (a, b) in enumerate(pairs):
        for col, e in enumerate((a, b)):
            mfcc = librosa.feature.mfcc(y=audio[e], sr=SR, n_mfcc=40,
                                        n_fft=N_FFT, hop_length=HOP)
            img = librosa.display.specshow(mfcc, sr=SR, hop_length=HOP,
                                           x_axis="time", ax=axes[row, col], cmap="coolwarm")
            axes[row, col].set_title(f"MFCC — {e}")
    fig.colorbar(img, ax=axes, shrink=0.6)
    fig.suptitle("MFCC heatmaps: contrastive emotion pairs", y=1.01)
    save(fig, "08_mfcc_pairs.png")


# ----------------------------------------------------------------- prosody ---
def prosody(feat: pd.DataFrame) -> None:
    print("[3/4] prosodic-feature EDA")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, (col, title) in zip(axes, [
        ("f0_mean", "Pitch F0 (Hz)"), ("rms_mean", "RMS energy"),
        ("zcr_mean", "Zero-crossing rate")]):
        sns.boxplot(data=feat, x="emotion", y=col, order=EMOTION_ORDER,
                    hue="emotion", palette=PALETTE, legend=False, ax=ax)
        ax.set_title(title); ax.set_xlabel("")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.suptitle("Prosodic features across emotions", y=1.02)
    fig.tight_layout()
    save(fig, "09_prosody_by_emotion.png")

    # F0 by emotion split by gender (pitch is gender-dependent)
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.violinplot(data=feat, x="emotion", y="f0_mean", hue="gender",
                   order=EMOTION_ORDER, palette="Set2", split=True, ax=ax)
    ax.set_title("Pitch (F0) by emotion and gender"); ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    save(fig, "10_pitch_by_gender.png")


# ------------------------------------------------------------ feature space ---
def feature_space(feat: pd.DataFrame) -> None:
    print("[4/4] feature-space EDA")
    label_cols = ["filename", "actor", "gender", "emotion", "emotion_code",
                  "intensity", "statement", "repetition", "duration_sec"]
    X = feat.drop(columns=[c for c in label_cols if c in feat.columns])
    X = X.select_dtypes("number").fillna(0.0)
    y = feat["emotion"].values
    Xs = StandardScaler().fit_transform(X)

    # correlation heatmap on an interpretable subset
    subset = [c for c in ["f0_mean", "f0_std", "rms_mean", "rms_std", "zcr_mean",
              "centroid_mean", "bandwidth_mean", "rolloff_mean", "voiced_ratio",
              "mfcc1_mean", "mfcc2_mean", "mfcc3_mean", "mfcc4_mean", "mfcc5_mean",
              "contrast1_mean", "contrast2_mean"] if c in X.columns]
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(X[subset].corr(), annot=True, fmt=".2f", cmap="vlag", center=0,
                square=True, cbar_kws={"shrink": 0.7}, annot_kws={"size": 8}, ax=ax)
    ax.set_title("Correlation among interpretable features")
    save(fig, "11_correlation_heatmap.png")

    # PCA
    pca = PCA(n_components=2, random_state=0).fit(Xs)
    pcs = pca.transform(Xs)
    fig, ax = plt.subplots(figsize=(11, 9))
    for e in EMOTION_ORDER:
        m = y == e
        ax.scatter(pcs[m, 0], pcs[m, 1], s=18, alpha=0.7,
                   color=PALETTE[e], label=e)
    ax.set_title("PCA (var explained: %.1f%% + %.1f%%)" %
                 (pca.explained_variance_ratio_[0] * 100,
                  pca.explained_variance_ratio_[1] * 100))
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(markerscale=1.5, fontsize=10, ncol=2)
    save(fig, "12_pca.png")

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, init="pca",
                learning_rate="auto", random_state=0)
    emb = tsne.fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(11, 9))
    for e in EMOTION_ORDER:
        m = y == e
        ax.scatter(emb[m, 0], emb[m, 1], s=18, alpha=0.7,
                   color=PALETTE[e], label=e)
    ax.set_title("t-SNE projection"); ax.legend(markerscale=1.5, fontsize=10, ncol=2)
    save(fig, "13_tsne.png")

    # UMAP
    try:
        import umap
        reducer = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1,
                            random_state=0)
        u = reducer.fit_transform(Xs)
        fig, ax = plt.subplots(figsize=(11, 9))
        for e in EMOTION_ORDER:
            m = y == e
            ax.scatter(u[m, 0], u[m, 1], s=18, alpha=0.7,
                       color=PALETTE[e], label=e)
        ax.set_title("UMAP projection"); ax.legend(markerscale=1.5, fontsize=10, ncol=2)
        save(fig, "14_umap.png")
    except Exception as exc:  # noqa: BLE001
        print(f"  UMAP skipped: {exc}")


def main() -> int:
    meta = pd.read_csv(META_CSV)
    dataset_level(meta)
    signal_level(meta)
    if os.path.exists(FEAT_CSV):
        feat = pd.read_csv(FEAT_CSV)
        prosody(feat)
        feature_space(feat)
    else:
        print("features_tabular.csv not found -> skipping prosody + feature-space EDA")
    print(f"\nAll figures in {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
