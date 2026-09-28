"""Extract a fixed-length tabular feature vector per clip for EDA + classical ML.

For every cleaned clip listed in ravdess_metadata.csv this computes a single
feature row by aggregating frame-level descriptors (mean/std) plus prosodic
statistics:

  MFCC (40)  mean + std                       -> 80
  MFCC delta (40) mean, delta2 (40) mean      -> 80
  chroma (12) mean                            -> 12
  spectral contrast (7) mean                  -> 7
  spectral centroid / bandwidth / rolloff     -> mean+std each (6)
  zero-crossing-rate, RMS energy              -> mean+std each (4)
  F0 (pyin)                                   -> mean, std, voiced_ratio (3)

Output: data/features_tabular.csv  (labels + ~192 numeric feature columns)
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META_CSV = os.path.join(ROOT, "ravdess_metadata.csv")
OUT_CSV = os.path.join(ROOT, "data", "features_tabular.csv")

SR = 16_000
N_FFT = 1024
HOP = 256
N_MFCC = 40
F0_MIN, F0_MAX = 65.0, 400.0   # speech pitch range (Hz)

LABEL_COLS = [
    "filename", "actor", "gender", "emotion", "emotion_code",
    "intensity", "statement", "repetition", "duration_sec",
]


def _stats(prefix: str, arr: np.ndarray, with_std: bool = True) -> dict:
    """mean (+ optional std) for each row of a (features, frames) array."""
    out = {}
    mean = arr.mean(axis=1)
    for i, v in enumerate(mean, 1):
        out[f"{prefix}{i}_mean"] = float(v)
    if with_std:
        std = arr.std(axis=1)
        for i, v in enumerate(std, 1):
            out[f"{prefix}{i}_std"] = float(v)
    return out


def _scalar_stats(name: str, arr: np.ndarray) -> dict:
    arr = np.asarray(arr).ravel()
    return {f"{name}_mean": float(arr.mean()), f"{name}_std": float(arr.std())}


def extract_one(args: tuple) -> dict:
    """Worker: compute one feature row. Returns {} keyed errors on failure."""
    filename, proc_rel = args
    path = os.path.join(ROOT, proc_rel)
    try:
        y, _ = librosa.load(path, sr=SR, mono=True)
        feats: dict = {"filename": filename}

        mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP)
        feats.update(_stats("mfcc", mfcc, with_std=True))
        feats.update(_stats("mfccd1_", librosa.feature.delta(mfcc), with_std=False))
        feats.update(_stats("mfccd2_", librosa.feature.delta(mfcc, order=2), with_std=False))

        chroma = librosa.feature.chroma_stft(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP)
        feats.update(_stats("chroma", chroma, with_std=False))

        contrast = librosa.feature.spectral_contrast(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP)
        feats.update(_stats("contrast", contrast, with_std=False))

        feats.update(_scalar_stats("centroid",
            librosa.feature.spectral_centroid(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP)))
        feats.update(_scalar_stats("bandwidth",
            librosa.feature.spectral_bandwidth(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP)))
        feats.update(_scalar_stats("rolloff",
            librosa.feature.spectral_rolloff(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP)))
        feats.update(_scalar_stats("zcr",
            librosa.feature.zero_crossing_rate(y, hop_length=HOP)))
        feats.update(_scalar_stats("rms",
            librosa.feature.rms(y=y, hop_length=HOP)))

        # F0 via probabilistic YIN
        f0, voiced_flag, _ = librosa.pyin(
            y, sr=SR, fmin=F0_MIN, fmax=F0_MAX, frame_length=N_FFT, hop_length=HOP
        )
        voiced = f0[~np.isnan(f0)]
        feats["f0_mean"] = float(voiced.mean()) if voiced.size else 0.0
        feats["f0_std"] = float(voiced.std()) if voiced.size else 0.0
        feats["voiced_ratio"] = float(np.mean(voiced_flag)) if voiced_flag.size else 0.0
        return feats
    except Exception as exc:  # noqa: BLE001
        return {"filename": filename, "_error": str(exc)}


def main() -> int:
    meta = pd.read_csv(META_CSV)
    tasks = list(zip(meta["filename"], meta["processed_path"]))

    workers = max(1, (os.cpu_count() or 2) - 1)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(extract_one, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="features", unit="clip"):
            rows.append(fut.result())

    feat_df = pd.DataFrame(rows)
    errors = feat_df[feat_df.get("_error").notna()] if "_error" in feat_df else feat_df.iloc[0:0]
    if len(errors):
        print(f"WARNING: {len(errors)} clip(s) failed feature extraction")
        feat_df = feat_df[feat_df.get("_error").isna()].drop(columns=["_error"])
    elif "_error" in feat_df:
        feat_df = feat_df.drop(columns=["_error"])

    merged = meta[LABEL_COLS].merge(feat_df, on="filename", how="inner")
    merged = merged.sort_values(["actor", "emotion_code"]).reset_index(drop=True)
    merged.to_csv(OUT_CSV, index=False)

    n_feat = merged.shape[1] - len(LABEL_COLS)
    print(f"\nWrote {OUT_CSV}")
    print(f"Rows: {len(merged)}  |  feature columns: {n_feat}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
