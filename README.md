# RAVDESS Speech Emotion Dataset — Collection & Preprocessing

Pipeline to download the **RAVDESS** speech audio dataset, preprocess it, and build a
metadata CSV with extracted acoustic features for emotion-recognition modeling.

- **Dataset:** Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)
- **Subset:** Audio-only **Speech** — 1440 clips, 24 actors (12 male, 12 female)
- **Source:** https://zenodo.org/records/1188976 (DOI 10.5281/zenodo.1188976)
- **License:** CC BY-NC-SA 4.0 — *non-commercial*; cite Livingstone & Russo (2018), PLoS ONE.

## Layout

```
DAP/
├── scripts/
│   ├── download_ravdess.py     # download + extract from Zenodo
│   └── preprocess_ravdess.py   # clean audio + extract features + build CSV
├── data/
│   ├── raw/Actor_01..24/*.wav  # original clips (48 kHz)
│   ├── processed/Actor_*/      # cleaned: mono, 16 kHz, trimmed, normalized
│   └── features/
│       ├── mel/*.npy           # log-mel spectrogram  (128, frames)
│       └── mfcc/*.npy          # MFCC + Δ + ΔΔ        (120, frames)
└── ravdess_metadata.csv        # one row per clip (labels + paths)
```

## Reproduce

```bash
pip install librosa soundfile pandas numpy requests tqdm
python scripts/download_ravdess.py      # ~208 MB
python scripts/preprocess_ravdess.py    # ~1.5 min for 1440 clips
```

## Preprocessing

Each clip is loaded to **mono** and resampled to **16 kHz**, silence-trimmed
(`top_db=30`), and **peak-normalized**. Features (`n_fft=1024`, `hop=256`):

- **log-mel spectrogram** — 128 mel bands, dB scale
- **MFCC** — 40 coefficients stacked with their delta and delta-delta (→ 120 rows)

## Metadata columns (`ravdess_metadata.csv`)

| column | description |
|---|---|
| `filename` | original RAVDESS file name |
| `actor` / `gender` | actor id 1–24 / male (odd) or female (even) |
| `emotion` / `emotion_code` | neutral, calm, happy, sad, angry, fearful, disgust, surprised / 1–8 |
| `intensity` / `intensity_code` | normal, strong / 1–2 (neutral has only normal) |
| `statement` / `statement_code` | spoken sentence / 1–2 |
| `repetition` | 1 or 2 |
| `vocal_channel` / `modality` | speech / audio-only |
| `duration_sec` / `sample_rate` / `n_frames` | clip length, 16000, feature frame count |
| `raw_path` / `processed_path` / `mel_path` / `mfcc_path` | repo-relative file paths |

**Label balance:** 192 clips per emotion (96 for neutral); 720 male / 720 female.

The RAVDESS filename is a 7-part code
`modality-vocalchannel-emotion-intensity-statement-repetition-actor`
(e.g. `03-01-06-01-02-01-12.wav`), decoded in `scripts/preprocess_ravdess.py`.
