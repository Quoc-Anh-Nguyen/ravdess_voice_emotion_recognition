"""Preprocess the RAVDESS speech audio and build a metadata CSV.

For every raw .wav file this script:
  1. Parses the 7-part RAVDESS filename into labels.
  2. Loads the audio as mono and resamples it to a fixed rate.
  3. Trims leading/trailing silence and peak-normalizes the waveform.
  4. Writes the cleaned waveform to data/processed/.
  5. Extracts log-mel-spectrogram and MFCC features (saved as .npy).
  6. Records one metadata row per file.

Output:
  data/processed/Actor_XX/*.wav      cleaned audio
  data/features/mel/*.npy            log-mel spectrogram  (n_mels, frames)
  data/features/mfcc/*.npy           MFCC + deltas        (3*n_mfcc, frames)
  ravdess_metadata.csv               one row per clip
"""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

# ---------------------------------------------------------------- config ----
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
MEL_DIR = ROOT / "data" / "features" / "mel"
MFCC_DIR = ROOT / "data" / "features" / "mfcc"
CSV_PATH = ROOT / "ravdess_metadata.csv"

SAMPLE_RATE = 16_000   # Hz, target rate (speech-appropriate)
TRIM_TOP_DB = 30       # silence threshold for librosa.effects.trim
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128
N_MFCC = 40

# --------------------------------------------------- RAVDESS label maps -----
EMOTION = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}
INTENSITY = {"01": "normal", "02": "strong"}
MODALITY = {"01": "full-AV", "02": "video-only", "03": "audio-only"}
VOCAL_CHANNEL = {"01": "speech", "02": "song"}
STATEMENT = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door",
}
REPETITION = {"01": "1st repetition", "02": "2nd repetition"}


def parse_filename(stem: str) -> dict:
    """Parse a RAVDESS file stem like '03-01-06-01-02-01-12' into labels."""
    p = stem.split("-")
    if len(p) != 7:
        raise ValueError(f"unexpected RAVDESS filename: {stem!r}")
    modality, vocal, emotion, intensity, statement, repetition, actor = p
    actor_id = int(actor)
    return {
        "modality": MODALITY.get(modality, modality),
        "vocal_channel": VOCAL_CHANNEL.get(vocal, vocal),
        "emotion_code": int(emotion),
        "emotion": EMOTION.get(emotion, emotion),
        "intensity_code": int(intensity),
        "intensity": INTENSITY.get(intensity, intensity),
        "statement_code": int(statement),
        "statement": STATEMENT.get(statement, statement),
        "repetition": int(repetition),
        "actor": actor_id,
        "gender": "male" if actor_id % 2 == 1 else "female",
    }


def process_one(wav_path: Path) -> dict:
    """Preprocess a single clip and return its metadata row."""
    stem = wav_path.stem
    labels = parse_filename(stem)
    actor_dir = f"Actor_{labels['actor']:02d}"

    # load -> mono + target sample rate
    y, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    # trim silence
    y, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
    # peak-normalize (guard against all-silence clips)
    peak = np.max(np.abs(y)) if y.size else 0.0
    if peak > 0:
        y = y / peak

    # save cleaned waveform
    proc_path = PROC_DIR / actor_dir / f"{stem}.wav"
    proc_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(proc_path, y, SAMPLE_RATE, subtype="PCM_16")

    # log-mel spectrogram (n_mels, frames)
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

    # MFCC + delta + delta-delta (3*n_mfcc, frames)
    mfcc = librosa.feature.mfcc(
        y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mfcc_stack = np.vstack(
        [mfcc, librosa.feature.delta(mfcc), librosa.feature.delta(mfcc, order=2)]
    ).astype(np.float32)

    mel_path = MEL_DIR / f"{stem}.npy"
    mfcc_path = MFCC_DIR / f"{stem}.npy"
    np.save(mel_path, log_mel)
    np.save(mfcc_path, mfcc_stack)

    labels.update(
        {
            "filename": wav_path.name,
            "raw_path": str(wav_path.relative_to(ROOT).as_posix()),
            "processed_path": str(proc_path.relative_to(ROOT).as_posix()),
            "mel_path": str(mel_path.relative_to(ROOT).as_posix()),
            "mfcc_path": str(mfcc_path.relative_to(ROOT).as_posix()),
            "sample_rate": SAMPLE_RATE,
            "duration_sec": round(len(y) / SAMPLE_RATE, 4),
            "n_frames": int(log_mel.shape[1]),
        }
    )
    return labels


def main() -> int:
    wav_files = sorted(RAW_DIR.rglob("*.wav"))
    if not wav_files:
        raise SystemExit(
            f"No .wav files under {RAW_DIR}. Run download_ravdess.py first."
        )
    for d in (PROC_DIR, MEL_DIR, MFCC_DIR):
        d.mkdir(parents=True, exist_ok=True)

    rows, errors = [], []
    for wav in tqdm(wav_files, desc="preprocessing", unit="clip"):
        try:
            rows.append(process_one(wav))
        except Exception as exc:  # noqa: BLE001 - keep going, report at end
            errors.append((wav.name, str(exc)))

    column_order = [
        "filename", "actor", "gender", "emotion", "emotion_code",
        "intensity", "intensity_code", "statement", "statement_code",
        "repetition", "vocal_channel", "modality",
        "duration_sec", "sample_rate", "n_frames",
        "raw_path", "processed_path", "mel_path", "mfcc_path",
    ]
    df = pd.DataFrame(rows)[column_order].sort_values(
        ["actor", "emotion_code", "intensity_code", "statement_code", "repetition"]
    )
    df.to_csv(CSV_PATH, index=False)

    print(f"\nProcessed {len(df)} clips -> {CSV_PATH.name}")
    print(f"Cleaned audio : {PROC_DIR}")
    print(f"Mel features  : {MEL_DIR}")
    print(f"MFCC features : {MFCC_DIR}")
    if errors:
        print(f"\n{len(errors)} file(s) failed:")
        for name, msg in errors[:10]:
            print(f"  {name}: {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
