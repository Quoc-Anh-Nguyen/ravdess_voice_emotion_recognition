"""Download and extract the RAVDESS Speech (audio-only) dataset from Zenodo.

Dataset: Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)
Subset:  Audio-only Speech (1440 .wav files, 24 actors)
Source:  https://zenodo.org/records/1188976  (DOI 10.5281/zenodo.1188976)
License: CC BY-NC-SA 4.0
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

ZENODO_URL = (
    "https://zenodo.org/records/1188976/files/"
    "Audio_Speech_Actors_01-24.zip?download=1"
)
ZIP_NAME = "Audio_Speech_Actors_01-24.zip"
EXPECTED_WAV_COUNT = 1440

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"


def download(url: str, dest: Path) -> None:
    """Stream a URL to disk with a progress bar; skip if already complete."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        if dest.exists() and total and dest.stat().st_size == total:
            print(f"[skip] {dest.name} already downloaded ({total:,} bytes)")
            return
        with open(dest, "wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                fh.write(chunk)
                bar.update(len(chunk))


def extract(zip_path: Path, out_dir: Path) -> int:
    """Extract the archive and return the number of .wav files present."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        for m in tqdm(members, desc="extracting", unit="file"):
            zf.extract(m, out_dir)
    return len(list(out_dir.rglob("*.wav")))


def main() -> int:
    zip_path = RAW_DIR / ZIP_NAME
    print(f"Downloading RAVDESS speech audio to {zip_path} ...")
    download(ZENODO_URL, zip_path)

    print(f"Extracting into {RAW_DIR} ...")
    n_wav = extract(zip_path, RAW_DIR)
    print(f"Found {n_wav} .wav files (expected {EXPECTED_WAV_COUNT}).")
    if n_wav != EXPECTED_WAV_COUNT:
        print("WARNING: wav count does not match the expected total.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
