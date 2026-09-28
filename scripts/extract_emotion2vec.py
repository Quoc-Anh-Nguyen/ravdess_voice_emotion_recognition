"""Extract frozen emotion2vec embeddings for RAVDESS (utterance + optional frame).

emotion2vec (Ma et al., ACL 2024) is an SSL model PRE-TRAINED specifically for
speech emotion, so its frozen embeddings are a strong representation — the authors
themselves report results with a simple linear probe on top.  Unlike WavLM it is
NOT a HuggingFace transformers model: it loads via FunASR
(`pip install -U funasr modelscope`), not `transformers.AutoModel`.

We therefore extract embeddings ONCE here, then evaluate them speaker-independently
with the existing classical pipeline:

    python scripts/extract_emotion2vec.py --model iic/emotion2vec_plus_large
    python scripts/train_ml.py --features data/features_emotion2vec.csv \
        --out_prefix emotion2vec_

This gives an apples-to-apples comparison (same GroupKFold, same SVM/LogReg/RF,
same report) against the 192-d hand-crafted features — the cleanest way to show
whether emotion-specialised pre-training beats a general encoder.

Output:
  data/features_emotion2vec.csv          labels + e1..eD utterance embedding
                                         (D=1024 for plus_large, 768 for *_base)
  data/features/emotion2vec/<stem>.npy   (only with --frame) frame-level (T, D),
                                         for a future attentive-pooling head.

Models (funasr/modelscope ids):
  iic/emotion2vec_plus_large  (1024-d, recommended)
  iic/emotion2vec_plus_base   (768-d)
  iic/emotion2vec_base        (768-d, original)
  iic/emotion2vec_plus_seed

Smoke test 4 clips first:  python scripts/extract_emotion2vec.py --limit 4 --device cpu
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META_CSV = os.path.join(ROOT, "ravdess_metadata.csv")
OUT_CSV = os.path.join(ROOT, "data", "features_emotion2vec.csv")
FRAME_DIR = os.path.join(ROOT, "data", "features", "emotion2vec")

# Same label columns as data/features_tabular.csv so train_ml.py treats the rest
# (e1..eD) as the numeric feature matrix unchanged.
LABEL_COLS = ["filename", "actor", "gender", "emotion", "emotion_code",
              "intensity", "statement", "repetition", "duration_sec"]


def _feats(res) -> np.ndarray:
    """Pull the embedding array out of a funasr generate() result robustly."""
    item = res[0] if isinstance(res, (list, tuple)) else res
    if isinstance(item, dict):
        arr = item.get("feats", item.get("feat", item.get("embedding")))
    else:
        arr = item
    return np.asarray(arr, dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="iic/emotion2vec_plus_large",
                    help="funasr/modelscope model id (see module docstring).")
    ap.add_argument("--device", default="cuda:0",
                    help="'cuda:0' or 'cpu'.")
    ap.add_argument("--frame", action="store_true",
                    help="Also save frame-level (T, D) embeddings as .npy "
                         "(enables a later attentive-pooling head).")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only the first N clips (smoke test).")
    args = ap.parse_args()

    # Imported here so this file stays importable / syntax-checkable without funasr.
    from funasr import AutoModel

    meta = pd.read_csv(META_CSV)
    if args.limit:
        meta = meta.head(args.limit)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    if args.frame:
        os.makedirs(FRAME_DIR, exist_ok=True)

    print(f"loading {args.model} on {args.device} ...", flush=True)
    model = AutoModel(model=args.model, device=args.device, disable_update=True)

    rows, errors = [], []
    tasks = list(zip(meta["filename"], meta["processed_path"]))
    for fn, proc_rel in tqdm(tasks, desc="emotion2vec", unit="clip"):
        path = os.path.join(ROOT, proc_rel)
        try:
            res = model.generate(path, granularity="utterance",
                                 extract_embedding=True)
            utt = _feats(res).ravel()                       # (D,)
            if args.frame:
                fr = _feats(model.generate(path, granularity="frame",
                                           extract_embedding=True))   # (T, D)
                np.save(os.path.join(FRAME_DIR, os.path.splitext(fn)[0] + ".npy"),
                        fr)
            row = {"filename": fn}
            row.update({f"e{i + 1}": float(v) for i, v in enumerate(utt)})
            rows.append(row)
        except Exception as exc:  # noqa: BLE001 — keep going, report at the end
            errors.append((fn, str(exc)))

    if not rows:
        raise SystemExit("No embeddings extracted — check the funasr install, "
                         "the model id, and that data/processed/ exists.")

    feat_df = pd.DataFrame(rows)
    merged = (meta[LABEL_COLS].merge(feat_df, on="filename", how="inner")
              .sort_values(["actor", "emotion_code"]).reset_index(drop=True))
    merged.to_csv(OUT_CSV, index=False)

    print(f"\nWrote {OUT_CSV}")
    print(f"Rows: {len(merged)}  |  embedding dim: {len(rows[0]) - 1}")
    if errors:
        print(f"WARNING: {len(errors)} clip(s) failed feature extraction:")
        for fn, msg in errors[:10]:
            print(f"  {fn}: {msg}")
    print("\nNext:  python scripts/train_ml.py "
          "--features data/features_emotion2vec.csv --out_prefix emotion2vec_")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
