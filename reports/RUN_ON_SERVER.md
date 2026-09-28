# Running the SSL fine-tuning on your server

The script `scripts/train_ssl.py` fine-tunes WavLM for speaker-independent RAVDESS
SER and prints/saves everything needed to iterate. Run it on the server, then
**copy the final `COPY THIS BACK TO CLAUDE` block** (and optionally
`reports/ssl/summary.json`) back here so we can tune further.

## 1. Setup (once)
```bash
pip install "torch" torchaudio --index-url https://download.pytorch.org/whl/cu124   # match your CUDA
pip install transformers soundfile librosa pandas numpy scikit-learn matplotlib tqdm
# NOTE: do NOT keep a mismatched torchvision — if `import transformers` fails with
# "operator torchvision::nms does not exist", run:  pip uninstall -y torchvision
```
Prerequisite files already produced locally (copy the repo, or at least
`ravdess_metadata.csv` + `data/processed/`): the script reads `processed_path`
from the metadata CSV.

## 2. Recommended command

### Server GPU >= 16 GB (fast, full fine-tune)
```bash
python scripts/train_ssl.py --mode cv --epochs 25 \
    --batch 16 --accum 1 --max_sec 5 \
    --unfreeze_top 12 --augment
```

### Small GPU (~4 GB) fallback
```bash
python scripts/train_ssl.py --mode cv --epochs 25 \
    --batch 2 --accum 8 --max_sec 4 \
    --unfreeze_top 12 --grad_ckpt --augment
```

Quick single-fold check of a config before the full 5-fold:
```bash
python scripts/train_ssl.py --mode quick --epochs 20 --batch 16 --augment
```

## 3. Knobs to push past 0.80 (try in this order)
| Knob | Default | Try | Effect |
|---|---|---|---|
| `--epochs` | 25 | 30–40 | more convergence (watch val curve for overfit) |
| `--batch` / `--accum` | 2 / 8 | bigger batch on big GPU | stabler gradients |
| `--max_sec` | 4 | 5–6 | keep more of each utterance |
| `--unfreeze_top` | 12 | 12 (all) | full fine-tune = highest ceiling |
| `--enc_lr` | 2e-5 | 1e-5 … 5e-5 | core fine-tuning LR |
| `--head_lr` | 2e-4 | 1e-4 … 5e-4 | new head LR |
| `--llrd` | 0.9 | 0.8–0.95 | layer-wise LR decay strength |
| `--label_smoothing` | 0.1 | 0.0–0.15 | regularization |
| `--mask_time_prob` | 0.08 | 0.05–0.12 | SpecAugment strength |
| `--augment` | off | on | waveform aug (gain/noise/shift) |
| `--model` | wavlm-base-plus | `microsoft/wavlm-large`, `facebook/hubert-large-ls960-ft` | bigger encoder → usually higher, needs more VRAM |

If `--model microsoft/wavlm-large` fits on the server, it is the single most
likely change to clear 0.80 comfortably.

## 4. What you get (under `reports/ssl/`)
- `config.json` — exact hyperparameters + GPU/torch info for the run
- `history.csv` — per-epoch: train_loss, train_acc, val_acc, val_uar, val_macro_f1, lr, sec, gpu_mb
- `fold_results.csv` — per-fold test: best_epoch, acc, uar, macro_f1, weighted_f1
- `classification_report.txt` — per-class precision/recall/F1 + confusion matrix (counts)
- `summary.json` — aggregate metrics, per-fold mean±std, per-class F1
- `oof_predictions.csv` — out-of-fold y_true / y_pred / actor (for any custom analysis)
- `figures/17_ssl_confusion_matrix.png` — normalized OOF confusion matrix

And stdout ends with a compact **COPY THIS BACK TO CLAUDE** block: per-fold +
aggregate acc/UAR/macro-F1, per-class F1, and a text confusion matrix. That block
alone is enough for me to recommend the next change.

## 5. Metrics glossary
- **acc / WA** — overall accuracy (weighted by class frequency).
- **UAR** — unweighted average recall (mean per-class recall) = balanced accuracy.
  Report this for SER because neutral has half the samples.
- **macro-F1** — mean per-class F1; the headline metric here.
- **weighted-F1** — F1 weighted by support.
Always report the **speaker-independent** numbers (this script's default).
