"""From-scratch RCNN (CNN→BiLSTM) baseline for RAVDESS SER — a deep-learning
comparison point sitting BETWEEN the classical tabular models (train_ml.py)
and the SSL fine-tuning (train_ssl.py).

Input  : log-mel spectrograms (128 mel bands × T frames) from data/features/mel.
Model  : 3 conv blocks (2-D, time-freq) → BiLSTM over time → attention pooling
         → MLP head.  "RCNN" = Recurrent-Convolutional: convolution learns local
         time-frequency texture, the LSTM models how it evolves over the
         utterance, attention pooling summarises the most emotion-bearing frames.

Why this is a fair comparison:
  * SAME protocol as train_ssl.py — speaker-independent GroupKFold by actor,
    speaker-disjoint inner-val for best-epoch selection, identical metrics and
    COPY-BACK block, multi-seed softmax ensemble.
  * SAME anti-leakage care — per-mel-band standardisation uses TRAIN frames only.
  * Writes reports/rcnn/oof_probs_<tag>.npz so it can join the SSL backbones in
    scripts/ensemble_ssl.py (identical deterministic GroupKFold partition).

It is expected to LOSE to WavLM (no pretraining, 1440 clips is tiny) — that gap
is exactly the point of the comparison: it quantifies what self-supervised
pretraining buys on this dataset.

Augmentation (spectrogram-domain, prosody-preserving):
  * SpecAugment — time + frequency masking (the standard for log-mel).
  * Random time-crop (train) / centre-crop (eval) to --max_frames.
  * Hidden-free input mixup on the batched spectrograms.

Examples:
  python scripts/train_rcnn.py --mode smoke                 # ~1 min correctness
  python scripts/train_rcnn.py --mode cv --epochs 60 --batch 32
  python scripts/train_rcnn.py --mode cv --epochs 60 --n_seeds 3   # ensemble
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, recall_score)
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META_CSV = os.path.join(ROOT, "ravdess_metadata.csv")
OUT_DIR  = os.path.join(ROOT, "reports", "rcnn")
FIG_DIR  = os.path.join(ROOT, "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

EMOTION_ORDER = ["neutral", "calm", "happy", "sad",
                 "angry", "fearful", "disgust", "surprised"]
LBL2ID = {e: i for i, e in enumerate(EMOTION_ORDER)}
ID2LBL = {i: e for e, i in LBL2ID.items()}
N_MELS = 128
N_CLS  = len(EMOTION_ORDER)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(s=0):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# --------------------------------------------------------------- data --------
def band_stats(mels, indices):
    """Per-mel-band mean/std over ALL frames of the TRAIN clips only.
    Computed per fold from train indices → no speaker/test leakage."""
    s = np.zeros(N_MELS, np.float64)
    s2 = np.zeros(N_MELS, np.float64)
    n = 0
    for i in indices:
        m = mels[i]                                   # (128, T) dB
        s  += m.sum(axis=1)
        s2 += (m.astype(np.float64) ** 2).sum(axis=1)
        n  += m.shape[1]
    mean = (s / n).astype(np.float32)
    std  = np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-8)).astype(np.float32)
    return mean[:, None], std[:, None]                # (128, 1)


class MelDataset(Dataset):
    """Standardised log-mel, time-cropped to max_frames, optional SpecAugment."""

    def __init__(self, mels, indices, labels, mean, std, max_frames,
                 augment=False, seed=0, time_mask=0, freq_mask=0, n_masks=2):
        self.mels, self.indices, self.labels = mels, indices, labels
        self.mean, self.std, self.max_frames = mean, std, max_frames
        self.augment = augment
        self.time_mask, self.freq_mask, self.n_masks = time_mask, freq_mask, n_masks
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.indices)

    def _specaug(self, m, length):
        for _ in range(self.n_masks):
            if self.freq_mask > 0:                     # frequency masking
                f  = self.rng.randint(0, self.freq_mask + 1)
                f0 = self.rng.randint(0, N_MELS - f + 1)
                m[f0:f0 + f, :] = 0.0
            if self.time_mask > 0 and length > 1:      # time masking
                t  = self.rng.randint(0, min(self.time_mask, length) + 1)
                t0 = self.rng.randint(0, length - t + 1)
                m[:, t0:t0 + t] = 0.0
        return m

    def __getitem__(self, k):
        i = self.indices[k]
        m = (self.mels[i] - self.mean) / self.std      # (128, T), 0-mean per band
        T = m.shape[1]
        if T > self.max_frames:                        # crop
            s = (self.rng.randint(0, T - self.max_frames + 1)
                 if self.augment else (T - self.max_frames) // 2)
            m = m[:, s:s + self.max_frames]
        m = np.ascontiguousarray(m, dtype=np.float32)
        length = m.shape[1]
        if self.augment:
            m = self._specaug(m, length)
        return m, length, int(self.labels[i])


def collate(batch):
    """Pad spectrograms to the batch-max time length; return a length tensor
    so the model can mask padded frames in the LSTM + attention pooling."""
    mels    = [b[0] for b in batch]
    lengths = torch.tensor([b[1] for b in batch], dtype=torch.long)
    labels  = torch.tensor([b[2] for b in batch], dtype=torch.long)
    t_max   = max(m.shape[1] for m in mels)
    x = np.zeros((len(batch), N_MELS, t_max), dtype=np.float32)
    for j, m in enumerate(mels):
        x[j, :, :m.shape[1]] = m
    return torch.from_numpy(x).unsqueeze(1), lengths, labels   # (B,1,128,T)


def _worker_init(worker_id):
    info = torch.utils.data.get_worker_info()
    if info is not None:
        info.dataset.rng = np.random.RandomState(info.seed % (2 ** 32))


# -------------------------------------------------------------- model --------
class ConvBlock(nn.Module):
    def __init__(self, cin, cout, pool, drop):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.MaxPool2d(pool), nn.Dropout2d(drop))

    def forward(self, x):
        return self.net(x)


class CRNN(nn.Module):
    """3 conv blocks → BiLSTM over time → masked attention pooling → MLP head.

    Time is downsampled by 4 (two time-pools of 2); frequency is pooled to 16
    then adaptively to `freq_keep` so the LSTM input stays compact."""

    TIME_REDUCTION = 4

    def __init__(self, n_cls=N_CLS, ch=(32, 64, 128), cnn_drop=0.1,
                 freq_keep=4, rnn_hidden=128, rnn_layers=2, p_drop=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(1,     ch[0], pool=(2, 2), drop=cnn_drop),   # 128→64, T/2
            ConvBlock(ch[0], ch[1], pool=(2, 2), drop=cnn_drop),   # 64→32,  T/4
            ConvBlock(ch[1], ch[2], pool=(2, 1), drop=cnn_drop))   # 32→16,  T/4
        self.freq_pool = nn.AdaptiveAvgPool2d((freq_keep, None))   # → freq_keep
        rnn_in = ch[2] * freq_keep
        self.lstm = nn.LSTM(rnn_in, rnn_hidden, rnn_layers,
                            batch_first=True, bidirectional=True,
                            dropout=p_drop if rnn_layers > 1 else 0.0)
        h = rnn_hidden * 2
        self.attn = nn.Sequential(nn.Linear(h, h // 2), nn.Tanh(),
                                  nn.Linear(h // 2, 1))
        self.head = nn.Sequential(
            nn.Linear(h, h), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(h, n_cls))

    def forward(self, x, lengths):
        h = self.conv(x)                                # (B, C, F', T')
        h = self.freq_pool(h)                           # (B, C, freq_keep, T')
        b, c, fk, tp = h.shape
        h = h.permute(0, 3, 1, 2).reshape(b, tp, c * fk)   # (B, T', C*fk)
        out_len = torch.clamp(lengths // self.TIME_REDUCTION, min=1, max=tp)

        seq, _ = self.lstm(h)                            # (B, T', 2*hidden)
        idx  = torch.arange(tp, device=x.device)[None, :]
        mask = idx < out_len[:, None]                    # (B, T') valid frames
        e = self.attn(seq).squeeze(-1).masked_fill(~mask, float("-inf"))
        a = torch.softmax(e, dim=1).unsqueeze(-1)        # (B, T', 1)
        pooled = (a * seq).sum(dim=1)                    # (B, 2*hidden)
        return self.head(pooled)


# ------------------------------------------------------------- losses --------
class FocalCE(nn.Module):
    """Class-weighted, label-smoothed CE with optional focal modulation."""

    def __init__(self, weight=None, label_smoothing=0.0, gamma=0.0):
        super().__init__()
        self.register_buffer("weight", weight)
        self.ls, self.gamma = label_smoothing, gamma

    def forward(self, logits, labels):
        ce = F.cross_entropy(logits, labels, weight=self.weight,
                             label_smoothing=self.ls, reduction="none")
        if self.gamma == 0.0:
            return ce.mean()
        with torch.no_grad():
            pt = (torch.softmax(logits, -1)
                  .gather(1, labels.view(-1, 1)).squeeze(1).clamp(min=1e-6))
        return ((1 - pt) ** self.gamma * ce).mean()


def cosine_warmup(opt, warmup, total):
    def fn(step):
        if step < warmup:
            return step / max(1, warmup)
        p = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


# -------------------------------------------------------- train / eval -------
@torch.no_grad()
def evaluate(model, loader, max_batches=0):
    model.eval()
    probs, gts = [], []
    for bi, (x, lengths, labels) in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        logits = model(x.to(DEVICE), lengths.to(DEVICE))
        probs.append(torch.softmax(logits, -1).cpu().numpy())
        gts.append(labels.numpy())
    return np.concatenate(probs), np.concatenate(gts)


def metric_pack(gts, preds):
    return {
        "acc":         float(accuracy_score(gts, preds)),
        "uar":         float(recall_score(gts, preds, average="macro", zero_division=0)),
        "macro_f1":    float(f1_score(gts, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(gts, preds, average="weighted", zero_division=0)),
    }


def train_fold(model, tr_loader, va_loader, class_w, args, fold):
    crit = FocalCE(weight=torch.tensor(class_w, dtype=torch.float32, device=DEVICE),
                   label_smoothing=args.label_smoothing, gamma=args.focal_gamma)
    opt  = torch.optim.AdamW(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)
    total = args.epochs * len(tr_loader)
    sched = cosine_warmup(opt, int(0.1 * total), total)

    history, best_f1, best_state, row_best = [], -1.0, None, 1
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot, correct, seen = time.perf_counter(), 0.0, 0, 0
        for i, (x, lengths, labels) in enumerate(tr_loader):
            if args.max_train_batches and i >= args.max_train_batches:
                break
            x, lengths, labels = x.to(DEVICE), lengths.to(DEVICE), labels.to(DEVICE)

            mix_lam, labels_b = 1.0, labels
            if args.mixup_alpha > 0 and labels.numel() > 1:
                mix_lam = float(np.random.beta(args.mixup_alpha, args.mixup_alpha))
                mix_lam = max(mix_lam, 1.0 - mix_lam)
                perm    = torch.randperm(labels.size(0), device=DEVICE)
                x       = mix_lam * x + (1 - mix_lam) * x[perm]
                lengths = torch.maximum(lengths, lengths[perm])
                labels_b = labels[perm]

            logits = model(x, lengths)
            loss = (mix_lam * crit(logits, labels)
                    + (1 - mix_lam) * crit(logits, labels_b))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()

            tot += loss.item()
            correct += (logits.detach().argmax(-1) == labels).sum().item()
            seen += labels.numel()

        vp, vg = evaluate(model, va_loader, args.max_eval_batches)
        vm = metric_pack(vg, vp.argmax(-1))
        row = {"fold": fold, "epoch": ep,
               "train_loss": round(tot / max(i + 1, 1), 4),
               "train_acc": round(correct / max(seen, 1), 4),
               "val_acc": round(vm["acc"], 4), "val_uar": round(vm["uar"], 4),
               "val_macro_f1": round(vm["macro_f1"], 4),
               "lr": round(sched.get_last_lr()[0], 7),
               "sec": round(time.perf_counter() - t0, 1)}
        history.append(row)
        print(f"    ep {ep:2d}  loss={row['train_loss']:.3f}  "
              f"tr_acc={row['train_acc']:.3f}  val_f1={row['val_macro_f1']:.3f}  "
              f"lr={row['lr']:.2e}  {row['sec']:.0f}s", flush=True)
        if vm["macro_f1"] > best_f1:
            best_f1, row_best = vm["macro_f1"], ep
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, row_best


# ------------------------------------------------------------- report --------
def text_confusion(cm):
    head = "          " + " ".join(f"{e[:4]:>5}" for e in EMOTION_ORDER)
    rows = [f"{e[:9]:>9} " + " ".join(f"{cm[i, j]:5d}" for j in range(N_CLS))
            for i, e in enumerate(EMOTION_ORDER)]
    return "\n".join([head] + rows)


def save_confusion_fig(gts, preds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(10, 9))
    ConfusionMatrixDisplay.from_predictions(
        [ID2LBL[g] for g in gts], [ID2LBL[p] for p in preds],
        labels=EMOTION_ORDER, normalize="true", cmap="Greens",
        xticks_rotation=45, values_format=".2f", ax=ax, colorbar=False)
    m = metric_pack(gts, preds)
    ax.set_title(f"RCNN (log-mel CRNN) — speaker-independent\n"
                 f"acc={m['acc']:.3f}  UAR={m['uar']:.3f}  "
                 f"macro-F1={m['macro_f1']:.3f}")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "18_rcnn_confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"saved {os.path.relpath(path, ROOT)}", flush=True)


# --------------------------------------------------------------- main --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cv", "quick", "smoke"], default="cv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n_seeds", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--max_frames", type=int, default=200)
    ap.add_argument("--num_workers", type=int, default=0)
    # losses
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--class_weight_beta", type=float, default=0.5,
                    help="Soften balanced class weights: w**beta, renorm to mean 1. "
                         "1.0=full balanced, 0.0=uniform.")
    ap.add_argument("--focal_gamma", type=float, default=0.0)
    ap.add_argument("--mixup_alpha", type=float, default=0.2)
    # spec-augment
    ap.add_argument("--time_mask", type=int, default=40)
    ap.add_argument("--freq_mask", type=int, default=16)
    ap.add_argument("--n_masks", type=int, default=2)
    # model
    ap.add_argument("--rnn_hidden", type=int, default=128)
    ap.add_argument("--rnn_layers", type=int, default=2)
    ap.add_argument("--p_drop", type=float, default=0.3)
    ap.add_argument("--cnn_drop", type=float, default=0.1)
    # protocol
    ap.add_argument("--inner_val_frac", type=float, default=0.25)
    ap.add_argument("--run_tag", default="")
    ap.add_argument("--max_train_batches", type=int, default=0)
    ap.add_argument("--max_eval_batches", type=int, default=0)
    args = ap.parse_args()

    if args.mode == "smoke":
        args.epochs = max(1, min(args.epochs, 2))
        args.max_train_batches = args.max_train_batches or 8
        args.max_eval_batches  = args.max_eval_batches  or 8
    args.n_seeds = max(1, args.n_seeds)

    env = {"device": DEVICE, "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"}
    print(json.dumps({"config": vars(args), "env": env}, indent=2), flush=True)
    with open(os.path.join(OUT_DIR, "config.json"), "w") as fh:
        json.dump({"config": vars(args), "env": env}, fh, indent=2)

    meta   = pd.read_csv(META_CSV)
    labels = meta["emotion"].map(LBL2ID).values
    groups = meta["actor"].values
    print(f"loading {len(meta)} log-mel spectrograms into memory ...", flush=True)
    mels = [np.load(os.path.join(ROOT, p)).astype(np.float32)
            for p in meta["mel_path"]]

    def make_loader(mels, idx, labels, mean, std, shuffle, aug, ds_seed):
        ds = MelDataset(mels, idx, labels, mean, std, args.max_frames,
                        augment=aug, seed=ds_seed,
                        time_mask=args.time_mask, freq_mask=args.freq_mask,
                        n_masks=args.n_masks)
        return DataLoader(ds, batch_size=args.batch, shuffle=shuffle,
                          collate_fn=collate, num_workers=args.num_workers,
                          pin_memory=(DEVICE == "cuda"),
                          worker_init_fn=(_worker_init if args.num_workers > 0 else None),
                          persistent_workers=(args.num_workers > 0))

    splits = list(GroupKFold(n_splits=args.folds).split(mels, labels, groups))
    if args.mode in ("quick", "smoke"):
        splits = splits[:1]

    seed_oof_probs, seed_summaries = [], []
    filled_mask = np.zeros(len(labels), dtype=bool)
    all_fold_rows, all_history = [], []
    t_start = time.perf_counter()

    for seed in range(args.seed, args.seed + args.n_seeds):
        set_seed(seed)
        if args.n_seeds > 1:
            print(f"\n{'#'*60}\nSEED {seed} "
                  f"({seed-args.seed+1}/{args.n_seeds})\n{'#'*60}", flush=True)
        oof_probs = np.zeros((len(labels), N_CLS))

        for fold, (tr, te) in enumerate(splits, 1):
            test_actors = sorted(int(a) for a in set(groups[te]))
            print(f"\n=== fold {fold}/{len(splits)} test_actors={test_actors} ===",
                  flush=True)

            uniq = np.unique(groups[tr])
            rng  = np.random.RandomState(seed + fold)
            n_va = min(max(2, int(round(len(uniq) * args.inner_val_frac))),
                       len(uniq) - 1)
            val_actors = set(rng.choice(uniq, size=n_va, replace=False))
            is_val = np.array([g in val_actors for g in groups[tr]])
            tr_idx, va_idx = tr[~is_val], tr[is_val]

            mean, std = band_stats(mels, tr_idx)        # train-only normalisation
            cw = compute_class_weight("balanced", classes=np.arange(N_CLS),
                                      y=labels[tr_idx])
            if args.class_weight_beta != 1.0:
                cw = np.clip(cw, 1e-8, None) ** args.class_weight_beta
                cw = cw * (N_CLS / cw.sum())

            tr_ldr = make_loader(mels, tr_idx, labels, mean, std,
                                 True,  True,  seed * 1000 + fold)
            va_ldr = make_loader(mels, va_idx, labels, mean, std,
                                 False, False, seed)

            model = CRNN(N_CLS, cnn_drop=args.cnn_drop, rnn_hidden=args.rnn_hidden,
                         rnn_layers=args.rnn_layers, p_drop=args.p_drop).to(DEVICE)
            if fold == 1 and seed == args.seed:
                npar = sum(p.numel() for p in model.parameters())
                print(f"  CRNN params {npar/1e6:.2f}M  (rnn_hidden={args.rnn_hidden}"
                      f" layers={args.rnn_layers})", flush=True)

            model, hist, best_ep = train_fold(model, tr_ldr, va_ldr, cw, args, fold)
            for r in hist:
                r["seed"] = seed
            all_history.extend(hist)

            te_ldr = make_loader(mels, te, labels, mean, std, False, False, seed)
            probs, _ = evaluate(model, te_ldr)
            oof_probs[te] = probs
            filled_mask[te] = True
            m = metric_pack(labels[te], probs.argmax(-1))
            all_fold_rows.append({"seed": seed, "fold": fold,
                                  "test_actors": str(test_actors),
                                  "best_epoch": best_ep,
                                  **{k: round(v, 4) for k, v in m.items()}})
            print(f"  >> seed {seed} fold {fold}  acc={m['acc']:.4f}  "
                  f"uar={m['uar']:.4f}  macro_f1={m['macro_f1']:.4f}  "
                  f"weighted_f1={m['weighted_f1']:.4f}", flush=True)
            del model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        seed_oof_probs.append(oof_probs)
        sm = metric_pack(labels[filled_mask], oof_probs[filled_mask].argmax(-1))
        seed_summaries.append({"seed": seed, **{k: round(v, 4) for k, v in sm.items()}})
        print(f"\n  == seed {seed} OOF  acc={sm['acc']:.4f}  uar={sm['uar']:.4f}  "
              f"macro_f1={sm['macro_f1']:.4f} ==", flush=True)

    # ---- ensemble across seeds + persist ----
    ens_probs = np.mean(np.stack(seed_oof_probs, axis=0), axis=0)
    gts   = labels[filled_mask]
    preds = ens_probs[filled_mask].argmax(-1)
    agg   = metric_pack(gts, preds)
    cm    = confusion_matrix(gts, preds, labels=list(range(N_CLS)))
    report = classification_report(gts, preds, labels=list(range(N_CLS)),
                                   target_names=EMOTION_ORDER, digits=3,
                                   zero_division=0)
    per_cls_f1 = f1_score(gts, preds, average=None,
                          labels=list(range(N_CLS)), zero_division=0)
    elapsed = round((time.perf_counter() - t_start) / 60, 1)

    pd.DataFrame(all_fold_rows).to_csv(
        os.path.join(OUT_DIR, "fold_results.csv"), index=False)
    pd.DataFrame(all_history).to_csv(
        os.path.join(OUT_DIR, "history.csv"), index=False)
    out_df = pd.DataFrame({"y_true": [ID2LBL[g] for g in gts],
                           "y_pred": [ID2LBL[p] for p in preds],
                           "actor": groups[filled_mask]})
    for ci, e in enumerate(EMOTION_ORDER):
        out_df[f"p_{e}"] = ens_probs[filled_mask][:, ci].round(4)
    out_df.to_csv(os.path.join(OUT_DIR, "oof_predictions.csv"), index=False)

    # ensemble-compatible probs (scripts/ensemble_ssl.py globs reports/*/oof_probs_*.npz)
    tag = f"rcnn-mel_s{args.seed}-{args.seed + args.n_seeds - 1}"
    if args.run_tag:
        tag += "_" + args.run_tag
    tag = "".join(c if (c.isalnum() or c in "-._") else "_" for c in tag)
    np.savez(os.path.join(OUT_DIR, f"oof_probs_{tag}.npz"),
             probs=ens_probs, labels=labels, filled=filled_mask,
             model="rcnn-mel", macro_f1=agg["macro_f1"])

    with open(os.path.join(OUT_DIR, "classification_report.txt"), "w") as fh:
        fh.write(report + "\n\n" + text_confusion(cm) + "\n")
    seed_f1s = np.array([s["macro_f1"] for s in seed_summaries])
    fold_f1_std = (float(pd.DataFrame(all_fold_rows).groupby("seed")["macro_f1"].std().mean())
                   if len(all_fold_rows) > 1 else 0.0)
    summary = {"config": vars(args), "env": env, "elapsed_min": elapsed,
               "ensemble_oof": agg, "per_seed_oof": seed_summaries,
               "seed_macro_f1_mean": round(float(seed_f1s.mean()), 4),
               "fold_macro_f1_std_mean": round(fold_f1_std, 4),
               "per_class_f1": {e: round(float(f), 4)
                                for e, f in zip(EMOTION_ORDER, per_cls_f1)},
               "folds": all_fold_rows}
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    if args.mode == "cv":
        save_confusion_fig(gts, preds)

    # ---- COPY-BACK ----
    print("\n" + "=" * 60)
    print("COPY THIS BACK TO CLAUDE")
    print("=" * 60)
    print(f"model=rcnn-mel (log-mel CRNN) | mode={args.mode} | "
          f"folds={len(splits)} | n_seeds={args.n_seeds} | "
          f"elapsed={elapsed}min | gpu={env['gpu']}")
    print(f"cfg: epochs={args.epochs} batch={args.batch} lr={args.lr} "
          f"wd={args.weight_decay} max_frames={args.max_frames} "
          f"rnn={args.rnn_hidden}x{args.rnn_layers} p_drop={args.p_drop} "
          f"ls={args.label_smoothing} cwbeta={args.class_weight_beta} "
          f"focal_g={args.focal_gamma} mixup={args.mixup_alpha} "
          f"specaug=t{args.time_mask}/f{args.freq_mask}x{args.n_masks}")
    print("--- per (seed, fold) test ---")
    for r in all_fold_rows:
        print(f"  s{r['seed']}f{r['fold']} ep={r['best_epoch']:>2}  "
              f"acc={r['acc']:.4f}  uar={r['uar']:.4f}  "
              f"macroF1={r['macro_f1']:.4f}  actors={r['test_actors']}")
    if len(seed_summaries) > 1:
        print("--- per-seed OOF ---")
        for s in seed_summaries:
            print(f"  seed{s['seed']}  macroF1={s['macro_f1']:.4f}")
    print(f"--- fold-to-fold macroF1 std (actor variance) = {fold_f1_std:.4f} ---")
    print(f"--- ENSEMBLE OOF (n={int(filled_mask.sum())}) ---")
    print(f"  acc={agg['acc']:.4f}  uar={agg['uar']:.4f}  "
          f"macroF1={agg['macro_f1']:.4f}  weightedF1={agg['weighted_f1']:.4f}")
    print("--- per-class F1 (ensemble OOF) ---")
    print("  " + "  ".join(f"{e}={f:.3f}" for e, f in zip(EMOTION_ORDER, per_cls_f1)))
    print("--- confusion matrix (rows=true, cols=pred) ---")
    print(text_confusion(cm))
    print("=" * 60)
    print(f"artifacts in {os.path.relpath(OUT_DIR, ROOT)}/  | "
          f"join the SSL ensemble with: python scripts/ensemble_ssl.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
