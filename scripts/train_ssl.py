"""Self-supervised fine-tuning for RAVDESS SER (PyTorch + HuggingFace).

Fine-tunes a pretrained SSL audio encoder (default microsoft/wavlm-base-plus)
for 8-way emotion recognition, evaluated SPEAKER-INDEPENDENTLY (GroupKFold by
actor). Target: 0.80 – 0.90 macro-F1.

WHY OLD AUGMENTATION HURT
--------------------------
  * pitch-shift / time-stretch: speech rate and pitch ARE the emotion cues on
    RAVDESS.  Stretching a 'sad' clip slower makes it sound 'calm'.
    These transforms corrupt the very signal the model must learn.
  * WavLM was pre-trained on clean speech.  Distorted audio sits off-manifold.
  * Dataset is tiny (~160 samples/class/fold); augmentation noise > diversity.
  → pitch_shift_prob and time_stretch_prob DEFAULT TO 0.0.

SOTA IMPROVEMENTS (v2)
-----------------------
  1. Supervised Contrastive Loss (SupCon, Khosla et al. 2020)
     Auxiliary loss on L2-normalised projections.  Explicitly pulls same-emotion
     embeddings together and pushes apart neutral/calm/sad/fearful, which
     standard CE cannot separate well via a linear boundary alone.
     A small projection head (2-layer MLP → 128-d) is trained jointly;
     only the main classifier is used at inference.
     --supcon_weight 0.3  --supcon_temp 0.07
     SupCon is computed on PRE-MIX clean embeddings; CE on mixed embeddings.
     This is the standard SupCon + mixup dual-objective pattern.

  2. Focal Cross-Entropy
     Modulates CE gradient by (1−p_t)^γ.  Easy examples (happy, angry, disgust)
     are down-weighted; the hard confusable cluster (neutral/calm/sad/fearful)
     gets amplified gradient.  γ=0 → standard CE.
     --focal_gamma 1.0  (try 0.5 – 2.0)

  3. Test-Time Augmentation (TTA)
     At inference, run K random-crop passes of each test clip and average
     softmax probabilities.  No training change; reduces crop-position variance
     on short RAVDESS clips for free.
     --tta_k 5

  Retained from v1:
  4. Hidden-level mixup (pooled embeddings, not raw waveforms).
  5. Random crop on train / centre crop on eval.
  6. Larger configurable inner-val actor fraction (--inner_val_frac 0.25).
  7. Audio-duration self-diagnostic on startup.
  8. Multi-seed softmax-averaging ensemble (--n_seeds).

Memory (4 GB GPU): --unfreeze_top 12 --grad_ckpt --batch 2 --accum 8
Memory (24 GB GPU): --unfreeze_top 16 --grad_ckpt --batch 8 --accum 2

Modes:
  --mode cv      full GroupKFold (headline result)
  --mode quick   1 fold (fast config check)
  --mode smoke   1 fold, capped batches (~1 min correctness test)

Example (4 GB laptop GPU):
  python train_ssl.py --mode cv --epochs 25 --batch 2 --accum 8 \
      --max_sec 4 --unfreeze_top 12 --grad_ckpt --augment \
      --supcon_weight 0.3 --focal_gamma 1.0 --tta_k 5

Example (24 GB GPU — wavlm-large, best speaker-independent recipe):
  python train_ssl.py --mode cv --model microsoft/wavlm-large \
      --epochs 30 --batch 16 --accum 1 --max_sec 4 \
      --unfreeze_top 16 --grad_ckpt --augment --num_workers 6 \
      --enc_lr 1e-5 --head_lr 1e-4 --llrd 0.9 \
      --proj_hidden 512 --p_drop 0.15 \
      --mixup_level hidden --inner_val_frac 0.3 --n_seeds 3 \
      --supcon_weight 0.4 --supcon_temp 0.07 --focal_gamma 1.0 \
      --class_weight_beta 0.25

NOTE on clip length / TTA: processed clips are silence-TRIMMED and short
(~1.9 s mean, p95 2.7 s, max 4.3 s).  max_sec=4 truncates <0.2%; raising it
only pads.  Crop-based --tta_k is therefore a near no-op (clips are shorter
than the crop window, so every pass returns the same audio) — leave tta_k=1.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, recall_score)
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoFeatureExtractor, AutoModel,
                          get_cosine_schedule_with_warmup)

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False

warnings.filterwarnings(
    "ignore", message=".*mismatched key_padding_mask and attn_mask.*")

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META_CSV = os.path.join(ROOT, "ravdess_metadata.csv")
OUT_DIR  = os.path.join(ROOT, "reports", "ssl")
FIG_DIR  = os.path.join(ROOT, "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

EMOTION_ORDER = ["neutral", "calm", "happy", "sad",
                 "angry", "fearful", "disgust", "surprised"]
LBL2ID = {e: i for i, e in enumerate(EMOTION_ORDER)}
ID2LBL = {i: e for e, i in LBL2ID.items()}
SR     = 16_000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(s=0):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# --------------------------------------------------------------- data --------
def audio_duration_report(paths, max_sec):
    """Header-only scan (no full decode).  Warns when >10% of clips will be
    truncated, which can silently clip the emotional onset/offset."""
    durs = []
    for p in paths:
        try:
            durs.append(sf.info(p).duration)
        except Exception:
            pass
    if not durs:
        print("[audio-duration] could not read headers — skipping.", flush=True)
        return
    durs = np.array(durs)
    frac = float((durs > max_sec).mean())
    print(f"[audio-duration] n={len(durs)}  mean={durs.mean():.2f}s  "
          f"p50={np.percentile(durs,50):.2f}s  p90={np.percentile(durs,90):.2f}s  "
          f"p95={np.percentile(durs,95):.2f}s  max={durs.max():.2f}s  |  "
          f"max_sec={max_sec:.1f}s  truncates {frac*100:.1f}% of clips",
          flush=True)
    if frac > 0.1:
        print(f"[audio-duration] WARNING: >10% exceed max_sec={max_sec}s — "
              "consider raising --max_sec to the p95 above.", flush=True)


class WavDataset(Dataset):
    """Loads 16 kHz mono wavs.

    augment=True  (train):  RANDOM crop + gain / noise / micro time-shift.
                            Pitch and time-stretch are DISABLED by default
                            (they corrupt the prosody cues that encode emotion).
    augment=False (eval):   deterministic CENTRE crop, no augmentation.
    tta=True:               RANDOM crop only — used during test-time augmentation
                            to sample K crop windows and average probabilities.
    """

    def __init__(self, paths, labels, max_len,
                 augment=False, tta=False, seed=0,
                 pitch_shift_prob=0.0, pitch_shift_steps=2.0,
                 time_stretch_prob=0.0, time_stretch_range=0.1):
        self.paths, self.labels, self.max_len = paths, labels, max_len
        self.augment = augment
        self.tta     = tta
        self.rng     = np.random.RandomState(seed)
        self.pitch_shift_prob   = pitch_shift_prob   if _HAS_LIBROSA else 0.0
        self.pitch_shift_steps  = pitch_shift_steps
        self.time_stretch_prob  = time_stretch_prob  if _HAS_LIBROSA else 0.0
        self.time_stretch_range = time_stretch_range

    def __len__(self):
        return len(self.paths)

    def _crop(self, y):
        if len(y) <= self.max_len:
            return y
        if self.augment or self.tta:            # random window (train + TTA)
            s = self.rng.randint(0, len(y) - self.max_len + 1)
        else:                                    # centre window (eval)
            s = (len(y) - self.max_len) // 2
        return y[s:s + self.max_len]

    def _aug(self, y):
        """Prosody-preserving augmentations only."""
        if self.rng.rand() < 0.5:              # gain jitter
            y = y * self.rng.uniform(0.8, 1.2)
        if self.rng.rand() < 0.5:              # additive noise (high SNR)
            snr_db   = self.rng.uniform(25, 45)
            p_sig    = np.mean(y ** 2) + 1e-10
            sigma    = math.sqrt(p_sig / (10 ** (snr_db / 10)))
            y        = y + self.rng.randn(len(y)).astype(np.float32) * sigma
        if self.rng.rand() < 0.3:              # micro shift (≤50 ms)
            y = np.roll(y, self.rng.randint(-800, 800))
        # pitch / time-stretch: OFF by default (opt-in via CLI for ablations)
        if self.rng.rand() < self.pitch_shift_prob:
            steps = self.rng.uniform(-self.pitch_shift_steps,
                                     self.pitch_shift_steps)
            y = librosa.effects.pitch_shift(y, sr=SR, n_steps=steps)
        if self.rng.rand() < self.time_stretch_prob:
            rate = self.rng.uniform(1 - self.time_stretch_range,
                                    1 + self.time_stretch_range)
            y = librosa.effects.time_stretch(y, rate=rate)
        return y.astype(np.float32)

    def __getitem__(self, i):
        y, _ = sf.read(self.paths[i], dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = self._crop(y)
        if self.augment:
            y = self._aug(y)
        return y, self.labels[i]


def make_collate(fe, max_len):
    def collate(batch):
        arrays = [b[0] for b in batch]
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
        enc = fe(arrays, sampling_rate=SR, padding=True, max_length=max_len,
                 truncation=True, return_attention_mask=True, return_tensors="pt")
        enc["labels"] = labels
        return enc
    return collate


def _worker_init(worker_id):
    info = torch.utils.data.get_worker_info()
    if info is not None:
        info.dataset.rng = np.random.RandomState(info.seed % (2 ** 32))


# ------------------------------------------------------- SOTA loss modules ---
class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

    Why this beats extra augmentation for RAVDESS:
    - CE only separates classes via a linear boundary on the final logit layer.
    - SupCon acts directly in embedding space: it pulls every pair of 'calm'
      embeddings together and pushes them away from 'neutral', 'sad', etc.
    - Computed on PRE-MIX, clean embeddings so label integrity is maintained
      while hidden mixup regularises the classification head separately.

    features: (B, D) L2-normalised projections on the unit sphere.
    labels:   (B,)  integer class indices.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        B      = features.shape[0]
        if B < 2:
            return features.new_tensor(0.0)
        device = features.device
        labels = labels.contiguous().view(-1, 1)
        # positive mask: same class, excluding self-pairs
        mask_pos  = torch.eq(labels, labels.T).float().to(device)
        mask_self = torch.eye(B, device=device)
        mask_pos  = mask_pos * (1 - mask_self)

        n_pos = mask_pos.sum(dim=1)
        if not (n_pos > 0).any():
            # no positive pairs in this batch (e.g. all different classes)
            return features.new_tensor(0.0)

        # pairwise cosine similarities (features are already L2-normalised)
        dot     = torch.matmul(features, features.T) / self.temperature
        dot_max = dot.max(dim=1, keepdim=True).values.detach()
        exp_dot = torch.exp(dot - dot_max) * (1 - mask_self)
        log_prob = (dot - dot_max) - torch.log(exp_dot.sum(1, keepdim=True) + 1e-8)

        # mean log-prob over positives; only include anchors that have positives
        has_pos           = n_pos > 0
        mean_log_prob_pos = (mask_pos * log_prob).sum(1) / (n_pos + 1e-8)
        loss              = -mean_log_prob_pos[has_pos].mean()
        return loss


class FocalCrossEntropyLoss(nn.Module):
    """CE with class weights + label smoothing + focal modulation (1−p_t)^γ.

    Why this helps:
    - Standard CE gives equal gradient weight to easy (angry, disgust) and
      hard (neutral, calm, sad) examples.
    - Focal loss concentrates training on the hard confusable cluster.
    - γ=0 → standard weighted CE (backward-compatible).
    - γ=1.0 is a safe default; try 0.5 – 2.0.
    """

    def __init__(self, weight=None, label_smoothing=0.0, gamma=0.0):
        super().__init__()
        self.register_buffer("weight", weight)
        self.label_smoothing = label_smoothing
        self.gamma           = gamma

    def forward(self, logits, labels):
        per_sample = F.cross_entropy(
            logits, labels,
            weight=self.weight, label_smoothing=self.label_smoothing,
            reduction="none")                                    # (B,)
        if self.gamma == 0.0:
            return per_sample.mean()
        with torch.no_grad():
            pt = (torch.softmax(logits.float(), dim=-1)
                  .gather(1, labels.view(-1, 1)).squeeze(1)
                  .clamp(min=1e-6))
        return ((1.0 - pt) ** self.gamma * per_sample).mean()


# -------------------------------------------------------------- model --------
class SSLEmotionNet(nn.Module):
    """WavLM backbone + layer-weighted sum + attentive stats pooling.

    Three forward paths (all share the same pooled representation):
      encode()   → (B, 2·dim) pooled embedding, no dropout yet
      project()  → (B, supcon_dim) L2-normalised projection for SupCon
      classify() → (B, C) final logits through dropout + MLP head
      forward()  → classify(encode())   ← standard inference path
    """

    def __init__(self, model_name, num_labels, unfreeze_top, grad_ckpt,
                 p_drop=0.1, proj_hidden=256, supcon_dim=128,
                 attn_hidden=128, mask_time_prob=0.08, mask_feature_prob=0.0):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        cfg            = self.backbone.config
        cfg.use_cache  = False
        cfg.apply_spec_augment  = (mask_time_prob > 0) or (mask_feature_prob > 0)
        cfg.mask_time_prob      = mask_time_prob
        cfg.mask_feature_prob   = mask_feature_prob

        if hasattr(self.backbone, "feature_extractor"):
            self.backbone.feature_extractor._freeze_parameters()
        for p in self.backbone.parameters():
            p.requires_grad = False
        layers       = self.backbone.encoder.layers
        self.n_layers = len(layers)
        k             = min(unfreeze_top, self.n_layers)
        for layer in layers[-k:]:
            for p in layer.parameters():
                p.requires_grad = True
        if grad_ckpt:
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})

        dim = cfg.hidden_size
        self.layer_weights = nn.Parameter(
            torch.zeros(cfg.num_hidden_layers + 1))
        self.attn = nn.Sequential(
            nn.Linear(dim, attn_hidden), nn.Tanh(), nn.Linear(attn_hidden, 1))
        self.pool_drop  = nn.Dropout(p_drop)
        # primary head
        self.classifier = nn.Sequential(
            nn.Linear(2 * dim, proj_hidden), nn.ReLU(),
            nn.Dropout(p_drop), nn.Linear(proj_hidden, num_labels))
        # SupCon projection head (only used during training)
        self.proj_head = nn.Sequential(
            nn.Linear(2 * dim, proj_hidden), nn.GELU(),
            nn.Linear(proj_hidden, supcon_dim))
        self.unfrozen = k

    # ------------------------------------------------------------------ paths
    def encode(self, input_values, attention_mask=None, **_):
        """Backbone → layer-weighted sum → attentive stats pooling.
        Returns (B, 2·dim) BEFORE dropout so SupCon sees clean embeddings."""
        out = self.backbone(input_values, attention_mask=attention_mask,
                            output_hidden_states=True)
        hs  = torch.stack(out.hidden_states, dim=1)            # (B, L+1, T, D)
        w   = torch.softmax(self.layer_weights, dim=0).view(1, -1, 1, 1)
        x   = (hs * w).sum(dim=1)                             # (B, T, D)

        if attention_mask is not None:
            fmask = self.backbone._get_feature_vector_attention_mask(
                x.shape[1], attention_mask)
        else:
            fmask = torch.ones(x.shape[:2], device=x.device)

        e = self.attn(x).squeeze(-1)
        e = e.masked_fill(fmask == 0, float("-inf"))
        a = torch.softmax(e, dim=1).unsqueeze(-1)
        mean = (a * x).sum(dim=1)
        std  = torch.sqrt(((a * x.pow(2)).sum(1) - mean.pow(2)).clamp(min=1e-8))
        return torch.cat([mean, std], dim=-1)                  # (B, 2·dim)

    def project(self, pooled):
        """L2-normalised projection onto the unit sphere for SupCon."""
        return F.normalize(self.proj_head(pooled), dim=-1)

    def classify(self, pooled):
        return self.classifier(self.pool_drop(pooled))

    def forward(self, input_values, attention_mask=None, **_):
        return self.classify(self.encode(input_values, attention_mask))


def build_param_groups(model, enc_lr, head_lr, llrd):
    n = model.n_layers
    head, by_layer = [], {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ".encoder.layers." in name:
            idx = int(name.split(".encoder.layers.")[1].split(".")[0])
            by_layer.setdefault(idx, []).append(p)
        else:
            head.append(p)
    groups = [{"params": head, "lr": head_lr}]
    for idx, ps in by_layer.items():
        groups.append({"params": ps,
                       "lr": enc_lr * (llrd ** (n - 1 - idx))})
    return groups


# -------------------------------------------------------- train / eval -------
@torch.no_grad()
def evaluate(model, loader, max_batches=0):
    """Single forward pass.  Returns (probs (N,C), gts (N,))."""
    model.eval()
    probs, gts = [], []
    for bi, enc in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        labels = enc.pop("labels")
        enc    = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            logits = model(**enc)
        probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        gts.append(labels.numpy())
    return np.concatenate(probs), np.concatenate(gts)


def evaluate_tta(model, make_loader_fn, idx, tta_k, seed, max_batches=0):
    """Test-Time Augmentation: K random-crop passes, average softmax probs.

    Why TTA helps on RAVDESS:
    - Each test clip has a unique best crop window that captures the clearest
      emotional expression.  With a single centre crop we always miss it by
      the same amount every run.
    - Averaging K=5 random crops reduces that positional variance at zero
      training cost.
    """
    if tta_k <= 1:
        ldr = make_loader_fn(idx, shuffle=False, aug=False,
                             ds_seed=seed, tta=False)
        return evaluate(model, ldr, max_batches)
    all_probs, gts = [], None
    for k in range(tta_k):
        # use coprime stride (7919) so crop seeds spread widely
        ldr = make_loader_fn(idx, shuffle=False, aug=False,
                             ds_seed=seed + k * 7919, tta=True)
        p, g = evaluate(model, ldr, max_batches)
        all_probs.append(p)
        if gts is None:
            gts = g
    return np.mean(np.stack(all_probs, axis=0), axis=0), gts


def metric_pack(gts, preds):
    return {
        "acc":         float(accuracy_score(gts, preds)),
        "uar":         float(recall_score(gts, preds, average="macro",
                                          zero_division=0)),
        "macro_f1":    float(f1_score(gts, preds, average="macro",
                                      zero_division=0)),
        "weighted_f1": float(f1_score(gts, preds, average="weighted",
                                      zero_division=0)),
    }


def train_fold(model, tr_loader, va_loader, class_w, args, fold):
    """Train one fold with SupCon + Focal CE + hidden mixup.

    Loss structure per step
    -----------------------
    1. encode()       → pooled (B, 2·dim)   clean, pre-mix
    2. project(pooled) → z (B, D) L2-norm   → SupConLoss on CLEAN labels
    3. [optional] hidden mixup on pooled     → pooled_mix, labels_b
    4. classify(pooled_mix) → logits
    5. FocalCE(logits, labels_clean) * λ  +  FocalCE(logits, labels_b) * (1−λ)
    6. total = focal_ce + supcon_weight * supcon_loss
    """
    w_t   = torch.tensor(class_w, dtype=torch.float32, device=DEVICE)
    if args.class_weight_beta != 1.0:
        # Soften the balanced class weights: w**beta, renormalised to mean 1.
        # beta=1.0 → full 'balanced' (default).  beta=0.0 → uniform (no weighting).
        # Lower beta when a STRONG model over-predicts the minority class
        # (neutral): a fixed 2x weight that suited a weak model over-corrects a
        # better-fitting one, collapsing minority-class precision.
        w_t = w_t.clamp(min=1e-8) ** args.class_weight_beta
        w_t = w_t * (len(w_t) / w_t.sum())
    crit  = FocalCrossEntropyLoss(weight=w_t,
                                  label_smoothing=args.label_smoothing,
                                  gamma=args.focal_gamma)
    sc    = SupConLoss(temperature=args.supcon_temp)
    opt   = torch.optim.AdamW(
        build_param_groups(model, args.enc_lr, args.head_lr, args.llrd),
        weight_decay=args.weight_decay)
    n_steps = args.epochs * math.ceil(len(tr_loader) / args.accum)
    sched   = get_cosine_schedule_with_warmup(opt, int(0.1 * n_steps), n_steps)
    scaler  = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")

    history, best_f1, best_state, row_best = [], -1.0, None, 1
    for ep in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        t0 = time.perf_counter()
        tot_loss = tot_ce = tot_sc = 0.0
        correct = seen = 0

        for i, enc in enumerate(tr_loader):
            if args.max_train_batches and i >= args.max_train_batches:
                break
            labels_clean = enc.pop("labels").to(DEVICE)
            enc          = {k: v.to(DEVICE) for k, v in enc.items()}

            with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
                # — clean pooled embedding (shared for SupCon + CE) —
                pooled = model.encode(**enc)                   # (B, 2·dim)

                # — SupCon on pre-mix clean projections —
                sc_loss = (sc(model.project(pooled), labels_clean)
                           if args.supcon_weight > 0
                           else pooled.new_tensor(0.0))

                # — optional hidden-level mixup for CE —
                mix_lam, labels_b = 1.0, labels_clean
                do_mix = (args.mixup_alpha > 0
                          and labels_clean.numel() > 1
                          and args.mixup_level != "off")
                if do_mix and args.mixup_level == "hidden":
                    mix_lam = float(np.random.beta(args.mixup_alpha,
                                                   args.mixup_alpha))
                    mix_lam = max(mix_lam, 1.0 - mix_lam)
                    perm    = torch.randperm(labels_clean.size(0), device=DEVICE)
                    pooled  = mix_lam * pooled + (1 - mix_lam) * pooled[perm]
                    labels_b = labels_clean[perm]
                elif do_mix and args.mixup_level == "waveform":
                    # legacy A/B option — not recommended for RAVDESS
                    mix_lam = float(np.random.beta(args.mixup_alpha,
                                                   args.mixup_alpha))
                    mix_lam = max(mix_lam, 1.0 - mix_lam)
                    perm    = torch.randperm(labels_clean.size(0), device=DEVICE)
                    enc["input_values"] = (
                        mix_lam * enc["input_values"]
                        + (1 - mix_lam) * enc["input_values"][perm])
                    if enc.get("attention_mask") is not None:
                        enc["attention_mask"] = torch.maximum(
                            enc["attention_mask"],
                            enc["attention_mask"][perm])
                    labels_b = labels_clean[perm]

                # — focal CE (with mixed labels if mixup active) —
                logits  = model.classify(pooled)
                ce_loss = (mix_lam * crit(logits, labels_clean)
                           + (1 - mix_lam) * crit(logits, labels_b))

                total = (ce_loss + args.supcon_weight * sc_loss) / args.accum

            scaler.scale(total).backward()
            if (i + 1) % args.accum == 0 or (i + 1) == len(tr_loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()

            tot_loss += total.item() * args.accum
            tot_ce   += ce_loss.item()
            tot_sc   += float(sc_loss)
            correct  += (logits.detach().argmax(-1) == labels_clean).sum().item()
            seen     += labels_clean.numel()

        vp, vg = evaluate(model, va_loader, args.max_eval_batches)
        vm      = metric_pack(vg, vp.argmax(-1))
        n_b     = max(1, min(i + 1, args.max_train_batches or i + 1))
        gpu_mb  = (torch.cuda.max_memory_allocated() / 1e6) if DEVICE == "cuda" else 0
        row = {
            "fold": fold, "epoch": ep,
            "train_loss": round(tot_loss / n_b, 4),
            "train_ce":   round(tot_ce   / n_b, 4),
            "train_sc":   round(tot_sc   / n_b, 4),
            "train_acc":  round(correct  / max(seen, 1), 4),
            "val_acc":       round(vm["acc"],       4),
            "val_uar":       round(vm["uar"],       4),
            "val_macro_f1":  round(vm["macro_f1"],  4),
            "lr":    round(sched.get_last_lr()[0], 8),
            "sec":   round(time.perf_counter() - t0, 1),
            "gpu_mb": round(gpu_mb),
        }
        history.append(row)
        print(f"    ep {ep:2d}  loss={row['train_loss']:.3f}  "
              f"ce={row['train_ce']:.3f}  sc={row['train_sc']:.3f}  "
              f"tr_acc={row['train_acc']:.3f}  val_f1={row['val_macro_f1']:.3f}  "
              f"lr={row['lr']:.2e}  {row['sec']:.0f}s  {row['gpu_mb']}MB",
              flush=True)
        if vm["macro_f1"] > best_f1:
            best_f1   = vm["macro_f1"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            row_best   = ep
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, row_best


# ------------------------------------------------------------- report --------
def text_confusion(cm):
    head = "          " + " ".join(f"{e[:4]:>5}" for e in EMOTION_ORDER)
    rows = [f"{e[:9]:>9} " + " ".join(f"{cm[i,j]:5d}" for j in range(len(EMOTION_ORDER)))
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
        labels=EMOTION_ORDER, normalize="true", cmap="Blues",
        xticks_rotation=45, values_format=".2f", ax=ax, colorbar=False)
    m = metric_pack(gts, preds)
    ax.set_title(f"WavLM — speaker-independent\n"
                 f"acc={m['acc']:.3f}  UAR={m['uar']:.3f}  "
                 f"macro-F1={m['macro_f1']:.3f}")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "17_ssl_confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {os.path.relpath(path, ROOT)}", flush=True)


# --------------------------------------------------------------- main --------
def main():
    ap = argparse.ArgumentParser()
    # backbone / CV
    ap.add_argument("--model", default="microsoft/wavlm-base-plus")
    ap.add_argument("--mode",  choices=["cv", "quick", "smoke"], default="cv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n_seeds", type=int, default=1,
                    help="Run full CV this many times (seeds seed…seed+N-1) and "
                         "average softmax probs.  Multiplies training time by N.")
    ap.add_argument("--seed", type=int, default=0)
    # training
    ap.add_argument("--epochs",   type=int,   default=25)
    ap.add_argument("--batch",    type=int,   default=2)
    ap.add_argument("--accum",    type=int,   default=8)
    ap.add_argument("--max_sec",  type=float, default=4.0)
    ap.add_argument("--unfreeze_top", type=int, default=12)
    ap.add_argument("--grad_ckpt",    action="store_true")
    ap.add_argument("--num_workers",  type=int, default=0)
    ap.add_argument("--max_train_batches", type=int, default=0)
    ap.add_argument("--max_eval_batches",  type=int, default=0)
    # optimiser
    ap.add_argument("--enc_lr",        type=float, default=2e-5)
    ap.add_argument("--head_lr",       type=float, default=2e-4)
    ap.add_argument("--llrd",          type=float, default=0.9)
    ap.add_argument("--weight_decay",  type=float, default=0.01)
    # SOTA losses
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--class_weight_beta", type=float, default=1.0,
                    help="Softening exponent on balanced class weights.  "
                         "1.0=full 'balanced' (default).  0.5=sqrt-softened.  "
                         "0.0=uniform (no class weighting).  Lower this if a "
                         "strong model over-predicts the minority class "
                         "(neutral): precision collapses when a fixed 2x weight "
                         "over-corrects a well-fitting encoder.")
    ap.add_argument("--focal_gamma", type=float, default=0.0,
                    help="Focal γ.  0=standard CE.  Try 0.5–2.0.  "
                         "Focuses training gradient on hard examples "
                         "(neutral/calm/sad cluster).")
    ap.add_argument("--supcon_weight", type=float, default=0.0,
                    help="Weight of SupCon auxiliary loss.  0=off.  "
                         "Recommend 0.2–0.4.  Explicitly clusters embeddings "
                         "by emotion class in projection space.")
    ap.add_argument("--supcon_temp", type=float, default=0.07,
                    help="SupCon temperature.  0.07 is the standard default.")
    ap.add_argument("--supcon_dim",  type=int,   default=128,
                    help="Projection head output dimension for SupCon.")
    # augmentation (mild only by default)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--mask_time_prob",    type=float, default=0.08)
    ap.add_argument("--mask_feature_prob", type=float, default=0.05)
    ap.add_argument("--mixup_alpha", type=float, default=0.2)
    ap.add_argument("--mixup_level",
                    choices=["hidden", "waveform", "off"], default="hidden",
                    help="'hidden' (default): mix pooled embeddings — safe for "
                         "RAVDESS.  'waveform': legacy, not recommended.  "
                         "'off': disable mixup.")
    ap.add_argument("--pitch_shift_prob", type=float, default=0.0,
                    help="DEFAULT OFF.  Pitch shift corrupts emotion prosody on "
                         "RAVDESS.  Enable only for ablations.  Needs librosa.")
    ap.add_argument("--pitch_shift_steps",  type=float, default=2.0)
    ap.add_argument("--time_stretch_prob",  type=float, default=0.0,
                    help="DEFAULT OFF.  Time-stretch changes speech rate which "
                         "IS an emotion cue.  Enable only for ablations.")
    ap.add_argument("--time_stretch_range", type=float, default=0.1)
    # model head
    ap.add_argument("--p_drop",      type=float, default=0.1)
    ap.add_argument("--proj_hidden", type=int,   default=256)
    # inference
    ap.add_argument("--tta_k", type=int, default=1,
                    help="Test-Time Augmentation crops.  1=off (centre crop).  "
                         "K>1: average probs over K random crops.  "
                         "Try 5.  Free lunch — no training change needed.")
    ap.add_argument("--inner_val_frac", type=float, default=0.25,
                    help="Fraction of train actors held out for best-epoch "
                         "selection.  Default 0.25 (~5 actors) is far more "
                         "stable than the old hardcoded ~2 actors.")
    ap.add_argument("--run_tag", default="",
                    help="Optional suffix on the saved oof_probs_*.npz so two "
                         "runs of the SAME model (e.g. different hyper-params) "
                         "do not overwrite each other before ensembling.")
    args = ap.parse_args()

    if args.mode == "smoke":
        args.epochs = max(1, min(args.epochs, 2))
        args.max_train_batches = args.max_train_batches or 8
        args.max_eval_batches  = args.max_eval_batches  or 8
    args.n_seeds = max(1, args.n_seeds)
    if (args.pitch_shift_prob > 0 or args.time_stretch_prob > 0) and not _HAS_LIBROSA:
        warnings.warn("librosa not installed — pitch/time-stretch disabled.  "
                      "pip install librosa to enable.")

    for flag, val in [("SupCon",    f"weight={args.supcon_weight}  "
                                     f"temp={args.supcon_temp}  "
                                     f"dim={args.supcon_dim}"),
                      ("FocalCE",   f"gamma={args.focal_gamma}"),
                      ("TTA",       f"k={args.tta_k}")]:
        active = ((flag == "SupCon"  and args.supcon_weight > 0) or
                  (flag == "FocalCE" and args.focal_gamma   > 0) or
                  (flag == "TTA"     and args.tta_k         > 1))
        if active:
            print(f"[{flag}] {val}", flush=True)
    set_seed(args.seed)

    env = {
        "device": DEVICE,
        "torch":  torch.__version__,
        "gpu":    (torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"),
        "vram_GB": (round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
                    if DEVICE == "cuda" else 0),
    }
    print(json.dumps({"config": vars(args), "env": env}, indent=2), flush=True)
    with open(os.path.join(OUT_DIR, "config.json"), "w") as fh:
        json.dump({"config": vars(args), "env": env}, fh, indent=2)

    meta    = pd.read_csv(META_CSV)
    paths   = np.array([os.path.join(ROOT, p) for p in meta["processed_path"]])
    labels  = meta["emotion"].map(LBL2ID).values
    groups  = meta["actor"].values
    max_len = int(args.max_sec * SR)
    n_cls   = len(EMOTION_ORDER)

    audio_duration_report(paths, args.max_sec)

    fe      = AutoFeatureExtractor.from_pretrained(args.model)
    collate = make_collate(fe, max_len)

    def make_loader(idx, shuffle, aug, ds_seed, tta=False):
        ds = WavDataset(
            paths[idx], labels[idx], max_len,
            augment=aug, tta=tta, seed=ds_seed,
            pitch_shift_prob=args.pitch_shift_prob,
            pitch_shift_steps=args.pitch_shift_steps,
            time_stretch_prob=args.time_stretch_prob,
            time_stretch_range=args.time_stretch_range)
        return DataLoader(
            ds, batch_size=args.batch, shuffle=shuffle,
            collate_fn=collate, num_workers=args.num_workers,
            pin_memory=(DEVICE == "cuda"),
            worker_init_fn=(_worker_init if args.num_workers > 0 else None),
            persistent_workers=(args.num_workers > 0))

    splits = list(GroupKFold(n_splits=args.folds).split(paths, labels, groups))
    if args.mode in ("quick", "smoke"):
        splits = splits[:1]

    seed_oof_probs = []
    seed_summaries = []
    filled_mask    = np.zeros(len(labels), dtype=bool)
    all_fold_rows, all_history = [], []
    t_start = time.perf_counter()

    for seed in range(args.seed, args.seed + args.n_seeds):
        set_seed(seed)
        if args.n_seeds > 1:
            print(f"\n{'#'*60}\nSEED {seed}  "
                  f"({seed - args.seed + 1}/{args.n_seeds})\n{'#'*60}",
                  flush=True)
        oof_probs = np.zeros((len(labels), n_cls))

        for fold, (tr, te) in enumerate(splits, 1):
            test_actors = sorted(int(a) for a in set(groups[te]))
            print(f"\n=== fold {fold}/{len(splits)}  "
                  f"test_actors={test_actors} ===", flush=True)
            if DEVICE == "cuda":
                torch.cuda.reset_peak_memory_stats()

            # speaker-disjoint inner-val (best-epoch selection)
            uniq  = np.unique(groups[tr])
            rng   = np.random.RandomState(seed + fold)
            n_va  = max(2, int(round(len(uniq) * args.inner_val_frac)))
            n_va  = min(n_va, len(uniq) - 1)
            val_actors = set(rng.choice(uniq, size=n_va, replace=False))
            is_val = np.array([g in val_actors for g in groups[tr]])
            tr_idx, va_idx = tr[~is_val], tr[is_val]

            cw = compute_class_weight(
                "balanced", classes=np.arange(n_cls), y=labels[tr_idx])

            tr_ldr = make_loader(tr_idx, True,  args.augment, seed * 1000 + fold)
            va_ldr = make_loader(va_idx, False, False,         seed)

            model = SSLEmotionNet(
                args.model, n_cls, args.unfreeze_top, args.grad_ckpt,
                args.p_drop, args.proj_hidden, args.supcon_dim,
                mask_time_prob=args.mask_time_prob,
                mask_feature_prob=args.mask_feature_prob).to(DEVICE)

            if fold == 1 and seed == args.seed:
                ntr  = sum(p.numel() for p in model.parameters() if p.requires_grad)
                ntot = sum(p.numel() for p in model.parameters())
                print(f"  trainable {ntr/1e6:.1f}M / {ntot/1e6:.1f}M  "
                      f"(top {model.unfrozen}/{model.n_layers} layers  "
                      f"ckpt={args.grad_ckpt}  mixup={args.mixup_level}  "
                      f"supcon={args.supcon_weight}  γ={args.focal_gamma}  "
                      f"inner_val={n_va} actors  tta_k={args.tta_k})",
                      flush=True)

            model, hist, best_ep = train_fold(
                model, tr_ldr, va_ldr, cw, args, fold)
            for row in hist:
                row["seed"] = seed
            all_history.extend(hist)

            # TTA applied to TEST only (not inner-val)
            probs, _ = evaluate_tta(
                model, make_loader, te, args.tta_k, seed, args.max_eval_batches)
            fold_preds = probs.argmax(-1)
            te_gts     = labels[te]
            oof_probs[te]   = probs
            filled_mask[te] = True
            m = metric_pack(te_gts, fold_preds)
            all_fold_rows.append({
                "seed": seed, "fold": fold,
                "test_actors": str(test_actors),
                "best_epoch": best_ep,
                **{k: round(v, 4) for k, v in m.items()},
            })
            print(f"  >> seed {seed} fold {fold}  "
                  f"acc={m['acc']:.4f}  uar={m['uar']:.4f}  "
                  f"macro_f1={m['macro_f1']:.4f}  "
                  f"weighted_f1={m['weighted_f1']:.4f}", flush=True)
            del model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        seed_oof_probs.append(oof_probs)
        sm = metric_pack(labels[filled_mask],
                         oof_probs[filled_mask].argmax(-1))
        seed_summaries.append({"seed": seed,
                               **{k: round(v, 4) for k, v in sm.items()}})
        print(f"\n  == seed {seed} OOF  acc={sm['acc']:.4f}  "
              f"uar={sm['uar']:.4f}  macro_f1={sm['macro_f1']:.4f} ==",
              flush=True)

    # ---- ensemble across seeds ----
    ens_probs  = np.mean(np.stack(seed_oof_probs, axis=0), axis=0)
    gts        = labels[filled_mask]
    preds      = ens_probs[filled_mask].argmax(-1)
    agg        = metric_pack(gts, preds)
    cm         = confusion_matrix(gts, preds, labels=list(range(n_cls)))
    report     = classification_report(gts, preds, labels=list(range(n_cls)),
                                       target_names=EMOTION_ORDER,
                                       digits=3, zero_division=0)
    per_cls_f1 = f1_score(gts, preds, average=None,
                          labels=list(range(n_cls)), zero_division=0)
    elapsed    = round((time.perf_counter() - t_start) / 60, 1)

    # ---- persist ----
    res = pd.DataFrame(all_fold_rows)
    res.to_csv(os.path.join(OUT_DIR, "fold_results.csv"), index=False)
    pd.DataFrame(all_history).to_csv(
        os.path.join(OUT_DIR, "history.csv"), index=False)

    out_df = pd.DataFrame({"y_true": [ID2LBL[g] for g in gts],
                           "y_pred": [ID2LBL[p] for p in preds],
                           "actor": groups[filled_mask]})
    for ci, e in enumerate(EMOTION_ORDER):
        out_df[f"p_{e}"] = ens_probs[filled_mask][:, ci].round(4)
    out_df.to_csv(os.path.join(OUT_DIR, "oof_predictions.csv"), index=False)

    # raw OOF probabilities for cross-model / cross-run ensembling.
    # GroupKFold is deterministic, so every run shares the same test partition
    # and these matrices align row-by-row → scripts/ensemble_ssl.py averages them.
    tag = args.model.split("/")[-1] + f"_s{args.seed}-{args.seed + args.n_seeds - 1}"
    if args.run_tag:
        tag += "_" + args.run_tag
    tag = "".join(c if (c.isalnum() or c in "-._") else "_" for c in tag)
    np.savez(os.path.join(OUT_DIR, f"oof_probs_{tag}.npz"),
             probs=ens_probs, labels=labels, filled=filled_mask,
             model=args.model, macro_f1=agg["macro_f1"])
    print(f"saved oof_probs_{tag}.npz  "
          "(average backbones with: python scripts/ensemble_ssl.py)", flush=True)

    with open(os.path.join(OUT_DIR, "classification_report.txt"), "w") as fh:
        fh.write(report + "\n\n" + text_confusion(cm) + "\n")

    seed_f1s   = np.array([s["macro_f1"] for s in seed_summaries])
    fold_f1_std = float(res.groupby("seed")["macro_f1"].std().mean()) \
        if len(res) > 1 else 0.0
    summary = {
        "config": vars(args), "env": env, "elapsed_min": elapsed,
        "ensemble_oof": agg,
        "per_seed_oof": seed_summaries,
        "seed_macro_f1_mean": round(float(seed_f1s.mean()), 4),
        "seed_macro_f1_std":  round(float(seed_f1s.std()) if len(seed_f1s) > 1 else 0.0, 4),
        "fold_macro_f1_std_mean": round(fold_f1_std, 4),
        "per_class_f1": {e: round(float(f), 4)
                         for e, f in zip(EMOTION_ORDER, per_cls_f1)},
        "folds": all_fold_rows,
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    if args.mode == "cv":
        save_confusion_fig(gts, preds)

    # ---- COPY-BACK block ----
    print("\n" + "=" * 60)
    print("COPY THIS BACK TO CLAUDE")
    print("=" * 60)
    print(f"model={args.model} | mode={args.mode} | folds={len(splits)} | "
          f"n_seeds={args.n_seeds} | elapsed={elapsed}min | gpu={env['gpu']}")
    print(f"cfg: unfreeze={args.unfreeze_top} ep={args.epochs} "
          f"bs={args.batch}×{args.accum} max_sec={args.max_sec} "
          f"enc_lr={args.enc_lr} head_lr={args.head_lr} llrd={args.llrd} "
          f"ls={args.label_smoothing} cwbeta={args.class_weight_beta} "
          f"aug={args.augment} ckpt={args.grad_ckpt} "
          f"mixup={args.mixup_alpha}/{args.mixup_level} p_drop={args.p_drop}")
    print(f"SOTA: supcon={args.supcon_weight}/τ{args.supcon_temp} "
          f"focal_γ={args.focal_gamma} tta_k={args.tta_k} "
          f"inner_val_frac={args.inner_val_frac} "
          f"pitch_p={args.pitch_shift_prob} stretch_p={args.time_stretch_prob}")
    print("--- per (seed, fold) test ---")
    for r in all_fold_rows:
        print(f"  s{r['seed']}f{r['fold']} ep={r['best_epoch']:>2}  "
              f"acc={r['acc']:.4f}  uar={r['uar']:.4f}  "
              f"macroF1={r['macro_f1']:.4f}  actors={r['test_actors']}")
    if len(seed_summaries) > 1:
        print("--- per-seed OOF ---")
        for s in seed_summaries:
            print(f"  seed{s['seed']}  acc={s['acc']:.4f}  "
                  f"uar={s['uar']:.4f}  macroF1={s['macro_f1']:.4f}")
        print(f"  seed macroF1 = {seed_f1s.mean():.4f} ± {seed_f1s.std():.4f}  "
              f"across {len(seed_summaries)} runs")
    print(f"--- fold-to-fold macroF1 std (actor variance) "
          f"avg over seeds = {fold_f1_std:.4f} ---")
    print(f"--- ENSEMBLE OOF (n={filled_mask.sum()}) ---")
    print(f"  acc={agg['acc']:.4f}  uar={agg['uar']:.4f}  "
          f"macroF1={agg['macro_f1']:.4f}  weightedF1={agg['weighted_f1']:.4f}")
    print("--- per-class F1 (ensemble OOF) ---")
    print("  " + "  ".join(f"{e}={f:.3f}"
                            for e, f in zip(EMOTION_ORDER, per_cls_f1)))
    print("--- confusion matrix (rows=true, cols=pred) ---")
    print(text_confusion(cm))
    print("=" * 60)
    print(f"artifacts in {os.path.relpath(OUT_DIR, ROOT)}/ "
          "(summary.json fold_results.csv history.csv "
          "oof_predictions.csv classification_report.txt)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())