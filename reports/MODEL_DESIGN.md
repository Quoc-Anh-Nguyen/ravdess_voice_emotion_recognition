# Model & Algorithm Design — RAVDESS Speech Emotion Recognition

This document covers the design decisions that follow the EDA: data-splitting
strategy, augmentation, feature engineering, and the model architectures
(classical ML → deep learning → self-supervised), plus the training/evaluation
protocol and known pitfalls. It is written to be lifted directly into the
*Methods* chapter of a thesis.

---

## 1. Task definition

8-way single-label classification of an utterance into
`{neutral, calm, happy, sad, angry, fearful, disgust, surprised}`.
1440 clips, 24 actors, balanced by gender. Inputs are mono 16 kHz waveforms
(cleaned: silence-trimmed, peak-normalized). The dominant constraint is **small
data with speaker structure** — every modelling choice below is driven by it.

---

## 2. Data-splitting strategy

| Strategy | Description | Verdict for SER |
|---|---|---|
| **Random split** | Shuffle all 1440 clips, split rows. | ❌ Leaks speaker identity: the same actor (and even the same sentence) appears in train and test. The model partly learns *who is speaking*, inflating accuracy by a large margin (we quantify this in `figures/16_split_comparison.png`). Use only as an upper-bound sanity check, never as the headline number. |
| **Speaker-independent (group) split** | Partition by **actor id** so no actor is shared between train/test (`GroupKFold`/`StratifiedGroupKFold`). | ✅ **Recommended default.** Measures generalization to unseen speakers — the realistic deployment condition. This is the headline metric. |
| **LOSO-CV** (Leave-One-Speaker-Out) | 24 folds, each holding out one actor. | ✅ **Recommended for the final reported result.** Gold standard in SER: maximal training data per fold, per-speaker variance is visible, and it removes the luck of a particular fold assignment. Cost: 24× training (cheap for classical ML, expensive for DL — use 5-fold grouped CV during development and LOSO only for the final table). |

**Recommendation.** Develop and tune with **5-fold speaker-independent
`GroupKFold`**; report the **final** number with **LOSO-CV** (and report mean ±
std across folds, plus macro-F1, not just accuracy). Keep actors split so that
gender stays roughly balanced across folds. Our baseline (`train_ml.py`) uses
5-fold `GroupKFold` and prints the random-split number alongside purely to expose
the leakage gap.

> Pitfall: RAVDESS has only 2 sentences. Even a speaker-independent split shares
> *lexical content* between train and test, so absolute numbers are optimistic
> relative to spontaneous speech. Disclose this.

---

## 3. Data augmentation

Augmentation is applied **on the training folds only**, after the split, never
before (otherwise augmented copies of a test clip leak into training).

| Method | What it does | Benefit | Risk for *acted* emotion |
|---|---|---|---|
| **Additive noise** (Gaussian / MUSAN / RIR) | Mix background noise / reverb | Robustness to recording conditions; strong regularizer | RAVDESS is studio-clean; heavy noise can move it off-distribution. Keep SNR high (15–30 dB). |
| **Time shift** | Roll waveform in time | Cheap, label-safe | Minimal risk; avoid wrapping speech past clip edges. |
| **Pitch shift** | Shift F0 by ±n semitones | Speaker/gender invariance | **High risk**: pitch *is* an emotion cue (anger/fear raise F0). Large shifts can change perceived emotion. Use small (±1–2 semitones). |
| **Time stretch** | Change tempo without pitch | Speaking-rate invariance | Speaking rate encodes arousal; keep rate in 0.9–1.1×. |
| **SpecAugment** (on spectrograms) | Mask time/freq bands | Best-in-class for spectrogram CNNs; cannot distort prosody globally | Low risk; **preferred** augmentation for the DL track. |

**Recommendation.** For the DL spectrogram models use **SpecAugment + mild
additive noise**; treat pitch/tempo shifts as small-magnitude only. Always run an
ablation (with vs without augmentation) so the thesis can report its effect.
General risk: augmentation on a small acted corpus can *erase* the very prosodic
cues that define the label — validate that augmented samples still sound like the
target emotion.

---

## 4. Feature engineering recommendations

### 4.1 Traditional ML (the `features_tabular.csv` set, 192 dims)
| Feature | Why it helps SER |
|---|---|
| **MFCC (40)** + mean/std | Compact spectral-envelope / timbre summary; the workhorse SER feature. |
| **Δ-MFCC, ΔΔ-MFCC** | Capture temporal dynamics (transitions), which carry arousal information lost in static means. |
| **Chroma (12)** | Pitch-class energy; weakly related to intonation patterns. |
| **Spectral contrast (7)** | Peak-vs-valley energy per sub-band; separates harmonic (voiced, calm) vs noisy (harsh, angry) spectra. |
| **Spectral centroid / bandwidth / roll-off** | "Brightness" and spread — higher for tense/high-arousal emotions. |
| **ZCR** | Proxy for noisiness/voicing; rises for fricatives, harsh anger. |
| **RMS energy** | Loudness ≈ arousal; one of the single most discriminative cues. |
| **F0 (pitch) mean/std + voiced ratio** | Intonation: high/variable F0 → high arousal (fear, surprise, anger); flat/low → sad, neutral. |

These are aggregated to one vector per clip (mean/std over frames) → ideal for
SVM / RF / gradient boosting.

### 4.2 Deep learning input representations
| Representation | Pros | Cons |
|---|---|---|
| **Log-mel spectrogram** (we precomputed `128×T`) | Best accuracy/effort trade-off; 2-D → CNNs; perceptually motivated | Needs fixed-size handling (pad/crop) or pooling. **Recommended default.** |
| **Raw waveform** (1-D CNN, e.g. SincNet) | No hand-tuned front-end; learns filters | Data-hungry; risky on 1440 clips without pretraining. |
| **MFCC "images"** (we precomputed `120×T` with Δ/ΔΔ) | Smaller, denoised input; fast | Discards fine spectral detail vs mel; usually a touch below log-mel for CNNs. |

### 4.3 Self-supervised (SSL) embeddings
With **wav2vec 2.0 / HuBERT / WavLM / emotion2vec**, manual feature engineering
is largely **unnecessary** — feed the (cleaned, 16 kHz) waveform and use the
model's hidden states. Best practice:
- **Freeze the encoder + train a lightweight head** first (logistic / attentive
  pooling) given the small dataset; fine-tune top layers only if it helps.
- Use a **weighted sum of layers** (different layers encode different info;
  emotion peaks in middle/upper layers for wav2vec2/WavLM).
- **emotion2vec** is purpose-built for emotion and is a strong frozen baseline.
- EDA for SSL = run PCA/t-SNE/UMAP **on the pooled embeddings** (same scripts,
  swap the feature matrix) to inspect class separability before training.

---

## 5. Model architectures

### 5.1 Classical ML baseline (implemented — `train_ml.py`)
`StandardScaler → {SVM-RBF, LogisticRegression, RandomForest}`, evaluated with
speaker-independent `GroupKFold`. SVM-RBF is the typical winner on aggregated
MFCC+prosody. Serves as the reference the deep models must beat. (Empirical
numbers in §7.)

### 5.2 Deep learning — recommended progression
1. **CNN on log-mel** (baseline DL): 3–4 conv blocks (Conv→BN→ReLU→Pool) →
   global average pool → FC → softmax. Handle variable length by padding/cropping
   to a fixed `T` (e.g. 3 s ≈ 188 frames at hop 256/16 kHz) **or** global pooling
   to be length-agnostic. Dropout + weight decay; class-weighted loss for the
   neutral imbalance.
2. **CRNN / CNN-BiLSTM**: CNN front-end + recurrent layer to model temporal
   evolution of emotion.
3. **CNN + attention / transformer encoder**: attentive temporal pooling — often
   the best non-pretrained architecture on RAVDESS.

Input length strategy: **pad/truncate to a fixed duration** (cover ~95th pct of
durations, ~4 s) for batching; reflect-pad short clips, center-crop long ones.
Alternatively use global pooling and variable length with bucketing.

### 5.3 SSL fine-tuning (recommended for best accuracy)
`wav2vec2/WavLM/HuBERT (frozen) → layer-weighted mean → attentive statistics
pooling → FC → softmax`. This is the current SOTA recipe on RAVDESS and the
recommended final model; expect a large jump over hand-crafted features.

---

## 6. Training & evaluation protocol
- **Metrics:** report **both** Weighted Accuracy (WA) and **Unweighted
  Accuracy / macro-F1 (UAR)** — neutral has half the samples, so macro metrics
  matter. Add a confusion matrix (`figures/15_confusion_matrix.png`).
- **CV:** 5-fold `GroupKFold` for development; LOSO for the final number; report
  mean ± std.
- **Optimization (DL):** AdamW, cosine/step LR, early stopping on a held-out
  speaker-disjoint validation fold, class-weighted cross-entropy, mixed precision.
- **Reproducibility:** fix seeds, log the exact fold→actor mapping.

---

## 7. Empirical baseline results
5-fold speaker-independent `GroupKFold` (by actor), `reports/ml_results.csv`:

| Model | Speaker-indep. acc | macro-F1 | Random-split acc | Leakage gap |
|---|---|---|---|---|
| **SVM (RBF)** | **0.512 ± 0.032** | **0.500** | 0.694 | **+0.183** |
| RandomForest | 0.490 ± 0.062 | 0.450 | 0.619 | +0.129 |
| LogReg | 0.463 ± 0.046 | 0.457 | 0.579 | +0.117 |

- **SVM-RBF wins** the hand-crafted-feature track at **0.512 acc / 0.500
  macro-F1** speaker-independent. This is the bar the deep models must beat.
- **Leakage gap ≈ +18 pts** (random 0.694 vs grouped 0.512) — quantitative proof
  that random splitting inflates SER results; see
  `figures/16_split_comparison.png`.
- Per-class (confusion matrix `figures/15_confusion_matrix.png`): strong on
  arousal-distinct classes (fearful 0.61, angry 0.60), weak on neutral (0.28,
  confused with calm/happy) and sad (0.39). Full interpretation in
  `reports/EDA_FINDINGS.md` §5.

**Conclusion:** hand-crafted features plateau near 0.50 macro-F1 → proceed to the
log-mel CNN/CRNN and SSL fine-tuning tracks (§5.2–5.3) to capture valence and
temporal dynamics that aggregated features miss.

---

## 8. Pitfalls & limitations of RAVDESS for SER
1. **Acted, not spontaneous** — prototypical, exaggerated emotions; models may
   not transfer to natural speech. State this as an external-validity limit.
2. **Only 2 sentences** — lexical content leaks across any split; report results
   as content-dependent.
3. **24 actors** — small speaker pool; speaker-independent variance is high →
   always report std / use LOSO.
4. **North-American English, studio-clean** — domain-narrow; add noise/RIR aug or
   external test sets for robustness claims.
5. **calm vs neutral** and **happy vs surprised / fear vs sad** confusions are
   intrinsic — expect them in the confusion matrix; discuss rather than over-tune.
6. **Class imbalance** (neutral = 96 vs 192) — use macro-F1 and class weights.
7. **Leakage** is the #1 reproducibility error in SER papers — never random-split.
