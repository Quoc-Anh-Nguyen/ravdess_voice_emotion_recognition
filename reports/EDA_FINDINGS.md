# EDA Findings — RAVDESS Speech Emotion (Audio-only)

Empirical analysis of 1440 cleaned clips (24 actors, 8 emotions). Every claim
below is backed by a figure in `figures/`. Figures are numbered to match the
order you would present them in a thesis.

---

## 1. Dataset-level analysis

**Class balance** ([01_emotion_distribution.png](../figures/01_emotion_distribution.png)).
Seven emotions have exactly **192** clips; **neutral has 96** — because RAVDESS
records neutral only at *normal* intensity (no *strong*). This is a mild,
structural imbalance (ratio 2:1, not severe).
→ *Implication:* report **macro-F1 / UAR**, not just accuracy, and consider class
weights. Do **not** oversample naively across speakers.

**Gender & intensity** ([02_gender_intensity.png](../figures/02_gender_intensity.png)).
Perfectly balanced gender (**720 / 720**). Intensity is balanced *strong vs
normal* for all emotions except neutral (normal only).
→ Gender is a confound for pitch (see §3) — keep it out of the label path and
verify it doesn't leak via speaker.

**Per-actor counts** ([03_utterances_per_actor.png](../figures/03_utterances_per_actor.png)).
Every actor contributes **60** clips (uniform). Clean grouping variable for
speaker-independent CV.

**Duration** ([04_duration_stats.png](../figures/04_duration_stats.png),
`dataset_summary.csv`). After silence-trimming: mean **1.90 s**, median 1.79 s,
std 0.42 s, range 1.15–4.25 s. Distribution is right-skewed and fairly tight.
→ *Implication:* a fixed model input of ~**3–4 s** (pad short / crop long) covers
almost all clips with little truncation. Duration differs only weakly by emotion,
so it is not a strong stand-alone cue (and is partly an artifact of acting pace).

---

## 2. Signal-level analysis (representative clips, actor 1)

**Waveforms** ([05_waveforms.png](../figures/05_waveforms.png)). High-arousal
emotions (angry, fearful, surprised) show larger, more abrupt amplitude envelopes
and faster onsets; neutral/calm/sad are smoother and lower-energy.

**STFT & mel spectrograms**
([06_stft_spectrograms.png](../figures/06_stft_spectrograms.png),
[07_mel_spectrograms.png](../figures/07_mel_spectrograms.png)). Active emotions
show more high-frequency energy and denser harmonic structure with steeper pitch
contours; calm/sad concentrate energy low and flat. This is the visual signature
that motivates **log-mel spectrograms as the DL input**.

**MFCC contrastive pairs** ([08_mfcc_pairs.png](../figures/08_mfcc_pairs.png)).
*angry vs calm* and *happy vs sad* differ visibly in the lower cepstral
coefficients and their temporal variability; *fearful vs neutral* differ in
dynamics. Confirms MFCCs carry usable emotion information.

---

## 3. Prosodic analysis ([09_prosody_by_emotion.png](../figures/09_prosody_by_emotion.png), [10_pitch_by_gender.png](../figures/10_pitch_by_gender.png))

A clear **arousal axis** emerges:
- **Pitch (F0):** highest and most variable for **fearful, angry, surprised,
  happy**; lowest for **calm, neutral, disgust, sad**.
- **RMS energy:** elevated for **surprised, calm, happy**; lowest for **sad,
  angry-normal, disgust** (energy alone does not separate anger — pitch/ZCR do).
- **ZCR:** highest for **angry, disgust, surprised** (harsher/noisier spectra),
  lowest for **neutral**.

**Pitch is strongly gender-dependent** ([10_pitch_by_gender.png]) — female F0 is
shifted up across all emotions. → Pitch features should be used *with* speaker
normalization, or the model may exploit gender/speaker rather than emotion.

→ *Interpretation for the thesis:* prosody encodes **arousal** well (high vs low)
but **valence** poorly (happy vs angry both high-arousal) — this predicts which
classes will confuse (see §5).

---

## 4. Feature-space structure

**Correlation** ([11_correlation_heatmap.png](../figures/11_correlation_heatmap.png)).
Spectral centroid / bandwidth / roll-off are mutually highly correlated (redundant
"brightness" family); F0 and ZCR are comparatively independent and thus
complementary. → Tree/linear models benefit from this redundancy awareness;
standardization + (optional) PCA helps.

**PCA** ([12_pca.png](../figures/12_pca.png)). First two PCs explain only
**14.5% + 7.5% ≈ 22%** of variance and classes **overlap heavily** — emotion is
not low-dimensional or linearly separable. **t-SNE / UMAP**
([13_tsne.png](../figures/13_tsne.png), [14_umap.png](../figures/14_umap.png))
show soft gradients (high- vs low-arousal regions) rather than crisp clusters,
with neutral/calm and happy/surprised/fearful interpenetrating.
→ *Implication:* simple linear classifiers will underperform; non-linear models
(SVM-RBF, CNN) and richer temporal features are needed. Expect a moderate ceiling
for hand-crafted features.

---

## 5. Baseline classification (speaker-independent)

`train_ml.py`, 5-fold **GroupKFold by actor** (`reports/ml_results.csv`,
[15_confusion_matrix.png](../figures/15_confusion_matrix.png),
[16_split_comparison.png](../figures/16_split_comparison.png)).

| Model | Speaker-indep. acc | macro-F1 | Random-split acc | Leakage gap |
|---|---|---|---|---|
| **SVM (RBF)** | **0.512 ± 0.032** | **0.500** | 0.694 | **+0.183** |
| RandomForest | 0.490 ± 0.062 | 0.450 | 0.619 | +0.129 |
| LogReg | 0.463 ± 0.046 | 0.457 | 0.579 | +0.117 |

**Headline methodological result:** the random split overstates accuracy by
**~18 points** (0.69 vs 0.51). Speaker identity leakage is the single biggest
reproducibility trap in SER — always group by actor.

**Per-class (SVM, confusion matrix):** best = **fearful 0.61, angry 0.60,
surprised 0.52, calm 0.55, disgust 0.54**; worst = **neutral 0.28, sad 0.39**.
Dominant confusions, all consistent with §3:
- **neutral → calm / happy** (low-arousal overlap; neutral also has fewest
  samples),
- **calm ↔ sad** (both low-arousal, low-energy),
- **happy ↔ fearful** (both high-arousal, similar pitch dynamics — valence not
  captured),
- **disgust → angry** (shared harshness/ZCR).

→ *Interpretation:* the model separates **arousal** well but struggles with
**valence** and with the under-represented neutral class — exactly what the
prosody and projection analyses predicted.

---

## 6. Takeaways for modelling
1. Use **speaker-independent CV** (GroupKFold for dev, LOSO for the final number);
   report macro-F1 with mean ± std.
2. Hand-crafted features cap around **~0.50 macro-F1** speaker-independent →
   motivates **log-mel CNN / CRNN** and especially **SSL fine-tuning**
   (wav2vec2 / WavLM / emotion2vec) to capture valence and temporal structure.
3. Address **neutral imbalance** (class weights) and **gender/pitch confound**
   (speaker normalization).
4. Expect and *discuss* the intrinsic confusions (neutral/calm, happy/fear) rather
   than over-tuning to remove them.

See [MODEL_DESIGN.md](MODEL_DESIGN.md) for the full architecture and training plan.
