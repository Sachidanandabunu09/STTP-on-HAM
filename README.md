# STTP-Net: Long-Tail Skin Lesion Classification on HAM10000

A PyTorch implementation of **STTP-Net** — a dual-head long-tail learning framework for dermoscopy image classification on the HAM10000 dataset. The pipeline addresses severe class imbalance (58× ratio between majority and minority classes) using a combination of contrastive representation learning, conditional generation, tail-aware sampling, hybrid augmentation, and balanced loss functions.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Dataset](#2-dataset)
3. [Architecture](#3-architecture)
4. [Component Details](#4-component-details)
5. [Training Strategy](#5-training-strategy)
6. [Augmentation Strategy](#6-augmentation-strategy)
7. [Experiment Log](#7-experiment-log)
8. [Comparison with Related Work](#8-comparison-with-related-work)
9. [Setup](#9-setup)
10. [Project Structure](#10-project-structure)
11. [Future Work](#11-future-work)

---

## 1. Problem Statement

Skin lesion classification from dermoscopy images is a clinically critical task. The HAM10000 dataset exhibits **severe long-tail distribution** — the most common class (melanocytic nevi) has 6705 training samples while the rarest class (dermatofibroma) has only 115. Standard training on such data causes:

- Model bias toward head classes
- Poor recall on clinically important rare classes
- Inflated overall accuracy masking poor tail performance
- Especially dangerous for melanoma — a potentially fatal class with medium frequency

**Our goal:** Maximize balanced accuracy across all 7 classes simultaneously, with particular focus on tail and median classes without sacrificing head class performance.

---

## 2. Dataset

**HAM10000** (Human Against Machine with 10000 training images) — ISIC skin cancer dermoscopy dataset.

| Class | Label | Train Count | Group | Clinical Importance |
|-------|-------|-------------|-------|---------------------|
| Melanocytic nevi | nv | 5609 | **Head** | Benign — lowest risk |
| Melanoma | mel | 781 | **Median** | Malignant — highest risk |
| Benign keratosis | bkl | 770 | **Median** | Generally benign |
| Basal cell carcinoma | bcc | 360 | **Tail** | Malignant |
| Actinic keratosis | akiec | 229 | **Tail** | Pre-malignant |
| Vascular lesions | vasc | 99 | **Tail** | Mostly benign |
| Dermatofibroma | df | 80 | **Tail** | Benign |

**Split:** 70% train / 15% val / 15% test — stratified by class

**Class grouping thresholds:**
- Head: frequency > 1000
- Median: 500 ≤ frequency ≤ 1000
- Tail: frequency < 500

---

## 3. Architecture

```
Input Image  (3 × 224 × 224)
      │
      ▼
┌─────────────────────────────────┐
│   Shared Encoder                │
│   ResNet-50 (ImageNet pretrained│
│   + Projection MLP)             │
│   Output: f ∈ ℝ^512             │
└────────────┬────────────────────┘
             │
     ┌───────┴────────┐
     │                │
     ▼                ▼
┌─────────┐    ┌──────────────┐
│  CVAE   │    │  SupCon      │
│  z∈ℝ128 │    │  Projection  │
│         │    │  128-d       │
└────┬────┘    └──────┬───────┘
     │                │
  CVAE Loss      SupCon Loss
  (ELBO)         (pulls same-class
                  embeddings together)
     │
     └──────────────────────────────┐
                                    │
              f (512-d encoder output)
                    │
          ┌─────────┴──────────┐
          │                    │
          ▼                    ▼
   ┌─────────────┐    ┌─────────────┐
   │   Head-1    │    │   Head-2    │
   │ head+median │    │    tail     │
   │ specialist  │    │ specialist  │
   └──────┬──────┘    └──────┬──────┘
          │                  │
       EBS Loss           EBS Loss
          │                  │
          └────────┬─────────┘
                   │
           Norm-scaled fusion
           (l1/‖w1‖ + l2/‖w2‖) × 0.5
                   │
           Final Prediction
```

**Full pipeline:**
```
Image → Pretrained Encoder → CVAE + SupCon → Class Grouping →
Tail-Aware Samplers → HybridMix → Dual-Head Classifier →
EBS Loss → Norm-Fused Prediction
```

---

## 4. Component Details

### 4.1 Shared Encoder

- **Backbone:** ResNet-50 pretrained on ImageNet (IMAGENET1K_V2 weights)
- **Projection MLP:** Linear(2048→1024) → BatchNorm → ReLU → Linear(1024→512)
- **Output:** 512-dimensional feature vector `f`
- **Normalization:** ImageNet mean/std `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`

### 4.2 CVAE (Conditional Variational Autoencoder)

Encodes `(feature, class_label)` → latent vector `z ∈ ℝ^128`

```
Encoder: Linear(512+7 → 256) → μ, log_var ∈ ℝ^128
Reparametrize: z = μ + ε·exp(0.5·log_var),  ε ~ N(0,I)
Decoder: Linear(128+7 → 256) → Linear(256 → 512)
Loss: ELBO = MSE(f_recon, f) + β·KL(q(z|f,y) || p(z))
```

**Purpose:**
- Learns a structured class-conditional latent space
- Enables class-aware representation refinement
- KL term forces compact, well-organized embeddings per class

### 4.3 Supervised Contrastive Loss (SupCon)

Pulls same-class embeddings together and pushes different-class embeddings apart:

```
L_supcon = -1/|P(i)| · Σ_{p∈P(i)} log[
    exp(z_i · z_p / τ) /
    Σ_{a≠i} exp(z_i · z_a / τ)
]

τ = 0.07 (temperature)
```

**Purpose:** Organizes the latent space so classes form tight, well-separated clusters — visible in t-SNE visualization.

### 4.4 Three Samplers

| Sampler | Weight formula | Target | Effect |
|---------|---------------|--------|--------|
| Instance | w = 1 (uniform) | Head classes | Natural distribution |
| Median | w = clip(median_n / freq, 1, 5) | Median classes | Moderate upsampling |
| Reverse | w = 1/freq (normalized) | Tail classes | Strong upsampling |

**Usage:** Instance + Median batches → Head-1. Reverse batches → Head-2.

### 4.5 HybridMix Augmentation

Two different mixing strategies applied based on class group:

**CMeO (CutMix-style) — for Head + Median samples:**
```python
# Random bounding box cut
mixed[:, :, y1:y2, x1:x2] = imgs_b[:, :, y1:y2, x1:x2]
lam_adj = 1 - (box_area / total_area)
```

**Mixup — for Tail samples:**
```python
mixed = lam * imgs_a + (1 - lam) * imgs_b
# lam ~ Beta(0.4, 0.4)
```

**Why different strategies:**
- Head/median classes have enough samples — CutMix adds spatial diversity
- Tail classes are rare — Mixup creates smoother interpolations, more synthetic diversity

### 4.6 Dual-Head Classifier

```
Head-1 (head + median specialist):
  Dropout(0.4) → Linear(512→256) → ReLU → Dropout(0.2) → Linear(256→7)
  Trained on: instance + median batches (with CMeO HybridMix)

Head-2 (tail specialist):
  Dropout(0.4) → Linear(512→256) → ReLU → Dropout(0.2) → Linear(256→7)
  Trained on: reverse sampler batches (with Mixup HybridMix)

Inference fusion:
  fused = (logits1 / ‖W1‖ + logits2 / ‖W2‖) × 0.5
```

**Why norm-scaled fusion:** Normalizing by weight matrix Frobenius norm balances the scale of each head's outputs, preventing one head from dominating the final prediction.

### 4.7 Effective Balanced Softmax Loss (EBS)

Adjusts the softmax with log-frequency prior to counteract class imbalance:

```
adj_logits = logits + log(n_i)
L_EBS = CrossEntropy(adj_logits, labels)

n_i = training frequency of class i
```

Classes with fewer samples get a higher prior, effectively making the classifier more willing to predict rare classes.

### 4.8 Total Loss

```
L_total = L_head1 + L_head2 + 0.3 × L_supcon + 0.2 × L_cvae

L_head1 = mixed_CE(logits1, la, lb, λ)   # HybridMix interpolated loss
L_head2 = mixed_CE(logits2, la, lb, λ)   # HybridMix interpolated loss
L_supcon = Supervised Contrastive Loss on clean instance batch
L_cvae   = ELBO reconstruction + KL divergence
```

### 4.9 Two-Phase Training

| Phase | Epochs | HybridMix | Purpose |
|-------|--------|-----------|---------|
| Phase 1 | 1–35 | ON | Learn rich mixed representations |
| Phase 2 | 36–45 | OFF | Sharpen class boundaries on clean samples |

### 4.10 Test-Time Augmentation (TTA)

At inference, average predictions over 5 forward passes:
- Pass 1: clean image (no augmentation)
- Passes 2–5: randomly augmented views (flip, rotate, mild color jitter)

```
final_prob = (1/5) × Σ softmax(fused_logits(aug_i(x)))
```

---

## 5. Training Strategy

| Setting | Value |
|---------|-------|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 1e-4 |
| LR schedule | CosineAnnealingLR (η_min=1e-6) |
| Gradient clipping | max_norm=2.0 |
| Batch size | 16 |
| Image size | 224×224 |
| Encoder dim | 512 |
| Latent dim | 128 |

---

## 6. Augmentation Strategy

Based on dermoscopy literature — only clinically safe augmentations are used.

### What is safe for dermoscopy

| Augmentation | Parameters | Reason safe |
|---|---|---|
| Horizontal Flip | p=0.5 | Lesions have no canonical orientation |
| Vertical Flip | p=0.5 | Same — no up/down in dermoscopy |
| RandomRotate90 | p=0.5–0.7 | Arbitrary acquisition angle |
| Rotate | limit=25–35°, p=0.6–0.7 | Small angle variation |
| ShiftScaleRotate | shift=0.05–0.07, scale=0.05–0.07 | Minor position/scale variation |
| ColorJitter | brightness/contrast/saturation=0.1–0.15, hue=0.02–0.03 | Device variation simulation |
| RandomGamma | gamma=(85–115), p=0.3–0.4 | Mild illumination variation |

### What was tried and rejected

| Augmentation | Reason rejected |
|---|---|
| ElasticTransform | Destroys asymmetry — primary melanoma indicator |
| GaussianBlur / MotionBlur | Destroys microstructures and texture |
| GaussNoise | Corrupts pigment networks and borders |
| CoarseDropout | Erases lesion borders — key diagnostic regions |
| Heavy ColorJitter (hue>0.05) | Creates biologically impossible colors |
| RandomShadow | Creates false dark regions mimicking lesions |
| HAM10000 normalization | Tiny std values destabilize pretrained ResNet weights |

### Class-conditional augmentation

Tail and median classes (mel, bkl, bcc, akiec, vasc, df) receive stronger augmentation than head classes (nv) — higher rotation limits and color jitter probabilities — to increase diversity of rare samples without creating more nv samples.

---

## 7. Experiment Log

### exp01 — Baseline
**Date:** May 2026
**Settings:**
- Basic torchvision augmentation (HFlip, VFlip, ColorJitter(0.2), Rotation(20))
- ImageNet normalization
- BATCH_SIZE=32, NUM_WORKERS=4
- EPOCHS: 30 (Phase1) + 10 (Phase2) = 40 total
- Single-pass evaluation

**Results:**

| Metric | Score |
|--------|-------|
| Test Accuracy | **0.9042** |
| Test Balanced Accuracy | **0.8686** |
| Macro F1 | 0.8538 |

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| nv | 0.9524 | 0.9538 | 0.9531 | 671 |
| mel | 0.7429 | 0.6964 | 0.7189 | 112 |
| bkl | 0.8545 | 0.8545 | 0.8545 | 110 |
| bcc | 0.8654 | 0.8654 | 0.8654 | 52 |
| akiec | 0.7143 | 0.7812 | 0.7463 | 32 |
| vasc | 0.7647 | 0.9286 | 0.8387 | 14 |
| df | 1.0000 | 1.0000 | 1.0000 | 11 |

**Key observations:**
- Strong overall performance despite severe imbalance
- df hit 100% recall with only 11 test samples — reverse sampler working well
- mel recall (69.6%) is the weakest point — clinically most important
- vasc at 92.9% recall — excellent for a tail class

**Status:** ✅ Best result so far — baseline to beat

---

### exp02 — Aggressive Augmentation (Interrupted)
**Date:** May 2026
**Changes from exp01:**
- Added heavy albumentations pipeline
- ElasticTransform (p=0.3–0.5)
- GaussNoise, Blur, CoarseDropout, RandomShadow
- Heavy ColorJitter (hue=0.15–0.2)
- BATCH_SIZE=64 → OOM → reduced to 16
- 40+15 epochs planned

**Why interrupted:**
- Out of memory error on RTX 5050 8GB with BATCH_SIZE=64
- After analysis, realized the augmentation pipeline was medically harmful
- ElasticTransform destroys asymmetry — key melanoma diagnostic feature
- Blur destroys microstructures critical for dermoscopy
- Heavy hue shift creates biologically impossible lesion colors

**Status:** ❌ Interrupted — augmentation strategy was fundamentally wrong

---

### exp03 — Wrong Normalization
**Date:** May 2026
**Changes from exp01:**
- Removed harmful augmentations ✅
- Changed normalization to HAM10000-specific stats:
  `mean=[0.7636, 0.5462, 0.5705], std=[0.0883, 0.1172, 0.1315]`
- CenterCrop(224) after Resize(256)
- 40+15 epochs

**Results:**

| Metric | Score |
|--------|-------|
| Test Accuracy | 0.8084 |
| Test Balanced Accuracy | 0.7611 |

| Class | Recall | vs exp01 |
|-------|--------|----------|
| nv | 0.891 | -6.3% |
| mel | 0.411 | **-28.5%** |
| bkl | 0.709 | -14.5% |
| bcc | 0.789 | -7.6% |
| akiec | 0.781 | ≈same |
| vasc | 0.929 | ≈same |
| df | 0.818 | **-18.2%** |

**Root cause of failure:**
The HAM10000 std values `[0.0883, 0.1172, 0.1315]` are extremely small. Dividing pixel values by these tiny numbers produced massive input values that completely destabilized the pretrained ResNet-50 weights. The pretrained ImageNet features were destroyed.

**Rule learned:** Always use ImageNet normalization when using ImageNet pretrained weights. HAM10000-specific normalization only makes sense when training from scratch.

**Status:** ❌ Failed — worst result, -9.6% accuracy vs exp01

---

### exp04 — Safe Augmentation + ImageNet Norm
**Date:** May 2026
**Changes from exp01:**
- Albumentations pipeline — only clinically safe transforms
- Class-conditional augmentation — tail/median classes get stronger aug
- ImageNet normalization restored ✅
- BATCH_SIZE=16, NUM_WORKERS=0 (Windows stability)
- 35 (Phase1) + 10 (Phase2) = 45 total epochs
- TTA ×5 at test evaluation
- Experiment folder tracking (config.json saved per run)

**Results:**

| Metric | Score | vs exp01 |
|--------|-------|----------|
| Test Accuracy | **0.9072** | +0.3% ✅ |
| Test Balanced Accuracy | 0.8561 | -1.3% ⚠️ |
| Macro Precision | **0.8659** | +2.4% ✅ |
| Macro F1 | **0.8605** | +0.7% ✅ |

| Class | Recall | vs exp01 |
|-------|--------|----------|
| nv | 0.9598 | +0.6% ✅ |
| mel | 0.7321 | +3.6% ✅ |
| bkl | 0.8091 | -4.6% ⚠️ |
| bcc | 0.9038 | +3.9% ✅ |
| akiec | 0.7500 | -3.1% ⚠️ |
| vasc | 0.9286 | ≈same |
| df | 0.9091 | -9.1% ⚠️ |

**Key wins:** mel +3.6%, bcc crossed 90%, overall accuracy +0.3%, macro precision +2.4%

**Key losses:** bkl -4.6% (tail_aug too aggressive for median class), balanced acc -1.3% (tail variance at small n)

**Root cause of balanced acc dip:** df (n=11) going 100%→90.9% = 1 wrong sample but pulls balanced metric significantly due to equal class weighting.

**Status:** ✅ Completed

---

### exp05 — EfficientNet-B4 + CLAHE + Focal+EBS Loss (Running)
**Date:** May 2026

**Changes from exp04:**

| Component | exp04 | exp05 |
|-----------|-------|-------|
| Encoder | ResNet-50 | **EfficientNet-B4** |
| in_features | 2048 | 1792 |
| Preprocessing | None | **CLAHE** (clip=2.0, tile=8×8) |
| Loss | EBS only | **Focal + EBS** (w=0.5 each) |
| Focal gamma | — | 2.0 |
| Focal alpha | — | 0.25 |
| Phase 1 epochs | 35 | **40** |
| tail_aug rotation | 35° | **30°** (recalibrated) |

**Why each change:**

**EfficientNet-B4:**
Compound scaling balances depth/width/resolution simultaneously. Stronger dermoscopy feature extraction, lighter than Swin Transformer, proven on medical imaging. in_features = 1792 vs 2048 for ResNet-50. Expected to close bcc gap and improve mel.

**CLAHE (Contrast Limited Adaptive Histogram Equalization):**
Applied on L channel only in LAB color space — preserves color information while enhancing local contrast. Standard dermoscopy preprocessing in clinical practice. Makes pigment networks, borders, and microstructures more visible. Applied consistently to train, val, test, and TTA images.

**Focal Loss combined with EBS:**
```
L_FocalEBS = 0.5 x L_EBS + 0.5 x L_Focal
L_Focal = -alpha x (1-pt)^gamma x log(pt)   gamma=2.0, alpha=0.25
```
EBS corrects for class frequency. Focal additionally downweights easy examples and focuses on hard ones. mel is hard not just due to frequency but visual similarity to nv — focal loss specifically targets these hard-to-classify samples.

**Recalibrated tail_aug (30° instead of 35°):**
In exp04, bkl (median class) dropped 4.6% — tail_aug rotation was too aggressive. Reduced to 30° to balance diversity vs stability. Should recover bkl while maintaining mel/bcc gains.

**Targets:** mel recall > 0.80, bkl recall > 0.85, bcc > 0.93, balanced acc > 0.87

**Status:** 🔄 Currently running

---

## 8. Comparison with Related Work

### vs. YOLOv11 + GAN (Fitzpatrick17k)

| Metric | YOLOv11+GAN | Our exp01 |
|--------|-------------|-----------|
| Overall Accuracy | **96.8%** | 90.4% |
| Macro Recall | **96.3%** | 86.9% |
| Dataset | Balanced (200/class) | Imbalanced (115–6705) |
| Training data | 50% synthetic (GAN) | 100% real |
| Real-world validity | Limited* | High |

*Their own paper admits: *"GAN images may not fully capture the complexity and subtle variations of real world lesions"*

**Key context:** Their 96.8% accuracy is on a perfectly balanced dataset with 50% synthetic training data. Our 90.4% is on severely imbalanced real clinical data — a significantly harder problem.

---

### vs. Swin Transformer + CVAE + SVM

| Metric | Swin+CVAE+SVM | exp01 (baseline) | exp04 (best so far) |
|--------|--------------|------------------|---------------------|
| Overall Accuracy | 90.0% | 90.4% | **90.7%** |
| Macro Recall | 86.0% | 86.9% | 85.6% |
| Macro Precision | 88.0% | 84.2% | **86.6%** |
| Macro F1 | **87.0%** | 85.4% | 86.1% |
| Test samples | 2003 | 1002 | 1002 |

**Per-class recall — three-way comparison:**

| Class | Swin+SVM | exp01 | exp04 | exp04 vs Swin |
|-------|----------|-------|-------|----------------|
| nv | 96.0% | 95.4% | **96.0%** | Tied ✅ |
| mel | 70.0% | 69.6% | **73.2%** | +3.2% ✅ |
| bkl | 78.0% | **85.5%** | 80.9% | +2.9% ✅ |
| bcc | **94.0%** | 86.5% | 90.4% | -3.6% ⚠️ |
| akiec | 78.0% | **78.1%** | 75.0% | -3.0% ⚠️ |
| vasc | 93.0% | **92.9%** | 92.9% | Tied ✅ |
| df | 91.0% | **100%** | 90.9% | Tied ✅ |

**Key insight:** Despite using ResNet-50 (older backbone vs Swin Transformer), exp04 matches or beats Swin+SVM on 5 out of 7 classes and achieves higher overall accuracy. The main remaining gap is bcc (90.4% vs 94%) — targeted by EfficientNet-B4 in exp05.

---

## 9. Setup

### Requirements
```bash
conda create -n sttp_net python=3.10 -y
conda activate sttp_net

# PyTorch — CUDA 12.8 for RTX 40/50 series (Blackwell)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# All other dependencies
pip install -r requirements.txt
```

### Data preparation
Download HAM10000 from Kaggle and place as:
```
data/
└── raw/
    ├── images/        # all .jpg dermoscopy images
    └── metadata.csv   # HAM10000_metadata.csv
```

Pre-split CSVs (train/val/test) are included in `data/splits/` — use these for reproducibility.

### Run
Open `notebooks/STTP_Net_HAM10000.ipynb` in VS Code, select the `sttp_net` kernel, and run all cells top to bottom.

**Important:** Always use **Restart Kernel and Run All Cells** — never run cells individually as constants defined in Cell 1 are required by all subsequent cells.

---

## 10. Project Structure

```
sttp_net_ham10000/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── STTP_Net_HAM10000.ipynb     # main notebook
├── data/
│   └── splits/
│       ├── train.csv               # stratified train split
│       ├── val.csv                 # stratified val split
│       └── test.csv                # stratified test split
└── experiments/
    ├── exp01_baseline/
    │   ├── config.json
    │   ├── best_model.pt
    │   ├── training_curves.png
    │   ├── confusion_matrix.png
    │   ├── per_class_acc.png
    │   ├── tsne_latent.png
    │   └── results_summary.json
    ├── exp03_wrong_norm/           # kept for reference
    │   └── results_summary.json
    ├── exp04_safe_aug_imagenet_norm/   # completed ✅
    │   ├── config.json
    │   ├── best_model.pt
    │   ├── training_curves.png
    │   ├── confusion_matrix.png
    │   ├── per_class_acc.png
    │   ├── tsne_latent.png
    │   └── results_summary.json
    └── exp05_efficientnetb4_clahe_focal/   # currently running 🔄
        └── ...
```

**Note:** Raw images (`data/raw/images/`) and model checkpoints (`*.pt`) are excluded from git via `.gitignore` due to file size.

---

## 11. Future Work

### exp05 — Currently Running
- [x] Replace ResNet-50 with EfficientNet-B4 encoder
- [x] Add CLAHE preprocessing for dermoscopy contrast enhancement
- [x] Combine Focal Loss with EBS for hard sample focus
- [x] Recalibrate tail_aug rotation (35° → 30°) to recover bkl
- [ ] Results pending — target mel > 0.80, bkl > 0.85, balanced acc > 0.87

### exp06 — If exp05 still struggles with mel
- [ ] Increase mel-specific sampling weight in reverse sampler
- [ ] Add mel as pseudo-tail class (force stronger upsampling)
- [ ] Longer Phase 1 (50 epochs) for more HybridMix exposure

### Longer term
- [ ] Swin Transformer encoder (if compute allows)
- [ ] Diffusion-based synthetic augmentation for extreme tail classes
- [ ] GradCAM / LIME explainability visualization
- [ ] Clinical validation on external dermoscopy datasets (ISIC 2019, BCN20000)
- [ ] Lightweight model distillation for edge deployment

---

## License

MIT

---

## Acknowledgements

- HAM10000 dataset: Tschandl et al., 2018
- SupCon Loss: Khosla et al., 2020 (NeurIPS)
- Balanced Softmax: Ren et al., 2020
- STTP-Net architecture inspired by long-tail learning literatureA