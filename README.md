<p align="center">
  <h1 align="center">MindCanvas</h1>
  <p align="center">
    <strong>Decode visual stimuli from EEG brain signals and reconstruct perceived images using deep learning.</strong>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" alt="Python">
    <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch" alt="PyTorch">
    <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
    <img src="https://img.shields.io/badge/EEG-BCI-orange" alt="BCI">
  </p>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Details](#pipeline-details)
  - [1. Signal Preprocessing](#1-signal-preprocessing)
  - [2. Wavelet Feature Extraction](#2-wavelet-feature-extraction)
  - [3. Deep Residual Classifier](#3-deep-residual-classifier)
  - [4. Cross-Validation](#4-cross-validation)
  - [5. GradCAM Interpretability](#5-gradcam-interpretability)
  - [6. Image Reconstruction](#6-image-reconstruction)
- [Configuration](#configuration)
- [Output](#output)
- [Project Structure](#project-structure)
- [Extending the Project](#extending-the-project)
- [Troubleshooting](#troubleshooting)
- [References](#references)
- [License](#license)

---

## Overview

**MindCanvas** is a Brain-Computer Interface (BCI) pipeline that answers the question:

> *Can we reconstruct what a person is looking at — purely from their brain waves?*

The system records **Visual Evoked Potentials (VEPs)** from a 14-channel EEG headset while a participant views images from four categories. It then:

1. **Decodes** the brain signal into a stimulus category using a deep residual CNN
2. **Reconstructs** a visual approximation of the perceived image via Stable Diffusion

```
 Visual Stimulus  →  EEG Recording  →  Deep CNN  →  Image Generation
   (Apple/Face/           (14-ch, 128Hz)      (ResNet-style)    (Stable Diffusion)
    Flower/Car)
```

### Categories

| Label | Category | Conditions | Prompt |
|-------|----------|------------|--------|
| 0 | Apple | A1, A2 | *"A red apple on a table"* |
| 1 | Human Face | P1, P2 | *"A human face portrait"* |
| 2 | Flower | F1, F2 | *"A flower in bloom"* |
| 3 | Car | C1, C2 | *"A car on the road"* |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MindCanvas Pipeline                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │  Raw EEG CSV  │───▶│  Bandpass    │───▶│  Z-Score              │  │
│  │  (14 channels)│    │  1–40 Hz     │    │  Normalisation        │  │
│  └──────────────┘    └──────────────┘    └───────────┬───────────┘  │
│                                                      │              │
│                                                      ▼              │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │  Stable       │◀───│  Prompt      │◀───│  Wavelet Decomposition│  │
│  │  Diffusion    │    │  Mapping     │    │  (5 bands × 14 ch     │  │
│  │  (Image Gen)  │    │              │    │   = 70 features)      │  │
│  └──────────────┘    └──────────────┘    └───────────┬───────────┘  │
│                                                      │              │
│                                                      ▼              │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │  GradCAM      │◀───│  Residual    │◀───│  1-Second Windows     │  │
│  │  Analysis     │    │  CNN         │    │  (128 samples)        │  │
│  └──────────────┘    └──────────────┘    └───────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Dataset

### Hardware

- **Headset:** [EMOTIV EPOC X](https://www.emotiv.com/products/epoc-x) — 14-channel wireless EEG
- **Sampling Rate:** 128 Hz
- **Channels:** AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4

### Data Organisation

```
VEP-DATA/
├── VEP-CSV/                     # Preprocessed CSV exports
│   ├── Apple/A1/, A2/           # 8 subjects each
│   ├── Car/C1/, C2/             # 8 subjects each
│   ├── Flower/F1/, F2/          # 8 subjects each
│   └── Human Face/P1/, P2/      # 8 subjects each
├── VEP-EDF/                     # Raw EDF recordings
├── Participant_info.xlsx        # Demographics
└── VVIQuestionnaire.pdf         # Vividness of Visual Imagery questionnaire
```

Each subject's CSV contains:
- `Timestamp` — Unix timestamp
- `EEG.*` — 14 raw EEG channel values (µV)
- `POW.*` — On-device FFT power bands (Theta, Alpha, BetaL, BetaH, Gamma) per channel
- `EEG.Counter` / `EEG.Interpolated` — Metadata columns (excluded from analysis)

> **Note:** The `VEP-DATA/` folder is excluded from Git due to size. Place it in the project root before running.

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended for Stable Diffusion; CPU-only for classification)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/amrrwael/mindcanvas.git
cd mindcanvas

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install Stable Diffusion for image generation
pip install diffusers transformers accelerate

# 5. Place your VEP-DATA/ folder in the project root
```

### Quick Verify

```bash
python -c "from main import Config, EEGClassifier; import torch; print('Setup OK — Device:', Config.DEVICE)"
```

---

## Quick Start

```bash
python main.py
```

This runs the complete pipeline:

1. Loads and preprocesses all 64+ EEG recordings
2. Extracts wavelet features (70 features per time-step)
3. Trains with 5-fold stratified cross-validation
4. Evaluates on a held-out test set
5. Generates GradCAM saliency maps
6. Reconstructs an image via Stable Diffusion (if installed)

All outputs (plots, model weights, images) are saved to `output/`.

**Expected runtime:**
| Component | CPU | GPU (CUDA) |
|-----------|-----|------------|
| 5-fold CV (20 epochs) | ~20–30 min | ~5–10 min |
| Stable Diffusion | N/A | ~30 sec/image |

---

## Pipeline Details

### 1. Signal Preprocessing

**Bandpass Filter (1–40 Hz)**

A 4th-order Butterworth filter removes:
- DC drift and slow artefacts below 1 Hz
- Muscle noise and line interference above 40 Hz

```python
from scipy.signal import butter, lfilter
b, a = butter(4, [1/64, 40/64], btype='band')  # nyq = 64 Hz at fs=128
filtered = lfilter(b, a, raw_eeg, axis=0)
```

**Z-Score Normalisation**

Each channel is independently standardised to zero mean and unit variance, removing baseline shifts between subjects and sessions.

### 2. Wavelet Feature Extraction

The **Discrete Wavelet Transform (DWT)** decomposes each EEG channel into 5 frequency sub-bands using the Daubechies-4 (`db4`) wavelet at level 4:

| Sub-band | Frequency Range | Associated Brain State |
|----------|----------------|----------------------|
| **D1** | 32–64 Hz | Gamma — cognitive binding, feature integration |
| **D2** | 16–32 Hz | Beta — alert attention, active thinking |
| **D3** | 8–16 Hz | Alpha — visual idling, relaxed awareness |
| **D4** | 4–8 Hz | Theta — memory encoding, drowsiness |
| **A4** | 0–4 Hz | Delta — deep sleep, slow cortical potentials |

**Result:** 14 channels × 5 bands = **70 features per time-step**

> Unlike FFT, DWT preserves temporal resolution, making it ideal for capturing transient VEP components like P100 and N200.

### 3. Deep Residual Classifier

A 3-block ResNet-style 1D-CNN processes the windowed EEG:

```
Input (B, 128, 70)
  │
  ├─ Stem: Conv1d(70→32, k=7) → BN → ReLU → MaxPool(2)
  │
  ├─ ResBlock 1: Conv(32→64) + skip connection → MaxPool(2)
  ├─ ResBlock 2: Conv(64→128) + skip connection → MaxPool(2)
  ├─ ResBlock 3: Conv(128→256) + skip connection → MaxPool(2)
  │
  ├─ Global Average Pooling → (B, 256)
  │
  └─ Classifier: FC(256→128) → ReLU → Dropout → FC(128→4)
```

**Total parameters:** ~741,348

**Key design choices:**
- **Residual connections** prevent vanishing gradients and allow deeper feature extraction
- **1D convolutions** along the time axis capture temporal VEP patterns
- **Adaptive average pooling** handles variable-length inputs
- **Gradient clipping** (max_norm=1.0) stabilises training on noisy EEG
- **AdamW** optimiser with weight decay (1e-4) for L2 regularisation
- **ReduceLROnPlateau** scheduler drops learning rate when validation stalls

### 4. Cross-Validation

**5-fold stratified cross-validation** ensures:
- Each fold maintains the same class distribution as the full dataset
- A fresh model is trained per fold (no data leakage)
- Best weights per fold are saved for potential ensembling
- Mean ± std accuracy provides a statistically reliable estimate

**Training hardening techniques:**
| Technique | Purpose |
|-----------|---------|
| Class-weight balancing | Compensates for uneven subject counts |
| Label smoothing (0.1) | Prevents overconfident predictions on noisy labels |
| Gradient clipping | Bounds gradient magnitude to prevent explosions |
| Weight decay (AdamW) | L2 regularisation against overfitting |
| LR scheduler | Automatic learning rate reduction on plateau |

### 5. GradCAM Interpretability

**Gradient-weighted Class Activation Mapping** highlights which time-steps in the 1-second EEG window most influenced the model's decision.

```
EEG Window (128 time-steps)
│████████░░░░░░██████░░░░░░░░░░░░░░░░░░░░│
        ↑                ↑
   High activation   High activation
   (P100 region)     (N200 region)
```

This validates whether the model is attending to physiologically plausible VEP components (typically 80–300 ms post-stimulus) or learning spurious artefacts.

### 6. Image Reconstruction

The predicted class is mapped to a descriptive text prompt, which is passed to [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) for image generation:

| Predicted Class | Prompt | Generated Image |
|-----------------|--------|-----------------|
| Apple | "A red apple on a table" | Photorealistic apple |
| Human Face | "A human face portrait" | Portrait photo |
| Flower | "A flower in bloom" | Flower close-up |
| Car | "A car on the road" | Car on road |

> This is a proof-of-concept. Advanced approaches would project EEG embeddings directly into the diffusion latent space (see [Extending the Project](#extending-the-project)).

---

## Configuration

All hyperparameters are centralised in the `Config` class at the top of `main.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FS` | 128 | EEG sampling rate (Hz) |
| `LOWCUT` / `HIGHCUT` | 1.0 / 40.0 | Bandpass filter cutoffs (Hz) |
| `WIN_SIZE` | 128 | Window length in samples (1 second) |
| `MAX_WIN_FILE` | 50 | Max windows extracted per subject file |
| `N_FOLDS` | 5 | Number of cross-validation folds |
| `EPOCHS` | 20 | Training epochs per fold |
| `BATCH_SIZE` | 32 | Mini-batch size |
| `LR` | 1e-3 | Initial learning rate |
| `WD` | 1e-4 | Weight decay (L2 regularisation) |
| `DROPOUT` | 0.3 | Dropout rate in classifier head |
| `WAVELET` | `'db4'` | Wavelet family for DWT |
| `WAVELET_LEVEL` | 4 | Decomposition depth |
| `SEED` | 42 | Random seed for reproducibility |

---

## Output

After a full run, the `output/` directory contains:

| File | Description |
|------|-------------|
| `training_curves.png` | Loss and accuracy curves across all folds |
| `confusion_matrix_test.png` | Heatmap of predictions vs. actual labels |
| `gradcam.png` | Saliency map showing model attention over time |
| `class_distribution.png` | Bar chart of windows per category |
| `best_fold{1-5}.pt` | Best model weights per CV fold |
| `generated_image.png` | Stable Diffusion reconstruction (if installed) |

---

## Project Structure

```
mindcanvas/
├── main.py                 # Complete pipeline (preprocessing → generation)
├── requirements.txt        # Python dependencies
├── .gitignore              # Git exclusion rules
├── README.md               # This file
├── output/                 # Generated artefacts (git-ignored)
│   ├── training_curves.png
│   ├── confusion_matrix_test.png
│   ├── gradcam.png
│   ├── class_distribution.png
│   ├── best_fold*.pt
│   └── generated_image.png
└── VEP-DATA/               # EEG dataset (git-ignored)
    ├── VEP-CSV/
    ├── VEP-EDF/
    ├── Participant_info.xlsx
    └── VVIQuestionnaire.pdf
```

---

## Extending the Project

Here are high-impact directions to build on this foundation:

### Signal Processing
- **Independent Component Analysis (ICA)** to remove eye-blink and muscle artefacts
- **Common Spatial Patterns (CSP)** for enhanced class discrimination
- **Channel importance analysis** — which of the 14 electrodes contribute most

### Model Architecture
- **Temporal Transformer encoder** to capture long-range dependencies in EEG
- **EEGNet-style depthwise separable convolutions** for parameter efficiency
- **Cross-subject domain adaptation** — train on all subjects, evaluate on held-out subjects

### Image Reconstruction
- **EEG-to-latent projection** — learn a mapping from EEG features to Stable Diffusion's latent space (bypassing text prompts)
- **CLIP-based evaluation** — measure cosine similarity between generated and original stimulus images
- **Contrastive learning (EEG-CLIP)** — align EEG embeddings with image embeddings in a shared space

### Evaluation
- **Per-subject accuracy breakdown** — identify which participants produce the most decodable signals
- **Time-resolved classification** — sliding window to find when stimulus information peaks
- **Permutation testing** — establish statistical significance against chance level (25%)

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: pywt` | `pip install PyWavelets` |
| `CUDA out of memory` | Reduce `BATCH_SIZE` in Config, or run on CPU |
| `VEP-DATA/ not found` | Ensure the dataset folder is in the project root |
| `Skipped X files` | Check CSV integrity — some recordings may be too short |
| `diffusers not installed` | Image generation is optional; classification still works |
| Low accuracy (< 40%) | Increase `EPOCHS`, check data quality, or reduce `MAX_WIN_FILE` |
| `ReduceLROnPlateau` warning | Harmless; LR just won't reduce further |

---

## References

1. **Lawhern, V. J. et al.** (2018). *EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces.* Journal of Neural Engineering, 15(5).
2. **Mallat, S. G.** (1989). *A Theory for Multiresolution Signal Decomposition: The Wavelet Representation.* IEEE TPAMI, 11(7).
3. **Selvaraju, R. R. et al.** (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV 2017.
4. **Rombach, R. et al.** (2022). *High-Resolution Image Synthesis with Latent Diffusion Models.* CVPR 2022.
5. **Subha, D. P. et al.** (2010). *EEG signal classification using wavelet feature extraction and a mixture of expert model.* Expert Systems with Applications.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with by <a href="https://github.com/amrrwael">Amr Wael</a>
</p>
