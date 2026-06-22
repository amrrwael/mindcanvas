import os, glob, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
from scipy.signal import butter, lfilter
import matplotlib
matplotlib.use('Agg')  # non-interactive; use 'TkAgg' for live windows
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False

class Config:
    SEED          = 42
    DATA_ROOT     = r"VEP-DATA\VEP-CSV"
    FS            = 128            # Sampling rate (Hz)
    LOWCUT        = 1.0            # Bandpass low cutoff (Hz)
    HIGHCUT       = 40.0           # Bandpass high cutoff (Hz)
    FILTER_ORDER  = 4
    WIN_SIZE      = 128            # 1-second windows at 128 Hz
    MAX_WIN_FILE  = 50             # Cap windows per file for balance
    N_FOLDS       = 5
    EPOCHS        = 20
    BATCH_SIZE    = 32
    LR            = 1e-3
    WD            = 1e-4           # weight decay (L2 regularization)
    DROPOUT       = 0.3
    WAVELET       = 'db4'
    WAVELET_LEVEL = 4
    N_CLASSES     = 4
    DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    OUT_DIR       = "output"

    CATEGORIES = {
        0: {"name": "Apple",      "subs": ["A1", "A2"], "prompt": "A red apple on a table"},
        1: {"name": "Human Face", "subs": ["P1", "P2"], "prompt": "A human face portrait"},
        2: {"name": "Flower",     "subs": ["F1", "F2"], "prompt": "A flower in bloom"},
        3: {"name": "Car",        "subs": ["C1", "C2"], "prompt": "A car on the road"},
    }

    # 14 standard EMOTIV EPOC X channels (excluding Counter & Interpolated)
    EEG_CHANNELS = [
        "EEG.AF3","EEG.F7","EEG.F3","EEG.FC5","EEG.T7","EEG.P7","EEG.O1",
        "EEG.O2","EEG.P8","EEG.T8","EEG.FC6","EEG.F4","EEG.F8","EEG.AF4",
    ]

# Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Signal Processing 
def _butter_bandpass(low, high, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return b, a

def bandpass(data, low=1.0, high=40.0, fs=128, order=4):
    b, a = _butter_bandpass(low, high, fs, order)
    return lfilter(b, a, data, axis=0)

def wavelet_decompose(signal, wavelet='db4', level=4):
    """
    Decompose each channel into 5 frequency sub-bands via DWT:
      D1  32-64 Hz (Gamma)   D2  16-32 Hz (Beta)
      D3   8-16 Hz (Alpha)   D4   4-8 Hz  (Theta)
      A4   0-4 Hz  (Delta)
    Returns array of shape [T, 5 * n_channels].
    """
    try:
        import pywt
    except ImportError:
        raise ImportError("pip install PyWavelets")
    n_ch = signal.shape[1]
    bands = []
    for ch in range(n_ch):
        coeffs = pywt.wavedec(signal[:, ch], wavelet, level=level)
        # coeffs = [cA4, cD4, cD3, cD2, cD1]
        for c in coeffs:
            bands.append(c)
    # Pad shorter bands to the max length so they stack cleanly
    max_len = max(b.shape[0] for b in bands)
    padded = np.array([np.pad(b, (0, max_len - len(b))) for b in bands])
    return padded.T.astype(np.float32)          # [T, 5*n_ch]

# Data Loading
def load_all_data(cfg: Config):
    """
    Dynamically discover every CSV under DATA_ROOT, apply bandpass filtering,
    z-score normalisation, and wavelet decomposition, then window & stack.
    """
    all_X, all_y, stats = [], [], {"total_files": 0, "total_windows": 0, "skipped": []}

    for label, info in cfg.CATEGORIES.items():
        for sub in info["subs"]:
            folder = os.path.join(cfg.DATA_ROOT, info["name"], sub)
            for fp in sorted(glob.glob(os.path.join(folder, "*.csv"))):
                stats["total_files"] += 1
                try:
                    df = pd.read_csv(fp)
                    present = [c for c in cfg.EEG_CHANNELS if c in df.columns]
                    if not present:
                        stats["skipped"].append(fp); continue

                    raw = df[present].fillna(0).values.astype(np.float32)
                    if len(raw) < cfg.WIN_SIZE * 2:
                        stats["skipped"].append(fp); continue

                    # 1) Bandpass 1-40 Hz
                    raw = bandpass(raw, cfg.LOWCUT, cfg.HIGHCUT, cfg.FS, cfg.FILTER_ORDER)
                    # 2) Z-score per channel
                    raw = (raw - raw.mean(0)) / (raw.std(0) + 1e-6)
                    # 3) Wavelet multi-band features
                    feats = wavelet_decompose(raw, cfg.WAVELET, cfg.WAVELET_LEVEL)
                    # 4) Window
                    n_win = min(feats.shape[0] // cfg.WIN_SIZE, cfg.MAX_WIN_FILE)
                    if n_win == 0:
                        stats["skipped"].append(fp); continue
                    seg = feats[:n_win * cfg.WIN_SIZE].reshape(n_win, cfg.WIN_SIZE, -1)
                    all_X.append(seg)
                    all_y.append(np.full(n_win, label, dtype=np.int64))
                    stats["total_windows"] += n_win
                except Exception as e:
                    stats["skipped"].append(fp)
                    print(f"  ⚠  {os.path.basename(fp)}: {e}")

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    return X, y, stats

# Model Architecture 
class ResBlock(nn.Module):
    """1-D Residual block with matched skip-connection (stride-aware)."""
    def __init__(self, in_c, out_c, ks=5, pool=2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, ks, padding=ks // 2)
        self.bn1   = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, ks, padding=ks // 2)
        self.bn2   = nn.BatchNorm1d(out_c)
        self.pool  = nn.MaxPool1d(pool)
        self.skip  = nn.Sequential(
            nn.Conv1d(in_c, out_c, 1),
            nn.BatchNorm1d(out_c),
            nn.MaxPool1d(pool),
        )
    def forward(self, x):
        residual = self.skip(x)              # (B, out_c, T/pool)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))      # (B, out_c, T)
        out = self.pool(out)                 # (B, out_c, T/pool)
        return F.relu(out + residual)

class EEGClassifier(nn.Module):
    """
    Deep residual CNN for EEG classification.
    Input : (B, T, F)  T=time-samples, F=features-per-sample
    Output: (B, n_classes) logits
    """
    def __init__(self, in_features, n_classes=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_features, 32, 7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.block1 = ResBlock(32, 64)
        self.block2 = ResBlock(64, 128)
        self.block3 = ResBlock(128, 256)
        self.gap    = nn.AdaptiveAvgPool1d(1)
        self.head   = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)        # (B, F, T)
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x).squeeze(-1)    # (B, 256)
        return self.head(x)

    # GradCAM
    def gradcam(self, x):
        """Return input-space saliency map via gradient-weighted class activation."""
        acts, grads = {}, {}
        target = self.block3

        def _fwd(m, i, o): acts['v'] = o
        def _bwd(m, gi, go): grads['v'] = go[0]

        h1 = target.register_forward_hook(_fwd)
        h2 = target.register_full_backward_hook(_bwd)

        x_t = x.clone().requires_grad_(True)
        logits = self(x_t)
        top = logits.argmax(1)
        self.zero_grad(); logits[0, top[0]].backward()

        a = acts['v'].detach()
        g = grads['v'].detach()
        w = g.mean(-1, keepdim=True)
        cam = (w * a).sum(1).squeeze(0)
        cam = F.relu(cam).cpu().numpy()

        h1.remove(); h2.remove()
        return cam, top.item()

# Train / Evaluate
def train_one_epoch(model, loader, opt, criterion, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # gradient clipping
        opt.step()
        loss_sum += loss.item() * xb.size(0)
        correct  += (out.argmax(1) == yb).sum().item()
        total    += xb.size(0)
    return loss_sum / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, preds, labels = 0.0, [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        loss_sum += criterion(out, yb).item() * xb.size(0)
        preds.append(out.argmax(1).cpu())
        labels.append(yb.cpu())
    all_p = torch.cat(preds).numpy()
    all_l = torch.cat(labels).numpy()
    return (loss_sum / len(all_p),
            accuracy_score(all_l, all_p),
            f1_score(all_l, all_p, average='macro'),
            all_p, all_l)

# Visualisations
def plot_training_curves(history, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for h in history:
        fold = h['fold']
        ax1.plot(h['tr_loss'], label=f'F{fold} train')
        ax1.plot(h['val_loss'], '--', label=f'F{fold} val')
        ax2.plot(h['tr_acc'],  label=f'F{fold} train')
        ax2.plot(h['val_acc'], '--', label=f'F{fold} val')
    ax1.set(xlabel='Epoch', ylabel='Loss', title='Loss Curves'); ax1.legend(fontsize=8)
    ax2.set(xlabel='Epoch', ylabel='Accuracy', title='Accuracy Curves'); ax2.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, 'training_curves.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f"  ✓ Saved {path}")

def plot_confusion_matrix(y_true, y_pred, names, out_dir, tag=""):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    if HAS_SNS:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=names, yticklabels=names, ax=ax)
    else:
        im = ax.imshow(cm, cmap='Blues')
        for i in range(len(cm)):
            for j in range(len(cm)):
                ax.text(j, i, str(cm[i,j]), ha='center', va='center')
        plt.colorbar(im)
        ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names); ax.set_yticklabels(names)
    ax.set(xlabel='Predicted', ylabel='Actual', title=f'Confusion Matrix {tag}')
    plt.tight_layout()
    path = os.path.join(out_dir, f'confusion_matrix{tag}.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f"  ✓ Saved {path}")

def plot_gradcam(cam, out_dir):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(cam)
    ax.set(xlabel='Time Step', ylabel='Activation',
           title='GradCAM — model attention over time')
    ax.fill_between(range(len(cam)), cam, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, 'gradcam.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f"  ✓ Saved {path}")

def plot_class_distribution(y, names, out_dir):
    unique, counts = np.unique(y, return_counts=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([names[u] for u in unique], counts, color='steelblue')
    ax.set(xlabel='Category', ylabel='Windows', title='Class Distribution')
    for u, c in zip(unique, counts):
        ax.text(u, c + 10, str(c), ha='center', fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'class_distribution.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f"  ✓ Saved {path}")

# Main
def main():
    cfg = Config()
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    set_seed(cfg.SEED)
    device = cfg.DEVICE
    cat_names = [cfg.CATEGORIES[i]["name"] for i in range(cfg.N_CLASSES)]
    print(f"Device: {device}")

    # Load & process all data 
    print("\n[1/5] Loading and preprocessing all EEG data …")
    X, y, stats = load_all_data(cfg)
    print(f"  Files loaded : {stats['total_files']}")
    print(f"  Total windows: {stats['total_windows']}")
    print(f"  Window shape : ({cfg.WIN_SIZE}, {X.shape[2]})")
    print(f"  Dataset shape: {X.shape}")
    print(f"  Skipped files: {len(stats['skipped'])}")
    plot_class_distribution(y, cat_names, cfg.OUT_DIR)

    # Build model 
    print("\n[2/5] Initialising model …")
    model = EEGClassifier(X.shape[2], cfg.N_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Class-weight balancing (handles uneven subject counts per category)
    counts = np.bincount(y, minlength=cfg.N_CLASSES).astype(np.float32)
    weights = torch.tensor(1.0 / (counts + 1e-6), device=device)
    weights = weights / weights.sum() * cfg.N_CLASSES
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    # Stratified K-Fold Cross-Validation
    print(f"\n[3/5] {cfg.N_FOLDS}-fold stratified cross-validation …")
    skf = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED)
    history, fold_acc, fold_f1 = [], [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        print(f"\n  ── Fold {fold+1}/{cfg.N_FOLDS} ──")
        tr_ds = TensorDataset(
            torch.tensor(X[tr_idx], dtype=torch.float32),
            torch.tensor(y[tr_idx], dtype=torch.long))
        va_ds = TensorDataset(
            torch.tensor(X[va_idx], dtype=torch.float32),
            torch.tensor(y[va_idx], dtype=torch.long))
        tr_dl = DataLoader(tr_ds, cfg.BATCH_SIZE, shuffle=True)
        va_dl = DataLoader(va_ds, cfg.BATCH_SIZE)

        # Fresh model + optimiser per fold
        fm = EEGClassifier(X.shape[2], cfg.N_CLASSES).to(device)
        opt = torch.optim.AdamW(fm.parameters(), lr=cfg.LR, weight_decay=cfg.WD)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)

        h = {"fold": fold+1, "tr_loss": [], "val_loss": [], "tr_acc": [], "val_acc": []}
        best_acc = 0.0
        for ep in range(cfg.EPOCHS):
            tr_l, tr_a = train_one_epoch(fm, tr_dl, opt, criterion, device)
            va_l, va_a, va_f, _, _ = evaluate(fm, va_dl, criterion, device)
            sch.step(va_l)
            h["tr_loss"].append(tr_l); h["val_loss"].append(va_l)
            h["tr_acc"].append(tr_a);  h["val_acc"].append(va_a)
            if va_a > best_acc:
                best_acc = va_a
                torch.save(fm.state_dict(), os.path.join(cfg.OUT_DIR, f"best_fold{fold+1}.pt"))
            print(f"    Ep {ep+1:2d} | tr_loss={tr_l:.4f} tr_acc={tr_a:.4f}"
                  f" | va_loss={va_l:.4f} va_acc={va_a:.4f} va_f1={va_f:.4f}")
        fold_acc.append(best_acc); fold_f1.append(va_f)
        history.append(h)

    plot_training_curves(history, cfg.OUT_DIR)
    print(f"\n  CV Accuracy : {np.mean(fold_acc):.4f} ± {np.std(fold_acc):.4f}")
    print(f"  CV F1-macro : {np.mean(fold_f1):.4f} ± {np.std(fold_f1):.4f}")

    # Final model training
    print("\n[4/5] Training final model on full dataset …")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=cfg.SEED, stratify=y)
    tr_dl = DataLoader(TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.long)), cfg.BATCH_SIZE, shuffle=True)
    te_dl = DataLoader(TensorDataset(
        torch.tensor(X_te, dtype=torch.float32),
        torch.tensor(y_te, dtype=torch.long)), cfg.BATCH_SIZE)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WD)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

    for ep in range(cfg.EPOCHS):
        tr_l, tr_a = train_one_epoch(model, tr_dl, opt, criterion, device)
        va_l, va_a, _, _, _ = evaluate(model, te_dl, criterion, device)
        sch.step(va_l)
        print(f"  Ep {ep+1:2d} | loss={tr_l:.4f} acc={tr_a:.4f} | val_acc={va_a:.4f}")

    _, test_acc, test_f1, preds, labels = evaluate(model, te_dl, criterion, device)
    print(f"\n  Test Accuracy : {test_acc:.4f}")
    print(f"  Test F1-macro : {test_f1:.4f}\n")
    print(classification_report(labels, preds, target_names=cat_names))
    plot_confusion_matrix(labels, preds, cat_names, cfg.OUT_DIR, tag="_test")

    # GradCAM analysis & image generation
    print("\n[5/5] GradCAM analysis & image generation …")
    model.eval()
    idx = np.random.randint(len(X_te))
    sample = torch.tensor(X_te[idx:idx+1], dtype=torch.float32).to(device)
    cam, pred_cls = model.gradcam(sample)
    plot_gradcam(cam, cfg.OUT_DIR)
    print(f"  Predicted : {cat_names[pred_cls]}")
    print(f"  Actual    : {cat_names[y_te[idx]]}")

    try:
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        pipe = pipe.to(device)
        prompt = cfg.CATEGORIES[pred_cls]["prompt"]
        print(f"  Prompt    : \"{prompt}\"")
        img = pipe(prompt).images[0]
        img.save(os.path.join(cfg.OUT_DIR, "generated_image.png"))
        print(f"  ✓ Image saved to {cfg.OUT_DIR}/generated_image.png")
    except ImportError:
        print("  ℹ  diffusers not installed — skipping image generation.")
    except Exception as e:
        print(f"  ⚠  Image generation error: {e}")

    print("\n All done! Outputs saved to:", cfg.OUT_DIR)

if __name__ == "__main__":
    main()
