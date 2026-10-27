"""Shared data loading and preprocessing for the Task 3-5 notebooks."""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CACHE_DIR = Path("cache")
IMG_SIZE = 64            # must match the value used in Task 2
SEED = 42


def load_cached_data():
    """Reload the arrays, labels and train/validation split that the Task 2 notebook saved."""
    split = np.load(CACHE_DIR / "split.npz")
    y_all = torch.from_numpy(split["y_all"]).long()
    class_to_idx = json.load(open(CACHE_DIR / "class_to_idx.json"))
    classes = sorted(class_to_idx, key=class_to_idx.get)          # class names in index order

    X_all = torch.from_numpy(np.load(CACHE_DIR / f"train_{IMG_SIZE}_{len(y_all)}.npy"))
    X_all = X_all.permute(0, 3, 1, 2).contiguous()                # (N,H,W,C) -> (N,C,H,W)
    test_file = sorted(CACHE_DIR.glob(f"test_{IMG_SIZE}_*.npy"))[0]
    X_test = torch.from_numpy(np.load(test_file)).permute(0, 3, 1, 2).contiguous()

    idx_tr, idx_va = split["idx_tr"], split["idx_va"]
    stats = {"train": (torch.from_numpy(split["mean"]), torch.from_numpy(split["std"])),
             "imagenet": (torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225]))}
    return dict(X_tr=X_all[idx_tr], y_tr=y_all[idx_tr], X_va=X_all[idx_va], y_va=y_all[idx_va],
                X_test=X_test, classes=classes, stats=stats)


def augment(x, max_shift=4):
    """x: float batch in [0,1], shape (B,3,H,W). Every image gets its own random changes."""
    B, C, H, W = x.shape
    flip = torch.rand(B) < 0.5                                    # 1. horizontal flip
    x = torch.where(flip[:, None, None, None], x.flip(3), x)

    p = max_shift                                                 # 2. random shift
    xp = F.pad(x, (p, p, p, p), mode="reflect")
    oy = torch.randint(0, 2 * p + 1, (B,)); ox = torch.randint(0, 2 * p + 1, (B,))
    rows = oy[:, None] + torch.arange(H)
    cols = ox[:, None] + torch.arange(W)
    x = xp[torch.arange(B)[:, None, None], :, rows[:, :, None], cols[:, None, :]].permute(0, 3, 1, 2)

    bright = (torch.rand(B, 1, 1, 1) - 0.5) * 0.4                 # 3. brightness / contrast
    contrast = 1 + (torch.rand(B, 1, 1, 1) - 0.5) * 0.4
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    return ((x - m) * contrast + m + bright).clamp(0, 1)


def prepare(xb_uint8, train, stats, use_aug=True):
    """uint8 batch -> float in [0,1] -> augment (training only) -> normalise with stats=(mean, std)."""
    x = xb_uint8.float() / 255
    if train and use_aug:
        x = augment(x)
    mean, std = stats
    return ((x - mean[None, :, None, None]) / std[None, :, None, None]).contiguous()
