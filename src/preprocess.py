"""src/preprocess.py
Common data-loading & preprocessing utilities.  Supports a synthetic
classification data set for smoke-tests and plugs in concrete benchmark data
sets (CIFAR-10 / CIFAR-100) via the Hugging Face 🤗 Datasets hub for the main
experiments.
"""
from __future__ import annotations
import random
from typing import Any, Dict, Tuple

import numpy as np
import torch
from sklearn.datasets import make_classification
from torch.utils.data import Dataset

# third-party
from datasets import load_dataset
import torchvision.transforms as T

# -------------------------------- synthetic data -----------------------------

class SyntheticClassificationDataset(Dataset):
    def __init__(self, n_samples: int = 2000, n_features: int = 20, n_classes: int = 2, seed: int = 0):
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=int(n_features * 0.6),
            n_redundant=int(n_features * 0.2),
            n_classes=n_classes,
            random_state=seed,
        )
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# -------------------------------- HF image data ------------------------------

class HFImageDataset(Dataset):
    """Light-weight wrapper around HF image datasets returning (tensor, label)."""

    def __init__(self, hf_ds, transform):
        self.ds = hf_ds  # huggingface Dataset object
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        example = self.ds[idx]
        img = example["img"]  # PIL.Image
        label = example["label"]
        img = self.transform(img)
        return img, label


def _cifar_transforms(train: bool = True):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    if train:
        return T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean, std),
        ])


# ------------------------------- dataset factory -----------------------------

def get_dataset(name: str, cfg: Dict[str, Any]) -> Tuple[Dataset, Dataset, int, int]:
    """Returns (train_dataset, val_dataset, input_dim, num_classes).

    Supported names:
      • synthetic – toy data for smoke tests
      • cifar10  – uoft-cs/cifar10 (HF hub)
      • cifar100 – uoft-cs/cifar100 (HF hub)
    """

    name = name.lower()

    # ---- synthetic (tabular) ------------------------------------------------
    if name == "synthetic":
        seed = cfg.get("seed", 0)
        n_features = cfg.get("n_features", 20)
        n_classes = cfg.get("n_classes", 2)
        full_ds = SyntheticClassificationDataset(2000, n_features, n_classes, seed=seed)
        n_train = int(0.8 * len(full_ds))
        n_val = len(full_ds) - n_train
        train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val])
        return train_ds, val_ds, n_features, n_classes

    # ---- CIFAR-10 / CIFAR-100 ----------------------------------------------
    if name in {"cifar10", "cifar100"}:
        hf_repo = f"uoft-cs/{name}"
        # splits: train (50k) / test (10k).  We create a 45k/5k train/val split.
        ds_train_full = load_dataset(hf_repo, split="train")
        ds_test = load_dataset(hf_repo, split="test")  # used as held-out test later if desired

        indices = list(range(len(ds_train_full)))
        random.shuffle(indices)
        val_size = 5000
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        hf_train = ds_train_full.select(train_indices)
        hf_val = ds_train_full.select(val_indices)

        train_ds = HFImageDataset(hf_train, _cifar_transforms(train=True))
        val_ds = HFImageDataset(hf_val, _cifar_transforms(train=False))

        input_dim = 3 * 32 * 32  # flattened dimension (unused for CNNs but kept for consistency)
        num_classes = 10 if name == "cifar10" else 100
        return train_ds, val_ds, input_dim, num_classes

    # ------------------------------------------------------------------------
    raise NotImplementedError(f"Dataset '{name}' not supported.  Implemented: synthetic, cifar10, cifar100.")
