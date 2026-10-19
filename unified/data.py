"""
Unified data pipeline.

One split, one preprocessing chain, one augmentation policy -- for all models.

The split is NOT recomputed here. It is loaded from the frozen artefacts
produced by "EDA & preprocessing/final-splitted-data.ipynb" (train/val/test
zips or folders). This is what puts every architecture on the same ground.
"""
from __future__ import annotations

import os
import zipfile
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import (UnifiedConfig, FEATURE_SETS, ID_COLS, TARGET_COL,
                    IMAGE_COL, N_TRAIN, N_VAL, N_TEST)

Image.MAX_IMAGE_PIXELS = None


# ---------------------------------------------------------------------------
#  Split discovery / extraction
# ---------------------------------------------------------------------------
def ensure_split(data_root: str, work_dir: str = "/kaggle/working/split") -> Dict[str, str]:
    """
    Locate the frozen 70/15/15 split. Accepts either already-extracted folders
    or the train.zip / val.zip / test.zip artefacts, anywhere under data_root.
    Returns {"train": dir, "val": dir, "test": dir}.
    """
    out: Dict[str, str] = {}
    for split in ("train", "val", "test"):
        found = None
        for root, dirs, files in os.walk(data_root):
            if os.path.basename(root) == split and f"{split}.csv" in files:
                found = root
                break
        if found:
            out[split] = found
            continue
        # fall back to the zip artefacts
        zip_path = None
        for root, dirs, files in os.walk(data_root):
            if f"{split}.zip" in files:
                zip_path = os.path.join(root, f"{split}.zip")
                break
        if zip_path is None:
            raise FileNotFoundError(
                f"Could not find the frozen '{split}' split under {data_root}. "
                "Attach the splitted-data dataset to the notebook.")
        dest = os.path.join(work_dir)
        os.makedirs(dest, exist_ok=True)
        if not os.path.isdir(os.path.join(dest, split)):
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(dest)
        out[split] = os.path.join(dest, split)
    return out


def load_split(data_root: str, feature_set: str = "full"
               ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], Dict[str, str]]:
    """Load the three frozen dataframes plus the agreed metadata column list."""
    dirs = ensure_split(data_root)
    dfs = {}
    for split, d in dirs.items():
        df = pd.read_csv(os.path.join(d, f"{split}.csv"))
        dfs[split] = df

    tr, va, te = dfs["train"], dfs["val"], dfs["test"]

    # Integrity: the split must be the frozen one, otherwise results are not
    # comparable across models and the run is aborted.
    for name, df, n in (("train", tr, N_TRAIN), ("val", va, N_VAL), ("test", te, N_TEST)):
        if len(df) != n:
            raise ValueError(
                f"{name} split has {len(df)} rows but the frozen split defines {n}. "
                "Refusing to run: the benchmark requires the identical partition.")

    meta_cols = build_feature_list(tr, feature_set)
    return tr, va, te, meta_cols, dirs


def build_feature_list(df: pd.DataFrame, feature_set: str) -> List[str]:
    if feature_set not in FEATURE_SETS:
        raise KeyError(f"unknown feature_set {feature_set!r}")
    drop_prefixes = FEATURE_SETS[feature_set]["drop_prefixes"]
    cols = [c for c in df.columns if c not in ID_COLS and c != TARGET_COL]
    cols = [c for c in cols if not any(c.startswith(p) for p in drop_prefixes)]
    return sorted(cols)


# ---------------------------------------------------------------------------
#  Transforms -- IDENTICAL for every architecture
# ---------------------------------------------------------------------------
def build_transforms(cfg: UnifiedConfig, train: bool) -> transforms.Compose:
    """
    The corpus images are already DullRazor-inpainted and resized to 224x224 by
    the shared preprocessing stage, so the deterministic chain is a square
    resize (a no-op at 224) rather than a resize-and-crop, which would upscale
    and then discard the lesion border.

    Training chain adds the conservative augmentation policy described in the
    paper. Augmentations are deliberately mild so that diagnostically relevant
    colour gradients and border morphology survive.
    """
    norm = transforms.Normalize(mean=cfg.mean, std=cfg.std)
    if not train:
        return transforms.Compose([
            transforms.Resize((cfg.img_size, cfg.img_size)),
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomResizedCrop(cfg.img_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomApply([transforms.RandomRotation(90)], p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        transforms.RandomAffine(degrees=0, shear=10, scale=(0.9, 1.1)),
        transforms.ToTensor(),
        norm,
        transforms.RandomErasing(p=0.3, scale=(0.01, 0.05), value="random"),
    ])


# ---------------------------------------------------------------------------
#  Dataset
# ---------------------------------------------------------------------------
class LesionDataset(Dataset):
    """Returns (image, metadata_vector, label). Metadata is already numeric."""

    def __init__(self, df: pd.DataFrame, img_dir: str, meta_cols: List[str],
                 tfm: transforms.Compose):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.meta_cols = meta_cols
        self.tfm = tfm
        self.meta = self.df[meta_cols].to_numpy(dtype=np.float32)
        self.labels = self.df[TARGET_COL].to_numpy(dtype=np.float32)
        self.files = self.df[IMAGE_COL].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        path = os.path.join(self.img_dir, self.files[i])
        with Image.open(path) as im:
            img = self.tfm(im.convert("RGB"))
        return img, torch.from_numpy(self.meta[i]), torch.tensor(self.labels[i])


def build_loaders(cfg: UnifiedConfig) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    tr, va, te, meta_cols, dirs = load_split(cfg.data_root, cfg.feature_set)
    pin = torch.cuda.is_available()
    common = dict(num_workers=cfg.num_workers, pin_memory=pin,
                  persistent_workers=cfg.num_workers > 0,
                  prefetch_factor=4 if cfg.num_workers > 0 else None)

    def mk(df, split, train):
        ds = LesionDataset(df, os.path.join(dirs[split], "images"), meta_cols,
                           build_transforms(cfg, train))
        return DataLoader(ds, batch_size=cfg.micro_batch, shuffle=train,
                          drop_last=train, **common)

    return (mk(tr, "train", True), mk(va, "val", False),
            mk(te, "test", False), meta_cols)
