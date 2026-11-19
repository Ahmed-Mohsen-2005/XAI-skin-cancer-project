"""
Unified data pipeline.

One split, one preprocessing chain, one augmentation policy -- for all models.

The split is NOT recomputed here. It is loaded from the frozen artefacts
produced by "EDA & preprocessing/final-splitted-data.ipynb" (train/val/test
zips or folders). This is what puts every architecture on the same ground.

Throughput design
-----------------
The corpus is 17k images of 224x224, i.e. 2.6 GB as uint8. Decoding JPEGs and
running PIL augmentations in worker processes was the bottleneck on Windows
(spawned workers, no fork), so instead:

  1. every split is decoded ONCE into a uint8 memmap (`cache_dir`);
  2. a background thread gathers shuffled batches from the memmap into pinned
     memory;
  3. augmentation runs on the GPU, batched, inside `GPUAugment`.

`loader.dataset` still behaves like a torch Dataset (`ds[i]` returns the
normalised CHW tensor, metadata vector and label), which is what the XAI code
uses.
"""
from __future__ import annotations

import hashlib
import math
import os
import queue
import threading
import zipfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

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
        zip_path = None
        for root, dirs, files in os.walk(data_root):
            if f"{split}.zip" in files:
                zip_path = os.path.join(root, f"{split}.zip")
                break
        if zip_path is None:
            raise FileNotFoundError(
                f"Could not find the frozen '{split}' split under {data_root}. "
                "Attach the splitted-data dataset to the notebook.")
        os.makedirs(work_dir, exist_ok=True)
        if not os.path.isdir(os.path.join(work_dir, split)):
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(work_dir)
        out[split] = os.path.join(work_dir, split)
    return out


def load_split(data_root: str, feature_set: str = "full"
               ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], Dict[str, str]]:
    """Load the three frozen dataframes plus the agreed metadata column list."""
    dirs = ensure_split(data_root)
    dfs = {split: pd.read_csv(os.path.join(d, f"{split}.csv")) for split, d in dirs.items()}
    tr, va, te = dfs["train"], dfs["val"], dfs["test"]

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
#  uint8 image cache
# ---------------------------------------------------------------------------
def default_cache_dir(cfg: UnifiedConfig) -> str:
    if cfg.cache_dir:
        return cfg.cache_dir
    if os.path.isdir("/kaggle/working"):
        return "/kaggle/working/cache"
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def vignette_flag(img: np.ndarray, thr: int = 40) -> bool:
    """True when the four 16x16 corners are mostly black (dermoscope ring)."""
    g = img.mean(-1) if img.ndim == 3 else img
    c = np.concatenate([g[:16, :16].ravel(), g[:16, -16:].ravel(),
                        g[-16:, :16].ravel(), g[-16:, -16:].ravel()])
    return bool((c < thr).mean() > 0.5)


def build_cache(img_dir: str, files: List[str], cache_dir: str, split: str,
                size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Decode a split once into (N, size, size, 3) uint8; returns (memmap, vignette)."""
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.md5(("|".join(files) + f"|{size}").encode()).hexdigest()[:12]
    arr_p = os.path.join(cache_dir, f"{split}_{size}_{key}.npy")
    vig_p = os.path.join(cache_dir, f"{split}_{size}_{key}_vig.npy")
    if os.path.exists(arr_p) and os.path.exists(vig_p):
        return np.load(arr_p, mmap_mode="r"), np.load(vig_p)

    print(f"  caching {split}: {len(files)} images -> {arr_p}", flush=True)
    tmp = arr_p + ".tmp"
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8,
                                    shape=(len(files), size, size, 3))
    vig = np.zeros(len(files), dtype=bool)
    for i, f in enumerate(files):
        with Image.open(os.path.join(img_dir, f)) as im:
            im = im.convert("RGB")
            if im.size != (size, size):
                im = im.resize((size, size), Image.BICUBIC)
            a = np.asarray(im, dtype=np.uint8)
        arr[i] = a
        vig[i] = vignette_flag(a)
    arr.flush()
    del arr
    os.replace(tmp, arr_p)
    np.save(vig_p, vig)
    return np.load(arr_p, mmap_mode="r"), vig


# ---------------------------------------------------------------------------
#  Dataset view (used by XAI and by anything that wants ds[i])
# ---------------------------------------------------------------------------
class CachedDataset:
    """Indexable view over the cache; returns NORMALISED, un-augmented tensors."""

    def __init__(self, df: pd.DataFrame, images: np.ndarray, vignette: np.ndarray,
                 meta_cols: List[str], mean: List[float], std: List[float]):
        self.df = df.reset_index(drop=True)
        self.images = images
        self.vignette = vignette
        self.meta_cols = meta_cols
        self.meta = self.df[meta_cols].to_numpy(dtype=np.float32)
        self.labels = self.df[TARGET_COL].to_numpy(dtype=np.float32)
        self.files = self.df[IMAGE_COL].astype(str).tolist()
        self._mean = torch.tensor(mean).view(3, 1, 1)
        self._std = torch.tensor(std).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.df)

    def raw(self, i: int) -> np.ndarray:
        """uint8 HWC image, exactly what the model sees before normalisation."""
        return np.asarray(self.images[i])

    def __getitem__(self, i: int):
        img = torch.from_numpy(self.raw(i).copy()).permute(2, 0, 1).float().div_(255)
        img = (img - self._mean) / self._std
        return img, torch.from_numpy(self.meta[i]), torch.tensor(self.labels[i])


# ---------------------------------------------------------------------------
#  GPU augmentation -- IDENTICAL for every architecture
# ---------------------------------------------------------------------------
_YIQ = torch.tensor([[0.299, 0.587, 0.114],
                     [0.596, -0.274, -0.322],
                     [0.211, -0.523, 0.312]])


class GPUAugment:
    """
    Batched, per-sample random augmentation on the GPU:
      * random resized crop + full rotation + flips, as ONE affine warp with
        reflection padding (no black corners, which would mimic a vignette)
      * brightness / contrast / saturation / small hue jitter
      * counterfactual dermoscope vignette (class-balanced, see config)
      * small random erasing
    Input/Output: float images in [0, 1], (B, 3, H, W).
    """

    def __init__(self, cfg: UnifiedConfig, vig_prob: Tuple[float, float]):
        self.cfg = cfg
        self.vig_prob = vig_prob          # P(add vignette | y=0), P(... | y=1)
        self._yiq: Optional[torch.Tensor] = None
        self._rgb: Optional[torch.Tensor] = None

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, y: torch.Tensor, has_vig: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        dev = x.device
        cfg = self.cfg

        # ---- geometry -----------------------------------------------------
        scale = torch.empty(b, device=dev).uniform_(*cfg.aug_scale).sqrt()   # side ratio
        ar = torch.exp(torch.empty(b, device=dev).uniform_(math.log(0.85), math.log(1.18)))
        sx, sy = scale * ar.sqrt(), scale / ar.sqrt()
        ang = torch.empty(b, device=dev).uniform_(-cfg.aug_rotate, cfg.aug_rotate) * math.pi / 180
        fx = torch.where(torch.rand(b, device=dev) < 0.5, -1.0, 1.0)
        fy = torch.where(torch.rand(b, device=dev) < 0.5, -1.0, 1.0)
        tx = (1 - sx).clamp(min=0) * torch.empty(b, device=dev).uniform_(-1, 1)
        ty = (1 - sy).clamp(min=0) * torch.empty(b, device=dev).uniform_(-1, 1)
        cos, sin = torch.cos(ang), torch.sin(ang)
        # theta = R(ang) @ diag(sx*fx, sy*fy): a rotation of a (flipped) crop,
        # with no shear component.
        ax, ay = sx * fx, sy * fy
        theta = torch.stack([
            torch.stack([cos * ax, -sin * ay, tx], -1),
            torch.stack([sin * ax, cos * ay, ty], -1)], 1)
        grid = F.affine_grid(theta, (b, 3, h, w), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection",
                          align_corners=False)

        # ---- colour -------------------------------------------------------
        c = cfg.aug_color
        if c > 0:
            bri = torch.empty(b, 1, 1, 1, device=dev).uniform_(1 - c, 1 + c)
            con = torch.empty(b, 1, 1, 1, device=dev).uniform_(1 - c, 1 + c)
            sat = torch.empty(b, 1, 1, 1, device=dev).uniform_(1 - c, 1 + c)
            x = x * bri
            m = x.mean(dim=(1, 2, 3), keepdim=True)
            x = (x - m) * con + m
            g = 0.299 * x[:, :1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
            x = (x - g) * sat + g
            # hue: rotate chroma in YIQ space by a small angle
            if self._yiq is None or self._yiq.device != dev:
                self._yiq = _YIQ.to(dev)
                self._rgb = torch.linalg.inv(self._yiq)
            hue = torch.empty(b, device=dev).uniform_(-0.05, 0.05) * 2 * math.pi
            q = torch.einsum("ij,bjhw->bihw", self._yiq, x)
            ch, sh = torch.cos(hue).view(b, 1, 1), torch.sin(hue).view(b, 1, 1)
            i2 = q[:, 1] * ch - q[:, 2] * sh
            q2 = q[:, 1] * sh + q[:, 2] * ch
            x = torch.einsum("ij,bjhw->bihw", self._rgb, torch.stack([q[:, 0], i2, q2], 1))
            x = x.clamp_(0, 1)

        # ---- counterfactual vignette -------------------------------------
        if cfg.vignette_debias:
            p = torch.where(y > 0.5, torch.full_like(y, self.vig_prob[1]),
                            torch.full_like(y, self.vig_prob[0]))
            add = (torch.rand(b, device=dev) < p) & (~has_vig.bool())
            n = int(add.sum())
            if n:
                yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=dev),
                                        torch.linspace(-1, 1, w, device=dev), indexing="ij")
                r = torch.empty(n, 1, 1, device=dev).uniform_(0.95, 1.25)
                cx = torch.empty(n, 1, 1, device=dev).uniform_(-0.08, 0.08)
                cy = torch.empty(n, 1, 1, device=dev).uniform_(-0.08, 0.08)
                soft = torch.empty(n, 1, 1, device=dev).uniform_(0.03, 0.15)
                d = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
                mask = torch.sigmoid((r - d) / soft).unsqueeze(1)           # 1 inside
                floor = torch.empty(n, 1, 1, 1, device=dev).uniform_(0.0, 0.08)
                x[add] = x[add] * mask + floor * (1 - mask)

        # ---- random erasing ----------------------------------------------
        if cfg.aug_erase_p > 0:
            er = torch.nonzero(torch.rand(b) < cfg.aug_erase_p).flatten().tolist()
            for i in er:
                eh = int(h * float(np.random.uniform(0.06, 0.14)))
                ew = int(w * float(np.random.uniform(0.06, 0.14)))
                y0 = int(np.random.randint(0, h - eh))
                x0 = int(np.random.randint(0, w - ew))
                x[i, :, y0:y0 + eh, x0:x0 + ew] = torch.rand(3, 1, 1, device=dev)
        return x


def vignette_probs(labels: np.ndarray, vig: np.ndarray, extra: float) -> Tuple[float, float]:
    """Per-class P(add) so that P(vignette | class) is equal for both classes."""
    rate = [float(vig[labels == c].mean()) if (labels == c).any() else 0.0 for c in (0, 1)]
    target = min(0.95, max(rate) + extra)
    p0, p1 = (max(0.0, (target - r) / max(1e-6, 1 - r)) for r in rate)
    return p0, p1


# ---------------------------------------------------------------------------
#  GPU batch loader
# ---------------------------------------------------------------------------
class GPULoader:
    """
    Yields (image, meta, label) already on `device`; images normalised and,
    for training, augmented on the GPU. A background thread prefetches pinned
    uint8 batches from the memmap so the GPU never waits on disk.
    """

    def __init__(self, ds: CachedDataset, batch_size: int, train: bool,
                 cfg: UnifiedConfig, device: torch.device,
                 augment: Optional[GPUAugment] = None, prefetch: int = 6,
                 limit: int = 0):
        self.dataset = ds
        self.batch_size = batch_size
        self.train = train
        self.cfg = cfg
        self.device = device
        self.augment = augment
        self.prefetch = prefetch
        self.limit = limit              # >0: only the first `limit` batches (smoke tests)
        self.epoch = 0
        self.mean = torch.tensor(cfg.mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg.std, device=device).view(1, 3, 1, 1)

    def __len__(self) -> int:
        n = len(self.dataset)
        k = n // self.batch_size if self.train else math.ceil(n / self.batch_size)
        return min(k, self.limit) if self.limit else k

    def _batches(self) -> List[np.ndarray]:
        n = len(self.dataset)
        if self.train:
            rng = np.random.default_rng(self.cfg.seed + self.epoch)
            idx = rng.permutation(n)
        else:
            idx = np.arange(n)
        return [idx[i * self.batch_size:(i + 1) * self.batch_size] for i in range(len(self))]

    def __iter__(self):
        batches = self._batches()
        self.epoch += 1
        ds = self.dataset
        pin = self.device.type == "cuda"
        q: "queue.Queue" = queue.Queue(maxsize=self.prefetch)
        stop = threading.Event()

        def producer():
            try:
                for bi in batches:
                    if stop.is_set():
                        return
                    order = np.argsort(bi)                  # sequential memmap reads
                    imgs = np.empty((len(bi),) + ds.images.shape[1:], dtype=np.uint8)
                    imgs[order] = ds.images[bi[order]]
                    imgs = torch.from_numpy(imgs)
                    meta = torch.from_numpy(ds.meta[bi])
                    y = torch.from_numpy(ds.labels[bi])
                    v = torch.from_numpy(ds.vignette[bi])
                    if pin:
                        imgs, meta, y = imgs.pin_memory(), meta.pin_memory(), y.pin_memory()
                    q.put((imgs, meta, y, v))
            except Exception as exc:  # noqa: BLE001
                q.put(exc)
            q.put(None)

        th = threading.Thread(target=producer, daemon=True)
        th.start()
        try:
            while True:
                item = q.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                imgs, meta, y, v = item
                x = imgs.to(self.device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
                meta = meta.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                if self.train and self.augment is not None:
                    x = self.augment(x, y, v.to(self.device))
                x = (x - self.mean) / self.std
                yield x.contiguous(), meta, y
        finally:
            stop.set()
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break


def build_loaders(cfg: UnifiedConfig, device: Optional[torch.device] = None,
                  limit: int = 0) -> Tuple[GPULoader, GPULoader, GPULoader, List[str]]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va, te, meta_cols, dirs = load_split(cfg.data_root, cfg.feature_set)
    cache = default_cache_dir(cfg)

    def mk_ds(df, split):
        files = df[IMAGE_COL].astype(str).tolist()
        arr, vig = build_cache(os.path.join(dirs[split], "images"), files, cache,
                               split, cfg.img_size)
        return CachedDataset(df, arr, vig, meta_cols, cfg.mean, cfg.std)

    ds_tr, ds_va, ds_te = mk_ds(tr, "train"), mk_ds(va, "val"), mk_ds(te, "test")
    vp = vignette_probs(ds_tr.labels, ds_tr.vignette, cfg.vignette_extra)
    aug = GPUAugment(cfg, vp)
    eval_bs = max(cfg.micro_batch, 32)
    return (GPULoader(ds_tr, cfg.micro_batch, True, cfg, device, aug, limit=limit),
            GPULoader(ds_va, eval_bs, False, cfg, device),
            GPULoader(ds_te, eval_bs, False, cfg, device),
            meta_cols)
