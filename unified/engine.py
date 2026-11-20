"""
Unified training / evaluation engine.

GPU throughput measures applied uniformly to every model:
  * mixed precision (bf16 autocast on Ampere+, fp16 + GradScaler otherwise)
  * uint8 memmap cache + GPU-side augmentation (see data.py)
  * channels_last memory format for convolutional backbones
  * cudnn.benchmark autotuning, TF32 matmuls
  * gradient accumulation to hold the effective batch at 32 regardless of VRAM
  * fused AdamW where available, set_to_none gradient zeroing
  * per-epoch resumable checkpoint (`last.pt`), so a sleep / logoff / crash
    costs at most one epoch instead of the whole run
  * Windows: the process keeps the machine awake while it trains

Evaluation: EMA weights, temperature scaling and threshold fitted on
VALIDATION, 8-view dihedral test-time augmentation applied to every model.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from config import UnifiedConfig, grad_accum_steps
from metrics import (TemperatureScaler, bootstrap_ci, choose_threshold,
                     compute_metrics)


# ---------------------------------------------------------------------------
def sigmoid(z: np.ndarray) -> np.ndarray:
    """Overflow-safe logistic."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def native_bf16() -> bool:
    """
    True only on GPUs with hardware bf16 (compute capability >= 8.0, Ampere+).
    torch.cuda.is_bf16_supported() also returns True on T4 / P100 / GTX 16xx
    through slow emulation, which would make Kaggle runs crawl.
    """
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


def amp_dtype(cfg: UnifiedConfig, device) -> torch.dtype:
    if device.type == "cuda" and cfg.amp_dtype == "bf16" and native_bf16():
        return torch.bfloat16
    return torch.float16


def keep_awake(on: bool = True) -> None:
    """Stop Windows from idle-sleeping while a run is in progress."""
    if os.name != "nt":
        return
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception:  # noqa: BLE001
        pass


class ModelEMA:
    """Exponential moving average of weights; evaluated instead of the raw model."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)

    def copy_to(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()
                  if k in self.shadow}
        sd = model.state_dict()
        for k, v in self.shadow.items():
            sd[k].copy_(v.to(sd[k].dtype))
        return backup

    @staticmethod
    def restore(model: nn.Module, backup: Dict[str, torch.Tensor]) -> None:
        sd = model.state_dict()
        for k, v in backup.items():
            sd[k].copy_(v)


class WarmupCosine:
    """Linear warmup then cosine decay, stepped per optimiser update."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float):
        self.opt = optimizer
        self.warmup, self.total, self.min_lr = max(1, warmup_steps), max(2, total_steps), min_lr
        self.base = [g["lr"] for g in optimizer.param_groups]
        self.step_n = 0
        self._apply()

    def _apply(self) -> None:
        for g, base in zip(self.opt.param_groups, self.base):
            if self.step_n <= self.warmup:
                lr = base * max(self.step_n, 1) / self.warmup
            else:
                p = (self.step_n - self.warmup) / max(1, self.total - self.warmup)
                lr = self.min_lr + 0.5 * (base - self.min_lr) * (1 + math.cos(math.pi * min(1.0, p)))
            g["lr"] = lr

    def step(self) -> None:
        self.step_n += 1
        self._apply()


def build_optimizer(model, cfg: UnifiedConfig):
    groups = model.param_groups(cfg.lr_head, cfg.lr_backbone, cfg.weight_decay)
    try:
        return torch.optim.AdamW(groups, fused=torch.cuda.is_available())
    except (TypeError, RuntimeError):
        return torch.optim.AdamW(groups)


# ---------------------------------------------------------------------------
# 8-view dihedral group: identity, 3 rotations, and the same after a flip.
def dihedral(x: torch.Tensor, k: int) -> torch.Tensor:
    if k >= 4:
        x = x.flip(-1)
    return torch.rot90(x, k % 4, dims=(-2, -1)) if k % 4 else x


@torch.no_grad()
def predict(model, loader, device, cfg: UnifiedConfig,
            channels_last: bool, tta: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (logits, labels) for a whole loader. TTA averages 8 views' logits."""
    model.eval()
    logits, labels = [], []
    views = range(8) if tta else range(1)
    dt = amp_dtype(cfg, device)
    for img, meta, y in loader:
        img = img.to(device, non_blocking=True)
        meta = meta.to(device, non_blocking=True)
        acc = None
        for k in views:
            v = dihedral(img, k)
            if channels_last:
                v = v.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=dt, enabled=cfg.amp and device.type == "cuda"):
                out = model(v, meta).float()
            acc = out if acc is None else acc + out
        logits.append((acc / len(views)).cpu())
        labels.append(y.detach().cpu() if torch.is_tensor(y) else torch.as_tensor(y))
    return torch.cat(logits).numpy(), torch.cat(labels).float().numpy()


def load_ema_weights(model: nn.Module, ckpt_path: str, device) -> nn.Module:
    """Load a best.pt (EMA weights) into `model`."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" in state:
        model.load_state_dict(state["model"])
    sd = model.state_dict()
    for k, v in state.get("ema", {}).items():
        if k in sd:
            sd[k].copy_(v.to(sd[k].dtype))
    return model.to(device).eval()


# ---------------------------------------------------------------------------
def train_one_model(model, loaders, cfg: UnifiedConfig, out_dir: str,
                    log=print) -> Dict:
    train_loader, val_loader, test_loader = loaders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.seed, cfg.deterministic)
    os.makedirs(out_dir, exist_ok=True)
    keep_awake(True)

    channels_last = cfg.channels_last and device.type == "cuda"
    model = model.to(device)
    if channels_last:
        try:
            model = model.to(memory_format=torch.channels_last)
        except RuntimeError:          # e.g. Swin-V2 holds rank-5 parameters
            model = model.to(memory_format=torch.contiguous_format)
            channels_last = False

    dt = amp_dtype(cfg, device)
    use_amp = cfg.amp and device.type == "cuda"
    accum = grad_accum_steps(cfg)
    steps_per_epoch = max(1, len(train_loader) // accum)
    opt = build_optimizer(model, cfg)
    sched = WarmupCosine(opt, cfg.warmup_epochs * steps_per_epoch,
                         cfg.epochs * steps_per_epoch, cfg.min_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and dt == torch.float16)
    lossf = nn.BCEWithLogitsLoss()
    ema = ModelEMA(model, cfg.ema_decay)

    best = {"score": -1.0, "epoch": -1}
    history, patience, start_epoch = [], 0, 1
    ckpt = os.path.join(out_dir, "best.pt")
    last = os.path.join(out_dir, "last.pt")
    t_start = time.time()

    # ---- resume -------------------------------------------------------------
    if os.path.exists(last):
        try:
            st = torch.load(last, map_location=device, weights_only=False)
            model.load_state_dict(st["model"])
            opt.load_state_dict(st["opt"])
            ema.shadow = {k: v.to(device) for k, v in st["ema"].items()}
            sched.step_n = st["sched_step"]
            sched._apply()
            best, history, patience = st["best"], st["history"], st["patience"]
            start_epoch = st["epoch"] + 1
            train_loader.epoch = st["epoch"]
            log(f"  resumed after epoch {st['epoch']} (best {best['score']:.4f} "
                f"@ epoch {best['epoch']})")
            del st
        except Exception as exc:  # noqa: BLE001
            log(f"  [warn] could not resume ({exc}); starting fresh")

    stopped = patience >= cfg.early_stop_patience
    for epoch in range(start_epoch, cfg.epochs + 1):
        if stopped:
            break
        model.train()
        t0, running, nb = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)

        for i, (img, meta, y) in enumerate(train_loader):
            if channels_last:
                img = img.contiguous(memory_format=torch.channels_last)
            if cfg.label_smooth > 0:
                y = y * (1 - cfg.label_smooth) + 0.5 * cfg.label_smooth

            with torch.autocast("cuda", dtype=dt, enabled=use_amp):
                out = model(img, meta)
            loss = lossf(out.float(), y) / accum
            scaler.scale(loss).backward()
            running += loss.item() * accum
            nb += 1

            if (i + 1) % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                ema.update(model)

        backup = ema.copy_to(model)
        vl, vy = predict(model, val_loader, device, cfg, channels_last)
        ema.restore(model, backup)

        vp = sigmoid(vl)
        thr = choose_threshold(vy, vp, cfg.threshold_rule)
        vm = compute_metrics(vy, vp, thr)
        score = vm["auc"] if cfg.monitor == "val_auc" else vm["f1"]
        row = {"epoch": epoch, "train_loss": running / max(1, nb),
               "secs": round(time.time() - t0, 1), **{f"val_{k}": v for k, v in vm.items()}}
        history.append(row)
        log(f"  epoch {epoch:2d}/{cfg.epochs} | loss {row['train_loss']:.4f} | "
            f"val AUC {vm['auc']:.4f} | val acc {vm['accuracy']:.4f} | "
            f"F1 {vm['f1']:.4f} | {row['secs']:.0f}s")

        if score > best["score"]:
            best = {"score": score, "epoch": epoch, "threshold": thr}
            # Only the EMA weights are saved: they are what evaluation uses.
            torch.save({"ema": ema.shadow, "epoch": epoch, "cfg": asdict(cfg)}, ckpt)
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                log(f"  early stop at epoch {epoch} (best epoch {best['epoch']})")
                stopped = True

        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "ema": ema.shadow, "sched_step": sched.step_n, "epoch": epoch,
                    "best": best, "history": history, "patience": patience}, last)

    # ---- final evaluation on the held-out test set --------------------------
    del opt
    state = torch.load(ckpt, map_location=device, weights_only=False)
    ema.shadow = {k: v.to(device) for k, v in state["ema"].items()}
    ema.copy_to(model)
    del state

    val_logits, val_y = predict(model, val_loader, device, cfg, channels_last, tta=cfg.tta)
    test_logits, test_y = predict(model, test_loader, device, cfg, channels_last, tta=cfg.tta)

    temperature = 1.0
    if cfg.calibrate:
        temperature = TemperatureScaler().fit(val_logits, val_y)
    val_prob = sigmoid(val_logits / temperature)
    test_prob = sigmoid(test_logits / temperature)

    # threshold chosen on VALIDATION, never on test
    thr = choose_threshold(val_y, val_prob, cfg.threshold_rule)
    val_metrics = compute_metrics(val_y, val_prob, thr)
    test_metrics = compute_metrics(test_y, test_prob, thr)
    ci = bootstrap_ci(test_y, test_prob, thr, cfg.n_bootstrap, cfg.seed)

    total, trainable = (sum(p.numel() for p in model.parameters()),
                        sum(p.numel() for p in model.parameters() if p.requires_grad))

    result = {
        "model_name": cfg.model_name,
        "feature_set": cfg.feature_set,
        "backbone_loaded": model.backbone_name(),
        "config": asdict(cfg),
        "best_epoch": best["epoch"],
        "epochs_run": len(history),
        "train_minutes": round((time.time() - t_start) / 60, 1),
        "params_total": total,
        "params_trainable": trainable,
        "temperature": temperature,
        "threshold": thr,
        "tta": cfg.tta,
        "val": val_metrics,
        "test": test_metrics,
        "test_ci95": ci,
        "history": history,
    }

    np.savez_compressed(os.path.join(out_dir, "predictions.npz"),
                        test_prob=test_prob, test_y=test_y,
                        val_prob=val_prob, val_y=val_y,
                        test_logits=test_logits, val_logits=val_logits)
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    if os.path.exists(last):
        os.remove(last)                       # run finished; keep only best.pt

    log(f"\n  TEST  acc {test_metrics['accuracy']:.4f} | bal-acc "
        f"{test_metrics['balanced_accuracy']:.4f} | prec {test_metrics['precision']:.4f} | "
        f"rec {test_metrics['recall']:.4f} | F1 {test_metrics['f1']:.4f} | "
        f"AUC {test_metrics['auc']:.4f}")
    return result
