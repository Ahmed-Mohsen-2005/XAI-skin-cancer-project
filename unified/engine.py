"""
Unified training / evaluation engine.

GPU throughput measures applied uniformly to every model:
  * mixed precision (fp16 autocast + GradScaler)
  * channels_last memory format for convolutional backbones
  * cudnn.benchmark autotuning
  * gradient accumulation to hold the effective batch at 32 regardless of VRAM
  * fused AdamW where available
  * non_blocking H2D copies from pinned memory, persistent workers, prefetch
  * set_to_none gradient zeroing
  * inference under no_grad + autocast
  * optional torch.compile (off by default; enable with --compile)
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import UnifiedConfig, grad_accum_steps
from metrics import (TemperatureScaler, bootstrap_ci, choose_threshold,
                     compute_metrics)


# ---------------------------------------------------------------------------
def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Overflow-safe logistic. The naive 1/(1+exp(-z)) overflows for |z| beyond
    about 700, which happens routinely once logits are divided by a small
    calibration temperature.
    """
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

    def step(self) -> None:
        self.step_n += 1
        for g, base in zip(self.opt.param_groups, self.base):
            if self.step_n <= self.warmup:
                lr = base * self.step_n / self.warmup
            else:
                p = (self.step_n - self.warmup) / max(1, self.total - self.warmup)
                lr = self.min_lr + 0.5 * (base - self.min_lr) * (1 + math.cos(math.pi * min(1.0, p)))
            g["lr"] = lr


def build_optimizer(model, cfg: UnifiedConfig):
    groups = model.param_groups(cfg.lr_head, cfg.lr_backbone, cfg.weight_decay)
    try:
        return torch.optim.AdamW(groups, fused=torch.cuda.is_available())
    except (TypeError, RuntimeError):
        return torch.optim.AdamW(groups)


# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(model, loader: DataLoader, device, cfg: UnifiedConfig,
            channels_last: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (logits, labels) for a whole loader."""
    model.eval()
    logits, labels = [], []
    for img, meta, y in loader:
        img = img.to(device, non_blocking=True)
        if channels_last:
            img = img.contiguous(memory_format=torch.channels_last)
        meta = meta.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.amp and device.type == "cuda"):
            out = model(img, meta)
        logits.append(out.float().cpu())
        labels.append(y)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def train_one_model(model, loaders, cfg: UnifiedConfig, out_dir: str,
                    log=print) -> Dict:
    train_loader, val_loader, test_loader = loaders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.seed, cfg.deterministic)
    os.makedirs(out_dir, exist_ok=True)

    channels_last = cfg.channels_last and device.type == "cuda"
    model = model.to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    accum = grad_accum_steps(cfg)
    steps_per_epoch = max(1, len(train_loader) // accum)
    opt = build_optimizer(model, cfg)
    sched = WarmupCosine(opt, cfg.warmup_epochs * steps_per_epoch,
                         cfg.epochs * steps_per_epoch, cfg.min_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    lossf = nn.BCEWithLogitsLoss()
    ema = ModelEMA(model, cfg.ema_decay)

    best = {"score": -1.0, "epoch": -1}
    history, patience = [], 0
    ckpt = os.path.join(out_dir, "best.pt")
    t_start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0, running, nb = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)

        for i, (img, meta, y) in enumerate(train_loader):
            img = img.to(device, non_blocking=True)
            if channels_last:
                img = img.contiguous(memory_format=torch.channels_last)
            meta = meta.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if cfg.label_smooth > 0:
                y = y * (1 - cfg.label_smooth) + 0.5 * cfg.label_smooth

            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=cfg.amp and device.type == "cuda"):
                loss = lossf(model(img, meta), y) / accum
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
            f"val AUC {vm['auc']:.4f} | val F1 {vm['f1']:.4f} | "
            f"bal-acc {vm['balanced_accuracy']:.4f} | {row['secs']:.0f}s")

        if score > best["score"]:
            best = {"score": score, "epoch": epoch, "threshold": thr}
            # Only the EMA weights are saved: they are what evaluation uses,
            # and storing the raw weights as well doubles a ~660 MB file per
            # model for no benefit. Non-float buffers (e.g. BatchNorm's
            # num_batches_tracked) are not EMA-tracked and do not affect
            # inference here.
            torch.save({"ema": ema.shadow, "epoch": epoch,
                        "cfg": asdict(cfg)}, ckpt)
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                log(f"  early stop at epoch {epoch} (best epoch {best['epoch']})")
                break

    # ---- final evaluation on the held-out test set --------------------------
    state = torch.load(ckpt, map_location=device, weights_only=False)
    if "model" in state:                      # checkpoints from before the change
        model.load_state_dict(state["model"])
    ema.shadow = {k: v.to(device) for k, v in state["ema"].items()}
    ema.copy_to(model)

    val_logits, val_y = predict(model, val_loader, device, cfg, channels_last)
    test_logits, test_y = predict(model, test_loader, device, cfg, channels_last)

    temperature = 1.0
    if cfg.calibrate:
        ts = TemperatureScaler()
        temperature = ts.fit(val_logits, val_y)
        val_logits_c, test_logits_c = val_logits / temperature, test_logits / temperature
    else:
        val_logits_c, test_logits_c = val_logits, test_logits

    val_prob = sigmoid(val_logits_c)
    test_prob = sigmoid(test_logits_c)

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
        "val": val_metrics,
        "test": test_metrics,
        "test_ci95": ci,
        "history": history,
    }

    np.savez_compressed(os.path.join(out_dir, "predictions.npz"),
                        test_prob=test_prob, test_y=test_y,
                        val_prob=val_prob, val_y=val_y)
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    log(f"\n  TEST  acc {test_metrics['accuracy']:.4f} | bal-acc "
        f"{test_metrics['balanced_accuracy']:.4f} | prec {test_metrics['precision']:.4f} | "
        f"rec {test_metrics['recall']:.4f} | F1 {test_metrics['f1']:.4f} | "
        f"AUC {test_metrics['auc']:.4f}")
    return result
