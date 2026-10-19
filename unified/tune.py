"""
Optuna hyper-parameter search -- deliberately SHARED and EQUAL-BUDGET.

Fairness rule enforced here: the search space and the trial budget are
identical for every architecture, and the objective is validation AUC only.
The test set is never touched during tuning. If tuning is used for one model
in the benchmark it must be used for all of them with the same budget;
otherwise the comparison silently rewards whichever model received more
search effort, which is one of the defects this rewrite exists to remove.

    python tune.py --model vit_early_fusion --trials 15
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from config import MODEL_ZOO, get_config
from data import build_loaders
from engine import (ModelEMA, WarmupCosine, build_optimizer, predict, sigmoid,
                    seed_everything)
from metrics import choose_threshold, compute_metrics
from models import build_model

# Shared search space -- identical for all 14 architectures.
SEARCH_SPACE = {
    "lr_head":      ("loguniform", 5e-5, 1e-3),
    "lr_backbone":  ("loguniform", 5e-6, 1e-4),
    "weight_decay": ("loguniform", 1e-3, 1e-1),
    "label_smooth": ("categorical", [0.0, 0.05, 0.1]),
    "ema_decay":    ("categorical", [0.995, 0.999, 0.9995]),
}


def _suggest(trial, name, spec):
    kind = spec[0]
    if kind == "loguniform":
        return trial.suggest_float(name, spec[1], spec[2], log=True)
    if kind == "uniform":
        return trial.suggest_float(name, spec[1], spec[2])
    if kind == "categorical":
        return trial.suggest_categorical(name, spec[1])
    raise ValueError(kind)


def _short_train(model, loaders, cfg, epochs: int) -> float:
    """Truncated training run scored on validation AUC."""
    train_loader, val_loader, _ = loaders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    channels_last = cfg.channels_last and device.type == "cuda"
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    accum = max(1, cfg.effective_batch // cfg.micro_batch)
    steps = max(1, len(train_loader) // accum)
    opt = build_optimizer(model, cfg)
    sched = WarmupCosine(opt, steps, epochs * steps, cfg.min_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    lossf = torch.nn.BCEWithLogitsLoss()
    ema = ModelEMA(model, cfg.ema_decay)

    best = 0.0
    for _ in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        for i, (img, meta, y) in enumerate(train_loader):
            img = img.to(device, non_blocking=True)
            if channels_last:
                img = img.contiguous(memory_format=torch.channels_last)
            meta, y = meta.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if cfg.label_smooth:
                y = y * (1 - cfg.label_smooth) + 0.5 * cfg.label_smooth
            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=cfg.amp and device.type == "cuda"):
                loss = lossf(model(img, meta), y) / accum
            scaler.scale(loss).backward()
            if (i + 1) % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                ema.update(model)
        backup = ema.copy_to(model)
        logits, y_true = predict(model, val_loader, device, cfg, channels_last)
        ema.restore(model, backup)
        prob = sigmoid(logits)
        thr = choose_threshold(y_true, prob, cfg.threshold_rule)
        best = max(best, compute_metrics(y_true, prob, thr)["auc"])
    return best


def tune_model(model_name: str, n_trials: int = 15, epochs: int = 4,
               data_root: str = "/kaggle/input",
               out_root: str = "/kaggle/working/outputs") -> dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    out_dir = os.path.join(out_root, f"{model_name}__tuning")
    os.makedirs(out_dir, exist_ok=True)

    def objective(trial):
        params = {k: _suggest(trial, k, v) for k, v in SEARCH_SPACE.items()}
        cfg = get_config(model_name, data_root=data_root, num_workers=4, **params)
        seed_everything(cfg.seed, cfg.deterministic)
        tr, va, te, meta_cols = build_loaders(cfg)
        model = build_model(model_name, len(meta_cols), cfg.img_size)
        try:
            return _short_train(model, (tr, va, te), cfg, epochs)
        finally:
            del model
            torch.cuda.empty_cache()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1))
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    out = {"model": model_name, "n_trials": n_trials, "tuning_epochs": epochs,
           "search_space": {k: list(v) for k, v in SEARCH_SPACE.items()},
           "best_value_val_auc": study.best_value, "best_params": study.best_params,
           "trials": [{"number": t.number, "value": t.value, "params": t.params}
                      for t in study.trials]}
    with open(os.path.join(out_dir, "optuna.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"  best val AUC {study.best_value:.4f} with {study.best_params}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_ZOO))
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--data-root", default="/kaggle/input")
    ap.add_argument("--out-root", default="/kaggle/working/outputs")
    a = ap.parse_args()
    tune_model(a.model, a.trials, a.epochs, a.data_root, a.out_root)
