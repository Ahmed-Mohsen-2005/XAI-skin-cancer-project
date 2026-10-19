"""
Single entry point for every experiment in the study.

    python run.py --model vit_early_fusion
    python run.py --model segformer_laca --feature-set no_confirm
    python run.py --all                       # sequential sweep
    python run.py --model davit_gated --tune 20   # Optuna search first

Writes, per run:
    outputs/<model>__<feature_set>/results.json     metrics + config + history
    outputs/<model>__<feature_set>/predictions.npz  test/val probabilities
    outputs/<model>__<feature_set>/best.pt          checkpoint
    outputs/<model>__<feature_set>/xai_panel.png    single-case XAI figure
    outputs/<model>__<feature_set>/shap_metadata.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_ZOO, FEATURE_SETS, get_config
from data import build_loaders
from engine import seed_everything, train_one_model
from models import build_model, count_params
import xai as X

warnings.filterwarnings("ignore", category=UserWarning)


def run_one(model_name: str, feature_set: str, args) -> dict:
    cfg = get_config(model_name, feature_set=feature_set,
                     data_root=args.data_root, out_root=args.out_root,
                     epochs=args.epochs, num_workers=args.workers)
    if args.micro_batch:
        cfg.micro_batch = args.micro_batch

    tag = f"{model_name}__{feature_set}"
    out_dir = os.path.join(cfg.out_root, tag)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 78)
    print(f"  {MODEL_ZOO[model_name]['display']}   [{feature_set}]")
    print(f"  {MODEL_ZOO[model_name]['note']}")
    print("=" * 78)

    seed_everything(cfg.seed, cfg.deterministic)
    train_loader, val_loader, test_loader, meta_cols = build_loaders(cfg)
    print(f"  split  train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
          f"test={len(test_loader.dataset)} | metadata features={len(meta_cols)}")

    model = build_model(model_name, len(meta_cols), cfg.img_size)
    total, trainable = count_params(model)
    print(f"  params {total/1e6:.1f}M ({trainable/1e6:.1f}M trainable) | "
          f"micro-batch {cfg.micro_batch} x accum {cfg.effective_batch // cfg.micro_batch} "
          f"= effective {cfg.effective_batch}\n")

    result = train_one_model(model, (train_loader, val_loader, test_loader),
                             cfg, out_dir)
    result["meta_cols"] = meta_cols
    result["display"] = MODEL_ZOO[model_name]["display"]
    result["family"] = MODEL_ZOO[model_name]["family"]
    result["origin"] = MODEL_ZOO[model_name]["origin"]
    result["novel_on_dataset"] = MODEL_ZOO[model_name]["novel_on_dataset"]

    # ---- single-case XAI -------------------------------------------------
    if not args.no_xai:
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            preds = np.load(os.path.join(out_dir, "predictions.npz"))
            case = X.select_case(preds["test_prob"], preds["test_y"], want_class=1)
            ds = test_loader.dataset
            case_id = str(ds.df.iloc[case].get("isic_id", f"idx{case}"))
            print(f"\n  XAI on a single case: {case_id} "
                  f"(p={preds['test_prob'][case]:.3f})")

            panel = X.render_xai_panel(model, cfg, ds, case, device, out_dir,
                                       result["display"], case_id)
            result["xai"] = {"case_index": int(case), "case_id": case_id, **
                             {k: v for k, v in panel.items() if k != "panel"}}
            print(f"    panel -> {panel['panel']}  ({', '.join(panel['methods'])})")

            n_bg = max(20, min(cfg.shap_background, len(ds)))
            bg = ds.meta[np.random.default_rng(cfg.seed)
                         .choice(len(ds), n_bg, replace=False)]
            img_fixed = ds[case][0][None].to(device)
            sv = X.shap_metadata(model, bg, ds.meta[case], img_fixed, device,
                                 cfg.shap_nsamples)
            if sv is not None:
                p = X.render_shap_bar(sv, meta_cols, out_dir, result["display"])
                result["xai"]["shap_values"] = {c: float(v) for c, v in zip(meta_cols, sv)}
                print(f"    shap  -> {p}")
        except Exception as exc:  # noqa: BLE001
            print(f"    [warn] XAI stage failed: {type(exc).__name__}: {exc}")
            result["xai_error"] = f"{type(exc).__name__}: {exc}"

    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, choices=list(MODEL_ZOO))
    ap.add_argument("--all", action="store_true", help="run every model in the zoo")
    ap.add_argument("--feature-set", default="full", choices=list(FEATURE_SETS))
    ap.add_argument("--ablation", action="store_true",
                    help="also run no_confirm / no_age / no_both for the model")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--micro-batch", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--data-root", default="/kaggle/input")
    ap.add_argument("--out-root", default="/kaggle/working/outputs")
    ap.add_argument("--tune", type=int, default=0, help="Optuna trials before the final run")
    ap.add_argument("--no-xai", action="store_true")
    args = ap.parse_args()

    if not args.model and not args.all:
        ap.error("pass --model NAME or --all")

    models = list(MODEL_ZOO) if args.all else [args.model]
    feature_sets = ([args.feature_set] if not args.ablation
                    else [args.feature_set] + [f for f in FEATURE_SETS if f != args.feature_set])

    if args.tune:
        from tune import tune_model
        for m in models:
            tune_model(m, n_trials=args.tune, data_root=args.data_root,
                       out_root=args.out_root)

    summary = []
    for m in models:
        for fs in feature_sets:
            t0 = time.time()
            try:
                r = run_one(m, fs, args)
                summary.append({"model": m, "feature_set": fs,
                                "auc": r["test"]["auc"], "f1": r["test"]["f1"],
                                "acc": r["test"]["accuracy"],
                                "minutes": round((time.time() - t0) / 60, 1)})
            except Exception as exc:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                summary.append({"model": m, "feature_set": fs, "error": str(exc)})

    print("\n" + "=" * 78)
    print("  SWEEP SUMMARY")
    print("=" * 78)
    for s in summary:
        if "error" in s:
            print(f"  {s['model']:24s} {s['feature_set']:11s} FAILED: {s['error'][:60]}")
        else:
            print(f"  {s['model']:24s} {s['feature_set']:11s} "
                  f"AUC {s['auc']:.4f}  F1 {s['f1']:.4f}  acc {s['acc']:.4f}  "
                  f"({s['minutes']} min)")
    os.makedirs(args.out_root, exist_ok=True)
    with open(os.path.join(args.out_root, "sweep_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
