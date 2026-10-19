"""
Regenerate the explainability outputs for an already-trained model.

Needed because a run keeps its checkpoint but its XAI stage can fail (or can
predate a fix) without the expensive training being invalid. This reloads
best.pt, restores the EMA weights that were used for evaluation, and redoes
the single-case panel and the SHAP attribution, then updates results.json in
place. Training results are never recomputed or altered.

    python redo_xai.py --runs ../../runs --data-root ../../data            # all
    python redo_xai.py --runs ../../runs --data-root ../../data \
        --models vit_early_fusion --shap-only

Run this when the GPU is free: it loads a model and will contend for VRAM with
an in-progress sweep on a small card.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from config import MODEL_ZOO, get_config
from data import build_loaders
from models import build_model
import xai as X

warnings.filterwarnings("ignore")


def redo(run_dir: str, data_root: str, shap_only: bool, device_str: str) -> bool:
    rp = os.path.join(run_dir, "results.json")
    cp = os.path.join(run_dir, "best.pt")
    if not (os.path.exists(rp) and os.path.exists(cp)):
        return False

    r = json.load(open(rp, encoding="utf-8"))
    model_name, feature_set = r["model_name"], r["feature_set"]
    saved = r.get("config", {})

    cfg = get_config(model_name, feature_set=feature_set, data_root=data_root,
                     num_workers=0)
    for k in ("img_size", "shap_background", "shap_nsamples", "ig_steps",
              "mean", "std"):
        if k in saved:
            setattr(cfg, k, saved[k])

    device = torch.device(device_str if (device_str != "cuda" or
                                         torch.cuda.is_available()) else "cpu")
    print(f"\n{r.get('display', model_name)}  [{feature_set}]  on {device}")

    _, _, test_loader, meta_cols = build_loaders(cfg)
    ds = test_loader.dataset

    model = build_model(model_name, len(meta_cols), cfg.img_size)
    state = torch.load(cp, map_location="cpu", weights_only=False)
    if "model" in state:          # checkpoints written before the EMA-only change
        model.load_state_dict(state["model"])
    # restore the EMA weights: these are the ones the reported metrics used
    sd = model.state_dict()
    for k, v in state.get("ema", {}).items():
        if k in sd:
            sd[k].copy_(v.to(sd[k].dtype))
    model = model.to(device).eval()

    r["backbone_loaded"] = model.backbone_name()

    preds = np.load(os.path.join(run_dir, "predictions.npz"))
    case = X.select_case(preds["test_prob"], preds["test_y"], want_class=1)
    case_id = str(ds.df.iloc[case].get("isic_id", f"idx{case}"))
    xai = r.get("xai", {}) or {}
    xai.update({"case_index": int(case), "case_id": case_id})

    if not shap_only:
        panel = X.render_xai_panel(model, cfg, ds, case, device, run_dir,
                                   r.get("display", model_name), case_id)
        xai.update({k: v for k, v in panel.items() if k != "panel"})
        print(f"  panel -> {', '.join(panel['methods'])}")

    n_bg = max(20, min(cfg.shap_background, len(ds)))
    bg = ds.meta[np.random.default_rng(cfg.seed).choice(len(ds), n_bg, replace=False)]
    img_fixed = ds[case][0][None].to(device)
    sv = X.shap_metadata(model, bg, ds.meta[case], img_fixed, device,
                         cfg.shap_nsamples)
    if sv is None or len(sv) != len(meta_cols):
        print(f"  [warn] SHAP unavailable (got {None if sv is None else len(sv)} "
              f"values for {len(meta_cols)} features)")
        r.pop("xai_error", None)
    else:
        X.render_shap_bar(sv, meta_cols, run_dir, r.get("display", model_name))
        xai["shap_values"] = {c: float(v) for c, v in zip(meta_cols, sv)}
        top = sorted(xai["shap_values"].items(), key=lambda kv: -abs(kv[1]))[:5]
        print("  shap top-5: " + ", ".join(f"{k}={v:+.3f}" for k, v in top))
        r.pop("xai_error", None)

    r["xai"] = xai
    json.dump(r, open(rp, "w", encoding="utf-8"), indent=2)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--shap-only", action="store_true")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--only-missing", action="store_true",
                    help="skip runs that already have SHAP values")
    a = ap.parse_args()

    dirs = sorted(glob.glob(os.path.join(a.runs, "*", "results.json")))
    n = 0
    for rp in dirs:
        d = os.path.dirname(rp)
        r = json.load(open(rp, encoding="utf-8"))
        if a.models and r["model_name"] not in a.models:
            continue
        if a.only_missing and (r.get("xai", {}) or {}).get("shap_values"):
            continue
        try:
            n += bool(redo(d, a.data_root, a.shap_only, a.device))
        except Exception as exc:  # noqa: BLE001
            print(f"  [error] {r['model_name']}: {type(exc).__name__}: {exc}")
    print(f"\nregenerated XAI for {n} run(s)")


if __name__ == "__main__":
    main()
