"""
Regenerate the explainability outputs for an already-trained model.

Reloads best.pt (the EMA weights used for evaluation) and reruns the full XAI
suite (scored single-case panel, TP/TN/FP/FN gallery, faithfulness and
localisation scores, grouped Shapley + permutation importance), then updates
results.json. Training results are never recomputed or altered.

    python redo_xai.py --runs outputs --data-root ../../data              # all runs
    python redo_xai.py --runs outputs --data-root ../../data --models vit_early_fusion
    python redo_xai.py --runs outputs --data-root ../../data --out /tmp/xai   # write elsewhere
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

from config import get_config
from data import build_loaders
from engine import keep_awake, load_ema_weights
from models import build_model
import xai as X

warnings.filterwarnings("ignore")


def redo(run_dir: str, data_root: str, device_str: str, out: str = "",
         eval_cases: int | None = None) -> bool:
    rp = os.path.join(run_dir, "results.json")
    cp = os.path.join(run_dir, "best.pt")
    if not (os.path.exists(rp) and os.path.exists(cp)):
        return False

    r = json.load(open(rp, encoding="utf-8"))
    model_name, feature_set = r["model_name"], r["feature_set"]
    cfg = get_config(model_name, feature_set=feature_set, data_root=data_root)
    if eval_cases is not None:
        cfg.xai_eval_cases = eval_cases

    device = torch.device(device_str if (device_str != "cuda" or
                                         torch.cuda.is_available()) else "cpu")
    print(f"\n{r.get('display', model_name)}  [{feature_set}]  on {device}")

    tr, _, te, meta_cols = build_loaders(cfg, device)
    model = load_ema_weights(build_model(model_name, len(meta_cols), cfg.img_size), cp, device)

    preds = np.load(os.path.join(run_dir, "predictions.npz"))
    out_dir = os.path.join(out, os.path.basename(os.path.normpath(run_dir))) if out else run_dir
    xai = X.run_xai_suite(model, cfg, te.dataset, tr.dataset, preds["test_prob"],
                          r.get("threshold", 0.5), r.get("temperature", 1.0), out_dir,
                          r.get("display", model_name), device)
    r["xai"] = xai
    r.pop("xai_error", None)
    dest = os.path.join(out_dir, "results.json") if out else rp
    json.dump(r, open(dest, "w", encoding="utf-8"), indent=2)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--feature-sets", nargs="*", default=None)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--out", default="", help="write outputs here instead of the run dir")
    ap.add_argument("--eval-cases", type=int, default=None)
    ap.add_argument("--only-missing", action="store_true",
                    help="skip runs that already have the scored XAI suite")
    a = ap.parse_args()
    keep_awake(True)

    n = 0
    for rp in sorted(glob.glob(os.path.join(a.runs, "*", "results.json"))):
        d = os.path.dirname(rp)
        r = json.load(open(rp, encoding="utf-8"))
        if a.models and r["model_name"] not in a.models:
            continue
        if a.feature_sets and r["feature_set"] not in a.feature_sets:
            continue
        if a.only_missing and (r.get("xai", {}) or {}).get("eval"):
            continue
        try:
            n += bool(redo(d, a.data_root, a.device, a.out, a.eval_cases))
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"  [error] {r['model_name']}: {type(exc).__name__}: {exc}")
    print(f"\nregenerated XAI for {n} run(s)")


if __name__ == "__main__":
    main()
