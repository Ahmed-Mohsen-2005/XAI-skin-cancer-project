"""
One-glance status of the training sweep.

    python status.py --runs ../../runs

Shows what has finished, what is training right now, the live epoch line, and
an ETA extrapolated from the epoch times actually observed so far.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time
from datetime import datetime, timedelta

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MODEL_ZOO
from run_all_local import PRIORITY


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="../../runs")
    a = ap.parse_args()
    root = a.runs

    done, epoch_secs = {}, []
    for p in sorted(glob.glob(os.path.join(root, "*", "results.json"))):
        r = json.load(open(p, encoding="utf-8"))
        key = f"{r['model_name']}__{r['feature_set']}"
        done[key] = r
        for h in r.get("history", []):
            if h.get("secs"):
                epoch_secs.append(h["secs"])

    # also take epoch times from live logs, so an ETA exists before the first
    # model has produced a results.json
    for log in glob.glob(os.path.join(root, "log_*.txt")):
        txt = open(log, encoding="utf-8", errors="replace").read()
        epoch_secs += [float(s) for s in re.findall(r"\|\s*([\d.]+)s\s*$", txt, re.M)]

    print("=" * 78)
    print(f"  SWEEP STATUS                              {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 78)

    # finished
    if done:
        print(f"\n  FINISHED ({len(done)}):")
        print(f"    {'model':34s} {'AUC':>7s} {'F1':>7s} {'Acc':>7s} {'BalAcc':>7s} {'min':>6s}")
        for k, r in sorted(done.items(), key=lambda kv: -kv[1]["test"]["auc"]):
            t = r["test"]
            print(f"    {k:34s} {t['auc']:7.4f} {t['f1']:7.4f} "
                  f"{t['accuracy']:7.4f} {t['balanced_accuracy']:7.4f} "
                  f"{r.get('train_minutes', 0):6.0f}")
    else:
        print("\n  FINISHED: none yet")

    # in flight
    live = None
    for log in glob.glob(os.path.join(root, "log_*.txt")):
        key = os.path.basename(log)[4:-4]
        if key in done:
            continue
        age = time.time() - os.path.getmtime(log)
        if age < 1800:
            live = (key, log, age)
    if live:
        key, log, age = live
        txt = open(log, encoding="utf-8", errors="replace").read()
        eps = re.findall(r"epoch\s+(\d+)/(\d+).*", txt)
        last = [l for l in txt.splitlines() if "epoch " in l]
        print(f"\n  TRAINING NOW: {key}")
        if last:
            print(f"    {last[-1].strip()}")
        if eps:
            cur, tot = int(eps[-1][0]), int(eps[-1][1])
            print(f"    progress: epoch {cur}/{tot}")
        print(f"    last log write: {age:.0f}s ago")
    else:
        print("\n  TRAINING NOW: nothing detected "
              "(sweep may be finished, between models, or stopped)")

    # eta
    remaining = [m for m in PRIORITY if f"{m}__full" not in done]
    if remaining and epoch_secs:
        # Use the most recent epochs rather than the all-time median: power
        # mode, thermal state and clock can change mid-sweep, and the recent
        # rate is the one that predicts what is left.
        # The fastest of the last few epochs approximates the steady-state rate
        # under the *current* power/clock state. A median over a window that
        # straddles a power-mode change stays pinned to the old, slower rate.
        med = min(epoch_secs[-3:]) if len(epoch_secs) >= 3 else min(epoch_secs)
        # crude per-model cost weight, relative to a ViT-class model
        weight = {"segformer_laca": 2.0, "swinv2_tabtransformer": 1.2,
                  "swinv2_gat": 1.1, "biovil_lcsa": 1.0, "vit_early_fusion": 1.0,
                  "hybrid_vit_mlp": 1.0, "gnn_maxvit": 0.9,
                  "maxvit_crossattn": 0.9, "dinov2_crossattn": 0.8,
                  "convnext_bicross": 0.7, "convnext_bilinear": 0.7,
                  "davit_gated": 0.7, "effnetv2_crossattn": 0.7,
                  "unet_crossattn": 0.6}
        epochs = list(done.values())[0]["config"]["epochs"] if done else 8
        secs = sum(med * epochs * weight.get(m, 1.0) + 480 for m in remaining)
        eta = datetime.now() + timedelta(seconds=secs)
        print(f"\n  REMAINING: {len(remaining)} model(s)")
        print(f"    current epoch rate  : {med:.0f}s")
        print(f"    estimated finish    : {eta:%Y-%m-%d %H:%M} "
              f"(~{secs/3600:.1f} h)")
        print(f"    queue: {', '.join(remaining)}")
    elif not remaining:
        print("\n  ALL MODELS COMPLETE -- run report.py then build_paper.py")
    print()


if __name__ == "__main__":
    main()
