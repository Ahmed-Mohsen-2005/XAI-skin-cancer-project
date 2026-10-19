"""
Local sweep driver for a 4 GB GPU.

Trains every architecture under the shared protocol, but with per-model
micro-batches sized for a 4 GB card. The EFFECTIVE batch stays at 32 for every
model via gradient accumulation, so the comparison is unaffected -- only the
number of forward passes per optimiser step changes.

On CUDA OOM the micro-batch is halved and the model retried, so one greedy
architecture cannot abort the sweep.

    python run_all_local.py --epochs 8 --data-root ../../data --out-root ../../runs
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_ZOO

# Micro-batch that fits 4 GB with AMP. Effective batch is restored to 32 by
# gradient accumulation inside engine.py.
VRAM_4GB = {
    "vit_early_fusion": 8,
    "hybrid_vit_mlp": 8,
    "biovil_lcsa": 8,
    "segformer_laca": 4,
    "swinv2_tabtransformer": 8,
    "swinv2_gat": 8,
    "convnext_bilinear": 16,
    "maxvit_crossattn": 8,
    "gnn_maxvit": 8,
    "convnext_bicross": 16,
    "davit_gated": 16,
    "dinov2_crossattn": 16,
    "effnetv2_crossattn": 16,
    "unet_crossattn": 32,
}

# Headline models first (the early-vs-late fusion contrast carries the paper),
# then ascending cost so partial sweeps still yield a usable table.
PRIORITY = [
    "vit_early_fusion",
    "hybrid_vit_mlp",
    "unet_crossattn",
    "effnetv2_crossattn",
    "davit_gated",
    "convnext_bicross",
    "convnext_bilinear",
    "dinov2_crossattn",
    "gnn_maxvit",
    "maxvit_crossattn",
    "swinv2_gat",
    "swinv2_tabtransformer",
    "biovil_lcsa",
    "segformer_laca",
]


def run(model: str, feature_set: str, args, micro: int) -> int:
    cmd = [sys.executable, "-u", "run.py",
           "--model", model, "--feature-set", feature_set,
           "--epochs", str(args.epochs), "--micro-batch", str(micro),
           "--workers", str(args.workers),
           "--data-root", args.data_root, "--out-root", args.out_root]
    if args.no_xai:
        cmd.append("--no-xai")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    log_path = os.path.join(args.out_root, f"log_{model}__{feature_set}.txt")
    os.makedirs(args.out_root, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n{'='*70}\n{model} [{feature_set}] micro={micro}\n{'='*70}\n")
        log.flush()
        return subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--feature-sets", nargs="*", default=["full"])
    ap.add_argument("--models", nargs="*", default=PRIORITY)
    ap.add_argument("--no-xai", action="store_true")
    ap.add_argument("--skip-done", action="store_true", default=True)
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    board = os.path.join(args.out_root, "sweep_progress.json")
    progress = {}
    if os.path.exists(board):
        progress = json.load(open(board, encoding="utf-8"))

    todo = [(m, fs) for fs in args.feature_sets for m in args.models]
    print(f"sweep: {len(todo)} runs, {args.epochs} epochs each", flush=True)

    for i, (model, fs) in enumerate(todo, 1):
        key = f"{model}__{fs}"
        done_marker = os.path.join(args.out_root, key, "results.json")
        if args.skip_done and os.path.exists(done_marker):
            print(f"[{i}/{len(todo)}] {key}: already done, skipping", flush=True)
            continue

        micro = VRAM_4GB.get(model, MODEL_ZOO[model]["micro_batch"])
        t0 = time.time()
        for attempt in range(3):
            print(f"[{i}/{len(todo)}] {key}: starting (micro={micro})", flush=True)
            rc = run(model, fs, args, micro)
            if rc == 0 and os.path.exists(done_marker):
                break
            log = os.path.join(args.out_root, f"log_{key}.txt")
            tail = ""
            if os.path.exists(log):
                tail = open(log, encoding="utf-8", errors="replace").read()[-3000:]
            if "out of memory" in tail.lower() and micro > 1:
                micro = max(1, micro // 2)
                print(f"    OOM -> retrying with micro={micro}", flush=True)
                continue
            print(f"    FAILED rc={rc}", flush=True)
            break

        mins = round((time.time() - t0) / 60, 1)
        if os.path.exists(done_marker):
            r = json.load(open(done_marker, encoding="utf-8"))
            t = r["test"]
            progress[key] = {"auc": t["auc"], "f1": t["f1"], "acc": t["accuracy"],
                             "bal_acc": t["balanced_accuracy"],
                             "precision": t["precision"], "recall": t["recall"],
                             "minutes": mins, "micro_batch": micro,
                             "epochs_run": r.get("epochs_run")}
            print(f"    done in {mins} min | AUC {t['auc']:.4f} F1 {t['f1']:.4f} "
                  f"acc {t['accuracy']:.4f}", flush=True)
        else:
            progress[key] = {"error": True, "minutes": mins}
        with open(board, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2)

    print("\nSWEEP COMPLETE", flush=True)
    for k, v in sorted(progress.items(), key=lambda kv: -kv[1].get("auc", 0)):
        if "auc" in v:
            print(f"  {k:36s} AUC {v['auc']:.4f}  F1 {v['f1']:.4f}  "
                  f"({v['minutes']} min)", flush=True)
        else:
            print(f"  {k:36s} FAILED", flush=True)


if __name__ == "__main__":
    main()
