"""
Unified-case XAI: every model is explained on the SAME test images.

The per-run XAI suite picks cases from each model's own probabilities, so its
scores are not comparable across models. This script fixes one case set for
all runs (model-independent, seed-fixed) and reruns attribution + faithfulness
+ localisation + grouped Shapley on it.

    python common_xai.py --data-root ../../data --out outputs/common_xai \
        --runs outputs "xai_results of jana, malak and abdelrahman/xai_results"
"""
from __future__ import annotations

import argparse, glob, json, os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from config import get_config
from data import build_loaders
from engine import load_ema_weights
from models import build_model
import xai as X

warnings.filterwarnings("ignore")

# Weights come from best.pt, so never fetch ImageNet/HF weights over the network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import timm  # noqa: E402
_create = timm.create_model
def _create_offline(*a, **k):
    k["pretrained"] = False
    return _create(*a, **k)
timm.create_model = _create_offline
try:  # SegFormer: build the MiT-B5 encoder from its config instead of the Hub
    from transformers import SegformerConfig, SegformerModel
    SegformerModel.from_pretrained = classmethod(lambda cls, *a, **k: cls(SegformerConfig(
        depths=[3, 6, 40, 3], hidden_sizes=[64, 128, 320, 512], num_attention_heads=[1, 2, 5, 8],
        sr_ratios=[8, 4, 2, 1], decoder_hidden_size=768)))
except ImportError:
    pass


def find_runs(roots, match=".*"):
    import re
    out = {}
    for r in roots:
        for p in glob.glob(os.path.join(r, "**", "results.json"), recursive=True):
            d = os.path.dirname(p)
            if os.path.exists(os.path.join(d, "best.pt")) and "common_xai" not in d and re.fullmatch(match, os.path.basename(d)):
                out[json.load(open(p, encoding="utf-8"))["model_name"]] = d
    return out


def pick_cases(runs, n_eval=32, seed=0):
    P, T, y = [], [], None
    for d in runs.values():
        z = np.load(os.path.join(d, "predictions.npz"))
        P.append(z["test_prob"]); y = z["test_y"].astype(int)
        T.append(json.load(open(os.path.join(d, "results.json")))["threshold"])
    pred = np.stack([p >= t for p, t in zip(P, T)])          # M x N
    correct = (pred == y[None]).sum(0)
    mp = np.stack(P).mean(0)
    M = len(P)
    g = {}
    g["TP (consensus)"] = int(np.where((y == 1) & (correct == M))[0][np.argmax(mp[(y == 1) & (correct == M)])])
    g["TN (consensus)"] = int(np.where((y == 0) & (correct == M))[0][np.argmin(mp[(y == 0) & (correct == M)])])
    fn = np.where(y == 1)[0]; g["FN (hardest)"] = int(fn[np.argmin(mp[fn])])
    fp = np.where(y == 0)[0]; g["FP (hardest)"] = int(fp[np.argmax(mp[fp])])
    mel = np.where(y == 1)[0]; g["Disputed melanoma"] = int(mel[np.argmin(np.abs(correct[mel] - M / 2))])
    rng = np.random.default_rng(seed)
    ev = np.concatenate([rng.choice(np.where(y == c)[0], n_eval // 2, replace=False) for c in (0, 1)])
    return g, np.sort(ev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="outputs/common_xai")
    ap.add_argument("--match", default=r".*__no_confirm", help="regex on run-folder name")
    ap.add_argument("--n-eval", type=int, default=32)
    ap.add_argument("--n-shap", type=int, default=12)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = find_runs(a.runs, a.match)
    gal, ev = pick_cases(runs, a.n_eval)
    json.dump({"gallery": gal, "eval": ev.tolist()}, open(os.path.join(a.out, "cases.json"), "w"), indent=1)
    print("models:", len(runs), "gallery:", gal)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for name, d in sorted(runs.items()):
        dest = os.path.join(a.out, f"{name}.json")
        if os.path.exists(dest):
            continue
        r = json.load(open(os.path.join(d, "results.json"), encoding="utf-8"))
        cfg = get_config(name, feature_set=r["feature_set"], data_root=a.data_root)
        tr, _, te, meta_cols = build_loaders(cfg, dev)
        ds = te.dataset
        try:
            model = load_ema_weights(build_model(name, len(meta_cols), cfg.img_size), os.path.join(d, "best.pt"), dev)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {name}: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
            continue
        prob = np.load(os.path.join(d, "predictions.npz"))["test_prob"]
        thr, T = r["threshold"], r.get("temperature", 1.0)
        spec = X.cam_spec(model)
        res = {"display": r.get("display", name), "gallery": {}, "eval": {}}
        maps_out = {}
        with X.frozen(model):
            for kind, i in gal.items():
                im, mt, _ = X._case_tensors(ds, i, dev)
                s = 1.0 if prob[i] >= thr else -1.0
                mp, _ = X.explain_case(model, im, mt, s, cfg, spec, fast=True)
                les, fv = X.lesion_and_fov(ds.raw(i))
                res["gallery"][kind] = {"idx": i, "prob": float(prob[i]),
                                        "scores": X._score_maps(model, im, mt, s, mp, les, fv)}
                for m, h in mp.items():
                    maps_out[f"{kind}|{m}"] = np.asarray(h, np.float16)
            agg = {}
            for i in ev:
                im, mt, _ = X._case_tensors(ds, int(i), dev)
                s = 1.0 if prob[i] >= thr else -1.0
                mp, _ = X.explain_case(model, im, mt, s, cfg, spec, fast=True)
                les, fv = X.lesion_and_fov(ds.raw(int(i)))
                for m, sc in X._score_maps(model, im, mt, s, mp, les, fv).items():
                    for k, v in sc.items():
                        agg.setdefault(m, {}).setdefault(k, []).append(float(v))
            res["eval"] = agg
            groups = X.meta_groups(ds.meta_cols)
            rng = np.random.default_rng(42)
            bg = tr.dataset.meta[rng.choice(len(tr.dataset), cfg.shap_background, replace=False)]
            phis = []
            for i in ev[:: max(1, len(ev) // a.n_shap)][: a.n_shap]:
                im, _, _ = X._case_tensors(ds, int(i), dev)
                phi, _, _ = X.grouped_shapley(model, im, ds.meta[int(i)], bg, groups, dev, T)
                phis.append(np.asarray(phi, float))
            res["shap_mean_abs"] = {n: float(v) for (n, _), v in zip(groups, np.abs(phis).mean(0))}
            res["shap_mean"] = {n: float(v) for (n, _), v in zip(groups, np.mean(phis, 0))}
            # grouped permutation importance on the SAME 512 test cases for every model
            pidx = np.sort(np.random.default_rng(0).choice(len(ds), min(512, len(ds)), replace=False))
            res["perm"] = X.grouped_permutation_importance(model, ds, pidx, groups, dev, 42)
        np.savez_compressed(os.path.join(a.out, f"{name}_maps.npz"), **maps_out)
        json.dump(res, open(dest, "w"), indent=1)
        print(f"done {name}", flush=True)
        del model; torch.cuda.empty_cache()

    # raw images of the gallery for plotting
    if not os.path.exists(os.path.join(a.out, "gallery_raw.npz")):
        name, d = next(iter(sorted(runs.items())))
        cfg = get_config(name, data_root=a.data_root)
        _, _, te, _ = build_loaders(cfg, dev)
        np.savez_compressed(os.path.join(a.out, "gallery_raw.npz"),
                            **{k: te.dataset.raw(i) for k, i in gal.items()})


if __name__ == "__main__":
    main()
