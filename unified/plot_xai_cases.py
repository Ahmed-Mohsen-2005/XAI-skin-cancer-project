"""Large, readable XAI comparison figures (run after common_xai.py + compare_all.py).

    python plot_xai_cases.py
Writes into outputs/comparison:
  fig_xai_case_<case>.png/pdf   one figure per test case: 14 models (2x7) x 3 methods
  fig_xai_tier1_<case>.png/pdf  Tier-1 models x all methods for one case
"""
import json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from compare_all import SHORT, save

import argparse
_ap = argparse.ArgumentParser(); _ap.add_argument("--xai", default="outputs/common_xai"); _ap.add_argument("--cmp", default="outputs/comparison")
_a = _ap.parse_args(); X, OUT = _a.xai, _a.cmp
plt.rcParams.update({"font.family": "serif"})
rank = pd.read_csv(os.path.join(OUT, "ranking.csv"))
order, tier, thr = list(rank.key), dict(zip(rank.key, rank.tier)), dict(zip(rank.key, rank.thr))
cases = json.load(open(os.path.join(X, "cases.json")))["gallery"]
raw = np.load(os.path.join(X, "gallery_raw.npz"))
J = {k: json.load(open(os.path.join(X, f"{k}.json"))) for k in order}
M = {k: np.load(os.path.join(X, f"{k}_maps.npz")) for k in order}
slug = lambda s: s.split(" (")[0].lower().replace(" ", "_")
GT = {"TP (consensus)": 1, "FN (hardest)": 1, "Disputed melanoma": 1, "TN (consensus)": 0, "FP (hardest)": 0}


def tile(ax, img, h, kind, title=None, p=None, t=None, fs=9):
    ax.imshow(img)
    if h is not None:
        ax.imshow(np.ma.masked_less(h, .2), cmap="inferno", alpha=.6, vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title, fontsize=fs, pad=3)
    if p is not None:
        col = "#1a9850" if (p >= t) == bool(GT[kind]) else "#d73027"
        ax.text(4, 26, f"p={p:.2f}", color="w", fontsize=fs - 1, fontweight="bold",
                bbox=dict(fc=col, alpha=.85, pad=1.5, lw=0))
        for s in ax.spines.values(): s.set_edgecolor(col); s.set_linewidth(2.2)


METHODS = ["Grad-CAM", "Integrated Gradients", "Occlusion"]
for kind in cases:
    img = raw[kind].astype(float) / 255
    gt = "melanoma" if GT[kind] else "benign"
    # ---------- all 14 models, 2 rows of 7 per method
    fig = plt.figure(figsize=(16, 15.5))
    gs = fig.add_gridspec(len(METHODS) * 2 + 1, 8, hspace=.45, wspace=.06,
                          height_ratios=[1.15] + [1] * (len(METHODS) * 2))
    ax0 = fig.add_subplot(gs[0, 3:5]); tile(ax0, img, None, kind)
    ax0.set_title(f"{kind} — ground truth: {gt}", fontsize=13, pad=6)
    for mi, m in enumerate(METHODS):
        for i, k in enumerate(order):
            r, c = 1 + mi * 2 + i // 7, 1 + i % 7
            ax = fig.add_subplot(gs[r, c])
            key = f"{kind}|{m}"
            h = M[k][key].astype(float) if key in M[k].files else None
            tile(ax, img, h, kind, f"#{i+1} {SHORT[k]}".replace(" + ", "\n+ ", 1), J[k]["gallery"][kind]["prob"], thr[k], fs=8)
            if i == 0:
                lab = fig.add_subplot(gs[r:r + 2, 0]); lab.axis("off")
                lab.text(.95, .5, m.replace(" ", "\n"), ha="right", va="center", fontsize=14, fontweight="bold")
    fig.text(.5, .005, "Green frame: correct prediction; red: wrong. Models ordered by final rank (#1 = best). "
             "Overlay shows normalised attribution ≥ 0.2.", ha="center", fontsize=10)
    save(fig, OUT, f"fig_xai_case_{slug(kind)}")

    # ---------- Tier-1 models x all methods
    t1 = [k for k in order if tier[k] == 1]
    meths = ["Grad-CAM", "Grad-CAM++", "Integrated Gradients", "Occlusion", "Attention (grad-rollout)"]
    fig, axs = plt.subplots(len(t1), len(meths) + 1, figsize=(2.3 * (len(meths) + 1), 2.4 * len(t1)))
    for r, k in enumerate(t1):
        tile(axs[r, 0], img, None, kind, "Input" if r == 0 else None, J[k]["gallery"][kind]["prob"], thr[k])
        axs[r, 0].set_ylabel(f"#{r+1} {SHORT[k]}", fontsize=10, fontweight="bold")
        for c, m in enumerate(meths, 1):
            name = m.replace(" (grad-rollout)", " rollout")
            key = f"{kind}|{m}"
            if key in M[k].files:
                tile(axs[r, c], img, M[k][key].astype(float), kind, name if r == 0 else None)
            else:
                axs[r, c].axis("off")
                axs[r, c].text(.5, .5, "n/a (not ViT)" if "rollout" in m else "n/a", ha="center", va="center",
                               color="gray", transform=axs[r, c].transAxes)
                if r == 0: axs[r, c].set_title(name, fontsize=9)
    fig.suptitle(f"Tier-1 models, all attribution methods — {kind} (GT: {gt})", fontsize=12)
    plt.tight_layout()
    save(fig, OUT, f"fig_xai_tier1_{slug(kind)}")
print("done")
