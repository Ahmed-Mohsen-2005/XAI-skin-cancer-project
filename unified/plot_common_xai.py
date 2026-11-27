"""Figures + stats for the unified-case XAI (run common_xai.py first).

    python plot_common_xai.py --xai outputs/common_xai --cmp outputs/comparison
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from compare_all import SHORT, save

plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.titlesize": 8})
METHODS = ["Grad-CAM", "Grad-CAM++", "Integrated Gradients", "Occlusion", "Attention (grad-rollout)"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--xai", default="outputs/common_xai"); ap.add_argument("--cmp", default="outputs/comparison")
    a = ap.parse_args(); out = a.cmp
    rank = pd.read_csv(os.path.join(a.cmp, "ranking.csv")); order = [k for k in rank.key if os.path.exists(os.path.join(a.xai, f"{k}.json"))]
    J = {k: json.load(open(os.path.join(a.xai, f"{k}.json"))) for k in order}
    cases = json.load(open(os.path.join(a.xai, "cases.json")))["gallery"]
    raw = np.load(os.path.join(a.xai, "gallery_raw.npz"))
    lab = {k: SHORT.get(k, k) for k in order}

    # ---------------- qualitative grids: one per method, rows = cases, cols = models
    for meth in ("Grad-CAM", "Integrated Gradients", "Occlusion"):
        ks = [k for k in order if f"{next(iter(cases))}|{meth}" in np.load(os.path.join(a.xai, f"{k}_maps.npz")).files]
        if not ks: continue
        fig, axs = plt.subplots(len(cases), len(ks) + 1, figsize=(7.2, 7.2 * len(cases) / (len(ks) + 1) + .6))
        for r, kind in enumerate(cases):
            img = raw[kind].astype(float) / 255
            axs[r, 0].imshow(img); axs[r, 0].set_ylabel(kind.replace(" (", "\n("), fontsize=5, rotation=0, ha="right", va="center")
            for c, k in enumerate(ks, 1):
                h = np.load(os.path.join(a.xai, f"{k}_maps.npz"))[f"{kind}|{meth}"].astype(float)
                axs[r, c].imshow(img); axs[r, c].imshow(np.ma.masked_less(h, .25), cmap="jet", alpha=.55, vmin=0, vmax=1)
                p = J[k]["gallery"][kind]["prob"]
                axs[r, c].text(3, 22, f"{p:.2f}", color="w", fontsize=4.2, bbox=dict(fc="k", alpha=.55, pad=.4, lw=0))
                if r == 0: axs[r, c].set_title(lab[k], fontsize=4.3, rotation=90, va="bottom")
        for ax in axs.ravel(): ax.set_xticks([]); ax.set_yticks([])
        axs[0, 0].set_title("Input", fontsize=5, rotation=90, va="bottom")
        plt.subplots_adjust(wspace=.03, hspace=.03)
        save(fig, out, "fig_xai_grid_" + meth.lower().replace(" ", "_").replace("-", ""))

    # ---------------- quantitative: per model x method
    recs = []
    for k in order:
        ev = J[k]["eval"]; rnd = ev["_random"]
        for m, d in ev.items():
            if m == "_random": continue
            n = len(d["deletion"])
            gap = np.array(d["insertion"]) - np.array(d["deletion"]); rgap = np.array(rnd["insertion"][:n]) - np.array(rnd["deletion"][:n])
            rec = dict(key=k, model=lab[k], method=m, deletion=np.mean(d["deletion"]), insertion=np.mean(d["insertion"]),
                       faith=np.mean(gap), faith_gain=np.mean(gap - rgap), wilcoxon_p=stats.wilcoxon(gap, rgap).pvalue if n > 5 else np.nan,
                       outside_fov=np.mean(d["energy_outside_fov"]))
            if "energy_in_lesion" in d:
                rec["in_lesion"] = np.mean(d["energy_in_lesion"]); rec["focus_gain"] = np.median(d["lesion_focus_gain"])
            recs.append(rec)
    X = pd.DataFrame(recs); X.to_csv(os.path.join(out, "xai_common_scores.csv"), index=False)
    meths = [m for m in METHODS if m in set(X.method)]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.6), sharey=True)
    for ax, (c, t, cm) in zip(axs, (("faith_gain", "(a) Faithfulness gain\n(Ins−Del) − random", "RdBu_r"),
                                    ("in_lesion", "(b) Attribution energy\ninside lesion", "viridis"),
                                    ("outside_fov", "(c) Energy outside\nfield of view (↓)", "magma_r"))):
        P = X.pivot(index="key", columns="method", values=c).reindex(index=order, columns=meths)
        lim = np.nanmax(abs(P.values)) if c == "faith_gain" else None
        im = ax.imshow(P.values, cmap=cm, aspect="auto", vmin=-lim if lim else None, vmax=lim)
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                if P.values[i, j] == P.values[i, j]: ax.text(j, i, f"{P.values[i, j]:.2f}", ha="center", va="center", fontsize=4.8)
        ax.set_xticks(range(len(meths))); ax.set_xticklabels([m.replace(" (grad-rollout)", "\nrollout").replace("Integrated ", "Int.\n") for m in meths], fontsize=5.5)
        ax.set_title(t, fontsize=7); plt.colorbar(im, ax=ax, fraction=.05)
    axs[0].set_yticks(range(len(order))); axs[0].set_yticklabels([lab[k] for k in order], fontsize=5.5)
    save(fig, out, "fig_xai_quant_heatmap")

    # method-level distribution + Friedman across methods common to all models
    common = [m for m in meths if X[X.method == m].key.nunique() == len(order)]
    F = X.pivot(index="key", columns="method", values="faith_gain")[common].dropna()
    fr = stats.friedmanchisquare(*[F[m] for m in common]) if len(common) > 2 else None
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.boxplot([X[X.method == m].faith_gain for m in meths], labels=[m.replace(" (grad-rollout)", "\nrollout").replace("Integrated ", "Int. ") for m in meths], widths=.5)
    ax.axhline(0, c="gray", ls=":"); ax.set_ylabel("Faithfulness gain over random"); ax.tick_params(axis="x", labelsize=5.5)
    if fr: ax.set_title(f"Friedman χ²={fr.statistic:.1f}, p={fr.pvalue:.1e} (n={len(F)} models)", fontsize=6.5)
    save(fig, out, "fig_xai_method_box")

    # ---------------- metadata attribution: grouped Shapley (common cases) + permutation importance
    S = pd.DataFrame({k: J[k]["shap_mean_abs"] for k in order}).T
    S = S.loc[:, (S != 0).any()]
    perm = {}
    for k in order:   # unified permutation importance from common_xai.py (same 512 cases for every model)
        if "perm" in J[k]: perm[k] = {g: v for g, v in J[k]["perm"].items() if not g.startswith("_")}
    if not perm:      # older common_xai outputs: fall back to each run's own XAI stage
        for f in glob.glob("outputs/*__*/results.json") + glob.glob("xai_results of jana, malak and abdelrahman/**/results.json", recursive=True):
            r = json.load(open(f, encoding="utf-8")); pi = r.get("xai", {}).get("meta_permutation_importance")
            if pi: perm[r["model_name"]] = {g: v for g, v in pi.items() if not g.startswith("_")}
    Pm = pd.DataFrame(perm).T.reindex(order)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    for ax, M, t in ((axs[0], S, "(a) Mean |grouped Shapley| (common cases)"), (axs[1], Pm, "(b) Permutation importance (ΔAUC, common 512 cases)")):
        im = ax.imshow(M.values.astype(float), cmap="YlOrRd", aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]): ax.text(j, i, f"{M.values[i, j]:.3f}", ha="center", va="center", fontsize=4.6)
        ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(M.columns, rotation=35, ha="right", fontsize=5.5); ax.set_title(t, fontsize=7)
        plt.colorbar(im, ax=ax, fraction=.04)
    axs[0].set_yticks(range(len(order))); axs[0].set_yticklabels([lab[k] for k in order], fontsize=5.5)
    save(fig, out, "fig_meta_attribution")
    # Kendall's W of feature rankings across models
    def kendall_w(M):
        Rk = M.rank(axis=1).values; m, n = Rk.shape; s = ((Rk.sum(0) - m * (n + 1) / 2) ** 2).sum(); return 12 * s / (m ** 2 * (n ** 3 - n))
    summ = dict(friedman=(fr.statistic, fr.pvalue) if fr else None, kendall_w_shap=kendall_w(S), kendall_w_perm=kendall_w(Pm.drop(columns=[c for c in Pm if "ALL" in c]).dropna()),
                method_means=X.groupby("method")[["faith_gain", "in_lesion", "outside_fov", "insertion", "deletion"]].mean().round(4).to_dict("index"),
                pct_sig_vs_random=X.groupby("method").wilcoxon_p.apply(lambda p: float((p < .05).mean())).to_dict(),
                shap_mean=S.mean().round(4).to_dict(), perm_mean=Pm.mean().round(4).to_dict())
    # correlation between explanation quality and accuracy
    Q = X[X.method == "Integrated Gradients"].set_index("key").reindex(order)
    summ["spearman_auc_vs_IG_faith"] = stats.spearmanr(rank.set_index("key").loc[order, "auc"], Q.faith_gain).correlation
    summ["spearman_auc_vs_IG_inlesion"] = stats.spearmanr(rank.set_index("key").loc[order, "auc"], Q.in_lesion).correlation
    json.dump(summ, open(os.path.join(out, "xai_summary.json"), "w"), indent=1, default=float)
    print(json.dumps(summ, indent=0, default=lambda v: round(float(v), 4))[:2500])


if __name__ == "__main__":
    main()
