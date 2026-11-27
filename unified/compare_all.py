"""
Head-to-head comparison of every trained model on the SAME frozen test set.

    python compare_all.py --data-root ../../data --out outputs/comparison \
        --runs outputs "xai_results of jana, malak and abdelrahman/xai_results"

Writes publication figures (PDF + 300-dpi PNG), LaTeX tables and a ranking CSV.
"""
from __future__ import annotations

import argparse, glob, json, os, itertools
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
                             brier_score_loss, confusion_matrix, cohen_kappa_score)

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                     "legend.fontsize": 6.5, "figure.dpi": 110, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})

SHORT = {"vit_early_fusion": "ViT-B Early Fusion", "hybrid_vit_mlp": "ViT-B Late Fusion (MLP)",
         "gnn_maxvit": "MaxViT-T + GATv2", "segformer_laca": "SegFormer-B5 + LACA",
         "davit_gated": "DaViT-T + Gated", "convnext_bicross": "ConvNeXt-T + BiCross",
         "swinv2_gat": "SwinV2-S + GAT", "maxvit_crossattn": "MaxViT-T + CrossAttn",
         "effnetv2_crossattn": "EffNetV2-S + CrossAttn", "convnext_bilinear": "ConvNeXt-T + Bilinear",
         "swinv2_tabtransformer": "SwinV2-S + TabTransf.", "biovil_lcsa": "ViT-B + LCSA (BioViL-T*)",
         "dinov2_crossattn": "DINOv2-S + CrossAttn", "unet_crossattn": "U-Net (scratch) + CrossAttn"}
FAM_COL = {"Early/Late": "#1b9e77", "Cross-attention": "#d95f02", "Graph": "#7570b3",
           "Gated": "#e7298a", "Bilinear": "#66a61e", "Scratch": "#777777"}
FAMILY = {"vit_early_fusion": "Early/Late", "hybrid_vit_mlp": "Early/Late", "gnn_maxvit": "Graph",
          "swinv2_gat": "Graph", "davit_gated": "Gated", "convnext_bilinear": "Bilinear",
          "unet_crossattn": "Scratch"}


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- statistics
def delong(y, p1, p2):
    """Paired DeLong test (Sun & Xu fast algorithm). Returns (auc1, auc2, z, p)."""
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()

    def comps(p):
        X, Y = p[pos], p[neg]
        allr = stats.rankdata(np.r_[X, Y]); xr = stats.rankdata(X); yr = stats.rankdata(Y)
        auc = (allr[:m].sum() - m * (m + 1) / 2) / (m * n)
        v10 = (allr[:m] - xr) / n
        v01 = 1 - (allr[m:] - yr) / m
        return auc, v10, v01
    a1, v10a, v01a = comps(p1); a2, v10b, v01b = comps(p2)
    s10 = np.cov(np.vstack([v10a, v10b])); s01 = np.cov(np.vstack([v01a, v01b]))
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n
    z = (a1 - a2) / np.sqrt(max(var, 1e-12))
    return a1, a2, z, 2 * stats.norm.sf(abs(z))


def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty_like(p); run = 0
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i]); adj[i] = min(1, run)
    return adj


def ece(y, p, bins=15):
    e = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(p, e) - 1, 0, bins - 1)
    return sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())


def sens_at_spec(y, p, spec):
    f, t, _ = roc_curve(y, p); return float(np.interp(1 - spec, f, t))


def spec_at_sens(y, p, sens):
    f, t, _ = roc_curve(y, p); return float(1 - f[np.searchsorted(t, sens)])


def boot_ci(y, p, fn, B=1000, seed=0):
    rng = np.random.default_rng(seed); n = len(y); v = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if y[i].min() != y[i].max():
            v.append(fn(y[i], p[i]))
    return np.percentile(v, [2.5, 97.5])


# ---------------------------------------------------------------- load
def load(roots, match=".*"):
    import re
    R = {}
    for r in roots:
        for f in glob.glob(os.path.join(r, "**", "results.json"), recursive=True):
            if "common_xai" in f or "comparison" in f or not re.fullmatch(match, os.path.basename(os.path.dirname(f))):
                continue
            j = json.load(open(f, encoding="utf-8")); d = os.path.dirname(f)
            z = np.load(os.path.join(d, "predictions.npz"))
            R[j["model_name"]] = dict(j=j, p=z["test_prob"].astype(float), y=z["test_y"].astype(int),
                                      vp=z["val_prob"].astype(float), vy=z["val_y"].astype(int), dir=d)
    ys = {k: hash(v["y"].tobytes()) for k, v in R.items()}
    assert len(set(ys.values())) == 1, "test sets differ between runs!"
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="outputs/comparison")
    ap.add_argument("--match", default=r".*__no_confirm", help="regex on run-folder name, e.g. '.*__patient'")
    a = ap.parse_args(); out = a.out; os.makedirs(out, exist_ok=True)
    R = load(a.runs, a.match); y = next(iter(R.values()))["y"]; N = len(y)
    fam = {k: FAMILY.get(k, "Cross-attention") for k in R}

    # ---------------- metrics table
    rows = []
    for k, v in R.items():
        p, thr = v["p"], v["j"]["threshold"]; yh = (p >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yh).ravel()
        ci = boot_ci(y, p, roc_auc_score)
        rows.append(dict(key=k, model=SHORT.get(k, k), family=fam[k], auc=roc_auc_score(y, p), auc_lo=ci[0], auc_hi=ci[1],
                         pr_auc=average_precision_score(y, p), acc=(tp + tn) / N, bal_acc=(tp / (tp + fn) + tn / (tn + fp)) / 2,
                         sens=tp / (tp + fn), spec=tn / (tn + fp), ppv=tp / (tp + fp), npv=tn / (tn + fn),
                         f1=2 * tp / (2 * tp + fp + fn), mcc=(tp * tn - fp * fn) / np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))),
                         sens_at_spec90=sens_at_spec(y, p, .90), spec_at_sens95=spec_at_sens(y, p, .95),
                         ece=ece(y, p), brier=brier_score_loss(y, p), T=v["j"]["temperature"], thr=thr,
                         params_M=v["j"]["params_total"] / 1e6, train_min=v["j"].get("train_minutes", np.nan),
                         best_epoch=v["j"]["best_epoch"], epochs=v["j"]["epochs_run"], val_auc=v["j"]["val"]["auc"]))
    df = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    order = list(df.key); best = order[0]

    # ---------------- DeLong vs best + full pairwise
    pv = np.ones((len(order), len(order)))
    for i, j in itertools.combinations(range(len(order)), 2):
        pv[i, j] = pv[j, i] = delong(y, R[order[i]]["p"], R[order[j]]["p"])[3]
    iu = np.triu_indices(len(order), 1); adj = np.ones_like(pv); adj[iu] = holm(pv[iu]); adj.T[iu] = adj[iu]
    df["p_vs_best"] = [1.0 if k == best else delong(y, R[best]["p"], R[k]["p"])[3] for k in order]
    df["p_vs_best_holm"] = holm(df.p_vs_best.values); df.loc[0, "p_vs_best_holm"] = 1.0

    # ---------------- composite ranking
    up = ["auc", "pr_auc", "bal_acc", "f1", "mcc", "sens_at_spec90", "spec_at_sens95"]; down = ["ece", "brier"]
    rk = pd.concat([df[c].rank(ascending=False) for c in up] + [df[c].rank() for c in down], axis=1)
    df["mean_rank"] = rk.mean(1)
    # statistical tier: tier 1 = not significantly worse than best AUC (Holm); else by greedy grouping
    tiers, t, anchor = [], 1, 0
    for i in range(len(order)):
        sig = df.p_vs_best_holm[i] < 0.05 if t == 1 else adj[anchor, i] < 0.05
        if i and sig:
            t += 1; anchor = i
        tiers.append(t)
    df["tier"] = tiers
    df = df.sort_values(["tier", "mean_rank"]).reset_index(drop=True); df.insert(0, "final_rank", range(1, len(df) + 1))
    df.to_csv(os.path.join(out, "ranking.csv"), index=False)
    order = list(df.key)
    col = {k: FAM_COL[fam[k]] for k in order}
    lab = {k: SHORT.get(k, k) for k in order}

    # ---------------- Fig: ROC + PR (all)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.3))
    cm = plt.cm.tab20(np.linspace(0, 1, len(order)))
    for c, k in zip(cm, order):
        f, t, _ = roc_curve(y, R[k]["p"]); ax[0].plot(f, t, lw=1, color=c, label=f"{lab[k]} ({roc_auc_score(y, R[k]['p']):.3f})")
        pr, rc, _ = precision_recall_curve(y, R[k]["p"]); ax[1].plot(rc, pr, lw=1, color=c)
    ax[0].plot([0, 1], [0, 1], "k:", lw=.6); ax[0].set(xlabel="1 - specificity", ylabel="Sensitivity", title="(a) ROC")
    ax[1].set(xlabel="Recall", ylabel="Precision", title="(b) Precision-recall", ylim=(.45, 1.01))
    ax[0].legend(loc="lower right", fontsize=5.2, frameon=False)
    a2 = ax[0].inset_axes([.42, .52, .3, .3]);
    for c, k in zip(cm, order):
        f, t, _ = roc_curve(y, R[k]["p"]); a2.plot(f, t, lw=.8, color=c)
    a2.set(xlim=(0, .2), ylim=(.7, 1)); a2.tick_params(labelsize=5)
    save(fig, out, "fig_roc_pr_all")

    # ---------------- Fig: forest plot AUC with CI + tier
    fig, ax = plt.subplots(figsize=(4.6, 4))
    d2 = df.sort_values("auc")
    ax.errorbar(d2.auc, range(len(d2)), xerr=[d2.auc - d2.auc_lo, d2.auc_hi - d2.auc], fmt="none", ecolor="gray", lw=1)
    ax.scatter(d2.auc, range(len(d2)), c=[col[k] for k in d2.key], s=22, zorder=3)
    for i, (_, r) in enumerate(d2.iterrows()):
        ax.text(r.auc_hi + .004, i, f"{r.auc:.3f}  T{r.tier}" + ("" if r.final_rank == 1 else ("  ns" if r.p_vs_best_holm >= .05 else "")), va="center", fontsize=6)
    ax.set_yticks(range(len(d2))); ax.set_yticklabels([lab[k] for k in d2.key])
    ax.axvline(df.auc.max(), ls=":", c="k", lw=.6); ax.set_xlabel("Test AUC-ROC (95% bootstrap CI)"); ax.set_xlim(right=.99)
    for f_, c_ in FAM_COL.items():
        ax.scatter([], [], c=c_, label=f_, s=15)
    ax.legend(loc="lower right", frameon=False, title="Fusion family", title_fontsize=6.5)
    save(fig, out, "fig_auc_forest")

    # ---------------- Fig: metric heatmap
    mets = ["auc", "pr_auc", "acc", "bal_acc", "f1", "mcc", "sens", "spec", "ppv", "npv", "sens_at_spec90", "spec_at_sens95", "ece", "brier"]
    names = ["AUC", "PR-AUC", "Acc", "BalAcc", "F1", "MCC", "Sens", "Spec", "PPV", "NPV", "Sens@Sp90", "Sp@Sens95", "ECE", "Brier"]
    M = df.set_index("key").loc[order, mets]
    Z = (M - M.min()) / (M.max() - M.min()); Z[["ece", "brier"]] = 1 - Z[["ece", "brier"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.2)); ax.imshow(Z.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M.values[i, j]:.3f}", ha="center", va="center", fontsize=5.4)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_yticks(range(len(order))); ax.set_yticklabels([f"{i+1}. {lab[k]}" for i, k in enumerate(order)])
    ax.set_title("Test-set metrics (column-normalised colour; ECE/Brier inverted)")
    save(fig, out, "fig_metric_heatmap")

    # ---------------- Fig: pairwise DeLong (Holm) heatmap
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    oi = [list(df.sort_values("auc", ascending=False).key).index(k) for k in order]
    orig = list(df.sort_values("auc", ascending=False).key)
    A = np.array([[adj[orig.index(a_), orig.index(b_)] for b_ in order] for a_ in order])
    D = np.array([[roc_auc_score(y, R[a_]["p"]) - roc_auc_score(y, R[b_]["p"]) for b_ in order] for a_ in order])
    im = ax.imshow(D, cmap="coolwarm", vmin=-.05, vmax=.05)
    for i in range(len(order)):
        for j in range(len(order)):
            if i != j:
                s = "***" if A[i, j] < .001 else "**" if A[i, j] < .01 else "*" if A[i, j] < .05 else ""
                ax.text(j, i, s, ha="center", va="center", fontsize=5)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([lab[k] for k in order], rotation=60, ha="right", fontsize=5.5)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([lab[k] for k in order], fontsize=5.5)
    plt.colorbar(im, ax=ax, fraction=.04, label="ΔAUC (row − column)")
    ax.set_title("Pairwise DeLong tests, Holm-adjusted (* p<.05, ** p<.01, *** p<.001)")
    save(fig, out, "fig_delong_pairwise")

    # ---------------- Fig: confusion matrices grid
    nc = 7; nr = int(np.ceil(len(order) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(7.2, 2.3 * nr))
    for ax, k in zip(axs.ravel(), order):
        c = confusion_matrix(y, R[k]["p"] >= R[k]["j"]["threshold"]); cn = c / c.sum(1, keepdims=True)
        ax.imshow(cn, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{c[i, j]}\n{cn[i, j]:.0%}", ha="center", va="center", fontsize=5.5, color="w" if cn[i, j] > .5 else "k")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xticklabels(["B", "M"], fontsize=6); ax.set_yticklabels(["B", "M"], fontsize=6)
        ax.set_title(lab[k], fontsize=5.6)
    for ax in axs.ravel()[len(order):]:
        ax.axis("off")
    save(fig, out, "fig_confusion_all")

    # ---------------- Fig: reliability diagrams + ECE
    fig, axs = plt.subplots(2, 7, figsize=(7.2, 2.6), sharex=True, sharey=True)
    e = np.linspace(0, 1, 11)
    for ax, k in zip(axs.ravel(), order):
        p = R[k]["p"]; idx = np.clip(np.digitize(p, e) - 1, 0, 9)
        conf = [p[idx == b].mean() if (idx == b).any() else np.nan for b in range(10)]
        acc = [y[idx == b].mean() if (idx == b).any() else np.nan for b in range(10)]
        ax.bar(e[:-1], acc, width=.1, align="edge", color=col[k], alpha=.7, edgecolor="k", lw=.3)
        ax.plot([0, 1], [0, 1], "k:", lw=.6); ax.set_title(f"{lab[k]}\nECE={ece(y, p):.3f}", fontsize=5)
        ax.tick_params(labelsize=5)
    fig.supxlabel("Predicted probability", fontsize=7); fig.supylabel("Observed melanoma rate", fontsize=7)
    save(fig, out, "fig_reliability_all")

    # ---------------- Fig: prediction agreement (kappa) + probability correlation, clustered
    P = np.stack([R[k]["p"] for k in order]); H = np.stack([R[k]["p"] >= R[k]["j"]["threshold"] for k in order])
    K = np.array([[cohen_kappa_score(H[i], H[j]) for j in range(len(order))] for i in range(len(order))])
    C = np.corrcoef(stats.rankdata(P, axis=1))
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.5))
    for a_, Mx, t_ in ((ax[0], K, "(a) Cohen's κ of hard predictions"), (ax[1], C, "(b) Spearman ρ of probabilities")):
        im = a_.imshow(Mx, cmap="viridis", vmin=Mx[np.triu_indices(len(order), 1)].min(), vmax=1)
        a_.set_xticks(range(len(order))); a_.set_xticklabels([lab[k] for k in order], rotation=70, ha="right", fontsize=5)
        a_.set_yticks(range(len(order))); a_.set_yticklabels([lab[k] for k in order], fontsize=5); a_.set_title(t_)
        plt.colorbar(im, ax=a_, fraction=.045)
    save(fig, out, "fig_agreement")

    # ---------------- Fig: per-sample difficulty
    correct = (H == y[None]).sum(0)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for c_, n_, cl in ((1, "melanoma", "#c0392b"), (0, "benign", "#2c7fb8")):
        ax[0].hist(correct[y == c_], bins=np.arange(len(order) + 2) - .5, alpha=.6, label=n_, color=cl)
    ax[0].set(xlabel=f"No. of models (of {len(order)}) correct", ylabel="Test images", title="(a) Per-image difficulty"); ax[0].legend(frameon=False)
    # oracle / majority / val-selected ensemble
    acc_ind = df.acc.values
    maj = ((H.mean(0) >= .5) == y).mean(); orc = (correct > 0).mean()
    ax[1].bar(range(len(order)), df.set_index("key").loc[order, "acc"], color=[col[k] for k in order])
    ax[1].axhline(maj, c="k", ls="--", lw=.8, label=f"majority vote {maj:.3f}")
    ax[1].axhline(orc, c="g", ls=":", lw=.8, label=f"oracle (any correct) {orc:.3f}")
    ax[1].set_xticks(range(len(order))); ax[1].set_xticklabels([lab[k] for k in order], rotation=70, ha="right", fontsize=5)
    ax[1].set(ylim=(.6, 1), ylabel="Accuracy", title="(b) Individual vs. collective"); ax[1].legend(frameon=False, fontsize=6)
    save(fig, out, "fig_difficulty")

    # ---------------- ensemble (greedy forward selection on VALIDATION, evaluated on test)
    chosen, best_v = [], 0
    cand = [k for k in order if k != "unet_crossattn"]
    while True:
        sc = {k: roc_auc_score(R[k]["vy"], np.mean([R[c]["vp"] for c in chosen + [k]], 0)) for k in cand if k not in chosen}
        if not sc: break
        k, v = max(sc.items(), key=lambda kv: kv[1])
        if v <= best_v + 1e-4: break
        chosen.append(k); best_v = v
    pe = np.mean([R[c]["p"] for c in chosen], 0)
    vpe = np.mean([R[c]["vp"] for c in chosen], 0)
    f_, t_, th = roc_curve(R[chosen[0]]["vy"], vpe); thr_e = th[np.argmax(t_ - f_)]
    ens = dict(members=chosen, val_auc=best_v, test_auc=roc_auc_score(y, pe), test_auc_ci=boot_ci(y, pe, roc_auc_score).tolist(),
               test_acc=float(((pe >= thr_e) == y).mean()), test_ece=float(ece(y, pe)),
               p_vs_best=float(delong(y, pe, R[best]["p"])[3]), majority_acc=float(maj), oracle_acc=float(orc),
               n_all_wrong=int((correct == 0).sum()), n_all_right=int((correct == len(order)).sum()))
    json.dump(ens, open(os.path.join(out, "ensemble.json"), "w"), indent=1)

    # ---------------- Fig: decision curve analysis
    fig, ax = plt.subplots(figsize=(3.5, 2.7)); pts = np.linspace(.05, .8, 60); prev = y.mean()
    for k in order[:5] + [order[-1]]:
        nb = [((R[k]["p"] >= t) & (y == 1)).mean() - ((R[k]["p"] >= t) & (y == 0)).mean() * t / (1 - t) for t in pts]
        ax.plot(pts, nb, lw=1, color=col[k], label=lab[k], ls="-" if k != order[-1] else ":")
    ax.plot(pts, [prev - (1 - prev) * t / (1 - t) for t in pts], "k--", lw=.6, label="treat all")
    ax.axhline(0, c="gray", lw=.6, label="treat none"); ax.set(ylim=(-.05, .55), xlabel="Threshold probability", ylabel="Net benefit", title="Decision-curve analysis")
    ax.legend(frameon=False, fontsize=5)
    save(fig, out, "fig_decision_curve")

    # ---------------- subgroup robustness
    te = pd.read_csv(glob.glob(os.path.join(a.data_root, "**", "test.csv"), recursive=True)[0])
    tr = pd.read_csv(glob.glob(os.path.join(a.data_root, "**", "train.csv"), recursive=True)[0])
    known = lambda d: set(d.patient_id.dropna().astype(str)) - {"Unknown"}   # 'Unknown' is a placeholder, not a patient
    overlap = len(known(te) & known(tr))
    def argcat(prefix):
        cs = [c for c in te.columns if c.startswith(prefix)]; return te[cs].values.argmax(1), [c[len(prefix):] for c in cs]
    groups = {}
    for pre, nm in (("sex_", "Sex"), ("age_group_", "Age"), ("anatom_site_general_", "Site"), ("dermoscopic_type_", "Dermoscope")):
        g, labs_ = argcat(pre)
        for gi, l in enumerate(labs_):
            msk = g == gi
            if msk.sum() >= 60 and 0 < y[msk].mean() < 1:
                groups[f"{nm}: {l}"] = msk
    unk = (te.patient_id.isna() | (te.patient_id.astype(str) == "Unknown")).values
    seen = te.patient_id.astype(str).isin(known(tr)).values & ~unk
    if seen.sum() >= 30:   # absent on the patient-disjoint split
        groups["Patient in train: yes"] = seen; groups["Patient in train: no"] = ~seen & ~unk
    groups["Patient ID unknown"] = unk
    vig = [v for v in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "test_224_*_vig.npy"))
           if len(np.load(v)) == len(y)]   # the cache that belongs to THIS test split
    if vig:
        vf = np.load(vig[0]).astype(bool)
        groups["Vignette: yes"] = vf; groups["Vignette: no"] = ~vf
    groups = {g: m for g, m in groups.items() if m.sum() >= 30 and 0 < y[m].mean() < 1}
    SG = pd.DataFrame({k: {g: roc_auc_score(y[m], R[k]["p"][m]) for g, m in groups.items()} for k in order})
    SG.insert(0, "n", [int(m.sum()) for m in groups.values()]); SG.insert(1, "mel_rate", [y[m].mean() for m in groups.values()])
    SG.to_csv(os.path.join(out, "subgroup_auc.csv"))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    V = SG[order].values; im = ax.imshow(V, cmap="magma", aspect="auto", vmin=max(.6, np.nanmin(V)), vmax=1)
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            ax.text(j, i, f"{V[i, j]:.2f}", ha="center", va="center", fontsize=4.8, color="w" if V[i, j] < .85 else "k")
    ax.set_xticks(range(len(order))); ax.set_xticklabels([lab[k] for k in order], rotation=60, ha="right", fontsize=5.5)
    ax.set_yticks(range(len(SG))); ax.set_yticklabels([f"{g} (n={n})" for g, n in zip(SG.index, SG.n)], fontsize=5.5)
    plt.colorbar(im, ax=ax, fraction=.03, label="AUC"); ax.set_title("Subgroup AUC on the common test set")
    save(fig, out, "fig_subgroup_auc")

    # ---------------- Fig: probability distributions
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    for i, k in enumerate(order):
        for c_, off, cl in ((0, -.18, "#2c7fb8"), (1, .18, "#c0392b")):
            vp = ax.violinplot(R[k]["p"][y == c_], positions=[i + off], widths=.34, showextrema=False)
            for b in vp["bodies"]:
                b.set_facecolor(cl); b.set_alpha(.55)
        ax.plot([i - .4, i + .4], [R[k]["j"]["threshold"]] * 2, "k-", lw=.7)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([lab[k] for k in order], rotation=60, ha="right", fontsize=5.5)
    ax.set_ylabel("Calibrated P(melanoma)"); ax.set_title("Score distributions (blue benign, red melanoma; bar = operating threshold)")
    save(fig, out, "fig_prob_violin")

    # ---------------- Fig: efficiency + learning curves
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for _, r in df.iterrows():
        ax[0].scatter(r.params_M, r.auc, s=8 + 2 * r.train_min if r.train_min == r.train_min else 20, c=col[r.key], alpha=.8, edgecolor="k", lw=.3)
        ax[0].annotate(lab[r.key], (r.params_M, r.auc), fontsize=4.6, xytext=(3, 2), textcoords="offset points")
    ax[0].set(xlabel="Parameters (M)", ylabel="Test AUC", title="(a) Accuracy vs. size (marker ∝ train time)")
    for k in order:
        h = R[k]["j"]["history"]; ax[1].plot([e_["epoch"] for e_ in h], [e_["val_auc"] for e_ in h], color=col[k], lw=.9, alpha=.85)
    ax[1].set(xlabel="Epoch", ylabel="Validation AUC (EMA)", title="(b) Learning curves", ylim=(.6, .96))
    save(fig, out, "fig_efficiency_learning")

    # ---------------- Fig: family boxplot
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    fams = [f_ for f_ in FAM_COL if f_ in set(fam.values())]
    ax.boxplot([df[df.family == f_].auc for f_ in fams], labels=fams, widths=.5)
    for i, f_ in enumerate(fams):
        v_ = df[df.family == f_].auc; ax.scatter(np.full(len(v_), i + 1) + np.random.uniform(-.08, .08, len(v_)), v_, c=FAM_COL[f_], s=14, zorder=3)
    ax.set_ylabel("Test AUC"); ax.tick_params(axis="x", labelsize=6); ax.set_title("AUC by fusion family")
    save(fig, out, "fig_family_box")

    # ---------------- tables
    def tex(d, cols, fmt, path, cap, lbl):
        lines = [r"\begin{table*}[!t]", r"\centering", r"\caption{" + cap + "}", r"\label{" + lbl + "}", r"\scriptsize",
                 r"\begin{tabular}{" + "l" * 2 + "c" * (len(cols) - 2) + "}", r"\toprule",
                 " & ".join(c[1] for c in cols) + r" \\", r"\midrule"]
        for _, r in d.iterrows():
            lines.append(" & ".join(fmt(c[0], r) for c in cols) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
        open(path, "w", encoding="utf-8").write("\n".join(lines))
    bestv = {c: (df[c].min() if c in ("ece", "brier") else df[c].max()) for c in mets}
    def fm(c, r):
        if c == "final_rank": return str(int(r[c]))
        if c == "model": return r[c].replace("&", r"\&")
        if c == "auc": s = f"{r.auc:.3f} [{r.auc_lo:.3f}, {r.auc_hi:.3f}]"; return r"\textbf{" + s + "}" if r.auc == bestv["auc"] else s
        if c == "tier": return str(int(r[c]))
        if c == "p_vs_best_holm": return "--" if r.final_rank == 1 or r.key == best else (f"{r[c]:.3f}" if r[c] >= .001 else "$<$.001")
        if c == "params_M": return f"{r[c]:.0f}"
        s = f"{r[c]:.3f}"; return r"\textbf{" + s + "}" if c in bestv and np.isclose(r[c], bestv[c]) else s
    tex(df, [("final_rank", "\\#"), ("model", "Model"), ("tier", "Tier"), ("auc", "AUC [95\\% CI]"), ("pr_auc", "PR-AUC"), ("bal_acc", "Bal.Acc"),
             ("f1", "F1"), ("mcc", "MCC"), ("sens", "Sens"), ("spec", "Spec"), ("sens_at_spec90", "Sens@Sp90"), ("ece", "ECE"), ("brier", "Brier"),
             ("p_vs_best_holm", "$p_{\\mathrm{Holm}}$"), ("params_M", "Params (M)")],
        fm, os.path.join(out, "table_ranking.tex"),
        "Leakage-controlled test-set performance of all 14 multimodal architectures on the identical held-out split ($n=2{,}559$). Models are ordered by statistical tier (DeLong, Holm-adjusted against the best AUC) and then by mean rank over nine metrics. Best value per column in bold.",
        "tab:ranking")
    SG.to_latex(os.path.join(out, "table_subgroup.tex"), float_format="%.3f")
    json.dump(dict(patient_overlap_train_test=overlap, n_test=int(N), ensemble=ens), open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(df[["final_rank", "model", "tier", "auc", "auc_lo", "auc_hi", "bal_acc", "f1", "mcc", "sens_at_spec90", "ece", "mean_rank", "p_vs_best_holm"]].round(4).to_string(index=False))
    print("ensemble", {k: (np.round(v, 4) if isinstance(v, float) else v) for k, v in ens.items()})
    print("patient overlap train/test:", overlap, "| worst subgroup per top model:", SG[order[0]].idxmin(), round(SG[order[0]].min(), 3))


if __name__ == "__main__":
    main()
