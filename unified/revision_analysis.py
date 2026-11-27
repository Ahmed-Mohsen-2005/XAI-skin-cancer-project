"""Analysis of the revision runs (run on ONE machine after everyone's outputs_revision is merged).

    python revision_analysis.py                  # tables + figures + full XAI on the patient split
    python revision_analysis.py --skip-xai       # statistics only (fast)

Put every teammate's outputs_revision/* folders into unified/outputs_revision/ first.
Writes into outputs_revision/comparison/:
  (1) everything compare_all.py makes, for the 14 models on the PATIENT-DISJOINT split
  (2) split_shift.csv / table_split_shift.tex / fig_split_shift     original (leaky) vs patient-disjoint AUC
  (3) seed_stability.csv / table_seeds.tex / fig_seed_stability     Tier-1 mean +- std over seeds 42, 1, 2
  (4) ablation.csv / table_ablation.tex / fig_ablation              full vs image-only vs no-melanocytic
and the unified XAI (common_xai -> plot_common_xai -> plot_xai_cases) into outputs_revision/common_xai.
"""
import argparse, glob, json, os, subprocess, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from compare_all import SHORT, save, delong  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.size": 8})
REV = os.path.join(HERE, "outputs_revision")
OLD = [os.path.join(HERE, "outputs"), os.path.join(HERE, "xai_results of jana, malak and abdelrahman", "xai_results")]
PATIENT = os.path.normpath(os.path.join(HERE, "..", "..", "data_patient"))
TIER1 = ["vit_early_fusion", "hybrid_vit_mlp", "gnn_maxvit", "davit_gated", "segformer_laca"]


def runs(roots):
    R = {}
    for r in roots:
        for f in glob.glob(os.path.join(r, "**", "results.json"), recursive=True):
            d = os.path.dirname(f)
            if "common_xai" in d or "comparison" in d or not os.path.exists(os.path.join(d, "predictions.npz")):
                continue
            z = np.load(os.path.join(d, "predictions.npz"))
            j = json.load(open(f, encoding="utf-8"))
            R[os.path.basename(d)] = dict(m=j["model_name"], p=z["test_prob"].astype(float),
                                          y=z["test_y"].astype(int), t=j["test"])
    return R


def paired_delta(y, p1, p2, B=2000, seed=0):
    """AUC(p1) - AUC(p2) with a paired bootstrap 95% CI and the DeLong p-value."""
    rng = np.random.default_rng(seed); n = len(y); v = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if y[i].min() != y[i].max():
            v.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p2[i]))
    a1, a2, _, p = delong(y, p1, p2)
    return a1 - a2, *np.percentile(v, [2.5, 97.5]), p


def ci(y, p, B=1000):
    rng = np.random.default_rng(0); n = len(y); v = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if y[i].min() != y[i].max(): v.append(roc_auc_score(y[i], p[i]))
    return np.percentile(v, [2.5, 97.5])


def tex(df, path, cap, label):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\\centering\\footnotesize\n\\caption{%s}\\label{%s}\n" % (cap, label))
        f.write("\\begin{tabular}{l" + "c" * (df.shape[1] - 1) + "}\\toprule\n")
        f.write(" & ".join(df.columns) + "\\\\\\midrule\n")
        for _, r in df.iterrows():
            f.write(" & ".join(str(x).replace("&", "\\&") for x in r) + "\\\\\n")
        f.write("\\bottomrule\\end{tabular}\n\\end{table}\n")


def sh(m):
    return SHORT.get(m, m)


def main():
    global REV
    ap = argparse.ArgumentParser(); ap.add_argument("--skip-xai", action="store_true")
    ap.add_argument("--rev", default=REV, help="merged outputs_revision folder"); a = ap.parse_args(); REV = a.rev
    out = os.path.join(REV, "comparison"); os.makedirs(out, exist_ok=True)
    py = [sys.executable, "-u"]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")

    # (1) full comparison of the 14 models on the patient-disjoint split
    subprocess.call(py + ["compare_all.py", "--data-root", PATIENT, "--runs", REV, "--out", out,
                          "--match", r".*__no_confirm__patient"], cwd=HERE, env=env)
    rp = os.path.join(out, "ranking.csv")
    rank_p = pd.read_csv(rp) if os.path.exists(rp) else None

    R, O = runs([REV]), runs(OLD)
    old = {v["m"]: v for k, v in O.items() if k.endswith("__no_confirm")}
    new = {v["m"]: v for k, v in R.items() if k.endswith("__no_confirm__patient")}

    # (2) original split (patients shared) vs patient-disjoint split
    rows = []
    for m in sorted(set(old) & set(new), key=lambda m: -new[m]["t"]["auc"]):
        o, n = old[m], new[m]
        lo, hi = ci(n["y"], n["p"])
        rows.append(dict(key=m, Model=sh(m), auc_orig=roc_auc_score(o["y"], o["p"]),
                         auc_patient=roc_auc_score(n["y"], n["p"]), lo=lo, hi=hi,
                         balacc_orig=o["t"]["balanced_accuracy"], balacc_patient=n["t"]["balanced_accuracy"]))
    S, rho = pd.DataFrame(rows), None
    if len(S) >= 3:
        S["delta"] = S.auc_patient - S.auc_orig
        S["rank_orig"] = S.auc_orig.rank(ascending=False).astype(int)
        S["rank_patient"] = S.auc_patient.rank(ascending=False).astype(int)
        rho = stats.spearmanr(S.auc_orig, S.auc_patient).correlation
        S.round(4).to_csv(os.path.join(out, "split_shift.csv"), index=False)
        T = pd.DataFrame({"Model": S.Model, "Orig. split": S.auc_orig.map("{:.3f}".format),
                          "Patient-disjoint [95\\% CI]": [f"{r.auc_patient:.3f} [{r.lo:.3f}--{r.hi:.3f}]" for r in S.itertuples()],
                          "$\\Delta$AUC": S.delta.map("{:+.3f}".format),
                          "Rank (orig.$\\to$pat.)": [f"{r.rank_orig}$\\to${r.rank_patient}" for r in S.itertuples()]})
        tex(T, os.path.join(out, "table_split_shift.tex"),
            f"Test AUC on the original split vs.\\ the patient-disjoint split (Spearman $\\rho$={rho:.2f}).",
            "tab:split_shift")
        fig, ax = plt.subplots(figsize=(3.5, 3.6))
        for r in S.itertuples():
            c = "#c0392b" if r.delta < S.delta.median() else "#2c3e50"
            ax.plot([0, 1], [r.auc_orig, r.auc_patient], "-o", color=c, ms=3, lw=1)
            ax.text(1.04, r.auc_patient, r.Model, fontsize=5.5, va="center")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Original split\n(patients shared)", "Patient-disjoint\nsplit"])
        ax.set_xlim(-.15, 1.9); ax.set_ylabel("Test AUC"); ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"Performance shift (Spearman $\\rho$={rho:.2f})", fontsize=8)
        save(fig, out, "fig_split_shift")
        print(f"[split shift] median dAUC {S.delta.median():+.4f}, rho {rho:.3f}")

    # (3) seed stability of the Tier-1 models
    rows = []
    for m in TIER1:
        got = [R[k] for k in (f"{m}__no_confirm__patient", f"{m}__no_confirm__patient_s1",
                              f"{m}__no_confirm__patient_s2") if k in R]
        if not got: continue
        auc = [roc_auc_score(g["y"], g["p"]) for g in got]; ba = [g["t"]["balanced_accuracy"] for g in got]
        sd = (lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
        rows.append(dict(key=m, Model=sh(m), n_seeds=len(got), auc_mean=np.mean(auc), auc_std=sd(auc),
                         balacc_mean=np.mean(ba), balacc_std=sd(ba), f1_mean=np.mean([g["t"]["f1"] for g in got]),
                         auc_seed_ensemble=roc_auc_score(got[0]["y"], np.mean([g["p"] for g in got], 0)), aucs=auc))
    D = pd.DataFrame(rows)
    if len(D):
        D = D.sort_values("auc_mean", ascending=False)
        D.drop(columns="aucs").round(4).to_csv(os.path.join(out, "seed_stability.csv"), index=False)
        tex(pd.DataFrame({"Model": D.Model, "Seeds": D.n_seeds,
                          "AUC (mean$\\pm$sd)": [f"{r.auc_mean:.3f}$\\pm${r.auc_std:.3f}" for r in D.itertuples()],
                          "Bal.Acc (mean$\\pm$sd)": [f"{r.balacc_mean:.3f}$\\pm${r.balacc_std:.3f}" for r in D.itertuples()],
                          "Seed-ensemble AUC": D.auc_seed_ensemble.map("{:.3f}".format)}),
            os.path.join(out, "table_seeds.tex"), "Tier-1 models over three training seeds (patient-disjoint split).", "tab:seeds")
        fig, ax = plt.subplots(figsize=(3.5, 2.4))
        for i, r in enumerate(D.itertuples()):
            ax.scatter([i] * len(r.aucs), r.aucs, s=12, color="#7f8c8d", zorder=3)
            ax.errorbar(i, r.auc_mean, yerr=r.auc_std, fmt="D", color="#c0392b", ms=4, capsize=3, zorder=4)
        ax.set_xticks(range(len(D))); ax.set_xticklabels(D.Model, rotation=25, ha="right", fontsize=6)
        ax.set_ylabel("Test AUC"); ax.set_title("Seed stability (grey: seeds; red: mean $\\pm$ sd)", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        save(fig, out, "fig_seed_stability")
        print(D.drop(columns="aucs").round(4).to_string(index=False))

    # (4) metadata ablations, paired on the same test images
    rows = []
    for m in ("vit_early_fusion", "gnn_maxvit", "davit_gated"):
        base = R.get(f"{m}__no_confirm__patient")
        if base is None: continue
        for fs, lab in (("image_only", "Image only"), ("no_melanocytic", "No melanocytic")):
            v = R.get(f"{m}__{fs}__patient")
            if v is None: continue
            d, lo, hi, p = paired_delta(base["y"], v["p"], base["p"])
            rows.append(dict(key=m, Model=sh(m), variant=lab, auc_full=roc_auc_score(base["y"], base["p"]),
                             auc_variant=roc_auc_score(v["y"], v["p"]), delta=d, lo=lo, hi=hi, delong_p=p,
                             balacc_variant=v["t"]["balanced_accuracy"]))
    A = pd.DataFrame(rows)
    if len(A):
        A.round(4).to_csv(os.path.join(out, "ablation.csv"), index=False)
        tex(pd.DataFrame({"Model": A.Model, "Variant": A.variant, "AUC full": A.auc_full.map("{:.3f}".format),
                          "AUC variant": A.auc_variant.map("{:.3f}".format),
                          "$\\Delta$AUC [95\\% CI]": [f"{r.delta:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]" for r in A.itertuples()],
                          "DeLong $p$": A.delong_p.map(lambda p: "$<$0.001" if p < 1e-3 else f"{p:.3f}")}),
            os.path.join(out, "table_ablation.tex"),
            "Metadata ablations on the patient-disjoint split ($\\Delta$ = variant $-$ full multimodal model; paired bootstrap CI).",
            "tab:ablation")
        fig, ax = plt.subplots(figsize=(3.5, 2.2))
        ax.barh(range(len(A)), A.delta, xerr=[A.delta - A.lo, A.hi - A.delta], capsize=2,
                color=["#2980b9" if v == "Image only" else "#e67e22" for v in A.variant])
        ax.axvline(0, color="k", lw=.7); ax.set_yticks(range(len(A)))
        ax.set_yticklabels([f"{r.Model}\n{r.variant}" for r in A.itertuples()], fontsize=5.5)
        ax.invert_yaxis(); ax.set_xlabel("$\\Delta$ test AUC vs. full multimodal model")
        ax.spines[["top", "right"]].set_visible(False)
        save(fig, out, "fig_ablation")
        print(A.round(4).to_string(index=False))

    json.dump(dict(n_patient_runs=len(new), spearman_orig_vs_patient=rho,
                   median_delta_auc=float(S.delta.median()) if "delta" in S else None,
                   patient_top5=rank_p[["key", "auc", "tier"]].head(5).to_dict("records") if rank_p is not None else None),
              open(os.path.join(out, "revision_summary.json"), "w"), indent=1, default=float)

    # (5) unified XAI on the patient-disjoint models (same cases for every model)
    if not a.skip_xai:
        X = os.path.join(REV, "common_xai")
        for cmd in (["common_xai.py", "--data-root", PATIENT, "--runs", REV, "--out", X, "--match", r".*__no_confirm__patient"],
                    ["plot_common_xai.py", "--xai", X, "--cmp", out],
                    ["plot_xai_cases.py", "--xai", X, "--cmp", out]):
            print(">>", " ".join(cmd), flush=True)
            if subprocess.call(py + cmd, cwd=HERE, env=env) != 0:
                print("   failed; it is resumable, just re-run", flush=True); break
    print("revision analysis done ->", out)


if __name__ == "__main__":
    main()
