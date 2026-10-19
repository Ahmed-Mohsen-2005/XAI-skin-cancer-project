"""
Aggregates every finished run into the paper's tables and figures.

    python report.py --out-root outputs --paper-dir ../paper/generated

Produces
    results_all.csv                 every run, every metric
    table_main.tex                  all models ranked by AUC (IEEEtran)
    table_ablation.tex              feature-drop ablation
    table_highlight.tex             which fusion designs are ours / novel here
    fig_roc_top3.png                ROC curves, three best by AUC
    fig_confusion_top3.png          confusion matrices, three best by AUC
    fig_auc_ranking.png             all models ranked, coloured by family
    baselines.json                  metadata-only and image-only reference points
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix

from config import MODEL_ZOO
from metrics import delong_like_pvalue

FAMILY_COLORS = {
    "Early fusion": "#1f77b4", "Late fusion": "#7f9fc4",
    "Cross-attention": "#d62728", "Sparse cross-attention": "#e377c2",
    "Gated": "#2ca02c", "Graph": "#9467bd", "Bilinear": "#ff7f0e",
}


def load_runs(out_root: str) -> List[Dict]:
    runs = []
    for path in sorted(glob.glob(os.path.join(out_root, "*", "results.json"))):
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        r["_dir"] = os.path.dirname(path)
        runs.append(r)
    return runs


def runs_to_frame(runs: List[Dict]) -> pd.DataFrame:
    rows = []
    for r in runs:
        t = r["test"]
        ci = r.get("test_ci95", {})
        zoo = MODEL_ZOO.get(r["model_name"], {})
        rows.append({
            "model": r["model_name"],
            "display": r.get("display", zoo.get("display", r["model_name"])),
            "family": r.get("family", zoo.get("family", "")),
            "origin": r.get("origin", zoo.get("origin", "")),
            "novel_on_dataset": r.get("novel_on_dataset", zoo.get("novel_on_dataset", False)),
            "feature_set": r["feature_set"],
            "accuracy": t["accuracy"], "balanced_accuracy": t["balanced_accuracy"],
            "precision": t["precision"], "recall": t["recall"], "f1": t["f1"],
            "f1_macro": t["f1_macro"], "specificity": t["specificity"],
            "auc": t["auc"], "pr_auc": t["pr_auc"], "ece": t["ece"],
            "auc_lo": ci.get("auc", (np.nan, np.nan))[0],
            "auc_hi": ci.get("auc", (np.nan, np.nan))[1],
            "threshold": t["threshold"], "temperature": r.get("temperature", 1.0),
            "params_m": r.get("params_total", 0) / 1e6,
            "epochs_run": r.get("epochs_run", 0),
            "best_epoch": r.get("best_epoch", 0),
            "train_minutes": r.get("train_minutes", 0),
            "backbone_loaded": r.get("backbone_loaded", ""),
            "tn": t["tn"], "fp": t["fp"], "fn": t["fn"], "tp": t["tp"],
            "_dir": r["_dir"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Reference baselines
# ---------------------------------------------------------------------------
def compute_baselines(data_root: str) -> Dict:
    """
    Metadata-only references. These bound how much of the headline score is
    attributable to clinical fields rather than to the dermoscopic image, and
    they are the reason the leakage ablation exists.
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data import load_split, build_feature_list
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from metrics import compute_metrics, choose_threshold

    tr, va, te, _, _ = load_split(data_root, "full")
    out: Dict[str, Dict] = {}
    for fs in ("full", "no_confirm", "no_age", "no_both"):
        cols = build_feature_list(tr, fs)
        for name, mk in (("rf", lambda: RandomForestClassifier(
                              n_estimators=400, random_state=42, n_jobs=-1,
                              class_weight="balanced")),
                         ("logreg", lambda: LogisticRegression(
                              max_iter=2000, class_weight="balanced"))):
            m = mk().fit(tr[cols], tr["class"])
            vp = m.predict_proba(va[cols])[:, 1]
            tp = m.predict_proba(te[cols])[:, 1]
            thr = choose_threshold(va["class"].to_numpy(), vp)
            out[f"metadata_only_{name}__{fs}"] = {
                "n_features": len(cols),
                **compute_metrics(te["class"].to_numpy(), tp, thr)}

    # the single-flag shortcut rule
    if "diagnosis_confirm_type_histopathology" in te.columns:
        rule = te["diagnosis_confirm_type_histopathology"].to_numpy().astype(float)
        out["shortcut_rule_histopathology"] = compute_metrics(
            te["class"].to_numpy(), rule, 0.5)
        nonhisto = te[te["diagnosis_confirm_type_histopathology"] == 0]
        out["shortcut_rule_histopathology"]["melanoma_among_non_histopathology"] = \
            int(nonhisto["class"].sum())
        out["shortcut_rule_histopathology"]["n_non_histopathology"] = int(len(nonhisto))
    return out


# ---------------------------------------------------------------------------
#  Figures
# ---------------------------------------------------------------------------
def fig_roc_top3(df: pd.DataFrame, out_path: str) -> List[str]:
    top = df.nlargest(3, "auc")
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    for _, row in top.iterrows():
        p = np.load(os.path.join(row["_dir"], "predictions.npz"))
        fpr, tpr, _ = roc_curve(p["test_y"], p["test_prob"])
        ax.plot(fpr, tpr, lw=2,
                label=f"{row['display']} (AUC {row['auc']:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC - three best architectures (held-out test set)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return top["display"].tolist()


def fig_confusion_top3(df: pd.DataFrame, out_path: str) -> None:
    top = df.nlargest(3, "auc")
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.0))
    for ax, (_, row) in zip(np.atleast_1d(axes), top.iterrows()):
        cm = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
        cmn = cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n({cmn[i, j]*100:.1f}%)",
                        ha="center", va="center", fontsize=10,
                        color="white" if cmn[i, j] > 0.55 else "black")
        ax.set_xticks([0, 1], ["pred. benign", "pred. melanoma"], fontsize=8)
        ax.set_yticks([0, 1], ["true benign", "true melanoma"], fontsize=8)
        ax.set_title(f"{row['display']}\nAUC {row['auc']:.4f} | F1 {row['f1']:.4f}",
                     fontsize=9)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    fig.suptitle("Confusion matrices - three best architectures", fontsize=11)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_auc_ranking(df: pd.DataFrame, out_path: str, baselines: Dict | None = None) -> None:
    d = df.sort_values("auc")
    fig, ax = plt.subplots(figsize=(8.4, 0.42 * len(d) + 1.8))
    colors = [FAMILY_COLORS.get(f, "#888888") for f in d["family"]]
    err = None
    if d["auc_lo"].notna().all():
        err = np.vstack([d["auc"] - d["auc_lo"], d["auc_hi"] - d["auc"]])
    ax.barh(range(len(d)), d["auc"], color=colors, xerr=err,
            error_kw=dict(lw=0.8, capsize=2, ecolor="#333333"))
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d["display"], fontsize=8)
    ax.set_xlabel("Test AUC-ROC (bars show bootstrap 95% CI)")
    lo = max(0.5, float(d["auc"].min()) - 0.05)
    ax.set_xlim(lo, 1.0)
    if baselines:
        key = "metadata_only_rf__full"
        if key in baselines:
            ax.axvline(baselines[key]["auc"], color="k", ls="--", lw=1.2)
            ax.text(baselines[key]["auc"], len(d) - 0.4,
                    f"  metadata-only RF ({baselines[key]['auc']:.3f})",
                    fontsize=7.5, va="top")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLORS.values()]
    ax.legend(handles, FAMILY_COLORS.keys(), fontsize=7, loc="lower right")
    ax.set_title("Architecture ranking by test AUC", fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
#  LaTeX tables
# ---------------------------------------------------------------------------
def _esc(s: str) -> str:
    return str(s).replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def table_main(df: pd.DataFrame, path: str) -> None:
    d = df.sort_values("auc", ascending=False).reset_index(drop=True)
    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Test-set performance of all %d multimodal architectures, "
        r"ranked by AUC-ROC. All models share one 70/15/15 split, one "
        r"preprocessing pipeline, and one training protocol "
        r"(Section~\ref{sec:setup}). Operating thresholds were selected on the "
        r"validation split by Youden's $J$ and applied unchanged to the test "
        r"split. $\dagger$ marks fusion mechanisms designed in this work."
        % len(d),
        r"\label{tab:main}}",
        r"\begin{tabular}{clccccccc}", r"\toprule",
        r"Rank & Model & Acc. & Bal.\,Acc. & Prec. & Rec. & F1 & AUC (95\% CI) & Fusion \\",
        r"\midrule",
    ]
    for i, r in d.iterrows():
        mark = r"$\dagger$" if r["origin"] == "ours" else ""
        ci = (f"{r['auc']:.4f} \\scriptsize{{({r['auc_lo']:.3f}--{r['auc_hi']:.3f})}}"
              if np.isfinite(r["auc_lo"]) else f"{r['auc']:.4f}")
        lines.append(
            f"{i+1} & {_esc(r['display'])}{mark} & {r['accuracy']:.4f} & "
            f"{r['balanced_accuracy']:.4f} & {r['precision']:.4f} & "
            f"{r['recall']:.4f} & {r['f1']:.4f} & {ci} & {_esc(r['family'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def table_ablation(df: pd.DataFrame, path: str, baselines: Dict | None = None) -> None:
    piv = df.pivot_table(index="display", columns="feature_set", values="auc")
    order = [c for c in ("full", "no_confirm", "no_age", "no_both") if c in piv.columns]
    piv = piv[order].sort_values(order[0], ascending=False)
    head = " & ".join(_esc(c.replace("_", "-")) for c in order)
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Metadata ablation (test AUC). \textit{no-confirm} removes the "
        r"diagnosis-confirmation field, which deterministically encodes the "
        r"label for roughly a quarter of the corpus; \textit{no-age} removes "
        r"the age scalar and age bands.}", r"\label{tab:ablation}",
        r"\begin{tabular}{l" + "c" * len(order) + "}", r"\toprule",
        f"Model & {head} \\\\", r"\midrule",
    ]
    for name, row in piv.iterrows():
        vals = " & ".join("--" if pd.isna(v) else f"{v:.4f}" for v in row)
        lines.append(f"{_esc(name)} & {vals} \\\\")
    if baselines:
        lines.append(r"\midrule")
        vals = []
        for c in order:
            k = f"metadata_only_rf__{c}"
            vals.append(f"{baselines[k]['auc']:.4f}" if k in baselines else "--")
        lines.append("Metadata-only RF (no image) & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def table_highlight(df: pd.DataFrame, path: str) -> None:
    d = df[df["feature_set"] == "full"].sort_values("auc", ascending=False)
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Provenance of each configuration. \emph{Designed here} marks "
        r"fusion mechanisms introduced by this work; \emph{new on this corpus} "
        r"marks backbones not previously evaluated on ISIC-DICM-17K.}",
        r"\label{tab:provenance}",
        r"\begin{tabular}{lccl}", r"\toprule",
        r"Model & Designed here & New on this corpus & Mechanism \\", r"\midrule",
    ]
    for _, r in d.iterrows():
        note = MODEL_ZOO.get(r["model"], {}).get("note", "")
        lines.append(
            f"{_esc(r['display'])} & {'yes' if r['origin']=='ours' else 'no'} & "
            f"{'yes' if r['novel_on_dataset'] else 'no'} & {_esc(note)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="outputs")
    ap.add_argument("--paper-dir", default="../paper/generated")
    ap.add_argument("--data-root", default=None,
                    help="if given, also fit the metadata-only baselines")
    a = ap.parse_args()
    os.makedirs(a.paper_dir, exist_ok=True)

    runs = load_runs(a.out_root)
    if not runs:
        raise SystemExit(f"no results.json found under {a.out_root}")
    df = runs_to_frame(runs)
    df.to_csv(os.path.join(a.paper_dir, "results_all.csv"), index=False)
    print(f"loaded {len(df)} runs ({df['model'].nunique()} distinct models)")

    baselines = None
    if a.data_root:
        baselines = compute_baselines(a.data_root)
        with open(os.path.join(a.paper_dir, "baselines.json"), "w", encoding="utf-8") as f:
            json.dump(baselines, f, indent=2)
        k = "metadata_only_rf__full"
        print(f"metadata-only RF (no image): AUC {baselines[k]['auc']:.4f}")

    main_df = df[df["feature_set"] == "full"].copy()
    if main_df.empty:
        main_df = df.copy()

    table_main(main_df, os.path.join(a.paper_dir, "table_main.tex"))
    table_highlight(df, os.path.join(a.paper_dir, "table_highlight.tex"))
    if df["feature_set"].nunique() > 1:
        table_ablation(df, os.path.join(a.paper_dir, "table_ablation.tex"), baselines)

    top = fig_roc_top3(main_df, os.path.join(a.paper_dir, "fig_roc_top3.png"))
    fig_confusion_top3(main_df, os.path.join(a.paper_dir, "fig_confusion_top3.png"))
    fig_auc_ranking(main_df, os.path.join(a.paper_dir, "fig_auc_ranking.png"), baselines)
    print("top-3 by AUC:", ", ".join(top))

    # is the winner genuinely ahead of the runner-up?
    best2 = main_df.nlargest(2, "auc")
    if len(best2) == 2:
        pa = np.load(os.path.join(best2.iloc[0]["_dir"], "predictions.npz"))
        pb = np.load(os.path.join(best2.iloc[1]["_dir"], "predictions.npz"))
        p = delong_like_pvalue(pa["test_y"], pa["test_prob"], pb["test_prob"])
        print(f"AUC difference {best2.iloc[0]['display']} vs "
              f"{best2.iloc[1]['display']}: bootstrap p = {p:.4f}")
        with open(os.path.join(a.paper_dir, "significance.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"model_a": best2.iloc[0]["display"],
                       "model_b": best2.iloc[1]["display"],
                       "auc_a": float(best2.iloc[0]["auc"]),
                       "auc_b": float(best2.iloc[1]["auc"]),
                       "bootstrap_p": p}, f, indent=2)
    print(f"wrote tables and figures to {a.paper_dir}")


if __name__ == "__main__":
    main()
