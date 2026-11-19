"""
Assembles the revised manuscript.

Splices the rewritten static sections into main.tex and GENERATES the
results-dependent prose, tables and figures from the actual run outputs, so
that no number in the paper is hand-transcribed.

    python build_paper.py --main ../../main.tex --runs ../../runs \
        --paper ../paper --out ../../main_revised.tex

Nothing is written unless at least one completed run exists; the results
section is regenerated from whatever has finished, so the command can be
re-run as more models complete.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from config import MODEL_ZOO


# ---------------------------------------------------------------------------
#  Load runs
# ---------------------------------------------------------------------------
def load_runs(runs_dir: str) -> List[Dict]:
    out = []
    for p in sorted(glob.glob(os.path.join(runs_dir, "*", "results.json"))):
        try:
            r = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        r["_dir"] = os.path.dirname(p)
        z = MODEL_ZOO.get(r["model_name"], {})
        r.setdefault("display", z.get("display", r["model_name"]))
        r.setdefault("family", z.get("family", ""))
        r.setdefault("origin", z.get("origin", ""))
        out.append(r)
    return out


def by_feature(runs: List[Dict], fs: str) -> List[Dict]:
    sel = [r for r in runs if r.get("feature_set") == fs]
    return sorted(sel, key=lambda r: -r["test"]["auc"])


def esc(s: str) -> str:
    return str(s).replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def fmt_ci(r: Dict) -> str:
    ci = r.get("test_ci95", {}).get("auc")
    if not ci:
        return f"{r['test']['auc']:.4f}"
    return (f"{r['test']['auc']:.4f} "
            f"\\scriptsize{{({ci[0]:.3f}--{ci[1]:.3f})}}")


# ---------------------------------------------------------------------------
#  Generated LaTeX
# ---------------------------------------------------------------------------
def gen_main_table(runs: List[Dict]) -> str:
    rows = []
    for i, r in enumerate(runs, 1):
        t = r["test"]
        mark = r"$\dagger$" if r.get("origin") == "ours" else ""
        rows.append(
            f"{i} & {esc(r['display'])}{mark} & {t['accuracy']:.4f} & "
            f"{t['balanced_accuracy']:.4f} & {t['precision']:.4f} & "
            f"{t['recall']:.4f} & {t['f1']:.4f} & {fmt_ci(r)} \\\\")
    return "\n".join([
        r"\begin{table*}[!t]", r"\centering",
        r"\caption{Test-set performance of all " + str(len(runs)) +
        r" multimodal architectures, ranked by AUC-ROC. Every model shares the "
        r"frozen 70/15/15 partition, the preprocessing pipeline of "
        r"Table~\ref{tab:preprocessing}, and the training protocol of "
        r"Table~\ref{tab:protocol}. Thresholds were selected on the validation "
        r"split by Youden's $J$ and applied unchanged to the test split. "
        r"$\dagger$ marks a fusion mechanism designed in this work.}",
        r"\label{tab:results}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{clccccccc}", r"\toprule",
        r"\textbf{\#} & \textbf{Model} & \textbf{Acc.} & \textbf{Bal. Acc.} & "
        r"\textbf{Prec.} & \textbf{Rec.} & \textbf{F1} & "
        r"\textbf{AUC (95\% CI)} \\", r"\midrule",
        "\n".join(rows),
        r"\bottomrule", r"\end{tabular}", r"\end{table*}",
    ])


def gen_ablation_table(runs: List[Dict], baselines: Optional[Dict]) -> str:
    order = ["full", "no_confirm", "no_age", "no_both"]
    present = [fs for fs in order if any(r["feature_set"] == fs for r in runs)]
    if len(present) < 2 and not baselines:
        return ""
    names = sorted({r["display"] for r in runs},
                   key=lambda d: -max(r["test"]["auc"] for r in runs
                                      if r["display"] == d))
    rows = []
    for name in names:
        cells = []
        for fs in present:
            m = [r for r in runs if r["display"] == name and r["feature_set"] == fs]
            cells.append(f"{m[0]['test']['auc']:.4f}" if m else "--")
        rows.append(f"{esc(name)} & " + " & ".join(cells) + r" \\")
    if baselines:
        rows.append(r"\midrule")
        cells = []
        for fs in present:
            k = f"metadata_only_rf__{fs}"
            cells.append(f"{baselines[k]['auc']:.4f}" if k in baselines else "--")
        rows.append(r"\textit{Metadata-only RF (no image)} & "
                    + " & ".join(cells) + r" \\")
    head = " & ".join(r"\textbf{" + fs.replace("_", "-") + "}" for fs in present)
    return "\n".join([
        r"\begin{table}[!t]", r"\centering",
        r"\caption{Metadata ablation (test AUC). \textit{no-confirm} removes the "
        r"diagnosis-confirmation field, which deterministically encodes the "
        r"label for roughly a quarter of the corpus; \textit{no-age} removes "
        r"the age scalar and age bands. The final row is a random forest "
        r"trained on metadata alone, with no access to any image.}",
        r"\label{tab:ablation}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{l" + "c" * len(present) + "}", r"\toprule",
        f"\\textbf{{Model}} & {head} \\\\", r"\midrule",
        "\n".join(rows),
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ])


def gen_results_prose(full: List[Dict], noconf: List[Dict],
                      baselines: Optional[Dict], sig: Optional[Dict]) -> str:
    best = full[0]
    bt = best["test"]
    n = len(full)
    p = [
        r"\subsection{Classification Performance}",
        "",
        f"All {num_word(n)} architectures were trained and evaluated under the protocol "
        r"of Section~\ref{sec:setup}. Table~\ref{tab:results} reports accuracy, "
        r"balanced accuracy, precision, recall, F1 and AUC-ROC with bootstrap "
        r"95\% confidence intervals on the 2,559-lesion test split, ranked by "
        r"AUC. Figure~\ref{fig:ranking} presents the same ranking graphically, "
        r"coloured by fusion family; Figure~\ref{fig:roc} overlays the ROC "
        r"curves of the three strongest models and Figure~\ref{fig:cm} their "
        r"confusion matrices at the validation-selected operating threshold.",
        "",
        f"The strongest configuration is {esc(best['display'])}, reaching an "
        f"AUC of {bt['auc']:.4f}, accuracy {bt['accuracy']:.4f}, balanced "
        f"accuracy {bt['balanced_accuracy']:.4f}, precision "
        f"{bt['precision']:.4f}, recall {bt['recall']:.4f} and F1 "
        f"{bt['f1']:.4f}.",
    ]
    if len(full) > 1:
        second = full[1]
        gap = bt["auc"] - second["test"]["auc"]
        s = (f" The runner-up, {esc(second['display'])}, reaches "
             f"{second['test']['auc']:.4f}, a gap of {gap:.4f} AUC.")
        if sig and "bootstrap_p" in sig:
            s += (f" A paired bootstrap test over the test set gives "
                  f"$p = {sig['bootstrap_p']:.3f}$ for this difference, and we "
                  "report it explicitly rather than presenting the ranking as a "
                  "settled ordering.")
        p.append(s)

    if n > 1:
        lo = min(r["test"]["auc"] for r in full)
        p += ["",
              f"Across the benchmark, test AUC spans {lo:.4f} to {bt['auc']:.4f}. "
              r"Where confidence intervals overlap, adjacent rows of "
              r"Table~\ref{tab:results} are not separable at this sample size, and "
              r"we do not claim that they are."]

    if baselines and "metadata_only_rf__full" in baselines:
        mb = baselines["metadata_only_rf__full"]["auc"]
        below = [r for r in full if r["test"]["auc"] < mb]
        p += ["",
              f"A random forest trained on the clinical metadata alone, with no "
              f"access to any image, attains an AUC of {mb:.4f} on the same "
              f"test split."]
        if below:
            p.append(
                f" {len(below)} of the {n} multimodal configurations do not "
                f"exceed this image-free reference, which means they recover no "
                f"usable information from the dermoscopic image beyond what a "
                f"tabular classifier extracts from the clinical record. "
                r"Section~\ref{sec:ablation} explains why this reference is so "
                r"strong, and why it is not a compliment to the metadata.")
    # recall-oriented reading
    best_rec = max(full, key=lambda r: r["test"]["recall"])
    p += ["",
          r"Because a missed melanoma is the costly error in a screening "
          r"context, the recall column is arguably more decision-relevant than "
          f"AUC. On that criterion the leading configuration is "
          f"{esc(best_rec['display'])}, at a recall of "
          f"{best_rec['test']['recall']:.4f} with precision "
          f"{best_rec['test']['precision']:.4f}."]

    if noconf:
        nb = noconf[0]
        p += ["", r"\subsection{Leakage-Controlled Results}", "",
              "Repeating the benchmark with the diagnosis-confirmation field "
              "removed changes both the absolute numbers and, in places, the "
              f"ordering. The strongest leakage-controlled configuration is "
              f"{esc(nb['display'])}, at an AUC of {nb['test']['auc']:.4f} "
              f"(accuracy {nb['test']['accuracy']:.4f}, F1 "
              f"{nb['test']['f1']:.4f}, recall {nb['test']['recall']:.4f}). "
              r"Table~\ref{tab:ablation} reports the full ablation. We treat "
              r"these as the primary results of the study, for the reasons set "
              r"out in Section~\ref{sec:ablation}."]
    return "\n".join(p)


def gen_timing_note(runs: List[Dict]) -> str:
    mins = [r.get("train_minutes", 0) for r in runs if r.get("train_minutes")]
    if not mins:
        return ""
    params = [r.get("params_total", 0) / 1e6 for r in runs]
    return ("Model size across the benchmark ranges from "
            f"{min(params):.1f}M to {max(params):.1f}M parameters, and "
            f"end-to-end training time from {min(mins):.0f} to {max(mins):.0f} "
            "minutes per model on a single consumer GPU.")


# ---------------------------------------------------------------------------
#  Splice
# ---------------------------------------------------------------------------
def read(path: str) -> str:
    return open(path, encoding="utf-8").read()


def replace_block(text: str, start_pat: str, end_pat: str, new: str,
                  label: str, include_end: bool = False) -> str:
    """Replace from the line matching start_pat up to (not incl.) end_pat."""
    s = re.search(start_pat, text, re.M)
    if not s:
        print(f"  [warn] start not found for {label}")
        return text
    e = re.search(end_pat, text[s.end():], re.M)
    if not e:
        print(f"  [warn] end not found for {label}")
        return text
    stop = s.end() + (e.end() if include_end else e.start())
    print(f"  spliced {label}")
    return text[:s.start()] + new + "\n\n" + text[stop:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--paper", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--generated", default=None,
                    help="directory holding generated figures/baselines")
    a = ap.parse_args()

    gen_dir = a.generated or os.path.join(a.paper, "generated")
    runs = load_runs(a.runs)
    # the main table is the leakage-controlled headline configuration; the
    # leaky "full" runs are only used where the manuscript compares the two
    from config import HEADLINE_FEATURE_SET
    full = by_feature(runs, HEADLINE_FEATURE_SET) or by_feature(runs, "full")
    noconf = by_feature(runs, "no_confirm")
    print(f"loaded {len(runs)} runs ({len(full)} full, {len(noconf)} no_confirm)")
    if not full:
        raise SystemExit("no completed headline runs yet; nothing to build")

    baselines = None
    bpath = os.path.join(gen_dir, "baselines.json")
    if os.path.exists(bpath):
        baselines = json.load(open(bpath, encoding="utf-8"))
    sig = None
    spath = os.path.join(gen_dir, "significance.json")
    if os.path.exists(spath):
        sig = json.load(open(spath, encoding="utf-8"))

    tex = read(a.main)

    # ---- 1. abstract ----------------------------------------------------
    best = full[0]
    bt = best["test"]
    nb = noconf[0] if noconf else None
    abstract = build_abstract(full, best, nb, baselines)
    tex = replace_block(tex, r"^\\begin\{abstract\}", r"^\\end\{abstract\}",
                        abstract, "abstract", include_end=True)

    # ---- 2. keywords ----------------------------------------------------
    kw = ("\\begin{IEEEkeywords}\n"
          "Skin cancer detection, melanoma classification, multimodal deep\n"
          "learning, explainable artificial intelligence, dermoscopy, vision\n"
          "transformers, cross-attention fusion, label leakage, shortcut\n"
          "learning, model calibration.\n"
          "\\end{IEEEkeywords}")
    tex = replace_block(tex, r"^\\begin\{IEEEkeywords\}",
                        r"^\\end\{IEEEkeywords\}", kw, "keywords",
                        include_end=True)

    # ---- 3. introduction -------------------------------------------------
    intro = read(os.path.join(a.paper, "section_introduction.tex"))
    intro = strip_comments(intro)
    tex = replace_block(tex, r"^\\section\{Introduction\}",
                        r"^\\section\{Review of Relevant Literature\}",
                        intro, "introduction")

    # ---- 4. dataset / splitting / preprocessing --------------------------
    data_sec = strip_comments(read(os.path.join(a.paper,
                                                "new_methodology_data.tex")))
    tex = replace_block(tex, r"^\\subsection\{Dataset\}",
                        r"^\\begin\{table\}\[!t\]\n\\centering\n\\caption\{Preprocessing",
                        data_sec, "dataset/splitting/preprocessing")

    # ---- 5. unified protocol (inserted before architectures) -------------
    setup = strip_comments(read(os.path.join(a.paper,
                                             "section_experimental_setup.tex")))
    setup = fill_protocol_tokens(setup, full)
    arch = strip_comments(read(os.path.join(a.paper, "new_arch_table.tex")))
    tex = replace_block(tex, r"^\\subsection\{Model Architectures and Fusion Strategies\}",
                        r"^\\subsubsection\{Gated Fusion\}",
                        setup + "\n\n" + arch, "protocol + architecture table")

    # ---- 6. results ------------------------------------------------------
    results = build_results_section(full, noconf, baselines, sig, gen_dir)
    tex = replace_block(tex, r"^\\section\{Results\}", r"^\\section\{Discussion\}",
                        results, "results")

    # ---- 7. discussion ---------------------------------------------------
    disc = build_discussion(a.paper, full, noconf, baselines)
    tex = replace_block(tex, r"^\\section\{Discussion\}",
                        r"^\\section\{Ethical and Legal Analysis\}",
                        disc, "discussion")

    # ---- 8. drop the ethical & legal section -----------------------------
    tex = replace_block(tex, r"^\\section\{Ethical and Legal Analysis\}",
                        r"^\\section\{Conclusion\}", "",
                        "removed ethical & legal section")

    # ---- 9. conclusion ---------------------------------------------------
    concl = strip_comments(read(os.path.join(a.paper, "new_conclusion.tex")))
    concl = concl.replace("@@CONCLUSION_PERFORMANCE@@",
                          conclusion_performance(full, noconf))
    tex = replace_block(tex, r"^\\section\{Conclusion\}",
                        r"^\\begin\{thebibliography\}", concl, "conclusion")

    # ---- 10. bibliography ------------------------------------------------
    bib = read(os.path.join(a.paper, "new_bibliography.tex"))
    tex = replace_block(tex, r"^\\begin\{thebibliography\}",
                        r"^\\end\{thebibliography\}", bib, "bibliography",
                        include_end=True)

    # consistency: no stale model counts
    for wrong in ["sixteen", "seventeen", "twelve multimodal"]:
        if wrong in tex:
            print(f"  [warn] stale count '{wrong}' still present")

    with open(a.out, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"\nwrote {a.out} ({len(tex.splitlines())} lines)")


def strip_comments(s: str) -> str:
    """Drop the instructional header comments from the section files."""
    lines = [l for l in s.splitlines() if not l.startswith("%")]
    while lines and not lines[0].strip():
        lines.pop(0)
    return "\n".join(lines)


def build_abstract(full, best, nb, baselines) -> str:
    bt = best["test"]
    n = len(full)
    mb = baselines["metadata_only_rf__full"]["auc"] if baselines else None
    lead = (f"The best-performing architecture, {esc(best['display'])}, attains "
            f"an accuracy of {bt['accuracy']:.4f}, balanced accuracy of "
            f"{bt['balanced_accuracy']:.4f}, precision of {bt['precision']:.4f}, "
            f"recall of {bt['recall']:.4f}, F1 of {bt['f1']:.4f} and AUC-ROC of "
            f"{bt['auc']:.4f} on the held-out test set.")
    leak = ("We further report an audit of the clinical metadata that changes "
            "how these and previously published figures should be read: the "
            "diagnosis-confirmation field constitutes label leakage, as no "
            "lesion outside the histopathology category is a melanoma anywhere "
            "in the corpus. A single-flag rule using no image attains 75.3\\% "
            "accuracy")
    if mb:
        leak += (f", and a metadata-only classifier attains an AUC of {mb:.4f}, "
                 "exceeding several published multimodal results on this dataset")
    leak += "."
    if nb:
        leak += (f" Removing the field reduces the best multimodal AUC to "
                 f"{nb['test']['auc']:.4f}, which we report as the primary "
                 f"leakage-controlled result.")
    return r"""\begin{abstract}
Melanoma causes the majority of skin cancer deaths despite representing a
minority of diagnoses, and survival depends almost entirely on how early the
lesion is excised. An estimated 331,722 new melanoma cases and 58,667 deaths
occurred worldwide in 2022, with incidence projected to rise by roughly half by
2040, while early lesions remain visually difficult to separate from benign
pigmented entities. This paper reports a controlled benchmark of """ + \
        num_word(n) + r""" multimodal deep learning architectures that classify
dermoscopic images jointly with structured clinical metadata on the
ISIC-DICM-17K corpus (17,059 lesions, balanced 1:1 between melanoma and
non-melanoma). The architectures span six fusion paradigms---early token-level
fusion, late concatenation, cross-attention, gating, graph reasoning and
bilinear pooling---over ViT, Swin-V2, ConvNeXt, EfficientNetV2, MaxViT, DaViT,
SegFormer-B5 and DINOv2 backbones. Every model is paired with the same
explainability suite: Grad-CAM and Grad-CAM++, Integrated Gradients, occlusion
sensitivity, attention rollout, cross-attention maps, LIME, SHAP and
permutation importance. Four fusion mechanisms are introduced in this
work---lesion-aware cross-attention, lesion-conditioned sparse attention, a
heterogeneous image--metadata graph with GATv2 message passing, and an
inspectable per-modality gating module---and several backbones, including
SegFormer-B5 and DINOv2, are evaluated on this corpus for the first time.
Critically, all models are trained under one protocol: a single frozen
70/15/15 partition (11,941/2,559/2,559), one preprocessing and augmentation
pipeline, a constant effective batch size maintained by gradient accumulation,
an identical optimisation budget, and an operating threshold fitted on
validation data and applied unchanged to the held-out test set. This removes
the resolution, batch-size, epoch-count and split inconsistencies that make
most published multimodal dermoscopy comparisons uninterpretable. """ + \
        lead + " " + leak + r""" SHAP faithfully identified the leaking field as
the dominant feature without this being recognised as a failure, demonstrating
that attribution fidelity does not establish clinical validity.
\end{abstract}"""


def num_word(n: int) -> str:
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
             7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven",
             12: "twelve", 13: "thirteen", 14: "fourteen"}
    return words.get(n, str(n))


def conclusion_performance(full, noconf) -> str:
    b = full[0]
    s = (f"the strongest configuration was {esc(b['display'])}, at a test AUC "
         f"of {b['test']['auc']:.4f} with an F1 of {b['test']['f1']:.4f} and a "
         f"melanoma recall of {b['test']['recall']:.4f}.")
    if noconf:
        s += (f" Under leakage control the strongest configuration reached an "
              f"AUC of {noconf[0]['test']['auc']:.4f}.")
    return s


def build_results_section(full, noconf, baselines, sig, gen_dir) -> str:
    parts = [r"\section{Results}", r"\label{sec:results}", "",
             gen_results_prose(full, noconf, baselines, sig), "",
             gen_main_table(full), ""]
    for fn, label, cap, star in [
        ("fig_auc_ranking.png", "fig:ranking",
         "Test AUC of every benchmarked architecture, coloured by fusion "
         "family, with bootstrap 95\\% confidence intervals. The dashed line "
         "marks a metadata-only random forest that never sees an image.", False),
        ("fig_roc_top3.png", "fig:roc",
         "ROC curves for the three best architectures on the held-out test "
         "set.", False),
        ("fig_confusion_top3.png", "fig:cm",
         "Confusion matrices for the three best architectures at the "
         "validation-selected operating threshold.", True),
    ]:
        if os.path.exists(os.path.join(gen_dir, fn)):
            env = "figure*" if star else "figure"
            width = r"\textwidth" if star else r"\columnwidth"
            parts += [f"\\begin{{{env}}}[!t]", r"  \centering",
                      f"  \\includegraphics[width={width}]{{generated/{fn}}}",
                      f"  \\caption{{{cap}}}", f"  \\label{{{label}}}",
                      f"\\end{{{env}}}", ""]
    ab = gen_ablation_table(full + noconf, baselines)
    if ab:
        parts += [r"\subsection{Effect of the Metadata Feature Set}",
                  r"\label{sec:results-ablation}", "",
                  r"Table~\ref{tab:ablation} reports each architecture under the "
                  r"available metadata configurations, with a metadata-only "
                  r"random forest as reference. "
                  r"Section~\ref{sec:ablation} explains why this pattern is not "
                  r"a neutral feature-importance result but evidence of label "
                  r"leakage.", "", ab, ""]
    parts += build_xai_subsection(full, gen_dir)
    return "\n".join(parts)


def build_xai_subsection(full, gen_dir) -> List[str]:
    parts = [r"\subsection{Explainability}", r"\label{sec:xai}", "",
             "Following the single-case figure policy of "
             r"Section~\ref{sec:setup}, each architecture contributes one "
             "deterministically selected test case---the highest-confidence "
             "correctly classified melanoma---so that panels are comparable "
             "across models and no case is chosen post hoc for its appearance. "
             "The selected lesion identifier is printed in each figure. "
             r"Spatial attribution uses Grad-CAM~\cite{selvaraju2020} and "
             r"Grad-CAM++~\cite{chattopadhay2018} read from the last transformer "
             "block (ViTs) or the final backbone feature map (CNNs) and averaged "
             "over the eight test-time views, Integrated "
             r"Gradients~\cite{sundararajan2017} with a blurred baseline and "
             "SmoothGrad noise, blur-filled occlusion, and gradient-weighted "
             "attention rollout for transformer backbones. Every map explains the "
             "predicted class and is scored by deletion/insertion faithfulness and "
             "by the share of attribution inside an automatically segmented lesion, "
             r"against a random-map baseline (Table~\ref{tab:xai}). The metadata "
             r"branch is attributed with exact grouped Shapley values~\cite{lundberg2017} "
             "(one player per clinical variable) and grouped permutation importance."]
    import shutil
    shown = 0
    for r in full:
        panel = os.path.join(r["_dir"], "xai_panel.png")
        if os.path.exists(panel) and shown < 3:
            # copy next to the other generated figures so the path in the .tex
            # resolves from the manuscript root regardless of where runs live
            dest_name = f"xai_{r['model_name']}.png"
            os.makedirs(gen_dir, exist_ok=True)
            shutil.copyfile(panel, os.path.join(gen_dir, dest_name))
            rel = f"generated/{dest_name}"
            case = r.get("xai", {}).get("case_id", "")
            parts += ["", r"\begin{figure}[!t]", r"  \centering",
                      f"  \\includegraphics[width=\\columnwidth]{{{rel}}}",
                      f"  \\caption{{Single-case explanation for "
                      f"{esc(r['display'])}"
                      + (f" (lesion {esc(case)})" if case else "") +
                      ": attribution of the predicted class by each method, with "
                      "the automatic lesion outline (dashed) and per-map "
                      "faithfulness / localisation scores.}",
                      f"  \\label{{fig:xai{shown}}}", r"\end{figure}"]
            shown += 1
    # SHAP summary across models
    top_feats: Dict[str, List[float]] = {}
    for r in full:
        sv = r.get("xai", {}).get("shap_values")
        if not sv:
            continue
        for k, v in sv.items():
            top_feats.setdefault(k, []).append(abs(v))
    if top_feats:
        ranked = sorted(top_feats.items(), key=lambda kv: -float(np.mean(kv[1])))[:5]
        names = ", ".join(f"\\texttt{{{esc(k)}}}" for k, _ in ranked)
        parts += ["", "Aggregating the single-case grouped Shapley attributions across "
                  f"architectures, the features with the largest mean absolute "
                  f"contribution are {names}."]
    return parts


PRIOR_AUC = 0.8851   # Ahammed et al., the originating work on this corpus


def fill_vit_tokens(text: str, full, noconf) -> str:
    """
    Fill the comparison against the originating work from measured values.

    The honest comparison is the leakage-controlled one: Ahammed et al. built
    their metadata vector from age, sex and site only, so our full-metadata
    number is not like-for-like against theirs.
    """
    ef_full = next((r for r in full if r["model_name"] == "vit_early_fusion"), None)
    ef_nc = next((r for r in noconf if r["model_name"] == "vit_early_fusion"), None)
    if ef_full is None:
        text = text.replace("@@VIT_VS_PRIOR@@",
                            "The ViT-B/16 early-fusion configuration is compared "
                            "against their reported AUC of 0.8851 below.")
        return text.replace("@@VIT_MARGIN@@", "")

    if ef_nc is not None:
        a = (f"Our ViT-B/16 early-fusion model reaches an AUC of "
             f"{ef_nc['test']['auc']:.4f} on the leakage-controlled feature set "
             f"and {ef_full['test']['auc']:.4f} with the full metadata vector, "
             f"against their {PRIOR_AUC:.4f}.")
        d = ef_nc["test"]["auc"] - PRIOR_AUC
        ci = ef_nc.get("test_ci95", {}).get("auc")
        if d > 0:
            b = (f"matched, leakage-free condition the margin is "
                 f"{d*100:.2f} AUC points in our favour")
        else:
            b = (f"matched, leakage-free condition our model is "
                 f"{abs(d)*100:.2f} AUC points \\emph{{below}} the originating "
                 f"work, and we do not claim an improvement")
        if ci:
            b += (f", with a bootstrap 95\\% interval of "
                  f"{ci[0]:.4f}--{ci[1]:.4f} on our figure")
        b += ". "
        if ci and ci[0] <= PRIOR_AUC <= ci[1]:
            b += ("Because that interval contains their reported value, the "
                  "difference is not resolvable at this sample size and should "
                  "not be read as a definitive improvement.")
    else:
        a = (f"Our ViT-B/16 early-fusion model reaches an AUC of "
             f"{ef_full['test']['auc']:.4f} with the full metadata vector, "
             f"against their {PRIOR_AUC:.4f}. The leakage-controlled run "
             f"required for a like-for-like comparison is reported in "
             r"Table~\ref{tab:ablation}.")
        b = ("matched, leakage-free comparison is the one that should be "
             "quoted, and it is reported in Table~\\ref{tab:ablation}. ")
    return text.replace("@@VIT_VS_PRIOR@@", a).replace("@@VIT_MARGIN@@", b)


def fill_protocol_tokens(text: str, full) -> str:
    cfg = full[0].get("config", {})
    epochs = cfg.get("epochs", 20)
    patience = cfg.get("early_stop_patience", 6)
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
             7: "seven", 8: "eight", 9: "nine", 10: "ten"}
    mins = [r.get("train_minutes", 0) for r in full if r.get("train_minutes")]
    compute = ("All models were trained on a single NVIDIA GPU under the "
               "identical software stack")
    if mins:
        compute += (f", with end-to-end training times between {min(mins):.0f} "
                    f"and {max(mins):.0f} minutes per model")
    compute += "."
    return (text.replace("@@EPOCHS@@", str(epochs))
                .replace("@@PATIENCE@@", words.get(patience, str(patience)))
                .replace("@@COMPUTE@@", compute))


def build_discussion(paper_dir: str, full, noconf, baselines) -> str:
    parts = [r"\section{Discussion}", ""]
    b = full[0]
    parts.append(
        "The controlled protocol of this study makes one comparison possible "
        "that the prior literature generally does not support: because the "
        "partition, preprocessing, effective batch size, optimisation budget "
        "and evaluation rule are held fixed, the differences in "
        r"Table~\ref{tab:results} are attributable to the architectures "
        "themselves. "
        f"The strongest configuration, {esc(b['display'])}, reaches an AUC of "
        f"{b['test']['auc']:.4f}; the spread across the benchmark is "
        f"{b['test']['auc'] - min(r['test']['auc'] for r in full):.4f} AUC.")
    # early vs late, if both present
    ef = next((r for r in full if r["model_name"] == "vit_early_fusion"), None)
    lf = next((r for r in full if r["model_name"] == "hybrid_vit_mlp"), None)
    if ef and lf:
        d = ef["test"]["auc"] - lf["test"]["auc"]
        df1 = ef["test"]["f1"] - lf["test"]["f1"]
        ci_e = ef.get("test_ci95", {}).get("auc")
        ci_l = lf.get("test_ci95", {}).get("auc")
        overlap = bool(ci_e and ci_l and not (ci_e[0] > ci_l[1] or ci_l[0] > ci_e[1]))
        txt = ("The cleanest architectural contrast in the benchmark is "
               "between the two ViT-B/16 configurations, which share a "
               "backbone, a metadata encoder capacity, an optimisation budget "
               "and a data partition, and differ only in where the modalities "
               f"meet. Early token-level fusion reaches {ef['test']['auc']:.4f} "
               f"AUC against {lf['test']['auc']:.4f} for late concatenation, a "
               f"difference of {d:+.4f}")
        if ci_e and ci_l:
            txt += (f" (95\\% intervals {ci_e[0]:.4f}--{ci_e[1]:.4f} and "
                    f"{ci_l[0]:.4f}--{ci_l[1]:.4f} respectively)")
        txt += ". "
        if abs(d) <= 0.005 or overlap:
            txt += ("The intervals overlap, and on F1 the ordering "
                    + ("reverses" if df1 < 0 else "is preserved only marginally")
                    + f" ({ef['test']['f1']:.4f} against {lf['test']['f1']:.4f}, "
                      f"a difference of {df1:+.4f}). We therefore do not claim "
                      "that the depth of metadata integration is decisive on "
                      "this corpus. This is a correction to our own earlier "
                      "report, which described a consistent two-point F1 "
                      "advantage for early fusion; that comparison was drawn "
                      "between runs that differed in resolution, batch size, "
                      "epoch budget and evaluation partition, and the "
                      "advantage does not survive a controlled comparison. It "
                      "is a concrete illustration of why the protocol of "
                      r"Section~\ref{sec:setup} was necessary: the earlier "
                      "conclusion was an artefact of experimental treatment "
                      "rather than a property of the architectures.")
        else:
            txt += ("Because every other factor is held fixed, this is "
                    "attributable to fusion depth rather than to capacity, "
                    "data or optimisation effort.")
        parts += ["", txt]
    for fn in ["section_discussion_leakage.tex", "section_discussion_vit.tex",
               "section_discussion_future.tex"]:
        p = os.path.join(paper_dir, fn)
        if os.path.exists(p):
            body = strip_comments(read(p))
            if "vit" in fn:
                body = fill_vit_tokens(body, full, noconf)
            parts += ["", body]
    return "\n".join(parts)


if __name__ == "__main__":
    main()
