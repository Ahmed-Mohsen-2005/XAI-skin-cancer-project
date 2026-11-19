"""
Emits one self-contained Kaggle notebook per architecture.

Each notebook embeds the unified package verbatim (single source of truth:
these files), so running it on Kaggle reproduces exactly what runs locally.

    python make_kaggle_notebooks.py --out ../kaggle_notebooks

On Kaggle:
  1. New Notebook -> File -> Import Notebook -> upload one of the generated .ipynb
  2. Add data: the splitted-data dataset (train/val/test zips or folders)
  3. Settings -> Accelerator: GPU T4 x2 (or P100), Internet: ON (for weights)
  4. Run All. Download outputs/<model>__full/ when finished.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List

from config import MODEL_ZOO

PKG_FILES = ["config.py", "data.py", "metrics.py", "models.py",
             "engine.py", "xai.py", "tune.py", "report.py", "run.py"]

INSTALL = """\
# Dependencies. Kaggle images already ship torch/timm/transformers; these pins
# only add what may be missing. Keep quiet to avoid flooding the log.
!pip install -q "timm>=1.0.9" "transformers>=4.40" shap lime optuna 2>/dev/null
import torch, timm, sys
print("torch", torch.__version__, "| CUDA", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("timm", timm.__version__)
"""

LOCATE = """\
# Locate the frozen 70/15/15 split that every model in this study shares.
import os, glob
CANDIDATES = glob.glob("/kaggle/input/*") + glob.glob("/kaggle/input/*/*")
print("attached inputs:")
for c in sorted(set(CANDIDATES))[:40]:
    print("  ", c)

DATA_ROOT = "/kaggle/input"
print("\\nDATA_ROOT =", DATA_ROOT)
"""

TRAIN_TMPL = """\
# ---------------------------------------------------------------------------
#  {display}
#  {note}
# ---------------------------------------------------------------------------
import sys, os
sys.path.insert(0, "/kaggle/working/unified")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL       = "{model}"
FEATURE_SET = "no_confirm"  # headline (leakage-free); "full" / "no_age" / "no_both" for ablations
EPOCHS      = {epochs}

import argparse
from run import run_one

args = argparse.Namespace(
    data_root="/kaggle/input", out_root="/kaggle/working/outputs",
    epochs=EPOCHS, micro_batch=0, workers=2, no_xai=False, tune=0,
    limit=0, xai_eval_cases=None)

result = run_one(MODEL, FEATURE_SET, args)
"""

SHOW = """\
# Final metric block -- exactly the metrics the paper reports.
import json, glob, os
for p in sorted(glob.glob("/kaggle/working/outputs/*/results.json")):
    r = json.load(open(p))
    t = r["test"]
    print(f"\\n{r.get('display', r['model_name'])}  [{r['feature_set']}]")
    print(f"  backbone loaded : {r.get('backbone_loaded','n/a')}")
    print(f"  best epoch      : {r['best_epoch']} / {r['epochs_run']}  "
          f"({r.get('train_minutes','?')} min)")
    print(f"  Accuracy        : {t['accuracy']:.4f}")
    print(f"  Balanced Acc.   : {t['balanced_accuracy']:.4f}")
    print(f"  Precision       : {t['precision']:.4f}")
    print(f"  Recall          : {t['recall']:.4f}")
    print(f"  F1              : {t['f1']:.4f}")
    print(f"  AUC-ROC         : {t['auc']:.4f}   PR-AUC {t['pr_auc']:.4f}")
    print(f"  Specificity     : {t['specificity']:.4f}   ECE {t['ece']:.4f}")
    print(f"  confusion       : TN={t['tn']} FP={t['fp']} FN={t['fn']} TP={t['tp']}")
"""

SHOW_XAI = """\
# Single-case explanation figures (one case per figure, as per the paper).
from IPython.display import Image, display
import glob
for p in sorted(glob.glob("/kaggle/working/outputs/*/*.png")):
    print(p)
    display(Image(filename=p))
"""


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


def package_cell(pkg_dir: str) -> dict:
    """One cell that materialises the unified package inside the Kaggle session."""
    parts: List[str] = [
        "# The unified framework, written out verbatim so this notebook is",
        "# self-contained and reproduces the local runs exactly.",
        "import os, pathlib",
        'os.makedirs("/kaggle/working/unified", exist_ok=True)',
        "FILES = {}",
        "",
    ]
    for fname in PKG_FILES:
        path = os.path.join(pkg_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            src = f.read()
        parts.append(f"FILES[{fname!r}] = r'''{src}'''")
        parts.append("")
    parts += [
        "for name, src in FILES.items():",
        '    p = pathlib.Path("/kaggle/working/unified") / name',
        '    p.write_text(src, encoding="utf-8")',
        'print("wrote", len(FILES), "modules to /kaggle/working/unified")',
    ]
    return code("\n".join(parts))


def build_notebook(model: str, pkg_dir: str, epochs: int) -> dict:
    z = MODEL_ZOO[model]
    cells = [
        md(f"# {z['display']}\n\n"
           f"**Fusion family:** {z['family']}  \n"
           f"**Mechanism:** {z['note']}  \n"
           f"**Designed in this work:** {'yes' if z['origin'] == 'ours' else 'no'}  \n"
           f"**New on ISIC-DICM-17K:** {'yes' if z['novel_on_dataset'] else 'no'}\n\n"
           "Every hyper-parameter comes from the shared `config.py`. The only "
           "per-model setting is the micro-batch size, which is compensated by "
           "gradient accumulation so the effective batch is 32 for all models.\n\n"
           "**Before running:** attach the splitted-data dataset, set the "
           "accelerator to GPU, and turn Internet on so pretrained weights can "
           "download."),
        md("## 1. Environment"),
        code(INSTALL),
        md("## 2. Unified framework"),
        package_cell(pkg_dir),
        md("## 3. Locate the frozen split"),
        code(LOCATE),
        md("## 4. Train and evaluate"),
        code(TRAIN_TMPL.format(model=model, display=z["display"],
                               note=z["note"], epochs=epochs)),
        md("## 5. Results"),
        code(SHOW),
        md("## 6. Single-case explanations"),
        code(SHOW_XAI),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.13"},
            "accelerator": "GPU",
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../kaggle_notebooks")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--models", nargs="*", default=list(MODEL_ZOO))
    a = ap.parse_args()

    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(a.out, exist_ok=True)

    order = sorted(a.models, key=lambda m: MODEL_ZOO[m]["micro_batch"], reverse=True)
    for i, model in enumerate(order, 1):
        nb = build_notebook(model, pkg_dir, a.epochs)
        path = os.path.join(a.out, f"{i:02d}_{model}.ipynb")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"  {path}   ({MODEL_ZOO[model]['display']})")
    print(f"\n{len(order)} notebooks written to {a.out}")


if __name__ == "__main__":
    main()
