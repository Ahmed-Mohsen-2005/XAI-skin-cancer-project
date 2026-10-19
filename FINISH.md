# Finishing the paper

Everything is built and validated. The only thing still running is training.

## Status

The 14-model sweep is running locally on the RTX 3050:

```
python unified/run_all_local.py --epochs 8 --workers 3 \
    --data-root ../data --out-root ../runs
```

Measured rate: ViT-B/16 takes **519 s/epoch** at micro-batch 8 (effective 32 via
gradient accumulation), so roughly 85 min for a ViT-class model end to end.
Expect **12--14 hours** for all fourteen. Progress is appended to
`runs/sweep_progress.json` after each model, and each model writes its own
`runs/<model>__full/results.json` as it finishes.

To watch:

```bash
tail -f runs/sweep.log
cat runs/sweep_progress.json
```

If the machine is interrupted, re-running the same command skips every model
that already has a `results.json` and resumes with the next one.

## When the sweep finishes — three commands

```bash
cd XAI-skin-cancer-project/unified

# 1. ablation on the top models (leakage-controlled, the headline result)
python run_all_local.py --epochs 8 --workers 3 \
    --data-root ../../data --out-root ../../runs \
    --feature-sets no_confirm \
    --models vit_early_fusion hybrid_vit_mlp convnext_bicross

# 2. tables, figures, metadata-only baselines, significance test
python report.py --out-root ../../runs --paper-dir ../../generated \
    --data-root ../../data

# 3. assemble the manuscript
python build_paper.py --main ../../main.tex --runs ../../runs \
    --paper ../paper --out ../../main_revised.tex --generated ../../generated

# 4. check it before sending to Overleaf
python lint_tex.py ../../main_revised.tex
```

Upload `main_revised.tex` plus the `generated/` folder to Overleaf. Nothing
else is needed — the bibliography is inline (`thebibliography`, 30 entries).

## What build_paper.py does

It never hand-copies a number. It reads `runs/*/results.json` and generates:

- the abstract, with the real best-model metrics and the leakage figures
- the ranked results table, every model, all five required metrics + AUC with
  bootstrap 95% CIs
- the results prose, including how many models fail to beat the image-free
  metadata baseline
- the ablation table against the metadata-only reference
- the ViT-vs-prior-work comparison, which automatically states the honest
  leakage-controlled margin and says so explicitly if the confidence interval
  contains Ahammed et al.'s 0.8851 rather than claiming a win
- the epoch count and compute note in the protocol table, from the actual config
- the single-case XAI figures, copied into `generated/`

Re-run it at any point; it builds from whatever has completed so far.

## Verified

- 14/14 architectures build, forward, backward
- full pipeline validated end-to-end (train, calibrate, threshold, metrics,
  bootstrap, single-case XAI, SHAP, report, LaTeX)
- assembled manuscript lints clean: **0 errors, 30 references, 30 cited**
- Ethical and Legal Analysis section removed
- all stale model counts ("sixteen", "seventeen", "twelve") eliminated

## Known constraints

- `best.pt` is ~660 MB per model (stores both raw and EMA weights). Fourteen
  models is about 9 GB; D: had 26 GB free when the sweep started.
- No LaTeX toolchain on this machine, so the manuscript was checked with
  `lint_tex.py` (environment balance, refs, citations, figure files) rather
  than a real compile. Compile once in Overleaf.
- Epochs are 8 here, not the 20 in the original protocol draft, because of the
  4 GB card. This is applied uniformly to every model, so comparisons remain
  valid, and `build_paper.py` writes the true number into the protocol table.
