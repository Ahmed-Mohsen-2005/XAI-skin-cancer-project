# Unified benchmark framework

Every architecture in this study is trained, evaluated and explained through
this package, on one frozen split, with one protocol.

## What is shared

| Dimension | Before (per notebook) | Now (enforced by `config.py`) |
|---|---|---|
| Split | `train_test_split(test_size=0.2)` in some, the frozen 70/15/15 in others | frozen 70/15/15 only — **11,941 / 2,559 / 2,559**, verified at load time |
| Metadata | every column, incl. the confirmation method | **`no_confirm`** headline: the label-leaking confirmation method is removed (see below); `full` kept as an ablation |
| Image size | 224 / 256 / 384 / 512 | 224 for every backbone |
| Batch size | 8 / 16 / 32 | effective **32** for every model (micro-batch + gradient accumulation) |
| Augmentation | mild PIL chain | GPU: random-resized crop, full rotation + flips (one affine warp, reflection padding), colour/hue jitter, erasing, **counterfactual vignette** (below) |
| Epochs | 1 – 40 | 15 locally / 20 on Kaggle, early stop on validation AUC (patience 5), EMA weights |
| Loss | focal / mixed / BCE | BCE with label smoothing (corpus is 1:1 balanced) |
| Threshold | 0.5, or tuned on the evaluation split | accuracy-optimal on **validation**, applied unchanged to test |
| Inference | single view | temperature-calibrated, **8-view dihedral TTA** for every model |
| Metrics | often AUC + F1 only | accuracy, balanced accuracy, precision, recall, F1, AUC, PR-AUC, specificity, ECE, confusion matrix, bootstrap 95% CI |
| XAI | 3–10 unscored samples | one deterministic case per panel + TP/TN/FP/FN gallery, **scored** for faithfulness and localisation |

## Layout

```
config.py          single source of truth: split, features, hyper-parameters, model zoo
data.py            frozen-split loader, uint8 memmap cache, GPU augmentation, GPU loader
models.py          the 14 architectures behind one interface: forward(image, meta) -> logit
engine.py          bf16 AMP + grad-accum + EMA + early stop + resume + TTA + calibration
metrics.py         the reported metric block, threshold rules, temperature scaling, CIs
xai.py             scored XAI suite (see below)
redo_xai.py        rerun the XAI suite on trained checkpoints
run.py             one model / one feature set
run_all_local.py   the 14-model sweep on a 4 GB GPU (OOM back-off, crash resume)
launch_sweep.ps1   starts the sweep detached from the shell
status.py          sweep progress
report.py          tables, figures, greedy ensemble, XAI-quality table -> outputs/report/
tune.py            Optuna, shared search space, equal trial budget
make_kaggle_notebooks.py   one self-contained Kaggle notebook per model (T4/P100)
```

Outputs per run, under `outputs/<model>__<feature_set>/`: `results.json`,
`predictions.npz` (val/test probabilities and logits), `best.pt` (EMA
weights), `xai_panel.png`, `xai_gallery.png`, `shap_metadata.png`.

## Running

```powershell
# full sweep, detached; re-running skips finished models and resumes the rest
powershell -ExecutionPolicy Bypass -File launch_sweep.ps1
python status.py

# one model
python run.py --model vit_early_fusion --data-root ../../data --out-root outputs
python run.py --model vit_early_fusion --feature-set full ...   # leaky comparison

# tables, figures, ensemble, XAI-quality summary
python report.py --data-root ../../data

# re-explain trained models (e.g. after changing xai.py)
python redo_xai.py --runs outputs --data-root ../../data
```

The first run decodes the corpus once into `cache/` (~2.6 GB of uint8
memmaps, about 3 minutes).

## GPU throughput

Measured on the local RTX 3050 Laptop GPU (4 GB):

* JPEG decode happens once; batches are gathered from a memmap by a background
  thread into pinned memory, and **all augmentation runs on the GPU**. This
  removed the input bottleneck (DaViT went from ~12 to ~1.5 min/epoch, the
  U-Net from ~4 to under 1).
* bf16 autocast (no loss scaling needed on Ampere), TF32 matmuls,
  `channels_last` where the backbone allows it, fused AdamW,
  `cudnn.benchmark`.
* Micro-batches in `run_all_local.py` are sized from measured peak memory;
  gradient accumulation keeps the effective batch at 32.
* Each epoch writes `last.pt`; a crash, sleep or logoff costs at most one
  epoch. The process also asks Windows not to idle-sleep while it trains. (The
  previous sweep lost 9 of 14 models when the laptop slept and logged off.)

The heaviest backbones (SegFormer-B5, MaxViT, Swin-V2) are 3–11 min/epoch here;
the Kaggle notebooks run the same code on a 16 GB T4/P100.

## Shortcuts found in the data, and how they are handled

**Confirmation-method leak.** `diagnosis_confirm_type` deterministically
encodes the label for a quarter of the corpus: every lesion that was not
biopsy-confirmed is benign. With it, every model scored **100 %** on the 647
non-histopathology test cases, and SHAP ranked it as the top feature for every
architecture — the explanation described the acquisition protocol rather than
the lesion. The headline configuration (`no_confirm`) removes it; `full` is
kept only as an ablation, and `report.py` prints metadata-only baselines.

**Dermoscope-vignette shortcut.** A black circular vignette is present on
24 % of melanomas but 9 % of benign lesions in the training split, and the old
Grad-CAMs lit up the ring instead of the lesion. Training adds synthetic
vignettes at class-specific rates so that the vignette is equally common in
both classes, which removes the shortcut without discarding images. The XAI
scores report how much attribution falls outside the field of view.

**Patient overlap.** 682 patients have images in both train and test in the
frozen split. The split is kept frozen for comparability with the earlier
results; this is a known limitation that inflates all models alike.

## Explainability (`xai.py`)

* **Grad-CAM / Grad-CAM++** read from the right layer: the input of the last
  transformer block for ViTs (the old code hooked the patch-embedding conv,
  which is why the ViT maps showed the dermoscope ring), the final feature map
  for CNN / hierarchical backbones. Averaged over the 8 TTA views.
* **Integrated Gradients** with a blurred baseline, midpoint Riemann sum and
  SmoothGrad noise; the completeness error is recorded.
* **Occlusion** with blur fill and per-pixel averaging.
* **Gradient-weighted attention rollout** (Chefer et al., 2021) for ViTs;
  cross-attention, LACA-gate and LCSA maps for those fusion models; DaViT gate
  activations are recorded.
* Every map explains the **predicted** class (benign predictions are
  explained as benign).
* **Scored**, not only drawn: deletion / insertion AUC (faithfulness),
  share of attribution inside an automatically segmented lesion and outside
  the field of view (localisation), each against a random-map baseline, over
  24 test cases spread across TP / TN / FP / FN.
* **Clinical metadata**: exact grouped Shapley values (one player per clinical
  variable, all 2^G coalitions, image held fixed) for the case, and grouped
  permutation importance on 512 test cases, including "ALL metadata".
