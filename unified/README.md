# Unified benchmark framework

Every architecture in this study is trained and evaluated through this package.
It exists to remove the defects that made the earlier per-member notebooks
non-comparable: each notebook had its own split, its own image size, its own
batch size, its own epoch count, and its own metric set.

## What is now shared

| Dimension | Before (per notebook) | Now (enforced by `config.py`) |
|---|---|---|
| Split | `train_test_split(test_size=0.2)` in some, the frozen 70/15/15 in others | frozen 70/15/15 only — **11,941 / 2,559 / 2,559**, verified at load time |
| Image size | 224 / 256 / 384 / 512 | 224 for every backbone |
| Batch size | 8 / 16 / 32 | effective **32** for every model (micro-batch + gradient accumulation) |
| Epochs | 1 / 2 / 3 / 5 / 10 / 20 / 25 / 30 / 40 | 20, early stop on validation AUC (patience 6) |
| Optimiser | AdamW / OneCycle / LBFGS mixtures | AdamW, warmup + cosine, discriminative LR |
| Loss | focal / mixed / BCE | BCE with label smoothing (corpus is 1:1 balanced) |
| Threshold | 0.5, or tuned on the evaluation split | Youden's J on **validation**, applied unchanged to test |
| Metrics | varied; often AUC + F1 only | accuracy, balanced accuracy, precision, recall, F1, AUC, PR-AUC, specificity, ECE, confusion matrix, bootstrap 95% CI |
| XAI figure | 3–10 samples, varied per notebook | **one** deterministically selected case per figure |

`run.py` refuses to start if the split row counts do not match the frozen
partition, so a model cannot silently drift off the shared ground again.

## Layout

```
config.py    single source of truth: split, preprocessing, hyper-parameters, model zoo
data.py      frozen-split loader, shared transforms, dataset
models.py    the 14 architectures behind one interface: forward(image, meta) -> logit
engine.py    AMP + channels_last + grad-accum + EMA + early stop + calibration
metrics.py   the reported metric block, threshold rule, temperature scaling, CIs
xai.py       Grad-CAM/++, Integrated Gradients, occlusion, rollout, SHAP, PDP
tune.py      Optuna, shared search space and equal trial budget for every model
report.py    aggregates runs into the paper's LaTeX tables and figures
run.py       entry point
make_kaggle_notebooks.py   emits one self-contained Kaggle notebook per model
```

## Running

Local:

```bash
python run.py --model vit_early_fusion --data-root ../../data --out-root outputs
python run.py --all --data-root ../../data --out-root outputs
python run.py --model vit_early_fusion --ablation          # feature-drop study
python report.py --out-root outputs --paper-dir ../paper/generated --data-root ../../data
```

Kaggle (recommended — the heavy backbones need more than 4 GB of VRAM):

```bash
python make_kaggle_notebooks.py --out ../kaggle_notebooks
```

Then upload a notebook, attach the split dataset, set the accelerator to GPU
and Internet to ON, and Run All.

## GPU throughput

Applied uniformly so the comparison is not confounded by optimisation effort:
mixed precision (fp16 autocast + GradScaler), `channels_last` for the
convolutional backbones, `cudnn.benchmark`, fused AdamW where available,
gradient accumulation to hold the effective batch constant, pinned-memory
non-blocking transfers, persistent workers with prefetch, and `set_to_none`
gradient zeroing. `torch.compile` is available but off by default because
compile time dominates for short runs.

## Metadata leakage

`diagnosis_confirm_type` deterministically encodes the label for about a
quarter of the corpus: of the test cases that are not histopathology-confirmed,
**zero** are melanoma. A model handed this field learns the acquisition
protocol rather than dermatology. The `no_confirm` feature set removes it, and
`report.py` reports metadata-only baselines so the contribution of the image is
visible rather than assumed. Run the ablation before quoting any headline
number.
