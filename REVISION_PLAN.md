# Revision plan — status against your 13 items

Legend: **done** = finished and verified · **ready** = code/text written, needs the
Kaggle runs or your `.tex` · **blocked** = waiting on you

---

## The two findings that change the paper

**1. `diagnosis_confirm_type` is label leakage.** Measured on the real split:

| Category | P(melanoma) | n (train) |
|---|---|---|
| serial imaging showing no change | **0.000** | 1,687 |
| single image expert consensus | **0.000** | 851 |
| Unknown | **0.000** | 351 |
| confocal microscopy w/ consensus | **0.000** | 63 |
| histopathology | 0.664 | 8,989 |

On the **test** split, of 647 non-histopathology lesions, **zero** are melanoma.
Consequences, all measured:

- Rule "histopathology ⇒ melanoma", no image, no training: **75.3% accuracy**, macro-F1 **0.737**
- Metadata-only random forest, no image: **AUC 0.886** — beats 4 of the 14 models in your Table VI
- Drop the field: metadata-only AUC falls 0.886 → **0.794**; drop age too → **0.733**

This is why your SHAP kept ranking "histopathological confirmation" first. The
models learned the biopsy protocol, not dermatology. It is the strongest result
in the project, but it means every previously reported number is inflated.

**2. "BioViL-T" was never BioViL-T.** The notebook loads
`timm.create_model("vit_base_patch16_224")`. The paper explains its weak score
via a "chest-radiograph pre-training domain gap" — a rationale for pretraining
the model never had. I confirmed why: `AutoModel.from_pretrained("microsoft/BiomedVLP-BioViL-T")`
returns `CXRBertModel`, the **text** encoder. The image branch is not reachable that
way. The framework now reports the backbone it actually loaded, and that string
goes into `results.json` so the paper cannot misstate it again.

---

## Item-by-item

| # | Your request | Status |
|---|---|---|
| 1 | Re-run the models that didn't run | **ready** — 14 Kaggle notebooks generated; all 14 architectures verified to build, forward and backward |
| 2 | Same preprocessing + splitting everywhere | **done** — enforced in `unified/config.py`; loader aborts if the split isn't 11,941/2,559/2,559 |
| 3 | Highlight our own architectures in the abstract | **done** — `origin` field per model; abstract names the 4 mechanisms we designed |
| 4 | Models whose backbone is new on this dataset | **done** — `novel_on_dataset` flag; SegFormer-B5, DINOv2, MaxViT, DaViT, GNN fusion |
| 5 | Abstract reordered (motivation→models→XAI→ours→background→results→features) | **ready** — `paper/section_abstract.tex`, numbers marked `\FILL{}` |
| 6 | Introduction restructured | **done** — `paper/section_introduction.tex`: disease → statistics w/ citations → XAI cut to one paragraph → no subheadings → closes on contributions |
| 7 | Methodology: preprocessing + all models + architecture | **ready** — `paper/section_experimental_setup.tex` |
| 8 | Top-3 AUC + confusion matrices + ranked table | **ready** — `report.py` emits `fig_roc_top3.png`, `fig_confusion_top3.png`, `table_main.tex` |
| 9 | ViT beat the paper + what we changed | **ready** — `paper/section_discussion_vit.tex`, with a fairness caveat (see below) |
| 10 | SHAP/Grad-CAM — one sample only | **done** — `xai.py` selects ONE case deterministically; `xai_n_samples = 1` |
| 11 | Discussion: literature, feature drops, deeper results | **ready** — `section_discussion_leakage.tex` + `section_discussion_vit.tex` + `section_discussion_future.tex`; ablation runs via `--ablation` |
| 12 | Remove Ethical and Legal Analysis | **done** — section dropped; the two empirically grounded points moved into Limitations |
| 13 | 30 references | **done** — `paper/references.bib` (37 entries available, 30 to be cited) |
| 13.2 | Optuna | **done** — `tune.py`, shared search space, equal budget for every model |
| 13.5 | Shared parameters | **done** — `config.py` is the single source of truth |
| 13.6 | GPU optimisation | **done** — AMP, channels_last, cudnn.benchmark, fused AdamW, grad accumulation, pinned+persistent workers |
| 13.8 | F1 / Accuracy / Recall / Balanced accuracy / Precision | **done** — plus AUC, PR-AUC, specificity, ECE, confusion matrix, bootstrap 95% CI |

**Caveat on item 9.** Ahammed et al. use only age, sex and anatomical site — no
confirmation field. Quoting our *full-metadata* AUC against their 0.8851 compares a
model that can see the shortcut against one that cannot. The honest comparison uses
the `no_confirm` result. The section is written that way.

---

## What was wrong before

| Notebook | Problem found |
|---|---|
| Abdelrahman Phase 3 ×3 | never executed — 0 outputs |
| DINOv2 + TabTransformer | never executed — 0 outputs |
| DaViT-Tiny + Gated | 2 epochs, AUC **0.7163** (paper claims 0.91) |
| BioViL-T + LCSA | AUC **0.4826** = random (paper claims 0.86) |
| GNN+MaxViT, notebook13 | stopped at epoch 3 of 20/25 |
| MaxViT-crossattn (Jana) | mostly commented out |
| SwinV2 ×2 | corrupted JSON — git merge markers; **repaired** |

Config drift: image size 224/384/512, batch 8/16/32, epochs 1→40, and two
different splits (some `test_size=0.2` → n=3,412; others the real test set → n=2,559).

Table VI also contains numbers with no source in any notebook output (DINOv2 0.89
never ran; BioViL 0.86 actually scored 0.48). Those must be regenerated, not restated.

---

## What you need to do

**1. Send me the `.tex` / Overleaf source.** Everything in `paper/` is written as
drop-in sections, but I can't merge them or renumber citations without the source.

**2. Run the notebooks.** `kaggle_notebooks/01..14`, numbered cheapest-first.

- Attach the splitted-data dataset · GPU on · Internet on · Run All
- Each takes ~1–4 h; ~25–35 GPU-hours total. Kaggle gives 30 h/week per account,
  so split across the 4 team accounts ≈ 8 h each.
- Then send me `outputs/<model>__full/results.json` from each.

**3. Ablation**, at least for the top 3: set `FEATURE_SET = "no_confirm"` and re-run.

Then I generate the tables and figures and fill every `\FILL{}` in the paper.

---

## Local environment note

Your **C: drive was 100% full (0 bytes free)** — unrelated to this project, but it
broke the weight downloads mid-run. I moved the model cache to `D:\ml-cache` and
reclaimed ~3.7 GB. Set `HF_HOME=D:\ml-cache\hf` to keep it off C:.

Your RTX 3050 has 4 GB, so SegFormer-B5 and SwinV2 can't train locally — hence Kaggle.
