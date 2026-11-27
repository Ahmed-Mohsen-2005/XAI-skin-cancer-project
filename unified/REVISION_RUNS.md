# Revision runs: patient-disjoint benchmark, ablations, seeds

Why: in the original split, test images from patients already seen in training score AUC ~0.94,
while unseen patients score ~0.87. The new split has **zero patient overlap**
(`patient_id == "Unknown"` is a placeholder for 8,668 ISIC-2019 images, so each of those images is treated
as its own patient). The split is frozen in `unified/splits/patient_*.csv`, so it is identical on every laptop.

## Every person (cmd, inside `unified/`)

```
git pull
python make_patient_split.py
python revision_runs.py --who YOUR_NAME --list
python revision_runs.py --who YOUR_NAME
```

`YOUR_NAME` is one of `ahmed`, `abdelrahman`, `malak` or `jana`.
- `make_patient_split.py` builds `../../data_patient` from `../../data` in about 1 minute, using hard links, so no extra disk space is used. It must print `patient overlap ... 0`.
- `--list` shows your jobs and an estimated number of hours.
- The last command runs your jobs. It is resumable: if the laptop stops, run the same command again.

| who | jobs | ~hours |
|---|---|---|
| ahmed | ViT-EF, GNN-MaxViT, SwinV2-GAT, U-Net, GNN no-melanocytic, GNN seeds 1-2 | 6.8 |
| abdelrahman | Hybrid-ViT, ConvNeXt-BiCross, EffNetV2, DINOv2, BioViL, GNN image-only, Hybrid seeds 1-2 | 7.6 |
| malak | DaViT, MaxViT-CA, ConvNeXt-Bilinear, SwinV2-Tab, ViT-EF & DaViT image-only / no-melanocytic, DaViT seeds 1-2 | 6.1 |
| jana | SegFormer, SegFormer seeds 1-2, ViT-EF seeds 1-2 | 7.8 |

When it prints `ALL DONE`, zip `unified/outputs_revision` (all of it, including `best.pt`) and send it to Ahmed.
If a job prints `FAILED`, send the matching `outputs_revision/log_*.txt`.

## Ahmed, after collecting everything

1. Unzip every teammate's folders into `unified/outputs_revision/`.
2. Run:

```
python revision_analysis.py
```

It writes the following to `outputs_revision/comparison/`:
- The 14-model ranking on the patient-disjoint split, with tiers, DeLong, calibration, DCA, subgroups and ensemble.
- Original vs patient-disjoint AUC: `fig_split_shift`, `table_split_shift.tex`.
- Seed stability for Tier 1: `fig_seed_stability`, `table_seeds.tex`.
- Image-only and no-melanocytic ablations: `fig_ablation`, `table_ablation.tex`.
- The unified XAI (same cases for all 14 models: faithfulness, localisation, Shapley and permutation importance, per-case figures), in `outputs_revision/common_xai/`.

Use `--skip-xai` for statistics only.
