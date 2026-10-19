"""
Unified experimental configuration -- SINGLE SOURCE OF TRUTH.

Every architecture in this study is trained and evaluated through this file.
Nothing below may be overridden inside a model notebook; the only per-model
degree of freedom is `micro_batch`, a pure memory/throughput knob that is
compensated by gradient accumulation so the EFFECTIVE batch size is identical
for every model.

Rationale for each shared value is documented inline so that the Methodology
section of the paper can cite this file directly.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any
import json

# ---------------------------------------------------------------------------
#  Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
#  Data split -- FIXED, PRE-COMPUTED, SHARED BY EVERY MODEL
#  Produced once by "EDA & preprocessing/final-splitted-data.ipynb" using a
#  stratified 70/15/15 partition with random_state=SEED. Models consume these
#  exact files; no model is permitted to re-split the corpus.
# ---------------------------------------------------------------------------
SPLIT_RATIOS = (0.70, 0.15, 0.15)
N_TRAIN, N_VAL, N_TEST, N_TOTAL = 11_941, 2_559, 2_559, 17_059

TARGET_COL = "class"
IMAGE_COL = "image_fixed"          # filename inside the images/ folder
ID_COLS = ["image", "isic_id", "patient_id", "year", "image_fixed"]

# ---------------------------------------------------------------------------
#  Metadata feature sets (ablation axis)
#
#  LEAKAGE WARNING, measured on the training split:
#    diagnosis_confirm_type_serial imaging showing no change -> P(mel)=0.000 n=1687
#    diagnosis_confirm_type_single image expert consensus    -> P(mel)=0.000 n= 851
#    diagnosis_confirm_type_Unknown                          -> P(mel)=0.000 n= 351
#    diagnosis_confirm_type_confocal microscopy w/ consensus -> P(mel)=0.000 n=  63
#
#  The confirmation-method field deterministically encodes the label for ~25%
#  of the corpus, because melanoma ground truth was only ever assigned to
#  biopsy-confirmed lesions. A model given this field learns the acquisition
#  protocol, not dermatology. FULL is reported for comparability with prior
#  work; NO_CONFIRM is the headline, leakage-controlled configuration.
# ---------------------------------------------------------------------------
CONFIRM_PREFIX = "diagnosis_confirm_type"
AGE_PREFIXES = ("age_scaled", "age_group")

FEATURE_SETS: Dict[str, Dict[str, Any]] = {
    "full":       {"drop_prefixes": ()},
    "no_confirm": {"drop_prefixes": (CONFIRM_PREFIX,)},
    "no_age":     {"drop_prefixes": AGE_PREFIXES},
    "no_both":    {"drop_prefixes": (CONFIRM_PREFIX,) + AGE_PREFIXES},
}
DEFAULT_FEATURE_SET = "full"       # main table; ablations are separate runs


# ---------------------------------------------------------------------------
#  Shared training protocol
# ---------------------------------------------------------------------------
@dataclass
class UnifiedConfig:
    # -- identity ----------------------------------------------------------
    model_name: str = "vit_early_fusion"
    feature_set: str = DEFAULT_FEATURE_SET
    run_tag: str = ""

    # -- image pipeline (IDENTICAL for every architecture) -----------------
    img_size: int = 224        # every backbone is instantiated at 224x224
    resize_to: int = 256       # shortest side, then centre-crop to img_size
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    # -- optimisation (IDENTICAL) ------------------------------------------
    effective_batch: int = 32  # held constant via gradient accumulation
    micro_batch: int = 32      # ONLY per-model knob (VRAM); see MODEL_ZOO
    epochs: int = 20
    warmup_epochs: int = 2
    lr_head: float = 3e-4
    lr_backbone: float = 3e-5  # discriminative LR: 10x lower on pretrained weights
    min_lr: float = 1e-6
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    label_smooth: float = 0.05
    ema_decay: float = 0.999
    early_stop_patience: int = 6
    monitor: str = "val_auc"

    # -- runtime -----------------------------------------------------------
    amp: bool = True
    channels_last: bool = True
    num_workers: int = 4
    seed: int = SEED
    deterministic: bool = False   # cudnn.benchmark instead, for throughput

    # -- evaluation protocol (IDENTICAL) -----------------------------------
    threshold_rule: str = "youden"  # chosen on VALIDATION, applied to TEST
    calibrate: bool = True          # temperature scaling fitted on VALIDATION
    tta: bool = False               # off for the main table (fair comparison)
    n_bootstrap: int = 1000         # 95% CI on test metrics

    # -- XAI (IDENTICAL) ---------------------------------------------------
    xai_sample_strategy: str = "single"  # ONE representative case per figure
    xai_n_samples: int = 1
    ig_steps: int = 50
    lime_n_samples: int = 1000
    # KernelExplainer evaluates (shap_nsamples x shap_background) forward passes
    # with the image held fixed. These values keep that product near 4k, which
    # is a few minutes even for SegFormer-B5; raising them improves the
    # attribution estimate at roughly linear cost.
    shap_background: int = 25
    shap_nsamples: int = 150

    # -- paths -------------------------------------------------------------
    data_root: str = "/kaggle/input"
    out_root: str = "/kaggle/working/outputs"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


# ---------------------------------------------------------------------------
#  Model zoo -- the 14 architectures of the benchmark.
#
#  micro_batch       tuned so the model fits a 16 GB T4 under AMP; gradient
#                    accumulation restores the common effective batch of 32.
#  origin            "ours"    -> fusion mechanism designed by this work
#                    "adapted" -> standard fusion with a standard backbone
#  novel_on_dataset  backbone not previously applied to ISIC-DICM-17K
# ---------------------------------------------------------------------------
MODEL_ZOO: Dict[str, Dict[str, Any]] = {
    "vit_early_fusion": dict(
        display="ViT-B/16 Early Fusion", family="Early fusion",
        backbone="vit_base_patch16_224", micro_batch=32,
        origin="ours", novel_on_dataset=False,
        note="metadata projected into token space, prepended before block 1"),

    "hybrid_vit_mlp": dict(
        display="Hybrid ViT-MLP (Late Fusion)", family="Late fusion",
        backbone="vit_base_patch16_224", micro_batch=32,
        origin="adapted", novel_on_dataset=False,
        note="independent encoders, concatenation at the classification head"),

    "convnext_bicross": dict(
        display="ConvNeXt-T + Bidirectional Cross-Attention", family="Cross-attention",
        backbone="convnext_tiny", micro_batch=32,
        origin="ours", novel_on_dataset=False,
        note="mutual image<->metadata attention, 2 stacked layers"),

    "swinv2_tabtransformer": dict(
        display="Swin-V2 + TabTransformer Cross-Attention", family="Cross-attention",
        backbone="swinv2_cr_small_224", micro_batch=16,
        origin="ours", novel_on_dataset=False,
        note="shifted-window features queried by clinical tokens"),

    "gnn_maxvit": dict(
        display="MaxViT-T + Graph Fusion (GATv2)", family="Graph",
        backbone="maxvit_tiny_tf_224", micro_batch=16,
        origin="ours", novel_on_dataset=True,
        note="heterogeneous image/metadata node graph, GATv2 message passing"),

    "segformer_laca": dict(
        display="SegFormer-B5 + Lesion-Aware Cross-Attention", family="Cross-attention",
        backbone="nvidia/mit-b5", micro_batch=8,
        origin="ours", novel_on_dataset=True,
        note="LACA: clinical CLS token gates per-patch saliency before fusion"),

    "davit_gated": dict(
        display="DaViT-T + Gated Fusion", family="Gated",
        backbone="davit_tiny", micro_batch=32,
        origin="ours", novel_on_dataset=True,
        note="per-modality sigmoid gates, directly inspectable as an XAI signal"),

    "maxvit_crossattn": dict(
        display="MaxViT-T + TabTransformer Cross-Attention", family="Cross-attention",
        backbone="maxvit_tiny_tf_224", micro_batch=16,
        origin="adapted", novel_on_dataset=True,
        note="multi-axis attention backbone, standard cross-attention fusion"),

    "effnetv2_crossattn": dict(
        display="EfficientNetV2-S + TabTransformer Cross-Attention", family="Cross-attention",
        backbone="tf_efficientnetv2_s", micro_batch=32,
        origin="adapted", novel_on_dataset=False,
        note="convolutional control for the cross-attention family"),

    "dinov2_crossattn": dict(
        display="DINOv2 ViT-S/14 + Bidirectional Cross-Attention", family="Cross-attention",
        backbone="vit_small_patch14_dinov2.lvd142m", micro_batch=16,
        origin="ours", novel_on_dataset=True,
        note="self-supervised features, no dermoscopic supervision in pretraining"),

    "biovil_lcsa": dict(
        display="BioViL-T + Lesion-Conditioned Sparse Attention", family="Sparse cross-attention",
        backbone="vit_base_patch16_224", micro_batch=16,
        origin="ours", novel_on_dataset=True,
        note="LCSA: top-k sparse attention conditioned on clinical context"),

    "convnext_bilinear": dict(
        display="ConvNeXt-T + FT-Transformer Bilinear Fusion", family="Bilinear",
        backbone="convnext_tiny", micro_batch=32,
        origin="ours", novel_on_dataset=True,
        note="second-order outer-product cross-modal interaction"),

    "swinv2_gat": dict(
        display="Swin-V2 + Graph Attention Fusion", family="Graph",
        backbone="swinv2_cr_small_224", micro_batch=16,
        origin="ours", novel_on_dataset=True,
        note="relational graph over clinical feature nodes, GAT message passing"),

    "unet_crossattn": dict(
        display="U-Net Encoder + Metadata Cross-Attention", family="Cross-attention",
        backbone="scratch_unet", micro_batch=32,
        origin="ours", novel_on_dataset=True,
        note="multi-scale encoder trained from scratch, no ImageNet initialisation"),
}

# Non-deep reference points, trained by report.py to contextualise the table.
BASELINES = ["metadata_only_rf", "metadata_only_lr", "image_only_vit"]


def get_config(model_name: str, **overrides) -> UnifiedConfig:
    if model_name not in MODEL_ZOO:
        raise KeyError(f"{model_name!r} not in MODEL_ZOO; choose from {list(MODEL_ZOO)}")
    cfg = UnifiedConfig(model_name=model_name,
                        micro_batch=MODEL_ZOO[model_name]["micro_batch"])
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise KeyError(f"unknown config field {k!r}")
        setattr(cfg, k, v)
    return cfg


def grad_accum_steps(cfg: UnifiedConfig) -> int:
    if cfg.effective_batch % cfg.micro_batch != 0:
        raise ValueError("effective_batch must be divisible by micro_batch")
    return cfg.effective_batch // cfg.micro_batch
