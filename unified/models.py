"""
The fourteen multimodal architectures, ported to one common interface.

Every model:
    forward(image, meta) -> logits of shape (B,)      [single logit, BCE]
    param_groups(...)    -> discriminative LR groups (backbone vs head)
    tokens/attention accessors used by xai.py

The fusion mechanisms are the ones designed in this project; only the plumbing
(interface, metadata vector, output head) has been unified so that the models
are comparable. Backbone choices and hyper-parameters now come from config.py.
"""
from __future__ import annotations

import math
import os
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:  # pragma: no cover
    timm = None


# ===========================================================================
#  Shared building blocks
# ===========================================================================
class TokenBackbone(nn.Module):
    """
    Wraps any image backbone into a common token interface.

    forward(x) -> (tokens, hw)
        tokens : (B, N, C)   N = 1+P for ViT-style (CLS first), P for CNNs
        hw     : (H, W) spatial grid for CNN-style backbones, else None
    """

    def __init__(self, name: str, img_size: int = 224, pretrained: bool = True,
                 drop_path: float = 0.1):
        super().__init__()
        self.name = name
        self.kind = "timm"
        self.has_cls = False
        self.actual_backbone = name

        if name == "scratch_unet":
            self.kind = "unet"
            self.net = UNetStyleEncoder(d=256)
            self.out_dim = 256
            self.hw = (4, 4)
            return

        if name.startswith("nvidia/"):
            from transformers import SegformerModel
            # nvidia/mit-b5 publishes TensorFlow-only weights on the Hub, so the
            # PyTorch encoder is taken from the fine-tuned SegFormer-B5 release
            # (identical MiT-B5 hierarchical encoder) and the decode head is
            # discarded -- we only use it as a feature extractor.
            candidates = [name,
                          "nvidia/segformer-b5-finetuned-ade-640-640",
                          "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
                          "nvidia/mit-b2"]
            last_err = None
            for cand in candidates:
                try:
                    self.net = SegformerModel.from_pretrained(cand)
                    self.actual_backbone = cand
                    break
                except Exception as exc:  # noqa: BLE001
                    last_err = exc
            else:
                raise RuntimeError(f"could not load a SegFormer encoder: {last_err}")
            self.kind = "segformer"
            self.out_dim = self.net.config.hidden_sizes[-1]
            self.hw = (img_size // 32, img_size // 32)
            return

        if name == "biovil":
            # The genuine BioViL-T checkpoint is distributed with custom modelling
            # code, so loading it requires trust_remote_code=True, i.e. executing
            # Python downloaded from the Hub. That is an explicit opt-in here
            # rather than a silent default: set UNIFIED_ALLOW_REMOTE_CODE=1 to
            # enable it. Without the opt-in we fall back to ImageNet ViT-B/16 and
            # RECORD that fact, so the paper never claims a biomedical
            # pretraining distribution the model did not actually have.
            if os.environ.get("UNIFIED_ALLOW_REMOTE_CODE") == "1":
                try:
                    from transformers import AutoModel
                    self.net = AutoModel.from_pretrained(
                        "microsoft/BiomedVLP-BioViL-T", trust_remote_code=True)
                    self.kind = "biovil"
                    self.out_dim = 768
                    self.hw = (img_size // 16, img_size // 16)
                    self.has_cls = True
                    self.actual_backbone = "microsoft/BiomedVLP-BioViL-T"
                    return
                except Exception as exc:  # noqa: BLE001
                    warnings.warn(f"BioViL-T unavailable ({exc}); "
                                  "falling back to ImageNet ViT-B/16.")
            name = "vit_base_patch16_224"
            self.actual_backbone = "vit_base_patch16_224 (BioViL-T not loaded)"

        kwargs = dict(pretrained=pretrained, num_classes=0, global_pool="",
                      drop_path_rate=drop_path)
        if "dinov2" in name:
            kwargs["img_size"] = img_size
        self.net = timm.create_model(name, **kwargs)
        self.out_dim = self.net.num_features

        with torch.no_grad():
            probe = self.net(torch.zeros(1, 3, img_size, img_size))
        if probe.ndim == 3:
            self.layout = "nlc"
            self.has_cls = probe.shape[1] % 2 == 1  # 197, 257 ... -> CLS present
            p = probe.shape[1] - (1 if self.has_cls else 0)
            side = int(round(math.sqrt(p)))
            self.hw = (side, side)
        elif probe.shape[1] == self.out_dim:
            self.layout = "nchw"
            self.hw = (probe.shape[2], probe.shape[3])
        else:
            self.layout = "nhwc"
            self.hw = (probe.shape[1], probe.shape[2])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[Tuple[int, int]]]:
        if self.kind == "unet":
            return self.net(x), self.hw
        if self.kind == "segformer":
            out = self.net(pixel_values=x, output_hidden_states=False).last_hidden_state
            h, w = out.shape[2], out.shape[3]
            return out.flatten(2).transpose(1, 2), (h, w)
        if self.kind == "biovil":
            out = self.net(pixel_values=x).last_hidden_state   # (B, 1+P, D)
            return out, self.hw

        out = self.net(x)
        if self.layout == "nlc":
            return out, self.hw
        if self.layout == "nchw":
            return out.flatten(2).transpose(1, 2), (out.shape[2], out.shape[3])
        return out.flatten(1, 2), (out.shape[1], out.shape[2])


class ConvBNAct(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.GELU())

    def forward(self, x):
        return self.block(x)


class UNetStyleEncoder(nn.Module):
    """Multi-scale encoder trained from scratch (no ImageNet initialisation)."""

    def __init__(self, d: int = 256):
        super().__init__()
        self.stem = ConvBNAct(3, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBNAct(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBNAct(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ConvBNAct(128, 256))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), ConvBNAct(256, 512))
        self.proj = nn.Conv2d(32 + 64 + 128 + 256 + 512, d, 1)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        f1 = self.stem(x)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        f4 = self.down3(f3)
        f5 = self.down4(f4)
        feats = [F.adaptive_avg_pool2d(f, (4, 4)) for f in (f1, f2, f3, f4, f5)]
        feat = self.proj(torch.cat(feats, dim=1))
        return self.norm(feat.flatten(2).transpose(1, 2))


class TabTransformerEncoder(nn.Module):
    """One token per clinical feature + a CLS token. Returns (B, 1+M, D)."""

    def __init__(self, num_features: int, embed_dim: int = 256, depth: int = 2,
                 num_heads: int = 4, ff_dim: int = 512, dropout: float = 0.15):
        super().__init__()
        self.num_features = num_features
        self.embed = nn.Parameter(torch.randn(num_features, embed_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_features, embed_dim))
        self.pos = nn.Parameter(torch.zeros(1, num_features, embed_dim))
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # vectorised feature tokenisation: (B, M) -> (B, M, D)
        tok = x.unsqueeze(-1) * self.embed.unsqueeze(0) + self.bias.unsqueeze(0)
        tok = tok + self.pos
        tok = torch.cat([self.cls.expand(x.size(0), -1, -1), tok], dim=1)
        return self.norm(self.tf(tok))


class FTTransformerEncoder(nn.Module):
    """FT-Transformer; returns the pooled CLS embedding (B, D)."""

    def __init__(self, num_features: int, d: int = 256, depth: int = 2,
                 heads: int = 4, dropout: float = 0.15):
        super().__init__()
        self.tab = TabTransformerEncoder(num_features, d, depth, heads, d * 2, dropout)

    def forward(self, x):
        return self.tab(x)[:, 0]


class MLPMetaEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, out_dim), nn.LayerNorm(out_dim), nn.GELU())

    def forward(self, x):
        return self.net(x)


class CrossAttentionBlock(nn.Module):
    """Image self-attention + metadata-queries-image cross-attention."""

    def __init__(self, dim: int, num_heads: int, ff_dim: int, dropout: float = 0.15):
        super().__init__()
        self.meta_to_img = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.img_self = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_img = nn.LayerNorm(dim)
        self.norm_meta = nn.LayerNorm(dim)
        self.ffn_img = _ffn(dim, ff_dim, dropout)
        self.ffn_meta = _ffn(dim, ff_dim, dropout)

    def forward(self, img, meta, need_attn: bool = False):
        sa, _ = self.img_self(self.norm_img(img), img, img, need_weights=False)
        img = img + sa
        img = img + self.ffn_img(img)
        ca, attn = self.meta_to_img(self.norm_meta(meta), img, img,
                                    need_weights=need_attn,
                                    average_attn_weights=False)
        meta = meta + ca
        meta = meta + self.ffn_meta(meta)
        return img, meta, attn


class BidirectionalCrossAttentionBlock(nn.Module):
    """Mutual attention: image attends to metadata and metadata to image."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.15):
        super().__init__()
        self.img_to_meta = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.meta_to_img = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.n_img, self.n_meta = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ffn_img, self.ffn_meta = _ffn(dim, dim * 2, dropout), _ffn(dim, dim * 2, dropout)
        self.nf_img, self.nf_meta = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, img, meta, need_attn: bool = False):
        a, attn = self.img_to_meta(img, meta, meta, need_weights=need_attn,
                                   average_attn_weights=False)
        img = self.n_img(img + self.drop(a))
        b, _ = self.meta_to_img(meta, img, img, need_weights=False)
        meta = self.n_meta(meta + self.drop(b))
        img = self.nf_img(img + self.drop(self.ffn_img(img)))
        meta = self.nf_meta(meta + self.drop(self.ffn_meta(meta)))
        return img, meta, attn


class LesionAwareCrossAttentionBlock(nn.Module):
    """
    LACA (ours). The clinical CLS token predicts a per-patch saliency gate that
    multiplicatively steers fusion toward clinically relevant lesion regions
    before cross-attention is applied. The gate is retained for XAI.
    """

    def __init__(self, dim: int, num_heads: int, ff_dim: int, n_patches: int,
                 dropout: float = 0.15):
        super().__init__()
        self.n_patches = n_patches
        self.saliency_gate = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim // 2, n_patches))
        self.meta_to_img = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.img_self = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_img, self.norm_meta = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ffn_img, self.ffn_meta = _ffn(dim, ff_dim, dropout), _ffn(dim, ff_dim, dropout)

    def forward(self, img, meta, need_attn: bool = False):
        gate = torch.sigmoid(self.saliency_gate(meta[:, 0]))          # (B, P)
        cls_img, patches = img[:, :1], img[:, 1:]
        p = patches.size(1)
        g = gate[:, :p] if gate.size(1) >= p else F.pad(gate, (0, p - gate.size(1)), value=1.0)
        patches = patches * g.unsqueeze(-1)
        img = torch.cat([cls_img, patches], dim=1)

        sa, _ = self.img_self(self.norm_img(img), img, img, need_weights=False)
        img = img + sa
        img = img + self.ffn_img(img)
        ca, attn = self.meta_to_img(self.norm_meta(meta), img, img,
                                    need_weights=need_attn, average_attn_weights=False)
        meta = meta + ca
        meta = meta + self.ffn_meta(meta)
        return img, meta, attn, g


class LesionConditionedSparseAttention(nn.Module):
    """
    LCSA (ours). Clinical context queries the image, but attention is restricted
    to the top-k most relevant patches per head, mimicking the way a clinician
    fixates on a small number of salient regions rather than scanning uniformly.
    """

    def __init__(self, img_dim: int, meta_dim: int, fusion_dim: int = 256,
                 n_heads: int = 8, topk: int = 16, dropout: float = 0.1):
        super().__init__()
        assert fusion_dim % n_heads == 0
        self.h, self.dh, self.topk = n_heads, fusion_dim // n_heads, topk
        self.scale = self.dh ** -0.5
        self.img_proj = nn.Linear(img_dim, fusion_dim)
        self.meta_proj = nn.Linear(meta_dim, fusion_dim)
        self.q = nn.Linear(fusion_dim, fusion_dim)
        self.k = nn.Linear(fusion_dim, fusion_dim)
        self.v = nn.Linear(fusion_dim, fusion_dim)
        self.out = nn.Linear(fusion_dim * 2, fusion_dim)
        self.norm = nn.LayerNorm(fusion_dim)
        self.drop = nn.Dropout(dropout)
        self.gate = nn.Sequential(nn.Linear(fusion_dim * 2, 2), nn.Softmax(dim=-1))
        self.last_attn: Optional[torch.Tensor] = None

    def _split(self, t):
        b, n, _ = t.shape
        return t.view(b, n, self.h, self.dh).transpose(1, 2)

    def forward(self, img_cls, patches, meta_cls, keep_attn: bool = False):
        p = self.img_proj(patches)                       # (B, P, F)
        m = self.meta_proj(meta_cls).unsqueeze(1)        # (B, 1, F)
        q, k, v = self._split(self.q(m)), self._split(self.k(p)), self._split(self.v(p))
        logits = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, 1, P)

        kk = min(self.topk, logits.size(-1))
        thresh = logits.topk(kk, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < thresh, float("-inf"))
        attn = logits.softmax(dim=-1)
        if keep_attn:
            self.last_attn = attn.detach()
        ctx = (self.drop(attn) @ v).transpose(1, 2).reshape(m.size(0), 1, -1).squeeze(1)

        img_pooled = self.img_proj(img_cls) if img_cls.dim() == 2 else p.mean(1)
        joint = torch.cat([ctx, img_pooled], dim=-1)
        w = self.gate(joint)
        fused = self.out(joint) * w[:, :1] + ctx * w[:, 1:]
        return self.norm(fused), w


class GatedFusion(nn.Module):
    """
    Per-modality sigmoid gates (ours). The gate activations are themselves an
    interpretability signal: a low image gate means the model fell back on
    clinical context for that patient.
    """

    def __init__(self, img_dim: int, meta_dim: int, fusion_dim: int, dropout: float = 0.3):
        super().__init__()
        c = img_dim + meta_dim
        self.gate_img = nn.Sequential(nn.Linear(c, c // 2), nn.GELU(),
                                      nn.Linear(c // 2, img_dim), nn.Sigmoid())
        self.gate_meta = nn.Sequential(nn.Linear(c, c // 2), nn.GELU(),
                                       nn.Linear(c // 2, meta_dim), nn.Sigmoid())
        self.img_proj = nn.Linear(img_dim, fusion_dim)
        self.meta_proj = nn.Linear(meta_dim, fusion_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(fusion_dim), nn.Linear(fusion_dim, fusion_dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(fusion_dim * 2, fusion_dim), nn.LayerNorm(fusion_dim))

    def forward(self, f_img, f_meta):
        c = torch.cat([f_img, f_meta], dim=-1)
        g_img, g_meta = self.gate_img(c), self.gate_meta(c)
        fused = self.img_proj(g_img * f_img) + self.meta_proj(g_meta * f_meta)
        return self.mlp(fused), {"gate_img": g_img.detach(), "gate_meta": g_meta.detach()}


class DenseGATv2(nn.Module):
    """
    Dense GATv2 layer over a small fully-connected node set.

    Implemented densely rather than via torch_geometric: the graph here has a
    fixed, modest node count, so dense attention is mathematically identical,
    removes a heavyweight dependency, and avoids the per-sample graph-batching
    that dominated runtime in the original implementation.
    """

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.15):
        super().__init__()
        assert dim % heads == 0
        self.h, self.dh = heads, dim // heads
        self.lin_l = nn.Linear(dim, dim)
        self.lin_r = nn.Linear(dim, dim)
        self.att = nn.Parameter(torch.empty(1, heads, 1, 1, self.dh))
        nn.init.xavier_uniform_(self.att)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):                                  # x: (B, N, D)
        b, n, _ = x.shape
        l = self.lin_l(x).view(b, n, self.h, self.dh).transpose(1, 2)   # (B,H,N,dh)
        r = self.lin_r(x).view(b, n, self.h, self.dh).transpose(1, 2)
        e = F.leaky_relu(l.unsqueeze(3) + r.unsqueeze(2), 0.2)          # (B,H,N,N,dh)
        logits = (e * self.att).sum(-1)                                 # (B,H,N,N)
        alpha = self.drop(logits.softmax(dim=-1))
        out = (alpha @ r).transpose(1, 2).reshape(b, n, -1)
        return self.norm(out + x)


def _ffn(dim: int, hidden: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(),
                         nn.Dropout(dropout), nn.Linear(hidden, dim), nn.Dropout(dropout))


# ===========================================================================
#  Base class
# ===========================================================================
class FusionModel(nn.Module):
    """Common interface. Subclasses set self.backbone (if any) and self.head."""

    BACKBONE_ATTRS: Tuple[str, ...] = ("backbone", "img_encoder", "vit")

    def param_groups(self, lr_head: float, lr_backbone: float, weight_decay: float
                     ) -> List[Dict]:
        backbone_ids = set()
        for attr in self.BACKBONE_ATTRS:
            mod = getattr(self, attr, None)
            if isinstance(mod, nn.Module):
                backbone_ids |= {id(p) for p in mod.parameters()}
        bb_decay, bb_nodecay, hd_decay, hd_nodecay = [], [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            no_decay = p.ndim <= 1 or name.endswith(".bias")
            is_bb = id(p) in backbone_ids
            (bb_nodecay if (is_bb and no_decay) else
             bb_decay if is_bb else
             hd_nodecay if no_decay else hd_decay).append(p)
        groups = [
            {"params": bb_decay, "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": bb_nodecay, "lr": lr_backbone, "weight_decay": 0.0},
            {"params": hd_decay, "lr": lr_head, "weight_decay": weight_decay},
            {"params": hd_nodecay, "lr": lr_head, "weight_decay": 0.0},
        ]
        return [g for g in groups if g["params"]]

    def backbone_name(self) -> str:
        """
        The checkpoint actually instantiated, recorded in results.json.

        This exists so the manuscript can never again name a backbone the run
        did not use: an earlier configuration was described as BioViL-T while
        loading ImageNet ViT-B/16 weights.
        """
        bb = getattr(self, "backbone", None)
        if isinstance(bb, TokenBackbone):
            return bb.actual_backbone
        explicit = getattr(self, "backbone_id", None)
        if explicit:
            return explicit
        vit = getattr(self, "vit", None)
        if vit is not None:
            cfg = getattr(vit, "default_cfg", {}) or {}
            return cfg.get("architecture") or cfg.get("tag") or vit.__class__.__name__
        return "n/a"


# ===========================================================================
#  1. ViT Early Fusion  (ours)
# ===========================================================================
class ViTEarlyFusion(FusionModel):
    """Metadata is projected into token space and prepended before block 1, so
    every transformer layer is conditioned on clinical context."""

    def __init__(self, meta_dim: int, img_size: int = 224, embed_dim: int = 768):
        super().__init__()
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True,
                                     num_classes=0, global_pool="")
        self.backbone_id = "vit_base_patch16_224"
        self.meta_proj = nn.Sequential(
            nn.Linear(meta_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, embed_dim), nn.LayerNorm(embed_dim))
        self.meta_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.meta_pos, std=0.02)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 256), nn.GELU(),
            nn.Dropout(0.5), nn.Linear(256, 1))
        self.last_tokens: Optional[torch.Tensor] = None

    def forward(self, image, meta):
        x = self.vit.patch_embed(image)
        x = self.vit._pos_embed(x)
        x = self.vit.patch_drop(x)
        x = self.vit.norm_pre(x)
        tok = self.meta_proj(meta).unsqueeze(1) + self.meta_pos
        x = torch.cat([x, tok], dim=1)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        self.last_tokens = x
        return self.head(x[:, 0]).squeeze(-1)


# ===========================================================================
#  2. Hybrid ViT-MLP (late fusion)
# ===========================================================================
class HybridViTMLP(FusionModel):
    def __init__(self, meta_dim: int, img_size: int = 224, embed_dim: int = 768):
        super().__init__()
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True,
                                     num_classes=0, global_pool="token")
        self.backbone_id = "vit_base_patch16_224"
        self.mlp = nn.Sequential(
            nn.Linear(meta_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 128), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(embed_dim + 128, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))

    def forward(self, image, meta):
        return self.head(torch.cat([self.vit(image), self.mlp(meta)], dim=1)).squeeze(-1)


# ===========================================================================
#  3/4/8/9. Cross-attention family (backbone + TabTransformer + CA)
# ===========================================================================
class CrossAttentionFusion(FusionModel):
    """Shared implementation for the standard cross-attention family."""

    BLOCK = CrossAttentionBlock

    def __init__(self, meta_dim: int, backbone: str, img_size: int = 224,
                 d: int = 256, heads: int = 8, n_layers: int = 2, dropout: float = 0.15):
        super().__init__()
        self.backbone = TokenBackbone(backbone, img_size, drop_path=dropout)
        self.img_proj = nn.Sequential(nn.Linear(self.backbone.out_dim, d),
                                      nn.LayerNorm(d), nn.GELU())
        self.img_cls = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.img_cls, std=0.02)
        self.meta_encoder = TabTransformerEncoder(meta_dim, d, 2, 4, d * 2, dropout)
        n_patches = self.backbone.hw[0] * self.backbone.hw[1]
        self.ca_layers = nn.ModuleList([self._make_block(d, heads, dropout, n_patches)
                                        for _ in range(n_layers)])
        self.head = nn.Sequential(
            nn.LayerNorm(d * 2), nn.Linear(d * 2, 256), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 1))
        self.attn_maps: List[torch.Tensor] = []
        self.gates: List[torch.Tensor] = []

    def _make_block(self, d, heads, dropout, n_patches):
        return self.BLOCK(d, heads, d * 2, dropout)

    def forward(self, image, meta, keep_attn: bool = False):
        tok, _ = self.backbone(image)
        img = self.img_proj(tok)
        if not self.backbone.has_cls:
            img = torch.cat([self.img_cls.expand(img.size(0), -1, -1), img], dim=1)
        met = self.meta_encoder(meta)
        self.attn_maps, self.gates = [], []
        for blk in self.ca_layers:
            out = blk(img, met, need_attn=keep_attn)
            if len(out) == 4:
                img, met, attn, gate = out
                if keep_attn and gate is not None:
                    self.gates.append(gate.detach())
            else:
                img, met, attn = out
            if keep_attn and attn is not None:
                self.attn_maps.append(attn.detach())
        return self.head(torch.cat([img[:, 0], met[:, 0]], dim=1)).squeeze(-1)


class SegFormerLACA(CrossAttentionFusion):
    """SegFormer-B5 + Lesion-Aware Cross-Attention (ours)."""
    BLOCK = LesionAwareCrossAttentionBlock

    def _make_block(self, d, heads, dropout, n_patches):
        return LesionAwareCrossAttentionBlock(d, heads, d * 2, n_patches, dropout)


# ===========================================================================
#  5. ConvNeXt-T + Bidirectional Cross-Attention (ours)
# ===========================================================================
class ConvNeXtBiCross(FusionModel):
    def __init__(self, meta_dim: int, img_size: int = 224, d: int = 256,
                 heads: int = 8, n_layers: int = 2, meta_tokens: int = 8,
                 dropout: float = 0.15):
        super().__init__()
        self.backbone = TokenBackbone("convnext_tiny", img_size, drop_path=dropout)
        self.img_proj = nn.Sequential(nn.Linear(self.backbone.out_dim, d),
                                      nn.LayerNorm(d), nn.GELU())
        self.meta_tokens = meta_tokens
        self.meta_proj = nn.Sequential(nn.Linear(meta_dim, d * meta_tokens),
                                       nn.LayerNorm(d * meta_tokens), nn.GELU())
        self.layers = nn.ModuleList([BidirectionalCrossAttentionBlock(d, heads, dropout)
                                     for _ in range(n_layers)])
        self.head = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(d // 2, 1))
        self.attn_maps: List[torch.Tensor] = []

    def forward(self, image, meta, keep_attn: bool = False):
        tok, _ = self.backbone(image)
        img = self.img_proj(tok)
        met = self.meta_proj(meta).view(meta.size(0), self.meta_tokens, -1)
        self.attn_maps = []
        for blk in self.layers:
            img, met, attn = blk(img, met, need_attn=keep_attn)
            if keep_attn and attn is not None:
                self.attn_maps.append(attn.detach())
        return self.head(img.mean(1)).squeeze(-1)


# ===========================================================================
#  6. MaxViT-T + Graph Fusion (ours)
# ===========================================================================
class GNNMaxViT(FusionModel):
    """
    Heterogeneous graph over pooled image-region nodes and metadata nodes,
    refined by GATv2 message passing.

    Image tokens are adaptively pooled to a fixed node budget so the dense
    attention stays tractable; this is what makes the graph model trainable in
    the same wall-clock envelope as the other architectures.
    """

    def __init__(self, meta_dim: int, backbone: str = "maxvit_tiny_tf_224",
                 img_size: int = 224, d: int = 256, heads: int = 4,
                 img_nodes: int = 16, meta_tokens: int = 8, dropout: float = 0.15):
        super().__init__()
        self.backbone = TokenBackbone(backbone, img_size, drop_path=dropout)
        self.img_nodes, self.meta_tokens = img_nodes, meta_tokens
        self.img_proj = nn.Sequential(nn.Linear(self.backbone.out_dim, d),
                                      nn.LayerNorm(d), nn.GELU())
        self.meta_proj = nn.Sequential(nn.Linear(meta_dim, d * meta_tokens),
                                       nn.LayerNorm(d * meta_tokens), nn.GELU())
        self.gat1 = DenseGATv2(d, heads, dropout)
        self.gat2 = DenseGATv2(d, heads, dropout)
        self.head = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(d // 2, 1))

    def forward(self, image, meta):
        tok, hw = self.backbone(image)
        if self.backbone.has_cls:
            tok = tok[:, 1:]
        b, n, c = tok.shape
        side = int(round(math.sqrt(self.img_nodes)))
        grid = tok.transpose(1, 2).reshape(b, c, hw[0], hw[1])
        grid = F.adaptive_avg_pool2d(grid, (side, side))
        img = self.img_proj(grid.flatten(2).transpose(1, 2))
        met = self.meta_proj(meta).view(b, self.meta_tokens, -1)
        nodes = torch.cat([img, met], dim=1)
        nodes = self.gat2(self.gat1(nodes))
        return self.head(nodes[:, :img.size(1)].mean(1)).squeeze(-1)


# ===========================================================================
#  7. DaViT-T + Gated Fusion (ours)
# ===========================================================================
class DaViTGated(FusionModel):
    def __init__(self, meta_dim: int, img_size: int = 224, fusion_dim: int = 256,
                 meta_embed: int = 128, dropout: float = 0.3):
        super().__init__()
        self.backbone = TokenBackbone("davit_tiny", img_size, drop_path=0.1)
        self.meta_encoder = MLPMetaEncoder(meta_dim, meta_embed, dropout)
        self.fusion = GatedFusion(self.backbone.out_dim, meta_embed, fusion_dim, dropout)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(fusion_dim, fusion_dim // 2), nn.GELU(),
            nn.Dropout(dropout * 0.5), nn.Linear(fusion_dim // 2, 1))
        self.last_gates: Dict[str, torch.Tensor] = {}

    def forward(self, image, meta):
        tok, _ = self.backbone(image)
        f_img = tok.mean(1)
        fused, gates = self.fusion(f_img, self.meta_encoder(meta))
        self.last_gates = gates
        return self.head(fused).squeeze(-1)


# ===========================================================================
#  10. DINOv2 + Bidirectional Cross-Attention (ours)
# ===========================================================================
class DINOv2BiCross(FusionModel):
    def __init__(self, meta_dim: int, img_size: int = 224, d: int = 256,
                 heads: int = 8, n_layers: int = 2, dropout: float = 0.15):
        super().__init__()
        self.backbone = TokenBackbone("vit_small_patch14_dinov2.lvd142m",
                                      img_size, drop_path=dropout)
        self.img_proj = nn.Sequential(nn.Linear(self.backbone.out_dim, d),
                                      nn.LayerNorm(d), nn.GELU())
        self.meta_encoder = TabTransformerEncoder(meta_dim, d, 2, 4, d * 2, dropout)
        self.layers = nn.ModuleList([BidirectionalCrossAttentionBlock(d, heads, dropout)
                                     for _ in range(n_layers)])
        self.head = nn.Sequential(
            nn.LayerNorm(d * 2), nn.Linear(d * 2, 256), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(256, 1))
        self.attn_maps: List[torch.Tensor] = []

    def forward(self, image, meta, keep_attn: bool = False):
        tok, _ = self.backbone(image)
        img = self.img_proj(tok)
        met = self.meta_encoder(meta)
        self.attn_maps = []
        for blk in self.layers:
            img, met, attn = blk(img, met, need_attn=keep_attn)
            if keep_attn and attn is not None:
                self.attn_maps.append(attn.detach())
        return self.head(torch.cat([img[:, 0], met[:, 0]], dim=1)).squeeze(-1)


# ===========================================================================
#  11. BioViL-T + LCSA (ours)
# ===========================================================================
class BioViLLCSA(FusionModel):
    def __init__(self, meta_dim: int, img_size: int = 224, d: int = 256,
                 topk: int = 16, dropout: float = 0.15):
        super().__init__()
        self.backbone = TokenBackbone("biovil", img_size, drop_path=dropout)
        self.meta_encoder = FTTransformerEncoder(meta_dim, d, 2, 4, dropout)
        self.fusion = LesionConditionedSparseAttention(
            self.backbone.out_dim, d, d, n_heads=8, topk=topk, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(d), nn.Dropout(dropout), nn.Linear(d, 128), nn.GELU(),
            nn.Dropout(dropout / 2), nn.Linear(128, 1))
        self.last_gate: Optional[torch.Tensor] = None

    def forward(self, image, meta, keep_attn: bool = False):
        tok, _ = self.backbone(image)
        cls, patches = (tok[:, 0], tok[:, 1:]) if self.backbone.has_cls else (tok.mean(1), tok)
        fused, gate = self.fusion(cls, patches, self.meta_encoder(meta), keep_attn=keep_attn)
        self.last_gate = gate.detach()
        return self.head(fused).squeeze(-1)


# ===========================================================================
#  12. ConvNeXt + FT-Transformer Bilinear Fusion (ours)
# ===========================================================================
class ConvNeXtBilinear(FusionModel):
    def __init__(self, meta_dim: int, img_size: int = 224, d: int = 256,
                 dropout: float = 0.15):
        super().__init__()
        self.backbone = TokenBackbone("convnext_tiny", img_size, drop_path=dropout)
        self.img_proj = nn.Sequential(nn.Linear(self.backbone.out_dim, d),
                                      nn.GELU(), nn.Dropout(dropout))
        self.tab = FTTransformerEncoder(meta_dim, d, 2, 4, dropout)
        self.tab_proj = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout))
        self.bilinear = nn.Bilinear(d, d, d)
        self.head = nn.Sequential(nn.Linear(d * 3, d * 2), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(d * 2, 1))

    def forward(self, image, meta):
        tok, _ = self.backbone(image)
        img = self.img_proj(tok.mean(1))
        tab = self.tab_proj(self.tab(meta))
        return self.head(torch.cat([img, tab, self.bilinear(img, tab)], dim=-1)).squeeze(-1)


# ===========================================================================
#  13. Swin-V2 + Graph Attention Fusion (ours)
# ===========================================================================
class SwinV2GAT(FusionModel):
    def __init__(self, meta_dim: int, img_size: int = 224, d: int = 256,
                 heads: int = 4, dropout: float = 0.15):
        super().__init__()
        self.backbone = TokenBackbone("swinv2_cr_small_224", img_size, drop_path=dropout)
        self.img_proj = nn.Sequential(nn.Linear(self.backbone.out_dim, d),
                                      nn.GELU(), nn.Dropout(dropout))
        self.node_embed = nn.Parameter(torch.randn(meta_dim, d) * 0.02)
        self.node_bias = nn.Parameter(torch.zeros(meta_dim, d))
        self.gat1 = DenseGATv2(d, heads, dropout)
        self.gat2 = DenseGATv2(d, heads, dropout)
        self.meta_pool = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout))
        self.fuse = nn.Sequential(
            nn.Linear(d * 3, d * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d * 2, d), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Linear(d, 1)

    def forward(self, image, meta):
        tok, _ = self.backbone(image)
        img = self.img_proj(tok.mean(1))
        nodes = meta.unsqueeze(-1) * self.node_embed.unsqueeze(0) + self.node_bias.unsqueeze(0)
        nodes = self.gat2(self.gat1(nodes))
        met = self.meta_pool(nodes.mean(1))
        fused = self.fuse(torch.cat([img, met, img * met], dim=-1))
        return self.head(fused).squeeze(-1)


# ===========================================================================
#  14. U-Net Encoder + Metadata Cross-Attention (ours)
# ===========================================================================
class UNetCrossAttn(FusionModel):
    def __init__(self, meta_dim: int, img_size: int = 224, d: int = 256,
                 heads: int = 8, n_layers: int = 2, dropout: float = 0.15):
        super().__init__()
        self.backbone = TokenBackbone("scratch_unet", img_size)
        self.meta_encoder = TabTransformerEncoder(meta_dim, d, 2, 4, d * 2, dropout)
        self.layers = nn.ModuleList([BidirectionalCrossAttentionBlock(d, heads, dropout)
                                     for _ in range(n_layers)])
        self.fuse = nn.Sequential(
            nn.Linear(d * 3, d * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d * 2, d), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Linear(d, 1)

    def forward(self, image, meta):
        img, _ = self.backbone(image)
        met = self.meta_encoder(meta)
        for blk in self.layers:
            img, met, _ = blk(img, met)
        i, m = img.mean(1), met.mean(1)
        return self.head(self.fuse(torch.cat([i, m, i * m], dim=-1))).squeeze(-1)


# ===========================================================================
#  Factory
# ===========================================================================
def build_model(model_name: str, meta_dim: int, img_size: int = 224) -> FusionModel:
    if model_name == "vit_early_fusion":
        return ViTEarlyFusion(meta_dim, img_size)
    if model_name == "hybrid_vit_mlp":
        return HybridViTMLP(meta_dim, img_size)
    if model_name == "convnext_bicross":
        return ConvNeXtBiCross(meta_dim, img_size)
    if model_name == "swinv2_tabtransformer":
        return CrossAttentionFusion(meta_dim, "swinv2_cr_small_224", img_size)
    if model_name == "gnn_maxvit":
        return GNNMaxViT(meta_dim, "maxvit_tiny_tf_224", img_size)
    if model_name == "segformer_laca":
        return SegFormerLACA(meta_dim, "nvidia/mit-b5", img_size)
    if model_name == "davit_gated":
        return DaViTGated(meta_dim, img_size)
    if model_name == "maxvit_crossattn":
        return CrossAttentionFusion(meta_dim, "maxvit_tiny_tf_224", img_size)
    if model_name == "effnetv2_crossattn":
        return CrossAttentionFusion(meta_dim, "tf_efficientnetv2_s", img_size)
    if model_name == "dinov2_crossattn":
        return DINOv2BiCross(meta_dim, img_size)
    if model_name == "biovil_lcsa":
        return BioViLLCSA(meta_dim, img_size)
    if model_name == "convnext_bilinear":
        return ConvNeXtBilinear(meta_dim, img_size)
    if model_name == "swinv2_gat":
        return SwinV2GAT(meta_dim, img_size)
    if model_name == "unet_crossattn":
        return UNetCrossAttn(meta_dim, img_size)
    raise KeyError(f"unknown model {model_name!r}")


def count_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
