"""
Unified explainability suite.

Figure policy (paper requirement): ONE representative case per figure.
The case is selected deterministically as the highest-confidence CORRECTLY
classified melanoma in the test set, so every model's XAI panel shows the same
kind of evidence and the figures are directly comparable across architectures.
The selected isic_id is written into the results so the figure caption can name
the exact case.

Methods
  image branch : Grad-CAM, Grad-CAM++, Integrated Gradients, LIME, occlusion
  token branch : attention rollout (ViT), cross-attention maps, LACA gate
  tabular      : SHAP (KernelExplainer), permutation importance, PDP
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import UnifiedConfig


# ---------------------------------------------------------------------------
#  Case selection -- deterministic, one case
# ---------------------------------------------------------------------------
def select_case(probs: np.ndarray, labels: np.ndarray, want_class: int = 1) -> int:
    """
    Highest-confidence correctly-classified case of `want_class`.
    Deterministic: no randomness, no cherry-picking across runs.
    """
    correct = (labels == want_class) & ((probs >= 0.5).astype(int) == want_class)
    if not correct.any():
        return int(np.argmax(probs if want_class == 1 else -probs))
    idx = np.where(correct)[0]
    return int(idx[np.argmax(probs[idx])])


def denormalize(img: torch.Tensor, mean: List[float], std: List[float]) -> np.ndarray:
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    out = (img.detach().cpu() * s + m).clamp(0, 1)
    return out.permute(1, 2, 0).numpy()


# ---------------------------------------------------------------------------
#  Grad-CAM / Grad-CAM++
# ---------------------------------------------------------------------------
def find_cam_layer(model: nn.Module, img: Optional[torch.Tensor] = None,
                   meta: Optional[torch.Tensor] = None,
                   min_grid: int = 7) -> Optional[nn.Module]:
    """
    Deepest convolution whose activation is still spatially informative.

    Naively taking the last Conv2d is wrong for several of these backbones: in
    the U-Net encoder it is a 1x1 projection over a 4x4 grid, which upsamples
    into meaningless blobs, and in ViT the only convolution is the patch
    embedding. The layer is therefore chosen by probing actual activation
    shapes and keeping the deepest one with a grid of at least `min_grid` per
    side, falling back to the coarsest available if none qualifies.
    """
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        return None
    if img is None:
        return convs[-1]

    shapes: Dict[int, Tuple[int, int]] = {}
    hooks = []

    def mk(idx):
        def hook(_m, _i, o):
            if isinstance(o, torch.Tensor) and o.ndim == 4:
                shapes[idx] = (o.shape[2], o.shape[3])
        return hook

    for i, m in enumerate(convs):
        hooks.append(m.register_forward_hook(mk(i)))
    try:
        with torch.no_grad():
            model(img, meta)
    except Exception:  # noqa: BLE001
        return convs[-1]
    finally:
        for h in hooks:
            h.remove()

    ok = [i for i, hw in shapes.items() if min(hw) >= min_grid]
    if ok:
        return convs[max(ok)]
    return convs[max(shapes)] if shapes else convs[-1]


class GradCAM:
    """Grad-CAM and Grad-CAM++ from a single target layer."""

    def __init__(self, model: nn.Module, layer: nn.Module):
        self.model, self.layer = model, layer
        self.acts: Optional[torch.Tensor] = None
        self.grads: Optional[torch.Tensor] = None
        self._h = [layer.register_forward_hook(self._fwd),
                   layer.register_full_backward_hook(self._bwd)]

    def _fwd(self, _m, _i, o):
        self.acts = o.detach() if isinstance(o, torch.Tensor) else o[0].detach()

    def _bwd(self, _m, _gi, go):
        self.grads = go[0].detach()

    def remove(self):
        for h in self._h:
            h.remove()

    def _to_map(self, cam: torch.Tensor, size: int) -> np.ndarray:
        cam = F.relu(cam)
        cam = F.interpolate(cam[None, None], size=(size, size),
                            mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        return (cam / (cam.max() + 1e-8)).cpu().numpy()

    def __call__(self, img: torch.Tensor, meta: torch.Tensor,
                 plus_plus: bool = False) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        out = self.model(img, meta)
        out.sum().backward()
        a, g = self.acts, self.grads
        if a is None or g is None or a.ndim != 4:
            raise RuntimeError("target layer did not produce 4-D activations")
        if plus_plus:
            g2, g3 = g.pow(2), g.pow(3)
            denom = 2 * g2 + (a * g3).sum(dim=(2, 3), keepdim=True)
            alpha = g2 / torch.where(denom != 0, denom, torch.ones_like(denom))
            w = (alpha * F.relu(g)).sum(dim=(2, 3), keepdim=True)
        else:
            w = g.mean(dim=(2, 3), keepdim=True)
        cam = (w * a).sum(1)[0]
        return self._to_map(cam, img.shape[-1])


# ---------------------------------------------------------------------------
#  Integrated Gradients (image branch)
# ---------------------------------------------------------------------------
def integrated_gradients(model, img, meta, steps: int = 50) -> np.ndarray:
    baseline = torch.zeros_like(img)
    total = torch.zeros_like(img)
    for a in torch.linspace(0, 1, steps):
        x = (baseline + a * (img - baseline)).clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        model(x, meta).sum().backward()
        total = total + x.grad.detach()
    attr = ((img - baseline) * total / steps)[0].abs().sum(0)
    attr = attr - attr.min()
    return (attr / (attr.max() + 1e-8)).cpu().numpy()


# ---------------------------------------------------------------------------
#  Occlusion sensitivity
# ---------------------------------------------------------------------------
@torch.no_grad()
def occlusion(model, img, meta, patch: int = 32, stride: int = 16) -> np.ndarray:
    base = torch.sigmoid(model(img, meta)).item()
    size = img.shape[-1]
    grid = np.zeros(((size - patch) // stride + 1, (size - patch) // stride + 1))
    for yi, y in enumerate(range(0, size - patch + 1, stride)):
        batch, coords = [], []
        for xi, x in enumerate(range(0, size - patch + 1, stride)):
            occ = img.clone()
            occ[0, :, y:y + patch, x:x + patch] = 0
            batch.append(occ)
            coords.append(xi)
        probs = torch.sigmoid(model(torch.cat(batch), meta.repeat(len(batch), 1)))
        for xi, p in zip(coords, probs):
            grid[yi, xi] = base - p.item()
    grid = grid - grid.min()
    grid = grid / (grid.max() + 1e-8)
    return np.array(torch.nn.functional.interpolate(
        torch.tensor(grid)[None, None].float(), size=(size, size),
        mode="bilinear", align_corners=False)[0, 0])


# ---------------------------------------------------------------------------
#  Attention rollout (ViT-style token models)
# ---------------------------------------------------------------------------
def attention_rollout(model, img, meta) -> Optional[np.ndarray]:
    """
    Abnar & Zuidema rollout over the ViT blocks.

    timm's Attention computes the attention matrix internally and returns only
    the projected output, so the matrix cannot be captured from the module's
    own forward hook. It is instead captured from `attn_drop`, which receives
    the post-softmax (B, H, N, N) matrix as its input. Fused SDPA has to be
    disabled for the duration, because in that path the matrix is never
    materialised at all.
    """
    attns: List[torch.Tensor] = []
    hooks = []
    toggled = []

    def grab(_m, inp, _o):
        a = inp[0]
        if isinstance(a, torch.Tensor) and a.ndim == 4 and a.shape[-1] == a.shape[-2]:
            attns.append(a.detach())

    for mod in model.modules():
        if mod.__class__.__name__ == "Attention" and hasattr(mod, "attn_drop"):
            if getattr(mod, "fused_attn", False):
                mod.fused_attn = False
                toggled.append(mod)
            hooks.append(mod.attn_drop.register_forward_hook(grab))

    if not hooks:
        return None
    try:
        with torch.no_grad():
            model(img, meta)
    finally:
        for h in hooks:
            h.remove()
        for m in toggled:
            m.fused_attn = True
    if not attns:
        return None
    result = torch.eye(attns[0].size(-1), device=attns[0].device)
    for a in attns:
        a = a.mean(1)[0]
        a = a + torch.eye(a.size(-1), device=a.device)
        a = a / a.sum(-1, keepdim=True)
        result = a @ result
    # Row 0 is the CLS query. Drop the CLS column, then keep only the leading
    # square block of patch tokens: early-fusion models append metadata tokens
    # after the patches, which are not part of the spatial grid.
    cls = result[0, 1:]
    side = int(cls.numel() ** 0.5)
    if side < 2:
        return None
    cls = cls[:side * side]
    m = cls.reshape(side, side)[None, None].float()
    m = F.interpolate(m, size=(img.shape[-1], img.shape[-1]),
                      mode="bilinear", align_corners=False)[0, 0]
    m = m - m.min()
    return (m / (m.max() + 1e-8)).cpu().numpy()


# ---------------------------------------------------------------------------
#  Tabular attribution
# ---------------------------------------------------------------------------
def shap_metadata(model, meta_background: np.ndarray, meta_sample: np.ndarray,
                  image_fixed: torch.Tensor, device, nsamples: int = 200
                  ) -> Optional[np.ndarray]:
    """
    SHAP over the metadata branch with the image held fixed at the selected
    case, so the attribution answers "given this lesion, what did the clinical
    fields contribute?".
    """
    try:
        import shap
    except ImportError:
        return None

    def f(x: np.ndarray) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(x), 32):
                chunk = torch.tensor(x[i:i + 32], dtype=torch.float32, device=device)
                img = image_fixed.repeat(len(chunk), 1, 1, 1)
                out.append(torch.sigmoid(model(img, chunk)).cpu().numpy())
        return np.concatenate(out).ravel()

    # k must be comfortably smaller than the number of background rows:
    # asking k-means for as many clusters as points yields a degenerate
    # summary whose weight vector does not line up with the data matrix,
    # which KernelExplainer rejects with
    # "# of weights must match data matrix!".
    n_bg = len(meta_background)
    k = max(1, min(10, n_bg // 2))
    bg = shap.kmeans(np.asarray(meta_background, dtype=float), k) if n_bg > 2 \
        else np.asarray(meta_background, dtype=float)
    expl = shap.KernelExplainer(f, bg)
    sample = np.asarray(meta_sample, dtype=float).reshape(1, -1)
    vals = expl.shap_values(sample, nsamples=nsamples, silent=True)
    if isinstance(vals, list):          # older shap returns one array per class
        vals = vals[-1]
    return np.asarray(vals).ravel()


@torch.no_grad()
def permutation_importance(model, loader, device, meta_cols: List[str],
                           cfg: UnifiedConfig, n_repeats: int = 3) -> Dict[str, float]:
    """Global metadata importance: AUC drop when a column is shuffled."""
    from sklearn.metrics import roc_auc_score

    imgs, metas, ys = [], [], []
    for img, meta, y in loader:
        imgs.append(img)
        metas.append(meta)
        ys.append(y)
    img = torch.cat(imgs)
    meta = torch.cat(metas)
    y = torch.cat(ys).numpy()

    def score(mt: torch.Tensor) -> float:
        probs = []
        for i in range(0, len(img), 64):
            b = img[i:i + 64].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=cfg.amp and device.type == "cuda"):
                probs.append(torch.sigmoid(model(b, mt[i:i + 64].to(device))).float().cpu())
        return roc_auc_score(y, torch.cat(probs).numpy())

    base = score(meta)
    rng = np.random.default_rng(cfg.seed)
    out: Dict[str, float] = {}
    for j, col in enumerate(meta_cols):
        drops = []
        for _ in range(n_repeats):
            mt = meta.clone()
            mt[:, j] = mt[rng.permutation(len(mt)), j]
            drops.append(base - score(mt))
        out[col] = float(np.mean(drops))
    return out


# ---------------------------------------------------------------------------
#  Figure rendering -- ONE case per figure
# ---------------------------------------------------------------------------
def _overlay(ax, base: np.ndarray, heat: np.ndarray, title: str, cmap="jet"):
    ax.imshow(base)
    ax.imshow(heat, cmap=cmap, alpha=0.45)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def render_xai_panel(model, cfg: UnifiedConfig, dataset, case_idx: int,
                     device, out_dir: str, display_name: str,
                     case_id: str = "") -> Dict[str, str]:
    """
    Single-case explainability panel:
        original | Grad-CAM | Grad-CAM++ | Integrated Gradients | rollout/occlusion
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    img_t, meta_t, label = dataset[case_idx]
    img = img_t[None].to(device)
    meta = meta_t[None].to(device)
    base = denormalize(img_t, cfg.mean, cfg.std)

    with torch.no_grad():
        prob = torch.sigmoid(model(img, meta)).item()

    maps: Dict[str, np.ndarray] = {}
    layer = find_cam_layer(model, img, meta)
    if layer is not None:
        cam = GradCAM(model, layer)
        try:
            maps["Grad-CAM"] = cam(img, meta, plus_plus=False)
            maps["Grad-CAM++"] = cam(img, meta, plus_plus=True)
        except Exception:
            pass
        finally:
            cam.remove()

    try:
        maps["Integrated Gradients"] = integrated_gradients(model, img, meta, cfg.ig_steps)
    except Exception:
        pass

    roll = attention_rollout(model, img, meta)
    if roll is not None:
        maps["Attention Rollout"] = roll
    elif "Grad-CAM" not in maps:
        try:
            maps["Occlusion"] = occlusion(model, img, meta)
        except Exception:
            pass

    n = 1 + len(maps)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.4))
    axes = np.atleast_1d(axes)
    axes[0].imshow(base)
    truth = "melanoma" if float(label) == 1 else "non-melanoma"
    axes[0].set_title(f"{case_id or 'test case'}\ntrue: {truth} | p={prob:.3f}", fontsize=9)
    axes[0].axis("off")
    for ax, (name, heat) in zip(axes[1:], maps.items()):
        _overlay(ax, base, heat, name)
    fig.suptitle(f"{display_name} - single-case explanation", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, "xai_panel.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {"panel": path, "prob": prob, "label": float(label), "methods": list(maps)}


def render_shap_bar(values: np.ndarray, meta_cols: List[str], out_dir: str,
                    display_name: str, top_k: int = 10) -> str:
    os.makedirs(out_dir, exist_ok=True)
    order = np.argsort(np.abs(values))[::-1][:top_k][::-1]
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(order) + 1.4))
    colors = ["#c0392b" if values[i] > 0 else "#2471a3" for i in order]
    ax.barh(range(len(order)), values[order], color=colors)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([meta_cols[i] for i in order], fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("SHAP value (impact on melanoma probability)", fontsize=9)
    ax.set_title(f"{display_name} - metadata attribution (single case)", fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, "shap_metadata.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path
