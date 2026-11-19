"""
Unified explainability suite.

What changed relative to the first version, and why
---------------------------------------------------
* Grad-CAM was hooked to "the deepest Conv2d". In ViT that is the patch
  embedding, so the maps showed the dermoscope ring, not the lesion. The
  target is now chosen per architecture: the input of the last transformer
  block (`blocks[-1].norm1`) for ViTs, the final backbone feature map for
  CNN / hierarchical backbones.
* Every gradient method explains the PREDICTED class (sign-flipped logit for a
  benign prediction), so TN / FN cases are explained correctly.
* Grad-CAM(++) is averaged over the same 8 dihedral views used for TTA
  prediction (the explanation describes the prediction that is reported).
* Integrated Gradients uses a blurred baseline (a black image is in-distribution
  for vignetted dermoscopy, so it is a poor "absence of signal"), a midpoint
  Riemann sum and SmoothGrad noise draws.
* Occlusion fills with the blurred image instead of zeros, for the same reason.
* Attention rollout is gradient-weighted (Chefer et al., 2021), which is
  class-specific; plain rollout is not.
* Explanations are SCORED, not only drawn:
    - faithfulness: deletion / insertion AUC (Petsiuk et al., 2018)
    - localisation: share of attribution energy inside the lesion (automatic
      Otsu segmentation) and outside the dermoscope field of view
    - a random map gives the chance level for every score.
* Metadata attribution is EXACT grouped Shapley: one-hot columns of one
  clinical variable form one player, 2^G coalitions are enumerated, and the
  image is held fixed. Global importance is grouped permutation importance.

Figure policy: ONE representative case per main panel (highest-confidence
correctly classified melanoma), plus a TP/TN/FP/FN gallery.
"""
from __future__ import annotations

import itertools
import math
import os
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import UnifiedConfig, META_GROUPS


# ---------------------------------------------------------------------------
#  Small utilities
# ---------------------------------------------------------------------------
def denormalize(img: torch.Tensor, mean: List[float], std: List[float]) -> np.ndarray:
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    out = (img.detach().cpu().float() * s + m).clamp(0, 1)
    return out.permute(1, 2, 0).numpy()


def dihedral(x: torch.Tensor, k: int) -> torch.Tensor:
    if k >= 4:
        x = x.flip(-1)
    return torch.rot90(x, k % 4, dims=(-2, -1)) if k % 4 else x


def inv_dihedral(x: torch.Tensor, k: int) -> torch.Tensor:
    if k % 4:
        x = torch.rot90(x, -(k % 4), dims=(-2, -1))
    return x.flip(-1) if k >= 4 else x


def _gauss_kernel(sigma: float, device) -> torch.Tensor:
    r = max(1, int(3 * sigma))
    t = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    return k / k.sum()


def blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of (B, C, H, W) with reflection padding."""
    k = _gauss_kernel(sigma, x.device)
    c = x.shape[1]
    r = (len(k) - 1) // 2
    x = F.pad(x, (r, r, r, r), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(x, k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def normalize_map(m: np.ndarray, pct: float = 99.5) -> np.ndarray:
    m = np.nan_to_num(np.asarray(m, dtype=np.float32))
    m = np.clip(m, 0, None)
    hi = np.percentile(m, pct) if m.max() > 0 else 0.0
    if hi <= 0:
        hi = m.max() + 1e-8
    return np.clip(m / (hi + 1e-12), 0, 1)


def _upsample(m: torch.Tensor, size: int) -> torch.Tensor:
    m = F.interpolate(m[None, None].float(), size=(size, size), mode="bicubic",
                      align_corners=False)[0, 0]
    return m.clamp_(min=0)


@contextmanager
def frozen(model: nn.Module):
    """Parameters do not need gradients for any attribution method here."""
    flags = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    was_training = model.training
    model.eval()
    try:
        yield model
    finally:
        for p, f in zip(model.parameters(), flags):
            p.requires_grad_(f)
        model.train(was_training)


def _target(model, x, meta, sign: float) -> torch.Tensor:
    """Logit of the class being explained: +logit (melanoma) or -logit (benign)."""
    return sign * model(x, meta).float()


# ---------------------------------------------------------------------------
#  Case selection -- deterministic
# ---------------------------------------------------------------------------
def select_case(probs: np.ndarray, labels: np.ndarray, want_class: int = 1,
                threshold: float = 0.5) -> int:
    """Highest-confidence correctly-classified case of `want_class`."""
    pred = (probs >= threshold).astype(int)
    correct = (labels == want_class) & (pred == want_class)
    if not correct.any():
        return int(np.argmax(probs if want_class == 1 else -probs))
    idx = np.where(correct)[0]
    score = probs[idx] if want_class == 1 else -probs[idx]
    return int(idx[np.argmax(score)])


def select_gallery(probs: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, int]:
    """Most confident TP, TN, FP and FN (the confident errors are the informative ones)."""
    labels = labels.astype(int)
    pred = (probs >= threshold).astype(int)
    out: Dict[str, int] = {}
    for name, yl, pl, conf in (("TP", 1, 1, probs), ("TN", 0, 0, -probs),
                               ("FP", 0, 1, probs), ("FN", 1, 0, -probs)):
        idx = np.where((labels == yl) & (pred == pl))[0]
        if len(idx):
            out[name] = int(idx[np.argmax(conf[idx])])
    return out


def select_eval_cases(probs: np.ndarray, labels: np.ndarray, threshold: float,
                      n: int) -> List[int]:
    """n cases spread evenly over each confusion quadrant (deterministic)."""
    labels = labels.astype(int)
    pred = (probs >= threshold).astype(int)
    per = max(1, n // 4)
    chosen: List[int] = []
    for yl, pl in ((1, 1), (0, 0), (0, 1), (1, 0)):
        idx = np.where((labels == yl) & (pred == pl))[0]
        if not len(idx):
            continue
        idx = idx[np.argsort(probs[idx])]
        take = np.unique(np.linspace(0, len(idx) - 1, min(per, len(idx))).round().astype(int))
        chosen += idx[take].tolist()
    return chosen


# ---------------------------------------------------------------------------
#  Lesion / field-of-view masks (for localisation scores)
# ---------------------------------------------------------------------------
def lesion_and_fov(raw: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Automatic masks from the uint8 image.
      fov    : pixels inside the dermoscope field of view (not the black ring)
      lesion : central dark connected region by Otsu inside the FOV, or None
               when segmentation is implausible (too small / too large).
    Used only to SCORE explanations; never seen by the model.
    """
    from scipy import ndimage as ndi
    from skimage.filters import threshold_otsu, gaussian
    from skimage.morphology import binary_opening, disk

    g = raw.astype(np.float32).mean(-1)
    h, w = g.shape
    fov = g > 30
    fov = ndi.binary_fill_holes(binary_opening(fov, disk(3)))
    if fov.mean() < 0.3:
        fov = np.ones_like(fov, dtype=bool)

    gb = gaussian(g, sigma=3, preserve_range=True)
    # specular flash / glare would drag the Otsu threshold up: ignore it
    valid = fov & (gb < 235)
    inner = ndi.binary_erosion(fov, iterations=8)
    vals = gb[valid & inner]
    if vals.size < 100 or vals.std() < 2:
        return None, fov
    t = threshold_otsu(vals)
    les = (gb < t) & inner
    les = ndi.binary_fill_holes(binary_opening(les, disk(3)))
    lab, n = ndi.label(les)
    if n == 0:
        return None, fov
    yy, xx = np.mgrid[0:h, 0:w]
    edge = np.zeros((h, w), dtype=bool)
    edge[:4], edge[-4:], edge[:, :4], edge[:, -4:] = True, True, True, True
    edge |= ~ndi.binary_erosion(fov, iterations=10)
    best, best_s = 0, -1.0
    for i in range(1, n + 1):
        comp = lab == i
        a = comp.sum()
        if a < 0.005 * h * w:
            continue
        cy, cx = yy[comp].mean() / h - 0.5, xx[comp].mean() / w - 0.5
        s = a * math.exp(-6 * (cy * cy + cx * cx))
        # components hugging the image / FOV border are shading, not lesion
        touch = (comp & edge).sum() / max(1, a)
        s *= math.exp(-8 * touch)
        # contrast with the surrounding skin
        ring = ndi.binary_dilation(comp, iterations=8) & ~comp & valid
        if ring.any():
            s *= max(0.05, (gb[ring].mean() - gb[comp].mean()) / 40.0)
        if s > best_s:
            best, best_s = i, s
    if best == 0:
        return None, fov
    les = lab == best
    frac = les.mean()
    touch = (les & edge).sum() / max(1, les.sum())
    if frac < 0.005 or frac > 0.70 or touch > 0.25:
        return None, fov
    # a real lesion is a compact blob; ragged skin-texture masks are rejected
    from skimage.measure import regionprops
    solidity = regionprops(les.astype(np.uint8))[0].solidity
    if solidity < 0.75:
        return None, fov
    return les, fov


# ---------------------------------------------------------------------------
#  Grad-CAM / Grad-CAM++ on the correct layer per architecture
# ---------------------------------------------------------------------------
def cam_spec(model: nn.Module) -> Optional[Dict]:
    """Where to read activations for CAM, and how to turn them into a grid."""
    from models import TokenBackbone
    vit = getattr(model, "vit", None)
    if vit is not None and hasattr(vit, "blocks"):
        return {"module": vit.blocks[-1].norm1, "kind": "seq",
                "prefix": int(getattr(vit, "num_prefix_tokens", 1)),
                "grid": tuple(vit.patch_embed.grid_size), "layer": "vit.blocks[-1].norm1"}
    bb = getattr(model, "backbone", None)
    if isinstance(bb, TokenBackbone):
        net = bb.net
        if bb.kind == "timm" and getattr(bb, "layout", "") == "nlc" and hasattr(net, "blocks"):
            return {"module": net.blocks[-1].norm1, "kind": "seq",
                    "prefix": int(getattr(net, "num_prefix_tokens", 1)),
                    "grid": tuple(bb.hw), "layer": "backbone.blocks[-1].norm1"}
        return {"module": bb, "kind": "backbone", "has_cls": bool(bb.has_cls),
                "layer": f"backbone output ({bb.kind})"}
    return None


class _Capture:
    def __init__(self, spec: Dict):
        self.spec = spec
        self.act: Optional[torch.Tensor] = None
        self.grad: Optional[torch.Tensor] = None
        self.hw = None
        self.h = spec["module"].register_forward_hook(self._hook)

    def _hook(self, _m, _i, o):
        if isinstance(o, (tuple, list)):
            t = o[0]
            if len(o) > 1 and isinstance(o[1], (tuple, list)):
                self.hw = tuple(o[1])
        else:
            t = o
        if not (isinstance(t, torch.Tensor) and t.requires_grad):
            return
        self.act = t
        t.register_hook(self._save)

    def _save(self, g):
        self.grad = g

    def remove(self):
        self.h.remove()

    def grid(self) -> Tuple[torch.Tensor, torch.Tensor]:
        a, g = self.act, self.grad
        if a is None or g is None:
            raise RuntimeError("CAM layer produced no activation/gradient")
        if a.ndim == 4:
            return a.detach().float(), g.detach().float()
        if self.spec["kind"] == "seq":
            gh, gw = self.spec["grid"]
            p = self.spec["prefix"]
            a, g = a[:, p:p + gh * gw], g[:, p:p + gh * gw]
        else:
            if self.spec.get("has_cls"):
                a, g = a[:, 1:], g[:, 1:]
            gh, gw = self.hw if self.hw else (int(math.sqrt(a.shape[1])),) * 2
        b, n, c = a.shape
        a = a.transpose(1, 2).reshape(b, c, gh, gw)
        g = g.transpose(1, 2).reshape(b, c, gh, gw)
        return a.detach().float(), g.detach().float()


def gradcam(model, img, meta, sign: float, spec: Dict, views: int = 8
            ) -> Tuple[np.ndarray, np.ndarray]:
    """Grad-CAM and Grad-CAM++, averaged over dihedral views (TTA-consistent)."""
    cap = _Capture(spec)
    cams, pps = [], []
    try:
        for k in range(views):
            x = dihedral(img, k).detach().clone().requires_grad_(True)
            out = _target(model, x, meta, sign).sum()
            out.backward()
            a, g = cap.grid()
            cam = F.relu((g.mean(dim=(2, 3), keepdim=True) * a).sum(1))[0]
            g2, g3 = g.pow(2), g.pow(3)
            denom = 2 * g2 + (a * g3).sum(dim=(2, 3), keepdim=True)
            alpha = g2 / torch.where(denom != 0, denom, torch.ones_like(denom))
            w = (alpha * F.relu(g)).sum(dim=(2, 3), keepdim=True)
            pp = F.relu((w * a).sum(1))[0]
            cams.append(inv_dihedral(cam, k))
            pps.append(inv_dihedral(pp, k))
            cap.act = cap.grad = None
    finally:
        cap.remove()
    size = img.shape[-1]
    cam = _upsample(torch.stack(cams).mean(0), size).cpu().numpy()
    pp = _upsample(torch.stack(pps).mean(0), size).cpu().numpy()
    return normalize_map(cam, 100), normalize_map(pp, 100)


# ---------------------------------------------------------------------------
#  Integrated Gradients (blurred baseline + SmoothGrad)
# ---------------------------------------------------------------------------
def integrated_gradients(model, img, meta, sign: float, steps: int = 32,
                         noise: int = 4, chunk: int = 8, noise_std: float = 0.15
                         ) -> Tuple[np.ndarray, float]:
    """Returns (map, completeness_error) with error = |sum(attr) - (f(x)-f(x0))| / |f(x)-f(x0)|."""
    base = blur(img, 12.0)
    alphas = (torch.arange(steps, device=img.device, dtype=torch.float32) + 0.5) / steps
    total = torch.zeros_like(img)
    gen = torch.Generator(device=img.device).manual_seed(0)
    for n in range(max(1, noise)):
        xi = img if n == 0 else img + noise_std * torch.randn(
            img.shape, device=img.device, generator=gen)
        diff = xi - base
        for i in range(0, steps, chunk):
            a = alphas[i:i + chunk].view(-1, 1, 1, 1)
            x = (base + a * diff).requires_grad_(True)
            out = _target(model, x, meta.expand(len(a), -1), sign).sum()
            gx, = torch.autograd.grad(out, x)
            total += gx.sum(0, keepdim=True) * diff
    attr = total / (steps * max(1, noise))
    with torch.no_grad():
        delta = float(_target(model, img, meta, sign) - _target(model, base, meta, sign))
    comp_err = abs(float(attr.sum()) - delta) / (abs(delta) + 1e-6)
    m = attr[0].sum(0, keepdim=True)[None]               # signed, (1,1,H,W)
    m = blur(F.relu(m), 2.0)[0, 0]
    return normalize_map(m.cpu().numpy()), comp_err


# ---------------------------------------------------------------------------
#  Occlusion (blur fill, per-pixel averaging)
# ---------------------------------------------------------------------------
@torch.no_grad()
def occlusion(model, img, meta, sign: float, patch: int = 32, stride: int = 16,
              chunk: int = 32) -> np.ndarray:
    size = img.shape[-1]
    fill = blur(img, 12.0)
    base = torch.sigmoid(_target(model, img, meta, sign)).item()
    pos = [(y, x) for y in range(0, size - patch + 1, stride)
           for x in range(0, size - patch + 1, stride)]
    heat = torch.zeros(size, size, device=img.device)
    cnt = torch.zeros(size, size, device=img.device)
    for i in range(0, len(pos), chunk):
        part = pos[i:i + chunk]
        batch = img.repeat(len(part), 1, 1, 1)
        for j, (y, x) in enumerate(part):
            batch[j, :, y:y + patch, x:x + patch] = fill[0, :, y:y + patch, x:x + patch]
        p = torch.sigmoid(_target(model, batch, meta.expand(len(part), -1), sign))
        for j, (y, x) in enumerate(part):
            heat[y:y + patch, x:x + patch] += base - p[j]
            cnt[y:y + patch, x:x + patch] += 1
    heat = heat / cnt.clamp(min=1)
    return normalize_map(blur(F.relu(heat)[None, None], 3.0)[0, 0].cpu().numpy())


# ---------------------------------------------------------------------------
#  Gradient-weighted attention rollout (ViT-family)
# ---------------------------------------------------------------------------
def _vit_layout(model) -> Optional[Tuple[int, Tuple[int, int]]]:
    vit = getattr(model, "vit", None)
    if vit is not None and hasattr(vit, "patch_embed"):
        return int(getattr(vit, "num_prefix_tokens", 1)), tuple(vit.patch_embed.grid_size)
    bb = getattr(model, "backbone", None)
    if bb is not None and getattr(bb, "layout", "") == "nlc" and hasattr(bb.net, "blocks"):
        return int(getattr(bb.net, "num_prefix_tokens", 1)), tuple(bb.hw)
    return None


def grad_rollout(model, img, meta, sign: float) -> Optional[np.ndarray]:
    """Chefer et al. (2021): R <- R + E_h[(dy/dA * A)^+] R over all blocks; CLS row."""
    lay = _vit_layout(model)
    if lay is None:
        return None
    prefix, (gh, gw) = lay
    attns: List[torch.Tensor] = []
    hooks, toggled = [], []

    def grab(_m, inp, _o):
        a = inp[0]
        if isinstance(a, torch.Tensor) and a.ndim == 4 and a.requires_grad:
            a.retain_grad()
            attns.append(a)

    for mod in model.modules():
        if type(mod).__name__ == "Attention" and hasattr(mod, "attn_drop") and hasattr(mod, "qkv"):
            if getattr(mod, "fused_attn", False):
                mod.fused_attn = False
                toggled.append(mod)
            hooks.append(mod.attn_drop.register_forward_hook(grab))
    if not hooks:
        return None
    try:
        x = img.detach().clone().requires_grad_(True)
        _target(model, x, meta, sign).sum().backward()
    finally:
        for h in hooks:
            h.remove()
        for m in toggled:
            m.fused_attn = True
    attns = [a for a in attns if a.grad is not None]
    if not attns:
        return None
    n = attns[0].shape[-1]
    r = torch.eye(n, device=img.device)
    for a in attns:
        if a.shape[-1] != n:
            continue
        cam = (a.grad * a.detach())[0].float().clamp(min=0).mean(0)
        r = r + cam @ r
    row = r[0, prefix:prefix + gh * gw]
    if row.numel() != gh * gw:
        return None
    m = _upsample(row.reshape(gh, gw), img.shape[-1]).cpu().numpy()
    return normalize_map(m, 100)


# ---------------------------------------------------------------------------
#  Model-specific spatial maps (fusion mechanisms)
# ---------------------------------------------------------------------------
@torch.no_grad()
def model_specific_maps(model, img, meta) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    from models import CrossAttentionFusion, BioViLLCSA, DaViTGated
    maps: Dict[str, np.ndarray] = {}
    stats: Dict[str, float] = {}
    size = img.shape[-1]
    try:
        if isinstance(model, CrossAttentionFusion):
            model(img, meta, keep_attn=True)
            gh, gw = model.backbone.hw
            if model.attn_maps:
                a = model.attn_maps[-1][0].float().mean(0)          # (Q, K)
                row = a[0, 1:]
                if row.numel() == gh * gw:
                    maps["Clinical->image attention"] = normalize_map(
                        _upsample(row.reshape(gh, gw), size).cpu().numpy(), 100)
            if model.gates:
                g = model.gates[-1][0].float()
                if g.numel() == gh * gw:
                    maps["LACA gate"] = normalize_map(
                        _upsample(g.reshape(gh, gw), size).cpu().numpy(), 100)
            model.attn_maps, model.gates = [], []
        elif isinstance(model, BioViLLCSA):
            model(img, meta, keep_attn=True)
            a = model.fusion.last_attn
            if a is not None:
                row = a[0].float().mean(0)[0]                        # (P,)
                gh, gw = model.backbone.hw
                if row.numel() == gh * gw:
                    maps["LCSA sparse attention"] = normalize_map(
                        _upsample(row.reshape(gh, gw), size).cpu().numpy(), 100)
            stats["lcsa_gate_cross"] = float(model.last_gate[0, 0])
        elif isinstance(model, DaViTGated):
            model(img, meta)
            g = model.last_gates
            stats["gate_image_mean"] = float(g["gate_img"].float().mean())
            stats["gate_meta_mean"] = float(g["gate_meta"].float().mean())
    except Exception as exc:  # noqa: BLE001
        stats["model_specific_error"] = 1.0
        print(f"    [warn] model-specific map failed: {type(exc).__name__}: {exc}")
    return maps, stats


# ---------------------------------------------------------------------------
#  Scores: faithfulness and localisation
# ---------------------------------------------------------------------------
@torch.no_grad()
def deletion_insertion(model, img, meta, sign: float, heat: np.ndarray,
                       steps: int = 20, chunk: int = 21) -> Tuple[float, float]:
    """
    Deletion: replace the most-attributed pixels by the blurred image first;
    a faithful map makes P(target) fall fast (low AUC is good).
    Insertion: start from the blurred image and restore the most-attributed
    pixels first; a faithful map makes P(target) rise fast (high AUC is good).
    """
    size = img.shape[-1]
    fill = blur(img, 12.0)
    rng = np.random.default_rng(0)
    order = np.argsort(-(heat.ravel() + 1e-6 * rng.random(heat.size)))
    order = torch.from_numpy(order.copy()).to(img.device)
    fracs = np.linspace(0, 1, steps + 1)
    masks = []
    for f in fracs:
        m = torch.zeros(size * size, device=img.device)
        m[order[:int(round(f * size * size))]] = 1
        masks.append(m.view(1, size, size))
    masks = torch.stack(masks)                                     # (S, 1, H, W)
    dels = img * (1 - masks) + fill * masks
    inss = fill * (1 - masks) + img * masks
    out = []
    for batch in (dels, inss):
        ps = []
        for i in range(0, len(batch), chunk):
            b = batch[i:i + chunk]
            ps.append(torch.sigmoid(_target(model, b, meta.expand(len(b), -1), sign)))
        p = torch.cat(ps).cpu().numpy()
        out.append(float(np.trapz(p, fracs)))
    return out[0], out[1]


def localisation(heat: np.ndarray, lesion: Optional[np.ndarray], fov: np.ndarray
                 ) -> Dict[str, float]:
    tot = heat.sum() + 1e-8
    out = {"energy_outside_fov": float(heat[~fov].sum() / tot)}
    if lesion is not None:
        e = float(heat[lesion].sum() / tot)
        out["energy_in_lesion"] = e
        out["lesion_area"] = float(lesion.mean())
        out["lesion_focus_gain"] = e / max(1e-6, float(lesion.mean()))
    return out


# ---------------------------------------------------------------------------
#  All maps for one case
# ---------------------------------------------------------------------------
def explain_case(model, img: torch.Tensor, meta: torch.Tensor, sign: float,
                 cfg: UnifiedConfig, spec: Optional[Dict], fast: bool = False
                 ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    maps: Dict[str, np.ndarray] = {}
    info: Dict[str, float] = {}
    if spec is not None:
        try:
            maps["Grad-CAM"], pp = gradcam(model, img, meta, sign, spec, views=8)
            maps["Grad-CAM++"] = pp
        except Exception as exc:  # noqa: BLE001
            print(f"    [warn] Grad-CAM failed: {type(exc).__name__}: {exc}")
    try:
        maps["Integrated Gradients"], info["ig_completeness_error"] = integrated_gradients(
            model, img, meta, sign, steps=cfg.ig_steps, noise=1 if fast else cfg.ig_noise)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        maps["Integrated Gradients"], info["ig_completeness_error"] = integrated_gradients(
            model, img, meta, sign, steps=cfg.ig_steps, noise=1, chunk=2)
    try:
        maps["Occlusion"] = occlusion(model, img, meta, sign)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        maps["Occlusion"] = occlusion(model, img, meta, sign, chunk=4)
    try:
        r = grad_rollout(model, img, meta, sign)
        if r is not None:
            maps["Attention (grad-rollout)"] = r
    except Exception as exc:  # noqa: BLE001
        print(f"    [warn] rollout failed: {type(exc).__name__}: {exc}")
    ms, st = model_specific_maps(model, img, meta)
    maps.update(ms)
    info.update(st)
    return maps, info


# ---------------------------------------------------------------------------
#  Metadata: exact grouped Shapley + grouped permutation importance
# ---------------------------------------------------------------------------
def meta_groups(meta_cols: List[str]) -> List[Tuple[str, List[int]]]:
    groups, used = [], set()
    for name, prefixes in META_GROUPS:
        idx = [i for i, c in enumerate(meta_cols) if any(c.startswith(p) for p in prefixes)]
        if idx:
            groups.append((name, idx))
            used |= set(idx)
    for i, c in enumerate(meta_cols):
        if i not in used:
            groups.append((c, [i]))
    return groups


def group_value_label(name: str, cols: List[str], idx: List[int], x: np.ndarray) -> str:
    on = [cols[i] for i in idx if x[i] > 0.5 and not cols[i].endswith("_scaled")]
    if len(idx) == 1 and len(on) <= 1:
        return f"{name} = {'yes' if x[idx[0]] > 0.5 else 'no'}"
    if not on:
        return name
    val = on[0]
    for p in ("age_group_", "sex_", "anatom_site_general_", "dermoscopic_type_",
              "history_of_mm_", "diagnosis_confirm_type_"):
        if val.startswith(p):
            val = val[len(p):]
    return f"{name} = {val}"


@torch.no_grad()
def grouped_shapley(model, img: torch.Tensor, x: np.ndarray, background: np.ndarray,
                    groups: List[Tuple[str, List[int]]], device, temperature: float = 1.0,
                    chunk: int = 64, amp: bool = True) -> Tuple[np.ndarray, float, float]:
    """
    Exact Shapley values of the melanoma probability over G metadata groups,
    interventional (background rows supply the absent groups), image fixed.
    Returns (phi[G], base_value, full_value); sum(phi) == full - base exactly.
    """
    g = len(groups)
    k = len(background)
    coalitions = list(itertools.product([0, 1], repeat=g))
    rows = np.repeat(background[None], len(coalitions), axis=0).copy()   # (C, K, D)
    for ci, s in enumerate(coalitions):
        for gi, on in enumerate(s):
            if on:
                rows[ci][:, groups[gi][1]] = x[groups[gi][1]]
    rows = torch.tensor(rows.reshape(-1, x.shape[0]), dtype=torch.float32, device=device)
    probs = []
    dt = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    for i in range(0, len(rows), chunk):
        r = rows[i:i + chunk]
        with torch.autocast("cuda", dtype=dt, enabled=amp and device.type == "cuda"):
            z = model(img.expand(len(r), -1, -1, -1), r).float()
        probs.append(torch.sigmoid(z / temperature))
    v = torch.cat(probs).view(len(coalitions), k).mean(1).cpu().numpy()
    vi = {s: v[i] for i, s in enumerate(coalitions)}
    phi = np.zeros(g)
    fact = math.factorial
    for s in coalitions:
        sz = sum(s)
        for gi in range(g):
            if s[gi]:
                continue
            w = fact(sz) * fact(g - sz - 1) / fact(g)
            s2 = list(s)
            s2[gi] = 1
            phi[gi] += w * (vi[tuple(s2)] - vi[s])
    return phi, float(vi[tuple([0] * g)]), float(vi[tuple([1] * g)])


@torch.no_grad()
def grouped_permutation_importance(model, ds, idx: np.ndarray,
                                   groups: List[Tuple[str, List[int]]], device,
                                   seed: int = 42, repeats: int = 2, chunk: int = 64
                                   ) -> Dict[str, float]:
    """Global: test-AUC drop when a clinical variable (all its columns) is shuffled."""
    from sklearn.metrics import roc_auc_score
    imgs = torch.stack([ds[int(i)][0] for i in idx])
    meta = torch.tensor(ds.meta[idx], dtype=torch.float32)
    y = ds.labels[idx]
    dt = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    def score(mt: torch.Tensor) -> float:
        out = []
        for i in range(0, len(imgs), chunk):
            with torch.autocast("cuda", dtype=dt, enabled=device.type == "cuda"):
                out.append(model(imgs[i:i + chunk].to(device),
                                 mt[i:i + chunk].to(device)).float().cpu())
        return float(roc_auc_score(y, torch.cat(out).numpy()))

    base = score(meta)
    rng = np.random.default_rng(seed)
    res: Dict[str, float] = {}
    all_idx = sorted({j for _, ix in groups for j in ix})
    for name, ix in groups + [("ALL metadata", all_idx)]:
        drops = []
        for _ in range(repeats):
            mt = meta.clone()
            perm = torch.from_numpy(rng.permutation(len(mt)))
            mt[:, ix] = meta[perm][:, ix]
            drops.append(base - score(mt))
        res[name] = float(np.mean(drops))
    res["_base_auc"] = base
    return res


# ---------------------------------------------------------------------------
#  Rendering
# ---------------------------------------------------------------------------
def _overlay(ax, base: np.ndarray, heat: np.ndarray, title: str,
             lesion: Optional[np.ndarray] = None, cmap: str = "jet"):
    ax.imshow(base)
    rgba = plt.get_cmap(cmap)(heat)
    rgba[..., 3] = np.clip(heat, 0, 1) ** 0.9 * 0.62
    ax.imshow(rgba)
    if lesion is not None:
        ax.contour(lesion.astype(float), levels=[0.5], colors="white",
                   linewidths=0.9, linestyles="--")
    ax.set_title(title, fontsize=8.5)
    ax.axis("off")


def _fmt_scores(s: Dict[str, float]) -> str:
    parts = []
    if "energy_in_lesion" in s:
        parts.append(f"lesion {100 * s['energy_in_lesion']:.0f}%")
    if "deletion" in s:
        parts.append(f"del {s['deletion']:.2f} ins {s['insertion']:.2f}")
    return " | ".join(parts)


def render_panel(base, maps, scores, lesion, title_left, suptitle, path):
    n = 1 + len(maps)
    fig, axes = plt.subplots(1, n, figsize=(2.9 * n, 3.45))
    axes = np.atleast_1d(axes)
    axes[0].imshow(base)
    if lesion is not None:
        axes[0].contour(lesion.astype(float), levels=[0.5], colors="white",
                        linewidths=0.9, linestyles="--")
    axes[0].set_title(title_left, fontsize=8.5)
    axes[0].axis("off")
    for ax, (name, heat) in zip(axes[1:], maps.items()):
        _overlay(ax, base, heat, f"{name}\n{_fmt_scores(scores.get(name, {}))}", lesion)
    fig.suptitle(suptitle, fontsize=10.5, y=1.03)
    fig.text(0.5, 0.01, "dashed = automatic lesion outline | lesion = share of attribution "
             "inside it | del (lower better) / ins (higher better) = faithfulness AUC",
             ha="center", fontsize=7, color="#444")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def render_gallery(rows: List[Dict], display: str, path: str,
                   methods: Tuple[str, ...]) -> None:
    ncol = 1 + len(methods)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(2.7 * ncol, 2.85 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, row in enumerate(rows):
        ax = axes[r, 0]
        ax.imshow(row["base"])
        if row["lesion"] is not None:
            ax.contour(row["lesion"].astype(float), levels=[0.5], colors="white",
                       linewidths=0.8, linestyles="--")
        col = "#1e8449" if row["kind"] in ("TP", "TN") else "#c0392b"
        ax.set_title(f"{row['kind']}  {row['case_id']}\nGT {row['gt']} | pred {row['pred']} "
                     f"| p={row['prob']:.3f}", fontsize=8, color=col)
        ax.axis("off")
        for c, m in enumerate(methods, start=1):
            ax = axes[r, c]
            if m in row["maps"]:
                _overlay(ax, row["base"], row["maps"][m],
                         f"{m}\n{_fmt_scores(row['scores'].get(m, {}))}", row["lesion"])
            else:
                ax.axis("off")
    fig.suptitle(f"{display} - explanations of the predicted class (TP / TN / FP / FN)",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_meta(local: List[Tuple[str, float]], base_v: float, full_v: float,
                glob: Dict[str, float], display: str, case_id: str, path: str) -> None:
    g = [(k, v) for k, v in glob.items() if not k.startswith("_")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 0.42 * max(len(local), len(g)) + 2.0))
    lbl = [l for l, _ in local][::-1]
    val = np.array([v for _, v in local][::-1])
    a1.barh(range(len(val)), val, color=["#c0392b" if v > 0 else "#2471a3" for v in val])
    a1.set_yticks(range(len(val)))
    a1.set_yticklabels(lbl, fontsize=8)
    a1.axvline(0, color="k", lw=0.8)
    a1.set_xlabel("exact grouped Shapley value (change in P(melanoma))", fontsize=8.5)
    a1.set_title(f"{case_id}: P(mel | image, clinical info marginalised) = {base_v:.3f}\n"
                 f"-> with this patient's clinical info = {full_v:.3f}", fontsize=9)
    g.sort(key=lambda kv: kv[1])
    a2.barh(range(len(g)), [v for _, v in g],
            color=["#7d3c98" if k == "ALL metadata" else "#5d6d7e" for k, _ in g])
    a2.set_yticks(range(len(g)))
    a2.set_yticklabels([k for k, _ in g], fontsize=8)
    a2.axvline(0, color="k", lw=0.8)
    a2.set_xlabel("test AUC drop when the variable is shuffled", fontsize=8.5)
    a2.set_title(f"global (n={glob.get('_n', 0):.0f} test cases, "
                 f"AUC {glob.get('_base_auc', 0):.3f})", fontsize=9)
    fig.suptitle(f"{display} - clinical metadata attribution", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
#  The whole suite
# ---------------------------------------------------------------------------
def _case_tensors(ds, i: int, device):
    img, meta, y = ds[i]
    return img[None].to(device), meta[None].to(device), float(y)


def _score_maps(model, img, meta, sign, maps, lesion, fov) -> Dict[str, Dict[str, float]]:
    scores: Dict[str, Dict[str, float]] = {}
    gen = torch.Generator().manual_seed(1)
    noise = torch.rand(img.shape[-2:], generator=gen)[None, None].to(img.device)
    rand = normalize_map(blur(noise, 8.0)[0, 0].cpu().numpy())
    for name, heat in list(maps.items()) + [("_random", rand)]:
        d, ins = deletion_insertion(model, img, meta, sign, heat)
        scores[name] = {"deletion": d, "insertion": ins, **localisation(heat, lesion, fov)}
    return scores


def run_xai_suite(model, cfg: UnifiedConfig, test_ds, train_ds, test_prob: np.ndarray,
                  threshold: float, temperature: float, out_dir: str, display: str,
                  device, log=print) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    labels = test_ds.labels.astype(int)
    spec = cam_spec(model)
    result: Dict = {"cam_layer": spec["layer"] if spec else None}

    def cid(i):
        return str(test_ds.df.iloc[i].get("isic_id", f"idx{i}"))

    with frozen(model):
        # ---- 1. main single-case panel -----------------------------------
        case = select_case(test_prob, labels, 1, threshold)
        img, meta, y = _case_tensors(test_ds, case, device)
        base = test_ds.raw(case).astype(np.float32) / 255
        lesion, fov = lesion_and_fov(test_ds.raw(case))
        sign = 1.0 if test_prob[case] >= threshold else -1.0
        maps, info = explain_case(model, img, meta, sign, cfg, spec)
        scores = _score_maps(model, img, meta, sign, maps, lesion, fov)
        truth = "melanoma" if y == 1 else "benign"
        pred = "melanoma" if sign > 0 else "benign"
        render_panel(base, maps, scores, lesion,
                     f"{cid(case)}\nGT {truth} | pred {pred} | p={test_prob[case]:.3f}",
                     f"{display} - why it predicted {pred} (single case)",
                     os.path.join(out_dir, "xai_panel.png"))
        result.update({"case_index": int(case), "case_id": cid(case),
                       "prob": float(test_prob[case]), "label": y,
                       "methods": list(maps), "case_scores": scores, **info})
        log(f"    panel   -> {', '.join(maps)}")

        # ---- 2. TP / TN / FP / FN gallery --------------------------------
        gal = select_gallery(test_prob, labels, threshold)
        rows = []
        for kind, i in gal.items():
            im, mt, yy = _case_tensors(test_ds, i, device)
            s = 1.0 if test_prob[i] >= threshold else -1.0
            mp, _ = explain_case(model, im, mt, s, cfg, spec, fast=True)
            les, fv = lesion_and_fov(test_ds.raw(i))
            sc = _score_maps(model, im, mt, s, mp, les, fv)
            rows.append({"kind": kind, "case_id": cid(i), "prob": float(test_prob[i]),
                         "gt": "melanoma" if yy == 1 else "benign",
                         "pred": "melanoma" if s > 0 else "benign",
                         "base": test_ds.raw(i).astype(np.float32) / 255,
                         "maps": mp, "scores": sc, "lesion": les})
        gm = tuple(m for m in ("Grad-CAM", "Integrated Gradients", "Occlusion",
                               "Attention (grad-rollout)") if any(m in r["maps"] for r in rows))
        if rows:
            render_gallery(rows, display, os.path.join(out_dir, "xai_gallery.png"), gm)
        result["gallery"] = {r["kind"]: {"case_id": r["case_id"], "prob": r["prob"],
                                         "gt": r["gt"], "pred": r["pred"]} for r in rows}
        log(f"    gallery -> {', '.join(r['kind'] for r in rows)}")

        # ---- 3. quantitative evaluation over n cases ----------------------
        ev_idx = select_eval_cases(test_prob, labels, threshold, cfg.xai_eval_cases)
        agg: Dict[str, Dict[str, List[float]]] = {}
        for i in ev_idx:
            im, mt, _ = _case_tensors(test_ds, i, device)
            s = 1.0 if test_prob[i] >= threshold else -1.0
            mp, _ = explain_case(model, im, mt, s, cfg, spec, fast=True)
            les, fv = lesion_and_fov(test_ds.raw(i))
            for name, sc in _score_maps(model, im, mt, s, mp, les, fv).items():
                for k, v in sc.items():
                    agg.setdefault(name, {}).setdefault(k, []).append(v)
        summary = {}
        for m, d in agg.items():
            summary[m] = {k: float(np.mean(v)) for k, v in d.items()}
            summary[m]["n"] = len(d["deletion"])
            summary[m]["n_with_lesion"] = len(d.get("energy_in_lesion", []))
        result["eval"] = {"n_cases": len(ev_idx), "summary": summary}
        log(f"    scored  -> {len(ev_idx)} cases")
        for m, d in summary.items():
            les_s = (f"{100 * d['energy_in_lesion']:.0f}%" if "energy_in_lesion" in d else "n/a")
            log(f"      {m:28s} del {d['deletion']:.3f} ins {d['insertion']:.3f} "
                f"lesion {les_s:>4s} outside-FOV {100 * d['energy_outside_fov']:.1f}%")

        # ---- 4. clinical metadata ----------------------------------------
        groups = meta_groups(test_ds.meta_cols)
        rng = np.random.default_rng(cfg.seed)
        bg = train_ds.meta[rng.choice(len(train_ds), cfg.shap_background, replace=False)]
        x = test_ds.meta[case]
        phi, v0, v1 = grouped_shapley(model, img, x, bg, groups, device, temperature)
        local = [(group_value_label(n, test_ds.meta_cols, ix, x), float(p))
                 for (n, ix), p in zip(groups, phi)]
        local.sort(key=lambda kv: -abs(kv[1]))
        n_perm = min(cfg.perm_n, len(test_ds))
        pidx = np.sort(rng.choice(len(test_ds), n_perm, replace=False))
        glob = grouped_permutation_importance(model, test_ds, pidx, groups, device, cfg.seed)
        glob["_n"] = n_perm
        render_meta(local, v0, v1, glob, display, cid(case),
                    os.path.join(out_dir, "shap_metadata.png"))
        result["shap_values"] = {n: float(p) for (n, _), p in zip(groups, phi)}
        result["shap_local_labeled"] = local
        result["shap_base_value"] = v0
        result["shap_full_value"] = v1
        result["meta_permutation_importance"] = glob
        log("    shapley -> " + ", ".join(f"{l}: {v:+.3f}" for l, v in local[:4]))

    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result
