"""Export-time patches that build the Pair-VPR ViT-B network without xFormers, omegaconf or the network.

`build_model(mode, device)` returns the loaded `PairVPRNet` in eval mode. What it changes, and why:

  1. XFORMERS_DISABLED=1 is set before `pairvpr.models` / DINOv2 are imported, so every attention takes the
     plain PyTorch matmul + softmax path (xFormers' memory_efficient_attention does not export to ONNX).
  2. `MemEffCrossAttention.forward` fallback called `super().forward(query, key, value, qpos, kpos)`, but
     `CrossAttention.forward` takes only (query, key, value): a TypeError on every decoder block without
     xFormers. Patched to drop the unused position arguments.
  3. DINOv2 is built by `torch.hub.load('facebookresearch/dinov2', ...)` (unpinned, twice, with the
     pretrained weights downloaded). While the model is built, `torch.hub.load` is redirected to the pinned
     commit DINOV2_SHA with pretrained=False: the Pair-VPR checkpoint holds every weight.
  4. DINOv2's `interpolate_pos_encoding` resizes the 37x37 positional grid to 23x23 (bicubic, antialiased) on
     every forward; that resize does not export. It is evaluated once, eagerly in fp32 at 322 x 322, and the
     method is replaced by one returning that constant [1, 530, 768] (exact: test_exact checks `torch.equal`
     against the live interpolation).
  5. The YAML config is read with PyYAML into an attribute dict (the repo's configs/__init__ needs omegaconf).
     pairvpr_speed.yaml is the complete config eval.py uses (it merges that same file over itself).

Speed mode is ViT-B (this checkpoint); performance mode is ViT-G (different weights, same 322 input and decoder
config). Only the ViT-B weights are available here.
"""
import os

os.environ["XFORMERS_DISABLED"] = "1"          # must precede any pairvpr.models / DINOv2 import

import types                                   # noqa: E402
from contextlib import contextmanager          # noqa: E402
from pathlib import Path                       # noqa: E402

import torch                                   # noqa: E402
import yaml                                    # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "pairvpr" / "configs" / "pairvpr_speed.yaml"
CKPT = ROOT / "trained_models" / "pairvpr-vitB.pth"
# facebookresearch/dinov2 main as of 2026-10-27 (git ls-remote ... HEAD).
DINOV2_SHA = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
IMG_RES = 322
NUM_TOKENS = (IMG_RES // 14) ** 2              # 529
EMBED = 768


class AttrDict(dict):
    """dict with attribute access; nested dicts are wrapped. Supports `Cls(**cfg.section)`."""

    def __init__(self, d: dict):
        super().__init__({k: AttrDict(v) if isinstance(v, dict) else v for k, v in d.items()})

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e


def load_config(path: Path = CONFIG) -> AttrDict:
    with open(path, encoding="utf-8") as f:
        return AttrDict(yaml.safe_load(f))


@contextmanager
def pinned_dinov2_hub():
    """torch.hub.load('facebookresearch/dinov2', name) -> pinned commit, pretrained=False."""
    orig = torch.hub.load

    def load(repo, model, *args, **kw):
        if repo == "facebookresearch/dinov2":
            repo = f"facebookresearch/dinov2:{DINOV2_SHA}"
            kw.update(pretrained=False, trust_repo=True, skip_validation=True)
        return orig(repo, model, *args, **kw)

    torch.hub.load = load
    try:
        yield
    finally:
        torch.hub.load = orig


def _fix_cross_attention() -> None:
    from pairvpr.models.tools import blocks

    def forward(self, query, key, value, qpos=None, kpos=None, attn_bias=None):
        if blocks.XFORMERS_AVAILABLE:
            raise RuntimeError("export expects XFORMERS_DISABLED=1")
        return blocks.CrossAttention.forward(self, query, key, value)

    blocks.MemEffCrossAttention.forward = forward


def freeze_pos_embed(model: torch.nn.Module) -> torch.Tensor:
    """Replace DINOv2's interpolate_pos_encoding by a constant [1, 530, 768] computed eagerly at 322."""
    vit = model.encoder.model
    # interpolate_pos_encoding takes the token sequence including the class token (npatch = x.shape[1] - 1).
    x = torch.zeros(1, NUM_TOKENS + 1, EMBED, device=next(model.parameters()).device)
    with torch.no_grad():
        const = vit.interpolate_pos_encoding(x, IMG_RES, IMG_RES).detach().clone()
    assert const.shape == (1, NUM_TOKENS + 1, EMBED), const.shape
    vit.register_buffer("frozen_pos_embed", const, persistent=False)

    def interpolate_pos_encoding(self, x, w, h):
        assert w == IMG_RES and h == IMG_RES, "pos embed frozen at 322 x 322"
        return self.frozen_pos_embed.to(x.dtype)

    vit.interpolate_pos_encoding = types.MethodType(interpolate_pos_encoding, vit)
    return const


def build_model(mode: str = "fast", device: str = "cpu", ckpt: Path | str = CKPT,
                freeze: bool = True) -> torch.nn.Module:
    """Loaded PairVPRNet, eval mode, fp32 weights (the wrappers cast for `mode`; `mode` is only validated).
    freeze=False keeps the live pos-embed interpolation (test_exact)."""
    assert mode in ("fast", "mixed", "precise"), mode
    _fix_cross_attention()
    from pairvpr.models.pairvpr import PairVPRNet
    cfg = load_config()
    with pinned_dinov2_hub():
        model = PairVPRNet(cfg)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.eval().to(device)
    if freeze:
        freeze_pos_embed(model)
    return model
