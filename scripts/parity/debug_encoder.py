"""Locate where the fp16 encoder diverges from fp32: per-block residual-stream comparison.

    python -m scripts.parity.debug_encoder --variant base            # eager torch only (fp32 vs fp16 per block)
    python -m scripts.parity.debug_encoder --variant base --trt      # + export/build/run a debug TRT plan
    python -m scripts.parity.debug_encoder --variant sm32 --trt      # attention softmax in fp32
    python -m scripts.parity.debug_encoder --variant attn32 --trt    # q.kT, scale and softmax in fp32

The probe encoder returns descriptor, dense and the residual stream after each of the 12 blocks (`h1..h12`,
[1,534,768]) as extra outputs. Extra outputs can change TensorRT's fusion (they force the block outputs to be
materialised), so the probe plan is a diagnostic, not the shipped plan. Variants (fp16 model only):
  base    the shipped fast graph (DINOv2 attention = scaled_dot_product_attention)
  sm32    q.kT in fp16, then scale + softmax in fp32, cast back before .V
  attn32  q.kT, scale, softmax in fp32 (TF32 off in the build), cast back before .V
Per block it prints the per-token cosine against the torch fp32 residual (p01, min, tokens < 0.98) and the fp32
residual's abs max (fp16 overflows at 65504). The study plans were built with TF32 on (off for attn32); the fp16
GEMMs are unaffected, and TF32 only matters for the fp32 heads. Plans/ONNX go to runs/debug/ (git-ignored). Needs an idle GPU for the
TensorRT build.
"""
import argparse
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pairvpr.export.patches import IMG_RES, ROOT, build_model
from pairvpr.export.wrappers import convert

from .common import DEVICE, REF_DIR

OUT = ROOT / "runs" / "debug"


def attention_forward(variant: str):
    def forward(self, x, is_causal: bool = False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = [t.transpose(1, 2) for t in torch.unbind(qkv, 2)]
        if variant == "attn32":
            attn = (q.float() @ k.float().transpose(-2, -1)) * self.scale
        else:
            attn = (q @ k.transpose(-2, -1)).float() * self.scale
        attn = attn.softmax(dim=-1).to(v.dtype)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))
    return forward


class Probe(nn.Module):
    def __init__(self, model, mode: str, variant: str):
        super().__init__()
        self.model = convert(model, mode)
        self.dtype = torch.float16 if mode == "fast" else torch.float32
        if variant != "base":
            for blk in model.encoder.model.blocks:
                blk.attn.forward = types.MethodType(attention_forward(variant), blk.attn)

    def forward(self, image):
        vit = self.model.encoder.model
        x = vit.prepare_tokens_with_masks(image.to(self.dtype), None)
        hs = []
        for blk in vit.blocks:
            x = blk(x)
            hs.append(x)
        x = vit.norm(x)
        desc = nn.functional.normalize(self.model.globalizer(x[:, 0].float()), p=2, dim=-1)
        return (desc, x[:, 5:], *hs)


class SubProbe(nn.Module):
    """Intermediates of one block: tokens in, norm1, attn, x+ls1(attn), norm2, fc1, gelu, fc2, block out."""

    def __init__(self, model, mode: str, block: int):
        super().__init__()
        self.model = convert(model, mode)
        self.dtype = torch.float16 if mode == "fast" else torch.float32
        self.block = block

    def forward(self, image):
        vit = self.model.encoder.model
        x = vit.prepare_tokens_with_masks(image.to(self.dtype), None)
        for blk in vit.blocks[:self.block]:
            x = blk(x)
        blk = vit.blocks[self.block]
        n1 = blk.norm1(x)
        a = blk.attn(n1)
        xa = x + blk.ls1(a)
        n2 = blk.norm2(xa)
        f1 = blk.mlp.fc1(n2)
        g = blk.mlp.act(f1)
        f2 = blk.mlp.fc2(g)
        out = xa + blk.ls2(f2)
        return (x, n1, a, xa, n2, f1, g, f2, out)


SUB_NAMES = ["tokens_in", "norm1", "attn", "x+ls1(attn)", "norm2", "fc1", "gelu", "fc2", "block_out"]


def tok_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a.astype(np.float64), b.astype(np.float64)
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


@torch.inference_mode()
def run_torch(probe, pre):
    outs = []
    for x in pre:
        o = probe(torch.from_numpy(x)[None].to(DEVICE))
        outs.append([t.float().cpu().numpy()[0] for t in o])
    return [np.stack(c) for c in zip(*outs)]


def report(name: str, outs, ref_outs) -> None:
    print(f"\n{name}: per-block residual cosine vs torch fp32 (p01 / min / tokens<0.98 of {outs[2].shape[0] * outs[2].shape[1]}):")
    for i in range(12):
        c = tok_cos(outs[2 + i], ref_outs[2 + i])
        print(f"  block {i + 1:2d}  fp32 absmax {np.abs(ref_outs[2 + i]).max():9.1f}   "
              f"{np.quantile(c, 0.01):.5f} / {c.min():.4f} / {(c < 0.98).sum()}")
    c = tok_cos(outs[1], ref_outs[1])
    d = (outs[0] * ref_outs[0]).sum(-1)
    print(f"  dense (after norm) {np.quantile(c, 0.01):.5f} / {c.min():.4f} / {(c < 0.98).sum()}   desc cos min {d.min():.6f}")


def sub_probe(block: int, pre) -> int:
    import Build
    from .compare import TrtRunner
    ref = run_torch(SubProbe(build_model("precise", DEVICE), "precise", block).eval(), pre)
    fast = SubProbe(build_model("fast", DEVICE), "fast", block).eval()
    eager = run_torch(fast, pre)
    OUT.mkdir(parents=True, exist_ok=True)
    onnx_path, plan = str(OUT / f"sub{block}.onnx"), str(OUT / f"sub{block}.plan")
    with torch.inference_mode():
        x0 = torch.randn(1, 3, IMG_RES, IMG_RES, device=DEVICE)
        fast(x0)
        torch.onnx.export(fast, (x0,), f=onnx_path, opset_version=18, input_names=["image"],
                          output_names=SUB_NAMES, do_constant_folding=False, dynamo=False)
    Build.build(onnx_path=onnx_path, plan_path=plan, mode="fast", opt_level=5, avg_timing_iterations=8, tiling_level=0,
                error_on_timing_cache_miss=False, weight_streaming=True, allow_tf32=True, timing_cache_path=None,
                verbose=False)
    run = TrtRunner(Path(plan))
    outs = [run({"image": x[None]}) for x in pre]
    trt = [np.stack([o[n][0] for o in outs]) for n in SUB_NAMES]
    print(f"\nblock {block + 1} intermediates, token cosine vs torch fp32 (p01 / min), plus rel. max err:")
    for i, n in enumerate(SUB_NAMES):
        row = []
        for tag, o in (("torch fast", eager), ("TRT fast", trt)):
            c = tok_cos(o[i], ref[i])
            rel = np.abs(o[i] - ref[i]).max() / np.abs(ref[i]).max()
            row.append(f"{tag} {np.quantile(c, 0.01):.5f} / {c.min():.5f} rel {rel:.1e}")
        print(f"  {n:12s} " + "   ".join(row))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variant", choices=["base", "sm32", "attn32"], default="base")
    ap.add_argument("--trt", action="store_true", help="also export, build and run a probe TensorRT plan")
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--sub-block", type=int, default=None,
                    help="instead: probe the intermediates of this block (0-based) in TRT vs torch fp32")
    args = ap.parse_args()
    pre = np.load(REF_DIR / "ref.npz")["pre"][:args.frames]

    if args.sub_block is not None:
        return sub_probe(args.sub_block, pre)
    ref_outs = run_torch(Probe(build_model("precise", DEVICE), "precise", "base").eval(), pre)
    fast = Probe(build_model("fast", DEVICE), "fast", args.variant).eval()
    report(f"torch fast ({args.variant})", run_torch(fast, pre), ref_outs)
    if not args.trt:
        return 0

    import Build
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"probe_{args.variant}"
    onnx_path, plan = str(OUT / f"{stem}.onnx"), str(OUT / f"{stem}.plan")
    names = ["descriptor", "dense"] + [f"h{i + 1}" for i in range(12)]
    with torch.inference_mode():
        x0 = torch.randn(1, 3, IMG_RES, IMG_RES, device=DEVICE)
        fast(x0)
        torch.onnx.export(fast, (x0,), f=onnx_path, opset_version=18, input_names=["image"], output_names=names,
                          do_constant_folding=False, dynamo=False)
    Build.build(onnx_path=onnx_path, plan_path=plan, mode="fast", opt_level=5, avg_timing_iterations=8, tiling_level=0,
                error_on_timing_cache_miss=False, weight_streaming=True, allow_tf32=args.variant != "attn32", timing_cache_path=None,
                verbose=False)
    from .compare import TrtRunner
    run = TrtRunner(Path(plan))
    outs = [run({"image": x[None]}) for x in pre]
    trt_outs = [np.stack([o[n][0] for o in outs]) for n in names]
    report(f"TRT probe ({args.variant})", trt_outs, ref_outs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
