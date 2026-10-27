"""
ONNX export for the Pair-VPR ViT-B model (speed mode), as two static engines: an encoder and a pair scorer.

Mirrors MAV-VO-MCH/Export.py. "Mode" here is the dtype recipe only; the speed / performance split of Pair-VPR
differs by weights (ViT-B vs ViT-G; same 322 input and decoder config) and by the rerank depth (500 candidates
in performance mode). Only the ViT-B weights (trained_models/pairvpr-vitB.pth) are available, so only the speed
networks are exported; the rerank depth is the consumer's choice.

  Encoder  image [1,3,322,322] fp32 (already ImageNet-normalised) -> descriptor [1,512] fp32 (L2-normalised),
           dense [1,529,768] (fp16 fast and mixed, fp32 precise).
  Pair<C>  query [1,529,768], candidates [C,529,768] (the dense dtype: fp16 for the fast pair, fp32 for the precise pair) -> score [C] fp32 =
           logit(q, c_i) + logit(c_i, q), higher = more similar. Both orders run as ONE 2C batch (the query is a
           runtime `expand`), so the consumer makes one call per query instead of two. C is static
           (--pair-batch, default 8); pad a short candidate list by repeating a row.

Modes:
  fast     fp16 weights and activations, with fp32 islands for the descriptor head (globalizer + L2 normalise)
           and the score head (classvprmodule + the order sum). Pair with Build.py --mode fast.
  mixed    encoder fp32 (TF32 off) with only `dense` cast to fp16 at the end (descriptor fp32); the pair scorer
           of this mode is the fast one (--mode mixed writes Encoder_mixed only and pairs with Pair<C>_fast). Why: TensorRT
           picks fp16-accumulate GEMM tactics whose error compounds through the 12 blocks (docs/tensorrt-export.md).
  precise  fp32 everywhere (reference-precision plan; TF32 cleared by Build.py --mode precise).

Conventions (same reasons as MAC-VO's export):
  * Static shapes, a plain trace, no dynamic_axes.
  * Opset 18. TensorRT 10.x silently mis-compiled an opset-20 graph of the MAC-VO model (fp32 plan EPE 0.94 px
    -> 6.4 px) while onnxruntime ran it correctly; 18 is the only validated choice.
  * A warm-up forward under torch.inference_mode() before tracing, so lazily built caches trace as constants.
  * do_constant_folding=False: TensorRT folds constants itself at build time.
  * Pre-processing lives in exactly one place, the consumer: resize 322x322 (bilinear, antialiased) + ImageNet
    normalisation. The graph takes the normalised tensor. Post-processing (top-K, sort) is the consumer's.
  * The DINOv2 positional embedding is frozen to a constant [1,530,768] (pairvpr/export/patches.py), the
    xFormers-free cross-attention fallback is fixed, and DINOv2 is pinned to one commit.
  * No onnxsim: nothing to gain on a 12-block transformer, and MAC-VO's measurements show it can change TensorRT's
    tactic choice.

Deliverables (--out-dir, default Model/):
  Model/PairVPR_ViTB_Encoder_<mode>.onnx
  Model/PairVPR_ViTB_Pair<C>_<mode>.onnx
(fp16 sizes: encoder 166 MiB, Pair8 221 MiB; fp32 is about twice that. All stay under the 2 GB protobuf limit, so
no `*.onnx.data` file appears.)
--mode mixed exports ONLY Encoder_mixed: the pair scorer of the recommended pipeline is Pair<C>_fast, exported by
--mode fast (re-exporting it is deterministic, but a screened plan's ONNX must not be touched by another mode).

Equivalent trtexec builder command (static ONNX, strongly typed, weight streaming, level 5):
  trtexec --onnx=PairVPR_ViTB_Encoder_fast.onnx --saveEngine=PairVPR_ViTB_Encoder_fast_o5.plan ^
    --stronglyTyped --allowWeightStreaming --builderOptimizationLevel=5 --avgTiming=8 ^
    --profilingVerbosity=detailed --timingCacheFile=PairVPR_ViTB_Encoder_fast_o5.timing.cache

Validate with scripts/parity/test_exact.py and scripts/parity/compare.py.
"""
import argparse
import os
import sys
from collections import Counter

import onnx
import torch

from pairvpr.export.patches import CKPT, DINOV2_SHA, EMBED, IMG_RES, NUM_TOKENS, build_model
from pairvpr.export.wrappers import DENSE_DTYPE, MODES, Encoder, PairScorer

_ONNX_DTYPE = {1: "fp32", 10: "fp16", 16: "bf16"}


def _dtype_name(t: onnx.TypeProto) -> str:
    return _ONNX_DTYPE.get(t.tensor_type.elem_type, str(t.tensor_type.elem_type))


def _shape(t: onnx.TypeProto) -> list[int]:
    return [d.dim_value for d in t.tensor_type.shape.dim]


def print_io(path: str) -> None:
    m = onnx.load(path, load_external_data=False)
    for kind, vals in (("input ", m.graph.input), ("output", m.graph.output)):
        for v in vals:
            print(f"  {kind} {v.name:12s} {_shape(v.type)} {_dtype_name(v.type)}", flush=True)
    hist = Counter(n.op_type for n in m.graph.node)
    print(f"  {len(m.graph.node)} nodes: " + ", ".join(f"{k}={n}" for k, n in hist.most_common(8)), flush=True)


def export_part(part: str, mode: str, pair_batch: int, opset: int, ckpt: str,
                out_dir: str, device: str) -> str:
    assert not (part == "pair" and mode == "mixed"), "mixed has no pair scorer of its own: export --mode fast"
    name = ("PairVPR_ViTB_Encoder_%s.onnx" % mode if part == "encoder"
            else "PairVPR_ViTB_Pair%d_%s.onnx" % (pair_batch, mode))
    path = os.path.join(out_dir, name)
    model = build_model(mode, device, ckpt=ckpt)
    dense = DENSE_DTYPE[mode]
    if part == "encoder":
        wrapper = Encoder(model, mode).eval()
        args = (torch.randn(1, 3, IMG_RES, IMG_RES, device=device),)
        in_names, out_names = ["image"], ["descriptor", "dense"]
    else:
        wrapper = PairScorer(model, mode).eval()
        args = (torch.randn(1, NUM_TOKENS, EMBED, device=device, dtype=dense),
                torch.randn(pair_batch, NUM_TOKENS, EMBED, device=device, dtype=dense))
        in_names, out_names = ["query", "candidates"], ["score"]

    print(f"[1/2] Tracing {part} -> {path} (mode={mode}, opset={opset}, static)...", flush=True)
    with torch.inference_mode():
        wrapper(*args)                       # warm-up: caches trace as constants
        torch.onnx.export(wrapper, args, f=path, opset_version=opset, input_names=in_names,
                          output_names=out_names, dynamic_axes=None, export_params=True,
                          do_constant_folding=False, dynamo=False)
    del wrapper, model
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"OK: {path} ({os.path.getsize(path) / 2**20:.1f} MiB)", flush=True)
    print_io(path)
    return path


def main() -> int:
    p = argparse.ArgumentParser(description="Export the Pair-VPR ViT-B encoder and pair scorer to ONNX")
    p.add_argument("--part", choices=("encoder", "pair", "both"), default="both")
    p.add_argument("--mode", choices=MODES, default="fast",
                   help="fast = fp16 weights/activations with fp32 descriptor and score heads; "
                        "mixed = fp32 encoder with fp16 dense output, fast pair scorer; precise = fp32 everywhere")
    p.add_argument("--pair-batch", type=int, default=8,
                   help="Static candidate count C of the pair scorer (one call scores C candidates in both "
                        "orders as a 2C batch); also the rerank top-K")
    p.add_argument("--opset", type=int, default=18,
                   help="ONNX opset (18: opset 20 compiled silently wrong on TensorRT 10.x for MAC-VO)")
    p.add_argument("--ckpt", default=str(CKPT), help="Pair-VPR ViT-B checkpoint (plain fp32 state_dict)")
    p.add_argument("--out-dir", default="Model", help="Output directory (default Model/)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"DINOv2 pinned to facebookresearch/dinov2@{DINOV2_SHA}", flush=True)
    if args.mode == "mixed" and args.part == "pair":
        p.error("--mode mixed has no pair scorer of its own: export the pair with --mode fast")
    parts = ("encoder", "pair") if args.part == "both" else (args.part,)
    if args.mode == "mixed":
        parts = ("encoder",)                 # mixed = Encoder_mixed only; the pair is Pair<C>_fast, exported separately
        print("mode mixed: exporting the encoder only (the pair scorer is --mode fast)", flush=True)
    for part in parts:
        export_part(part, args.mode, args.pair_batch, args.opset, args.ckpt, args.out_dir,
                    args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
