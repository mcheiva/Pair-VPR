"""
ONNX -> TensorRT engine builder for the Pair-VPR ViT-B encoder and pair scorer.
Adapted from MAV-VO-MCH/Build.py: same flags, timing-cache semantics, miss detection, sidecar schema and write
order; only the defaults and names differ. All modes build a strongly-typed network so TRT respects the dtypes
baked into the ONNX literally (pair with the matching --mode in Export.py).

Parts (--part encoder|pair|both, default both), static shapes, so no optimization profile is created:
  encoder  Model/PairVPR_ViTB_Encoder_<mode>.onnx -> Model/PairVPR_ViTB_Encoder_<mode>_o<level>.plan
  pair     Model/PairVPR_ViTB_Pair<C>_<mode>.onnx -> Model/PairVPR_ViTB_Pair<C>_<mode>_o<level>.plan   (C = --pair-batch)
Each plan gets <plan>.json (sidecar) and <plan minus .plan>.timing.cache next to it (the cache is on by default;
--no-timing-cache disables it; an explicit --timing-cache needs a single --part).

Modes:
  fast     ONNX has fp16 weights/activations (fp32 descriptor and score heads) baked. Strongly-typed + weight
           streaming + opt-level 5.
  mixed    encoder ONNX fp32 with an fp16 dense output; encoder only (the pair is built by --mode fast).
  precise  ONNX has fp32 baked. Strongly-typed (so TRT keeps fp32 throughout).
TF32 is cleared in every mode (--tf32 opts in). A strongly typed build honours TF32 for any fp32 MatMul, including
the tiny fp32 heads of the fast graphs (globalizer, classvprmodule), and the torch-fast yardstick runs with TF32
off: the fast plans of the first screening were built with TF32 on (allow_tf32 true in their sidecars); the
deployed Pair8_fast plan is built with it off.

Weight streaming is on by default. MAC-VO measurements (RTX 4090, TRT 10.13.2): on a dynamic-axis ONNX the flag
forces the faster all-Myelin graph; on a static batch-1 ONNX it changed nothing (identical tactics, same
latency). It costs nothing at the default (full) budget: nothing is streamed at runtime. --no-weight-streaming
builds without it.

Optimization level 5 (default): the MAC-VO engines were built at 5 and validated end to end; MAC-VO issue #18
reported corrupted flow at higher levels on its build.

The timing cache is valid only on the same GPU model and TensorRT version: it is loaded with
ignore_mismatch=False, so another GPU or TensorRT refuses it. A build on another GPU picks its own tactics,
which can change numerics: screen that plan (scripts/parity/compare.py) before deploying it.

Equivalent trtexec invocations (static ONNX, no shape flags):
  fast:
    trtexec --onnx=PairVPR_ViTB_Encoder_fast.onnx --saveEngine=PairVPR_ViTB_Encoder_fast_o5.plan ^
      --stronglyTyped --allowWeightStreaming --builderOptimizationLevel=5 --avgTiming=8 ^
      --profilingVerbosity=detailed --timingCacheFile=PairVPR_ViTB_Encoder_fast_o5.timing.cache
  precise (fp32, TF32 off): the same with --noTF32 and the _precise names.
  Add --errorOnTimingCacheMiss for --error-on-timing-cache-miss.

Writes <plan>.json next to the plan (ONNX path/size/hash, build settings, timing-cache status, TensorRT version,
GPU), same fields and order as the MAC-VO builder.

Note: engines are only loadable by the exact TensorRT version that built them ("Version tag does not match").
"""
import argparse
import json
import os
import sys

import tensorrt as trt


MODEL_DIR = "Model"


class CountingLogger(trt.ILogger):
    """trt.Logger that counts ERROR / INTERNAL_ERROR messages and the
    timing-cache-miss warnings (one multi-line "Engine generation failed with
    backend strategy N. Error message: ... Adding new entry to timing cache is
    not allowed if `BuilderFlag::kERROR_ON_TIMING_CACHE_MISS` is set")."""

    def __init__(self, verbose: bool) -> None:
        trt.ILogger.__init__(self)
        self.verbose = verbose
        self.errors = 0
        self.strategy_failures = 0

    def log(self, severity, msg) -> None:
        if severity in (trt.ILogger.Severity.INTERNAL_ERROR,
                        trt.ILogger.Severity.ERROR):
            self.errors += 1
            print(f"[TRT ERROR] {msg}", file=sys.stderr)
        elif severity == trt.ILogger.Severity.WARNING:
            if ("kERROR_ON_TIMING_CACHE_MISS" in msg
                    or "Adding new entry to timing cache" in msg):
                self.strategy_failures += 1
            print(f"[TRT WARN ] {msg}", file=sys.stderr)
        elif self.verbose:
            print(f"[TRT INFO ] {msg}", file=sys.stderr)


def fnv1a64_hex(path: str) -> str | None:
    """64-bit FNV-1a of a file's bytes as 16 hex digits; None on read failure."""
    h = 14695981039346656037
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                for b in chunk:
                    h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    except OSError:
        return None
    return f"{h:016x}"


def all_inputs_static(network) -> bool:
    """True if every dimension of every network input is fixed (no -1)."""
    for i in range(network.num_inputs):
        if any(d < 0 for d in network.get_input(i).shape):
            return False
    return True


def gpu_info() -> tuple[str | None, str | None]:
    """(name, compute capability) via torch.cuda if importable, else (None, None)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None, None
        dev = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(dev)
        return torch.cuda.get_device_name(dev), f"{major}.{minor}"
    except Exception:
        return None, None


def write_sidecar(plan_path: str, onnx_path: str, mode: str, is_static: bool,
                  min_batch: int, opt_batch: int, max_batch: int,
                  height: int, width: int, opt_level: int,
                  avg_timing_iterations: int, tiling_level: int,
                  error_on_timing_cache_miss: bool, weight_streaming: bool,
                  allow_tf32: bool, timing_cache_path: str | None,
                  cache_loaded: bool, bytes_loaded: int,
                  bytes_saved: int) -> None:
    """Writes <plan_path>.json; same fields and order as the C++ builder."""
    try:
        onnx_size = os.path.getsize(onnx_path)
    except OSError:
        onnx_size = None
    # tensorrt.__version__ is "MAJOR.MINOR.PATCH.BUILD"; the C++ builder's
    # tensorrt_version is (MAJOR * 100 + MINOR) * 100 + PATCH.
    # The 4th component is the header's NV_TENSORRT_BUILD (the C++ builder writes the same); null if absent.
    parts = trt.__version__.split(".")
    major, minor, patch = (int(x) for x in parts[:3])
    build_header = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else None
    used = timing_cache_path is not None
    gpu_name, cc = gpu_info()
    side = {
        "onnx_path": os.path.abspath(onnx_path),
        "onnx_size_bytes": onnx_size,
        "onnx_fnv1a64": fnv1a64_hex(onnx_path),
        "mode": mode,
        "static": is_static,
        "min_batch": None if is_static else min_batch,
        "opt_batch": None if is_static else opt_batch,
        "max_batch": None if is_static else max_batch,
        "height": None if is_static else height,
        "width": None if is_static else width,
        "optimization_level": opt_level,
        "avg_timing_iterations": avg_timing_iterations,
        "tiling_optimization_level": tiling_level,
        "error_on_timing_cache_miss": error_on_timing_cache_miss,
        "weight_streaming": weight_streaming,
        "allow_tf32": allow_tf32,
        "timing_cache": os.path.abspath(timing_cache_path) if used else None,
        "timing_cache_status": (("loaded" if cache_loaded else "started_empty")
                                if used else None),
        "timing_cache_bytes_loaded": bytes_loaded if used else None,
        "timing_cache_bytes_saved": bytes_saved if used else None,
        "tensorrt_version": (major * 100 + minor) * 100 + patch,
        "tensorrt_version_str": f"{major}.{minor}.{patch}",
        "tensorrt_build_header": build_header,
        "gpu_name": gpu_name,
        "compute_capability": cc,
    }
    with open(plan_path + ".json", "w", encoding="utf-8") as f:
        json.dump(side, f, indent=2)
        f.write("\n")
    print(f"Wrote {plan_path}.json", file=sys.stderr, flush=True)


def build(onnx_path: str,
          plan_path: str,
          mode: str,
          opt_level: int,
          avg_timing_iterations: int,
          tiling_level: int,
          error_on_timing_cache_miss: bool,
          weight_streaming: bool,
          allow_tf32: bool,
          timing_cache_path: str | None,
          verbose: bool,
          parse_only: bool = False) -> None:
    logger = CountingLogger(verbose)

    # A stale sidecar next to a failed build would describe the wrong plan.
    if not parse_only:
        try:
            os.remove(plan_path + ".json")
        except FileNotFoundError:
            pass
        if timing_cache_path is not None:
            parent = os.path.dirname(os.path.abspath(timing_cache_path))
            if not os.path.isdir(parent):
                raise RuntimeError(
                    f"Timing cache directory does not exist: {parent}")

    builder = trt.Builder(logger)

    # Always strongly-typed: TRT respects the ONNX-baked dtypes literally.
    # For the fast (fp16) ONNX: fp16 with the fp32 heads literally.
    # For the precise (fp32) ONNX: full fp32 throughout.
    # Strongly-typed is incompatible with the legacy BuilderFlag.FP16 /
    # OBEY_PRECISION_CONSTRAINTS path; do not set those flags here.
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flag)

    parser = trt.OnnxParser(network, logger)
    # parse_from_file resolves the external-data sidecar (`*.onnx.data`)
    # relative to the ONNX path, regardless of the caller's CWD.
    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            print(parser.get_error(i), file=sys.stderr)
        raise RuntimeError(f"ONNX parse failed: {onnx_path}")

    is_static = all_inputs_static(network)
    if parse_only:
        print(f"parsed {onnx_path}: inputs "
              f"{'all static' if is_static else 'dynamic'}, optimization "
              f"profile would be {'NOT added' if is_static else 'added'}",
              flush=True)
        return

    config = builder.create_builder_config()
    config.builder_optimization_level = opt_level
    config.avg_timing_iterations = avg_timing_iterations
    # The repo's tensorrt stubs predate TilingOptimizationLevel (TRT 10.8+).
    config.tiling_optimization_level = trt.TilingOptimizationLevel(  # pyright: ignore[reportAttributeAccessIssue]
        tiling_level)
    if error_on_timing_cache_miss:
        config.set_flag(trt.BuilderFlag.ERROR_ON_TIMING_CACHE_MISS)

    if weight_streaming:
        # Makes the builder offload every Myelin-supported op to Myelin
        # (one graph instead of many islands); see the module docstring.
        config.set_flag(trt.BuilderFlag.WEIGHT_STREAMING)

    # Makes the plan inspectable (per-layer precision/format/tactic). Also
    # raises the NVTX verbosity, which TensorRT says may add enqueueV3
    # latency.
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    if not allow_tf32:
        # Disables TF32 in fp32 matmul kernels. TF32 truncates the mantissa
        # to 10 bits, which makes a fp32-baked plan diverge from fp32
        # PyTorch (allow_tf32=False is the PyTorch default).
        config.clear_flag(trt.BuilderFlag.TF32)

    if not is_static:
        raise RuntimeError("the ONNX has dynamic inputs; Export.py writes static shapes only, so no "
                           "optimization profile is supported: re-export")
    print("ONNX inputs are static: no optimization profile", file=sys.stderr, flush=True)

    # Timing cache: reusing a cache from a prior build makes tactic choice
    # repeatable in practice; TensorRT does not guarantee it. Check by
    # diffing TacticName in `trtexec --dumpLayerInfo` of two plans. The cache
    # object must outlive the build, so it is kept in a local.
    timing_cache = None
    cache_loaded = False
    bytes_loaded = 0
    bytes_saved = 0
    if timing_cache_path is not None:
        cache_bytes = b""
        if os.path.exists(timing_cache_path):
            with open(timing_cache_path, "rb") as f:
                cache_bytes = f.read()
        cache_loaded = len(cache_bytes) > 0
        bytes_loaded = len(cache_bytes)
        timing_cache = config.create_timing_cache(cache_bytes)
        if timing_cache is None:
            raise RuntimeError("create_timing_cache failed")
        if not config.set_timing_cache(timing_cache, ignore_mismatch=False):
            raise RuntimeError("set_timing_cache failed (cache was built for "
                               "different device properties?): "
                               f"{timing_cache_path}")
        if cache_loaded:
            print(f"Timing cache loaded: {timing_cache_path} "
                  f"({bytes_loaded} bytes)", file=sys.stderr, flush=True)
        else:
            print(f"Timing cache not found, starting empty: "
                  f"{timing_cache_path}", file=sys.stderr, flush=True)

    print(f"Building TRT {trt.__version__} engine "
          f"(mode={mode}, weight_streaming={weight_streaming}, "
          f"allow_tf32={allow_tf32}, opt_level={opt_level}, "
          f"avg_timing={avg_timing_iterations}, tiling={tiling_level}). "
          f"Takes several minutes...", flush=True)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("build_serialized_network returned None "
                           "(likely OOM or builder error)")
    # In TRT 10.13.2 a miss does not fail the build: the logger gets one
    # multi-line warning ("Engine generation failed with backend strategy N.
    # Error message: ... Adding new entry to timing cache is not allowed ..."),
    # and TensorRT falls back to another strategy and writes a plan. Primary
    # signal: the cache as this build leaves it, serialized before the on-disk
    # merge, is larger than the loaded bytes, i.e. TensorRT timed new entries.
    # Secondary: the logger saw an error or the miss warning.
    if error_on_timing_cache_miss and timing_cache is not None:
        own_size = len(bytes(timing_cache.serialize()))
        if own_size > bytes_loaded:
            raise RuntimeError(
                "--error-on-timing-cache-miss: the build's timing cache grew "
                f"from {bytes_loaded} to {own_size} bytes (TensorRT timed new "
                "entries); plan not written")
    if error_on_timing_cache_miss and (logger.errors or logger.strategy_failures):
        raise RuntimeError(
            f"--error-on-timing-cache-miss: the logger saw {logger.errors} "
            f"error(s) and {logger.strategy_failures} timing-cache-miss "
            "warning(s) during the build; plan not written")

    plan_bytes = bytes(serialized)
    with open(plan_path, "wb") as f:
        f.write(plan_bytes)
    print(f"Wrote {plan_path}: {len(plan_bytes) / (1024 * 1024):.2f} MiB",
          flush=True)

    cache_blob = None
    if timing_cache is not None and timing_cache_path is not None:
        # Another build (e.g. on the other GPU) may have saved since we
        # loaded: merge its entries so neither run's work is lost.
        if os.path.exists(timing_cache_path):
            with open(timing_cache_path, "rb") as f:
                disk = f.read()
            if disk:
                on_disk = config.create_timing_cache(disk)
                if on_disk is None or not timing_cache.combine(
                        on_disk, ignore_mismatch=False):
                    print("Timing cache: could not merge the on-disk copy, "
                          "overwriting it with this build's cache",
                          file=sys.stderr, flush=True)
        cache_blob = bytes(timing_cache.serialize())
        bytes_saved = len(cache_blob)

    # Sidecar goes right after the plan, before the cache is touched.
    write_sidecar(plan_path, onnx_path, mode, is_static, None, None,
                  None, None, None, opt_level, avg_timing_iterations,
                  tiling_level, error_on_timing_cache_miss, weight_streaming,
                  allow_tf32, timing_cache_path, cache_loaded, bytes_loaded,
                  bytes_saved)

    if cache_blob is not None and timing_cache_path is not None:
        tmp = timing_cache_path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(cache_blob)
        # os.replace() overwrites an existing file on Windows too.
        os.replace(tmp, timing_cache_path)
        print(f"Timing cache saved: {timing_cache_path} "
              f"({len(cache_blob)} bytes)", file=sys.stderr, flush=True)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Build TensorRT engines from the Pair-VPR ViT-B encoder / pair-scorer ONNX")
    p.add_argument("--part", choices=("encoder", "pair", "both"), default="both",
                   help="Which engine(s) to build (default both)")
    p.add_argument("--mode", choices=("fast", "mixed", "precise"), default="fast",
                   help="fast = fp16 baked ONNX (fp32 descriptor and score heads); mixed = fp32 encoder with fp16 "
                        "dense output (the pair part builds as fast); precise = fp32 baked ONNX")
    p.add_argument("--pair-batch", type=int, default=8,
                   help="Candidate count C of the pair-scorer ONNX (names the files Pair<C>)")
    p.add_argument("--model-dir", default=MODEL_DIR,
                   help="Directory of the ONNX files and the output plans (default Model/)")
    p.add_argument("--onnx", default=None,
                   help="Input ONNX file (single --part only; default: derived from --part/--mode)")
    p.add_argument("--out", default=None,
                   help="Output plan path (single --part only; default: derived)")
    p.add_argument("--opt-level", type=int, default=5, choices=range(6),
                   metavar="{0..5}",
                   help="Builder optimization level (0-5, default 5). The "
                        "MAC-VO engines were built at 5 and validated end to "
                        "end; MAC-VO #18 reported corrupted flow at higher "
                        "levels on its build.")
    p.add_argument("--avg-timing", type=int, default=8,
                   help="Timing runs averaged per tactic (trtexec default 8, "
                        "API default 1). Only affects timing-cache misses; the "
                        "existing production tactics were timed at 8.")
    p.add_argument("--tiling-level", type=int, default=0, choices=range(4),
                   metavar="{0..3}",
                   help="Tiling optimization level: 0 none, 1 fast, "
                        "2 moderate, 3 full (default 0)")
    p.add_argument("--error-on-timing-cache-miss", action="store_true",
                   help="Set BuilderFlag.ERROR_ON_TIMING_CACHE_MISS and fail "
                        "the build (no plan written) if the build's timing "
                        "cache grew beyond the loaded bytes (TensorRT timed "
                        "new entries) or TensorRT logged an error or the "
                        "cache-miss warning. Reproduces a plan exactly from a "
                        "populated --timing-cache.")
    p.add_argument("--weight-streaming", dest="weight_streaming",
                   action=argparse.BooleanOptionalAction, default=None,
                   help="Enable BuilderFlag.WEIGHT_STREAMING (default: on for "
                        "both modes). MAC-VO measurements (RTX 4090): on a "
                        "dynamic-axis ONNX TRT 10.13 then builds the all-Myelin graph: 942 vs "
                        "3222 layers, 26.3 vs 38.4 ms per call with a CUDA "
                        "graph (28.4 vs 54.2 without). On the static batch-1 "
                        "ONNX the flag changed nothing "
                        "(same tactics, 964 layers, 23.92-23.98 ms). At the "
                        "default budget nothing is streamed at runtime.")
    p.add_argument("--tf32", dest="allow_tf32",
                   action=argparse.BooleanOptionalAction, default=None,
                   help="Allow TF32 in fp32 matmuls. Default: off in every mode "
                        "(matches PyTorch allow_tf32=False, the parity yardstick). A strongly typed "
                        "build honours TF32 for any fp32 MatMul: the precise/mixed graphs and the "
                        "small fp32 heads (globalizer, classvprmodule) of the fast graphs.")
    p.add_argument("--timing-cache", default=None,
                   help="Path to a timing-cache file (single --part only; default: <plan minus .plan>"
                        ".timing.cache next to the plan), directory must exist. "
                        "Loaded if present, merged with any newer on-disk copy "
                        "and saved atomically after the build. Reuse makes "
                        "tactic choice repeatable in practice (TensorRT does "
                        "not guarantee it); verify by diffing TacticName in "
                        "trtexec --dumpLayerInfo.")
    p.add_argument("--no-timing-cache", action="store_true",
                   help="Build without a timing cache (default: one per plan, next to it)")
    p.add_argument("--parse-only", action="store_true",
                   help=argparse.SUPPRESS)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    # Resolve mode-derived defaults.
    if args.weight_streaming is None:
        args.weight_streaming = True  # on for both modes by default
    if args.avg_timing < 1:
        p.error("--avg-timing must be >= 1")

    parts = ("encoder", "pair") if args.part == "both" else (args.part,)
    if len(parts) > 1 and (args.onnx or args.out or args.timing_cache):
        p.error("--onnx / --out / --timing-cache need a single --part")
    if args.mode == "mixed":
        if args.part == "pair":
            p.error("--mode mixed has no pair scorer of its own: build the pair with --mode fast")
        parts = ("encoder",)
        print("mode mixed: building the encoder only (the pair is --mode fast)", file=sys.stderr, flush=True)
    for part in parts:      # sequential: never two builds at once
        mode = args.mode
        tf32 = bool(args.allow_tf32)
        stem = (f"PairVPR_ViTB_Encoder_{mode}" if part == "encoder"
                else f"PairVPR_ViTB_Pair{args.pair_batch}_{mode}")
        onnx = args.onnx or os.path.join(args.model_dir, stem + ".onnx")
        out = args.out or os.path.join(args.model_dir, stem + f"_o{args.opt_level}.plan")
        if args.no_timing_cache:
            cache = None
        else:
            cache = args.timing_cache or os.path.splitext(out)[0] + ".timing.cache"
        print(f"== {part}: {onnx} -> {out}", file=sys.stderr, flush=True)
        build(
            onnx_path=onnx,
            plan_path=out,
            mode=mode,
            opt_level=args.opt_level,
            avg_timing_iterations=args.avg_timing,
            tiling_level=args.tiling_level,
            error_on_timing_cache_miss=args.error_on_timing_cache_miss,
            weight_streaming=args.weight_streaming,
            allow_tf32=tf32,
            timing_cache_path=cache,
            verbose=args.verbose,
            parse_only=args.parse_only,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
