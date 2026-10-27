"""Compare a candidate (PyTorch wrappers / ONNX / TRT plans / C++ tool outputs) against runs/parity/ref.npz.

    python -m scripts.parity.compare --torch fast --json runs/parity/torch_fast.json
    python -m scripts.parity.compare --torch precise                 # the fp32 wrappers (an exactness check)
    python -m scripts.parity.compare --torch mixed                   # fp32 encoder + fast pair scorer
    python -m scripts.parity.compare --onnx Model/PairVPR_ViTB_Encoder_fast.onnx [--pair-onnx <path>]
    python -m scripts.parity.compare --trt  Model/PairVPR_ViTB_Encoder_fast_o5.plan [--pair-trt <path>] \
        --vs runs/parity/torch_fast.json                             # gate: engine no further from fp32 than torch
    python -m scripts.parity.compare --cpp runs/parity/cpp_out

The pair engine defaults to the encoder's name with Encoder -> Pair<C> (C from the reference file); the mixed
encoder's pair is the fast one (`_mixed` -> `_fast`). `--pair-onnx` / `--pair-trt` override.
Candidates run the full pipeline from the reference's preprocessed input: encode every frame, then score each
query's candidate list with the candidate's OWN dense maps ("e2e"); "iso" scores the pair stage alone on the
fp32 reference dense maps cast to the stage's dtype.

TRT runner: `set_input_shape`, `set_tensor_address`, `execute_async_v3` on a stream, fresh output tensors per
call, the calls the C++ runtime will make.

C++ output layout (--cpp <dir>), written by the later C++ tool, index i = frame order of the reference npz (the
frame at position i of runs/parity/ref.npz; `name`/`seq` arrays), q = query frame index, candidates in the
order of `cand[q]`:
  descriptor_<i>.npy   [512] or [1,512]   float32
  dense_<i>.npy        [529,768] or [1,529,768]   float16 or float32
  scores_<q>.npy       [C]   float32      logit(q,c) + logit(c,q), c = cand[q][j], computed on the tool's dense maps
Reads inputs from the reference's `u8` array (the tool must do its own resize + normalise from uint8 704^2 RGB,
the single preprocessing definition in common.preprocess). `python -m scripts.parity.export_cpp_inputs <dir>`
writes the tool's inputs from the reference: u8_<i>.npy (uint8 [704,704,3] HWC), pre_<i>.npy (float32
[3,322,322], the reference preprocessing, to check the tool's own resize against) and cand.npy (int64 [N,C]).

Metrics (the "worse" direction in brackets): descriptor cosine min/mean (lower), dense per-token cosine p01/min
(lower), score |delta| max/p99 (higher), retrieval top-1 agreement and top-5 overlap of the descriptor ranking
over the reference set, self excluded (lower), Kendall tau of the candidate order per query, mean/min (lower).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .common import DEVICE, REF_DIR, build_wrappers

# metric -> True if larger is worse
WORSE_HIGH = {"desc_cos_min": False, "desc_cos_mean": False, "dense_cos_p01": False, "dense_cos_min": False,
              "score_abs_max": True, "score_abs_p99": True, "iso_abs_max": True, "iso_abs_p99": True,
              "top1_agree": False, "top5_overlap": False, "tau_mean": False, "tau_min": False}


class TorchCand:
    def __init__(self, mode: str, C: int):
        self.enc, self.pair = build_wrappers(mode, C)
        self.dtype = self.pair.dtype

    @torch.inference_mode()
    def encode(self, pre: np.ndarray):
        d, f = [], []
        for x in pre:
            desc, dense = self.enc(torch.from_numpy(x)[None].to(DEVICE))
            d.append(desc.float().cpu().numpy()[0])
            f.append(dense.float().cpu().numpy()[0])
        return np.stack(d), np.stack(f)

    @torch.inference_mode()
    def score(self, q: np.ndarray, c: np.ndarray):
        out = self.pair(torch.from_numpy(q)[None].to(DEVICE, self.dtype), torch.from_numpy(c).to(DEVICE, self.dtype))
        return out.float().cpu().numpy()


class OnnxCand:
    def __init__(self, enc_path: Path, pair_path: Path):
        import onnxruntime as ort
        prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self.enc = ort.InferenceSession(str(enc_path), providers=prov)
        self.pair = ort.InferenceSession(str(pair_path), providers=prov)
        print("onnxruntime providers:", self.enc.get_providers(), flush=True)
        self.dt = np.float16 if "float16" in self.pair.get_inputs()[0].type else np.float32

    def encode(self, pre):
        d, f = [], []
        for x in pre:
            desc, dense = self.enc.run(None, {"image": x[None]})
            d.append(desc[0].astype(np.float32))
            f.append(dense[0].astype(np.float32))
        return np.stack(d), np.stack(f)

    def score(self, q, c):
        return self.pair.run(None, {"query": q[None].astype(self.dt), "candidates": c.astype(self.dt)})[0]


class TrtRunner:
    def __init__(self, path: Path):
        import tensorrt as trt
        self.trt = trt
        self.engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(Path(path).read_bytes())
        assert self.engine is not None, f"failed to deserialize {path}"
        self.ctx = self.engine.create_execution_context()
        self.io = [(self.engine.get_tensor_name(i), self.engine.get_tensor_mode(self.engine.get_tensor_name(i)))
                   for i in range(self.engine.num_io_tensors)]

    def torch_dtype(self, name):
        return {self.trt.float32: torch.float32, self.trt.float16: torch.float16}[self.engine.get_tensor_dtype(name)]

    def __call__(self, inputs: dict) -> dict:
        keep = []
        for name, mode in self.io:
            if mode == self.trt.TensorIOMode.INPUT:
                t = torch.from_numpy(inputs[name]).to(DEVICE, self.torch_dtype(name)).contiguous()
                keep.append(t)
                self.ctx.set_input_shape(name, tuple(t.shape))
                self.ctx.set_tensor_address(name, t.data_ptr())
        outs = {}
        for name, mode in self.io:
            if mode == self.trt.TensorIOMode.OUTPUT:
                outs[name] = torch.empty(tuple(self.ctx.get_tensor_shape(name)), device=DEVICE,
                                         dtype=self.torch_dtype(name))
                self.ctx.set_tensor_address(name, outs[name].data_ptr())
        assert self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream), "execute_async_v3 failed"
        torch.cuda.synchronize()
        return {k: v.float().cpu().numpy() for k, v in outs.items()}


class TrtCand:
    def __init__(self, enc_path: Path, pair_path: Path):
        self.enc, self.pair = TrtRunner(enc_path), TrtRunner(pair_path)

    def encode(self, pre):
        d, f = [], []
        for x in pre:
            o = self.enc({"image": x[None]})
            d.append(o["descriptor"][0])
            f.append(o["dense"][0])
        return np.stack(d), np.stack(f)

    def score(self, q, c):
        return self.pair({"query": q[None], "candidates": c})["score"]


class CppCand:
    """Reads the C++ tool's .npy files (layout in the module docstring)."""

    def __init__(self, folder: Path):
        self.dir = Path(folder)

    def encode(self, pre):
        d = [np.load(self.dir / f"descriptor_{i}.npy").reshape(512).astype(np.float32) for i in range(len(pre))]
        f = [np.load(self.dir / f"dense_{i}.npy").reshape(529, 768).astype(np.float32) for i in range(len(pre))]
        return np.stack(d), np.stack(f)

    def score(self, q, c):
        raise NotImplementedError

    def e2e_scores(self, n: int):
        return np.stack([np.load(self.dir / f"scores_{q}.npy").reshape(-1).astype(np.float32) for q in range(n)])


def kendall(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import kendalltau
    return float(kendalltau(a, b)[0])      # NaN (constant input) stays NaN, so distance_ok marks it worse


def metrics(ref: dict, desc, dense, scores, iso=None) -> dict:
    n = len(desc)
    m = {}
    cos = np.sum(desc * ref["desc"], axis=1) / (np.linalg.norm(desc, axis=1) * np.linalg.norm(ref["desc"], axis=1))
    m["desc_cos_min"], m["desc_cos_mean"] = float(cos.min()), float(cos.mean())
    a, b = dense.astype(np.float64), ref["dense"].astype(np.float64)
    tc = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    m["dense_cos_p01"], m["dense_cos_min"] = float(np.quantile(tc, 0.01)), float(tc.min())
    d = np.abs(scores - ref["scores"])
    m["score_abs_max"], m["score_abs_p99"] = float(d.max()), float(np.quantile(d, 0.99))
    if iso is not None:
        di = np.abs(iso - ref["scores"])
        m["iso_abs_max"], m["iso_abs_p99"] = float(di.max()), float(np.quantile(di, 0.99))
    sim_r, sim_c = ref["desc"] @ ref["desc"].T, desc @ desc.T
    np.fill_diagonal(sim_r, -np.inf)
    np.fill_diagonal(sim_c, -np.inf)
    top_r, top_c = np.argsort(-sim_r, 1)[:, :5], np.argsort(-sim_c, 1)[:, :5]
    m["top1_agree"] = float(np.mean(top_r[:, 0] == top_c[:, 0]))
    m["top5_overlap"] = float(np.mean([len(set(top_r[i]) & set(top_c[i])) / 5 for i in range(n)]))
    taus = [kendall(scores[q], ref["scores"][q]) for q in range(n)]
    m["tau_mean"], m["tau_min"] = float(np.mean(taus)), float(np.min(taus))
    return m


def distance_ok(m: dict, base: dict) -> dict:
    """metric -> True if `m` is no further from fp32 than `base` (a small epsilon absorbs float noise)."""
    out = {}
    for k, high in WORSE_HIGH.items():
        if k in m and k in base:
            eps = 1e-6 * max(1.0, abs(base[k]))
            out[k] = (m[k] <= base[k] + eps) if high else (m[k] >= base[k] - eps)
    return out


def default_pair(path: Path, C: int) -> Path:
    """Encoder_<mode> -> Pair<C>_<mode>; the mixed encoder pairs with the fast pair scorer."""
    return path.with_name(path.name.replace("Encoder", f"Pair{C}").replace("_mixed", "_fast"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--torch", choices=["fast", "mixed", "precise"])
    g.add_argument("--onnx", type=Path, help="encoder ONNX")
    g.add_argument("--trt", type=Path, help="encoder plan")
    g.add_argument("--cpp", type=Path, help="folder of .npy outputs of the C++ tool")
    ap.add_argument("--pair-onnx", type=Path)
    ap.add_argument("--pair-trt", type=Path)
    ap.add_argument("--ref", type=Path, default=REF_DIR / "ref.npz")
    ap.add_argument("--vs", type=Path, help="baseline --json (e.g. torch fast): report whether each metric is no "
                                            "further from fp32 than the baseline")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    ref = dict(np.load(args.ref))
    n, C = ref["cand"].shape
    cand_idx = ref["cand"]
    if args.torch:
        cand = TorchCand(args.torch, C)
    elif args.onnx:
        cand = OnnxCand(args.onnx, args.pair_onnx or default_pair(args.onnx, C))
    elif args.trt:
        cand = TrtCand(args.trt, args.pair_trt or default_pair(args.trt, C))
    else:
        cand = CppCand(args.cpp)

    desc, dense = cand.encode(ref["pre"])
    if args.cpp:
        scores, iso = cand.e2e_scores(n), None
    else:
        scores = np.stack([cand.score(dense[q], dense[cand_idx[q]]) for q in range(n)])
        iso = np.stack([cand.score(ref["dense"][q], ref["dense"][cand_idx[q]]) for q in range(n)])
    m = metrics(ref, desc, dense, scores, iso)

    for k, v in m.items():
        print(f"{k:16s} {v:14.6g}")
    out = {"metrics": m}
    if args.vs:
        base = json.loads(args.vs.read_text())["metrics"]
        ok = distance_ok(m, base)
        print(f"\nvs {args.vs.name}: " + ", ".join(f"{k} {'ok' if v else 'WORSE'}" for k, v in ok.items()))
        out["vs"] = {"baseline": str(args.vs), "no_worse": ok, "all_ok": all(ok.values())}
        print("ALL OK" if out["vs"]["all_ok"] else "SOME WORSE")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=1))
        print("->", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
