"""Place-recognition recall on the EIVA dives with GT poses: torch fp32 vs the exported engines.

    python -m scripts.eval_eiva_vpr --fp32 --trt Model/PairVPR_ViTB_Encoder_fast_o5.plan \
        --seqs vobster_quay/plane_nose vobster_quay/plane_tail1 welland/Welland_test_first_dive --json runs/parity/recall.json

Definition (one dive at a time; frames = the frames of pose_gt.txt whose image exists, in file order):
  * database = every STRIDE-th frame of the dive (STRIDE 4 by default); queries = the same frames. A query's
    candidates exclude itself and every database frame whose TIMESTAMP is within MIN_GAP_S seconds of the query's
    (default 60 s; timestamps parsed from the image names as MACVO_TRT scripts/gt_to_tum.py does), so a match must
    be a revisit, not the same moment seen again. (A fixed 16-index gap, the first protocol, is about 8 s on
    Welland, so 588 of 590 queries "matched" frames seconds apart; it stays available as --index-gap N and is
    only a parity check between engines, not a recall measurement.)
  * a database frame is a TRUE match for a query when the GT camera centres are within 1.0 m AND the optical axes
    (camera +z of T_i = H_0 inv(H_i), see common.read_gt) are within 30 degrees (both inclusive).
  * queries with no true match in the gap-filtered database are skipped (the revisit-query count is printed per
    dive); every backend sees the same set.
  * Recall@K (K = 1, 5, 10): a query counts if any of its top-K contains a true match.
      global     ranking by descriptor cosine (what faiss L2 on the L2-normalised descriptors gives in eval.py)
      reranked   the top-C global candidates (C = the pair batch, 8) are re-ordered by the summed both-order
                 pair score (eval.py:172-198 does this for the top 100); ranks beyond C keep the global order,
                 so Recall@10 mixes the reranked top 8 with global ranks 9-10.
  * Welland is long (4718 GT frames): its stride is --welland-stride (default 8, so 590 frames) to keep the
    fp32 reference run to a few minutes; the other dives use --stride (default 4).

Backends: --fp32 (original PairVPRNet, fp32, TF32 off), --torch fast|mixed|precise (the export wrappers), --onnx <enc>,
--trt <enc plan> (the pair plan name is derived: Encoder -> Pair<C>).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from .parity.common import (DEVICE, REF_DIR, build_fp32, centres_axes, encode_fp32, load_frame_u8, pre_version, preprocess,
                            read_gt, score_fp32)
from .parity.compare import OnnxCand, TorchCand, TrtCand, default_pair

NEAR_M, NEAR_DEG = 1.0, 30.0
KS = (1, 5, 10)


class Fp32Cand:
    def __init__(self):
        self.m = build_fp32()

    def encode(self, pre):
        return encode_fp32(self.m, pre)

    def score(self, q, c):
        return score_fp32(self.m, q, c)


def load_pre(seq: str, idx: np.ndarray, names: list[str]) -> np.ndarray:
    cache = REF_DIR / f"eiva_pre_{pre_version()}_{seq.replace('/', '_')}_{len(idx)}_{idx[0]}_{idx[-1]}.npy"
    if cache.exists():
        return np.load(cache)
    pre = np.stack([preprocess(load_frame_u8(seq, names[i])) for i in idx])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, pre)
    return pre


def true_matches(centre, axis, ts, idx, min_gap_s, index_gap):
    """([M, M] true match, [M, M] allowed candidate): db j for query i, gap filter applied."""
    d = np.linalg.norm(centre[:, None] - centre[None], axis=-1)
    ang = np.degrees(np.arccos(np.clip(axis @ axis.T, -1, 1)))
    if index_gap:
        far = np.abs(idx[:, None] - idx[None]) >= index_gap
    else:
        far = np.abs(ts[:, None] - ts[None]) >= min_gap_s
    return (d <= NEAR_M) & (ang <= NEAR_DEG) & far, far


def recall(rank: np.ndarray, match: np.ndarray, qs: np.ndarray) -> dict:
    out = {}
    for k in KS:
        out[f"R@{k}"] = float(np.mean([match[q, rank[q, :k]].any() for q in qs])) if len(qs) else float("nan")
        out[f"hits@{k}"] = int(sum(match[q, rank[q, :k]].any() for q in qs))
    return out


def evaluate(cand, pre, match, allowed, C) -> dict:
    t = time.time()
    desc, dense = cand.encode(pre)
    t_enc = time.time() - t
    sim = desc @ desc.T
    sim[~allowed] = -np.inf
    rank = np.argsort(-sim, axis=1)[:, :max(KS)]
    qs = np.where(match.any(1))[0]
    res = {"queries": int(len(qs)), "global": recall(rank, match, qs)}
    t = time.time()
    reranked = rank.copy()
    for q in qs:
        top = rank[q, :C]
        s = cand.score(dense[q], dense[top])
        reranked[q, :C] = top[np.argsort(-s, kind="stable")]
    res["reranked"] = recall(reranked, match, qs)
    res["seconds"] = {"encode": round(t_enc, 1), "rerank": round(time.time() - t, 1)}
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seqs", nargs="+", default=["vobster_quay/plane_nose", "vobster_quay/plane_tail1",
                                                  "welland/Welland_test_first_dive"])
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--welland-stride", type=int, default=8)
    ap.add_argument("--min-gap-s", type=float, default=60.0,
                    help="exclude database frames within this many seconds of the query (default 60)")
    ap.add_argument("--index-gap", type=int, default=0,
                    help="parity-check protocol: exclude by |frame index gap| < N instead of by time (e.g. 16)")
    ap.add_argument("--pair-batch", type=int, default=8, help="rerank top-K = C of the pair engine")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--torch", choices=["fast", "mixed", "precise"], action="append", default=[])
    ap.add_argument("--onnx", type=Path, action="append", default=[])
    ap.add_argument("--trt", type=Path, action="append", default=[])
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    C = args.pair_batch

    backends = {}
    if args.fp32:
        backends["fp32"] = Fp32Cand
    for m in args.torch:
        backends[f"torch-{m}"] = lambda m=m: TorchCand(m, C)
    for p in args.onnx:
        backends[f"onnx:{p.name}"] = lambda p=p: OnnxCand(p, default_pair(p, C))
    for p in args.trt:
        backends[f"trt:{p.name}"] = lambda p=p: TrtCand(p, default_pair(p, C))

    results = {}
    for seq in args.seqs:
        names, T, ts = read_gt(seq)
        stride = args.welland_stride if "welland" in seq.lower() else args.stride
        idx = np.arange(0, len(names), stride)
        centre, axis = centres_axes(T[idx])
        match, allowed = true_matches(centre, axis, ts[idx], idx, args.min_gap_s, args.index_gap)
        np.fill_diagonal(allowed, False)
        print(f"\n== {seq}: {len(names)} GT frames, stride {stride} -> {len(idx)} db/query frames, "
              f"{int(match.any(1).sum())} revisit queries (true match >= gap away)", flush=True)
        if not match.any():
            print("   no revisit queries on this dive; skipped", flush=True)
            results[seq] = {"frames": len(idx), "stride": stride, "queries": 0}
            continue
        pre = load_pre(seq, idx, names)
        results[seq] = {"frames": len(idx), "stride": stride}
        for name, make in backends.items():
            cand = make()
            r = evaluate(cand, pre, match, allowed, C)
            results[seq][name] = r
            g, rr = r["global"], r["reranked"]
            print(f"{name:42s} global R@1/5/10 {g['hits@1']}/{g['hits@5']}/{g['hits@10']}   "
                  f"reranked {rr['hits@1']}/{rr['hits@5']}/{rr['hits@10']}   of {r['queries']} queries  "
                  f"({r['seconds']})", flush=True)
            del cand
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=1))
        print("->", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
