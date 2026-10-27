"""Generate the fp32 PyTorch reference set for the parity harness.

    python -m scripts.parity.make_reference                 # 3 dives x 16 frames, C=8 candidates per query
    python -m scripts.parity.make_reference --pair-batch 8 --out runs/parity/ref.npz

Frames: 16 evenly spaced frames per dive (frames that have a GT pose) of vobster_quay/plane_nose,
vobster_quay/plane_tail1 and welland/Welland_test_first_dive -> 48 frames. Each frame is also a query for the
pair scorer, with C candidates: the C/2 GT-nearest frames of the same dive (smallest camera-centre distance;
`near` records whether they also meet the 1.0 m / 30 degree match criterion of eval_eiva_vpr.py, both with
<= like the eval), then the C/4 farthest frames of the same dive (2 for C = 8) and frames of the other dives
to fill the rest (negatives).

The .npz holds (N = 48, C = pair batch):
  u8 [N,3,704,704] uint8      inputs as the consumer sees them (cv2 INTER_AREA from the 2816^2 JPEG)
  pre [N,3,322,322] float32   reference preprocessing (common.preprocess)
  desc [N,512], dense [N,529,768]   fp32 original-forward outputs (TF32 off)
  cand [N,C] int, near [N,C] bool   candidate frame indices per query and their GT label
  scores [N,C] float32        original-forward logit(q,c)+logit(c,q), fp32
  seq [N] str, name [N] str   dive and image name of each frame
"""
import argparse
import time
from pathlib import Path

import numpy as np

from .common import REF_DIR, SEQS, build_fp32, centres_axes, encode_fp32, load_frame_u8, preprocess, read_gt, score_fp32

NEAR_M, NEAR_DEG = 1.0, 30.0


def pick_candidates(seq_of: np.ndarray, centre: np.ndarray, axis: np.ndarray, C: int, rng) -> tuple[np.ndarray, np.ndarray]:
    n = len(seq_of)
    cand = np.zeros((n, C), int)
    near = np.zeros((n, C), bool)
    for q in range(n):
        same = np.where((seq_of == seq_of[q]) & (np.arange(n) != q))[0]
        other = np.where(seq_of != seq_of[q])[0]
        dist = np.linalg.norm(centre[same] - centre[q], axis=1)
        order = same[np.argsort(dist)]
        n_pos = C // 2
        n_far_same = max(1, (C - n_pos) // 2)
        chosen = list(order[:n_pos]) + list(order[::-1][:n_far_same])
        chosen += list(rng.choice(other, C - len(chosen), replace=False))
        cand[q] = chosen
        for j, c in enumerate(chosen):
            if seq_of[c] == seq_of[q]:
                ang = np.degrees(np.arccos(np.clip(axis[q] @ axis[c], -1, 1)))
                near[q, j] = np.linalg.norm(centre[c] - centre[q]) <= NEAR_M and ang <= NEAR_DEG
    return cand, near


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REF_DIR / "ref.npz")
    ap.add_argument("--frames-per-seq", type=int, default=16)
    ap.add_argument("--pair-batch", type=int, default=8)
    args = ap.parse_args()

    u8, seq_of, names, centre, axis = [], [], [], [], []
    for seq in SEQS:
        frames, T, _ = read_gt(seq)
        c, a = centres_axes(T)
        for i in np.linspace(0, len(frames) - 1, args.frames_per_seq).round().astype(int):
            u8.append(load_frame_u8(seq, frames[i]))
            seq_of.append(seq)
            names.append(frames[i])
            centre.append(c[i])
            axis.append(a[i])
        print(f"{seq}: {len(frames)} frames with GT", flush=True)
    seq_of, centre, axis = np.array(seq_of), np.array(centre), np.array(axis)
    u8 = np.stack(u8)
    pre = np.stack([preprocess(x) for x in u8])
    cand, near = pick_candidates(seq_of, centre, axis, args.pair_batch, np.random.default_rng(0))
    print(f"candidates: {near.sum()} of {near.size} are GT-near (<={NEAR_M} m, <={NEAR_DEG} deg)", flush=True)

    model = build_fp32()
    t = time.time()
    desc, dense = encode_fp32(model, pre)
    scores = np.stack([score_fp32(model, dense[q], dense[cand[q]]) for q in range(len(u8))])
    print(f"fp32 reference in {time.time() - t:.1f} s", flush=True)
    pos = scores[near].mean() if near.any() else float("nan")
    print(f"mean score near {pos:.3f} vs not near {scores[~near].mean():.3f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, u8=u8, pre=pre, desc=desc, dense=dense, cand=cand, near=near, scores=scores,
             seq=seq_of, name=np.array(names))
    print("->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
