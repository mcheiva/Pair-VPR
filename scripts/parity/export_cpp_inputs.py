"""Write the reference inputs for the C++ parity tool: python -m scripts.parity.export_cpp_inputs <dir>

From runs/parity/ref.npz (make_reference.py) into <dir>:
  u8_<i>.npy    uint8   [704,704,3]  HWC RGB, the frame as the consumer sees it (cv2 INTER_AREA from the JPEG)
  pre_<i>.npy   float32 [3,322,322]  the reference preprocessing (common.preprocess), to compare the C++ resize +
                                     normalise against
  cand.npy      int64   [N,C]        candidate frame indices per query (row q = the candidates scored against q)
i = frame order of the reference. The tool writes descriptor_<i>.npy / dense_<i>.npy / scores_<q>.npy back (layout
in scripts/parity/compare.py) and `compare --cpp <dir>` scores them.
"""
import argparse
from pathlib import Path

import numpy as np

from .common import REF_DIR


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--ref", type=Path, default=REF_DIR / "ref.npz")
    args = ap.parse_args()
    ref = np.load(args.ref)
    args.out.mkdir(parents=True, exist_ok=True)
    for i, (u8, pre) in enumerate(zip(ref["u8"], ref["pre"])):
        np.save(args.out / f"u8_{i}.npy", np.ascontiguousarray(u8.transpose(1, 2, 0)))
        np.save(args.out / f"pre_{i}.npy", pre.astype(np.float32))
    np.save(args.out / "cand.npy", ref["cand"].astype(np.int64))
    print(f"wrote {len(ref['u8'])} frames and cand {ref['cand'].shape} to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
