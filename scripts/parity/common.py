"""Shared helpers for the Pair-VPR parity harness (mirrors MAV-VO-MCH Scripts/Parity/common.py).

Run everything from the repo root: `python -m scripts.parity.<module>`.

The reference preprocessing defined here is THE one definition the C++ consumer must match:
  uint8 704x704 RGB (cv2 INTER_AREA from the 2816x2816 processed JPEG) -> float [0,1]
  -> torchvision resize to 322x322, bilinear, antialias=True (a squash, like Pair-VPR's PIL Resize)
  -> ImageNet normalise (mean 0.485/0.456/0.406, std 0.229/0.224/0.225).
Pair-VPR's own eval pipeline (PIL Resize BILINEAR on the original JPEG) differs by interpolation rounding and
by starting from the full-resolution image; test_exact.py reports the gap, with no gate.

Domain note: this harness feeds cv2 INTER_AREA 704x704 frames, while the C++ loader decodes the JPEGs with nvJPEG's
1/4 DCT-scaled decode. At runtime MAC-VO and Pair-VPR both see the loader's images, so engine parity (what this
harness measures: same input, different engine) is unaffected; only the absolute recall numbers belong to this
input path.

The fp32 reference runs with the MATH scaled-dot-product-attention backend (DINOv2 attention calls SDPA), so it
does not depend on which fused kernel PyTorch picks.
"""
import hashlib
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from pairvpr.export.patches import CKPT, EMBED, IMG_RES, NUM_TOKENS, ROOT, build_model  # noqa: F401
from pairvpr.export.wrappers import Encoder, PairScorer  # noqa: F401

torch.backends.cudnn.allow_tf32 = False          # the harness means fp32 when it says fp32
torch.backends.cuda.matmul.allow_tf32 = False

RES = 704
DATA_ROOT = Path("D:/Datasets/EIVA")
SEQS = ["vobster_quay/plane_nose", "vobster_quay/plane_tail1", "welland/Welland_test_first_dive"]
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
REF_DIR = ROOT / "runs" / "parity"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def pre_version() -> str:
    """Hash of everything the reference preprocessing depends on (cache key of cached preprocessed frames)."""
    import cv2
    import torchvision
    spec = (RES, IMG_RES, MEAN, STD, "cv2.INTER_AREA", "tv.resize bilinear antialias=True", "tv.normalize",
            torchvision.__version__, cv2.__version__)
    return hashlib.sha1(repr(spec).encode()).hexdigest()[:8]


def build_fp32(device: str = DEVICE) -> torch.nn.Module:
    """The original PairVPRNet (frozen pos-embed), fp32. Reference for everything."""
    return build_model("precise", device)


def build_wrappers(mode: str, pair_batch: int = 8, device: str = DEVICE) -> tuple[Encoder, PairScorer]:
    """(Encoder, PairScorer) for 'fast', 'mixed' (fp32 encoder, fast pair scorer) or 'precise' (separate model
    copies: the wrappers cast in place)."""
    pair_mode = "fast" if mode == "mixed" else mode
    return (Encoder(build_model(mode, device), mode).eval(), PairScorer(build_model(pair_mode, device), pair_mode).eval())


# Timestamp parsing as MACVO_TRT scripts/gt_to_tum.py:parse_ts: datetime forms (optional D / SYSTEM_ prefix,
# fractional separator '.' or '-', 3-6 digits) or unix nanoseconds.
_TS_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})T(\d{2})-?(\d{2})-?(\d{2})[.\-](\d{3,6})")
_UNIX_NS_RE = re.compile(r"(?<!\d)(\d{16,19})(?!\d)")


def parse_ts(name: str) -> float:
    m = _TS_RE.search(name)
    if m:
        yr, mo, dy, hh, mm, ss = m.group(1, 2, 3, 4, 5, 6)
        frac = m.group(7).ljust(6, "0")
        return datetime.fromisoformat(f"{yr}-{mo}-{dy}T{hh}:{mm}:{ss}.{frac}+00:00").timestamp()
    m = _UNIX_NS_RE.search(name)
    if m:
        return int(m.group(1)) / 1e9
    raise ValueError(f"no timestamp in {name}")


def read_gt(seq: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Frames of a dive that have a GT pose: (left image names, T [N,4,4], timestamps [N] in seconds).

    pose_gt.txt line: `<image name> x y z r00 r01 r02 r10 r11 r12 r20 r21 r22` = H_target_world (world ->
    camera). As in MACVO_TRT scripts/gt_to_tum.py, T_i = H_0 @ inv(H_i) is the pose of camera i in the first
    camera's frame: centre = T_i[:3,3], optical axis (camera +z) = T_i[:3,:3] @ [0,0,1]."""
    names, Hs = [], []
    with open(DATA_ROOT / seq / "pose_gt.txt") as f:
        for line in f:
            parts = line.split()
            if not parts or line.startswith("#"):
                continue
            nums = list(map(float, parts[1:13]))
            H = np.eye(4)
            H[:3, :3] = np.array(nums[3:12]).reshape(3, 3)
            H[:3, 3] = nums[0:3]
            names.append(parts[0])
            Hs.append(H)
    H0 = Hs[0]
    T = np.stack([H0 @ np.linalg.inv(H) for H in Hs])
    left = DATA_ROOT / seq / "processed" / "left"
    keep = [i for i, n in enumerate(names) if (left / n).exists()]
    return [names[i] for i in keep], T[keep], np.array([parse_ts(names[i]) for i in keep])


def centres_axes(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return T[:, :3, 3], T[:, :3, :3] @ np.array([0.0, 0.0, 1.0])


def load_image_u8(path: Path) -> np.ndarray:
    """-> uint8 [3,704,704] RGB. cv2 INTER_AREA from the 2816x2816 JPEG; candidates see these exact bytes."""
    import cv2
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert im is not None, path
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (RES, RES), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(im.transpose(2, 0, 1))


def load_frame_u8(seq: str, name: str) -> np.ndarray:
    return load_image_u8(DATA_ROOT / seq / "processed" / "left" / name)


def preprocess(u8: np.ndarray) -> np.ndarray:
    """REFERENCE preprocessing: uint8 [3,H,W] -> float32 [3,322,322] normalised (see module docstring)."""
    import torchvision.transforms.functional as TF
    x = torch.from_numpy(u8).float().div_(255.0)
    x = TF.resize(x, [IMG_RES, IMG_RES], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
    x = TF.normalize(x, MEAN, STD)
    return x.numpy()


@torch.inference_mode()
def encode_fp32(model, pre: np.ndarray, device: str = DEVICE) -> tuple[np.ndarray, np.ndarray]:
    """Original forward(mode='global') one image at a time: (descriptor [N,512], dense [N,529,768]) fp32."""
    d, f = [], []
    for x in pre:
        with sdpa_kernel(SDPBackend.MATH):
            dense, desc = model(torch.from_numpy(x)[None].to(device), None, mode="global")
        d.append(desc.float().cpu().numpy()[0])
        f.append(dense.float().cpu().numpy()[0])
    return np.stack(d), np.stack(f)


@torch.inference_mode()
def score_fp32(model, q_dense: np.ndarray, c_dense: np.ndarray, device: str = DEVICE) -> np.ndarray:
    """Original forward(mode='pairvpr') in both orders, summed: [C] fp32 (Pair-VPR eval.py:172-198)."""
    q = torch.from_numpy(q_dense)[None].to(device).expand(len(c_dense), -1, -1)
    c = torch.from_numpy(c_dense).to(device)
    a = model(q, c, "pairvpr")
    b = model(c, q, "pairvpr")
    return (a + b).squeeze(1).float().cpu().numpy()
