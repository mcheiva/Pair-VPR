"""Self-checks for the export rewrites. Run: python -m scripts.parity.test_exact

  1. Encoder / PairScorer wrappers (fp32, 'precise') vs the original PairVPRNet.forward built with the LIVE
     pos-embed interpolation (global mode and pairvpr mode, both orders summed): max |diff| < 1e-5 for the
     descriptor and dense map, and for the score.
  2. Frozen pos-embed == the live interpolation, bit-identical (torch.equal) on prepare_tokens_with_masks of real
     images, live model vs frozen model; plus the checkpoint's dec_pos_embed_cls == cat(zero row, dec_pos_embed).
  3. Fused symmetric scorer == two separate original calls (covered by 1; also checked on random dense maps).
  4. Reference preprocessing vs Pair-VPR's PIL pipeline (PIL Resize BILINEAR + ToTensor + Normalize): REPORT ONLY.
"""
import sys

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from .common import DEVICE, REF_DIR, build_model, encode_fp32, preprocess, score_fp32, DATA_ROOT
from pairvpr.export.patches import CKPT, EMBED, IMG_RES, NUM_TOKENS
from pairvpr.export.wrappers import Encoder, PairScorer

FAILS = []


def check(name: str, value: float, limit: float) -> None:
    ok = value < limit
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {value:.3e} (< {limit:.0e})", flush=True)
    if not ok:
        FAILS.append(name)


@torch.inference_mode()
def main() -> int:
    # The wrappers and the original run under the same (MATH) SDPA backend, like the fp32 reference: the test is about
    # the rewrites, not about which fused fp32 attention kernel PyTorch picks (those differ by ~1e-3 on dense).
    with sdpa_kernel(SDPBackend.MATH):
        return run()


def run() -> int:
    ref = np.load(REF_DIR / "ref.npz")
    pre = ref["pre"][::6]                                   # 8 frames
    orig = build_model("precise", DEVICE, freeze=False)     # original network, live interpolation
    enc = Encoder(build_model("precise", DEVICE), "precise").eval()
    pair = PairScorer(build_model("precise", DEVICE), "precise").eval()

    # 1. wrappers vs original
    desc_o, dense_o = encode_fp32(orig, pre)
    d_err = f_err = 0.0
    for x, do, fo in zip(pre, desc_o, dense_o):
        d, f = enc(torch.from_numpy(x)[None].to(DEVICE))
        d_err = max(d_err, (d.cpu().numpy()[0] - do).__abs__().max())
        f_err = max(f_err, (f.cpu().numpy()[0] - fo).__abs__().max())
    check("encoder descriptor vs original forward(global), max abs", float(d_err), 1e-5)
    check("encoder dense vs original forward(global), max abs", float(f_err), 1e-5)

    s_err = 0.0
    g = torch.Generator().manual_seed(0)
    for q in range(3):                                      # real dense maps
        cands = dense_o[[(q + 1 + j) % len(dense_o) for j in range(8)]]
        want = score_fp32(orig, dense_o[q], cands)
        got = pair(torch.from_numpy(dense_o[q])[None].to(DEVICE), torch.from_numpy(cands).to(DEVICE)).cpu().numpy()
        s_err = max(s_err, float(np.abs(want - got).max()))
    check("pair scorer (fused 2C batch) vs two original calls summed, max abs", s_err, 1e-5)
    rq = torch.randn(NUM_TOKENS, EMBED, generator=g).numpy()
    rc = torch.randn(8, NUM_TOKENS, EMBED, generator=g).numpy()
    want = score_fp32(orig, rq, rc)
    got = pair(torch.from_numpy(rq)[None].to(DEVICE), torch.from_numpy(rc).to(DEVICE)).cpu().numpy()
    check("fused scorer vs two separate calls on random dense maps, max abs (scale %.1f)" % np.abs(want).max(),
          float(np.abs(want - got).max()), 1e-5)

    # 2. frozen pos-embed vs the live interpolation, on the real token path (prepare_tokens_with_masks of real
    # images through the live model and the frozen model; the live model never saw the frozen method)
    live_vit, frozen_vit = orig.encoder.model, enc.model.encoder.model
    same = frozen_vit.frozen_pos_embed.shape == (1, NUM_TOKENS + 1, EMBED)
    for x in torch.from_numpy(pre[:4]).to(DEVICE):
        same &= torch.equal(live_vit.prepare_tokens_with_masks(x[None], None),
                            frozen_vit.prepare_tokens_with_masks(x[None], None))
    print(f"{'PASS' if same else 'FAIL'}  prepare_tokens_with_masks: frozen [1,530,768] pos-embed bit-identical to "
          "the live interpolation (live model vs frozen model, real images)", flush=True)
    if not same:
        FAILS.append("pos-embed")

    # 3. checkpoint invariant the decoder relies on: dec_pos_embed_cls == cat(zero row, dec_pos_embed)
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)
    cls_ok = torch.equal(sd["dec_pos_embed_cls"][0], torch.zeros_like(sd["dec_pos_embed_cls"][0])) and         torch.equal(sd["dec_pos_embed_cls"][1:], sd["dec_pos_embed"])
    print(f"{'PASS' if cls_ok else 'FAIL'}  checkpoint dec_pos_embed_cls == cat(zero row, dec_pos_embed)", flush=True)
    if not cls_ok:
        FAILS.append("dec_pos_embed_cls")

    # 4. preprocessing report (no gate)
    from PIL import Image
    import torchvision.transforms as T
    pil_tf = T.Compose([T.Resize((IMG_RES, IMG_RES), interpolation=T.InterpolationMode.BILINEAR), T.ToTensor(),
                        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    print("\nreference preprocessing vs Pair-VPR PIL pipeline (report only):")
    for label, make in (("from the uint8 704^2 frame", lambda u8, p: Image.fromarray(u8.transpose(1, 2, 0))),
                        ("from the full 2816^2 JPEG", lambda u8, p: Image.open(p).convert("RGB"))):
        diffs, dd = [], []
        for i in range(0, len(ref["u8"]), 12):
            path = DATA_ROOT / str(ref["seq"][i]) / "processed" / "left" / str(ref["name"][i])
            a = pil_tf(make(ref["u8"][i], path)).numpy()
            b = preprocess(ref["u8"][i])
            diffs.append((a - b))
            d1, _ = encode_fp32(orig, np.stack([a]))
            d2, _ = encode_fp32(orig, np.stack([b]))
            dd.append(float(d1[0] @ d2[0]))
        e = np.abs(np.stack(diffs))
        print(f"  {label}: |dx| mean {e.mean():.4f} max {e.max():.3f} (normalised units); "
              f"descriptor cosine PIL vs reference min {min(dd):.5f}", flush=True)

    print("\nALL PASS" if not FAILS else f"\nFAILED: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
