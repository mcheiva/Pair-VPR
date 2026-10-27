"""Parity of the MACVO_TRT C++ Pair-VPR place-recognition RANKING against the Pair-VPR repo's own Python pipeline.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python -m scripts.parity.ranking_parity \
        --mode all --json runs/parity/ranking_parity.json

Inputs: the C++ detect logs `<stem>.places.csv` (place_seq, frame_id, t, median_depth) and `<stem>.loops.csv`
(one row per query: n_eligible, best_*, topk = `frame:cosine:score` items joined by `|` in cosine-rank order,
verified, accepted). The queries are the places themselves (the C++ queries every place_stride-th frame, then
stores it). Frame id -> image: the sorted `processed/left/*.jpg` of the dataset (checked by timestamp, see
`check_frame_mapping`).

What the C++ does per query at seq q (place_recognizer.cpp): eligible places = place_seq + G <= q (prefix of the
places); rank by cosine of the L2-normalised 512-d descriptors, ties to the lower place index; keep top K = 5;
score them with the pair scorer (logit(q,c) + logit(c,q)); best = argmax score, first maximum (better cosine rank)
wins ties. G is read from the run: the CSV's n_eligible fixes it (the brief said 32; the detect_all2 runs are
consistent only with G in 9..16, see the report of `consistency`), so --gap defaults to the largest G the data
allows and is checked.

Modes (--mode):
  consistency  C++ CSV internal consistency (no GPU)
  parity       pipelines A (repo path), B (C++-like input + TRT engines) and the ablations, vs the C++ CSV
  all          both

Pipelines:
  A       PIL open(jpg).convert('RGB') at full resolution -> the repo's eval transform (Resize((322,322)) bilinear,
          ToTensor, Normalize) -> PairVPRNet fp32 (TF32 off, MATH sdpa) via pairvpr.export.patches.build_model.
  B       cv2.imread(IMREAD_REDUCED_COLOR_4) (libjpeg DCT scaling, ~ nvJPEG's), RGB -> common.preprocess ->
          TRT Encoder_mixed_o5 + Pair8_fast_o5 (what the C++ runs), candidates padded to 8 with the last one.
  A_on_B  A's model on B's inputs (splits the deviation: decode/resize vs precision)
  B_on_A  B's engines on A's inputs
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from .common import DATA_ROOT, DEVICE, ROOT, build_fp32, parse_ts, preprocess
from .compare import TrtCand

CPP_ROOT = Path("C:/Users/mch/Documents/git/MACVO_TRT")
RUNS = {
    "plane_nose": dict(seq="vobster_quay/plane_nose", stem=CPP_ROOT / "runs/p4/plane_nose/detect_all2"),
    "welland": dict(seq="welland/Welland_test_first_dive", stem=CPP_ROOT / "runs/p4cal/welland/detect_all2"),
}
PAIR_C = 8
ENC_PLAN = "PairVPR_ViTB_Encoder_mixed_o5.plan"
PAIR_PLAN = "PairVPR_ViTB_Pair8_fast_o5.plan"


# ------------------------------------------------------------------ CSV
def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_run(stem: Path):
    places = [dict(seq=int(r["place_seq"]), frame=int(r["frame_id"]), t=float(r["t"])) for r in read_csv(f"{stem}.places.csv")]
    loops = []
    for r in read_csv(f"{stem}.loops.csv"):
        items = []
        for it in r["topk"].split("|"):
            fr, c, s = it.split(":")
            items.append((int(fr), float(c), float(s)))
        loops.append(dict(seq=int(r["query_seq"]), frame=int(r["query_frame"]), t=float(r["query_t"]),
                          n_eligible=int(r["n_eligible"]), best_frame=int(r["best_place_frame"]),
                          best_cos_rank=int(r["best_cos_rank"]), best_score=float(r["best_score"]), topk=items,
                          verified=int(r["verified"]), accepted=int(r["accepted"])))
    return places, loops


def infer_gap(places, loops):
    """G values (>= 0) for which n_eligible of every query equals #{places: seq + G <= q}."""
    ok = []
    for g in range(0, 129):
        if all(sum(p["seq"] + g <= l["seq"] for p in places) == l["n_eligible"] for l in loops):
            ok.append(g)
    return ok


# ------------------------------------------------------------------ frame -> image
def left_images(seq: str):
    d = DATA_ROOT / seq / "processed" / "left"
    return sorted(d / n for n in __import__("os").listdir(d) if n.endswith(".jpg"))


def check_frame_mapping(seq, places, loops):
    """C++ DatasetLoader: sorted *.jpg of processed/left, id = running counter over successfully decoded frames
    (cnt++), so id == index into the sorted list unless a frame failed to decode or had no timestamp. Verified here
    by comparing every place's / query's logged timestamp with the timestamp parsed from the file name at that index."""
    imgs = left_images(seq)
    worst = 0.0
    bad = []
    for r in [dict(frame=p["frame"], t=p["t"], seq=p["seq"]) for p in places] + loops:
        if r["frame"] >= len(imgs):
            bad.append((r["frame"], "out of range"))
            continue
        dt = abs(parse_ts(imgs[r["frame"]].name) - r["t"])
        worst = max(worst, dt)
        if dt > 1e-3:
            bad.append((r["frame"], dt))
    seq_eq_frame = all(p["seq"] == p["frame"] for p in places)
    return dict(n_images=len(imgs), n_rows_checked=len(places) + len(loops), max_abs_dt_s=worst, n_bad=len(bad),
                bad_first=bad[:5], place_seq_equals_frame_id=seq_eq_frame), imgs


# ------------------------------------------------------------------ check 1: C++ internal consistency
def consistency(name, places, loops, gap_arg, k_arg):
    k_top = k_arg if k_arg is not None else max(len(l["topk"]) for l in loops)   # K the run used
    gaps = infer_gap(places, loops)
    gap = gap_arg if gap_arg is not None else (max(gaps) if gaps else 32)
    seq_of = {p["frame"]: p["seq"] for p in places}
    idx_of = {p["frame"]: i for i, p in enumerate(places)}
    bad = {k: [] for k in ("cos_not_descending", "tie_order", "best_not_argmax", "best_rank_mismatch", "ineligible",
                           "dup_candidate", "topk_size", "n_eligible", "unknown_frame")}
    for l in loops:
        tk = l["topk"]
        cs = [c for _, c, _ in tk]
        if any(cs[i] < cs[i + 1] for i in range(len(cs) - 1)):
            bad["cos_not_descending"].append(l["seq"])
        if any(cs[i] == cs[i + 1] and idx_of.get(tk[i][0], 0) > idx_of.get(tk[i + 1][0], 0) for i in range(len(cs) - 1)):
            bad["tie_order"].append(l["seq"])
        sc = [s for _, _, s in tk]
        am = int(np.argmax(sc))                   # first maximum = better cosine rank
        if tk[am][0] != l["best_frame"]:
            bad["best_not_argmax"].append(l["seq"])
        if am != l["best_cos_rank"]:
            bad["best_rank_mismatch"].append(l["seq"])
        if len({f for f, _, _ in tk}) != len(tk):
            bad["dup_candidate"].append(l["seq"])
        for f, _, _ in tk:
            if f not in seq_of:
                bad["unknown_frame"].append((l["seq"], f))
            elif seq_of[f] + gap > l["seq"]:
                bad["ineligible"].append((l["seq"], f))
        if len(tk) != min(k_top, l["n_eligible"]):
            bad["topk_size"].append(l["seq"])
        if l["n_eligible"] != sum(p["seq"] + gap <= l["seq"] for p in places):
            bad["n_eligible"].append(l["seq"])
    exact_ties = sum(1 for l in loops for i in range(len(l["topk"]) - 1) if l["topk"][i][1] == l["topk"][i + 1][1])
    score_ties = sum(1 for l in loops if sorted([s for _, _, s in l["topk"]])[-2:][0] == l["topk"][int(np.argmax([s for _, _, s in l["topk"]]))][2] and len(l["topk"]) > 1)
    out = dict(run=name, k_used=k_top, n_queries=len(loops), n_places=len(places), gap_used=gap, gaps_consistent_with_n_eligible=gaps,
               exact_cosine_ties=exact_ties, exact_score_ties_at_best=score_ties,
               violations={k: len(v) for k, v in bad.items()}, violation_rows={k: v[:10] for k, v in bad.items() if v})
    out["ok"] = all(len(v) == 0 for v in bad.values())
    return out, gap, k_top


# ------------------------------------------------------------------ inputs
def pre_A(paths):
    import torchvision.transforms as T
    from PIL import Image
    from pairvpr.export.patches import IMG_RES
    tf = T.Compose([T.Resize((IMG_RES, IMG_RES), interpolation=T.InterpolationMode.BILINEAR), T.ToTensor(),
                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    return np.stack([tf(Image.open(p).convert("RGB")).numpy() for p in paths])


def pre_B(paths):
    import cv2
    out = []
    for p in paths:
        im = cv2.imread(str(p), cv2.IMREAD_REDUCED_COLOR_4)
        assert im is not None and im.shape[:2] == (704, 704), (p, None if im is None else im.shape)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        out.append(preprocess(np.ascontiguousarray(im.transpose(2, 0, 1))))
    return np.stack(out)


def pre_B_area(paths):
    """Fork-harness input: full decode, cv2 INTER_AREA to 704 (not what the C++ does; extra probe of the decode gap)."""
    from .common import load_image_u8
    return np.stack([preprocess(load_image_u8(p)) for p in paths])


# ------------------------------------------------------------------ pipelines
class TorchA:
    """The repo's PairVPRNet fp32, MATH sdpa, forward(mode='global') / forward(mode='pairvpr') both orders."""
    name = "fp32-torch"

    def __init__(self):
        self.m = build_fp32()

    @torch.inference_mode()
    def encode(self, pre):
        d, f = [], []
        for x in pre:
            with sdpa_kernel(SDPBackend.MATH):
                dense, desc = self.m(torch.from_numpy(x)[None].to(DEVICE), None, mode="global")
            d.append(desc.float().cpu().numpy()[0])
            f.append(dense.float().cpu().numpy()[0])
        return np.stack(d), f

    @torch.inference_mode()
    def score(self, q, cands):
        qq = torch.from_numpy(q)[None].to(DEVICE).expand(len(cands), -1, -1)
        c = torch.from_numpy(np.stack(cands)).to(DEVICE)
        with sdpa_kernel(SDPBackend.MATH):
            a = self.m(qq, c, "pairvpr")
            b = self.m(c, qq, "pairvpr")
        return (a + b).squeeze(1).float().cpu().numpy()


class TrtB:
    """The C++ engines: encoder (dense fp16), pair scorer padded to 8 with the last candidate."""
    name = "trt-mixed+fast"

    def __init__(self):
        self.c = TrtCand(ROOT / "Model" / ENC_PLAN, ROOT / "Model" / PAIR_PLAN)

    def encode(self, pre):
        d, f = [], []
        for x in pre:
            o = self.c.enc({"image": x[None]})
            d.append(o["descriptor"][0])
            f.append(o["dense"][0].astype(np.float16))     # the C++ stores fp16 dense maps
        return np.stack(d), f

    def score(self, q, cands):
        k = len(cands)
        padded = list(cands) + [cands[-1]] * (PAIR_C - k)
        return self.c.pair.__call__({"query": q[None], "candidates": np.stack(padded)})["score"][:k]


def run_queries(pipe, desc, dense, places, loops_sel, gap, K):
    """The C++ query procedure on this pipeline's descriptors / dense maps. -> {query_seq: dict}."""
    seqs = np.array([p["seq"] for p in places])
    idx_of_seq = {p["seq"]: i for i, p in enumerate(places)}
    res = {}
    for l in loops_sel:
        qi = idx_of_seq[l["seq"]]
        n_el = int((seqs + gap <= l["seq"]).sum())          # places are in seq order: a prefix
        cos = (desc[:n_el].astype(np.float32) @ desc[qi].astype(np.float32)).astype(np.float32)
        order = np.lexsort((np.arange(n_el), -cos))[:min(K, n_el)]       # cosine desc, ties to lower index
        scores = pipe.score(dense[qi], [dense[j] for j in order])
        best = int(np.argmax(scores))                        # first maximum
        res[l["seq"]] = dict(n_eligible=n_el, frames=[places[j]["frame"] for j in order],
                             cos=[float(cos[j]) for j in order], scores=[float(s) for s in scores], best=best)
    return res


# ------------------------------------------------------------------ metrics
def stats(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return dict(n=0)
    return dict(n=int(len(x)), median=float(np.median(x)), p95=float(np.quantile(x, 0.95)), max=float(x.max()))


def margin(scores, best):
    s = sorted(scores, reverse=True)
    return float(s[0] - s[1]) if len(s) > 1 else float("nan")


def compare_to_cpp(label, res, loops_sel):
    from scipy.stats import kendalltau
    top1, jac, cos1, taus, dcos, dsc, diffs = [], [], [], [], [], [], []
    for l in loops_sel:
        r = res[l["seq"]]
        cf = [f for f, _, _ in l["topk"]]
        cbest = l["best_frame"]
        pbest = r["frames"][r["best"]]
        top1.append(cbest == pbest)
        cos1.append(cf[0] == r["frames"][0])
        jac.append(len(set(cf) & set(r["frames"])) / len(set(cf) | set(r["frames"])))
        cmap = {f: (c, s) for f, c, s in l["topk"]}
        pmap = {f: (c, s) for f, c, s in zip(r["frames"], r["cos"], r["scores"])}
        common = [f for f in cf if f in pmap]
        for f in common:
            dcos.append(abs(cmap[f][0] - pmap[f][0]))
            dsc.append(abs(cmap[f][1] - pmap[f][1]))
        if len(common) >= 2:
            t = kendalltau([cmap[f][0] for f in common], [pmap[f][0] for f in common])[0]
            if not np.isnan(t):
                taus.append(float(t))
        if cbest != pbest:
            csc = [s for _, _, s in l["topk"]]
            row = dict(query_seq=l["seq"], cpp_best=cbest, pipe_best=pbest,
                       cpp_margin=margin(csc, l["best_cos_rank"]), pipe_margin=margin(r["scores"], r["best"]),
                       cpp_best_score_in_cpp=cmap[cbest][1], pipe_best_score_in_pipe=pmap[pbest][1],
                       cpp_best_score_in_pipe=pmap[cbest][1] if cbest in pmap else None,
                       pipe_best_score_in_cpp=cmap[pbest][1] if pbest in cmap else None,
                       cpp_verified=l["verified"], cpp_accepted=l["accepted"])
            diffs.append(row)
    n = len(loops_sel)
    return dict(label=label, n_queries=n, top1_best_agree_pct=100.0 * np.mean(top1),
                global_top1_agree_pct=100.0 * np.mean(cos1), top5_jaccard_mean=float(np.mean(jac)),
                top5_jaccard_min=float(np.min(jac)), top5_identical_set_pct=100.0 * np.mean([j == 1.0 for j in jac]),
                kendall_cos_order=dict(n=len(taus), mean=float(np.mean(taus)) if taus else None,
                                       min=float(np.min(taus)) if taus else None,
                                       exact_order_pct=100.0 * float(np.mean([t == 1.0 for t in taus])) if taus else None),
                abs_dcosine=stats(dcos), abs_dscore=stats(dsc), n_best_differs=len(diffs), best_differs=diffs)


# ------------------------------------------------------------------ main
def select_queries(loops, max_q):
    if max_q <= 0 or len(loops) <= max_q:
        return loops, 1
    step = int(np.ceil(len(loops) / max_q))
    return loops[::step], step


def fmt_stats(s):
    return "n=0" if s["n"] == 0 else f"{s['median']:.2e} / {s['p95']:.2e} / {s['max']:.2e}"


def print_cmp(c):
    k = c["kendall_cos_order"]
    print(f"  {c['label']:<14} top1 best {c['top1_best_agree_pct']:6.2f}%  glob-top1 {c['global_top1_agree_pct']:6.2f}%  "
          f"top5 Jaccard mean {c['top5_jaccard_mean']:.4f} min {c['top5_jaccard_min']:.3f} (identical set {c['top5_identical_set_pct']:.1f}%)  "
          f"tau mean {k['mean'] if k['mean'] is None else round(k['mean'], 4)} min {k['min'] if k['min'] is None else round(k['min'], 3)} "
          f"(exact {k['exact_order_pct'] if k['exact_order_pct'] is None else round(k['exact_order_pct'], 1)}%, n={k['n']})")
    print(f"  {'':<14} |dcos| med/p95/max {fmt_stats(c['abs_dcosine'])}   |dscore| med/p95/max {fmt_stats(c['abs_dscore'])}   "
          f"best differs: {c['n_best_differs']}")
    for d in c["best_differs"]:
        f = lambda v: "-" if v is None else f"{v:.3f}"
        print(f"     q{d['query_seq']:>5}: cpp best {d['cpp_best']} (margin {f(d['cpp_margin'])}, score {f(d['cpp_best_score_in_cpp'])}) "
              f"vs pipe best {d['pipe_best']} (margin {f(d['pipe_margin'])}, score {f(d['pipe_best_score_in_pipe'])}); "
              f"cpp-best score in pipe {f(d['cpp_best_score_in_pipe'])}, pipe-best score in cpp {f(d['pipe_best_score_in_cpp'])}; "
              f"cpp verified={d['cpp_verified']} accepted={d['cpp_accepted']}")


def run_parity(name, cfg, places, loops, imgs, gap, K, max_q, ablation, extra_area=False):
    sel, step = select_queries(loops, max_q)
    print(f"\n== parity {name}: {len(sel)} of {len(loops)} queries (every {step}th), all {len(places)} places encoded, gap {gap}", flush=True)
    paths = [imgs[p["frame"]] for p in places]
    t0 = time.time()
    timing = {}
    pa = pre_A(paths); timing["decode_A_s"] = time.time() - t0; t0 = time.time()
    pb = pre_B(paths); timing["decode_B_s"] = time.time() - t0
    print(f"  decode A (PIL full-res) {timing['decode_A_s']:.1f}s, B (cv2 reduced 4) {timing['decode_B_s']:.1f}s; "
          f"input diff A vs B (322 normalised): mean|d| {np.abs(pa - pb).mean():.4f}, max {np.abs(pa - pb).max():.3f}", flush=True)
    jobs = [("A", "torch", pa), ("B", "trt", pb)]
    if ablation:
        jobs += [("A_on_B", "torch", pb), ("B_on_A", "trt", pa)]
    if extra_area:
        jobs.append(("B_area704", "trt", pre_B_area(paths)))
    pipes = {}
    results, cmps = {}, {}
    for label, kind, pre in jobs:
        t0 = time.time()
        if kind not in pipes:
            pipes[kind] = TorchA() if kind == "torch" else TrtB()
        pipe = pipes[kind]
        desc, dense = pipe.encode(pre)
        t1 = time.time()
        res = run_queries(pipe, desc, dense, places, sel, gap, K)
        t2 = time.time()
        timing[label] = dict(encode_s=t1 - t0, query_s=t2 - t1)
        results[label] = res
        cmps[label] = compare_to_cpp(label, res, sel)
        print(f"  [{label}] encode {t1 - t0:.1f}s, queries {t2 - t1:.1f}s", flush=True)
        del dense
    print("\n  A vs C++ / B vs C++" + (" / ablations" if ablation else ""))
    for c in cmps.values():
        print_cmp(c)
    # A vs B directly
    ab = {}
    if "A" in results and "B" in results:
        ab = dict(top1_best_agree_pct=100.0 * np.mean([results["A"][l["seq"]]["frames"][results["A"][l["seq"]]["best"]] ==
                                                       results["B"][l["seq"]]["frames"][results["B"][l["seq"]]["best"]] for l in sel]))
        print(f"  A vs B top-1 best agreement: {ab['top1_best_agree_pct']:.2f}%")
    return dict(n_queries=len(sel), query_step=step, gap=gap, timing=timing, comparisons=cmps, A_vs_B=ab,
                query_seqs=[l["seq"] for l in sel])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=["consistency", "parity", "all"], default="all")
    ap.add_argument("--runs", nargs="+", default=list(RUNS), choices=list(RUNS))
    ap.add_argument("--gap", type=int, default=None, help="exclusion gap G (default: the largest G the CSV allows)")
    ap.add_argument("--max-queries", type=int, default=150, help="subsample queries to about this many per run (0 = all)")
    ap.add_argument("--k", type=int, default=None, help="rerank K (default: the largest topk size in the CSV)")
    ap.add_argument("--extra-area", action="store_true", help="also B engines on cv2 INTER_AREA 704 inputs (fork harness path)")
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--json", type=Path, default=ROOT / "runs" / "parity" / "ranking_parity.json")
    args = ap.parse_args()

    t_start = time.time()
    out = {}
    for name in args.runs:
        cfg = RUNS[name]
        places, loops = load_run(cfg["stem"])
        rep = {}
        fm, imgs = check_frame_mapping(cfg["seq"], places, loops)
        rep["frame_mapping"] = fm
        print(f"\n## {name}: {len(places)} places, {len(loops)} queries\n frame mapping: {fm}")
        cons, gap, K = consistency(name, places, loops, args.gap, args.k)
        rep["consistency"] = cons
        print(" consistency:", json.dumps({k: v for k, v in cons.items() if k != "run"}, default=str))
        if args.mode in ("parity", "all"):
            rep["parity"] = run_parity(name, cfg, places, loops, imgs, gap, K, args.max_queries, not args.no_ablation, args.extra_area)
        out[name] = rep
    out["total_runtime_s"] = time.time() - t_start
    print(f"\ntotal runtime {out['total_runtime_s']:.0f}s")
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print("->", args.json)


if __name__ == "__main__":
    main()
