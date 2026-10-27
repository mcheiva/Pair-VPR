# Pair-VPR ViT-B: PyTorch -> ONNX -> TensorRT

Two static engines, mirroring the MAC-VO recipe (MAV-VO-MCH `Export.py` / `Build.py` / `Scripts/Parity`):

* **Encoder**: `image [1,3,322,322]` fp32 (already ImageNet-normalised) -> `descriptor [1,512]` fp32 (L2-normalised), `dense [1,529,768]` (fp16 in fast and mixed, fp32 in precise).
* **Pair<C>** (C = 8): `query [1,529,768]`, `candidates [8,529,768]` -> `score [8]` fp32 = logit(q,c) + logit(c,q), higher = more similar. Both orders run as one 2C batch; the query is a runtime `expand`.

Pair-VPR's "speed" mode is the ViT-B network of this checkpoint; "performance" mode is ViT-G weights (not available here) plus 500 rerank candidates. The 322 input and the decoder config are identical, so only the speed networks are exported; the rerank depth is the consumer's choice.

## Modes

| mode | encoder | pair scorer |
|---|---|---|
| `fast` | fp16 weights/activations; fp32 islands for `globalizer` + L2 normalise | fp16; fp32 `classvprmodule` + order sum |
| `mixed` | **fp32 (TF32 off)**, only `dense` cast to fp16 at the very end; descriptor fp32 | the `fast` one (`Pair8_fast`) |
| `precise` | fp32 (TF32 off) | fp32 |

`--mode mixed` exports and builds **only** `Encoder_mixed`. The pair scorer of the pipeline is `Pair<C>_fast`, exported and built by `--mode fast --part pair` (mixed never touches it, so a screened plan's ONNX stays untouched; `PairScorer(model, "mixed")` asserts).

**TF32 is off in every build** (`Build.py` default; `--tf32` opts in). A strongly typed build honours TF32 for any fp32 MatMul, including the small fp32 heads (`globalizer`, `classvprmodule`) of the fast graphs, and the torch-fast yardstick runs with TF32 off. The first-screening fast plans were built with TF32 on (`allow_tf32: true` in their sidecars): the fast **encoder** plan still is (it is not deployed; the fast + fast rows below use it), and the deployed `Pair8_fast_o5.plan` was rebuilt with TF32 off (sidecar `allow_tf32: false`; its ONNX hash `01d2c4b84ac9fd4f` matches the sidecar).

## Recipe

```
set CUDA_DEVICE_ORDER=PCI_BUS_ID & set CUDA_VISIBLE_DEVICES=1
python Export.py --mode mixed           # Encoder_mixed only (fast / precise export encoder + pair;  --pair-batch 8)
python Export.py --mode fast --part pair
python Build.py  --mode mixed           # idle GPU only, never two builds at once; mixed = encoder only
python Build.py  --mode fast --part pair
python -m scripts.parity.make_reference # fp32 reference, runs/parity/ref.npz
python -m scripts.parity.compare --torch fast --json runs/parity/torch_fast.json
python -m scripts.parity.compare --trt Model/PairVPR_ViTB_Encoder_mixed_o5.plan --vs runs/parity/torch_fast.json
python -m scripts.parity.test_exact
python -m scripts.eval_eiva_vpr --fp32 --trt Model/PairVPR_ViTB_Encoder_mixed_o5.plan
python -m scripts.parity.export_cpp_inputs runs/parity/cpp_in     # inputs for the C++ parity tool (below)
python -m scripts.parity.debug_encoder --variant base --trt     # per-block divergence (below; study tooling)
```

Export: static shapes, opset 18 (opset 20 compiled silently wrong on TRT 10.x for MAC-VO), a warm-up forward under `inference_mode`, `do_constant_folding=False`, precision fixed in PyTorch (strongly typed build), preprocessing only in the consumer (`scripts/parity/common.preprocess`). Build: strongly typed, `parse_from_file`, optimization level 5, avg timing 8, tiling 0, weight streaming on, DETAILED profiling verbosity, timing cache loaded with `ignore_mismatch=False` and saved atomically, `.plan.json` sidecar in the MAC-VO schema, write order sidecar-delete, plan, sidecar, cache. Plans are named `_o<opt level>` (`_o5` here).

Export patches (`pairvpr/export/patches.py`): `XFORMERS_DISABLED=1` before imports; the cross-attention fallback called `super().forward(q, k, v, qpos, kpos)` on a 3-argument method (TypeError); DINOv2 pinned to `facebookresearch/dinov2@7764ea0f912e53c92e82eb78a2a1631e92725fc8` with `pretrained=False` (the checkpoint holds all 480 tensors; `strict=True` loads); DINOv2's bicubic antialiased 37->23 pos-embed resize frozen to a constant `[1,530,768]` (bit-identical to the live interpolation); PyYAML attribute dict instead of omegaconf. ONNX export is deterministic (re-export gives the same file hash). No onnxsim.

Environment of the export and all numbers here: Python 3.11.5, torch 2.6.0+cu124, torchvision 0.21.0+cu124, onnx 1.18.0, onnxruntime 1.26.0 (CPU provider only), TensorRT 10.13.2.6, opencv 4.11.0, numpy 2.4.0, PyYAML 6.0; RTX 4090. Data: `DATA_ROOT = D:/Datasets/EIVA` (`<seq>/processed/left/*.jpg`, `<seq>/pose_gt.txt`).

C++ parity tool inputs: `python -m scripts.parity.export_cpp_inputs <dir>` writes `u8_<i>.npy` (uint8 [704,704,3] HWC), `pre_<i>.npy` (float32 [3,322,322], the reference preprocessing) and `cand.npy` (int64 [N,C]); the tool writes `descriptor_<i>.npy`, `dense_<i>.npy`, `scores_<q>.npy` back and `compare --cpp <dir>` scores them (layout in `scripts/parity/compare.py`).

The fp32 reference runs with the MATH SDPA backend (DINOv2 attention calls SDPA) so it does not depend on the fused kernel PyTorch picks; the choice moved the reference by at most 1.3e-3 on dense values, 4e-7 on descriptors and 3.9e-5 on scores. The harness feeds cv2 INTER_AREA 704x704 frames while the C++ loader uses nvJPEG's 1/4 DCT-scaled decode; both networks see the loader's images at runtime, so engine parity is unaffected (domain note only).

## Deliverables (`Model/`, git-ignored except `*.plan.json`)

`PairVPR_ViTB_Encoder_{fast,mixed,precise}.onnx`, `PairVPR_ViTB_Pair8_{fast,precise}.onnx`, each with `_o5.plan`, `.plan.json`, `.timing.cache`. TRT 10.13.2.6, RTX 4090 (sm 8.9). Recommended pipeline: `Encoder_mixed_o5.plan` + `Pair8_fast_o5.plan`.

## Why fast drifts (root cause)

The fast TRT encoder has 0.75 % of dense tokens with cosine < 0.98 to fp32 (190 of 25392; worst 0.20), versus 4 tokens for torch fp16 (worst 0.905), while the descriptor stays at cosine >= 0.9999.

* **Which tokens.** Not a fixed set: 125 distinct token indices over 42 of 48 frames, none more than 6 times. Not high-norm tokens (fp32 norm 25.7 for the outliers vs 24.2 on average; the residual stream peaks at 73, far from fp16's 65504, so no overflow). 21 of the 33 tokens torch fp16 gets below 0.995 are among TRT's outliers: both hit the same fragile tokens, TRT much harder.
* **Where it starts** (`debug_encoder.py`, residual stream after each block, per-token cosine vs torch fp32, p01 / min / tokens < 0.98 of 25632). TRT diverges from block 1 and compounds; torch fp16 stays about 5x closer until block 9. The probe plan reproduces the shipped numbers exactly (dense 0.98668 / 0.1999 / 190), so the extra outputs did not change the defect.

| block | 1 | 4 | 6 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|
| torch fp16 | 1.00000 / 1.0000 / 0 | 0.99998 / 0.9997 / 0 | 0.99997 / 0.9996 / 0 | 0.99997 / 0.9997 / 0 | 0.99993 / 0.9990 / 0 | 0.99984 / 0.9921 / 0 | 0.99973 / 0.9728 / 2 | 0.99963 / 0.9207 / 4 |
| TRT fp16 | 0.99996 / 0.9999 / 0 | 0.99918 / 0.9949 / 0 | 0.99780 / 0.9794 / 1 | 0.99808 / 0.9747 / 4 | 0.99712 / 0.9441 / 17 | 0.99401 / 0.6418 / 86 | 0.99050 / 0.3333 / 152 | 0.98862 / 0.2643 / 181 |

* **Hypotheses.** Attention logits / softmax: no. `sm32` (fp16 q.kT, fp32 scale + softmax) and `attn32` (q.kT too in fp32) leave the outliers unchanged (dense min 0.215 / 0.224, 183 / 191 tokens < 0.98). LayerNorm in fp32: no (an earlier `fast_ln32` variant gave dense min 0.211). Block-1 intermediates (`--sub-block 0`): the error is already 5x torch's at the patch-embed output and after every Linear (e.g. fc2: TRT rel. max error 1.4e-2 vs 3.0e-3), with no single op to blame.
* **Cause (strongly supported, accumulator type inferred from tactic names).** All 49 encoder GEMMs / convs are `sm80_xmma_*_f16f16_f16f16_f16` tactics. In isolation, one block-3 `fc1` Linear on real activations (fp64 reference): torch fp16 mean abs error 2.9e-4 / max 1.1e-2 (pure output rounding), TRT fp16 7.6e-4 / 5.3e-2, fp32 2e-7. Restricting tactic sources (cuBLAS only, cuBLASLt only, none) picks the same tactic family with the same error, and a strongly typed network offers no accumulator control. The extra GEMM error compounds through 12 blocks and flips the fragile tokens. The pair scorer shares the cause: the pair stage alone on fp32 dense maps is 0.0136 max for TRT fast vs 0.0047 for torch fp16.
* **TF32 is not enough.** An fp32 encoder with TF32 on (2.38 ms): dense min 0.847, worse than torch fp16 (0.905); rejected. Full fp32 is needed for the fragile tokens.

No fp16 variant that passes the yardstick at < 1.5 ms was found (the softmax fix does nothing), so no fix was applied to the pair scorer either.

## Latency (trtexec `--useCudaGraph --noDataTransfers --iterations=200`, idle GPU 1, mean GPU ms)

| engine | fast | mixed | precise |
|---|---|---|---|
| Encoder | 0.849 (TF32-on plan) | 3.565 | 3.485 |
| Pair8 (16 pairs) | 12.21 (TF32 off; 12.52 with TF32 on) | (= fast) | 74.24 |

`precise` was rebuilt on an idle GPU 1 after its first build overlapped a VO run; the tactics changed in 37 of 127 encoder and 26 of 237 pair layers, and it re-measured 3.24 -> 3.49 ms and 73.8 -> 74.2 ms. Attention is fused in the fast engines (`scripts/parity/layers.py`, counted per layer): 12 layers with a Myelin `_gemm_mha_v2` in the encoder (91 layers, all Myelin) and 24 in the pair scorer (168 layers, 166 Myelin; the TF32-off rebuild has 3 layers more than the TF32-on plan, which showed 165); the per-layer profile is flat (largest layer 1.6 % / 1.9 %), so no graph rewrite was tried.

## Parity vs the fp32 reference (48 frames, 3 dives; `compare.py`)

| candidate | desc cos min | dense cos p01 / min | score max / p99 (e2e) | iso max / p99 | top1 / top5 | tau mean / min |
|---|---|---|---|---|---|---|
| torch fast | 0.999993 | 0.99958 / 0.9046 | 0.0456 / 0.0315 | 0.0047 / 0.0039 | 1 / 1 | 0.9985 / 0.929 |
| ONNX fast (ORT CPU) | 0.999995 | 0.99983 / 0.9873 | 0.0542 / 0.0401 | 0.0048 / 0.0041 | 1 / 1 | 0.9985 / 0.929 |
| TRT fast | 0.999899 | 0.98668 / 0.1999 | 0.1935 / 0.1112 | 0.0136 / 0.0084 | 1 / 1 | 0.9970 / 0.929 |
| TRT mixed enc + fast pair (TF32-off pair, deployed) | 0.9999999 | 1.0000 / 0.9999999 | 0.0151 / 0.0086 | 0.0162 / 0.0099 | 1 / 1 | 1 / 1 |
| TRT mixed enc + fast pair (first, TF32-on pair) | 0.9999999 | 1.0000 / 0.9999999 | 0.0118 / 0.0087 | 0.0136 / 0.0084 | 1 / 1 | 1 / 1 |
| TRT mixed enc (TF32 on) + fast pair (TF32 on) | 0.999998 | 0.99985 / 0.8466 | 0.0298 / 0.0222 | 0.0136 / 0.0084 | 1 / 1 | 0.9985 / 0.929 |
| TRT precise | 1.000000 | 1.0000 / 1.0000 | 3.7e-5 / 3.1e-5 | 5e-6 / 4e-6 | 1 / 1 | 1 / 1 |

Rows other than the mixed + TF32-off-pair one were measured against the first reference (before the MATH-SDPA change; differences <= 4e-5 on scores); torch fast on the final reference: dense 0.999584 / 0.9046, score 0.0456 / 0.0315, iso 0.0054 / 0.0043, so the pair-stage gap to torch fast is 0.0162 vs 0.0054. Turning TF32 off in the pair plan did **not** shrink the iso gap (0.0136 -> 0.0162, a different tactic draw): TF32 is not what separates the TRT pair stage from torch; the fp16 GEMM drift above is. "iso" = the pair stage alone on fp32 dense maps; it is the pair engine's own error and is identical for every row that uses `Pair8_fast`. ONNX ran on the CPU provider (onnxruntime has no CUDA provider here), so it checks graph correctness only.

## Recall (`scripts/eval_eiva_vpr.py`, time-gap protocol)

Database = every 4th frame (Welland: every 8th), queries = the same frames, candidates exclude frames within 60 s of the query (timestamps parsed as `gt_to_tum.py:parse_ts`), true match = GT centres within 1.0 m and optical axes within 30 degrees, queries without a true match skipped. Hits of N revisit queries: global R@1/5/10, then reranked R@1/5/10 over the top 8.

| dive (revisit queries) | fp32 | torch fast | TRT mixed + fast pair | TRT precise |
|---|---|---|---|---|
| plane_nose (54) | 28/43/47, 40/47/47 | 28/43/48, 40/47/48 | 28/43/47, 40/47/47 | 28/43/47, 40/47/47 |
| plane_tail1 (9, too few to conclude) | 4/7/8, 7/8/8 | 4/7/8, 7/8/8 | 4/7/8, 7/8/8 | 4/7/8, 7/8/8 |
| Welland (516, stride 8) | 260/326/358, 325/349/358 | 259/326/358, 325/349/358 | 260/326/358, 325/349/358 | 260/326/358, 325/349/358 |

Mixed + fast and precise reproduce the fp32 counts exactly on every dive. Reranking lifts R@1 from 28 to 40 of 54 (nose), 4 to 7 of 9 (tail1), 260 to 325 of 516 (Welland). Welland still has 516 of 590 queries with a revisit at 60 s (the dive passes the same places repeatedly); plane_tail1 has only 9, so its counts say little.

Parity check only (the first protocol, a fixed 16-index gap, about 8 s on Welland, so 588 of 590 Welland queries matched frames seconds apart; it measures re-identification, not revisits; `--index-gap 16`): fp32 and mixed + fast agreed to within 1 query on every dive there too (reranked R@1/5/10: nose 73/85/87, tail1 44/51/52, Welland 546/568/574).

## Recommendation

| mode combination | encoder ms | pair ms | dense cos min / p01 | score max / p99 | R@1/5 (rerank) per dive | yardstick |
|---|---|---|---|---|---|---|
| fast + fast (TF32-on plans) | 0.85 | 12.5 | 0.200 / 0.987 | 0.194 / 0.111 | not re-run under the time protocol | FAIL (all metrics but top-1, tau min) |
| **mixed + fast** | 3.57 | 12.2 | 0.9999999 / 1.0000 | 0.0151 / 0.0086 | 40/47, 7/8, 325/349 | PASS end to end; the pair-stage iso metric is 0.0162 vs torch fast 0.0054 |
| precise + precise | 3.49 | 74.2 | 1.0 / 1.0 | 2.4e-5 / 1.8e-5 | 40/47, 7/8, 325/349 | PASS |

Pass criterion for mixed + fast: every end-to-end metric (descriptor, dense, e2e score, retrieval, tau) no further from fp32 than torch fast; the pair-stage "iso" metric is reported separately and is worse (0.0162 vs 0.0054, scores are 5-25), which is the pair engine's own fp16 GEMM drift, not the encoder's.

Recommend **mixed encoder + fast pair**: 15.8 ms per query-with-8-candidates against 77.7 ms for precise, recall identical to fp32, dense maps at fp16-rounding level. `precise` stays as the reference plan. Plain `fast` is 4x cheaper in the encoder but has a numerical defect on fragile tokens.

## Rejected alternatives

opset 20 (silent miscompile risk); onnxsim (removed: no expected gain on this graph, it can change TensorRT's tactic choice, and a plain onnxsim call drops MAC-VO's safeguards); xFormers attention and live pos-embed interpolation (do not export); two separate scorer calls (one 2C batch halves launches); dense constants for the query broadcast (runtime `expand` instead); fp32 LayerNorm inside the fast graph (`fast_ln32`: no effect on the drift, +25 % encoder time, removed); TF32 in the fast pair plan (not the cause of the iso gap); fp32 attention softmax / logits (no effect); fp32 encoder with TF32 (2.38 ms, still worse than torch fp16 on the fragile tokens).
