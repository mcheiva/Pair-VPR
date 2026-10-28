Pair-VPR ViT-B TensorRT engines for MACVO_TRT loop closure (static, 2026-10-27) (branch feature/tensorrt-export,
docs/tensorrt-export.md; consumer: MACVO_TRT docs/loop-closure.md, section 4)

Deployed pair (what MACVO_TRT --loop-closure detect|on loads)
   Encoder:     PairVPR_ViTB_Encoder_mixed*   fp32 encoder (TF32 off), only the dense output cast to fp16 at the end.
                image [1,3,322,322] fp32 (ImageNet-normalised) -> descriptor [1,512] fp32 (L2-normalised),
                dense [1,529,768] fp16.
   Pair scorer: PairVPR_ViTB_Pair8_fast*      fp16 with fp32 heads (classvprmodule, order sum), TF32 off.
                query [1,529,768], candidates [8,529,768] -> score [8] fp32 = logit(q,c) + logit(c,q).
   Source weights: ../trained_models/pairvpr-vitB.pth (the Pair-VPR authors' ViT-B "speed" model; download link in
   ../trained_models/models_readme.md). Exported with Export.py on this branch (patches in pairvpr/export/patches.py).
   Removed from this folder on 2026-10-28 (not used by MACVO_TRT): Encoder_fast (fp16 GEMM drift: 0.75% of the dense
   tokens below cosine 0.98, worst 0.20; see the doc), Encoder_precise and Pair8_precise (the fp32 reference plans;
   the precise pair takes 74 ms). Regenerate them from the .pth if needed: python Export.py --mode fast|precise, then
   python Build.py --mode fast|precise (parity rows in the doc that name them were measured on those plans).

What to copy to another machine
   Same GPU model (RTX 4090, sm 8.9) and TensorRT 10.13.2: the two _o5.plan files into the MACVO_TRT working directory
     (the repo root, next to the MAC-VO plan), or point --vpr-encoder / --vpr-matcher at them. Nothing else.
   Any other GPU or TensorRT version: the two .onnx files and build them there (below). The ONNX files are
     self-contained (weights embedded, no .onnx.data), so the .pth, PyTorch and this repo's Python code are not
     needed to build or run. Take the .timing.cache files too only if that machine has the same GPU model and
     TensorRT version.
   The .pth is only needed to re-export an ONNX (another precision mode, batch or opset) or to regenerate the fp32
     parity reference (runs/parity/ref.npz).

Files per engine
   .onnx                Export.py output: static shapes, opset 18 (opset 20 compiled silently wrong on TRT 10.x for
                        MAC-VO), precision fixed in the graph, weights embedded. Re-export is deterministic (same hash).
                        No preprocessing inside: the consumer resizes to 322x322 and normalises.
   _o5.plan + .plan.json   TensorRT engine from that ONNX, Build.py defaults: strongly typed, optimization level 5,
                        avg timing 8, tiling 0, weight streaming on, TF32 off, detailed profiling verbosity; TRT 10.13.2,
                        RTX 4090. The sidecar records the ONNX path/size/hash, the settings, the timing-cache status and
                        the GPU. An engine loads only on the TensorRT version that built it.
   _o5.timing.cache     the per-layer tactic timings TensorRT measured when it built the plan (RTX 4090, TRT 10.13.2).
                        Only valid on the same GPU model and TensorRT version: Build.py loads it with
                        ignore_mismatch=False, so another GPU or TensorRT refuses it.

Building an engine (idle GPU, never two builds at once)
   Build.py (this repo, Python tensorrt 10.13.2 wheel), from the repo root:
     python Build.py --mode mixed                  # Encoder_mixed only
     python Build.py --mode fast --part pair       # Pair8_fast
     add --onnx <in> --out <plan> --timing-cache <cache> --error-on-timing-cache-miss to name the files explicitly
     (an explicit --timing-cache needs a single --part). Writes the .plan.json sidecar.
   trtexec (no Python), run from this folder (trtexec splits some path arguments on ':', so use relative paths):
     trtexec --onnx=PairVPR_ViTB_Encoder_mixed.onnx --saveEngine=PairVPR_ViTB_Encoder_mixed_o5.plan ^
       --stronglyTyped --allowWeightStreaming --builderOptimizationLevel=5 --avgTiming=8 --noTF32 ^
       --profilingVerbosity=detailed
     trtexec --onnx=PairVPR_ViTB_Pair8_fast.onnx --saveEngine=PairVPR_ViTB_Pair8_fast_o5.plan ^
       --stronglyTyped --allowWeightStreaming --builderOptimizationLevel=5 --avgTiming=8 --noTF32 ^
       --profilingVerbosity=detailed
     Same GPU and TensorRT: add --timingCacheFile=<name>_o5.timing.cache --errorOnTimingCacheMiss.
     trtexec writes no .plan.json sidecar; MACVO_TRT does not need one to load the plan.
   - --stronglyTyped is required: it keeps the fp16 / fp32 types baked into the graph.
   - --noTF32 is required for the mixed encoder: an fp32 encoder with TF32 on has dense min cosine 0.847 (the defect
     the mixed mode exists to avoid). The deployed pair plan is also built with TF32 off.
   - Needs TensorRT 10 (MACVO_TRT needs >= 10.7). Built and measured only on an RTX 4090.
   Re-export the ONNX (needs the .pth and this repo's Python environment; versions in docs/tensorrt-export.md:
   Python 3.11.5, torch 2.6.0+cu124, onnx 1.18.0, TensorRT 10.13.2.6):
     python Export.py --mode mixed                 # Encoder_mixed only
     python Export.py --mode fast --part pair      # Pair8_fast
   DINOv2 is pinned to facebookresearch/dinov2@7764ea0f912e53c92e82eb78a2a1631e92725fc8 through torch.hub (the code
   only, pretrained=False: the .pth holds all weights), so the first export needs internet or a filled hub cache.

Timing cache
   Same GPU model and TensorRT version: reproduce the shipped plan with --timing-cache <name>_o5.timing.cache
   --error-on-timing-cache-miss (trtexec: --timingCacheFile ... --errorOnTimingCacheMiss). The build fails if anything
   has to be timed anew. Verified 2026-10-28 with Build.py on an idle RTX 4090: both rebuilds hit the cache fully and
   reproduce the deployed tactics, encoder 127 of 127 layers and pair scorer 168 of 168 (runs/rebuild_check/).
   Plan bytes are never identical between two builds, even from the same cache. Compare tactic lists instead:
     trtexec --loadEngine=<plan> --dumpLayerInfo --profilingVerbosity=detailed --exportLayerInfo=<json>
   then the sorted (Name, TacticName) pairs.
   Different GPU model or TensorRT version: build without these caches (Build.py then saves a new one next to the
   plan). That build picks its own tactics, and some tactic choices change numerics, so screen the new plan.

Screening a rebuilt plan
   - This repo's parity tools (needs its Python environment, not the .pth, if runs/parity/ref.npz and
     runs/parity/torch_fast.json are copied along):
       python -m scripts.parity.compare --trt Model/PairVPR_ViTB_Encoder_mixed_o5.plan --vs runs/parity/torch_fast.json
     Pass = every end-to-end metric no further from fp32 than torch fast.
   - The C++ path: MACVO_TRT pairvpr_check <in_dir> <out_dir> (inputs from python -m scripts.parity.export_cpp_inputs,
     runs/parity/cpp_in here), scored with python -m scripts.parity.compare --cpp <out_dir>.
   - End to end without Python: run MACVO_TRT --loop-closure on on a reference dive and compare its loops and ATE with
     the reference machine's run (MACVO_TRT docs/loop-closure.md, section 8). The outputs will not be byte-identical
     when the tactics differ.

Performance (RTX 4090, TRT 10.13.2, trtexec --useCudaGraph --noDataTransfers, mean GPU ms)
   Encoder_mixed 3.57 ms, Pair8_fast 12.2 ms (16 pairs: both orders of 8 candidates). In MACVO_TRT per query: describe
   4.0 ms (preprocessing + encoder), rerank 12.4 ms.

Accuracy (parity against the fp32 reference, 48 frames from 3 dives; recall with the time-gap protocol)
   Mixed encoder + fast pair: descriptor cosine min 0.9999999, dense cosine min 0.9999999, end-to-end score error
   max 0.0151 (scores are 5-25), top-1 / top-5 identical to fp32. Reranked R@1/5/10 hits equal to fp32 on every dive
   (plane_nose 40/47/47 of 54 revisit queries, Welland 325/349/358 of 516).
   C++ (MACVO_TRT) against this repo's Python: best place identical on 95.7% (plane_nose) and 97.3% (Welland) of the
   queries; the gap is JPEG decode and resize, not the engines (scripts/parity/ranking_parity.py).

SHA-256
18e7b95ba57d4d578fbf0a06a1846fc2dfb976c74da56bfd7d6a68afbcea8423  ../trained_models/pairvpr-vitB.pth
5506d81d53cbd882e772c6898928e181e587074e413956ed3a031fb879c5b6e7  PairVPR_ViTB_Encoder_mixed.onnx
560c7a65302c656ce2b8dc3362e463ab2b6a0c6d40fa45b5d010142165f41d1c  PairVPR_ViTB_Encoder_mixed_o5.plan
e94f42ce1182d54db5e947bf7d29c83a2f75accefb883a5c48ec409010ed0f3e  PairVPR_ViTB_Encoder_mixed_o5.plan.json
eb07ff4fcf1315979102fbe0b77fa4a302aa7a6122acecffd9f435a1f07d7121  PairVPR_ViTB_Encoder_mixed_o5.timing.cache
0afe973f29bc863e6c24fc8108748baf767bcb223db12961ff8d0cb8d5d86797  PairVPR_ViTB_Pair8_fast.onnx
daadd6f8107e6b7250184b63936eaedc184ece9b6f89fe962d15829b72a37c23  PairVPR_ViTB_Pair8_fast_o5.plan
5224cc790c8b1ef8620d461a4c51ed1d29865bb04005f6483e341a20c7a1b469  PairVPR_ViTB_Pair8_fast_o5.plan.json
b653340b0137c903dd14ab8f325d5b69b8a032f8eb6d1496136903623d1f62c4  PairVPR_ViTB_Pair8_fast_o5.timing.cache
The two plans are byte-identical to the copies in the MACVO_TRT repo root.
