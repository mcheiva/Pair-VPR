"""Dump (layer name, tactic) pairs of a plan as JSON via the engine inspector, and report attention fusion.

    python -m scripts.parity.layers Model/PairVPR_ViTB_Encoder_fast_o5.plan [out.json]

Mirrors MAV-VO-MCH/MACVO_TRT_BuildCpp/verify_out/layers.py (needs a plan built with ProfilingVerbosity DETAILED,
which Build.py sets). Prints the layer count and, per layer, how many are Myelin / fused MHA, so a reviewer
sees whether attention was fused without reading the dump.
"""
import json
import sys

import tensorrt as trt


def main() -> int:
    rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    with open(sys.argv[1], "rb") as f:
        eng = rt.deserialize_cuda_engine(f.read())
    info = json.loads(eng.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
    layers = info["Layers"]
    rows = [[l["Name"], l.get("TacticName")] for l in layers]
    dumps = [json.dumps(l).lower() for l in layers]
    n_mha = sum("mha" in d for d in dumps)            # layers whose name / tactic / metadata mention a fused MHA
    n_myl = sum("myl" in d or "myelin" in d for d in dumps)   # Myelin-generated layers
    print(f"{len(layers)} layers; {n_myl} Myelin, {n_mha} with a fused MHA (counted per layer)")
    for r in rows[:6]:
        print("  ", r[0][:100], "|", (r[1] or "")[:80])
    if len(sys.argv) > 2:
        with open(sys.argv[2], "w", encoding="utf-8") as f:
            json.dump(rows, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
