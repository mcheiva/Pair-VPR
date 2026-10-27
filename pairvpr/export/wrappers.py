"""ONNX export wrappers: an image encoder and a pair scorer (both orders fused into one 2C batch).

Dtype policy (fixed in PyTorch so TensorRT builds the network strongly typed):
  fast     fp16 weights and activations; the L2-normalised descriptor head (`globalizer`, normalize) and the
           score head (`classvprmodule`, the order sum) stay fp32 islands, a negligible cost.
  mixed    encoder in fp32 (TF32 off) with only the `dense` output cast to fp16 at the very end; descriptor fp32.
           The cure for TensorRT's fp16 GEMM drift in the encoder (docs/tensorrt-export.md); pairs with the
           `fast` pair scorer, which takes the fp16 dense maps. `mixed` has no pair scorer of its own.
  precise  fp32 everywhere.
`dense` / `query` / `candidates` use the mode's dense dtype (fp16 fast, fp32 precise); `descriptor` and `score`
are always fp32.
"""
import torch
from torch import nn

DENSE_DTYPE = {"fast": torch.float16, "mixed": torch.float16, "precise": torch.float32}   # dense / pair inputs
COMPUTE_DTYPE = {"fast": torch.float16, "mixed": torch.float32, "precise": torch.float32}
MODES = tuple(DENSE_DTYPE)


def convert(model: nn.Module, mode: str) -> nn.Module:
    """Cast the model for `mode` (in place); fp32 islands stay fp32."""
    if mode == "fast":
        model.half()
        model.globalizer.float()
        model.classvprmodule.float()
    return model


class Encoder(nn.Module):
    """image [1,3,322,322] fp32 (ImageNet-normalised) -> descriptor [1,512] fp32, dense [1,529,768]."""

    def __init__(self, model: nn.Module, mode: str):
        super().__init__()
        self.model = convert(model, mode)
        self.dtype = COMPUTE_DTYPE[mode]
        self.dense_dtype = DENSE_DTYPE[mode]

    def forward(self, image):
        f, c = self.model.encoder(image.to(self.dtype))          # f [B,529,768], c [B,768] (registers dropped)
        desc = self.model.globalizer(c.float())
        desc = nn.functional.normalize(desc, p=2, dim=-1)        # fp32
        return desc, f.to(self.dense_dtype)                      # mixed: the only fp32 -> fp16 cast


class PairScorer(nn.Module):
    """query [1,529,768], candidates [C,529,768] -> score [C] fp32 = logit(q, c_i) + logit(c_i, q).

    Rows [0, C) of the 2C batch are (x=query, y=cand_i), rows [C, 2C) are (x=cand_i, y=query). The query is
    broadcast with expand (a runtime broadcast, not a dense constant). Higher = more similar."""

    def __init__(self, model: nn.Module, mode: str):
        super().__init__()
        assert mode in ("fast", "precise"), "mixed has no pair scorer of its own: use the fast one"
        self.model = convert(model, mode)
        self.dtype = DENSE_DTYPE[mode]

    def forward(self, query, candidates):
        m = self.model
        C = candidates.shape[0]
        q = m.decoder_embed(query.to(self.dtype)).expand(C, -1, -1)
        c = m.decoder_embed(candidates.to(self.dtype))
        x = torch.cat((q, c), dim=0)                             # [2C, 529, 768]: stream x (gets the cls token)
        y = torch.cat((c, q), dim=0)
        x = torch.cat((m.decoder_clstoken.expand(2 * C, -1, -1).to(x.dtype), x), dim=1) + m.dec_pos_embed_cls
        y = y + m.dec_pos_embed
        for blk in m.dec_blocks:
            x, y = blk(x, y)
        x = m.dec_norm(x)
        logit = m.classvprmodule(x[:, 0].float()).squeeze(-1)    # [2C] fp32
        return logit[:C] + logit[C:]
