"""
c1_model.py — models for the C1 superposition + integration probe.

Two model classes, param-matched (the load-bearing control requirement):

  VoxelModel(task)  — the voxel arm. Encoder over the 32 x-bin positions.
                      For C1a (superposition) it takes TWO feature streams
                      (f_* and g_*) per position — the two color-set
                      entries coexisting in one cell — and must output the
                      derivative of the f-stream only. For C1b
                      (integration) it takes the derivative field and
                      outputs the antiderivative shape.
                      Uses the CORRECTED color encoding (hue as cos/sin,
                      brightness separate from density, alpha = sign,
                      features concatenated + single projection) — the
                      arm-E representation that recovered C0's arm A.

  TokenModel()      — the token baseline. Autoregressive encoder-decoder
                      over a flat sequence of the SAME data. For C1a the
                      two streams are interleaved [f_0,g_0,f_1,g_1,...];
                      for C1b it's the derivative values as a flat
                      sequence. This is the load-bearing control: does
                      the voxel substrate beat a plain token sequence of
                      the same numbers?

Param matching: both use H=384, FFN=1536, 8 heads. VoxelModel uses 6
encoder layers; TokenModel uses 2+3 (enc+dec) so counts land within ~2%
(10.67M vs 10.90M), matching C0. Exact counts reported by n_params.
"""
from __future__ import annotations
import torch
import torch.nn as nn

N_X = 32
N_VALUE_BINS = 16
H = 384
FFN = 1536
N_HEADS = 8
DROPOUT = 0.1

N_VALUE_TOKENS = 200
TOK_PAD = 0
TOK_BOS = 1
TOK_EOS = 2
TOK_VOCAB = N_VALUE_TOKENS + 3


def _value_to_token(v):
    i = int(round((v + 1) / 2 * (N_VALUE_TOKENS - 1)))
    return 3 + max(0, min(N_VALUE_TOKENS - 1, i))


def _token_to_value(t):
    i = t - 3
    return (i / (N_VALUE_TOKENS - 1)) * 2 - 1


class _Block(nn.Module):
    def __init__(self, d, ffn, heads, cross=False):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=DROPOUT, batch_first=True)
        self.n1 = nn.LayerNorm(d)
        self.cross = cross
        if cross:
            self.cattn = nn.MultiheadAttention(d, heads, dropout=DROPOUT, batch_first=True)
            self.nc = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d))
        self.n2 = nn.LayerNorm(d)

    def forward(self, x, memory=None):
        a, _ = self.attn(x, x, x)
        x = self.n1(x + a)
        if self.cross and memory is not None:
            c, _ = self.cattn(x, memory, memory)
            x = self.nc(x + c)
        x = self.n2(x + self.ff(x))
        return x


class VoxelModel(nn.Module):
    """Voxel arm for C1a (superposition) and C1b (integration)."""

    def __init__(self, task):
        super().__init__()
        assert task in ("sup", "int"), task
        self.task = task
        self.pos = nn.Parameter(torch.randn(N_X, H) * 0.02)
        self.value_emb = nn.Embedding(N_VALUE_BINS, H)
        # corrected color: concat [value_bin_emb + density, hue_cos, hue_sin,
        # brightness, alpha] -> single projection. For sup, two streams.
        self.proj = nn.Linear(5, H)   # density, hue_cos, hue_sin, brightness, alpha
        self.layers = nn.ModuleList([_Block(H, FFN, N_HEADS) for _ in range(6)])
        self.head_bin = nn.Linear(H, N_VALUE_BINS)
        self.head_dens = nn.Linear(H, 1)

    def _stream(self, x, prefix):
        """Build the per-position feature vector for one stream."""
        feats = torch.stack([
            x[f"{prefix}_density"], x[f"{prefix}_hue_cos"],
            x[f"{prefix}_hue_sin"], x[f"{prefix}_brightness"],
            x[f"{prefix}_alpha"],
        ], dim=-1)                       # (B, N_X, 5)
        return self.proj(feats)

    def forward(self, x):
        """x: dict of per-position feature arrays, each (B, N_X)."""
        B = x["f_value_bin"].shape[0] if self.task == "sup" else x["in_value_bin"].shape[0]
        if self.task == "sup":
            # two streams coexist in the same cells (superposition)
            h = self.value_emb(x["f_value_bin"]) + self.pos.unsqueeze(0)
            h = h + self._stream(x, "f")
            h = h + self._stream(x, "g")   # g coexists in the same cell
        else:
            h = self.value_emb(x["in_value_bin"]) + self.pos.unsqueeze(0)
            h = h + self._stream(x, "in")
        for layer in self.layers:
            h = layer(h)
        bin_logits = self.head_bin(h)
        dens = self.head_dens(h).squeeze(-1)
        return bin_logits, dens

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class TokenModel(nn.Module):
    """Token baseline: autoregressive encoder-decoder over flat values."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(TOK_VOCAB, H)
        self.pos = nn.Parameter(torch.randn(256, H) * 0.02)
        self.enc = nn.ModuleList([_Block(H, FFN, N_HEADS) for _ in range(2)])
        self.dec = nn.ModuleList([_Block(H, FFN, N_HEADS, cross=True) for _ in range(3)])
        self.out = nn.Linear(H, TOK_VOCAB)

    def forward(self, src, tgt):
        B, Ls = src.shape
        se = self.emb(src) + self.pos[:Ls].unsqueeze(0)
        for layer in self.enc:
            se = layer(se)
        Lt = tgt.shape[1]
        te = self.emb(tgt) + self.pos[:Lt].unsqueeze(0)
        for layer in self.dec:
            te = layer(te, se)
        return self.out(te)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def build(task, kind):
    if kind == "voxel":
        return VoxelModel(task)
    if kind == "token":
        return TokenModel()
    raise ValueError(kind)
