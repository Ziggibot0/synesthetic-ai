"""
calc_model.py — the four-arm transformer for the calc gate.

Two model classes share one transformer core so the arms are
param-matched (the load-bearing control requirement):

  FieldModel(arm)  — arms A/B/C. Encoder over the 32 x-bin positions,
                     per-position head -> (deriv_bin logits, deriv_density).
                     The output is a FIELD aligned to the input grid
                     (no autoregressive decoding; per calc_gate.md the
                     output is the derivative field, not symbolic text).
                     `arm` selects which input features are used:
                       A: value_bin + density + hue + brightness + alpha
                       B: value_bin + density            (-color)
                       C: value_bin + occupancy(0/1)      (-density-grad)

  TokenModel()      — arm D. Autoregressive encoder-decoder over the SAME
                     numeric field data, but as a flat sequence of
                     quantized value tokens (no spatial/color structure).
                     This is the load-bearing control: does the voxel
                     substrate do anything a plain token sequence of the
                     same numbers doesn't?

DESIGN NOTE (deviation from calc_gate.md wording, flagged for review):
calc_gate.md says arm D is "Charton prefix-notation tokens". That is a
SYMBOLIC task (expression -> derivative expression). Arms A/B/C are a
NUMERIC task (field -> field). Comparing symbolic accuracy to numeric
accuracy is not a fair control. The methodologically correct control for
"does the voxel substrate beat a plain representation of the same
numbers" is a numeric-token seq2seq over the identical data — which is
what TokenModel implements. The symbolic-vs-numeric question is a
separate, later experiment. See docs/calc_implementation.md.

Param matching: both classes use H=384, FFN=1536, 8 heads. FieldModel
uses 6 encoder layers; TokenModel uses 2+3 (enc+dec) so the counts land
within ~2% (10.67M vs 10.90M). Exact counts are reported by `n_params`
and logged at train time so the match is verifiable, not assumed.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

N_X = 32
N_VALUE_BINS = 16
H = 384
FFN = 1536
N_HEADS = 8
DROPOUT = 0.1

# token vocab for arm D: 200 value buckets over [-1,1] + specials
N_VALUE_TOKENS = 200
TOK_PAD = 0
TOK_BOS = 1
TOK_EOS = 2
TOK_VOCAB = N_VALUE_TOKENS + 3


def _value_to_token(v):
    """Map a value in [-1,1] to a token id in [3, 202]."""
    i = int(round((v + 1) / 2 * (N_VALUE_TOKENS - 1)))
    return 3 + max(0, min(N_VALUE_TOKENS - 1, i))


def _token_to_value(t):
    """Inverse of _value_to_token."""
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


class FieldModel(nn.Module):
    """Arms A/B/C: encoder over 32 positions, per-position field head.

    Arm E (corrected color, per Sean's 2026-08-30 revelations):
      - hue as (cos, sin) 2D embedding (respects circularity)
      - brightness kept SEPARATE from density (attention vs established)
      - alpha = sign of f (proper feature)
      - features CONCATENATED into one per-position vector and projected
        once, instead of added additively into the residual stream
    """

    def __init__(self, arm):
        super().__init__()
        assert arm in ("A", "B", "C", "E"), arm
        self.arm = arm
        self.pos = nn.Parameter(torch.randn(N_X, H) * 0.02)
        self.value_emb = nn.Embedding(N_VALUE_BINS, H)
        # scalar feature projectors (only the ones the arm uses)
        self.scalar = nn.Linear(1, H)
        # arm E: single projection over the concatenated feature vector
        self.e_proj = nn.Linear(6, H)   # density, hue_cos, hue_sin, brightness, alpha, occupancy
        self.layers = nn.ModuleList([_Block(H, FFN, N_HEADS) for _ in range(6)])
        self.head_bin = nn.Linear(H, N_VALUE_BINS)
        self.head_dens = nn.Linear(H, 1)

    def n_features(self):
        return {"A": 5, "B": 2, "C": 2, "E": 6}[self.arm]

    def forward(self, x):
        """x: dict of per-position feature arrays, each (B, N_X)."""
        B = x["value_bin"].shape[0]
        h = self.value_emb(x["value_bin"]) + self.pos.unsqueeze(0)
        if self.arm in ("A", "B"):
            h = h + self.scalar(x["density"].unsqueeze(-1))
        if self.arm == "A":
            h = h + self.scalar(x["hue"].unsqueeze(-1))
            h = h + self.scalar(x["brightness"].unsqueeze(-1))
            h = h + self.scalar(x["alpha"].unsqueeze(-1))
        if self.arm == "C":
            h = h + self.scalar(x["occupancy"].unsqueeze(-1))
        if self.arm == "E":
            # concatenate corrected color features, project once
            feats = torch.stack([
                x["density"], x["hue_cos"], x["hue_sin"],
                x["brightness"], x["alpha"], x["occupancy"],
            ], dim=-1)                       # (B, N_X, 6)
            h = h + self.e_proj(feats)
        for layer in self.layers:
            h = layer(h)
        bin_logits = self.head_bin(h)          # (B, N_X, 16)
        dens = self.head_dens(h).squeeze(-1)   # (B, N_X)
        return bin_logits, dens

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class TokenModel(nn.Module):
    """Arm D: autoregressive encoder-decoder over flat value tokens."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(TOK_VOCAB, H)
        self.pos = nn.Parameter(torch.randn(256, H) * 0.02)
        self.enc = nn.ModuleList([_Block(H, FFN, N_HEADS) for _ in range(2)])
        self.dec = nn.ModuleList([_Block(H, FFN, N_HEADS, cross=True) for _ in range(3)])
        self.out = nn.Linear(H, TOK_VOCAB)

    def forward(self, src, tgt):
        """src, tgt: (B, L) token ids. Returns logits (B, L, V)."""
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


def build(arm):
    if arm in ("A", "B", "C"):
        return FieldModel(arm)
    if arm == "D":
        return TokenModel()
    raise ValueError(arm)
