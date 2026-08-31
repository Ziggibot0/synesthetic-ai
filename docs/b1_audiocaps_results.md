# B1 — AudioCaps binding gate results (pre-registered)

Status: COMPLETE, 2026-08-30. Design pre-registered in audiocaps_pairs.py
and docs/research_roadmap.md.

## What was tested

Pre-registered question: do meaning-locked invariance pairs (same-clip
captions = re-renderings of the same audio) create more semantic
structure than random pairs?

  arm A  same     — pairs of human captions from the same 10s AudioCaps clip
  arm B  shuffled — same pair count, captions from different clips

Mechanism: VICReg invariance on same-clip pairs should make the encoder's
representation preserve meaning across surface variation. If the
mechanism works, struct-rho (rank-correlation of pairwise distances vs
CLAP's reference) should be higher for same than shuffled.

Pre-registered gates:
  PASS : rho(A) >= 0.15 and rho(A) - rho(B) >= 0.10
  KILL : rho(A) < 0.05
  GRAY : otherwise

## Results

  arm        struct-rho
  same       0.6072
  shuffled   0.6245

  delta = same - shuffled = -0.017
  verdict: GRAY (but see honest interpretation below)

## Honest interpretation — the mechanism test FAILED

The GRAY verdict is technically correct but misleading. The real finding
is worse than GRAY:

1. BOTH arms achieved high rho (~0.61-0.62), far above the 0.15 PASS bar.
   But both arms' training loss went to 0.000 — the MLP trivially
   preserves CLAP's pre-existing semantic structure regardless of which
   pairs it sees. CLAP is already a trained text encoder; the small MLP
   just passes it through.

2. Shuffled pairs scored HIGHER than same-clip pairs (0.625 vs 0.607,
   delta = -0.017). The meaning-locked pairs did not create MORE
   structure than random pairs. The pre-registered hypothesis
   ("invariance pairs from co-occurrence create semantic structure") is
   not supported.

3. The high rho is not FROM the pairs — it is FROM CLAP. The pair
   mechanism (the entire point of the test) contributes nothing
   measurable above random pairing. The original VoxelNet's rho ≈ 0 was
   about VoxelNet's own random embeddings, not about pair quality. The
   fix was never "better pairs"; it was "better encoder" — a different
   problem.

## Design caveat (honest)

The test may have been too easy for the encoder: a small MLP on top of
CLAP's 512-d embeddings can trivially preserve structure without learning
from pairs at all. The VICReg invariance term is non-binding when the
encoder already satisfies variance/covariance constraints. A stronger
test would use a from-scratch encoder (no CLAP shortcut) or a harder
bottleneck (e.g. 8-d projection) where the encoder MUST learn from pairs
to preserve structure. Whether meaning-locked pairs would help under
that harder setup is an open question — but this test does not answer it,
and the roadmap should not assume they will.

## What this means for the roadmap

- The binding lane (B1 → R1 → M1 → V1) was built on "meaning-locked
  pairs create structure." This test does not support that mechanism.
- The superposition result (C1a) is unaffected — it validated a
  representational capability, not a binding mechanism.
- Phase 2 needs redesign before proceeding: either a harder encoder
  bottleneck that forces pair-learning, or a different binding mechanism
  entirely.
- The "so what" (E1 consistency eval) cannot be reached without a
  working binding mechanism, so the full dissertation arc is paused
  until this is resolved.