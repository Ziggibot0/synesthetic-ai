# S3 — front-3-as-location results (pre-registered)

Status: COMPLETE, 2026-09-01. Design pre-registered in docs/calc_s3.md
BEFORE the run. Fixed mapping, zero training, ~2 min.

## Verdict: KILL (clean)

| arm | grid | hue | rho vs embedding | null | cells used |
|-----|------|-----|------------------|------|------------|
| A   | 8    | cos/sin | 0.0030 | 0.0054 | 249/512 |
| B   | 8    | wavelength | 0.0591 | 0.0520 | 249/512 |
| C   | 16   | cos/sin | -0.0097 | -0.0123 | 926/4096 |
| D   | 4    | cos/sin | 0.0755 | 0.0550 | 52/64 |
| E   | —    | identity | 1.0000 (ceiling) | — | — |

  G1 front-3 holds structure:  FAIL  (0.003, bar > 0.50)
  G2 beats random null:       FAIL  (0.003 vs 0.005)
  G3 within 70% of ceiling:   FAIL
  G4 grid size matters:        PASS  (D 0.076 > A 0.003 — smaller grid helps)
  G5 hue encoding:             PASS  (wavelength 0.059 > cos/sin 0.003)

## The finding

The first 3 matryoshka dims are NOT spatially coherent. Mapping them to
grid coordinates preserves essentially ZERO pairwise structure (rho 0.003,
indistinguishable from a random cell shuffle). "Location means something
by construction" is FALSE for the coarse matryoshka head.

Crucially, this is NOT a collapse-to-a-line artifact: the front-3 dims
have a balanced 3-D spread (eigen-fraction 0.418 / 0.376 / 0.207). The
dims genuinely span 3-D — they just don't place similar sentences near
each other. Matryoshka's "best 3-dim approximation" is a tangled manifold,
not a spatial map. Nearby points in the front-3 space are unrelated.

The grid-size sweep (G4) is the one informative positive: smaller grids
(D, 4^3) preserve more structure than larger ones (A, 8^3). This is
consistent with quantization acting as a coarse clustering that partially
recovers structure the raw dims lack — but even the best arm (D, 0.076)
is far below any useful bar.

## What this means for the roadmap

- "Location means something" is NOT free from the matryoshka head. The
  fixed front-3->xyz mapping is dead.
- The alternative is the LEARNED adjacency loss (S3c): train the encoder
  so that spatial displacement between thoughts tracks semantic
  similarity. That's the non-bandaid path to a meaningful grid — but it
  requires training, not a fixed map.
- This is a clean, cheap negative: it rules out the cheapest possible
  version of the core claim, and tells us the grid must be LEARNED to
  mean something, not assumed.

## Reproducibility

  py -3.12 s3_front3_location.py   # ~2 min, no training

Artifacts: results/s3_results.json.