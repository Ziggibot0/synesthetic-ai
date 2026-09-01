# S3 — front-3-as-location: does the coarse matryoshka head make a semantic map?

Status: PRE-REGISTERED 2026-09-01, BEFORE this run.

## The idea (Sean's, 2026-09-01)

Matryoshka embeddings guarantee the first k dims are the best k-dim
summary. So the first 3 dims are the COARSEST semantic summary ("what is
this about, roughly") and the back dims are FINE detail. Sean's proposal:
map the front 3 dims -> xyz grid coordinates (quantized), map the back
dims -> color. Then location MEANS something BY CONSTRUCTION — similar
sentences land near each other, no learned loss term needed. This is the
cheapest possible test of "location means something": a FIXED mapping,
zero training.

## The question, falsifiable

Does mapping the first 3 matryoshka dims to grid coordinates (and the
back dims to color) preserve semantic structure — i.e. is the coarse head
spatially coherent, so that nearby cells hold similar meaning?

This is the load-bearing bet. If the front-3 dims are NOT spatially
coherent (nearby points unrelated), the whole "location means something"
architecture fails at the source and we need the learned adjacency loss
instead. If they ARE coherent, we get a semantic map for free.

## Design

Corpus: STS-B unique sentences (same cache as S2). Embed with nomic v1.5,
matryoshka-truncate to 64 dims (same as S2).

Mapping (fixed, no training):
  location = quantize(front 3 dims) -> (x,y,z) cell
  color    = back dims (3..63) -> per-cell color (wavelength/energy or
             cos/sin — see arms)
  density  = NONE (per Sean: density can fuck off; amplitude/energy is
             folded into color, not a separate identity channel)

Metric: pairwise-distance Spearman rho of the mapped representation vs
the original 64-dim embedding (same metric as V0/V1/S2). Plus STS-B
correlation as a secondary check.

## Arms (pre-registered)

  A  front3->xyz (8^3), back->cos/sin hue, no density
  B  front3->xyz (8^3), back->wavelength/energy hue, no density
  C  front3->xyz (16^3), back->cos/sin hue, no density   (grid-size sweep)
  D  front3->xyz (4^3),  back->cos/sin hue, no density   (grid-size sweep)
  E  control: full 64-dim identity (ceiling, rho=1.0 by definition)

Grid-size sweep (A vs C vs D) tests quantization loss: how much does
discretizing the front-3 dims to cells cost?

## Pre-registered gates

  S3-G1  A rho > 0.50            (front-3-as-location holds structure)
  S3-G2  A - random-null > 0.20   (it's the mapping, not capacity)
  S3-G3  A >= 0.70 x E            (within 70% of the identity ceiling)
  S3-G4  C > A or D > A           (grid size matters — quantization is a
                                   real cost, not a free lunch)
  S3-G5  A vs B: which hue encoding is better (no pre-registered direction)

PASS = G1 and G2 and G3. KILL = G1 FAIL (front-3 dims not spatially
coherent). GRAY otherwise.

## What each outcome buys

- PASS -> "location means something" is real and FREE. The architecture
  (location-as-choice + adjacency + wavelength color) has a foundation.
  Next: S3b — the full superposition test with this fixed mapping
  (two sentences' front-3 locations + back colors, reader unfolds).
- KILL -> the front-3 dims are not spatially coherent. We need the
  learned adjacency loss (S3c) to force location to mean something, or
  a different coarse head. Documented negative, cheaply.

## Honest risks (stated before the run)

- The front-3 dims might be a tangled manifold, not a spatial map.
  That's exactly what G1 tests.
- The front-3 dims might collapse to a line/plane (1-D embedding in a
  3-D costume). We'll check the 3-D spread of centroids explicitly.
- Quantization to 8^3 might destroy too much. The grid sweep (G4) tests
  this.

## Reproducibility

  py -3.12 s3_front3_location.py   # fixed mapping, no training, ~2 min

Artifacts: results/s3_results.json, results/s3_*.npy (mapped reps).