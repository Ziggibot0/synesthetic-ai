# V0 — Can chromavox space hold semantic structure? (pre-registered)

STATUS: DESIGNED, NOT RUN. This is the foundational gate: before testing
computation (C0/C1) or binding (B1), test whether the chromavox
representation can preserve semantic structure at all. If it can't,
nothing downstream matters.

**Chromavox** (chroma + voxel): the colored voxel semantic space — a
3D grid of cells, each carrying density and a bounded color-set (up to
K=2 entries, each with hue/brightness/alpha), quantized to 2 sig figs.
The representation this project tests.

## Why this experiment comes first

Every prior experiment assumed the voxel space could hold semantic
content and tested something on top of that assumption. C0 tested
computation, C1 tested superposition, B1 tested binding. None tested
the foundation: does the representation preserve meaning? V0 tests that
directly by projecting a known-good semantic embedding (matryoshka
nomic v1.5) into the voxel space and measuring how much structure
survives the projection + constraints.

## The question (falsifiable)

A matryoshka embedding at 64 dims has measurable semantic structure
(pairwise distances correlate with human semantic similarity). If we
project those 64 dims into a constrained voxel field (8³=512 cells,
bounded color-sets, 2-sig-fig quantization), how much structure is
preserved?

  PASS: the voxel projection preserves most of the embedding's structure.
  KILL: the voxel constraints destroy semantic structure.

## Design

### Corpus

2,000 sentences from STS-B (Semantic Textual Similarity Benchmark) — a
standard NLP dataset where humans rated sentence pairs for meaning
similarity (0-5 scale). STS-B gives two things: (1) enough sentences
for a pairwise rho measurement, and (2) human similarity judgments for
a direct ground-truth comparison. Text-only, no audio, no video.

STS-B is the Semantic Textual Similarity Benchmark (Cer et al. 2017,
ACL). ~8,000 sentences, ~1,500 human-annotated pairs. It is the standard
ground truth for testing whether a representation preserves semantic
meaning.

### Embedding

nomic-embed-text v1.5, matryoshka truncated to 64 dims (layer-norm +
keep first 64). This is the project's target compression level and is
already working on the machine.

### Voxel field (the constrained representation)

512 cells (8³ grid). Each cell carries:
  - density: float [0,1], quantized to 2 sig figs
  - color-set: up to K=2 entries, each (hue_cos, hue_sin, brightness,
    alpha), quantized to 2 sig figs
  - empty cells: density=0, no color entries

Effective dimensionality: 512 × (1 + 2×4) = 512 × 9 = 4,608. But the
constraints (quantization, bounded sets, many cells empty) reduce the
effective information capacity below the raw dimensionality.

### Projection

Learned linear projection: 64-dim embedding → voxel field parameters,
with constraints baked INTO the forward pass (not post-hoc):
  - density: sigmoid(Linear) → [0,1], then quantize to 2 sig figs
  - hue: cos/sin of (Linear angle) → unit circle, respecting circularity
  - brightness: sigmoid(Linear) → [0,1]
  - alpha: sigmoid(Linear) → [0,1]
  - color-set existence: learned gate (sigmoid) per entry per cell

The optimizer must work WITHIN the constraints. This is the fix for the
"linear expansion trivially preserves structure" critique — if the
constraints destroy structure, the constrained projection will have low
rho even though an unconstrained linear map to the same dimensionality
(arm C) would have high rho.

An MLP arm is included as a secondary check (does non-linearity help?).

### Distance metric

Primary: cosine distance on the flattened voxel field vector (density +
color features per cell). This treats the voxel field as a structured
vector, which is what it is — the structure is in how it was produced
(constrained, quantized, spatially organized), not in how distances are
computed.

Secondary: Earth Mover's Distance (EMD) over the 3D grid, which respects
spatial neighborhoods (moving mass from cell A to nearby cell B is cheap,
to distant cell C is expensive). If EMD gives higher rho than flat
cosine, the spatial organization is contributing to structure
preservation, not just the raw dimensionality.

NOTE: the flat-cosine metric is the conservative choice — if the voxel
space can't preserve structure under flat cosine, it can't preserve it
under any metric. EMD is the "does spatial structure help?" test on top.

### Arms (pre-registered)

  A  voxel-full      64-dim → voxel field (density + color-set, quantized)
  B  voxel-density   64-dim → voxel field (density only, no color, quantized)
  C  unconstrained   64-dim → 4,608-dim linear, no constraints (capacity ceiling)
  D  random          64-dim → random projection to 4,608 dims (null)
  E  identity        64-dim → 64-dim (no projection, raw embedding ceiling)
  F  voxel-quant     64-dim → voxel field but WITHOUT quantization (continuous)
                     -- isolates quantization loss from structural loss

### Metrics

  1. Structure preservation (primary): Spearman rho of pairwise distances
     in projected space vs original 64-dim embedding, on 400 held-out
     sentences (2,000 pairs).

  3. Ground-truth correlation: Spearman rho of projected distances vs
     STS-B human similarity scores (for the STS-B annotated pairs).

### Pre-registered gates

  V0-G1  A (voxel-full) rho vs embedding > 0.50
         -- the space holds meaningful structure (not random)
  V0-G2  A (voxel-full) > D (random) by > 0.20
         -- it's not just capacity, the structure is preserved
  V0-G3  A (voxel-full) >= 0.70 × E (identity ceiling)
         -- most structure survives, not just some
  V0-G4  A (voxel-full) > B (voxel-density)
         -- color-set channels add capacity beyond density alone
  V0-G5  F (voxel-quant) - A (voxel-full) < 0.10
         -- quantization costs less than 0.10 rho (not the bottleneck)

  PASS = G1 AND G2 AND G3. The space holds semantic structure.
  G4 PASS = color channels earn their keep for structure preservation.
  G5 PASS = quantization is not the limiting factor.
  KILL = G1 FAIL or G2 FAIL. The constraints destroy structure.

### What each gate protects against

  G1: is there ANY structure? (catches "the space is random")
  G2: is it the PROJECTION preserving structure, or just high dims? (catches
      the dimensionality trap — any 4,608-dim space has capacity)
  G3: is MOST structure preserved, or just a little? (catches "technically
      nonzero but practically useless")
  G4: do color channels matter, or is density sufficient? (catches "color
      is decoration")
  G5: is quantization the bottleneck, or is it the structure? (separates
      "2 sig figs is too coarse" from "the representation is wrong")

### Controls and baselines

  Arm C (unconstrained linear to same dims) is the load-bearing control.
  If A ≈ C, the voxel constraints don't hurt — but the advantage is just
  capacity, not structure. If A < C, the constraints cost something. If
  A > C, the spatial/color structure helps beyond raw capacity (surprising
  and interesting, but not expected).

  Arm E (identity, 64-dim) is the ceiling — no projection at all. A can't
  exceed E. G3 requires A to be within 70% of E.

  Arm D (random projection) is the null. Any learned projection should
  beat it. G2 requires the margin to be significant.

### Non-goals

  * Not testing whether a model can LEARN to produce voxel fields from
    text (that's step 2, after V0 passes).
  * Not testing binding (B1's territory).
  * Not testing computation (C0/C1's territory).
  * Not testing superposition dynamics (belief-update, convergence).
  * This is a CAPACITY test: can the space hold it, not can a model
    produce it.

  IMPORTANT (the "so what" guard): a linear projection from a known-good
  embedding into a high-dimensional space will trivially preserve some
  structure — that's true of ANY sufficiently high-dimensional space.
  V0 is necessary but not sufficient: if the space CAN'T hold structure,
  the project is dead. If it CAN, the next step (train a model to PRODUCE
  voxel fields from text) is where the representation's structure must
  earn its keep against a plain high-dimensional vector. V0 does not
  prove the voxel representation is better than a random 4,608-dim
  vector — arm C tests that comparison. V0 proves the constrained
  representation is not BROKEN for semantics, which is the prerequisite
  for everything else.

### If V0 passes

  Step 2: train a small encoder to produce voxel fields from text
  (text → voxels directly), measure whether the learned projection
  preserves rho. Step 3: add superposition (multiple semantic entries
  per cell). Step 4: add belief-update dynamics.

### If V0 fails

  The voxel/color-set representation cannot preserve semantic structure
  even with a learned projection from a known-good embedding. The
  representation is fundamentally broken for semantics. Stop. The
  superposition result (C1a) stands as a representational capability on
  toy data, but the substrate has no path to semantic meaning.

### Reproducibility

  CPU-only (no GPU needed for linear projection + rho computation).
  ~5 min total: embed 5k sentences (nomic, ~30s), train linear projection
  (numpy, ~10s), compute rho on held-out set (~5s). Full reproducibility
  with deterministic seed.