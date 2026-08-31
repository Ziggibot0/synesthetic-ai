# V1 — Can a model learn to produce chromavox fields from text? (pre-registered)

STATUS: DESIGNED, NOT RUN. Follows V0 (which proved the space can hold
semantic structure with a fixed orthogonal projection). V1 asks whether a
LEARNED encoder can do it — the gate between "the space can hold meaning"
and "a model can fill it with meaning."

## Why this experiment comes next

V0 proved chromavox constraints don't destroy semantic structure: a fixed
orthogonal projection from 64-dim embeddings into the 4,608-dim chromavox
field preserved 98.4% of pairwise distance structure. But that projection
wasn't learned — it was a random rotation that preserves distances by
construction (Johnson-Lindenstrauss). No model had to figure out WHERE in
the grid to place semantic content.

V1 tests the learned version: can a neural network, trained by gradient
descent, discover how to map embeddings into chromavox fields such that
semantic structure is preserved? The constraints (sigmoid density, cos/sin
hue, bounded color-sets, 2-sig-fig quantization) make this harder than a
plain linear map because they're nonlinear and (for quantization)
non-differentiable. The question is whether learning can overcome them.

## The question (falsifiable)

Can a learned encoder (MLP, trained on text embeddings) produce chromavox
fields that preserve semantic structure as well as the fixed orthogonal
projection (V0's ceiling)?

  PASS: the learned encoder preserves most of the structure the fixed
        projection did — the model can fill the space with meaning.
  KILL: learning the constrained mapping is too hard — the encoder
        collapses or preserves far less structure than the fixed projection.

## Design

### Input

64-dim matryoshka nomic-embed-text v1.5 embeddings (frozen — we're
testing the projection, not the encoder). Same embeddings as V0.

### Output

Chromavox field: 512 cells (8³), each with density + up to K=2 color-set
entries (hue_cos, hue_sin, brightness, alpha). Constraints baked into the
forward pass:
  - density: sigmoid(Linear) → [0,1]
  - hue: cos/sin of (Linear angle) → unit circle
  - brightness: sigmoid(Linear) → [0,1]
  - alpha: sigmoid(Linear) → [0,1]
  - quantization: straight-through estimator (round in forward pass,
    pass gradient through unchanged in backward — standard trick for
    learning through non-differentiable quantization)

### Encoder architecture

MLP: 64 → 256 → 4,608 (one hidden layer, GELU activation). ~131K params.
Simple enough to train on CPU in minutes. The hidden layer gives it
enough capacity to learn a nonlinear mapping (the constraints require
nonlinearity — a linear map followed by sigmoid is still monotone, but
the hidden layer lets it learn WHICH cells to fill and which to leave
empty).

### Training objectives (two arms, head-to-head)

  Arm A — distance preservation loss:
    Sample pairs of sentences. Compute their pairwise distances in the
    original 64-dim embedding (d_ref) and in the chromavox output (d_vox).
    Loss = MSE(d_vox, d_ref). Directly optimizes for structure preservation.

  Arm B — autoencoder loss:
    Encoder: 64 → chromavox (4,608-dim, constrained).
    Decoder: chromavox → 64 (linear, unconstrained).
    Loss = MSE(decode(encode(x)), x). If the decoder can reconstruct the
    original embedding from the chromavox field, the field must contain
    the semantic information.

  These test two different definitions of "holds meaning":
    A: pairwise distances are preserved (structure)
    B: the information can be recovered (information content)

### Corpus

Full STS-B: train (5,749 pairs) + validation (1,500 pairs) for training,
test (1,379 pairs) for evaluation. Embed all unique sentences with nomic
v1.5, matryoshka-truncate to 64 dims. ~8,000 unique sentences.

### Arms (pre-registered)

  A  learned-dist      MLP encoder, distance-preservation loss, constrained
  B  learned-autoenc   MLP encoder + decoder, reconstruction loss, constrained
  C  fixed-ortho       V0's fixed orthogonal projection (ceiling — not learned)
  D  random            random projection (null)
  E  identity          64-dim → 64-dim (no projection, raw ceiling)

### Metrics (same as V0)

  1. Structure preservation: Spearman rho of pairwise distances in
     projected space vs original 64-dim embedding, on held-out test set
     (400 sentences, 2,000 pairs).
  2. Ground-truth correlation: Spearman rho of projected distances vs
     STS-B human similarity scores.

### Pre-registered gates

  V1-G1  A (learned-dist) rho > 0.50
         -- a learned encoder CAN preserve structure
  V1-G2  A (learned-dist) > D (random) by > 0.20
         -- it's learning, not just capacity
  V1-G3  A (learned-dist) >= 0.70 × C (fixed-ortho ceiling)
         -- learning doesn't lose too much vs the free projection
  V1-G4  B (autoencoder) rho >= 0.50
         -- information content is preserved (not just distances)
  V1-G5  A (learned-dist) > B (autoencoder) OR B > A
         -- tells us which objective is better (no pre-registered direction)

  PASS = G1 AND G2 AND G3. A model can learn to produce chromavox fields.
  G4 PASS = reconstruction works (information is recoverable).
  KILL = G1 FAIL or G2 FAIL. Learning the constrained mapping is too hard.

### What each gate protects against

  G1: can the model learn ANY structure preservation? (catches collapse)
  G2: is it LEARNING or just high dims? (catches the capacity trap)
  G3: how much does LEARNING cost vs a free rotation? (catches "technically
      works but much worse than the ceiling")
  G4: is the information RECOVERABLE, or just distances? (separates
      "distances look right" from "the content is actually there")
  G5: which training objective is better? (informs future architecture)

### Training details

  - Optimizer: AdamW, lr=3e-4, weight_decay=1e-4
  - Batch: 256 sentences (sample 2,000 random pairs per epoch for arm A)
  - Epochs: 200 (small model, small data — converges fast)
  - Device: CPU (model is 131K params — no GPU needed)
  - Seed: 1337 (deterministic)
  - Time estimate: ~5-10 min per arm on CPU

### Non-goals (same discipline as V0)

  * Not testing superposition (multiple meanings per cell) — that's V2.
  * Not testing dynamic updates (eliminate/add hypotheses) — that's V3.
  * Not testing convergence to stable state — that's V4.
  * Not testing whether chromavox structure is BETTER than plain vectors
    — arm C (fixed-ortho) is the ceiling, not a competitor to beat.
  * This is a LEARNABILITY test: can a model learn to fill the space?
    V0 proved the space can hold it. V1 proves a model can put it there.

### If V1 passes

  V2: test semantic superposition — feed ambiguous sentences, see if the
  learned encoder places multiple candidate interpretations as separate
  color-set entries in one cell. This connects V0 (space holds meaning) +
  V1 (model fills it) + C1a (superposition works) into one test.

### If V1 fails

  The constraints make learning too hard. The model can't discover how to
  map embeddings into the constrained chromavox field through gradient
  descent. Options: relax constraints (remove quantization, increase K),
  use a bigger model (transformer encoder instead of MLP), or use a
  different training objective. But the base case is that the chromavox
  representation is too hard for a model to learn to produce — which would
  be a real finding, not a failure of execution.

### Reproducibility

  CPU-only. ~20 min total: embed 8k sentences (~5 min), train 2 arms
  (~10 min), evaluate (~1 min). Full reproducibility with deterministic
  seed. Caches V0's embeddings if available.