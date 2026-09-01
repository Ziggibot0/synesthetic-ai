# Chromavox: Unblended Color-Set Superposition Outperforms Token Sequences on a Disentanglement Task

Sean Kellogg

## Abstract

**Chromavox** (chroma + voxel): the colored voxel semantic space
representation in which each cell of a 3D grid carries a bounded set of
color entries (up to K=2), each with its own hue, brightness, and alpha.
Colors in a set coexist without blending — red AND blue, never purple —
matching the "simultaneous color experience" reported in the synesthesia
literature and providing a native representation for uncertainty and
superposition.

We test three claims in sequence: (1) the chromavox space can hold semantic
structure projected from a pretrained text embedding (V0, structure
preservation rho=0.984); (2) a learned MLP encoder can produce chromavox
fields from 64-dim matryoshka embeddings while preserving 95-97% of
pairwise distance structure (V1, rho=0.954-0.973); and (3) on a
superposition task where two streams share the same spatial positions,
the chromavox representation decisively outperforms a param-matched token
baseline (C1a, relative error 0.114 vs 1.004, token exact-match 0%).

The superposition result is the first demonstration that an unblended
color-set representation offers a task-specific advantage over flat token
sequences. We honestly report two negative results on the same substrate
(differentiation and integration, where tokens dominate) and discuss
the scope limitations of the positive finding.

## 1. Introduction

Representation determines capability. Abacus embeddings (Jelassi et al.
2024) showed that the right positional embedding can take a transformer
from near-0% to near-perfect on arithmetic. Lample & Charton (2020) beat
Mathematica at symbolic integration through a representation choice
(prefix-notation tokenization). The geometry of the representation, not
the model architecture, was the lever.

This paper asks whether a *structured* representation — hue as an ordinal
axis, color-sets as un-collapsed superpositions, brightness as salience,
an ego-centered spatial grid — gives a model capabilities that a plain
token sequence does not. The representation is motivated by synesthesia:
the author has synesthetic perception in which multiple colors coexist
at the same spatial location without blending. We treat this as a
computational primitive (a bounded set, not an average) and test whether
it earns its keep.

## 2. The chromavox representation

A chromavox field is an 8³ grid (512 cells). Each cell carries:
- density: float [0,1], quantized to 2 significant figures
- color-set: up to K=2 entries, each (hue_cos, hue_sin, brightness, alpha)
- empty cells: density=0, no color entries

Key properties:
- Colors never blend. Two entries in the same cell coexist as distinct
  values — the set {red, blue} is not purple. This is the structural
  claim that distinguishes chromavox from additive embedding methods.
- Spatial positions are discrete (512 cells). Thoughts occupy specific
  regions; simple thoughts use few cells, complex ones pack the space.
- Quantization to 2 sig figs provides a finite, bounded representation.

Effective dimensionality: 512 × 9 = 4,608. But the constraints (sigmoid
bounding, cos/sin hue, 2-sig-fig quantization) reduce effective capacity
below the raw dimensionality. The question is how much.

## 3. Experiments

All experiments use pre-registered gates, param-matched controls, and
deterministic seeds. Code, logs, and result JSONs are at
github.com/Ziggibot0/synesthetic-ai (commit 53c6b79).

### 3.1 V0: Can chromavox space hold semantic structure?

**Method.** Embed 2,000 STS-B test sentences with nomic-embed-text v1.5
(matryoshka-truncated to 64 dims). Project into chromavox via a fixed
orthogonal projection (Johnson-Lindenstrauss), with constraints (sigmoid,
cos/sin, 2-sig-fig quantization) baked into the forward pass. Measure
Spearman rho of pairwise distances: projected vs original embedding.

**Arms.** A (voxel-full), B (density-only), C (unconstrained linear),
D (random projection), E (identity/no projection), F (no quantization).

**Results.**

| Arm | rho (vs embedding) | rho (vs STS-B) | dims |
|-----|-------------------|----------------|------|
| A (voxel-full) | 0.984 | -0.824 | 4,608 |
| B (density-only) | 0.999 | -0.824 | 4,608 |
| C (unconstrained) | 1.000 | -0.824 | 4,608 |
| D (random) | 0.031 | +0.051 | 4,608 |
| E (identity) | 1.000 | -0.824 | 64 |
| F (no quant) | 0.984 | -0.824 | 4,608 |

**Finding.** Chromavox constraints preserve 98.4% of embedding structure
(rho=0.984 vs random 0.031). Quantization costs <0.001 rho (G5 pass).
The space is not broken for semantics. However, color channels do not
add capacity beyond density alone (G4 fail: 0.984 vs 0.999) — 512
density dimensions already exceed the 64-dim input. This is a capacity
test, not an emergence test: any sufficiently high-dimensional space
would pass. It establishes the prerequisite, not the claim.

### 3.2 V1: Can a model learn to produce chromavox fields?

**Method.** Train an MLP encoder (64 → 256 → 4,608, ~131K params) on
5,749 STS-B training pairs with two objectives: (A) distance-preservation
loss (minimize MSE of pairwise distances) and (B) autoencoder
reconstruction loss (encode then decode back to 64-dim). Constraints
(sigmoid, cos/sin, straight-through quantization) baked into the encoder
forward pass. Evaluate on 2,552 held-out test sentences.

**Results.**

| Arm | rho (vs embedding) | dims | params |
|-----|-------------------|------|--------|
| A (learned-dist) | 0.954 | 4,608 | 131K |
| B (learned-autoenc) | 0.973 | 4,608 | 262K |
| C (fixed-ortho) | 0.983 | 4,608 | 0 |
| D (random) | 0.002 | 4,608 | 0 |
| E (identity) | 1.000 | 64 | 0 |

**Finding.** A learned MLP encoder preserves 95.4% of embedding structure
(arm A) and 97.3% via autoencoder reconstruction (arm B), reaching 97% of
the fixed-orthogonal ceiling (arm C: 0.983). All gates pass (G1-G4).
The autoencoder objective (B) outperforms the distance-preservation
objective (A), suggesting that reconstructable information content is a
stronger training signal than pairwise distance matching. A model can
learn to fill chromavox space with meaning.

### 3.3 C1a: Superposition — chromavox beats tokens at disentanglement

**Method.** Two functions f, g sampled on the same 32 x-bins, voxelized
into the SAME chromavox cells. Each cell holds two color-set entries (one
for f, one for g). Task: output the derivative of f ONLY. The model must
attend to the f-stream and ignore the g-stream. Param-matched token
baseline (10.67M vs 10.90M, within 2.2%). Token baseline interleaves
[f0,g0,f1,g1,...] as a flat sequence.

Three token serialization controls: interleaved (original), concatenated
[f0..f31,g0..g31], and a corrected-color (cos/sin hue) voxel arm.

**Results.**

| Arm | test_in rel_err | exact | extrap rel_err | params |
|-----|-----------------|-------|----------------|--------|
| voxel (cos/sin) | 0.114 | 0.889 | 0.563 | 10,674,065 |
| voxel (energy) | 0.116 | 0.887 | 0.565 | 10,674,065 |
| token (interleaved) | 1.004 | 0.000 | 1.004 | 10,903,115 |
| token (concatenated) | 1.004 | 0.000 | 1.004 | 10,903,115 |

**Finding.** The chromavox voxel arm achieves relative error 0.114 while
BOTH token baselines completely fail (rel_err 1.004, exact-match 0.000).
The token failure is not a serialization artifact — concatenated tokens
fail identically to interleaved. The color-set representation genuinely
outperforms tokens at keeping co-located streams separate and extracting
the correct one.

The energy/wavelength encoding (200-1000nm, UV-IR) ties the cos/sin
encoding (0.116 vs 0.114), confirming the advantage comes from the set
structure, not the specific color values.

### 3.4 Negative results (honestly reported)

Two other operators on the same substrate went to tokens:

- **Differentiation (C0):** token baseline 0.006 vs best voxel 0.10.
  Tokens are 15x better at pointwise differentiation. The voxel substrate
  offers no advantage on local operators where attention already has
  full context.
- **Integration (C1b):** token baseline 0.007 vs voxel 0.225. The
  design hypothesis that integration (nonlocal accumulation) would
  favor the substrate was falsified by the data.

We also tested whether meaning-locked invariance pairs (same-clip
AudioCaps captions) create more semantic structure than random pairs
(B1). They do not (same rho 0.607 vs shuffled 0.625, delta -0.017). The
structure came from the pretrained encoder (CLAP), not from the pair
mechanism. The binding hypothesis remains unproven.

## 4. Discussion

The positive result is narrow but genuine: on a task that exercises
chromavox's one distinctive capability (multiple things in one cell,
kept separate), it decisively outperforms a param-matched token baseline
that completely fails. This is the first demonstration of a task-specific
advantage for an unblended color-set representation.

The negative results define the boundary: the advantage does not extend
to local pointwise operators (differentiation, integration) where token
attention already excels. The substrate is not a general-purpose
computational engine — it is a representation with a specific structural
capability (superposition/disentanglement) that tokens lack.

The V0 and V1 results establish that chromavox space can hold and
recover semantic structure from pretrained embeddings, with a learned
encoder preserving 95-97% of pairwise distance structure through the
constrained projection. This is necessary but not sufficient: it proves
the space is not broken for semantics, not that its structure is better
than a plain high-dimensional vector.

## 5. Scope limitations

1. The superposition result is on synthetic calculus (two functions
   sharing 32 bins), not semantic content. Whether the advantage extends
   to semantic ambiguity (multiple word senses, candidate referents) is
   the open question (V2).
2. V0 and V1 use a frozen pretrained encoder (nomic-embed-text v1.5) as
   input. Whether a model trained end-to-end (text → chromavox, no frozen
   encoder) can produce meaningful fields is untested.
3. No dynamic belief-update mechanism has been tested. The vision of
   "hold possibilities, update with new info, converge" requires temporal
   modeling that none of these experiments exercise.
4. STS-B is a sentence-similarity benchmark, not a reasoning or
   downstream-task benchmark. Structure preservation on STS-B does not
   imply task performance.

## 6. Related work

- **Abacus embeddings** (Jelassi et al. 2024): representation geometry
  determines arithmetic capability. Chromavox tests the same principle
  for a different representation.
- **Lample & Charton (2020)**: prefix-notation tokenization beats
  Mathematica at symbolic integration. The token baseline in C1a is
  in this lineage.
- **Anthropic toy models of superposition (2022)**: features
  superpose in neural networks as directions in activation space.
  Chromavox makes superposition an explicit, bounded set rather than
  an emergent direction.
- **Ramachandran & Hubbard (2001)**: synesthesia as cross-modal
  binding. Chromavox is motivated by but does not claim to model
  synesthetic perception.
- **Eagleman et al. (2007)**: clinical diagnostic battery for
  synesthesia (consistency, automaticity). The operational definition
  of "machine synesthesia-like binding" in this project draws from
  this framework.

## 7. Reproducibility

All code, pre-registered designs, per-epoch logs, and result JSONs are
at github.com/Ziggibot0/synesthetic-ai (commit 53c6b79). Experiments are
deterministic (seed 1337) and reproducible with:
- `py -3.12 v0_semantic_gate.py` (V0, ~5 min CPU)
- `py -3.12 v1_learned_encoder.py --epochs 200` (V1, ~20 min CPU)
- `py -3.12 c1_run_all.py --epochs 60` (C1, ~2h iGPU)

## References

1. Jelassi, S. et al. (2024). Transformers Can Do Arithmetic with the
   Right Embeddings. arXiv:2405.17399.
2. Lample, G. & Charton, F. (2020). Deep Learning for Symbolic
   Mathematics. ICLR 2020.
3. Anthropic (2022). Toy Models of Superposition.
   transformer-circuits.pub.
4. Ramachandran, V.S. & Hubbard, E.M. (2001). Synaesthesia — A Window
   into Perception, Thought and Language. J. Consciousness Studies 8(12).
5. Eagleman, D.M. et al. (2007). A standardized test battery for the
   study of synesthesia. Journal of Neuroscience Methods 159(1).
6. Cer, D. et al. (2017). Semantic Textual Similarity Benchmark.
   ACL 2017.