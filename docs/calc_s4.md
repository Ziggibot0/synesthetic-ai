# S4 — Scene encoder: text -> sparse chromavox scene -> meaning back out

Status: PRE-REGISTERED 2026-09-01, BEFORE this run. This is the
experiment the original thesis always described and no prior experiment
actually tested.

## Why this experiment exists (the drift correction)

In ten experiments, a sentence has NEVER been encoded as a scene. The
encodings tested were: one cell (v1 argmax collapse), ALL 512 cells
(V1 AE, 510.6 lit — the anti-scene), and exactly one cell with one color
(S3/S3b/S3c fixed mappings). Point or smear, never a shape.

The information arithmetic that explains every failure so far: a
sentence carries ~64 numbers of meaning (64-dim matryoshka). One cell
with position + one color carries ~5 numbers. You cannot pour 64 into 5;
every fixed single-cell mapping was guaranteed to spill meaning, and did
(S3 rho 0.003, S3c color adds +0.0007). A k-cell scene holds k x ~5
numbers; k=16 gives ~80 slots — the first encoding with ROOM for the
meaning. The scene is not decoration; it is the only encoding with
capacity.

Superposition's mixed record also traces to this: C1a WON (0.11 vs
1.00) with sparse, separable toy fields; S2 LOST (0.41 vs 0.998) with
dense smears overlapping 99.5%. The variable that flipped was
scene-ness, not superposition. Superposition of scenes is untestable
until a scene encoder exists.

## The question, falsifiable

Can a LEARNED encoder map sentences into sparse chromavox SCENES
(few cells lit, colors per cell, most of the grid empty) such that
(a) meaning survives round-trip, and (b) the encoding actually is a
scene — occupancy bounded well below the grid, and different sentences
landing in different regions?

This is the stated goal verbatim: "encode text SOMEHOW as a chromavox
semantic scene and then get those meanings back out later."

## Design

Corpus: STS-B unique sentences, sentence-disjoint 85/15 split (seed
1337), same embedding cache as S2 (nomic v1.5, 64-d matryoshka). Train
pool 10,000; held-out 400 (never in any training).

Architecture: autoencoder with structured bottleneck.
  encoder: MLP 64 -> 256 -> 4,608 (as V1 arm B)
  field:   512 cells x K=3 color slots (user-specified K=3; was 2),
           each slot (freq, brightness, amplitude); NO density channel
           (user directive 2026-09-01: "density can fuck off").
           Frequency encoding (user-specified), NOT cos/sin:
           slot hue emitted as a raw frequency value, 380-700nm
           normalized; ordered axis, not periodic circle.
  sparsity: top-k occupancy per sentence. The field's per-cell salience
           is computed, and only the top-k cells (k=16) are kept lit;
           all others zeroed. Straight-through so gradients flow.
           This is the scene constraint — learned shape, forced budget.
  decoder: MLP 4,608 -> 256 -> 64, reconstruct the 64-dim embedding.

Loss = reconstruction MSE (the round-trip), as V1 arm B. The scene
constraints are architectural (top-k + frequency slots), not loss
add-ons. Position-diversity is NOT separately enforced — whether
sentences naturally spread across cells under sparsity pressure is a
MEASURED outcome (M3), not a forced one. Forcing it would be the
bandaid pattern we rejected; we test whether sparsity alone induces
spatial spread.

## Arms (pre-registered)

  A  scene       top-k=16 occupancy + K=3 freq slots, no density (the test)
  B  dense       V1 arm-B replica (no top-k, cos/sin slots) — the S2-era
                 baseline re-run at this config, the "smear" control
  C  scene-k8    top-k=8  (sparsity sweep)
  D  scene-k32   top-k=32 (sparsity sweep)
  E  identity    64-dim -> 64-dim (ceiling, rho = 1 by definition)

All arms: same corpus, same split, same seed, same decoder capacity,
same training budget (200 epochs).

## Metrics (held-out 400 sentences, 2,000 random pairs, seed 1337)

  M1 round-trip:    Spearman rho of decoded-vs-embedding pairwise
                    distances (structure preservation through the scene)
  M2 occupancy:      mean lit cells per sentence (is it a scene?)
  M3 separation:    mean pairwise lit-cell Jaccard overlap between
                    different held-out sentences (do they live in
                    different places?)
  M4 position-alive: rho of position-only distances (cell coords only,
                    no colors) vs embedding distances (S2a measured
                    -0.16; is position alive now?)
  M5 color-alive:   rho of color-only distances vs embedding distances,
                    and frequency-bin usage across 380-700nm (diversity;
                    the 'all red' pathology check)

## Pre-registered gates

  S4-G1  meaning survives the scene: arm A M1 >= 0.85
         (V1 arm B hit 0.973 dense; the honest question is the cost of
         scene constraints. 0.85 = ~88% of that ceiling.)
  S4-G2  it IS a scene: arm A M2 in [8, 32] AND M3 <= 0.30
         (bounded occupancy, and overlap far below the 0.995 pathology)
  S4-G3  position is alive: arm A M4 >= 0.10
         (beats the -0.16 dead-position null by a real margin; the
         scene must carry meaning IN WHERE, not just in color)
  PASS = G1 and G2 and G3.
  KILL  = G1 fails at M1 < 0.60 (scene constraints destroy meaning —
          the thesis's encoding cannot hold what it must hold)
  GRAY  = otherwise. If G1 passes but G2/G3 fail: meaning survives but
          sparsity alone doesn't make scenes (need position-diversity
          pressure after all — S4b, learned, pre-registered then).

## What each outcome buys

  PASS -> the foundation exists for the first time: real scenes that
  hold meaning. Next (in order, each pre-registered): superposition of
  scenes (two sparse shapes sharing a cell -> the K=3 slot mechanism,
  S5), then evidence-resolution on scenes (S2r design, now testable),
  then time/JEPA (only then).
  KILL -> the thesis's own encoding can't hold meaning under its own
  constraints, documented cleanly; the honest fork is relaxing k or
  admitting the scene is not the right encoding.
  GRAY -> informative middle; the failing gate names the missing piece.

## Honest risks, stated before the run

- Top-k straight-through may train poorly (discrete gate at k=16 with
  sigmoid-squashed salience); if A fails everywhere, C/D tell us if
  it's the k value or the mechanism.
- Frequency as raw ordered value may let the model collapse all slots
  to one frequency (the S2 all-red pathology in new clothes); M5's bin
  usage will show it and that is a finding, not a bug.
- Decoder may learn to read the 4,608-dim field as an unstructured
  vector (as V1 did), making M1 pass trivially while M2-M4 reveal the
  scene is cosmetic. This is exactly why G2/G3 gate on scene-ness
  itself, not just reconstruction.
- 200 epochs on CPU-sized MLPs on the iGPU: ~30-45 min for all arms.

## Reproducibility

  py -3.12 s4_scene_encoder.py --smoke   # pipeline check, ~5 min
  py -3.12 s4_scene_encoder.py          # full run, ~30-45 min iGPU

Artifacts: results/s4_results.json, results/s4_encoder.pt,
results/s4_run.log