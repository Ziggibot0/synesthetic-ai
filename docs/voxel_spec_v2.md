# Voxel representation spec, v2 — density, translucency, occupancy output

STATUS: design proposal (Sean, 2026-08-30, live-riffed). NOT implemented.
Supersedes the per-cell slot encoding in `voxel_model.py` ONLY after the
AudioCaps pairs gate passes; nothing here changes the current experiment.
Viewer: v2 needs alpha + density channels; deferred until the model emits
them.

## Motivation (phenomenology -> representation)

Sean's STORY.md additions: shapes "merge, diverge, pass through each
other, or swirl together without mixing"; colors superimpose but never
blend into a mix; some colors "DNE in the physical world"; time itself
has shape. The v2 claim: each phenomenological property maps to its own
quantized dimension, rather than being squeezed into hue/sat/light.

## The voxel record

Each cell holds:

  density ρ          one value per cell (mass there; ADDS like mass adds)
  color-set          up to K=2 entries, each (hue, brightness, alpha)
                     - the superimposed set; entries alternate in time
                     in the renderer, never average (2^k branch ceiling)

Position = the cell index itself. Value dims quantized to 2 significant
figures (~100 buckets each; axes are integer cell indices already).

CRITICAL (Sean's catch, 2026-08-30): the color structure is a SET per
cell, not a single record - a cell has ONE density (mass adds) but TWO
coexisting colors (they alternate, they never blend). A one-tuple-per-
cell design would force co-located red+blue to emit a single hue -
i.e., mixing, the exact failure the architecture exists to prevent.

Multiplicity decomposition (why bounded colors, unbounded cells):

  * spatial multiplicity is UNBOUNDED: a thought lights as many cells
    as its content needs ("simple thoughts take a small part of vision;
    complex ones pack a closed space").
  * per-cell color multiplicity is BOUNDED (cap K=2): uncertainty has a
    branch ceiling at one location; k superimposed colors = 2^k
    reachable branches downstream (the transition-kernel reading).
  * density is per-cell single-valued for a reason: the field's
    gradient/integral need ONE mass value; density superposes by
    addition, hue superposes as a set. Mass adds; colors don't.

## Dim semantics — the load-bearing table

| dim            | phenomenology            | computed role                     | calculus reading (proposed)            |
|----------------|--------------------------|-----------------------------------|----------------------------------------|
| x, y, z        | where it sits in space   | cell index (8^3 grid)             | argument / support of the function     |
| hue            | "the color itself"       | type/kind (ordinal red->violet)   | which operator is in play (periodic)   |
| brightness     | how loud / salient now   | attention gate (existing act*bri) | local magnitude, weight in operations  |
| translucency α | passes through, no mix   | confidence/persistence; composited via over-operator | accumulated evidence, noisy-OR style |
| density ρ      | how much substance       | multiplicity/mass at the cell     | the FUNCTION VALUE (integrable mass)   |

Key distinctions Sean is drawing that v1 collapsed:

* brightness is NOT translucency. A bright thing can be see-through
  (stage fog lit from behind). Salience (attention) and persistence
  (evidence strength) are different variables and must not share a dim.
* translucency gets COMPOSITING semantics, not averaging: Porter-Duff
  "over" accumulation C_out = C_a + C_b*(1-alpha_a) is exactly the
  accumulator of sequential evidence / noisy-OR. Two translucent shapes
  overlap, BOTH stay visible, neither becomes a mix - the mathematical
  version of "red and blue at the same time but not purple".
* density is the INTEGRATION dim: amount of stuff at a cell. Density
  turns "shape vs field" (open question from the riff backlog) into a
  computable property: shapes = connected high-density components
  (flood fill over threshold), fields = diffuse low-density mass.

## Variable-length output: the occupancy field

Requirement: model emits as many entries as the thought needs ("talk
until the prompt is satisfied"), not a fixed slot count.

Design: model outputs a full OCCUPANCY FIELD over the 8^3 grid - per
cell, density plus a bounded color-set of K=2 entries (each hue,
brightness, alpha). "As many as needed" = however many cells get
nonzero density; adaptive compute emerges from sparsity, no stop-token
machinery, and the whole-present-moment read stays a single O(512)
pass. Cost: 512 cells x (1 + K*3) quantized dims is still trivial next
to any transformer's logits.

This also retires the v1 single-argmax-cell limit (thoughts currently
collapse to one point - the known model-side gap) WITHOUT giving up
superposition: spatial multiplicity is free, per-cell color multiplicity
is bounded at the branch ceiling. Head shape per cell: 1 density logit +
K x (hue logits, brightness, alpha, existence) - a categorical + soft
attention set, exactly the structure the renderer already draws.

## The calculus connection (why this might actually compute)

The discrete differential operators are NATIVE to a grid: gradient =
differences along the 6 neighbors; Laplacian/divergence/curl = stencil
convolutions; integrals = Riemann sums over density. In a continuous
embedding you cannot take "the difference between adjacent dims"
meaningfully - dims are arbitrary. In voxel space, adjacency IS the
semantics, so differential operators are O(1) stencil matrix multiplies
the substrate gives you for free. That is the sharpened Abacus bet:
don't teach the transformer calculus; give it a substrate where the
operators are built-in, then test whether it learns to use them.

Two tiers, honest about difficulty:

* T1 (testable soon, same iGPU): seq2seq transformer in the
  Lample-Charton setup, but functions are voxelized 1-D density fields
  (lit cells with quantized attrs); task: input field of f -> output
  field of f' (then integral). Ablations: full voxels vs -density vs
  -alpha vs plain token baseline at matched params. If the ablations
  bite, the representation - not scale - is doing the calculus. This is
  the Abacus-paper logic transplanted to calc, which (searched
  2026-08-30) nobody has published.
* T2 (north star, later): the model's internal state IS the field and
  computation is stencil dynamics (neural cellular automaton flavor).
  Much harder; do not start here.

Precision honesty: 2 sig figs is comfortable for function VALUES;
derivative targets may want 3-4 (Charton used 4 on the mantissa). Keep
the semantics layer at 2 sig figs; let the calculus head use wider
buckets if ablation says so.

## Explicit non-goals / sequencing

* Gate #1 (AudioCaps caption pairs) still runs first. This spec is the
  v2 design target, not a detour: it is implemented only after the
  mechanism (meaning-locked pairs -> structure) is confirmed.
* Translucency and density are ADDED dims; nothing in gates or eval
  changes. Viewer gets alpha + density rendering only at v2.
* Saturation: dropped from the v2 record (its job - kind-intensity -
  is folded into brightness/density). Say so in any paper text so the
  v1->v2 change is documented, not silent.