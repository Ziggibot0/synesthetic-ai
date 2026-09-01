# Research roadmap — synesthetic-ai

A numbered, ordered plan of every experiment in this project's arc, so
each has a place and the sequence is methodical. Each entry: what it
tests, why (which claim), status, pre-registered gate/kill, and where its
artifacts will live. New experiments are appended here BEFORE they run,
keeping the "pre-register then run" discipline that caught the leak in
the sibling project.

Status legend: ✅ DONE · 🔶 DESIGNED/NOT RUN · ⬜ PROPOSED · ⬛ DEFERRED

Boundary: this roadmap is synesthetic-ai ONLY. The embedding-vibes arc
(JEPA predictor, matryoshka dimension sweep, trajectory-shape probes)
lives in that repo and is deliberately not mixed in here.

---

## Phase 0 — substrate existence proof (COMPLETE)

### C0. Voxel-calculus gate — ✅ DONE (docs/calc_results.md)
Map f -> f' (pointwise derivative) on voxelized fields vs a token
baseline. Result: KILL on the pre-registered gate (token baseline D
0.0064 beats every voxel arm; best voxel 0.10). Controlled finding: arm
E recovered 1.19->0.10 via color-ENCODING fix (color itself was never
the problem), but ties density-only. Scope limit recorded: only a LOCAL
operator was tested; superposition and integration were not.

---

## Phase 1 — test the substrate's untested claims

### C1. Superposition + integration probe — ✅ DONE (docs/calc_c1_results.md)
SPLIT verdict. SUPERPOSITION is the substrate's FIRST real advantage:
voxel 0.1137 vs token 1.004 (token failed, exact 0.00) — the color set
representation genuinely beats tokens at keeping co-located streams
separate. INTEGRATION is another tokens-win (token 0.007 vs voxel 0.22;
voxel didn't even reach the compute gate) — the design's hypothesis that
integration favors the substrate was FALSIFIED by the data. Only
superposition is validated; do not build on integration as a native edge.
Phase 2 binding (B1) is still the next gate — semantic meaning is NOT yet
claimed (toy calculus), B1/E1 still required before publication. Code:
`c1_*.py` — built, ran, results recorded.

### C2. Symbolic differentiation — ⬛ DEFERRED
Question: can the substrate do SYMBOLIC differentiation (expression ->
expression), the task arm D's control was originally misworded to be?
Why deferred: the C0 arm-D deviation already established numeric-vs-
symbolic as a separate question; C1's structural findings should inform
whether this is worth a dedicated run. Artifacts: `docs/calc_c2.md`.

---

## Phase 2 — binding lane (induced synesthesia from co-occurrence)

### B1. AudioCaps caption-pair gate — ✅ DONE (docs/b1_audiocaps_results.md)
MECHANISM FAILED. Same-clip pairs (rho 0.607) did NOT beat shuffled pairs
(0.625; delta -0.017, gate required +0.10). Both arms' loss went to 0.000
— the MLP trivially preserves CLAP's pre-existing structure regardless of
pairs. The high rho comes from CLAP, not from the pair mechanism. The
hypothesis "meaning-locked invariance pairs create semantic structure" is
not supported. Design caveat: the test may have been too easy (MLP on
CLAP 512-d can pass without learning from pairs); a harder bottleneck
(from-scratch encoder or 8-d projection) is needed to truly test the
mechanism. Phase 2 needs redesign before proceeding.
Scripts: `scripts/prepare_audiocaps.py` + `audiocaps_pairs.py`.

### S2. Semantic superposition (co-located dual-stream recoverability) — ✅ DONE (docs/s2_results.md)
KILL, clean. Full run 2026-08-31 (33 min iGPU, pre-registered gates
docs/calc_s2.md): colocated reader recovers rho 0.41 vs solo ceiling
0.9975; all three gates fail; encoder valid (0.967), swap 0.51. MECHANISM
found: the arm-B AE spreads every sentence across ~all 512 cells (paired
lit-cell Jaccard 0.995), so superposed density (mass ADDS) is
un-invertible and no channel carries stream identity. Consequence per
pre-registration: S2r (evidence resolution) and V2E are MOOT at this
layer; next is a representation-side change — occupancy/sparsity prior
so streams partition cells — then re-run these same gates. C1a's
superposition claim stands, scope sharpened: the set mechanism needs
separability IN the field geometry, co-location alone is not enough.

### S2r. Evidence-resolved superposition (belief-update gate) — ⬜ PROPOSED
The goal's temporal clause — "hold multiple states until evidence resolves
them" — and the paper's scope-limitation #3, untested anywhere in this
repo. State = superposed field (S2 construction), evidence = an added
input semantically near one stream; an update net reads (field, evidence)
and emits a resolved field. Pre-registered gates (written after S2's
verdict, before S2r runs): pre-evidence both streams recoverable (S2-G1
reused); post-evidence the compatible stream is preserved while the
incompatible stream degrades by >= threshold; the wrong-branch control is
symmetric (resolution is content-driven, not a decay knob); and the
substrate clause holds: update-from-field >= re-encode-from-text — the
held field must be the substrate that resolved, not a cache. Blocked on
S2. Artifacts: docs/calc_s2r.md, `s2r_*.py`.

### S2b. Learned semantic superposition full (V2, Lane 1) — ⬜ PROPOSED
Follows S2 on a PASS. With the V1 learned encoder + the C1a sequence model,
feed two REAL sentences into the same cells, task = "extract stream A" —
C1a's disentanglement task transplanted onto semantic content. Reuses C1a's
architecture + V1's encoder (retrained ~20min, weights not currently saved).
Needs a fresh param-matched token baseline. If the set advantage survives
semantics, this is the "so what" C3 payoff. Blocked on S2 PASS.
Artifacts: `s2b_*`, docs/calc_s2b.md.

### S2c. Harder-bottleneck binding redo (B1, Lane 2) — ⬜ PROPOSED
From-scratch encoder (no CLAP shortcut) or 8-d projection where the model
MUST learn from pairs, to test whether B1's mechanism really is dead or just
never got exercised. Heavier (real iGPU training). Unlocks R1/M1/E1. Backstop:
even if this KILLs, embedding-vibes fallacy-trajectory work remains Sean's
stronger dissertation candidate. Blocked on S2 (or run in parallel as the
fallback lane). Artifacts: docs/calc_s2c.md.

### R1. Neutral-prose re-rendering retrain — ⬜ PROPOSED
Question: with meaning-locked pairs, does a retrain on a neutral-prose /
re-rendering corpus (same claim, different serialization) fix the
original rho ≈ 0 and give the model a multi-cell distribution (the known
single-argmax-voxel gap)?
Blocked on: corpus decision (self-contained neutral-prose corpus vs
refactor-first). Artifacts: `train/` + updated voxel model.
Note: this is where the ANCHOR_WORDS hardcoded dictionary is removed on
principle — no color dictated; every color traceable to co-occurrence.

### M1. Multimodal all-pairs VICReg — ⬛ DEFERRED
Question: extend binding across modalities — CLAP audio + frame hues +
text, all as renderings of one event, VICReg over all pairs (Barlow
triplets). Colors learned from co-occurrence, never dictated.
Depends on B1/R1 passing (binding before multimodality). Artifacts:
`train/multimodal/`.

### V1. Video stage — ⬛ DEFERRED (heaviest, last)
Question: frames as the natural anchor; precompute frame/audio/text
embeddings OFFLINE, voxel net trains on frozen embeddings (never
backprops through pixels). Depends on M1. Artifacts: `train/video/`.

---

## Phase 3 — evaluation & interpretability

### S1. Shape-vs-field probe — ⬜ PROPOSED
Question: what makes a shape vs a field? Candidate: shapes = connected
high-density hue components (flood-fill over <=512 cells); fields =
diffuse mass. Cheap, deterministic, no training. Validates the
shape/field vocabulary the representation uses. Artifacts: `eval/`.

### H1. Hue-ordinal-axis experiment — ⬜ PROPOSED
Question: does assigning hue an ordinal value (red=1..violet=7) make a
measurable, predictable "wrongness drift" in hue space (an axis you can
read off), versus an unanchored control? Turns a decoration into a
measurable claim. Artifacts: `eval/hue_axis/`.

### E1. Consistency / specificity eval — ⬜ PROPOSED
Question: does the trained model meet clinical-style "machine synesthesia"
criteria — test-retest consistency across seeds/paraphrases, structure
above a shuffled-pair null, automaticity in the forward pass (operational
definition ported from Eagleman et al. 2007)?
This is the "so what" answer and the paper's core evaluation. Depends on
B1/R1 producing a trained model. Artifacts: `eval/consistency/`.

### F1. Camera-palette vs personal-palette divergence — ⬛ DEFERRED
Question: the curve between perceptual color truth (camera palette, free,
at scale) and personal color truth (Sean's drawings/eval, scarce) as a
measurable divergence figure.
Note: hand-painting is EVAL-ONLY per the no-human-color-supervision
pivot (a model that copies Sean proves memorization, not binding).
Artifacts: `eval/divergence/`.

### V2. Trajectory movies (viewer upgrade) — ⬛ DEFERRED
Question: voxel-frame movies of a thought over time, colored by
hue-over-time. The paper's "watch the model think" hook. Depends on a
trained model (B1/R1). Artifacts: `viewer/` upgrade.

---

## Sequencing rule (the method)

Each experiment is pre-registered (this doc / its own gate doc) BEFORE
it runs, with gates written down. Phase gates are load-bearing:
- C1 must clarify the substrate before Phase 2 binding is trusted at all.
- B1 is the gate for the whole binding lane; if it KILLs, R1/M1/V1 are
  moot and the repo becomes a documented negative.
- E1 is the dissertation "so what" — nothing above it is a publication
  until consistency is measurable.

Only run the next experiment after the current one's verdict is written.
No parallel science; no post-hoc gate-movíng.
