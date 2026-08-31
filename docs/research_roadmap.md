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

### C1. Superposition + integration probe — 🔶 PROPOSED
Question: does the substrate's signature machinery — multiple values
coexisting in one cell as separate color-set entries (superposition),
and NONLOCAL accumulation (integration/antidifferentiation) — confer any
advantage the local derivative task could not reveal?
Why it matters: C0 only ever exercised a local operator where attention
on flat tokens is already optimal. Whether the substrate wins where it
should is still open.
Task design (to be fleshed out): (a) superposition — two functions share
cells, model must keep their color-sets separate and correctly assign
derivatives; (b) integration — f' -> f (antiderivative), which needs
global accumulation (Riemann over density) that local attention does not
naturally do.
Gate: voxel arm beats/ties the token baseline on the nonlocal task; kill
if tokens still dominate on both. Artifacts: `eval/` + `docs/calc_c1.md`.

### C2. Symbolic differentiation — ⬛ DEFERRED
Question: can the substrate do SYMBOLIC differentiation (expression ->
expression), the task arm D's control was originally misworded to be?
Why deferred: the C0 arm-D deviation already established numeric-vs-
symbolic as a separate question; C1's structural findings should inform
whether this is worth a dedicated run. Artifacts: `docs/calc_c2.md`.

---

## Phase 2 — binding lane (induced synesthesia from co-occurrence)

### B1. AudioCaps caption-pair gate — 🔶 DESIGNED/NOT RUN
Question: do meaning-locked pairing pairs (same clip, ~5 paraphrased
captions) lift the structure-preservation rho off zero, versus a shuffled
control? THE gate for the binding mechanism.
Scripts exist: `scripts/prepare_audiocaps.py` (verified cdjkim CSV URLs,
clip id = youtube_id + start) + `audiocaps_pairs.py`.
Gate: PASS rho >= 0.15 AND delta >= 0.10 vs shuffled; KILL rho < 0.05.
Dep: `laion-clap`. Artifacts: `results/pairs_verdict.json`.

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
