# S2 — semantic superposition disentanglement results (pre-registered)

Status: COMPLETE, 2026-08-31. Design pre-registered in docs/calc_s2.md
BEFORE the run (encoder validity bar, gates, KILL condition all written
first). Full run: 33.4 min on the 8050S iGPU (43 ms/step; the CPU path was
pathologically slow and had 2 zombie processes eating cores — killed).

## Verdict: KILL (clean)

  rho_solo                0.9975   (ceiling: reader recovers solo record ~perfectly)
  rho_colocated           0.4119   (the test — vs G1 bar 0.85, KILL bar 0.60)
  interference            0.5856   (vs G2 bar 0.05)
  role asymmetry          rho 0.426 (slot0) vs 0.349 (slot1) — both far below bar
  G1 capacity             FAIL
  G2 interference         FAIL
  G3 symmetry             FAIL
  encoder sanity          rho 0.9667 (valid, >= 0.90 bar), swap rate 0.51

All three gates fail. This is not a borderline case: the colocated reader
recovered only 41% of pairwise structure vs 99.8% solo, and its training
loss plateaued at ~0.0174 from epoch 25 of 300 — it converged, and where
it converged is bad. Capacity/optimization is not the story.

## The mechanism (why this died — worth more than the number)

1. The V1 arm-B autoencoder spreads every sentence's record across
   essentially the WHOLE grid: mean lit-cell Jaccard between paired
   sentences = 0.995. Any two unrelated sentences light the same ~all
   cells. Superposing two such fields gives the reader a field where
   every cell's content is a sum/mix of two unknown streams.
2. Density — the channel that actually carries the semantics (V0-G4
   already showed color adds nothing over density) — is superposed by
   ADDITION (min(d1+d2,1), per spec v2). Addition of two unknown
   densities is not invertible into per-stream values. The reader must
   guess the split of every cell's mass. The per-channel numbers confirm
   the channels degrade TOGETHER (density 0.410, color 0.413): there is
   no channel the reader can hide behind.
3. Slot randomization (swap 0.51) correctly killed the positional
   shortcut, so the reader has NO cue to attribute cell content to a
   stream — by design. C1a's fields never faced this because its two
   functions were engineered to be sparse and separable; these fields
   are dense and maximally overlapped.

So the honest causal chain is: dense field geometry (every sentence
everywhere) -> superposition is destructive mixing in the only channel
that matters -> unfolding is impossible, not merely hard. Note the
family resemblance to the known v1 single-argmax gap: v1 fields either
collapse to one cell (argmax) or fill the whole grid (this AE). NEITHER
uses the grid the way the phenomenology describes ("simple thoughts take
a small part of vision"). The representation, not the reader, is what
needs to change.

## What this means for the roadmap (pre-registered consequences)

- KILL = S2r (evidence-resolved superposition) and V2E (emergent
  multi-entry fields) are MOOT at this layer. They were gated on S2 PASS
  by design; running them on fields that cannot even be unfolded would
  be build-on-sand.
- The negative localizes the fix: not readers, not losses — the ENCODER
  must produce SPARSE, cell-partitioned fields so that two streams
  occupy (mostly) DISJOINT cell subsets, with the color-set mechanism
  reserved for the genuinely shared cells. That is a v2 occupancy-field
  design change (sparsity/occupancy structure in the training signal),
  and it is also exactly what STORY.md's phenomenology predicts real
  usage looks like (Jaccard 0.995 is the pathology; sparse overlap is
  the design intent).
- The superposition capability claim (C1a) stands UNCHANGED but its
  scope sharpens further: the set mechanism works when streams are
  separable IN THE FIELD GEOMETRY. "Holds multiple states until
  evidence resolves them" is blocked one level lower than we hoped —
  at field geometry, not at reader capacity.

Next representation-side test (to pre-register before any run): train
the AE with an occupancy/sparsity prior (e.g. top-k cells per sentence,
k tuned so mean pairwise lit Jaccard drops to <0.2), then re-run S2's
gates unchanged. If co-located rho recovers toward the solo ceiling,
the substrate thesis survives with a corrected encoder; if not, the
multi-state clause is dead at the semantic layer and the documented
negative stands.

## Reproducibility

  py -3.12 s2_disentangle.py --smoke   # pipeline check (~4 min iGPU)
  py -3.12 s2_disentangle.py           # full run, 33 min iGPU, seed 1337

Artifacts: results/s2_results.json (full metrics), s2_encoder.pt,
s2_readers.pt, s2_embs.json (sentence-keyed embedding cache),
s2_full.log. Unit check for superposition/assignment math included in
session log (build_superposed slot wiring + assign_best both verified
against toy fields before the run; a slot-cross-wiring bug was caught
and fixed pre-run by that check).