# S5 — superposition on scenes: results (pre-registered)

Status: COMPLETE, 2026-09-01. Design pre-registered in docs/calc_s5.md
BEFORE the run (with one dated amendment, also before the full run:
readers train on pool fields only, after the smoke exposed a
train/eval contamination that saturated at rho 1.0). Frozen S4
encoders, reader-only training, 4.6 min iGPU.

## Verdict: KILL (clean)

| arm | colocated rho | recover_acc |
|-----|---------------|-------------|
| scene k=32 (Jaccard 0.565) | 0.1975 | 0.357 |
| dense union control        | 0.2432 | 0.475 |
| scene solo (validity)      | 1.0000 | 1.000 |

  G0 validity (solo generalizes):        PASS
  G1 unfold works on scenes (>= 0.60):   FAIL  (0.20)
  G2 scenes beat smears (>= +0.10):      FAIL  (-0.05 — scenes did NOT help)
  G3 crowd regime tested (J in range):   PASS  (0.565)

## What this says, plainly

1. The solo scene is perfectly readable (rho 1.0, recall 1.0) by a
   reader that trained on OTHER sentences' scenes. The encoding
   generalizes. The substrate is fine alone.

2. Put two scenes in the same grid and the meaning is destroyed
   (solo 1.0 -> colocated 0.20). This is with the faithful un-blended
   construction: shared cells carry both colors in separate slots,
   slot order randomized, private cells preserved. The red-AND-blue
   machinery was given its exact intended job — and the reader still
   cannot use it.

3. The crowding hypothesis is falsified: scenes (56% overlap) did not
   beat the dense union control (~100% overlap). 0.20 vs 0.24, if
   anything slightly worse. Overlap fraction is not the variable that
   matters.

## Mechanism (what we actually learned)

For slot-based unfolding to work, a reader must cluster cells into
streams by color content — "these cells belong to the same thought."
That requires each sentence's colors to be SELF-COHERENT (a palette
identifiable across cells). S4 measured exactly this: M5 color-rho was
0.54 — colors carry some meaning but a sentence's palette is not a
tight signature. The unfold fails because the colors don't cohere into
per-stream clusters, not because the cells overlap. C1a worked on toy
calculus because two functions' color streams were coherent by
construction (hue = value); real-language scenes have no such
constraint.

This retro-explains all three superposition results with one variable:
- C1a PASS: toy streams, coherent colors by construction
- S2 FAIL: dense smears + incoherent palettes + destructive addition
- S5 FAIL: sparse scenes + faithful slots + incoherent palettes

## What this means for the thesis

Sean's goal clause 2 — "hold multiple states at once until evidence
resolves them" — is now KILLed at the semantic layer under every
encoding tried: dense smears (S2), sparse position-alive scenes (S5),
with the un-blended slot mechanism implemented faithfully. The one
remaining lever is a learned per-stream color-coherence pressure
(make each sentence's palette a recognizable signature), which edges
toward assigning each thought a color identity — a design fork with
its own problems (it approaches arbitrary color-naming, the thing the
no-human-prior principle forbids doing by hand; learned coherence is
not forbidden but is a real build with an honest risk of just being a
per-sentence fingerprint).

## The honest stopping point

The paper's claims all stand (V0, V1, C1a + scope limits). S5 closes
scope-limitation #1 decisively: the superposition advantage does NOT
transfer to real language under any encoding tested. That is a
complete, publishable negative arc: one novel positive (C1a), one
validated scene encoder (S4: sparse, position-alive, meaning-preserving
solo), and a clean boundary around when the color-set mechanism works.

## Reproducibility

  py -3.12 s5_superposition.py --smoke   # ~2 min
  py -3.12 s5_superposition.py           # ~5 min iGPU

Artifacts: results/s5_results.json, results/s5_full.log,
results/s5_smoke.log. Frozen encoders from S4 (results/s4_encoders.pt).