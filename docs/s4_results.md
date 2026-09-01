# S4 — scene encoder results (pre-registered)

Status: COMPLETE, 2026-09-01. Design pre-registered in docs/calc_s4.md
BEFORE the run. Full run 9.7 min on iGPU (4 arms, 200 epochs, seed 1337).
Pipeline validated by smoke + unit checks (the STE hard/soft gate bug was
caught by the unit check BEFORE any run; two eval bugs caught by smoke).

## Verdict: GRAY (informative — each gate names a different piece)

| arm | M1 round-trip | M2 occupancy | M3 Jaccard | M4 position | M5 color | freq bins |
|-----|---------------|--------------|------------|-------------|----------|-----------|
| A scene-k16 | 0.8005 | 16.0 | 0.4446 | 0.1737 | 0.5504 | 10/10 |
| B dense (S2-era control) | 0.9979 | 511.9 | 0.9995 | -0.0971 | 0.9688 | 10/10 |
| C scene-k8 | 0.6061 | 8.0 | 0.2067 | 0.1222 | 0.4637 | 10/10 |
| D scene-k32 | 0.8934 | 32.0 | 0.4089 | 0.1963 | 0.5395 | 10/10 |
| E identity ceiling | 1.0 | — | — | — | — | — |

  G1 meaning survives the scene (M1 >= 0.85):  FAIL  (A = 0.80; D = 0.89)
  G2 it IS a scene (occ in [8,32], Jac <= 0.30): FAIL  (A Jac 0.44)
  G3 position alive (M4 >= 0.10):               PASS  (A 0.17, C 0.12, D 0.20)

## What actually happened — read in order

1. THE SCENE MECHANIC WORKS. Top-k enforced exactly (16.0/8.0/32.0 cells
   lit — never 1, never 510). Frequency colors span all 10 bins — the
   "all red" pathology is dead. Occupancy is no longer a knob; it is
   structural.

2. POSITION IS ALIVE FOR THE FIRST TIME. Every prior encoder measured
   position rho -0.16 to -0.10 (dead). Under sparsity pressure alone,
   position carries 0.17-0.20 — with NO adjacency loss, NO forced
   position-diversity. The dense control confirms the causal story:
   same architecture minus top-k = position dead again (-0.10).
   Sparsity is what makes location meaningful. This is the S3c
   hypothesis, and it didn't even need the loss term.

3. MEANING *MOSTLY* SURVIVES. Round-trip rho 0.80 at k=16 (vs dense
   0.998). The cost of the scene is real but bounded, and the k-sweep
   gives the capacity curve: k=8 -> 0.61, k=16 -> 0.80, k=32 -> 0.89.
   Extrapolating, k~64 recovers ~0.95. The scene is not free — 16 cells
   x ~10 numbers hold ~160 slots vs the embedding's 64, but top-k makes
   the encoder spend them redundantly.

4. THE ONE FAILING PIECE: SEPARATION (M3 Jaccard 0.44 vs gate's 0.30).
   Different sentences still share ~44% of lit cells. Sparsity induced
   position-life but not full spatial spread. Two thoughts compete here:
   (a) at k=16, two 16-cell sets in a 512-cell grid collide on ~44% —
   some overlap is combinatorially expected when the decoder wants
   reconstructive cells; (b) G2's 0.30 bar may have been optimistic at
   k=16. C (k=8) hit 0.21 — under the bar. What M3 does NOT show is the
   0.995 smear: overlap is now the minority of each scene, not the
   whole thing.

## Honest interpretation

The drift correction worked. This is the first encoding in the entire
project that is simultaneously sparse, colorful, position-alive, and
meaning-preserving. But it is a GRAY because the pre-registered bar
said PASS requires all three gates, and separation + round-trip fell
short of the written numbers. No gate-moving: A failed G1 at 0.80
(needs 0.85), and that is the honest read.

The k-sweep says the knobs trade cleanly: smaller k = more separation,
less meaning; bigger k = more meaning, less separation. The gate
structure assumed independence; they are coupled.

## What this buys (per pre-registration)

The GRAY branch applies: "meaning survives but sparsity alone doesn't
make scenes (need position-diversity pressure after all — S4b, learned,
pre-registered then)." The data sharpens that fork:

- S4b option 1: k~64 scene (k-sweep says M1 -> ~0.95, M3 -> worse).
  Buys round-trip, loses separation further.
- S4b option 2: keep k=16, add a learned overlap penalty (repulsion
  between different sentences' cell sets). Buys separation, risks
  position-life (pressure can push it to degenerate spread).
- S4b option 3 (the interesting one): accept k=32 + M3 0.41 as the
  operating point and test whether SUPERPOSITION still works on these
  scenes — the C1a win was on sparse separable fields; S2 lost on
  smears; S4 fields are sparse-but-partially-overlapping, the honest
  middle. The 0.44 overlap is not noise — it may be exactly the regime
  the color-set slots were built for (shared cells carry two colors).

That last option is the one I lean toward, but it is a design decision,
not a mechanical one. Pre-register S4b/S5 with the user.

## Reproducibility

  py -3.12 s4_scene_encoder.py --smoke   # ~1 min
  py -3.12 s4_scene_encoder.py           # ~10 min iGPU

Unit checks (run before smoke): SceneField top-k forward (exactly k
lit, content survives, non-top-k zeroed), STE hard-forward/soft-
backward (4 lit in forward, 512 with gradient in backward),
DenseEncoder shape. Artifacts: results/s4_results.json,
results/s4_encoders.pt, results/s4_full.log.