# Calc gate — results (pre-registered, 5 arms)

Status: COMPLETE, 2026-08-30. This document records what was actually
measured and the honest interpretation. The gates were pre-registered in
[docs/calc_gate.md](calc_gate.md) before any run; nothing was changed
after the fact.

## Task recap

Map a function f (voxelized on a 1-D strip / token sequence) to its
derivative f'. Numeric field output; sympy ground truth; normalized.
Test sets: in-domain (poly+trig, x in [-5,5]) and extrapolation
(disjoint exp+combo families, x in [-8,8]). 60 epochs each, param-matched
models, deterministic seed.

## Arms and results (test_in rel_err)

| arm | input representation                          | rel_err | exact  | extrap rel_err | params   |
|-----|------------------------------------------------|---------|--------|----------------|----------|
| A   | naive color (hue scalar, brightness=density, additive) | 1.1919 | 0.315 | 1.2927 | 10,672,529 |
| B   | density only (no color)                        | 0.1031 | 0.931 | 0.4298 | 10,672,529 |
| C   | binary occupancy (no density magnitude)        | 0.1211 | 0.869 | 0.5398 | 10,672,529 |
| D   | token baseline (flat numeric sequence)         | 0.0064 | 0.982 | 0.0064 | 10,903,115 |
| E   | corrected color (cos/sin hue, brightness kept separate, concat) | 0.1036 | 0.927 | 0.4441 | 10,675,217 |

## Pre-registered gate verdicts

- G1  compute (A rel_err < 0.15):                FAIL  (1.1919)
- G2  beat control (A <= D, in + extrap):        FAIL
- G3  color pays (A < B):                        FAIL  (1.1919 vs 0.1031)
- Overall verdict per pre-registered rule:       KILL
  "voxel/color substrate does not support computation a token baseline
   doesn't already get."

A fails G1 and D succeeds, so the pre-registered kill condition was met.

## The controlled finding that survived (arm E)

Arm E was added after the A/B/C/D matrix to test the hypothesis that the
A-vs-B gap was caused by the *encoding* of color, not color itself.
E differs from A ONLY in how color is represented (hue as (cos,sin)
2-D embedding; brightness kept separate from density; features
concatenated into one projection instead of added additively). Same
data, same seed, same parameter count.

- E1  corrected color computes (< 0.15):          PASS  (0.1036)
- E2  corrected color beats naive A:              PASS  (0.1036 vs 1.1919, ~11.5x better)
- E3  corrected color beats density-only B:       FAIL  (0.1036 vs 0.1031, a tie)

Result: the naive color encoding was a genuine liability, and correcting
its representation recovered full function — but corrected color adds no
advantage over density-only on this task.

## Honest interpretation

1. On numeric differentiation, a plain token sequence (D) is far more
   accurate than any voxel/color representation (0.0064 vs best voxel
   0.10, ~16x). Differentiation is a LOCAL, pointwise operator; a
   transformer with self-attention has full local context from flat
   tokens, so spatial structure confers no benefit. This is the cleanest
   reading of the KILL verdict.

2. Arm A's failure was the color ENCODING, not color per se. Arm E's
   recovery (1.19 -> 0.10 with a representation-only change) is an
   isolated, reproducible demonstration of that claim.

3. Color does not earn a measurable advantage over density for this
   task even when correctly represented (E ~ B). Saturation/brightness
   semantics as designed do not help a pointwise derivative.

4. SCOPE LIMITATION (the honest caveat that keeps this from over-reading):
   the experiment only ever exercised a LOCAL operator and a per-cell
   single value. It never tested the substrate's distinctive claims:
   superposition (multiple values coexisting in one cell, kept as
   separate color-set entries) or integration/antidifferentiation
   (NONLOCAL accumulation, where spatial structure could plausibly beat
   token attention). Those mechanisms remain UNTESTED. "KILL" here means
   "the voxel/color substrate does not beat tokens at pointwise
   differentiation," not a blanket verdict on the substrate.

## Reproducibility

- `py -3.12 calc_data.py` (train/test_in/test_extrap, seed 1337)
- `py -3.12 calc_data.py --out calc_data_v2.npz` (same + hue_cos/hue_sin)
- `py -3.12 calc_run_all.py --epochs 60` (arms A-D)
- `py -3.12 calc_run_e.py --epochs 60` (arm E on v2 data, auto-fire after D)
- `py -3.12 calc_report.py` (verdict, reads all 5 results)

Full logging/checkpointing contract in [docs/calc_implementation.md](calc_implementation.md);
design and the arm-D control deviation in [docs/calc_gate.md](calc_gate.md).
