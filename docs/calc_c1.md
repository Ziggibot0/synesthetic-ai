# C1 — superposition + integration probe (pre-registered)

STATUS: DESIGNED, NOT RUN. Phase 1 of the roadmap. Follows the C0 calc
gate (docs/calc_results.md), which KILLed the substrate on pointwise
differentiation but explicitly left superposition and integration
untested. C1 tests exactly those two untested claims.

## Why this experiment

C0's verdict was scoped: differentiation is a LOCAL, pointwise operator,
where a transformer with self-attention on flat tokens already has full
local context — so spatial structure conferred no benefit (token baseline
D: 0.0064 vs best voxel 0.10). But the substrate's two DISTINCTIVE
claims were never exercised:

  1. SUPERPOSITION — multiple values coexisting in one cell as separate
     color-set entries (the "red AND blue, never purple" structure).
  2. INTEGRATION — NONLOCAL accumulation (Riemann over density), which
     local attention does not naturally do.

C1 asks: does the substrate win where it should? If it still loses to a
token baseline on BOTH, the kill is a blanket verdict. If it wins on
either, the substrate has a real, task-specific advantage.

## Task C1a — superposition (disentanglement)

Input : two functions f, g sampled on the same 32 x-bins, voxelized into
        the SAME cells. Each cell holds TWO color-set entries (one for f,
        one for g), each with its own hue/brightness/alpha (corrected
        encoding: hue as cos/sin, brightness separate from density).
Task  : output f' (the derivative of the FIRST function only). The model
        must attend to the f-stream and ignore the g-stream.
Why it tests the claim: the voxel arm represents the two streams as
distinct color-set entries in the same cell — the signature capability.
The token baseline must interleave [f_0,g_0,f_1,g_1,...] and disentangle
by position parity, which is harder. If the voxel arm wins, superposition
earns its keep.

## Task C1b — integration (antidifferentiation)

Input : f' (derivative field, voxelized).
Task  : output f (antiderivative), normalized so the constant is
        absorbed (target = f normalized by its own max; the model learns
        the SHAPE, which is well-posed).
Why it tests the claim: integration is NONLOCAL — it needs accumulation
across the whole domain (Riemann sum over density). Local attention on
tokens does not naturally accumulate; the voxel arm has density sums
built in. If the voxel arm wins, integration earns its keep.

## Arms (pre-registered)

For EACH task (C1a, C1b):
  V  voxel arm (corrected color encoding, per arm E of C0)
  T  token baseline (param-matched, flat sequence over the same data)

T is the load-bearing control, exactly as arm D was in C0. If T wins
both tasks, the substrate has no task-specific advantage and the kill is
blanket. If V wins either, the substrate has a real edge.

## Pre-registered gates

  C1a-G1  V beats T on superposition (test_in rel_err V < T)
  C1b-G1  V beats T on integration (test_in rel_err V < T)
  C1a-G2  V computes on superposition (rel_err < 0.15)
  C1b-G2  V computes on integration (rel_err < 0.15)

PASS (substrate has a real edge) = V beats T on at least one task AND
computes on it. KILL (blanket) = T beats V on both tasks.

## Non-goals

* No symbolic output (numeric field -> field, as in C0).
* No real-world data, no binding — Phase 2 (B1 AudioCaps) is separate.
* No T2 stencil-dynamics internal state.

## Reproducibility

Same discipline as C0: deterministic seed, per-epoch status files,
resumable checkpoints, live monitor. Artifacts in results/c1_*.

## Schedule

Data gen is CPU-only (sympy, ~seconds for smoke, ~14 min full). Training
~10-20M params on the iGPU, ~16s/epoch, 60 epochs per arm, 4 arms total
(2 tasks x 2 arms) ~ 65 min. Runs when the user gives the word.
