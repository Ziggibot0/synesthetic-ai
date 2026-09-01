# C1 — superposition + integration probe results (pre-registered)

Status: COMPLETE, 2026-08-30. Design pre-registered in
[docs/calc_c1.md](calc_c1.md) before any run.

## Task recap

Two tasks, each a voxel arm (V) vs a param-matched token baseline (T):

  C1a SUPERPOSITION: two functions f,g share the same cells; model must
      output the derivative of the f-stream only. V represents the two
      streams as separate color-set entries in one cell; T must
      disentangle an interleaved [f0,g0,f1,g1,...] sequence.
  C1b INTEGRATION: input = f' (derivative field), output = f
      (antiderivative shape). Tests NONLOCAL accumulation.

All arms: 60 epochs, deterministic seed 1337, param-matched (V 10.67M,
T 10.90M). Same generator, same families, same splits.

## Results (test_in rel_err / exact; extrap rel_err)

| arm          | test_in rel_err | exact | extrap rel_err | params   |
|--------------|-----------------|-------|----------------|----------|
| sup_voxel    | 0.1137          | 0.889 | 0.5627         | 10,674,065 |
| sup_token    | 1.0040          | 0.000 | 1.0040         | 10,903,115 |
| int_voxel    | 0.2246          | 0.796 | 0.7835         | 10,674,065 |
| int_token    | 0.0070          | 0.980 | 0.0033         | 10,903,115 |

## Gate verdicts

  C1a-G1  V beats T on superposition:  PASS  (0.1137 vs 1.0040)
  C1a-G2  V computes on superposition: PASS  (0.1137 < 0.15)
  C1b-G1  V beats T on integration:    FAIL  (0.2246 vs 0.0070)
  C1b-G2  V computes on integration:   FAIL  (0.2246 > 0.15)

## Honest interpretation (this is the important part)

The pre-registered rule says PASS if V beats T on at least one task AND
computes on it. By that letter, C1 PASSES on superposition. But the
overall story is NOT "the substrate is validated." It is SPLITTED:

1. SUPERPOSITION is the substrate's first REAL, reproducible advantage.
   V beat T 0.11 vs 1.00, and T failed outright (exact 0.0000 — it could
   not disentangle the interleaved streams at all). This is the task
   that leverages the substrate's ONE genuinely distinctive capability —
   multiple values coexisting in a cell as separate color-set entries.
   This is the first place across C0 (5 arms) and C1 (4 arms) where the
   voxel substrate beats tokens, and it is precisely the regime where it
   SHOULD.

2. INTEGRATION is another tokens-win, same as integration's sibling
   differentiation in C0. V did not even reach the compute gate (0.22),
   and T was near-perfect (0.007). CAUTION: the design doc's own
   hypothesis ("integration needs nonlocal accumulation, which local
   token attention does not naturally do") was FALSIFIED by the data —
   tokens handle it fine, and the voxel field does not. My stated
   expectation in the design was wrong, and the result overrides it.

## What this means for the roadmap

- Superposition is validated as a real substrate capability. This is the
  scientific finding worth building on: the color-set representation
  genuinely outperforms tokens at keeping co-located streams separate.
- Integration is NOT validated; do not build on it. Drop M1/V1's reliance
  on "integration as a native substrate edge" (they were about binding,
  but if they leaned on integration they should be re-scoped).
- Phase 2 (binding lane, B1 AudioCaps) remains the next gate. The
  superposition result gives a REASON to continue, but the semantic-
  meaning claim is still NOT made — this is toy calculus on a 2-color
  cap, not language. E1 (consistency) and B1 (binding) are still
  required before any publication claim.

## Reproducibility

  py -3.12 c1_data.py                # data (sup + int, seed 1337)
  py -3.12 c1_run_all.py --epochs 60 # all 4 arms (~2.4h)
  py -3.12 c1_report.py              # verdict

Full logging/checkpointing contract as in the C0 arms.
