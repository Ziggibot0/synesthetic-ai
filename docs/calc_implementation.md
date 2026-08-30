# Calc gate — implementation notes

Companion to `docs/calc_gate.md` (the pre-registered design). This file
records the concrete implementation decisions, the rationale for each,
and every place the code deviates from the pre-registered wording — so a
reviewer can see exactly what was built and why, and can hold the
experiment to its own gates.

Status: code complete, staged not run (iGPU busy with exp9). All four
files are self-contained; zero dependency on embedding-vibes or Ollama.

## Files

  calc_data.py    synthetic dataset generator (sympy ground truth)
  calc_model.py   the four-arm transformer (FieldModel A/B/C, TokenModel D)
  calc_train.py   train + evaluate one arm; writes results/calc_<arm>.json
  calc_report.py  reads the 4 arm results, prints G1/G2/G3 + kill verdict

## The task, concretely

Input  : a function f sampled on 32 x-bins in [-5,5] (or [-8,8] for the
         extrapolation split), voxelized into a 1-D occupancy strip.
Output : the derivative f' as a field on the same 32 bins.

Per x-bin the input carries:
  value_bin  : coarse value bin (16 bins over [-1,1]) — the "position"
               of the function's value
  density    : |f| quantized to 2 sig figs — "amount of stuff"
  hue        : (f+1)/2*360 — a redundant color channel (arm A only)
  brightness : |f| — redundant (arm A only)
  alpha      : sign of f (0.5/1.0) — redundant (arm A only)
  occupancy  : density>0 (arm C only)

Target per x-bin:
  deriv_bin  : coarse bin of f' (16 bins over [-1,1])
  deriv_density : |f'| quantized to 2 sig figs
  deriv_value   : continuous f' (2 sig figs) — used for metrics and arm D

## Design decisions and rationale

### 1. Per-example normalization (f and f' each scaled to [-1,1] by own max)

Rationale: the derivative of a function is scale-dependent (d/dx of
2f is 2f'), so a model that must generalize across functions of wildly
different magnitudes needs a scale-invariant target. Normalizing each
example by its own max makes the task "predict the normalized derivative
SHAPE", which is well-defined and learnable. The absolute scale is
recoverable from the input's own max, so no information needed for the
task is lost. This is the standard trick in symbolic-integration work
(Lample & Charton normalize similarly).

### 2. Color is a REDUNDANT channel derived from the value (arm A)

Rationale: the color dims (hue/brightness/alpha) are computed from f,
NOT from f'. They therefore cannot leak the answer — G3 (does color
earn its keep?) is a fair test of whether a redundant, structured,
value-derived channel helps the model learn, not a test of whether the
model can read the answer off the color. This is the honest way to test
"does the color substrate matter" without contaminating the target.

### 3. Arm D is a NUMERIC token seq2seq, not Charton's symbolic prefix

DEVIATION FROM PRE-REGISTERED WORDING — flagged for review.

calc_gate.md says arm D is "Charton prefix-notation tokens". Charton's
task is SYMBOLIC: expression -> expression (e.g. integrate x^2 -> x^3/3).
Arms A/B/C here are NUMERIC: field -> field (values on a grid). Comparing
symbolic accuracy to numeric accuracy is not a fair control — they are
different tasks with different difficulty and different output spaces.

The methodologically correct control for "does the voxel substrate beat
a plain representation of the SAME numbers" is a numeric-token seq2seq
over the identical field data: the function's 32 values as a flat token
sequence, predicting the derivative's 32 values as a token sequence.
That is what TokenModel (arm D) implements. It shares the same data,
the same normalization, and the same numeric output space as arms A/B/C,
so any accuracy difference is attributable to the REPRESENTATION (voxel
field vs flat token sequence), which is the question the gate asks.

The symbolic-vs-numeric question (can the voxel substrate do symbolic
differentiation?) is a separate, later experiment, explicitly out of
scope here (calc_gate.md non-goals: "No symbolic output").

### 4. Param matching is verifiable, not assumed

Both model classes use H=384, FFN=1536, 8 heads. FieldModel uses 6
encoder layers; TokenModel uses 4+4 (enc+dec). The exact parameter
counts are reported in each arm's results JSON (`n_params`) and printed
by calc_report.py, so the match is checkable rather than asserted. If
the counts differ materially, the control is invalid and must be
re-matched before the gates are read.

### 5. Metrics

  rel_err : mean |pred - true| / |true| over positions where |true| > 0.05
            (the 0.05 floor avoids division blow-up near zeros; the
            derivative of a polynomial is often near zero)
  exact   : fraction of positions where predicted bin == true bin
  extrap  : rel_err on the disjoint-family, wider-domain split

G1 uses test_in rel_err < 0.15. G2 uses test_in AND test_extrap rel_err
(A <= D on both). G3 uses test_in rel_err (A < B).

## Reproducibility

  * calc_data.py --seed 1337 is deterministic (numpy default_rng).
  * calc_train.py --seed 1337 sets python/numpy/torch seeds before
    training; the model is deterministic given the seed.
  * All hyperparameters are CLI args with defaults recorded in the
    results JSON.

## Data generation timing (measured)

Full dataset (20k train + 4k test_in + 4k test_extrap) takes ~14 min
on CPU (sympy lambdify + numpy). It is CPU-only, so it can run in
parallel with exp9's iGPU work. The 2k smoke dataset took ~85s.

## Known limitations (honest)

  * The 16-bin value quantization is coarse; exact-match is a hard
    metric and will understate the model's true accuracy. rel_err is the
    primary metric for this reason.
  * Arm D uses teacher-forcing at eval (no autoregressive decoding), so
    its exact-match is an upper bound on what it would achieve
    free-running. This is a conservative choice for the control (it can
    only make D look better, which strengthens any A-beats-D result).
  * The color channel is deliberately redundant (see #2); G3 tests
    redundancy value, not novel color information.
