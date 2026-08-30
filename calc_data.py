"""
calc_data.py — synthetic calculus dataset generator for the calc gate.

Generates function -> derivative pairs, voxelized into 1-D occupancy
strips, with three splits:
  * train       : poly + trig families, x in [-5, 5]
  * test_in     : poly + trig families, x in [-5, 5]  (held-out functions,
                  SAME families/domain -> in-domain generalization)
  * test_extrap : exp + combo families, x in [-8, 8]  (DISJOINT families
                  AND wider domain -> extrapolation, per calc_gate.md)

Design decisions (full rationale in docs/calc_implementation.md):
  * Per-example normalization: f and f' are each scaled to [-1, 1] by
    their own max magnitude. The model learns the normalized derivative
    SHAPE — scale-invariant and well-defined.
  * Value dims quantized to 2 significant figures (the spec's "2 sig
    figs"); position uses 16 coarse value-bins over [-1, 1].
  * Color (arm A) is a REDUNDANT channel derived from the value
    (hue = value->hue, brightness = |value|, alpha = sign). It does NOT
    encode the derivative, so it cannot leak the answer; G3 tests whether
    this redundant structured channel helps learning.
  * Output is the derivative FIELD (value-bin + density), not symbolic
    text (per calc_gate.md non-goals).

Run:  py -3.12 calc_data.py [--n-train 20000 --n-test 4000 --seed 1337]
Writes: results/calc_data.npz
"""
from __future__ import annotations
import argparse, os
import numpy as np
import sympy as sp

X_TRAIN = (-5.0, 5.0)
X_TEST_IN = (-5.0, 5.0)
X_TEST_EXTRAP = (-8.0, 8.0)
N_X = 32               # x-bins
N_VALUE_BINS = 16      # coarse value bins over [-1,1]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def sig2(x):
    """Round to 2 significant figures (spec: value dims 2 sig figs)."""
    x = float(x)
    if x == 0 or not np.isfinite(x):
        return 0.0
    return float(round(x, 2 - int(np.floor(np.log10(abs(x)))) - 1))


def sample_function(rng, family):
    """Return a sympy expression for a random function in `family`."""
    x = sp.Symbol("x")
    if family == "poly":
        deg = int(rng.integers(1, 7))            # 1..6
        coeffs = [float(rng.uniform(-1, 1)) for _ in range(deg + 1)]
        return sum(c * x**i for i, c in enumerate(coeffs))
    if family == "trig":
        a = float(rng.uniform(-1, 1)); b = float(rng.uniform(0.5, 2.0))
        c = float(rng.uniform(-3, 3)); d = float(rng.uniform(-1, 1))
        e = float(rng.uniform(0.5, 2.0)); f = float(rng.uniform(-3, 3))
        return a * sp.sin(b * x + c) + d * sp.cos(e * x + f)
    if family == "exp":
        a = float(rng.uniform(-1, 1)); b = float(rng.uniform(-0.5, 0.5))
        return a * sp.exp(b * x)
    if family == "combo":
        p = sample_function(rng, "poly")
        t = sample_function(rng, "trig")
        e = sample_function(rng, "exp")
        w1, w2, w3 = (float(rng.uniform(-1, 1)) for _ in range(3))
        return w1 * p + w2 * t + w3 * e
    raise ValueError(family)


def voxelize(expr, x_lo, x_hi):
    """Evaluate expr and its derivative on the grid; return input+target."""
    x = sp.Symbol("x")
    xg = np.linspace(x_lo, x_hi, N_X)
    f = np.atleast_1d(sp.lambdify(x, expr, "numpy")(xg)).astype(float)
    if f.size == 1:
        f = np.full(N_X, f[0])
    fp = sp.diff(expr, x)
    fpd = np.atleast_1d(sp.lambdify(x, fp, "numpy")(xg)).astype(float)
    if fpd.size == 1:
        fpd = np.full(N_X, fpd[0])
    # normalize each to [-1,1] by own max magnitude
    mf = np.max(np.abs(f)) + 1e-9
    mfp = np.max(np.abs(fpd)) + 1e-9
    fn = f / mf
    fpn = fpd / mfp
    # input features per x-bin
    value_bin = np.clip(((fn + 1) / 2 * N_VALUE_BINS).astype(int), 0, N_VALUE_BINS - 1)
    density = np.array([sig2(abs(v)) for v in fn])
    fine_value = np.array([sig2(v) for v in fn])
    hue = np.array([sig2((v + 1) / 2 * 360) for v in fn])
    brightness = np.array([sig2(abs(v)) for v in fn])
    alpha = np.array([1.0 if v >= 0 else 0.5 for v in fn])
    # targets (derivative field)
    deriv_bin = np.clip(((fpn + 1) / 2 * N_VALUE_BINS).astype(int), 0, N_VALUE_BINS - 1)
    deriv_density = np.array([sig2(abs(v)) for v in fpn])
    deriv_value = np.array([sig2(v) for v in fpn])   # continuous, for metrics + arm D
    return dict(
        value_bin=value_bin, density=density, fine_value=fine_value,
        hue=hue, brightness=brightness, alpha=alpha,
        deriv_bin=deriv_bin, deriv_density=deriv_density, deriv_value=deriv_value,
    )


def make(n, families, x_lo, x_hi, rng):
    rows = [voxelize(sample_function(rng, rng.choice(families)), x_lo, x_hi)
            for _ in range(n)]
    return {k: np.stack([r[k] for r in rows]) for k in rows[0]}


def generate(n_train, n_test, seed):
    rng = np.random.default_rng(seed)
    train = make(n_train, ["poly", "trig"], *X_TRAIN, rng)
    test_in = make(n_test, ["poly", "trig"], *X_TEST_IN, rng)
    test_extrap = make(n_test, ["exp", "combo"], *X_TEST_EXTRAP, rng)
    return train, test_in, test_extrap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-test", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    train, test_in, test_extrap = generate(a.n_train, a.n_test, a.seed)
    path = os.path.join(OUT, "calc_data.npz")
    np.savez(path,
             **{f"tr_{k}": v for k, v in train.items()},
             **{f"ti_{k}": v for k, v in test_in.items()},
             **{f"tx_{k}": v for k, v in test_extrap.items()})
    print(f"wrote {path}")
    print(f"train      : poly+trig, x in {X_TRAIN}, n={a.n_train}")
    print(f"test_in    : poly+trig, x in {X_TEST_IN}, n={a.n_test} (in-domain)")
    print(f"test_extrap: exp+combo, x in {X_TEST_EXTRAP}, n={a.n_test} (disjoint+wider)")


if __name__ == "__main__":
    main()
