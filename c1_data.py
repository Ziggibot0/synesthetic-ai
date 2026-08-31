"""
c1_data.py — dataset generator for the C1 superposition + integration probe.

Two tasks (see docs/calc_c1.md):

  C1a SUPERPOSITION: two functions f, g on the same 32 x-bins, voxelized
      into the SAME cells. Each cell holds TWO color-set entries (f and
      g). Target = derivative of f ONLY. The model must disentangle the
      f-stream from the g-stream.
  C1b INTEGRATION: input = f' (derivative field), target = f
      (antiderivative shape, constant absorbed by per-example
      normalization).

Each task is generated with the SAME function families and seed so the
voxel arm (V) and token baseline (T) see identical data.

Design decisions (rationale in docs/calc_c1.md):
  * Per-example normalization to [-1,1] by own max (as in C0).
  * Corrected color encoding (hue as cos/sin, brightness separate from
    density, alpha = sign) — the arm-E representation that recovered
    C0's arm A.
  * For C1a, the two streams are stored as SEPARATE per-position feature
    arrays (f_* and g_*), so the voxel arm can represent them as two
    color-set entries in one cell. The token baseline interleaves them
    as a flat sequence [f_0,g_0,f_1,g_1,...].

Run:  py -3.12 c1_data.py [--n-train 20000 --n-test 4000 --seed 1337]
Writes: results/c1_data.npz
"""
from __future__ import annotations
import argparse, os
import numpy as np
import sympy as sp

X_TRAIN = (-5.0, 5.0)
X_TEST_IN = (-5.0, 5.0)
X_TEST_EXTRAP = (-8.0, 8.0)
N_X = 32
N_VALUE_BINS = 16
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def sig2(x):
    x = float(x)
    if x == 0 or not np.isfinite(x):
        return 0.0
    return float(round(x, 2 - int(np.floor(np.log10(abs(x)))) - 1))


def sample_function(rng, family):
    x = sp.Symbol("x")
    if family == "poly":
        deg = int(rng.integers(1, 7))
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


def _eval(expr, xg):
    x = sp.Symbol("x")
    v = np.atleast_1d(sp.lambdify(x, expr, "numpy")(xg)).astype(float)
    if v.size == 1:
        v = np.full(N_X, v[0])
    return v


def _color_features(v):
    """Corrected color encoding for a normalized value array v in [-1,1]."""
    value_bin = np.clip(((v + 1) / 2 * N_VALUE_BINS).astype(int), 0, N_VALUE_BINS - 1)
    density = np.array([sig2(abs(x)) for x in v])
    fine_value = np.array([sig2(x) for x in v])
    hue_deg = (v + 1) / 2 * 360
    hue_cos = np.array([sig2(np.cos(np.deg2rad(h))) for h in hue_deg])
    hue_sin = np.array([sig2(np.sin(np.deg2rad(h))) for h in hue_deg])
    brightness = np.array([sig2(abs(x)) for x in v])
    alpha = np.array([1.0 if x >= 0 else 0.5 for x in v])
    return dict(value_bin=value_bin, density=density, fine_value=fine_value,
                hue_cos=hue_cos, hue_sin=hue_sin, brightness=brightness,
                alpha=alpha)


def _norm(v):
    m = np.max(np.abs(v)) + 1e-9
    return v / m


def make_superposition(n, families, x_lo, x_hi, rng):
    """C1a: two functions f,g; target = derivative of f only."""
    xg = np.linspace(x_lo, x_hi, N_X)
    rows = []
    for _ in range(n):
        f = sample_function(rng, rng.choice(families))
        g = sample_function(rng, rng.choice(families))
        fv = _norm(_eval(f, xg))
        gv = _norm(_eval(g, xg))
        fp = _norm(_eval(sp.diff(f, sp.Symbol("x")), xg))
        ff = _color_features(fv)
        gf = _color_features(gv)
        tgt = _color_features(fp)
        rows.append({
            **{f"f_{k}": v for k, v in ff.items()},
            **{f"g_{k}": v for k, v in gf.items()},
            **{f"t_{k}": v for k, v in tgt.items()},
        })
    return {k: np.stack([r[k] for r in rows]) for k in rows[0]}


def make_integration(n, families, x_lo, x_hi, rng):
    """C1b: input = f' (derivative), target = f (antiderivative shape)."""
    xg = np.linspace(x_lo, x_hi, N_X)
    rows = []
    for _ in range(n):
        f = sample_function(rng, rng.choice(families))
        fv = _norm(_eval(f, xg))
        fp = _norm(_eval(sp.diff(f, sp.Symbol("x")), xg))
        inp = _color_features(fp)   # input is the derivative field
        tgt = _color_features(fv)   # target is the antiderivative shape
        rows.append({
            **{f"in_{k}": v for k, v in inp.items()},
            **{f"t_{k}": v for k, v in tgt.items()},
        })
    return {k: np.stack([r[k] for r in rows]) for k in rows[0]}


def generate(n_train, n_test, seed):
    rng = np.random.default_rng(seed)
    return {
        "sup_train": make_superposition(n_train, ["poly", "trig"], *X_TRAIN, rng),
        "sup_test_in": make_superposition(n_test, ["poly", "trig"], *X_TEST_IN, rng),
        "sup_test_extrap": make_superposition(n_test, ["exp", "combo"], *X_TEST_EXTRAP, rng),
        "int_train": make_integration(n_train, ["poly", "trig"], *X_TRAIN, rng),
        "int_test_in": make_integration(n_test, ["poly", "trig"], *X_TEST_IN, rng),
        "int_test_extrap": make_integration(n_test, ["exp", "combo"], *X_TEST_EXTRAP, rng),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-test", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="c1_data.npz")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    d = generate(a.n_train, a.n_test, a.seed)
    path = os.path.join(OUT, a.out)
    np.savez(path, **{f"{k}_{kk}": v for k, dd in d.items() for kk, v in dd.items()})
    print(f"wrote {path}")
    print(f"superposition: train {a.n_train}, test_in {a.n_test}, extrap {a.n_test}")
    print(f"integration  : train {a.n_train}, test_in {a.n_test}, extrap {a.n_test}")


if __name__ == "__main__":
    main()
