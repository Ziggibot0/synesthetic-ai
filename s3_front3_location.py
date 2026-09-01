"""
s3_front3_location.py — S3: does the coarse matryoshka head make a semantic map?

Pre-registered: docs/calc_s3.md (2026-09-01, before this run).

IDEA (Sean): matryoshka front-3 dims = coarsest semantic summary. Map
front-3 -> xyz grid (quantized), back dims -> color, no density. Then
location means something BY CONSTRUCTION. This is a FIXED mapping, zero
training — the cheapest test of "location means something."

Metric: pairwise-distance Spearman rho vs the 64-dim embedding (same as
V0/V1/S2). Gates in docs/calc_s3.md.

USAGE:
  py -3.12 s3_front3_location.py
"""
import json, os
import numpy as np
from scipy.stats import spearmanr

import s2_disentangle as s2

OUT = s2.OUT
SEED = s2.SEED


def quantize_to_grid(vals, grid):
    """Map a batch of continuous values to integer cell coords in [0, grid)."""
    # min-max normalize to [0,1] then to [0, grid-1]
    lo, hi = vals.min(), vals.max()
    span = hi - lo + 1e-9
    norm = (vals - lo) / span
    return np.clip((norm * grid).astype(int), 0, grid - 1)


def build_rep(emb64, grid, hue_mode):
    """Fixed mapping: front-3 -> xyz cell, back dims -> color. No density.

    Returns (N, 3*grid^3) sparse-ish vector: for each sentence, one cell
    gets a color vector (wavelength or cos/sin), all other cells zero.
    """
    N = emb64.shape[0]
    front = emb64[:, :3]
    back = emb64[:, 3:]
    x = quantize_to_grid(front[:, 0], grid)
    y = quantize_to_grid(front[:, 1], grid)
    z = quantize_to_grid(front[:, 2], grid)
    cell = x * grid * grid + y * grid + z          # flat cell index

    if hue_mode == "cossin":
        # back dims -> cos/sin hue: use first 2 back dims as angle
        ang = np.arctan2(back[:, 1], back[:, 0])
        color = np.stack([np.cos(ang), np.sin(ang)], axis=1)   # (N,2)
    elif hue_mode == "wavelength":
        # back dims -> wavelength/energy: normalize first back dim to [380,700]nm
        b0 = back[:, 0]
        lo, hi = b0.min(), b0.max()
        wl = 380 + (b0 - lo) / (hi - lo + 1e-9) * (700 - 380)  # (N,)
        # energy = hc/lambda, normalized; represent as (wavelength, energy)
        energy = 1.0 / wl
        color = np.stack([wl / 700.0, energy / energy.max()], axis=1)  # (N,2)
    else:
        raise ValueError(hue_mode)

    # build sparse field: (N, grid^3 * 2) — one lit cell per sentence
    n_cells = grid ** 3
    rep = np.zeros((N, n_cells * 2), dtype=np.float32)
    for i in range(N):
        c = cell[i]
        rep[i, c * 2:c * 2 + 2] = color[i]
    return rep, cell


def pairwise_rho(rep, ref, n_pairs=2000, seed=SEED):
    N = rep.shape[0]
    rng = np.random.default_rng(seed)
    pairs = set()
    while len(pairs) < min(n_pairs, N * (N - 1) // 2):
        i, j = rng.integers(0, N, 2)
        if i != j:
            pairs.add((int(min(i, j)), int(max(i, j))))
    pairs = sorted(pairs)
    ii = np.array([i for i, _ in pairs]); jj = np.array([j for _, j in pairs])
    d_rep = np.linalg.norm(rep[ii] - rep[jj], axis=1)
    d_ref = np.linalg.norm(ref[ii] - ref[jj], axis=1)
    rho, _ = spearmanr(d_rep, d_ref)
    return float(rho)


def main():
    uniq = s2.load_unique_sentences()
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(uniq))
    sents = [uniq[i] for i in perm[:2000]]   # 2000 sentences, same pool as S2
    emb64 = s2.matryoshka(s2.embed_sentences(sents))

    # 3-D spread of the front-3 dims (is it a line/plane or a volume?)
    front = emb64[:, :3]
    cov = np.cov(front.T)
    evals = np.linalg.eigvalsh(cov)
    evals = np.sort(evals)[::-1]
    spread = evals / (evals.sum() + 1e-9)

    results = {"n_sentences": len(sents), "front3_eigen_frac": spread.tolist()}
    print("=" * 64)
    print("S3 — front-3-as-location (fixed mapping, no training)")
    print("=" * 64)
    print(f"front-3 dims eigen-fraction: {spread[0]:.3f} / {spread[1]:.3f} / {spread[2]:.3f}")
    print("  (if dim1 >> dim2,3 -> the front-3 is a LINE, not a volume)")

    # identity ceiling (E)
    rho_E = pairwise_rho(emb64, emb64)   # trivially 1.0
    results["E_identity"] = rho_E

    arms = {
        "A": (8, "cossin"),
        "B": (8, "wavelength"),
        "C": (16, "cossin"),
        "D": (4, "cossin"),
    }
    for name, (grid, hue) in arms.items():
        rep, cell = build_rep(emb64, grid, hue)
        rho = pairwise_rho(rep, emb64)
        # random null: shuffle cell assignments
        rng2 = np.random.default_rng(SEED + 1)
        cell_shuf = rng2.permutation(cell)
        rep_null = rep.copy()
        for i in range(rep.shape[0]):
            c = cell_shuf[i]
            rep_null[i, :] = 0
            rep_null[i, c * 2:c * 2 + 2] = rep[i, cell[i] * 2:cell[i] * 2 + 2]
        rho_null = pairwise_rho(rep_null, emb64)
        # occupancy: how many distinct cells used
        n_used = len(set(cell.tolist()))
        results[name] = {"grid": grid, "hue": hue, "rho": rho,
                         "rho_null": rho_null, "cells_used": n_used}
        print(f"  arm {name} (grid={grid}, {hue:>9}): rho={rho:.4f}  "
              f"null={rho_null:.4f}  cells_used={n_used}/{grid**3}")

    # gates
    A = results["A"]
    g1 = A["rho"] > 0.50
    g2 = (A["rho"] - A["rho_null"]) > 0.20
    g3 = A["rho"] >= 0.70 * rho_E
    g4 = results["C"]["rho"] > A["rho"] or results["D"]["rho"] > A["rho"]
    g5 = results["B"]["rho"] > A["rho"] or A["rho"] > results["B"]["rho"]
    gates = {"G1_front3_holds": g1, "G2_beats_null": g2, "G3_within_ceiling": g3,
             "G4_grid_matters": g4, "G5_hue_encoding": g5}
    verdict = "PASS" if (g1 and g2 and g3) else ("KILL" if not g1 else "GRAY")
    results["gates"] = gates
    results["verdict"] = verdict

    print("-" * 64)
    for g, v in gates.items():
        print(f"  {g}: {'PASS' if v else 'FAIL'}")
    print(f"VERDICT: {verdict}")

    path = os.path.join(OUT, "s3_results.json")
    json.dump(results, open(path, "w"), indent=2)
    print(f"saved {path}")


if __name__ == "__main__":
    main()