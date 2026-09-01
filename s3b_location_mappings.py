"""
s3b_location_mappings.py — S3 diagnostic: is there ANY linear map that makes
location spatially coherent?

S3 KILLed front-3-as-location (rho 0.003). Sean's follow-up: "if the first
3 dims aren't enough, conflate dims — average groups of 3 from the first 9."
The principled version is PCA-3 (best 3-dim similarity-preserving projection).

This is a DIAGNOSTIC (no gates, like S2a): it tests several candidate
location mappings to see if ANY gives spatial coherence. If even PCA-3
fails, no linear map works and we need the learned adjacency loss (S3c).

Mappings tested (all -> 3 location dims, quantized to 8^3 grid, cos/sin
hue from back dims, no density — hue unit held CONSTANT this time):
  front3      raw first 3 dims (S3 baseline, expect ~0.003)
  avg9        average groups of 3 from first 9 dims (Sean's idea)
  pca3_64     PCA-3 of the full 64-dim embedding
  pca3_9      PCA-3 of the first 9 dims
  pca3_16     PCA-3 of the first 16 dims
"""
import json, os
import numpy as np
from scipy.stats import spearmanr

import s2_disentangle as s2

OUT = s2.OUT
SEED = s2.SEED


def quantize(vals, grid=8):
    lo, hi = vals.min(), vals.max()
    norm = (vals - lo) / (hi - lo + 1e-9)
    return np.clip((norm * grid).astype(int), 0, grid - 1)


def build_rep(emb64, loc3, grid=8):
    """loc3: (N,3) location dims. Back dims -> cos/sin hue. No density."""
    N = emb64.shape[0]
    back = emb64[:, 3:]
    x = quantize(loc3[:, 0], grid); y = quantize(loc3[:, 1], grid); z = quantize(loc3[:, 2], grid)
    cell = x * grid * grid + y * grid + z
    ang = np.arctan2(back[:, 1], back[:, 0])
    color = np.stack([np.cos(ang), np.sin(ang)], axis=1)
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
    sents = [uniq[i] for i in perm[:2000]]
    emb64 = s2.matryoshka(s2.embed_sentences(sents))

    # candidate location dims
    front3 = emb64[:, :3]
    # Sean's avg9: average groups of 3 from first 9
    f9 = emb64[:, :9]
    avg9 = np.stack([f9[:, 0:3].mean(1), f9[:, 3:6].mean(1), f9[:, 6:9].mean(1)], axis=1)
    # PCA of various prefixes
    def pca(X, k=3):
        Xc = X - X.mean(0)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        return Xc @ Vt[:k].T
    pca3_64 = pca(emb64)
    pca3_9 = pca(emb64[:, :9])
    pca3_16 = pca(emb64[:, :16])

    mappings = {
        "front3": front3,
        "avg9": avg9,
        "pca3_64": pca3_64,
        "pca3_9": pca3_9,
        "pca3_16": pca3_16,
    }

    print("=" * 64)
    print("S3b — location mapping diagnostic (hue unit held constant: cos/sin)")
    print("=" * 64)
    results = {}
    for name, loc in mappings.items():
        rep, cell = build_rep(emb64, loc)
        rho = pairwise_rho(rep, emb64)
        n_used = len(set(cell.tolist()))
        results[name] = {"rho": rho, "cells_used": n_used}
        print(f"  {name:<10} rho={rho:.4f}  cells_used={n_used}/512")

    # also report the raw location-only rho (no color) for the best candidate
    # to separate "location carries it" from "color carries it"
    best = max(results, key=lambda k: results[k]["rho"])
    loc = mappings[best]
    rep_loc, _ = build_rep(emb64, loc)
    # location-only: use cell index as a 1-hot-ish distance (just cell coords)
    x = quantize(loc[:, 0]); y = quantize(loc[:, 1]); z = quantize(loc[:, 2])
    loc_only = np.stack([x, y, z], axis=1).astype(np.float32)
    rho_loc_only = pairwise_rho(loc_only, emb64)
    results["_best"] = best
    results["_best_loc_only_rho"] = rho_loc_only
    print(f"\n  best ({best}) location-only rho (no color): {rho_loc_only:.4f}")

    path = os.path.join(OUT, "s3b_results.json")
    json.dump(results, open(path, "w"), indent=2)
    print(f"saved {path}")


if __name__ == "__main__":
    main()