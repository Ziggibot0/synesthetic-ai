"""
s3c_residual_color.py — S3 diagnostic: does the post-PCA-3 residual make
diverse, meaningful colors via FREQUENCY?

Per the agreed design (Sean + Hermes, 2026-09-01):
  location = PCA-3 of the embedding (coarse/where)
  color    = RESIDUAL (full minus PCA-3) -> FREQUENCY, NOT cos/sin

Color's job is differentiation, not reconstruction. So the right tests:
  1. DIVERGENCE: are the residual-derived colors spread out, or collapsed
     to a point (the "everything red" pathology)?
  2. MEANING: do residual-color distances track residual distances
     (i.e. finer-grained semantic differences)?
  3. ORTHOGONALITY: does residual color carry info the location does NOT
     already carry? (location+color together > location alone)

Frequency encoding (user's spec): map the residual onto a frequency axis
(e.g. 380-700 nm <-> frequency 4e14-8e14 Hz). Uses one dominant residual
direction (PCA-1 of the residual) as the frequency value -> an ORDERED,
non-periodic color axis (red at one end, violet at the other), unlike the
periodic cos/sin circle.

DIAGNOSTIC, no gates (like S2a/S3b): it informs whether the "residual ->
color" design is worth building, before any trained experiment.
"""
import json, os
import numpy as np
from scipy.stats import spearmanr

import s2_disentangle as s2

OUT = s2.OUT
SEED = s2.SEED


def pca(X, k):
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    scores = Xc @ Vt[:k].T      # (N, k) in PC coordinates
    return scores, Vt           # Vt is (min(N,64), 64); Vt[:k] are the PC axes


def wrap_angle(deg):
    return (deg + 180) % 360 - 180


def main():
    uniq = s2.load_unique_sentences()
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(uniq))
    sents = [uniq[i] for i in perm[:2000]]
    emb64 = s2.matryoshka(s2.embed_sentences(sents))
    N = emb64.shape[0]

    # location = PCA-3; residual = full minus the PCA-3 reconstruction
    Xc = emb64 - emb64.mean(0)
    loc3_scores, Vt = pca(emb64, 3)
    loc3_64 = loc3_scores @ Vt[:3]         # PCA-3 reconstructed, in 64-dim
    resid = Xc - loc3_64                   # 64-dim residual (fine detail)
    loc3 = loc3_scores                     # (N,3) for metrics

    # color = frequency from residual: PCA-1 of residual -> 380..700 nm
    r1_scores, VtR = pca(resid, 1)
    r1 = r1_scores[:, 0]
    lo, hi = r1.min(), r1.max()
    nm = 380 + (r1 - lo) / (hi - lo + 1e-9) * (700 - 380)      # wavelength (ordered)
    freq = 3e8 / (nm * 1e-9)                                   # Hz (energy-like, ordered)
    color = np.stack([freq / freq.max(), nm / 700.0], axis=1)  # (N,2) frequency color

    # also derive a diversity metric on the color itself
    # (spread of the underlying frequency value)
    frac = (nm - nm.min()) / (nm.max() - nm.min() + 1e-9)
    diversity = frac.std()                       # 0 = all same, high = spread out

    # metric 1: rho of color distances vs residual distances (does color
    # track the FINE structure it came from?)
    rng_p = np.random.default_rng(SEED)
    pairs = set()
    while len(pairs) < 2000:
        i, j = rng_p.integers(0, N, 2)
        if i != j:
            pairs.add((int(min(i, j)), int(max(i, j))))
    pairs = sorted(pairs)
    ii = np.array([i for i, _ in pairs]); jj = np.array([j for _, j in pairs])

    def rho_of(A, B):
        dA = np.linalg.norm(A[ii] - A[jj], axis=1)
        dB = np.linalg.norm(B[ii] - B[jj], axis=1)
        r, _ = spearmanr(dA, dB)
        return float(r)

    rho_color_resid = rho_of(color, resid)          # color tracks residual?
    rho_color_loc = rho_of(color, loc3)             # color vs location (should be LOW)
    # metric 2: does color ADD to location? measure rho of combined vs full embedding
    combined = np.concatenate([loc3, color], axis=1)
    rho_combined_full = rho_of(combined, emb64)
    rho_loc_only_full = rho_of(loc3, emb64)

    # frequency distribution
    n_bins = 10
    hist, _ = np.histogram(nm, bins=n_bins)
    bins_used = int((hist > 0).sum())

    results = {
        "rho_color_vs_residual": rho_color_resid,   # does color track fine structure
        "rho_color_vs_location": rho_color_loc,     # orthogonality (want LOW)
        "rho_location_only_vs_full": rho_loc_only_full,
        "rho_combined_vs_full": rho_combined_full,  # location+color vs full
        "color_adds_to_location": rho_combined_full - rho_loc_only_full,
        "nm_range": [round(float(nm.min()), 1), round(float(nm.max()), 1)],
        "freq_std_fraction": round(float(diversity), 4),
        "nm_hist_bins_used": bins_used,
        "nm_hist": hist.tolist(),
    }

    print("=" * 64)
    print("S3c — residual -> color via FREQUENCY (diagnostic)")
    print("=" * 64)
    print(f"  color vs residual rho:     {rho_color_resid:.4f}   (does color track fine structure)")
    print(f"  color vs location rho:     {rho_color_loc:.4f}    (orthogonal? want LOW)")
    print(f"  location-only vs full:     {rho_loc_only_full:.4f}")
    print(f"  location+color vs full:    {rho_combined_full:.4f}")
    print(f"  color ADDS to location:    {rho_combined_full - rho_loc_only_full:+.4f}")
    print(f"  wavelength range: {nm.min():.0f}-{nm.max():.0f} nm, "
          f"spread(std-frac)={diversity:.3f}, bins-used={bins_used}/10")
    print(f"  nm histogram: {hist.tolist()}")
    print(f"  (if bins_used high -> DIVERSE; if all in 1-2 bins -> 'all red' "
          f"pathology again)")

    path = os.path.join(OUT, "s3c_results.json")
    json.dump(results, open(path, "w"), indent=2)
    print(f"saved {path}")


if __name__ == "__main__":
    main()