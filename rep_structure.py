"""rep_structure.py — mission-aligned eval for synesthetic-ai (numpy-only).

The mission is NOT classification. It is: does the voxel+color
representation form a non-random, semantically-aligned space?

Runs entirely on saved artifacts (results/reps_<setup>.npy from
eval_probe.py + the nomic teacher npy), so it never contends for the GPU:

  1. Structure preservation: Spearman rho between rep-pair cosines and
     reference-space (nomic) pair cosines over 20k random pairs.
  2. Geometry: effective rank (participation ratio), per-dim std.
  3. Rep self-similarity: mean pairwise cosine (0 = spread, 1 = collapse).

Run:  py -3.12 rep_structure.py
"""
import os
import numpy as np
from scipy.stats import spearmanr

import voxel_model as V


def main():
    texts, _, _ = V.load_data()
    teacher = np.load(V.TEACHER_NPY).astype(np.float32)
    Tn = teacher / np.linalg.norm(teacher, axis=1, keepdims=True)

    rng = np.random.RandomState(0)
    ia, ib = rng.randint(0, len(texts), 20000), rng.randint(0, len(texts), 20000)
    keep = ia != ib
    ia, ib = ia[keep], ib[keep]
    tcos = (Tn[ia] * Tn[ib]).sum(1)

    print(f"reference (nomic) self-sim: mean cos {tcos.mean():+.3f}\n")
    print(f"{'setup':9s} {'struct-rho':>10s} {'mean-cos':>9s} {'eff-rank':>9s} "
          f"{'perdim-std':>10s}")
    for setup in ["free", "distill", "anchor"]:
        p = os.path.join(V.OUT, f"reps_{setup}.npy")
        if not os.path.exists(p):
            print(f"{setup:9s} {'(no saved reps)'}")
            continue
        R = np.load(p)
        Rn = R / np.linalg.norm(R, axis=1, keepdims=True)
        rcos = (Rn[ia] * Rn[ib]).sum(1)
        rho = float(spearmanr(tcos, rcos).statistic)
        vr = R.var(0)
        eff_rank = float(vr.sum() ** 2 / (vr ** 2).sum())
        print(f"{setup:9s} {rho:>+10.3f} {rcos.mean():>+9.3f} "
              f"{eff_rank:>9.1f} {R.std(0).mean():>10.2f}")

    print("\ninterpretation:")
    print("  struct-rho ~ 0  -> voxel space is semantically random "
          "(the failure this iteration must fix)")
    print("  struct-rho > 0  -> voxel geometry tracks a real semantic space")
    print("  mean-cos -> 1   -> collapse (what happened to the anchor run)")


if __name__ == "__main__":
    main()