"""
s2_channel_audit.py — which channel carries the semantics in the S2 encoder?

Tests Sean's escape-hatch hypothesis (2026-09-01): density soaks up the
semantic signal and color dims coast. Method: take the trained S2 encoder,
encode the 400 held-out sentences, then compute pairwise-distance Spearman
rho (vs the 64-d embedding) using ONLY selected slices of the field:

  full          all 4,608 dims                        (reference)
  density       512 dims   (the 'escape hatch')
  color both    2,048 dims (hue/brightness/alpha, slots 0+1)
  color slot0   1,024 dims
  hue only      1,024 dims (cos/sin pairs, both slots)
  bright+alpha  1,024 dims

If escape-hatch is right: density-only ~0.97, color-only ~small.
If colors carry meaning: color-only stays high even with density dropped.
"""
import json, os, sys
import numpy as np
import torch
from scipy.stats import spearmanr

import s2_disentangle as s2

OUT = s2.OUT
rng_master = np.random.default_rng(s2.SEED)


def main():
    # rebuild the exact same sentence split as s2_disentangle.main()
    uniq = s2.load_unique_sentences()
    rng = np.random.default_rng(s2.SEED)
    perm = rng.permutation(len(uniq))
    n_hold = 400
    hold_sents = [uniq[i] for i in perm[:n_hold]]

    cache = json.load(open(os.path.join(OUT, "s2_embs.json")))
    emb768 = np.array([cache[s] for s in hold_sents], dtype=np.float32)
    emb64 = s2.matryoshka(emb768)

    enc = s2.Encoder().to(s2.DEVICE)
    state = torch.load(os.path.join(OUT, "s2_encoder.pt"), map_location=s2.DEVICE)
    enc.load_state_dict(state["encoder"])
    enc.eval()

    fields = s2.encode_fields(enc, emb64)            # (400, 4608) on device
    f = fields.view(n_hold, s2.N_CELLS, s2.FEATS_PER_CELL).cpu()

    slices = {
        "full": f.reshape(n_hold, -1).numpy(),
        "density": f[:, :, 0].numpy(),
        "color_both": f[:, :, 1:9].reshape(n_hold, -1).numpy(),
        "color_slot0": f[:, :, 1:5].reshape(n_hold, -1).numpy(),
        "hue_only": f[:, :, [1, 2, 5, 6]].reshape(n_hold, -1).numpy(),
        "bright_alpha": f[:, :, [3, 4, 7, 8]].reshape(n_hold, -1).numpy(),
    }
    # occupancy pattern: which cells are (meaningfully) lit
    occ = (f[:, :, 0] > 0.01).numpy().astype(np.float32)
    slices["occupancy_binary"] = occ

    rng_p = np.random.default_rng(s2.SEED)
    pset = set()
    while len(pset) < 2000:
        i, j = rng_p.integers(0, n_hold, 2)
        if i != j:
            pset.add((int(min(i, j)), int(max(i, j))))
    pairs = sorted(pset)
    ii = np.array([i for i, _ in pairs]); jj = np.array([j for _, j in pairs])
    d_ref = np.linalg.norm(emb64[ii] - emb64[jj], axis=1)

    print(f"held-out sentences: {n_hold}, pairs: {len(pairs)}")
    print(f"{'channel':<18} {'rho vs embedding':>18}")
    for name, mat in slices.items():
        d = np.linalg.norm(mat[ii] - mat[jj], axis=1)
        rho, _ = spearmanr(d, d_ref)
        print(f"{name:<18} {rho:>18.4f}")

    # occupancy overlap (how much two sentences share lit cells)
    jac = []
    for k in range(n_hold // 2):
        a, b = occ[2 * k] > 0, occ[2 * k + 1] > 0
        jac.append((a & b).sum() / max(1, (a | b).sum()))
    print(f"\nmean lit-cell Jaccard (adjacent pairs): {np.mean(jac):.4f}")
    print(f"mean lit cells per sentence: {occ.sum(1).mean():.1f} / 512")


if __name__ == "__main__":
    main()