"""
audiocaps_pairs.py - GATE test (b): does caption-multiplicity lift rho?

Pre-registered question: the text encoder's invariance pairs were weak
(article vs content-masked) and the pooled rep space came out semantically
random (struct-rho -0.000 free / +0.002 distill, rep_structure.py 088712b).
AudioCaps ships ~5 independent human captions per 10-second clip. If
same-clip captions are true re-renderings (same meaning, different words),
training the same architecture on SAME-CLIP caption pairs should lift
struct-rho above zero where SHUFFLED pairs (different clips, same count)
do not. That contrast IS the mechanism test:

    arm A  same        - pairs of captions from the same clip
    arm B  shuffled    - same pair count, captions from different clips
    verdict (audiocaps_pairs.py --report):
        PASS : rho(A) >= 0.15 and rho(A)-rho(B) >= 0.10
        FAIL : rho(A) < 0.05   (STORY.md kill metric)
        GRAY : otherwise - inspect per-arm loss curves before deciding

Prereqs (NOT downloaded at setup time; runtime downloads only what it
names explicitly):
  * pip install laion-clap           (CLAP 512-d text embeddings)
  * data/audiocaps_pairs.json built by scripts/prepare_audiocaps.py
  * this script caches CLAP text embeddings to data/audiocaps_text_emb.npy
  * scipy for the rho metric (present on this box)

Design notes:
  * Input here is the CLAP text embedding (512-d), not BPE ids: the test
    targets the PAIR MECHANISM, not the tokenizer. A small MLP encoder +
    the same pooled-rep geometry stands in for the full VoxelNet; if the
    mechanism lifts rho here, the same pairs go into the real VoxelNet.
  * Struct-rho mirror of rep_structure.py: rank-corr of pairwise distances
    (reps vs CLAP embeddings) on a fixed 400-caption sample.

Run:   py -3.12 audiocaps_pairs.py --arm same
       py -3.12 audiocaps_pairs.py --arm shuffled
       py -3.12 audiocaps_pairs.py --report
Writes results/history_pairs_<arm>.json and results/pairs_verdict.json.
"""
from __future__ import annotations
import argparse, json, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

GATES = {"rho_pass": 0.15, "delta_pass": 0.10, "kill": 0.05}


def load_clap(max_rows=20000):
    """CLAP text embeddings, cached to data/audiocaps_text_emb.npy."""
    import torch
    emb_path = os.path.join(HERE, "data", "audiocaps_text_emb.npy")
    meta_path = os.path.join(HERE, "data", "audiocaps_pairs.json")
    if os.path.exists(emb_path):
        return np.load(emb_path)
    if not os.path.exists(meta_path):
        sys.exit(f"missing {meta_path} - build it first with "
                 f"scripts/prepare_audiocaps.py (this script refuses to "
                 f"fetch anything on its own)")
    rows = json.load(open(meta_path))[:max_rows]
    from laion_clap import CLAP_Module
    m = CLAP_Module(enable_fusion=False)
    m.load_ckpt()                      # explicit: 630k-audioset checkpoint
    texts = [r["caption"] for r in rows]
    embs = []
    for i in range(0, len(texts), 256):
        e = m.get_text_embedding(texts[i:i + 256], use_tensor=False)
        embs.append(np.asarray(e, dtype=np.float32))
        print(f"clap embed {min(i + 256, len(texts))}/{len(texts)}", flush=True)
    E = np.concatenate(embs).astype(np.float32)
    np.save(emb_path, E)
    return E


def make_pairs(rows, arm, seed=1337):
    """same: caption pairs sharing a clip. shuffled: same count, captions
    from different clips (kills the semantic lock, keeps pair count)."""
    import random
    rng = random.Random(seed)
    by_clip = {}
    for i, r in enumerate(rows):
        by_clip.setdefault(r["clip_id"], []).append(i)
    same = [(i, j) for ids in by_clip.values() if len(ids) > 1
            for a, i in enumerate(ids) for j in ids[a + 1:]]
    if arm == "same":
        return same
    out, tries = [], 0
    while len(out) < len(same) and tries < len(same) * 50:
        tries += 1
        i, j = rng.sample(range(len(rows)), 2)
        if rows[i]["clip_id"] != rows[j]["clip_id"]:
            out.append((i, j))
    return out


def vicreg(z1, z2, sim_c=10.0, var_c=10.0, cov_c=5.0):
    """Same hinge as voxel_model.vicreg, inlined (no data deps)."""
    import torch
    import torch.nn.functional as F
    std1 = torch.sqrt(z1.var(0) + 1e-4)
    std2 = torch.sqrt(z2.var(0) + 1e-4)
    var = F.relu(1.0 - std1).mean() + F.relu(1.0 - std2).mean()
    z1n = (z1 - z1.mean(0)) / (std1 + 1e-4)
    z2n = (z2 - z2.mean(0)) / (std2 + 1e-4)
    N, D = z1.shape
    c12 = (z1n.T @ z2n) / N
    c11 = (z1n.T @ z1n) / N
    c22 = (z2n.T @ z2n) / N
    off = ((c12.pow(2).sum() - c12.diag().pow(2).sum())
           + (c11.pow(2).sum() - c11.diag().pow(2).sum())
           + (c22.pow(2).sum() - c22.diag().pow(2).sum())) / (3 * D * (D - 1))
    return sim_c * F.mse_loss(z1, z2) + var_c * var + cov_c * off


def make_net(in_dim, latent=64, hid=256):
    """Standalone stand-in encoder: same pooled-rep geometry contract as
    VoxelNet's rep (64-d), fed by CLAP text embeddings. Kept separate so
    the gate test cannot be confounded by leftover checkpoint state."""
    import torch
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(in_dim, 256), nn.GELU(),
        nn.Linear(256, latent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["same", "shuffled"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    resdir = os.path.join(HERE, "results")
    os.makedirs(resdir, exist_ok=True)

    if args.report:
        out = {}
        for arm in ["same", "shuffled"]:
            p = os.path.join(resdir, f"history_pairs_{arm}.json")
            if os.path.exists(p):
                out[arm] = json.load(open(p))["rho"]
        print(json.dumps(out, indent=1))
        if len(out) == 2:
            d = out["same"] - out["shuffled"]
            verdict = ("PASS" if out["same"] >= GATES["rho_pass"]
                       and d >= GATES["delta_pass"]
                       else "FAIL" if out["same"] < GATES["kill"] else "GRAY")
            print(f"delta {d:+.3f} -> gate {verdict}")
            json.dump({"rho_same": out["same"], "rho_shuffled": out["shuffled"],
                       "delta": d, "verdict": verdict, "gates": GATES},
                      open(os.path.join(resdir, "pairs_verdict.json"), "w"),
                      indent=1)
        return

    if not args.arm:
        sys.exit("--arm same | shuffled   (or --report after both arms)")

    import torch
    E = load_clap()
    rows = json.load(open(os.path.join(HERE, "data", "audiocaps_pairs.json")))
    pairs = make_pairs(rows[:len(E)], args.arm)
    print(f"arm={args.arm}: {len(E)} captions, {len(pairs)} pairs")

    dev = os.environ.get("VOXEL_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    net = make_net(E.shape[1]).to(dev).train()
    torch.manual_seed(1337); np.random.seed(1337)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.01)

    T = torch.tensor(E, dtype=torch.float32, device=dev)
    P = torch.tensor(pairs, dtype=torch.long, device=dev)   # [n,2]
    hist, t0 = [], time.time()
    BS = 128
    for ep in range(args.epochs):
        perm = torch.randperm(len(P), device=dev)
        tot = n = 0
        for s in range(0, len(P), BS):
            sel = P[perm[s:s + BS]]
            r1 = net(T[sel[:, 0]])
            r2 = net(T[sel[:, 1]])
            loss = vicreg(r1, r2)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); n += 1
        hist.append(dict(epoch=ep, loss=tot / max(1, n),
                         elapsed=time.time() - t0))
        print(f"ep{ep:3d} L={tot/max(1,n):.3f}")

    # ---- structure evaluation: rho(rep distances, clap distances) ----
    from scipy.stats import spearmanr
    from scipy.spatial.distance import pdist, squareform
    net.eval()
    with torch.no_grad():
        sub = torch.randperm(len(T))[:400]
        R = net(T[sub]).cpu().numpy()
        C = T[sub].cpu().numpy()
    iu = np.triu_indices(len(R), 1)
    rho = float(__import__("scipy.stats", fromlist=["spearmanr"])
                .spearmanr(pdist(R), pdist(C)).statistic)
    print(f"struct-rho ({args.arm}): {rho:+.3f}   "
          f"[pass >= {GATES['rho_pass']}, kill < {GATES['kill']}]")

    torch.save(net.state_dict(),
               os.path.join(resdir, f"model_pairs_{args.arm}.pt"))
    np.save(os.path.join(resdir, f"reps_pairs_{args.arm}.npy"), R)
    with open(os.path.join(resdir, f"history_pairs_{args.arm}.json"), "w") as f:
        json.dump(dict(arm=args.arm, rho=rho, epochs=args.epochs,
                       n_pairs=len(pairs), hist=hist), f, indent=1)
    print("saved. run --report after both arms for the verdict.")


if __name__ == "__main__":
    main()