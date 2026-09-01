"""
s4_scene_encoder.py — S4: text -> sparse chromavox scene -> meaning back out

Pre-registered: docs/calc_s4.md (2026-09-01, BEFORE this run).

THE TEST: can a learned encoder map sentences into sparse SCENES (few
cells lit, K=3 frequency-color slots per cell, no density) such that
(a) meaning survives the round-trip and (b) it actually IS a scene?

ARMS:
  A scene     top-k=16, K=3 freq slots, no density      (the test)
  B dense     V1 arm-B replica (no top-k, cos/sin)      (smear control)
  C scene-k8  top-k=8
  D scene-k32 top-k=32
  E identity  ceiling

GATES (pre-registered):
  S4-G1  meaning survives: arm A round-trip rho >= 0.85
  S4-G2  it IS a scene:    occupancy in [8,32] AND pairwise Jaccard <= 0.30
  S4-G3  position alive:   position-only rho >= 0.10
  KILL = G1 fails with rho < 0.60. GRAY otherwise.

USAGE:
  py -3.12 s4_scene_encoder.py --smoke   # ~5 min
  py -3.12 s4_scene_encoder.py           # ~30-45 min iGPU
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

import s2_disentangle as s2  # reuse data/cache/embedding helpers

OUT = s2.OUT
SEED = s2.SEED
DEVICE = s2.DEVICE

GRID = 8
N_CELLS = GRID ** 3              # 512
K_SLOTS = 3                      # user-specified K=3
FEATS_PER_CELL = 1 + K_SLOTS * 3 # salience + K*(freq, brightness, amplitude)
TOTAL_DIMS = N_CELLS * FEATS_PER_CELL
EMB_DIM = 64


def log(m):
    print(m, flush=True)


# --- field constraint: top-k salience + frequency slots, no density ---

class SceneField(nn.Module):
    """Constrained field: salience gate (top-k), K=3 frequency slots.

    Per cell the raw output is: [salience, (freq, bri, amp) x K].
    - salience: sigmoid
    - freq: normalized 380-700nm value via sigmoid (ordered, NOT periodic)
    - brightness, amplitude: sigmoid
    - top-k: only the k highest-salience cells keep their content; the
      rest are zeroed entirely (the scene budget). Straight-through.
    """

    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, x):
        B = x.shape[0]
        f = x.reshape(B, N_CELLS, FEATS_PER_CELL)
        out = torch.zeros_like(f)
        out[:, :, 0] = torch.sigmoid(f[:, :, 0])          # salience
        for k in range(K_SLOTS):
            b = 1 + k * 3
            out[:, :, b] = torch.sigmoid(f[:, :, b])       # freq (ordered 0-1)
            out[:, :, b + 1] = torch.sigmoid(f[:, :, b + 1])  # brightness
            out[:, :, b + 2] = torch.sigmoid(f[:, :, b + 2])  # amplitude

        if self.k is not None:
            # top-k scene budget: keep only k cells alive per sentence.
            # Hard mask in forward (exactly k cells lit); STE gives the
            # backward pass the soft salience-scaled gradient so cells
            # can still compete for the top-k set.
            sal = out[:, :, 0]
            topk = torch.topk(sal, self.k, dim=1).indices   # (B, k)
            mask = torch.zeros_like(sal)
            mask.scatter_(1, topk, 1.0)
            hard = out * mask.unsqueeze(-1)
            soft = out * sal.unsqueeze(-1)
            out = soft + (hard - soft).detach()  # forward=hard, backward=soft
        return out.reshape(B, -1)


class Encoder(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(EMB_DIM, 256), nn.GELU(),
                                nn.Linear(256, TOTAL_DIMS))
        self.field = SceneField(k)

    def forward(self, x):
        return self.field(self.mlp(x))


class DenseField(nn.Module):
    """V1 arm-B replica: cos/sin slots, no top-k (the smear control)."""

    def __init__(self):
        super().__init__()
        self.k_slots = 2

    def forward(self, x):
        B = x.shape[0]
        f = x.reshape(B, N_CELLS, 1 + self.k_slots * 4)
        out = torch.zeros_like(f)
        out[:, :, 0] = torch.sigmoid(f[:, :, 0])
        for k in range(self.k_slots):
            b = 1 + k * 4
            ang = f[:, :, b]
            out[:, :, b] = torch.cos(ang)
            out[:, :, b + 1] = torch.sin(ang)
            out[:, :, b + 2] = torch.sigmoid(f[:, :, b + 2])
            out[:, :, b + 3] = torch.sigmoid(f[:, :, b + 3])
        return out.reshape(B, -1)


class DenseEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(EMB_DIM, 256), nn.GELU(),
                                 nn.Linear(256, N_CELLS * (1 + 2 * 4)))
        self.field = DenseField()

    def forward(self, x):
        return self.field(self.mlp(x))


class Decoder(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, 256), nn.GELU(),
                                 nn.Linear(256, EMB_DIM))

    def forward(self, x):
        return self.mlp(x)


def train_arm(encoder, emb_train, epochs, lr=3e-4, wd=1e-4, bs=256):
    torch.manual_seed(SEED)
    enc = encoder.to(DEVICE)
    dec = Decoder(enc.mlp[-1].out_features).to(DEVICE)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()),
                            lr=lr, weight_decay=wd)
    x = torch.tensor(emb_train, dtype=torch.float32, device=DEVICE)
    n = x.shape[0]
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n, device=DEVICE)
        tot = nb = 0
        for i in range(0, n, bs):
            b = x[perm[i:i + bs]]
            loss = F.mse_loss(dec(enc(b)), b)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if ep % 25 == 0 or ep == epochs:
            log(f"    ep {ep}/{epochs} recon={tot/nb:.4f}")
    enc.eval(); dec.eval()
    return enc, dec


@torch.no_grad()
def encode_all(enc, embs, bs=512):
    outs = []
    for i in range(0, embs.shape[0], bs):
        xb = torch.tensor(embs[i:i + bs], dtype=torch.float32, device=DEVICE)
        outs.append(enc(xb))
    return torch.cat(outs)


def rho_pairs(A, B, pairs):
    ii = np.array([i for i, _ in pairs]); jj = np.array([j for _, j in pairs])
    dA = np.linalg.norm(A[ii] - A[jj], axis=1)
    dB = np.linalg.norm(B[ii] - B[jj], axis=1)
    r, _ = spearmanr(dA, dB)
    return float(r)


@torch.no_grad()
def eval_arm(name, enc, dec, hold_embs, pairs, feats_per_cell):
    fields = encode_all(enc, hold_embs)             # (400, D) on device
    f = fields.view(fields.shape[0], N_CELLS, feats_per_cell).cpu().numpy()
    decoded = torch.cat([dec(fields[i:i + 512]) for i in range(0, fields.shape[0], 512)])
    decoded = decoded.cpu().numpy()

    m1 = rho_pairs(decoded, hold_embs, pairs)        # round-trip structure

    lit = f[:, :, 0] > 0.01
    m2 = float(lit.sum(1).mean())                    # occupancy

    jac = []
    n_hold = f.shape[0]
    for i in range(0, n_hold - 1, 2):
        a, b = lit[i], lit[i + 1]
        jac.append((a & b).sum() / max(1, (a | b).sum()))
    m3 = float(np.mean(jac))                         # separation

    # position-only: cell coordinates of lit cells (centroid), no colors
    coords = np.zeros((f.shape[0], 3), dtype=np.float32)
    for i in range(f.shape[0]):
        idxs = np.where(lit[i])[0]
        if len(idxs):
            xs = idxs % GRID; ys = (idxs // GRID) % GRID; zs = idxs // (GRID * GRID)
            coords[i] = [xs.mean(), ys.mean(), zs.mean()]
    m4 = rho_pairs(coords, hold_embs, pairs)         # position alive

    # color-only: slot contents (freq/bri/amp), ignore salience/coords
    col = f[:, :, 1:].reshape(f.shape[0], -1)
    m5 = rho_pairs(col, hold_embs, pairs)
    # frequency-bin usage across 380-700nm (slot 0 freqs of lit cells)
    freqs = []
    for i in range(f.shape[0]):
        idxs = np.where(lit[i])[0]
        if len(idxs):
            freqs.extend(f[i, idxs, 1].tolist())
    hist, _ = np.histogram(np.array(freqs) if freqs else np.zeros(1), bins=10)
    m5_bins = int((hist > 0).sum())

    return {"name": name, "M1_roundtrip_rho": round(m1, 4),
            "M2_occupancy": round(m2, 1), "M3_jaccard": round(m3, 4),
            "M4_position_rho": round(m4, 4),
            "M5_color_rho": round(m5, 4), "M5_freq_bins_used": m5_bins}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    if a.smoke:
        n_pool, n_hold, epochs = 2000, 100, 60
    else:
        n_pool, n_hold, epochs = 10000, 400, 200

    log("=== S4: scene encoder — text -> sparse scene -> meaning back out ===")
    log(f"mode={'smoke' if a.smoke else 'full'} seed={SEED}")

    uniq = s2.load_unique_sentences()
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(uniq))
    hold_sents = [uniq[i] for i in perm[:n_hold]]
    pool_sents = [uniq[i] for i in perm[n_hold:n_hold + n_pool]]
    log(f"sentences: pool={len(pool_sents)} heldout={len(hold_sents)}")

    pool = s2.matryoshka(s2.embed_sentences(pool_sents))
    hold = s2.matryoshka(s2.embed_sentences(hold_sents))

    # held-out pair set (seed fixed)
    rng_p = np.random.default_rng(SEED)
    pset = set()
    while len(pset) < 2000:
        i, j = rng_p.integers(0, n_hold, 2)
        if i != j:
            pset.add((int(min(i, j)), int(max(i, j))))
    pairs = sorted(pset)

    results = {"mode": "smoke" if a.smoke else "full", "n_hold": n_hold}

    log("=== arm A: scene (top-k=16, K=3 freq slots, no density) ===")
    encA, decA = train_arm(Encoder(k=16), pool, epochs)
    results["A"] = eval_arm("scene-k16", encA, decA, hold, pairs, FEATS_PER_CELL)

    log("=== arm B: dense control (V1 arm-B replica) ===")
    encB, decB = train_arm(DenseEncoder(), pool, epochs)
    results["B"] = eval_arm("dense", encB, decB, hold, pairs, 1 + 2 * 4)

    log("=== arm C: scene-k8 ===")
    encC, decC = train_arm(Encoder(k=8), pool, epochs)
    results["C"] = eval_arm("scene-k8", encC, decC, hold, pairs, FEATS_PER_CELL)

    log("=== arm D: scene-k32 ===")
    encD, decD = train_arm(Encoder(k=32), pool, epochs)
    results["D"] = eval_arm("scene-k32", encD, decD, hold, pairs, FEATS_PER_CELL)

    results["E"] = {"name": "identity", "M1_roundtrip_rho": 1.0}

    # gates on arm A
    A = results["A"]
    g1 = A["M1_roundtrip_rho"] >= 0.85
    g2 = (8 <= A["M2_occupancy"] <= 32) and (A["M3_jaccard"] <= 0.30)
    g3 = A["M4_position_rho"] >= 0.10
    if not g1 and A["M1_roundtrip_rho"] < 0.60:
        verdict = "KILL"
    elif g1 and g2 and g3:
        verdict = "PASS"
    else:
        verdict = "GRAY"
    results["gates"] = {"G1_meaning_survives": g1, "G2_is_a_scene": g2,
                        "G3_position_alive": g3}
    results["verdict"] = verdict

    log("\n" + "=" * 64)
    log("S4 verdict")
    log("=" * 64)
    for arm in "ABCDE":
        if arm in results:
            r = results[arm]
            log(f"  {arm} ({r['name']:<10}) M1={r['M1_roundtrip_rho']:+.4f}  "
                f"M2={r.get('M2_occupancy', '-')}  M3={r.get('M3_jaccard', '-')}  "
                f"M4={r.get('M4_position_rho', '-')}  M5={r.get('M5_color_rho', '-')}  "
                f"bins={r.get('M5_freq_bins_used', '-')}")
    for g, v in results["gates"].items():
        log(f"  {g}: {'PASS' if v else 'FAIL'}")
    log(f"VERDICT: {verdict}")

    torch.save({"A": encA.state_dict(), "B": encB.state_dict(),
                "C": encC.state_dict(), "D": encD.state_dict()},
               os.path.join(OUT, "s4_encoders.pt"))
    path = os.path.join(OUT, "s4_results.json")
    json.dump(results, open(path, "w"), indent=2)
    log(f"saved {path}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()