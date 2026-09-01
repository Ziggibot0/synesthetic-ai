"""
s5_superposition.py — S5: can a reader unfold two overlapping SCENES?

Pre-registered: docs/calc_s5.md (2026-09-01, BEFORE this run).

Reuses S4's FROZEN encoders (no retrain). Tests the reframe "crowding
allows for superposition": two co-located sentences, each a sparse
position-alive scene (top-k=32, K=3 freq slots, no density), share ~44%
of their lit cells. A query-free reader must recover BOTH streams,
using each scene's private cells to identify it and the un-blended
color-set slots to handle the shared (crowded) cells.

Smear control = S4 arm B dense field, run through the identical reader
setup (within-experiment comparison, not a historical quote).

Gates (docs/calc_s5.md): G1 unfold rho>=0.60; G2 scenes beat smears by
>=0.10; G3 mean Jaccard in [0.15,0.70]. KILL < 0.50.

USAGE:
  py -3.12 s5_superposition.py --smoke
  py -3.12 s5_superposition.py
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

import s2_disentangle as s2
import s4_scene_encoder as s4  # frozen encoders + scene field

OUT = s2.OUT
SEED = s2.SEED
DEVICE = s2.DEVICE

N_CELLS = 512
GRID = 8

# S4 frozen encoder field shapes
SCENE_K = 32
SCENE_FPC = 1 + 3 * 3      # salience + K=3*(freq,bri,amp) = 10
SCENE_DIM = N_CELLS * SCENE_FPC            # 5120
DENSE_FPC = 1 + 2 * 4      # 9 (V1 arm-B replica)
DENSE_DIM = N_CELLS * DENSE_FPC            # 4608
EMB_DIM = 64

def log(m):
    print(m, flush=True)


# --- load frozen S4 encoders ---

def load_frozen():
    ckpt = torch.load(os.path.join(OUT, "s4_encoders.pt"), map_location="cpu")
    enc_scene = s4.Encoder(k=SCENE_K)
    enc_scene.load_state_dict(ckpt["D"])           # scene-k32
    enc_scene.to(DEVICE).eval()
    enc_dense = s4.DenseEncoder()
    enc_dense.load_state_dict(ckpt["B"])           # dense smeear control
    enc_dense.to(DEVICE).eval()
    return enc_scene, enc_dense


@torch.no_grad()
def encode(enc, emb):
    x = torch.tensor(emb, dtype=torch.float32, device=DEVICE)
    return enc(x).reshape(emb.shape[0], N_CELLS, -1)  # (N, 512, fpc)


def superpose(fieldA, fieldB, fpc, flip):
    """Salience-union superposition, un-blended color-set slots.

    A-only cell: slot-0 holds A's slot-0 color; slot-1 empty.
    B-only cell: slot-0 holds B's slot-0 color; slot-1 empty.
    SHARED cell: slot-0 holds one stream's color, slot-1 the other's,
                 never blended (the red-AND-blue case).
    Which stream sits in slot-0 of a SHARED cell is randomized per field
    (the `flip` mask), killing the slot-index/disjoint-blocks shortcut.
    Private cells are never flipped (their content is unambiguous).

    Returns flattened field of width fpc*512.
    """
    B = fieldA.shape[0]
    sw = 3 if fpc == SCENE_FPC else 4   # slot-0 width
    salA = fieldA[:, :, 0]; salB = fieldB[:, :, 0]
    inA = salA > 0.01; inB = salB > 0.01
    shared = inA & inB
    aonly = inA & ~inB
    bonly = ~inA & inB

    out = torch.zeros_like(fieldA)
    out[:, :, 0] = torch.max(salA, salB)            # occupancy union

    out[aonly, 1:1 + sw] = fieldA[aonly, 1:1 + sw]      # A-private -> slot-0
    out[bonly, 1:1 + sw] = fieldB[bonly, 1:1 + sw]      # B-private -> slot-0

    out[shared, 1:1 + sw] = fieldA[shared, 1:1 + sw]    # shared slot-0 = A
    out[shared, 1 + sw:1 + 2 * sw] = fieldB[shared, 1:1 + sw]  # shared slot-1 = B

    # randomize which stream is slot-0, ONLY on shared cells
    fsh = flip.to(fieldA.device).bool().view(B, 1)          # (B,1)
    swap_mask = shared & fsh.expand(B, N_CELLS)             # (B,512) bool
    s0 = out[:, :, 1:1 + sw].clone()
    s1 = out[:, :, 1 + sw:1 + 2 * sw].clone()
    new_s0 = torch.where(swap_mask.unsqueeze(-1), s1, s0)
    new_s1 = torch.where(swap_mask.unsqueeze(-1), s0, s1)
    out[:, :, 1:1 + sw] = new_s0
    out[:, :, 1 + sw:1 + 2 * sw] = new_s1
    return out.reshape(B, -1)


def jaccard_pair(fA, fB):
    a = fA[:, :, 0] > 0.01
    b = fB[:, :, 0] > 0.01
    return float((a & b).sum(1).float().mean() / a.sum(1).float().mean())


class Reader(nn.Module):
    def __init__(self, in_dim, n_streams=2):
        super().__init__()
        self.n_streams = n_streams
        self.mlp = nn.Sequential(nn.Linear(in_dim, 1024), nn.GELU(),
                                 nn.Linear(1024, EMB_DIM * n_streams))

    def forward(self, x):
        return self.mlp(x)


def rho_pairs(A, B, pairs):
    ii = np.array([i for i, _ in pairs]); jj = np.array([j for _, j in pairs])
    dA = np.linalg.norm(A[ii] - A[jj], axis=1)
    dB = np.linalg.norm(B[ii] - B[jj], axis=1)
    r, _ = spearmanr(dA, dB)
    return float(r)


def content_addressed_recover(records, hold_embs):
    """Map each stream record to a true sentence by best cosine match.

    records: (ns, 2, 64) per-field streams; hold_embs: (2*ns, 64).
    Returns a recovered full matrix (2*ns, 64) aligned by field index and
    slot, so pairwise structure is comparable to the true embeddings.
    """
    norm = hold_embs / (np.linalg.norm(hold_embs, axis=1, keepdims=True) + 1e-9)
    rec_full = np.empty((hold_embs.shape[0], hold_embs.shape[1]), dtype=np.float32)
    ns = records.shape[0]
    for i in range(ns):
        for slot in range(2):
            v = records[i, slot].reshape(1, -1)
            vn = v / (np.linalg.norm(v) + 1e-9)
            idx = int(np.argmax(vn @ norm.T))
            rec_full[2 * i + slot] = hold_embs[idx]
    return rec_full


def eval_colocated(reader, X, Ypairs, hold_embs):
    """X: (ns, in_dim) colocated fields -> recovered structure rho."""
    with torch.no_grad():
        out = reader(torch.tensor(X, dtype=torch.float32, device=DEVICE))
        out = out.reshape(X.shape[0], 2, EMB_DIM).cpu().numpy()   # (ns,2,64)
    rec_full = content_addressed_recover(out, hold_embs)          # (2*ns,64)
    r = rho_pairs(rec_full, hold_embs, Ypairs)
    # also report per-slot recovery accuracy (did the right sentence win?)
    norm = hold_embs / (np.linalg.norm(hold_embs, axis=1, keepdims=True) + 1e-9)
    acc = 0.0
    ns = X.shape[0]
    for i in range(ns):
        for slot in range(2):
            vn = out[i, slot] / (np.linalg.norm(out[i, slot]) + 1e-9)
            idx = int(np.argmax(vn @ norm.T))
            if idx == 2 * i + slot:
                acc += 1
    acc /= (2 * ns)
    return {"rho_coloc": round(r, 4), "recover_acc": round(acc, 4)}


def train_reader(X, target, epochs, bs=256, lr=3e-4, n_streams=2):
    torch.manual_seed(SEED)
    n = X.shape[0]
    in_dim = X.shape[1]
    r = Reader(in_dim, n_streams).to(DEVICE)
    opt = torch.optim.AdamW(r.parameters(), lr=lr)
    x = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    t = torch.tensor(target, dtype=torch.float32, device=DEVICE)
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n, device=DEVICE)
        tot = nb = 0
        for i in range(0, n, bs):
            b = x[perm[i:i + bs]]; tb = t[perm[i:i + bs]]
            loss = F.mse_loss(r(b), tb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if ep % 100 == 0 or ep == epochs:
            log(f"    reader ({in_dim}->{EMB_DIM*n_streams}) ep {ep}/{epochs} loss={tot/nb:.5f}")
    r.eval()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    if a.smoke:
        n_pool, n_hold, epochs = 2000, 100, 120
    else:
        n_pool, n_hold, epochs = 10000, 400, 300

    log("=== S5: superposition on scenes — can a reader unfold two? ===")
    log(f"mode={'smoke' if a.smoke else 'full'} seed={SEED}")

    enc_scene, enc_dense = load_frozen()
    log("frozen encoders loaded (scene-k32, dense)")

    rng = np.random.default_rng(SEED)
    uniq = s2.load_unique_sentences()
    perm = rng.permutation(len(uniq))
    hold_sents = [uniq[i] for i in perm[:n_hold]]
    pool_sents = [uniq[i] for i in perm[n_hold:n_hold + n_pool]]
    log(f"sentences: pool={len(pool_sents)} heldout={len(hold_sents)}")

    pool = s2.matryoshka(s2.embed_sentences(pool_sents)).astype(np.float32)
    hold = s2.matryoshka(s2.embed_sentences(hold_sents)).astype(np.float32)

    rng_p = np.random.default_rng(SEED)
    pset = set()
    ns = n_hold // 2
    NROW = 2 * ns   # superposed fields recover 2*ns rows (2 per field)
    maxpairs = NROW * (NROW - 1) // 2
    target = min(2000, maxpairs)   # smoke ns=50 -> NROW=100 -> 4950, fine
    while len(pset) < target:
        i, j = int(rng_p.integers(0, NROW)), int(rng_p.integers(0, NROW))
        if i != j:
            pset.add((min(i, j), max(i, j)))
    Ypairs = sorted(pset)

    results = {"mode": "smoke" if a.smoke else "full"}
    mode_rand = np.random.default_rng(SEED).integers(0, 2, size=ns).astype(np.float32)

    # --- SCENE arm ---
    log("\n=== scene arm (k=32): superpose held-out pairs ===")
    # READERS TRAIN ON POOL FIELDS ONLY (never on the held-out fields they
    # are evaluated on — S2's discipline; otherwise the reader memorizes
    # field->embedding and saturates at rho 1.0, as the smoke run showed).
    n_train_fields = min(n_pool // 2, 1000 if a.smoke else 4096)
    pool_scene = encode(enc_scene, pool[:2 * n_train_fields])   # (2f,512,10)
    tr_rand = np.random.default_rng(SEED + 1).integers(0, 2, size=n_train_fields).astype(np.float32)
    tr_fields = []; tr_targets = []
    for i in range(n_train_fields):
        A = pool_scene[2 * i]; B = pool_scene[2 * i + 1]
        sp = superpose(A.unsqueeze(0), B.unsqueeze(0), SCENE_FPC,
                       torch.tensor([[tr_rand[i]]]))
        tr_fields.append(sp.reshape(-1).cpu().numpy())
        tr_targets.append(np.concatenate([pool[2 * i], pool[2 * i + 1]]))
    Xt = np.stack(tr_fields); Tt = np.stack(tr_targets)
    log(f"  train fields (pool): {Xt.shape[0]}")

    # eval fields: superposed HELD-OUT pairs (never trained on)
    hold_scene = encode(enc_scene, hold[:2 * ns])   # (2ns,512,10)
    fields = []; jacs = []
    for i in range(ns):
        A = hold_scene[2 * i]; B = hold_scene[2 * i + 1]
        sp = superpose(A.unsqueeze(0), B.unsqueeze(0), SCENE_FPC,
                       torch.tensor([[mode_rand[i]]]))
        fields.append(sp.reshape(-1).cpu().numpy())
        jacs.append(jaccard_pair(A.unsqueeze(0), B.unsqueeze(0)))
    X = np.stack(fields)
    m3 = float(np.mean(jacs))
    results["scene_Jaccard"] = round(m3, 4)
    log(f"  mean paired-scene Jaccard: {m3:.3f}")

    # solo reader (ceiling): train on pool solo scenes, eval on held-out
    log("  reader: solo scene (ceiling)")
    solo_reader = train_reader(pool_scene.reshape(2 * n_train_fields, SCENE_DIM),
                               pool[:2 * n_train_fields], epochs, n_streams=1)
    hold_scene_flat = hold_scene.reshape(2 * ns, SCENE_DIM).cpu().numpy()
    with torch.no_grad():
        outS = solo_reader(torch.tensor(hold_scene_flat, dtype=torch.float32, device=DEVICE))
        outS = outS.reshape(2 * ns, EMB_DIM).cpu().numpy()   # one stream per field
    # content-address each solo field to its sentence; rho against truth
    norm = hold[:2 * ns] / (np.linalg.norm(hold[:2 * ns], axis=1, keepdims=True) + 1e-9)
    recS = np.empty_like(outS)
    accS = 0.0
    for i in range(2 * ns):
        vn = outS[i] / (np.linalg.norm(outS[i]) + 1e-9)
        idx = int(np.argmax(vn @ norm.T))
        recS[i] = hold[:2 * ns][idx]
        accS += (idx == i)
    accS /= (2 * ns)
    solo_rho = float(rho_pairs(recS, hold[:2 * ns], Ypairs))
    results["scene_solo"] = {"rho": round(solo_rho, 4), "recall_acc": round(accS, 4)}
    log(f"  scene solo rho: {solo_rho:.4f} (recall_acc {accS:.3f})  (validity >=0.50)")

    log("  reader: colocated scene (train pool, eval held-out)")
    col_reader = train_reader(Xt, Tt, epochs)
    cold_scene = eval_colocated(col_reader, X, Ypairs, hold[:2 * ns])
    results["scene_colocated"] = cold_scene
    log(f"  scene colocated rho: {cold_scene['rho_coloc']:.4f} (recover_acc {cold_scene['recover_acc']:.3f})")

    # --- DENSE arm (smear control, identical setup) ---
    log("\n=== dense arm (control): superpose held-out pairs ===")
    pool_dense = encode(enc_dense, pool[:2 * n_train_fields])   # (2f,512,9)
    dtf = []; dtt = []
    for i in range(n_train_fields):
        A = pool_dense[2 * i]; B = pool_dense[2 * i + 1]
        sp = superpose(A.unsqueeze(0), B.unsqueeze(0), DENSE_FPC,
                       torch.tensor([[tr_rand[i]]]))
        dtf.append(sp.reshape(-1).cpu().numpy())
        dtt.append(np.concatenate([pool[2 * i], pool[2 * i + 1]]))
    Xdt = np.stack(dtf); Tdt = np.stack(dtt)

    hold_dense = encode(enc_dense, hold[:2 * ns])   # (2ns,512,9)
    df = []
    for i in range(ns):
        A = hold_dense[2 * i]; B = hold_dense[2 * i + 1]
        sp = superpose(A.unsqueeze(0), B.unsqueeze(0), DENSE_FPC,
                       torch.tensor([[mode_rand[i]]]))
        df.append(sp.reshape(-1).cpu().numpy())
    Xd = np.stack(df)
    log("  reader: colocated dense (train pool, eval held-out)")
    cd_reader = train_reader(Xdt, Tdt, epochs)
    cold_dense = eval_colocated(cd_reader, Xd, Ypairs, hold[:2 * ns])
    results["dense_colocated"] = cold_dense
    log(f"  dense colocated rho: {cold_dense['rho_coloc']:.4f} (recover_acc {cold_dense['recover_acc']:.3f})")

    # gates
    g_col = cold_scene["rho_coloc"]
    G0 = solo_rho >= 0.50
    G1 = g_col >= 0.60
    G2 = g_col - cold_dense["rho_coloc"] >= 0.10
    G3 = 0.15 <= m3 <= 0.70
    results["gates"] = {"G0_validity_solo": bool(G0), "G1_unfold": bool(G1),
                        "G2_scenes_beat_smears": bool(G2), "G3_crowd": bool(G3)}
    verdict = "PASS" if (G1 and G2 and G3) else ("KILL" if g_col < 0.50 else "GRAY")
    results["verdict"] = verdict

    log("\n" + "=" * 64)
    log("S5 verdict")
    log("=" * 64)
    log(f"  scene Jaccard            : {m3:.3f}")
    log(f"  scene solo rho (validity): {solo_rho:.4f}")
    log(f"  SCENE colocated rho      : {cold_scene['rho_coloc']:.4f}")
    log(f"  DENSE colocated rho      : {cold_dense['rho_coloc']:.4f}")
    log(f"  scenes - smears          : {cold_scene['rho_coloc']-cold_dense['rho_coloc']:+.4f}")
    for g, v in results["gates"].items():
        log(f"  {g}: {'PASS' if v else 'FAIL'}")
    log(f"VERDICT: {verdict}")

    json.dump(results, open(os.path.join(OUT, "s5_results.json"), "w"), indent=2)
    log(f"saved results/s5_results.json  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()