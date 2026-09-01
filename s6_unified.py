"""
s6_unified.py — S6: end-to-end unification. Can gradient pressure make
language streams color-coherent?

Pre-registered: docs/calc_s6.md (2026-09-01, commit dd697e5, BEFORE this code).

Every prior superposition test used a frozen pipeline: the encoder never
felt the unfold task. S6 trains encoder -> superpose -> reader JOINTLY
under one loss, so unfold error backpropagates into the encoder. If the
S5 mechanism story is right (unfold needs per-stream palette coherence),
the encoder should be FORCED to invent it. No hand-designed coherence
anywhere: the only new pressure is the task loss.

Arms:
  J  joint training, scene k=32 field (the test)
  F  identical schedule, encoder FROZEN at S4 weights (baseline; must
     reproduce S5's 0.198 within noise, or the run is invalid)
  D  joint training on the dense (arm-B) encoder (substrate control)

Gates (docs/calc_s6.md): G0 validity |rho_F - 0.198| <= 0.08 and solo
round-trip(J) >= 0.80; G1 rho_J >= 0.45; G2 rho_J - rho_F >= 0.15;
G3 rho_J - rho_D >= 0.10; G4 M5 color-rho(J) >= S4's 0.54 + 0.10.
KILL = G0 passes and rho_J < 0.35.

USAGE:
  py -3.12 s6_unified.py --smoke
  py -3.12 s6_unified.py
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

import s2_disentangle as s2
import s4_scene_encoder as s4
import s5_superposition as s5

OUT = s2.OUT
SEED = s2.SEED
DEVICE = s2.DEVICE

N_CELLS = 512
SCENE_K = 32
SCENE_FPC = s5.SCENE_FPC          # 10
DENSE_FPC = s5.DENSE_FPC          # 9
SCENE_DIM = N_CELLS * SCENE_FPC   # 5120
DENSE_DIM = N_CELLS * DENSE_FPC   # 4608
EMB_DIM = 64
LAMBDA_RT = 1.0                   # pre-registered round-trip weight

S4_M5_BASELINE = 0.54             # pre-registered (S4 scene-k32 M5)
S5_SCENE_RHO = 0.198              # pre-registered (S5 frozen result)


def log(m):
    print(m, flush=True)


# ---------------------------------------------------------------- models

def make_scene_encoder(warm: bool):
    """Scene k=32 encoder; warm-started from S4 arm D weights."""
    enc = s4.Encoder(k=SCENE_K)
    if warm:
        ckpt = torch.load(os.path.join(OUT, "s4_encoders.pt"), map_location="cpu")
        enc.load_state_dict(ckpt["D"])
    return enc.to(DEVICE)


def make_dense_encoder(warm: bool):
    enc = s4.DenseEncoder()
    if warm:
        ckpt = torch.load(os.path.join(OUT, "s4_encoders.pt"), map_location="cpu")
        enc.load_state_dict(ckpt["B"])
    return enc.to(DEVICE)


# ------------------------------------------------- differentiable superpose
# Reuses s5.superpose verbatim: it is differentiable w.r.t. field CONTENTS
# (masks route values; advanced-indexing assignment carries gradients).
# The occupancy masks themselves are non-diff, matching the S4 STE spirit.

def superpose_batch(fA, fB, fpc, flip):
    """fA/fB: (B, 512, fpc) field tensors WITH grad. flip: (B,) float."""
    return s5.superpose(fA, fB, fpc, flip.view(-1, 1))


# ------------------------------------------------------------------ metrics

def rho_pairs(A, B, pairs):
    ii = np.array([i for i, _ in pairs]); jj = np.array([j for _, j in pairs])
    dA = np.linalg.norm(A[ii] - A[jj], axis=1)
    dB = np.linalg.norm(B[ii] - B[jj], axis=1)
    r, _ = spearmanr(dA, dB)
    return float(r)


@torch.no_grad()
def fields_of(enc, embs, bs=512):
    outs = []
    for i in range(0, embs.shape[0], bs):
        xb = torch.tensor(embs[i:i + bs], dtype=torch.float32, device=DEVICE)
        outs.append(enc(xb))
    return torch.cat(outs)


@torch.no_grad()
def m5_color_rho(enc, hold_embs, pairs, fpc):
    """S4's M5, verbatim: color-only field content vs true embeddings."""
    f = fields_of(enc, hold_embs).view(-1, N_CELLS, fpc).cpu().numpy()
    col = f[:, :, 1:].reshape(f.shape[0], -1)
    return round(rho_pairs(col, hold_embs, pairs), 4)


@torch.no_grad()
def palette_coherence(enc, hold_embs, fpc, max_n=200):
    """Diagnostic (F4): within-sentence vs between-sentence lit-cell color
    similarity, and palette-vs-meaning alignment. Not a gate."""
    f = fields_of(enc, hold_embs[:max_n]).view(-1, N_CELLS, fpc).cpu().numpy()
    lit = f[:, :, 0] > 0.01
    sigs, withins = [], []
    for i in range(f.shape[0]):
        idxs = np.where(lit[i])[0]
        if len(idxs) < 2:
            sigs.append(np.zeros(fpc - 1)); withins.append(0.0); continue
        cols = f[i, idxs, 1:]                                   # (n, fpc-1)
        cn = cols / (np.linalg.norm(cols, axis=1, keepdims=True) + 1e-9)
        sim = cn @ cn.T
        n = len(idxs)
        withins.append(float((sim.sum() - n) / (n * n - n)))
        sigs.append(cols.mean(0))
    sigs = np.stack(sigs)
    sn = sigs / (np.linalg.norm(sigs, axis=1, keepdims=True) + 1e-9)
    cross = sn @ sn.T
    n = sigs.shape[0]
    between = float((cross.sum() - n) / (n * n - n))
    within = float(np.mean(withins))
    # palette-vs-meaning: do similar sentences have similar palettes?
    en = hold_embs[:max_n] / (np.linalg.norm(hold_embs[:max_n], axis=1, keepdims=True) + 1e-9)
    esim = (en @ en.T)[np.triu_indices(n, 1)]
    psim = cross[np.triu_indices(n, 1)]
    r, _ = spearmanr(esim, psim)
    return {"within": round(within, 4), "between": round(between, 4),
            "within_minus_between": round(within - between, 4),
            "palette_vs_meaning_rho": round(float(r), 4)}


# ------------------------------------------------------------- joint training

def train_joint(enc, freeze_enc, fpc, pool, epochs, tag,
                pairs_per_epoch, bs=128, lr=3e-4):
    """One model, one loss: unfold (order-invariant) + lambda*round-trip."""
    torch.manual_seed(SEED)
    in_dim = N_CELLS * fpc
    reader = s5.Reader(in_dim, n_streams=2).to(DEVICE)
    dec = s4.Decoder(in_dim).to(DEVICE)
    params = list(reader.parameters()) + list(dec.parameters())
    if not freeze_enc:
        params += list(enc.parameters())
        enc.train()
    else:
        enc.eval()
        for p in enc.parameters():
            p.requires_grad_(False)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    x = torch.tensor(pool, dtype=torch.float32, device=DEVICE)
    n = x.shape[0]
    g = torch.Generator(device="cpu").manual_seed(SEED)
    for ep in range(1, epochs + 1):
        # F3: RANDOM pairing, RESAMPLED every epoch (no fixed partner)
        perm = torch.randperm(n, generator=g)[: 2 * pairs_per_epoch]
        ia, ib = perm[0::2], perm[1::2]
        flips = torch.randint(0, 2, (ia.shape[0],), generator=g).float()
        tot = nb = 0
        for i in range(0, ia.shape[0], bs):
            ea = x[ia[i:i + bs]]; eb = x[ib[i:i + bs]]
            fA = enc(ea).view(-1, N_CELLS, fpc)
            fB = enc(eb).view(-1, N_CELLS, fpc)
            sp = superpose_batch(fA, fB, fpc, flips[i:i + bs].to(DEVICE))
            out = reader(sp).view(-1, 2, EMB_DIM)
            # order-invariant unfold loss: min over the 2 assignments
            l_ab = F.mse_loss(out[:, 0], ea, reduction="none").mean(1) \
                 + F.mse_loss(out[:, 1], eb, reduction="none").mean(1)
            l_ba = F.mse_loss(out[:, 0], eb, reduction="none").mean(1) \
                 + F.mse_loss(out[:, 1], ea, reduction="none").mean(1)
            l_unfold = torch.minimum(l_ab, l_ba).mean()
            # round-trip on SOLO fields (guards content)
            l_rt = F.mse_loss(dec(fA.view(fA.shape[0], -1)), ea) \
                 + F.mse_loss(dec(fB.view(fB.shape[0], -1)), eb)
            loss = l_unfold + LAMBDA_RT * 0.5 * l_rt
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if ep % 20 == 0 or ep == epochs:
            log(f"    [{tag}] ep {ep}/{epochs} loss={tot/nb:.5f}")
    enc.eval(); reader.eval(); dec.eval()
    return enc, reader, dec


# --------------------------------------------------------------------- eval

@torch.no_grad()
def eval_pairing(enc, reader, fpc, hold, Ypairs, flip, order):
    """Build held-out superposed fields under a given pairing order and run
    S5's content-addressed evaluation (verbatim machinery)."""
    ns = len(order) // 2
    embs = hold[order]                                     # reordered truth
    f = fields_of(enc, embs).view(-1, N_CELLS, fpc)
    Xs, jacs = [], []
    for i in range(ns):
        A = f[2 * i:2 * i + 1]; B = f[2 * i + 1:2 * i + 2]
        sp = s5.superpose(A, B, fpc, torch.tensor([[flip[i]]]))
        Xs.append(sp.reshape(-1).cpu().numpy())
        jacs.append(s5.jaccard_pair(A, B))
    X = np.stack(Xs)
    res = s5.eval_colocated(reader, X, Ypairs, embs)
    res["jaccard"] = round(float(np.mean(jacs)), 4)
    return res


@torch.no_grad()
def solo_roundtrip(enc, dec, hold, pairs):
    f = fields_of(enc, hold)
    decd = torch.cat([dec(f[i:i + 512]) for i in range(0, f.shape[0], 512)]).cpu().numpy()
    return round(rho_pairs(decd, hold, pairs), 4)


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    if a.smoke:
        n_pool, n_hold, epochs, ppe = 2000, 100, 30, 800
    else:
        n_pool, n_hold, epochs, ppe = 10000, 400, 200, 4096

    log("=== S6: end-to-end unification (joint encoder+reader) ===")
    log(f"mode={'smoke' if a.smoke else 'full'} seed={SEED} device={DEVICE}")

    rng = np.random.default_rng(SEED)
    uniq = s2.load_unique_sentences()
    perm = rng.permutation(len(uniq))
    hold_sents = [uniq[i] for i in perm[:n_hold]]
    pool_sents = [uniq[i] for i in perm[n_hold:n_hold + n_pool]]
    log(f"sentences: pool={len(pool_sents)} heldout={len(hold_sents)}")

    pool = s2.matryoshka(s2.embed_sentences(pool_sents)).astype(np.float32)
    hold = s2.matryoshka(s2.embed_sentences(hold_sents)).astype(np.float32)

    # held-out pair set over 2*ns recovered rows (S5 convention)
    ns = n_hold // 2
    NROW = 2 * ns
    rng_p = np.random.default_rng(SEED)
    pset = set()
    target = min(2000, NROW * (NROW - 1) // 2)
    while len(pset) < target:
        i, j = int(rng_p.integers(0, NROW)), int(rng_p.integers(0, NROW))
        if i != j:
            pset.add((min(i, j), max(i, j)))
    Ypairs = sorted(pset)
    # separate pair set over raw held-out indices for M5 / round-trip metrics
    rng_q = np.random.default_rng(SEED + 7)
    qset = set()
    while len(qset) < min(2000, n_hold * (n_hold - 1) // 2):
        i, j = int(rng_q.integers(0, n_hold)), int(rng_q.integers(0, n_hold))
        if i != j:
            qset.add((min(i, j), max(i, j)))
    Qpairs = sorted(qset)

    flips = np.random.default_rng(SEED).integers(0, 2, size=ns).astype(np.float32)
    # F3: three distinct held-out pairing orders (identity + 2 reshuffles)
    orders = [np.arange(NROW)]
    rng_o = np.random.default_rng(SEED + 13)
    for _ in range(2):
        orders.append(rng_o.permutation(NROW))

    results = {"mode": "smoke" if a.smoke else "full",
               "lambda_rt": LAMBDA_RT, "epochs": epochs}

    arms = {}
    for tag, mk, freeze, fpc in [
            ("J", lambda: make_scene_encoder(warm=True), False, SCENE_FPC),
            ("F", lambda: make_scene_encoder(warm=True), True, SCENE_FPC),
            ("D", lambda: make_dense_encoder(warm=True), False, DENSE_FPC)]:
        log(f"\n=== arm {tag} ({'frozen' if freeze else 'joint'}, fpc={fpc}) ===")
        enc = mk()
        enc, reader, dec = train_joint(enc, freeze, fpc, pool, epochs, tag, ppe)
        pruns = [eval_pairing(enc, reader, fpc, hold, Ypairs, flips, o)
                 for o in orders]
        rhos = [p["rho_coloc"] for p in pruns]
        arm = {"pairings": pruns,
               "rho_mean": round(float(np.mean(rhos)), 4),
               "rho_std": round(float(np.std(rhos)), 4),
               "recover_acc": round(float(np.mean([p["recover_acc"] for p in pruns])), 4),
               "solo_roundtrip_rho": solo_roundtrip(enc, dec, hold, Qpairs),
               "M5_color_rho": m5_color_rho(enc, hold, Qpairs, fpc),
               "coherence": palette_coherence(enc, hold, fpc)}
        arms[tag] = arm
        results[tag] = arm
        log(f"  arm {tag}: rho={arm['rho_mean']:.4f}±{arm['rho_std']:.4f} "
            f"acc={arm['recover_acc']:.3f} solo_rt={arm['solo_roundtrip_rho']:.4f} "
            f"M5={arm['M5_color_rho']:.4f} coh={arm['coherence']['within_minus_between']:.4f}")
        if tag == "J":
            torch.save(enc.state_dict(), os.path.join(OUT, "s6_encoder.pt"))

    rJ, rF, rD = arms["J"]["rho_mean"], arms["F"]["rho_mean"], arms["D"]["rho_mean"]
    G0 = abs(rF - S5_SCENE_RHO) <= 0.08 and arms["J"]["solo_roundtrip_rho"] >= 0.80
    G1 = rJ >= 0.45
    G2 = rJ - rF >= 0.15
    G3 = rJ - rD >= 0.10
    G4 = arms["J"]["M5_color_rho"] >= S4_M5_BASELINE + 0.10
    results["gates"] = {"G0_validity": bool(G0), "G1_unfold_lifts": bool(G1),
                        "G2_gradient_did_it": bool(G2),
                        "G3_substrate_matters": bool(G3),
                        "G4_coherence_rose": bool(G4)}
    if G0 and G1 and G2 and G4:
        verdict = "PASS" + ("" if G3 else " (substrate-agnostic)")
    elif G0 and rJ < 0.35:
        verdict = "KILL"
    elif G0 and G1 and G2 and not G4:
        verdict = "SUSPECT-FINGERPRINT"
    elif not G0:
        verdict = "INVALID"
    else:
        verdict = "GRAY"
    results["verdict"] = verdict

    log("\n" + "=" * 64)
    log("S6 verdict")
    log("=" * 64)
    log(f"  rho J (joint scene)   : {rJ:.4f} ± {arms['J']['rho_std']:.4f}")
    log(f"  rho F (frozen scene)  : {rF:.4f}  (S5 was {S5_SCENE_RHO})")
    log(f"  rho D (joint dense)   : {rD:.4f}")
    log(f"  J solo round-trip     : {arms['J']['solo_roundtrip_rho']:.4f} (F2 >= 0.80)")
    log(f"  J M5 color rho        : {arms['J']['M5_color_rho']:.4f} (S4 base {S4_M5_BASELINE})")
    log(f"  J coherence w-b       : {arms['J']['coherence']['within_minus_between']:.4f} "
        f"(F base {arms['F']['coherence']['within_minus_between']:.4f})")
    log(f"  J palette~meaning rho : {arms['J']['coherence']['palette_vs_meaning_rho']:.4f}")
    for g, v in results["gates"].items():
        log(f"  {g}: {'PASS' if v else 'FAIL'}")
    log(f"VERDICT: {verdict}")

    fn = "s6_smoke.json" if a.smoke else "s6_results.json"
    json.dump(results, open(os.path.join(OUT, fn), "w"), indent=2)
    log(f"saved results/{fn}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
