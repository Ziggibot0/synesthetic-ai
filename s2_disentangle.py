"""
s2_disentangle.py — S2: semantic superposition disentanglement (pre-registered)

Design: docs/calc_s2.md (PRE-REGISTERED 2026-08-31, before this run).

QUESTION: can a learned reader recover BOTH semantic streams from a chromavox
field in which two unrelated real sentences share the same cells — and what
does co-location cost vs reading a solo field?

Architecture per arm:
  encoder: V1 arm-B recipe (64 -> 256 -> 4,608 constrained field + decoder),
           retrained here because V1 saved no weights.
  reader:  4,608 -> 1,024 (GELU) -> 5,120 (two stream records of 2,560).
           Query-free (no sentence input): it can ONLY act on the field.

Superposition construction (per spec v2 semantics):
  density = min(d1 + d2, 1)     mass adds
  slot0 = S1 slot-0 attrs, slot1 = S2 slot-0 attrs (never blended)
  which sentence takes slot0 is RANDOMIZED per field (kills the slot-index
  shortcut — the disjoint-blocks confound caught 2026-08-31).

Arms:
  colocated : reader(superposed field) -> both records      (the test)
  solo      : reader(padded solo record) -> same record     (ceiling)

Gates (pre-registered, docs/calc_s2.md):
  S2-G1 capacity:     rho_colocated >= 0.85
  S2-G2 interference: rho_solo - rho_colocated <= 0.05
  S2-G3 symmetry:     |delta_S1 - delta_S2| <= 0.05
  PASS = G1 and G2 and G3; KILL = rho_colocated < 0.60; GRAY otherwise.
  Encoder validity precondition: solo structure rho >= 0.90 or run invalid.

USAGE:
  py -3.12 s2_disentangle.py --smoke   # end-to-end validation, ~3-5 min
  py -3.12 s2_disentangle.py           # full run, ~35-45 min CPU, seed 1337
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
SEED = 1337

GRID = 8
N_CELLS = GRID ** 3              # 512
K_SLOTS = 2
FEATS_PER_CELL = 1 + K_SLOTS * 4  # 9
TOTAL_DIMS = N_CELLS * FEATS_PER_CELL  # 4,608
REC_DIM = N_CELLS * (1 + 4)      # stream record: density + slot-0 attrs = 2,560
EMB_DIM = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# measured 2026-08-31: 43 ms/step on 8050S iGPU vs 2600 ms/step on CPU
# (CPU path pathologically slow while other model containers run); CPU
# fallback kept for portability.

OLLAMA = "http://localhost:11434/api/embeddings"

torch.set_num_threads(min(8, os.cpu_count() or 1))  # small MLPs thrash on 16


def log(msg):
    print(msg, flush=True)


# --- data ---

def load_unique_sentences():
    cache = os.path.join(OUT, "stsb_full_cache.json")
    d = json.load(open(cache))
    seen, uniq = set(), []
    for s in d["train_sentences"] + d["val_sentences"] + d["test_sentences"]:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def pick_ollama_tag():
    try:
        import urllib.request
        raw = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=10).read()
        tags = [m["name"] for m in json.loads(raw).get("models", [])
                if m["name"].startswith("nomic-embed-text")]
        for pref in ("nomic-embed-text:v1.5", "nomic-embed-text:latest"):
            if pref in tags:
                return pref
        return tags[0] if tags else None
    except Exception as e:
        log(f"  ollama tag probe failed: {e}")
        return None


def embed_sentences(sentences):
    """Embed with nomic via ollama; sentence-keyed JSON cache (order-proof)."""
    path = os.path.join(OUT, "s2_embs.json")
    cache = json.load(open(path)) if os.path.exists(path) else {}
    tag = pick_ollama_tag()
    if tag is None:
        log("FATAL: no nomic-embed model in ollama"); sys.exit(2)
    cached_tag = cache.get("__tag__")
    todo = [s for s in sentences if s not in cache]
    if todo and cached_tag not in (None, tag):
        log(f"FATAL: cache tag {cached_tag} != current {tag}; refusing to mix"); sys.exit(2)
    log(f"  embedding {len(todo)} sentences with {tag} ({len(cache)-1 if cache else 0} cached)")
    import urllib.request
    t0 = time.time()
    for n, s in enumerate(todo):
        req = urllib.request.Request(
            OLLAMA, data=json.dumps({"model": tag, "prompt": s}).encode(),
            headers={"Content-Type": "application/json"})
        cache[s] = json.loads(urllib.request.urlopen(req, timeout=60).read())["embedding"]
        if (n + 1) % 500 == 0:
            json.dump({**cache, "__tag__": tag}, open(path, "w"))
            log(f"    {n+1}/{len(todo)} ({time.time()-t0:.0f}s)")
    json.dump({**cache, "__tag__": tag}, open(path, "w"))
    emb = np.array([cache[s] for s in sentences], dtype=np.float32)
    return matryoshka(emb)


def matryoshka(emb768, dim=EMB_DIM):
    t = emb768[:, :dim]
    return (t - t.mean(axis=1, keepdims=True)) / (t.std(axis=1, keepdims=True) + 1e-8)


# --- encoder (V1 arm-B recipe) ---

class ChromavoxField(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B = x.shape[0]
        f = x.reshape(B, N_CELLS, FEATS_PER_CELL)
        out = torch.zeros_like(f)
        out[:, :, 0] = torch.sigmoid(f[:, :, 0])
        for k in range(K_SLOTS):
            b = 1 + k * 4
            ang = f[:, :, b]
            out[:, :, b] = torch.cos(ang)
            out[:, :, b + 1] = torch.sin(ang)
            out[:, :, b + 2] = torch.sigmoid(f[:, :, b + 2])
            out[:, :, b + 3] = torch.sigmoid(f[:, :, b + 3])
        out = out + (torch.round(out * 100) / 100 - out).detach()  # STE quant
        return out.reshape(B, -1)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(EMB_DIM, 256), nn.GELU(), nn.Linear(256, TOTAL_DIMS))
        self.field = ChromavoxField()

    def forward(self, x):
        return self.field(self.mlp(x))


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(TOTAL_DIMS, 256), nn.GELU(), nn.Linear(256, EMB_DIM))

    def forward(self, x):
        return self.mlp(x)


def train_encoder(embs, epochs):
    torch.manual_seed(SEED)
    enc, dec = Encoder().to(DEVICE), Decoder().to(DEVICE)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()),
                            lr=3e-4, weight_decay=1e-4)
    x = torch.tensor(embs, dtype=torch.float32, device=DEVICE)
    n = x.shape[0]
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n, device=DEVICE)
        tot = nb = 0
        for i in range(0, n, 256):
            b = x[perm[i:i + 256]]
            loss = F.mse_loss(dec(enc(b)), b)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if ep % 25 == 0 or ep == epochs:
            log(f"  ep {ep}/{epochs} recon={tot/nb:.4f}")
    enc.eval()
    return enc


# --- fields / records / superposition ---

def encode_fields(enc, embs, bs=512):
    outs = []
    with torch.no_grad():
        for i in range(0, embs.shape[0], bs):
            xb = torch.tensor(embs[i:i + bs], dtype=torch.float32, device=DEVICE)
            outs.append(enc(xb))
    return torch.cat(outs)  # (N, 4608) on DEVICE


def stream_record(field):
    """(N,4608) -> (N,2560): density + slot-0 attrs."""
    f = field.reshape(field.shape[0], N_CELLS, FEATS_PER_CELL)
    return f[:, :, :5].contiguous().view(field.shape[0], REC_DIM)


def build_superposed(f1, f2, swap):
    """(B,4608) x 2 + swap bits -> superposed field (B,4608)."""
    B = f1.shape[0]
    a = f1.reshape(B, N_CELLS, FEATS_PER_CELL)
    b = f2.reshape(B, N_CELLS, FEATS_PER_CELL)
    out = torch.zeros_like(a)
    out[:, :, 0] = torch.clamp(a[:, :, 0] + b[:, :, 0], max=1.0)  # mass adds
    out[:, :, 1:5] = a[:, :, 1:5]   # slot0 = stream A's slot-0 attrs
    out[:, :, 5:9] = b[:, :, 1:5]   # slot1 = stream B's slot-0 attrs
    flipped = torch.zeros_like(out)
    flipped[:, :, 0] = out[:, :, 0]
    flipped[:, :, 1:5] = b[:, :, 1:5]
    flipped[:, :, 5:9] = a[:, :, 1:5]
    m = swap.view(B, 1, 1).float()
    return ((1 - m) * out + m * flipped).view(B, -1)


# --- reader ---

class Reader(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(TOTAL_DIMS, 1024), nn.GELU(), nn.Linear(1024, 2 * REC_DIM))

    def forward(self, x):
        return self.mlp(x).reshape(x.shape[0], 2, REC_DIM)


def train_colocated_reader(enc, fields, epochs, fields_per_epoch):
    """fields: (Ntrain,4608) encoded train fields; pairs sampled on the fly."""
    torch.manual_seed(SEED + 1)
    r = Reader().to(DEVICE)
    opt = torch.optim.AdamW(r.parameters(), lr=3e-4, weight_decay=1e-4)
    n = fields.shape[0]
    steps = fields_per_epoch // 256
    g = torch.Generator().manual_seed(SEED + 2)
    for ep in range(1, epochs + 1):
        tot = nb = 0
        for _ in range(steps):
            idx = torch.randint(0, n, (fields_per_epoch,), generator=g)
            jdx = torch.randint(0, n, (fields_per_epoch,), generator=g)
            keep = idx != jdx
            idx, jdx = idx[keep], jdx[keep]
            f1, f2 = fields[idx].to(DEVICE), fields[jdx].to(DEVICE)
            swap = torch.randint(0, 2, (f1.shape[0],), generator=g).float().to(DEVICE)
            sup = build_superposed(f1, f2, swap)
            t = torch.stack([stream_record(f1), stream_record(f2)], dim=1)
            o = r(sup)
            l_direct = F.mse_loss(o, t)
            l_swap = F.mse_loss(o, t.flip(1))
            loss = torch.minimum(l_direct, l_swap)  # assignment-free
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if ep % 25 == 0 or ep == epochs:
            log(f"  colocated ep {ep}/{epochs} loss={tot/nb:.5f}")
    r.eval()
    return r


def train_solo_reader(enc, records, epochs):
    torch.manual_seed(SEED + 3)
    r = Reader().to(DEVICE)
    opt = torch.optim.AdamW(r.parameters(), lr=3e-4, weight_decay=1e-4)
    x = records.to(DEVICE)  # (N, 2560) true solo records
    n = x.shape[0]
    pad = torch.zeros(n, TOTAL_DIMS - REC_DIM, device=DEVICE)
    inp = torch.cat([x, pad], dim=1)
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n, device=DEVICE)
        tot = nb = 0
        for i in range(0, n, 256):
            b = inp[perm[i:i + 256]]
            t = x[perm[i:i + 256]]
            o = r(b)                       # (B,2,2560)
            tgt = torch.stack([t, t], dim=1)
            loss = F.mse_loss(o, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if ep % 25 == 0 or ep == epochs:
            log(f"  solo ep {ep}/{epochs} loss={tot/nb:.6f}")
    r.eval()
    return r


# --- eval ---

def assign_best(out, t1, t2):
    """out: (2,2560) outputs; returns (rec_slot0role, rec_slot1role) matched
    to (t1, t2) by best total squared error."""
    d_direct = ((out[0] - t1) ** 2).sum() + ((out[1] - t2) ** 2).sum()
    d_swap = ((out[0] - t2) ** 2).sum() + ((out[1] - t1) ** 2).sum()
    if d_direct <= d_swap:
        return out[0], out[1]
    return out[1], out[0]


def rho_on_pairs(recon, truth, pairs):
    ii = np.array([i for i, _ in pairs]); jj = np.array([j for _, j in pairs])
    d_r = np.linalg.norm(recon[ii] - recon[jj], axis=1)
    d_t = np.linalg.norm(truth[ii] - truth[jj], axis=1)
    rho, _ = spearmanr(d_r, d_t)
    return float(rho)


def pairs0(p): return np.array([i for i, _ in p])
def pairs1(p): return np.array([j for _, j in p])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--validity-bar", type=float, default=None,
                    help="encoder solo-rho validity bar (default: 0.90 full, "
                         "0.80 smoke — smoke is pipeline validation only; the "
                         "pre-registered 0.90 applies to the full run)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t_start = time.time()

    if a.smoke:
        n_train_pool, n_eval, enc_ep, read_ep, fpe = 2000, 100, 60, 120, 1024
        validity_bar = a.validity_bar if a.validity_bar is not None else 0.80
    else:
        n_train_pool, n_eval, enc_ep, read_ep, fpe = 10000, 400, 200, 300, 4096
        validity_bar = a.validity_bar if a.validity_bar is not None else 0.90

    log("=== S2: semantic superposition disentanglement ===")
    log(f"mode={'smoke' if a.smoke else 'full'} seed={SEED} "
        f"enc_ep={enc_ep} read_ep={read_ep} fields/ep={fpe}")

    uniq = load_unique_sentences()
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(uniq))
    n_hold = n_eval
    hold_idx = perm[:n_hold]
    pool_idx = perm[n_hold:n_hold + n_train_pool]
    hold_sents = [uniq[i] for i in hold_idx]
    pool_sents = [uniq[i] for i in pool_idx]
    log(f"sentences: train_pool={len(pool_sents)} heldout={len(hold_sents)}")

    log("=== embedding ===")
    pool_embs = embed_sentences(pool_sents)
    hold_embs = embed_sentences(hold_sents)

    log("=== encoder (V1 arm-B recipe) ===")
    t0 = time.time()
    enc = train_encoder(pool_embs, enc_ep)
    log(f"  ({time.time()-t0:.0f}s)")

    # validity precondition: solo structure rho on held-out vs 64-d embedding
    hold_fields = encode_fields(enc, hold_embs)
    hold_np = hold_fields.cpu().numpy()
    emb_np = hold_embs
    rng_m = np.random.default_rng(SEED)
    m = min(2000, n_hold * (n_hold - 1) // 2)
    pr = set()
    while len(pr) < m:
        i, j = rng_m.integers(0, n_hold, 2)
        if i != j:
            pr.add((int(i), int(j)))
    pr = sorted(pr)
    d_e = np.linalg.norm(emb_np[pairs0(pr)] - emb_np[pairs1(pr)], axis=1)
    d_f = np.array([np.linalg.norm(hold_np[i] - hold_np[j]) for i, j in pr])
    enc_rho, _ = spearmanr(d_e, d_f)
    log(f"  encoder solo structure rho = {enc_rho:.4f} (validity bar >= {validity_bar:.2f})")
    if enc_rho < validity_bar:
        log("INVALID RUN — encoder below validity bar; gates NOT evaluated.")
        json.dump({"invalid": True, "enc_rho": enc_rho,
                   "validity_bar": validity_bar,
                   "mode": "smoke" if a.smoke else "full"},
                  open(os.path.join(OUT, "s2_results.json"), "w"), indent=2)
        sys.exit(3)

    # records + solo reader
    hold_recs = stream_record(hold_fields)
    pool_fields = encode_fields(enc, pool_embs)
    pool_recs = stream_record(pool_fields)
    log("=== solo reader (ceiling) ===")
    t0 = time.time()
    r_solo = train_solo_reader(enc, pool_recs, read_ep)
    log(f"  ({time.time()-t0:.0f}s)")

    log("=== colocated reader (the test) ===")
    t0 = time.time()
    r_col = train_colocated_reader(enc, pool_fields, read_ep, fpe)
    log(f"  ({time.time()-t0:.0f}s)")

    # eval: n_hold sentences -> n_hold//2 fields; each sentence one stream
    log("=== evaluation on held-out fields ===")
    hf = hold_fields
    hr = hold_recs
    n_f = n_hold // 2
    swap = torch.tensor([float(rng.integers(0, 2)) for _ in range(n_f)],
                        device=hf.device)
    sup = build_superposed(hf[0::2], hf[1::2], swap)
    with torch.no_grad():
        out_col = r_col(sup)                       # (n_f, 2, 2560)
        pad = torch.zeros(n_hold, TOTAL_DIMS - REC_DIM, device=hr.device)
        out_solo = r_solo(torch.cat([hr, pad], dim=1))[:, 0, :]  # (n_hold,2560)

    rec_col = hr.clone()
    role_of = np.zeros(n_hold, dtype=int)          # 0 = slot0-role, 1 = slot1-role
    for k in range(n_f):
        t1, t2 = hr[2 * k], hr[2 * k + 1]
        r0, r1 = assign_best(out_col[k], t1, t2)
        rec_col[2 * k] = r0
        rec_col[2 * k + 1] = r1
        if swap[k] > 0.5:  # sentence 2k went to slot1
            role_of[2 * k] = 1
            role_of[2 * k + 1] = 0
        else:
            role_of[2 * k] = 0
            role_of[2 * k + 1] = 1

    # pairwise rho metrics
    rng_p = np.random.default_rng(SEED)
    pset = set()
    while len(pset) < 2000:
        i, j = rng_p.integers(0, n_hold, 2)
        if i != j:
            pset.add((int(min(i, j)), int(max(i, j))))
    pr = sorted(pset)

    rec_col_np, rec_solo_np, truth_np = (
        rec_col.cpu().numpy(), out_solo.cpu().numpy(), hr.cpu().numpy())
    res = {}
    res["rho_solo"] = rho_on_pairs(rec_solo_np, truth_np, pr)
    res["rho_colocated"] = rho_on_pairs(rec_col_np, truth_np, pr)
    res["rho_density_solo"] = rho_on_pairs(rec_solo_np[:, :N_CELLS], truth_np[:, :N_CELLS], pr)
    res["rho_density_colocated"] = rho_on_pairs(rec_col_np[:, :N_CELLS], truth_np[:, :N_CELLS], pr)
    res["rho_color_solo"] = rho_on_pairs(rec_solo_np[:, N_CELLS:], truth_np[:, N_CELLS:], pr)
    res["rho_color_colocated"] = rho_on_pairs(rec_col_np[:, N_CELLS:], truth_np[:, N_CELLS:], pr)

    # role symmetry
    pr0 = [(i, j) for (i, j) in pr if role_of[i] == 0 and role_of[j] == 0]
    pr1 = [(i, j) for (i, j) in pr if role_of[i] == 1 and role_of[j] == 1]
    rho_role0 = rho_on_pairs(rec_col_np, truth_np, pr0) if len(pr0) > 10 else float("nan")
    rho_role1 = rho_on_pairs(rec_col_np, truth_np, pr1) if len(pr1) > 10 else float("nan")

    delta_solo = res["rho_solo"] - res["rho_colocated"]
    delta_role0 = res["rho_solo"] - rho_role0
    delta_role1 = res["rho_solo"] - rho_role1

    # lit-cell overlap stat
    lit = (hf.reshape(n_hold, N_CELLS, FEATS_PER_CELL)[:, :, 0] > 0.01).cpu().numpy()
    jac = []
    for k in range(n_f):
        i, j = 2 * k, 2 * k + 1
        A, Bc = lit[i], lit[j]
        jac.append((A & Bc).sum() / max(1e-9, (A | Bc).sum()))
    mean_jac = float(np.mean(jac))

    gates = {
        "G1_capacity": res["rho_colocated"] >= 0.85,
        "G2_interference": delta_solo <= 0.05,
        "G3_symmetry": abs(delta_role0 - delta_role1) <= 0.05,
    }
    if res["rho_colocated"] < 0.60:
        verdict = "KILL"
    elif all(gates.values()):
        verdict = "PASS"
    else:
        verdict = "GRAY"

    results = {
        "experiment": "S2 semantic superposition disentanglement",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": SEED, "mode": "smoke" if a.smoke else "full",
        "n_train_pool": len(pool_sents), "n_heldout": n_hold, "n_fields": n_f,
        "n_pairs": len(pr), "ollama_tag": pick_ollama_tag(),
        "enc_epochs": enc_ep, "reader_epochs": read_ep,
        "encoder_solo_rho": round(enc_rho, 4), "validity_bar": round(validity_bar, 2),
        "mean_lit_jaccard": round(mean_jac, 4),
        "swap_rate": float(swap.mean()),
        "rho": {k: round(v, 4) for k, v in res.items()},
        "rho_role0": round(rho_role0, 4), "rho_role1": round(rho_role1, 4),
        "delta_solo_colocated": round(delta_solo, 4),
        "delta_role0": round(delta_role0, 4), "delta_role1": round(delta_role1, 4),
        "gates": {k: bool(v) for k, v in gates.items()},
        "verdict": verdict,
        "wall_minutes": round((time.time() - t_start) / 60, 1),
    }
    path = os.path.join(OUT, "s2_results.json")
    json.dump(results, open(path, "w"), indent=2)
    torch.save({"encoder": enc.state_dict()}, os.path.join(OUT, "s2_encoder.pt"))
    torch.save({"solo": r_solo.state_dict(), "colocated": r_col.state_dict()},
               os.path.join(OUT, "s2_readers.pt"))

    log("\n" + "=" * 64)
    log("S2 verdict")
    log("=" * 64)
    for k in ("rho_solo", "rho_colocated", "rho_density_solo", "rho_density_colocated",
              "rho_color_solo", "rho_color_colocated"):
        log(f"  {k:<24} {res[k]:+.4f}")
    log(f"  interference (solo-coloc) {delta_solo:+.4f}   role0 {delta_role0:+.4f}  role1 {delta_role1:+.4f}")
    log(f"  encoder sanity rho {enc_rho:.4f}  lit Jaccard {mean_jac:.3f}  swap rate {swap.mean():.2f}")
    for g, v in gates.items():
        log(f"  {g}: {'PASS' if v else 'FAIL'}")
    log(f"VERDICT: {verdict}")
    log(f"saved {path}")


if __name__ == "__main__":
    main()