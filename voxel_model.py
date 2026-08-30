"""
voxel_model.py — a model that LEARNS to assign each text a voxel
position + a LIST of HSL colors (not a mix) in a small 3D grid.

User's training guidelines (this file):
  * The model assigns colors ITSELF — nothing hardcodes "X is green".
  * Soft anchoring session only: the sentence about green should come
    out greenish (a gentle pull, not a rule).
  * A no-guardrail session: pure Barlow Twins on the voxel rep,
    no teacher, no anchor — does it stay non-collapsed on its own?
  * Distill session: pull toward an off-the-shelf embedder
    (nomic-embed-text, precomputed in exp1) — checked via
    embedding cosine similarity.
  * Anti-collapse measures, logged every epoch:
      - BT off-diagonal (redundancy) and diagonal (invariance)
      - mean pairwise cosine of pooled reps (→1 means collapse)
      - per-dim std (degenerate dims → 0)
      - slot-activation entropy (did it use all color slots?)
      - position entropy across grid cells
      - hue histogram (used-color spread)

Voxel model structure:
  text -> BPE ids -> TransformerEncoder(4x256) -> h
    h -> pos_head  -> (x, y, z)  logits per grid axis
    h -> slot_head -> K slots, each = [activation, h, s, l]
  pooled voxel rep = MLP(pos one-hot, active-slot features) -> 64-dim
  BT / distill / anchor losses act on the pooled rep + hue slot.

Setup flags:
  free    : Barlow Twins on (text, masked-text) views only
  anchor  : free + soft pull of dominant-slot hue toward named
            color words (green -> ~120deg, ...)
  distill : free + 1 - cos(pooled_rep, teacher_embed)

Run:  py -3.12 voxel_model.py free | anchor | distill | all
"""
from __future__ import annotations
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn

SEED = 1337
GRID = 8            # 8^3 = 512 cells
K_SLOTS = 8         # up to 8 colors per voxel (a LIST, never a mix)
LATENT = 64         # pooled "A-space" dim
HID = 256
LAYERS = 4
N_HEADS = 4
VOCAB = 1024
EPOCHS = 60
BATCH = 64
LR = 3e-4

DATA_CSV = r"C:\Users\skell\Desktop\embedding-vibes\data\logical-fallacy-repo\data"
TEACHER_NPY = (r"C:\Users\skell\Desktop\embedding-vibes"
               r"\experiments\exp1_linear_probe\results"
               r"\embeddings_nomic_embed_text.npy")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# a few color words for the anchor session (soft, low weight — the
# model still chooses all of its own colors; this just checks that
# "the thought of green reads green" is REACHABLE, not forced)
ANCHOR_WORDS = {
    "green": 120, "red": 0, "blue": 240, "yellow": 60,
    "orange": 30, "purple": 280, "white": -1, "black": -1,
}

def set_seed(s=SEED):
    import random; random.seed(s); np.random.seed(s)
    import torch; torch.manual_seed(s)

def load_data():
    """Return (texts, masked_texts, labels) from the edu splits (2449)."""
    import pandas as pd
    rows = []
    for split in ["edu_train.csv", "edu_dev.csv", "edu_test.csv"]:
        df = pd.read_csv(os.path.join(DATA_CSV, split))
        for _, r in df.iterrows():
            t = str(r["source_article"]).replace("MSK<0>", " MSK ").replace("MSK<1>", " MSK ")
            m = str(r["masked_articles"])
            for tok in ["MSK<0>", " MSK<1>", " MSK<2>", " MSK<3>"]:
                m = m.replace(tok, " MSK ")
            t = t.replace("\n", " ")
            m = m.replace("\n", " ")
            if t and m:
                rows.append((t, m, str(r["updated_label"])))
    texts = [r[0] for r in rows]
    masked = [r[1] for r in rows]
    labels = [r[2] for r in rows]
    return texts, masked, labels

def train_bpe(texts):
    """Pure-python BPE (tokenizers crate is broken on this box: os 123),
    trained on a deterministic subsample, then cached to disk."""
    from bpe_pure import Bpe
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "bpe_cache.json")
    b = Bpe(vocab_size=VOCAB)
    if os.path.exists(cache):
        with open(cache) as f:
            d = json.load(f)
        b.w2i = {k: int(v) for k, v in d["w2i"].items()}
        b.merges = [tuple(m) for m in d["merges"]]
        return b
    sample = texts if len(texts) <= 400 else texts[:400]
    b.train(sample)
    with open(cache, "w") as f:
        json.dump({"w2i": b.w2i, "merges": b.merges}, f)
    return b

class VoxelNet(nn.Module):
    def __init__(self, vocab_size):
        nn.Module.__init__(self)
        self.tok = None  # set externally
        self.emb = nn.Embedding(vocab_size + 3, HID)  # pad = id 0, masked
        enc = nn.TransformerEncoderLayer(HID, N_HEADS, 1024, 0.1, batch_first=True)
        # LAYERS=0 -> bag-of-words encoder (no attention): debugging bypass
        self.enc = (nn.TransformerEncoder(enc, LAYERS, enable_nested_tensor=False)
                    if LAYERS else nn.Identity())
        self.pos_head = nn.Linear(HID, 3 * GRID)
        self.slot_head = nn.Linear(HID, K_SLOTS * 4)  # [a, h, s, l] each
        # pooled rep builder  (input: 512 pos-onehot + 8 slots × (64+32+32)
        #                      + DIRECT encoder state = differentiable path)
        self.sl_emb_h = nn.Embedding(32, HID // 4)  # hue buckets  -> 64
        self.sl_emb_s = nn.Embedding(16, HID // 8)  # sat buckets  -> 32
        self.sl_emb_l = nn.Embedding(16, HID // 8)  # brightness   -> 32
        sl_dim = K_SLOTS * (HID//4 + HID//8 + HID//8)
        self.pool = nn.Sequential(
            nn.Linear(3 * GRID + sl_dim + HID, 256),
            nn.GELU(), nn.Linear(256, LATENT))

    def forward(self, ids):
        h = self.emb(ids)
        h = self.enc(h, src_key_padding_mask=(ids == 0))
        # attention-masked mean (ignore pad token id 0)
        m = (ids != 0).to(h.dtype).unsqueeze(-1)
        h = (h * m).sum(1) / m.sum(1).clamp(min=1)
        pos_logits = self.pos_head(h).view(-1, 3, GRID)
        slots = self.slot_head(h).view(-1, K_SLOTS, 4)
        act = torch.sigmoid(slots[..., 0])             # [B, K]
        hue = torch.sigmoid(slots[..., 1]) * 360.0     # [B, K]
        sat = torch.sigmoid(slots[..., 2])
        bri = torch.sigmoid(slots[..., 3])
        # pooled rep = f(position logits [3*8], slot features [K*(4+dims)])
        pos_feat = pos_logits.flatten(1)
        hb = (hue / 360.0 * 31).long().clamp(0, 31)
        sb = (sat * 15).long().clamp(0, 15)
        lb = (bri * 15).long().clamp(0, 15)
        sl_feat = torch.cat([
            act.unsqueeze(-1) * self.sl_emb_h(hb),
            act.unsqueeze(-1) * self.sl_emb_s(sb),
            act.unsqueeze(-1) * self.sl_emb_l(lb),
        ], -1).flatten(1)
        # direct differentiable content path (encoder mean is the ONLY
        # gradient highway for shared-token invariance; bucketed hues
        # are quantized and pass no gradient)
        rep = self.pool(torch.cat([pos_feat, sl_feat, h], -1))
        return dict(pos_logits=pos_logits, act=act, hue=hue, sat=sat,
                    bri=bri, rep=rep)

    def argmax3(self, pos_logits):
        a = pos_logits.argmax(dim=-1)  # [B,3]
        return a[:, 0] * GRID * GRID + a[:, 1] * GRID + a[:, 2]

def barlow_twins(r1, r2, lam=0.005):
    import torch, torch.nn as nn
    r1 = nn.functional.normalize(r1, dim=1)
    r2 = nn.functional.normalize(r2, dim=1)
    c = (r1.T @ r2) / r1.size(0)
    d = (c.diag() - 1).pow(2).sum()
    o = (c.pow(2).sum() - c.diag().pow(2).sum()) / (c.size(0) - 1)
    return d + lam * o, (d.item(), o.item())

def vicreg(z1, z2, sim_c=10.0, var_c=10.0, cov_c=5.0):
    """VicReg: variance hinge + covariance decorrelation + MSE invariance.
    Robust where Barlow Twins starves (small batch, weak-view pairs)."""
    import torch.nn.functional as F
    std1 = torch.sqrt(z1.var(0) + 1e-4)
    std2 = torch.sqrt(z2.var(0) + 1e-4)
    var = F.relu(1.0 - std1).mean() + F.relu(1.0 - std2).mean()
    z1n = (z1 - z1.mean(0)) / (std1 + 1e-4)
    z2n = (z2 - z2.mean(0)) / (std2 + 1e-4)
    N = z1.size(0)
    c12 = (z1n.T @ z2n) / N
    c11 = (z1n.T @ z1n) / N
    c22 = (z2n.T @ z2n) / N
    D = z1.size(1)
    off = ((c12.pow(2).sum() - c12.diag().pow(2).sum()) +
           (c11.pow(2).sum() - c11.diag().pow(2).sum()) +
           (c22.pow(2).sum() - c22.diag().pow(2).sum())) / (3 * D * (D - 1))
    inv = F.mse_loss(z1, z2)
    return sim_c * inv + var_c * var + cov_c * off, \
        (float(inv), float(var), float(off))

def cosine_to_teacher(rep, T):
    """Distill loss vs the ( PCA-projected, 64-dim ) teacher."""
    import torch.nn as nn
    r = nn.functional.normalize(rep, dim=1)
    t = nn.functional.normalize(T, dim=1)
    return (1 - (r * t).sum(1)).mean(), (r * t).sum(1).mean(), (1 - (r * t).sum(1)).var().sqrt()

def anchor_loss(out, row_hues):
    """row_hues: list of (local_row, hue_deg) within the batch."""
    import torch
    if not row_hues:
        return torch.tensor(0.0, device=out["hue"].device), 0.0
    rows = [r for r, _ in row_hues]
    sel = torch.tensor(rows, device=out["hue"].device)
    sal = (out["bri"] * out["act"])[sel]
    dom = sal.argmax(dim=-1)
    h = out["hue"][sel, dom] / 360.0             # [n] normalized
    tgt = torch.tensor([hh / 360.0 for _, hh in row_hues],
                       device=h.device)
    diff = (h - tgt + 0.5) % 1.0 - 0.5            # circular shortest arc
    return diff.pow(2).mean() * 2.0, h.mean().item()

def collapse_metrics(rep, out, texts):
    r = rep.detach().cpu().numpy().astype(np.float64)
    r = (r - r.mean(0)) / (r.std(0) + 1e-8)
    D = r @ r.T
    iu = np.triu_indices(len(r), 1)
    mpc = float(D[iu].mean())
    perdim = float(r.std(0).mean())
    act = out["act"].detach().cpu().numpy()
    p = np.clip(act.mean(0), 1e-6, 1)
    slot_ent = float(-(p * np.log(p)).sum()) / math.log(len(p))
    act_tot = act.mean()
    p2 = np.clip(act, 1e-6, 1); p2 = p2 / p2.sum(1, keepdims=True)
    sl_ent = float(-(p2 * np.log(p2)).sum(1).mean() / K_SLOTS)
    return dict(mean_pair_cos=mpc, perdim_std=perdim,
                slot_util=sl_ent, act_frac=float(act_tot),
                n_active_slots=float((act > 0.5).mean()))

def run(setup: str, log=print):
    set_seed()
    os.makedirs(OUT, exist_ok=True)
    dev = os.environ.get("VOXEL_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    if dev == "cuda":
        # ROCm mem-efficient/flash attention kernels are experimental on
        # this box and appear to return input-independent garbage; force
        # the math kernel (slower, correct).
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    log(f"[{setup}] device={dev} torch={torch.__version__}")

    texts, masked, labels = load_data()
    T = np.load(TEACHER_NPY)
    assert len(T) == len(texts), (len(T), len(texts))
    T = torch.tensor(T, dtype=torch.float32, device=dev)
    T = (T - T.mean(0)) / (T.std(0) + 1e-8)
    if T.size(1) != LATENT:
        # deterministic PCA of the 768-d teacher down to LATENT dims
        # (SVD on CPU: 2.4k x 768 takes <1s and avoids ROCm LAPACK paths)
        Tc = T.cpu() - T.cpu().mean(0)
        U, S, Vh = torch.linalg.svd(Tc, full_matrices=False)
        T = (Tc @ Vh[:LATENT].T).to(dev)
        T = (T - T.mean(0)) / (T.std(0) + 1e-8)
        exp_var = float((S[:LATENT] ** 2).sum() / (S ** 2).sum())
        log(f"teacher 768 -> PCA {LATENT} (explained var {exp_var:.2f})")
    # teacher per masked view: not available in npy; use same-row teacher
    # for original only (distill applied to original view).

    tok = train_bpe(texts)
    MAXLEN = 128
    _wcache = {}

    def enc(s):
        """Per-word memoized BPE (per-word encoding is exact: no
        cross-word merges)."""
        out = []
        for w in tok.words(s):
            if w not in _wcache:
                _wcache[w] = tok.encode(w)
            out += _wcache[w]
        return out

    # window start per row: ~32 tokens before the first place the
    # masked view diverges from the original (so the BT contrast --
    # the removed claim -- is always inside the window)
    starts = []
    for t, m in zip(texts, masked):
        a = enc(t); b = enc(m)
        d = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), 0)
        starts.append(max(0, d - 32))

    def encw(ids, start):
        if len(ids) <= MAXLEN:
            return ids
        start = max(0, min(start, len(ids) - MAXLEN))
        return ids[start:start + MAXLEN]

    L = MAXLEN
    # fixed-length right-padding (pad id 0 at the end); mask = (ids != 0)
    def P(ids):
        arr = np.zeros(L, dtype=np.int64)
        arr[:len(ids)] = ids
        return torch.tensor(arr, device=dev)
    Xp = [P(encw(enc(t), s)) for t, s in zip(texts, starts)]
    Xp2 = [P(encw(enc(m), s)) for m, s in zip(masked, starts)]
    XL = Xp; XL2 = Xp2

    net = VoxelNet(VOCAB)
    net = net.to(dev)
    import torch.nn as nn
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    # anchor rows: sentences containing a single color word as the main subject
    anchor_pairs = []
    for i, t in enumerate(texts):
        tl = t.lower()
        for w, hue in ANCHOR_WORDS.items():
            if hue >= 0 and (w + " ") in tl and len(tl.split()) <= 12:
                anchor_pairs.append((i, hue)); break

    hist = []
    t0 = time.time()
    snap = None
    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(len(texts), device=dev)
        tot = d_tot = o_tot = n = 0
        for s in range(0, len(texts), BATCH):
            idx = perm[s:s + BATCH].tolist()
            ids = torch.stack([XL[i] for i in idx])
            ids2 = torch.stack([XL2[i] for i in idx])
            # views: original & masked (left-padded; mask handled inside net)
            out1 = net(ids)
            out2 = net(ids2)
            loss, (di, va, of) = vicreg(out1["rep"], out2["rep"])
            d_tot += di * len(idx); o_tot += of * len(idx); n += len(idx)
            if setup == "distill":
                Tb = T[torch.tensor(idx, device=dev)]
                cl, mean_cos, cos_sd = cosine_to_teacher(out1["rep"], Tb)
                loss = loss + 0.5 * cl
            if setup == "anchor":
                row_hues = [(i - idx[0], h)
                            for (i, h) in anchor_pairs if i in idx]
                al, dom_h = anchor_loss(out1, row_hues)
                if row_hues:
                    loss = loss + 0.25 * al
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        with torch.no_grad():
            gnorm = float(torch.sqrt(sum(
                (p.grad ** 2).sum() for p in net.parameters()
                if p.grad is not None)))
            pdelta = (float(((net.pos_head.weight - snap) ** 2).sum() ** 0.5)
                      if snap is not None else -1.0)
            snap = net.pos_head.weight.detach().clone()
            net.eval()
            rep = net(torch.stack(XL))["rep"]
            out = net(torch.stack(XL))
        cm = collapse_metrics(rep, out, texts)
        rec = dict(epoch=ep, loss=loss.item(), bt_d=d_tot / n, bt_off=o_tot / n,
                   **cm,
                   elapsed=s.time() if False else time.time() - t0)
        hist.append(rec)
        if setup == "distill":
            rec["mean_cos_teacher"] = float(mean_cos)
        log(f"[{setup}] ep{ep:3d} L={loss.item():.3f} BTd={rec['bt_d']:.3f} "
            f"off={rec['bt_off']:.3f} mpc={cm['mean_pair_cos']:.3f} "
            f"pstd={cm['perdim_std']:.3f} slots={cm['n_active_slots']:.2f} "
            f"gn={gnorm:.2e} pd={pdelta:.2e}")
    with open(os.path.join(OUT, f"history_{setup}.json"), "w") as f:
        json.dump(hist, f, indent=1)
    torch.save(net.state_dict(),
               os.path.join(OUT, f"model_{setup}.pt"))

    # final: report anchor reachability
    if True:
        with torch.no_grad():
            net.eval()
            for probe in ["green", "red", "blue", "purple", "starlight",
                          "the feeling of certainty"]:
                ids = enc(probe)[:MAXLEN]
                arr = np.zeros(L, dtype=np.int64); arr[:len(ids)] = ids
                o = net(torch.tensor(arr, device=dev).unsqueeze(0))
                sal = (o["bri"] * o["act"])[0].cpu().numpy()
                order = sal.argsort()[::-1][:3]
                cols = [(o["hue"][0, i].item(), o["sat"][0, i].item(),
                         o["bri"][0, i].item(), o["act"][0, i].item())
                        for i in order]
                pos = o["pos_logits"][0].argmax(-1).cpu().numpy()
                log(f"  probe '{probe}': pos={pos.tolist()} "
                    f"colors(Hdeg,S,L,a)={[(round(a,1),round(b,2),round(c,2),round(d,2)) for a,b,c,d in cols]}")
    return hist

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "all":
        for s in ["free", "anchor", "distill"]:
            run(s)
    else:
        run(which)
