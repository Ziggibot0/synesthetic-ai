"""
v1_learned_encoder.py — Can a model learn to produce chromavox fields from text?

Pre-registered design: docs/v1_learned_encoder.md

WHAT IT TESTS:
  V0 proved a fixed orthogonal projection preserves semantic structure in
  chromavox space (rho 0.984). V1 asks: can a LEARNED encoder (MLP, trained
  by gradient descent) do it? The constraints (sigmoid, cos/sin, 2-sig-fig
  quantization) make this harder than a plain linear map.

ARMS:
  A  learned-dist    MLP encoder, distance-preservation loss, constrained
  B  learned-autoenc MLP encoder + decoder, reconstruction loss, constrained
  C  fixed-ortho     V0's fixed orthogonal projection (ceiling)
  D  random          random projection (null)
  E  identity        64-dim → 64-dim (raw ceiling)

GATES:
  V1-G1  A rho > 0.50
  V1-G2  A - D > 0.20
  V1-G3  A >= 0.70 × C
  V1-G4  B rho >= 0.50
  V1-G5  which objective is better (A vs B)

USAGE:
  py -3.12 v1_learned_encoder.py              # run all, print verdict
  py -3.12 v1_learned_encoder.py --epochs 500 # more training

CPU-only. ~20 min total (embedding is the slow part; training is ~5 min).
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

GRID = 8
N_CELLS = GRID ** 3
K_SLOTS = 2
FEATURES_PER_CELL = 1 + K_SLOTS * 4  # 9
TOTAL_DIMS = N_CELLS * FEATURES_PER_CELL  # 4,608
EMB_DIM = 64
HIDDEN = 256

DEVICE = "cpu"  # intentionally CPU — model is tiny


def matryoshka_truncate(emb: np.ndarray, dim: int = EMB_DIM) -> np.ndarray:
    truncated = emb[:, :dim]
    mean = truncated.mean(axis=1, keepdims=True)
    std = truncated.std(axis=1, keepdims=True) + 1e-8
    return (truncated - mean) / std


def fetch_stsb_all():
    """Fetch all STS-B splits from HuggingFace."""
    cache = os.path.join(OUT, "stsb_full_cache.json")
    if os.path.exists(cache):
        d = json.load(open(cache))
        return (d["train_sentences"], d["train_pairs"], d["train_scores"],
                d["val_sentences"], d["val_pairs"], d["val_scores"],
                d["test_sentences"], d["test_pairs"], d["test_scores"])

    from datasets import load_dataset
    splits = {}
    for split_name in ["train", "validation", "test"]:
        ds = load_dataset("sentence-transformers/stsb", split=split_name)
        sentences, sent_idx, pairs, scores = [], {}, [], []
        for row in ds:
            s1, s2, score = row["sentence1"], row["sentence2"], float(row["score"])
            for s in (s1, s2):
                if s not in sent_idx:
                    sent_idx[s] = len(sentences)
                    sentences.append(s)
            pairs.append((sent_idx[s1], sent_idx[s2]))
            scores.append(score)
        splits[split_name] = (sentences, pairs, scores)

    result = {
        "train_sentences": splits["train"][0],
        "train_pairs": [[p[0], p[1]] for p in splits["train"][1]],
        "train_scores": splits["train"][2],
        "val_sentences": splits["validation"][0],
        "val_pairs": [[p[0], p[1]] for p in splits["validation"][1]],
        "val_scores": splits["validation"][2],
        "test_sentences": splits["test"][0],
        "test_pairs": [[p[0], p[1]] for p in splits["test"][1]],
        "test_scores": splits["test"][2],
    }
    json.dump(result, open(cache, "w"))
    return (result["train_sentences"], result["train_pairs"], result["train_scores"],
            result["val_sentences"], result["val_pairs"], result["val_scores"],
            result["test_sentences"], result["test_pairs"], result["test_scores"])


def embed_sentences(sentences: list[str]) -> np.ndarray:
    """Embed with nomic-embed-text:v1.5, matryoshka-truncate to 64."""
    cache = os.path.join(OUT, "v0_nomic_768.npy")
    if os.path.exists(cache):
        emb768 = np.load(cache)
        if emb768.shape[0] >= len(sentences):
            return matryoshka_truncate(emb768[:len(sentences)], EMB_DIM)

    import urllib.request, json as _json
    model_name = "nomic-embed-text:v1.5"
    embs = []
    t0 = time.time()
    for i in range(0, len(sentences), 64):
        for s in sentences[i:i+64]:
            payload = _json.dumps({"model": model_name, "prompt": s}).encode()
            req = urllib.request.Request(
                "http://localhost:11434/api/embeddings",
                data=payload, headers={"Content-Type": "application/json"})
            resp = urllib.request.urlopen(req, timeout=30)
            data = _json.loads(resp.read())
            embs.append(data["embedding"])
        if (i + 64) % 256 == 0 or i + 64 >= len(sentences):
            print(f"  embedded {min(i+64, len(sentences))}/{len(sentences)} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    emb768 = np.array(embs, dtype=np.float32)
    np.save(cache, emb768)
    return matryoshka_truncate(emb768, EMB_DIM)


# --- Chromavox constraints (baked into forward pass) ---

class ChromavoxField(nn.Module):
    """Apply chromavox constraints to a flat vector of TOTAL_DIMS values."""
    def __init__(self, quantize=True):
        super().__init__()
        self.quantize = quantize

    def forward(self, x):
        """x: (B, TOTAL_DIMS) → (B, TOTAL_DIMS) constrained."""
        B = x.shape[0]
        field = x.reshape(B, N_CELLS, FEATURES_PER_CELL)
        out = torch.zeros_like(field)
        # density: sigmoid
        out[:, :, 0] = torch.sigmoid(field[:, :, 0])
        for k in range(K_SLOTS):
            base = 1 + k * 4
            angle = field[:, :, base]
            out[:, :, base] = torch.cos(angle)
            out[:, :, base + 1] = torch.sin(angle)
            out[:, :, base + 2] = torch.sigmoid(field[:, :, base + 2])
            out[:, :, base + 3] = torch.sigmoid(field[:, :, base + 3])

        if self.quantize:
            # straight-through estimator: round in forward, pass grad in backward
            out = out + (torch.round(out * 100) / 100 - out).detach()

        return out.reshape(B, -1)


class Encoder(nn.Module):
    """64-dim → chromavox field (constrained)."""
    def __init__(self, quantize=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(EMB_DIM, HIDDEN), nn.GELU(),
            nn.Linear(HIDDEN, TOTAL_DIMS),
        )
        self.field = ChromavoxField(quantize=quantize)

    def forward(self, x):
        return self.field(self.mlp(x))


class Decoder(nn.Module):
    """chromavox field → 64-dim (unconstrained reconstruction)."""
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(TOTAL_DIMS, HIDDEN), nn.GELU(),
            nn.Linear(HIDDEN, EMB_DIM),
        )

    def forward(self, x):
        return self.mlp(x)


# --- Metrics ---

def pairwise_rho(projected, reference, n_pairs=2000, seed=1337):
    n = projected.shape[0]
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(min(n_pairs, n * (n - 1) // 2)):
        i, j = rng.integers(0, n, size=2)
        if i != j:
            pairs.append((i, j))
    pairs = np.array(pairs)
    d_proj = np.linalg.norm(projected[pairs[:, 0]] - projected[pairs[:, 1]], axis=1)
    d_ref = np.linalg.norm(reference[pairs[:, 0]] - reference[pairs[:, 1]], axis=1)
    rho, _ = spearmanr(d_proj, d_ref)
    return float(rho)


def stsb_rho(projected, pairs, scores):
    if not pairs:
        return float('nan')
    valid = [(i, j, s) for (i, j), s in zip(pairs, scores)
             if i < projected.shape[0] and j < projected.shape[0]]
    if not valid:
        return float('nan')
    d = np.array([np.linalg.norm(projected[i] - projected[j]) for i, j, _ in valid])
    s = np.array([sc for _, _, sc in valid])
    rho, _ = spearmanr(d, s)
    return float(rho)


# --- Fixed projections (from V0) ---

def project_fixed_ortho(emb):
    rng = np.random.default_rng(1337)
    rand_mat = rng.standard_normal((TOTAL_DIMS, EMB_DIM))
    Q, _ = np.linalg.qr(rand_mat)
    # apply constraints
    projected = (emb @ Q.T).astype(np.float32)
    field = projected.reshape(-1, N_CELLS, FEATURES_PER_CELL)
    out = np.zeros_like(field)
    out[:, :, 0] = 1.0 / (1.0 + np.exp(-field[:, :, 0]))
    for k in range(K_SLOTS):
        base = 1 + k * 4
        out[:, :, base] = np.cos(field[:, :, base])
        out[:, :, base + 1] = np.sin(field[:, :, base + 1])
        out[:, :, base + 2] = 1.0 / (1.0 + np.exp(-field[:, :, base + 2]))
        out[:, :, base + 3] = 1.0 / (1.0 + np.exp(-field[:, :, base + 3]))
    out = np.round(out * 100) / 100  # quantize
    return out.reshape(emb.shape[0], -1)


def project_random(emb):
    rng = np.random.default_rng(42)
    return rng.standard_normal((emb.shape[0], TOTAL_DIMS)).astype(np.float32)


# --- Training ---

def train_dist_preservation(emb_train, emb_test, epochs, seed=1337):
    """Arm A: train encoder to preserve pairwise distances."""
    torch.manual_seed(seed)
    enc = Encoder(quantize=True).to(DEVICE)
    opt = torch.optim.AdamW(enc.parameters(), lr=3e-4, weight_decay=1e-4)
    emb_t = torch.tensor(emb_train, dtype=torch.float32, device=DEVICE)

    n = emb_t.shape[0]
    n_pairs_per_epoch = min(2000, n * (n - 1) // 2)

    for ep in range(1, epochs + 1):
        enc.train()
        idx = torch.randint(0, n, (n_pairs_per_epoch, 2), device=DEVICE)
        mask = idx[:, 0] != idx[:, 1]
        idx = idx[mask]
        x1 = emb_t[idx[:, 0]]
        x2 = emb_t[idx[:, 1]]
        v1 = enc(x1)
        v2 = enc(x2)
        d_ref = (x1 - x2).norm(dim=1)
        d_vox = (v1 - v2).norm(dim=1)
        loss = F.mse_loss(d_vox, d_ref)
        opt.zero_grad(); loss.backward(); opt.step()

        if ep % 50 == 0 or ep == epochs:
            print(f"  A ep {ep}/{epochs} loss={loss.item():.4f}", flush=True)

    # eval
    enc.eval()
    with torch.no_grad():
        projected = enc(torch.tensor(emb_test, dtype=torch.float32, device=DEVICE)).cpu().numpy()
    return projected


def train_autoencoder(emb_train, emb_test, epochs, seed=1337):
    """Arm B: train encoder + decoder for reconstruction."""
    torch.manual_seed(seed)
    enc = Encoder(quantize=True).to(DEVICE)
    dec = Decoder().to(DEVICE)
    params = list(enc.parameters()) + list(dec.parameters())
    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-4)
    emb_t = torch.tensor(emb_train, dtype=torch.float32, device=DEVICE)
    n = emb_t.shape[0]
    batch = 256

    for ep in range(1, epochs + 1):
        enc.train(); dec.train()
        perm = torch.randperm(n, device=DEVICE)
        tot, nb = 0.0, 0
        for i in range(0, n, batch):
            b = perm[i:i+batch]
            x = emb_t[b]
            vox = enc(x)
            recon = dec(vox)
            loss = F.mse_loss(recon, x)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1

        if ep % 50 == 0 or ep == epochs:
            print(f"  B ep {ep}/{epochs} loss={tot/nb:.4f}", flush=True)

    enc.eval()
    with torch.no_grad():
        projected = enc(torch.tensor(emb_test, dtype=torch.float32, device=DEVICE)).cpu().numpy()
    return projected


# --- Main ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    print("=== fetching STS-B (all splits) ===", flush=True)
    tr_s, tr_p, tr_sc, val_s, val_p, val_sc, test_s, test_p, test_sc = fetch_stsb_all()
    print(f"  train: {len(tr_s)} sentences, {len(tr_p)} pairs", flush=True)
    print(f"  test:  {len(test_s)} sentences, {len(test_p)} pairs", flush=True)

    print("=== embedding all sentences ===", flush=True)
    # embed unique sentences across all splits
    all_sentences = list(set(tr_s + val_s + test_s))
    print(f"  {len(all_sentences)} unique sentences", flush=True)
    emb_all = embed_sentences(all_sentences)
    # build index
    sent_idx = {s: i for i, s in enumerate(all_sentences)}

    # split embeddings
    emb_train = np.array([emb_all[sent_idx[s]] for s in tr_s], dtype=np.float32)
    emb_test = np.array([emb_all[sent_idx[s]] for s in test_s], dtype=np.float32)

    # test pairs reference indices into test_s directly (from fetch_stsb_all)
    tp_filtered = [(p[0], p[1]) for p in test_p
                   if p[0] < len(test_s) and p[1] < len(test_s)]

    results = {}

    # Arm A: learned distance preservation
    print("\n=== arm A (learned-dist) ===", flush=True)
    t0 = time.time()
    proj_A = train_dist_preservation(emb_train, emb_test, a.epochs)
    rho_A = pairwise_rho(proj_A, emb_test)
    rho_stsb_A = stsb_rho(proj_A, tp_filtered, test_sc)
    print(f"  rho vs embedding: {rho_A:.4f}", flush=True)
    print(f"  rho vs STS-B:     {rho_stsb_A:.4f}", flush=True)
    print(f"  ({time.time()-t0:.1f}s)", flush=True)
    results["A"] = {"name": "learned-dist", "rho_embedding": round(rho_A, 4),
                    "rho_stsb": round(rho_stsb_A, 4), "dims": TOTAL_DIMS}

    # Arm B: learned autoencoder
    print("\n=== arm B (learned-autoenc) ===", flush=True)
    t0 = time.time()
    proj_B = train_autoencoder(emb_train, emb_test, a.epochs)
    rho_B = pairwise_rho(proj_B, emb_test)
    rho_stsb_B = stsb_rho(proj_B, tp_filtered, test_sc)
    print(f"  rho vs embedding: {rho_B:.4f}", flush=True)
    print(f"  rho vs STS-B:     {rho_stsb_B:.4f}", flush=True)
    print(f"  ({time.time()-t0:.1f}s)", flush=True)
    results["B"] = {"name": "learned-autoenc", "rho_embedding": round(rho_B, 4),
                    "rho_stsb": round(rho_stsb_B, 4), "dims": TOTAL_DIMS}

    # Arm C: fixed orthogonal (ceiling)
    print("\n=== arm C (fixed-ortho) ===", flush=True)
    proj_C = project_fixed_ortho(emb_test)
    rho_C = pairwise_rho(proj_C, emb_test)
    rho_stsb_C = stsb_rho(proj_C, tp_filtered, test_sc)
    print(f"  rho vs embedding: {rho_C:.4f}", flush=True)
    results["C"] = {"name": "fixed-ortho", "rho_embedding": round(rho_C, 4),
                    "rho_stsb": round(rho_stsb_C, 4), "dims": TOTAL_DIMS}

    # Arm D: random (null)
    print("\n=== arm D (random) ===", flush=True)
    proj_D = project_random(emb_test)
    rho_D = pairwise_rho(proj_D, emb_test)
    print(f"  rho vs embedding: {rho_D:.4f}", flush=True)
    results["D"] = {"name": "random", "rho_embedding": round(rho_D, 4),
                    "rho_stsb": round(stsb_rho(proj_D, tp_filtered, test_sc), 4),
                    "dims": TOTAL_DIMS}

    # Arm E: identity (raw ceiling)
    rho_E = 1.0  # identity is by definition perfect
    results["E"] = {"name": "identity", "rho_embedding": 1.0,
                    "rho_stsb": round(stsb_rho(emb_test, tp_filtered, test_sc), 4),
                    "dims": EMB_DIM}

    # Verdict
    print("\n" + "=" * 60)
    print("V1 — Can a model learn to produce chromavox fields from text?")
    print("=" * 60)
    for arm_id in "ABCDE":
        if arm_id in results:
            r = results[arm_id]
            print(f"  arm {arm_id} ({r['name']:<16}) rho_emb={r['rho_embedding']:+.4f}  "
                  f"rho_stsb={r['rho_stsb']:+.4f}  dims={r['dims']}")
    print("-" * 60)

    g1 = rho_A > 0.50
    g2 = (rho_A - rho_D) > 0.20
    g3 = rho_A >= 0.70 * rho_C
    g4 = rho_B >= 0.50
    print(f"V1-G1  A rho > 0.50:           {'PASS' if g1 else 'FAIL'}  ({rho_A:.4f})")
    print(f"V1-G2  A - D > 0.20:           {'PASS' if g2 else 'FAIL'}  ({rho_A:.4f} vs {rho_D:.4f})")
    print(f"V1-G3  A >= 0.70 × C:          {'PASS' if g3 else 'FAIL'}  ({rho_A:.4f} vs {0.70*rho_C:.4f})")
    print(f"V1-G4  B rho >= 0.50:          {'PASS' if g4 else 'FAIL'}  ({rho_B:.4f})")
    print(f"V1-G5  A vs B:                 A={'better' if rho_A > rho_B else 'worse'}  "
          f"({rho_A:.4f} vs {rho_B:.4f})")
    print("-" * 60)

    if g1 and g2 and g3:
        print("VERDICT: PASS — a model can learn to produce chromavox fields.")
        if g4:
            print("         Information is recoverable (G4).")
        print("         Next: V2 — test semantic superposition with the learned encoder.")
    else:
        print("VERDICT: KILL — learning the constrained mapping is too hard.")
        print("         The model can't discover how to fill chromavox space with meaning.")

    path = os.path.join(OUT, "v1_results.json")
    with open(path, "w") as f:
        json.dump({"arms": results, "gates": {
            "G1": g1, "G2": g2, "G3": g3, "G4": g4,
        }}, f, indent=2)
    print(f"\nresults saved to {path}")


if __name__ == "__main__":
    main()