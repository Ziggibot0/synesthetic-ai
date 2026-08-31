"""
v0_semantic_gate.py — Can chromavox space hold semantic structure?

Pre-registered design: docs/v0_semantic_gate.md

WHAT IT TESTS:
  Project a known-good semantic embedding (matryoshka nomic v1.5, 64-dim)
  into the constrained chromavox field (512 cells, bounded color-sets,
  2-sig-fig quantization). Measure how much semantic structure survives.

  PASS = the chromavox space can hold semantics. KILL = it can't.

ARMS (pre-registered):
  A  voxel-full      64-dim → chromavox field (density + color-set, quantized)
  B  voxel-density   64-dim → chromavox field (density only, no color)
  C  unconstrained   64-dim → 4,608-dim linear (no constraints, same dims)
  D  random          64-dim → random projection to 4,608 dims (null)
  E  identity        64-dim → 64-dim (no projection, ceiling)
  F  voxel-quant     64-dim → chromavox field WITHOUT quantization (continuous)

GATES:
  V0-G1  A rho vs embedding > 0.50
  V0-G2  A > D (random) by > 0.20
  V0-G3  A >= 0.70 × E (identity ceiling)
  V0-G4  A > B (color channels beat density-only)
  V0-G5  F - A < 0.10 (quantization is not the bottleneck)

USAGE:
  py -3.12 v0_semantic_gate.py              # run all arms, print verdict
  py -3.12 v0_semantic_gate.py --arms A C D E  # run subset

CPU-only. ~5 min total (nomic embedding is the slow part).
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
GRID = 8
N_CELLS = GRID ** 3       # 512
N_VALUE_BINS = 16
K_SLOTS = 2               # max color-set entries per cell

# chromavox field dims per cell: 1 density + K×4 color features
# color features per entry: hue_cos, hue_sin, brightness, alpha
FEATURES_PER_CELL = 1 + K_SLOTS * 4    # 9
TOTAL_DIMS = N_CELLS * FEATURES_PER_CELL  # 4,608

# Matryoshka truncation
EMB_DIM = 64


def _patch_torch_distributed():
    """Windows single-process patch for laion-clap / torch.distributed."""
    try:
        import torch.distributed as dist
        for _n in ['group', 'ReduceOp', 'broadcast', 'all_reduce',
                   'all_gather', 'reduce_scatter', 'barrier']:
            if not hasattr(dist, _n):
                setattr(dist, _n, lambda *a, **k: None)
        _g = dist.group
        if not hasattr(_g, 'WORLD'):
            _g.WORLD = None
        if not hasattr(dist, 'ReduceOp') or not hasattr(dist.ReduceOp, 'SUM'):
            class _RO:
                SUM = 0; PRODUCT = 1; MIN = 2; MAX = 3
            dist.ReduceOp = _RO
        import torch.distributed.nn as _dnn
        _dnn.broadcast = lambda *a, **k: None
        _dnn.all_reduce = lambda *a, **k: None
        _dnn.all_gather = lambda *a, **k: None
    except Exception:
        pass


def matryoshka_truncate(emb: np.ndarray, dim: int = EMB_DIM) -> np.ndarray:
    """Truncate to `dim` dims with layer-norm (matryoshka recipe)."""
    from sklearn.preprocessing import StandardScaler
    truncated = emb[:, :dim]
    # layer-norm per sample
    mean = truncated.mean(axis=1, keepdims=True)
    std = truncated.std(axis=1, keepdims=True) + 1e-8
    return (truncated - mean) / std


def fetch_stsb(max_sentences: int = 2000):
    """Fetch STS-B (Semantic Textual Similarity Benchmark) from HuggingFace.

    Returns (sentences, pairs, scores) where:
      sentences: list of unique sentences
      pairs: list of (i, j) indices into sentences
      scores: list of human similarity scores (0-5)
    """
    cache = os.path.join(OUT, "stsb_cache.json")
    if os.path.exists(cache):
        d = json.load(open(cache))
        return d["sentences"], [tuple(p) for p in d["pairs"]], d["scores"]

    # STS-B via HuggingFace datasets (sentence-transformers/stsb)
    try:
        from datasets import load_dataset
        ds = load_dataset("sentence-transformers/stsb", split="test")
        sentences = []
        sent_idx = {}
        pairs = []
        scores = []
        for row in ds:
            s1, s2 = row["sentence1"], row["sentence2"]
            score = float(row["score"])
            for s in (s1, s2):
                if s not in sent_idx:
                    sent_idx[s] = len(sentences)
                    sentences.append(s)
            pairs.append((sent_idx[s1], sent_idx[s2]))
            scores.append(score)
    except Exception as e:
        sys.exit(f"could not load STS-B: {e}. pip install datasets")

    sentences = sentences[:max_sentences]
    # filter pairs to those within range
    pairs = [(i, j) for i, j in pairs if i < max_sentences and j < max_sentences]
    json.dump({"sentences": sentences, "pairs": [[p[0], p[1]] for p in pairs],
                "scores": scores}, open(cache, "w"))
    return sentences, pairs, scores


def embed_sentences(sentences: list[str]) -> np.ndarray:
    """Embed sentences with nomic-embed-text v1.5 via Ollama, then
    matryoshka-truncate to EMB_DIM."""
    cache = os.path.join(OUT, "v0_nomic_768.npy")
    if os.path.exists(cache):
        emb768 = np.load(cache)
        if emb768.shape[0] >= len(sentences):
            return matryoshka_truncate(emb768[:len(sentences)], EMB_DIM)

    import subprocess
    # Use Ollama's nomic-embed-text:v1.5 (always returns 768-dim)
    model_name = "nomic-embed-text:v1.5"
    embs = []
    batch = 64
    t0 = time.time()
    for i in range(0, len(sentences), batch):
        batch_sents = sentences[i:i+batch]
        import urllib.request, json as _json
        results = []
        for s in batch_sents:
            payload = _json.dumps({"model": model_name, "prompt": s}).encode()
            req = urllib.request.Request(
                "http://localhost:11434/api/embeddings",
                data=payload,
                headers={"Content-Type": "application/json"})
            resp = urllib.request.urlopen(req, timeout=30)
            data = _json.loads(resp.read())
            results.append(data["embedding"])
        embs.extend(results)
        if (i + batch) % 256 == 0 or i + batch >= len(sentences):
            el = time.time() - t0
            print(f"  embedded {min(i+batch, len(sentences))}/{len(sentences)} "
                  f"({el:.0f}s)", flush=True)

    emb768 = np.array(embs, dtype=np.float32)
    np.save(cache, emb768)
    return matryoshka_truncate(emb768, EMB_DIM)


# --- Projections ---

def project_voxel_full(emb: np.ndarray, quantize: bool = True) -> np.ndarray:
    """Arm A/F: 64-dim → chromavox field (density + color-set, constrained).

    Linear projection from EMB_DIM to TOTAL_DIMS, then reshape to
    (N_CELLS, FEATURES_PER_CELL), apply constraints via sigmoid/cos-sin
    (baked into forward pass), optionally quantize to 2 sig figs.
    """
    # learn a linear projection (ridge regression to identity-ish)
    # but we don't have targets — so just use random orthogonal projection
    # and measure whether structure survives constraints
    # Actually: the projection should be LEARNED to preserve distances.
    # Simplest: orthogonal random projection (Johnson-Lindenstrauss).
    # But we want to LEARN the best projection, so use gradient descent
    # to minimize ||dist(proj(x)) - dist(x)||^2 on a train set.
    # For simplicity and CPU speed: use SVD-based orthogonal projection.
    # An orthogonal projection preserves distances exactly in unconstrained
    # space; the question is what the constraints cost.
    # So: random orthogonal projection, then apply constraints.

    rng = np.random.default_rng(1337)
    # Orthogonal projection from 64 to 4608 dims
    # QR of (4608, 64) gives Q of shape (4608, 64), then emb @ Q.T = (N, 4608)
    rand_mat = rng.standard_normal((TOTAL_DIMS, EMB_DIM))
    Q, _ = np.linalg.qr(rand_mat)  # Q: (TOTAL_DIMS, EMB_DIM) with orthonormal columns
    projected = emb @ Q.T  # (N, TOTAL_DIMS)

    # reshape to (N, N_CELLS, FEATURES_PER_CELL)
    field = projected.reshape(-1, N_CELLS, FEATURES_PER_CELL)

    # apply constraints
    # dim 0: density → sigmoid → [0,1]
    density = 1.0 / (1.0 + np.exp(-field[:, :, 0]))

    # dims 1-4: color entry 1 (hue_cos, hue_sin, brightness, alpha)
    # dims 5-8: color entry 2
    out = np.zeros_like(field)
    out[:, :, 0] = density
    for k in range(K_SLOTS):
        base = 1 + k * 4
        # hue: cos/sin of learned angle
        angle = field[:, :, base]  # use first dim as angle
        out[:, :, base] = np.cos(angle)     # hue_cos
        out[:, :, base + 1] = np.sin(angle)  # hue_sin
        # brightness: sigmoid
        out[:, :, base + 2] = 1.0 / (1.0 + np.exp(-field[:, :, base + 2]))
        # alpha: sigmoid
        out[:, :, base + 3] = 1.0 / (1.0 + np.exp(-field[:, :, base + 3]))

    if quantize:
        # 2 sig fig quantization
        out = np.round(out * 100) / 100

    return out.reshape(emb.shape[0], -1)  # flatten to (N, TOTAL_DIMS)


def project_voxel_density(emb: np.ndarray, quantize: bool = True) -> np.ndarray:
    """Arm B: 64-dim → chromavox field (density only, no color)."""
    rng = np.random.default_rng(1337)
    rand_mat = rng.standard_normal((N_CELLS, EMB_DIM))
    Q, _ = np.linalg.qr(rand_mat)
    density = 1.0 / (1.0 + np.exp(-(emb @ Q.T)))  # (N, N_CELLS)
    if quantize:
        density = np.round(density * 100) / 100
    # pad to TOTAL_DIMS with zeros
    out = np.zeros((emb.shape[0], TOTAL_DIMS), dtype=np.float32)
    out[:, :N_CELLS] = density  # density in first N_CELLS dims
    return out


def project_unconstrained(emb: np.ndarray) -> np.ndarray:
    """Arm C: 64-dim → 4,608-dim linear (no constraints)."""
    rng = np.random.default_rng(1337)
    rand_mat = rng.standard_normal((TOTAL_DIMS, EMB_DIM))
    Q, _ = np.linalg.qr(rand_mat)
    return (emb @ Q.T).astype(np.float32)


def project_random(emb: np.ndarray) -> np.ndarray:
    """Arm D: 64-dim → random projection to 4,608 dims (null)."""
    rng = np.random.default_rng(42)  # different seed
    return rng.standard_normal((emb.shape[0], TOTAL_DIMS)).astype(np.float32)


def project_identity(emb: np.ndarray) -> np.ndarray:
    """Arm E: 64-dim → 64-dim (no projection, ceiling)."""
    return emb.astype(np.float32)


# --- Metrics ---

def pairwise_rho(projected: np.ndarray, reference: np.ndarray,
                 n_pairs: int = 2000, seed: int = 1337) -> float:
    """Spearman rho of pairwise distances: projected vs reference."""
    n = projected.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(n_pairs, n * (n - 1) // 2), replace=False)
    # sample pairs
    pairs = []
    for _ in range(min(n_pairs, n * (n - 1) // 2)):
        i, j = rng.integers(0, n, size=2)
        if i != j:
            pairs.append((i, j))
    pairs = np.array(pairs)

    # distances
    d_proj = np.linalg.norm(projected[pairs[:, 0]] - projected[pairs[:, 1]], axis=1)
    d_ref = np.linalg.norm(reference[pairs[:, 0]] - reference[pairs[:, 1]], axis=1)

    rho, _ = spearmanr(d_proj, d_ref)
    return float(rho)


def stsb_rho(projected: np.ndarray, pairs: list, scores: list) -> float:
    """Spearman rho of projected distances vs STS-B human scores."""
    if not pairs:
        return float('nan')
    # filter to pairs within range and zip with scores
    valid = [(i, j, s) for (i, j), s in zip(pairs, scores)
             if i < projected.shape[0] and j < projected.shape[0]]
    if not valid:
        return float('nan')
    d = np.array([np.linalg.norm(projected[i] - projected[j]) for i, j, _ in valid])
    s = np.array([sc for _, _, sc in valid])
    rho, _ = spearmanr(d, s)
    return float(rho)


# --- Main ---

ARMS = {
    "A": ("voxel-full", project_voxel_full, {}),
    "B": ("voxel-density", project_voxel_density, {}),
    "C": ("unconstrained", project_unconstrained, {}),
    "D": ("random", project_random, {}),
    "E": ("identity", project_identity, {}),
    "F": ("voxel-quant", project_voxel_full, {"quantize": False}),
}

GATES = {
    "V0-G1": ("A rho > 0.50", 0.50),
    "V0-G2": ("A - D > 0.20", 0.20),
    "V0-G3": ("A >= 0.70 × E", 0.70),
    "V0-G4": ("A > B", 0.0),
    "V0-G5": ("F - A < 0.10", 0.10),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=list(ARMS.keys()),
                    help="subset of arms to run")
    ap.add_argument("--n-sentences", type=int, default=2000)
    ap.add_argument("--n-pairs", type=int, default=2000)
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)

    # 1. Fetch STS-B
    print("=== fetching STS-B ===", flush=True)
    sentences, stsb_pairs, stsb_scores = fetch_stsb(a.n_sentences)
    print(f"  {len(sentences)} sentences, {len(stsb_pairs)} annotated pairs",
          flush=True)

    # 2. Embed
    print("=== embedding with nomic-embed-text v1.5 ===", flush=True)
    emb = embed_sentences(sentences)
    print(f"  embeddings shape: {emb.shape}", flush=True)

    # 3. Run arms
    results = {}
    for arm_id in a.arms:
        name, func, kwargs = ARMS[arm_id]
        print(f"=== arm {arm_id} ({name}) ===", flush=True)
        t0 = time.time()
        projected = func(emb, **kwargs)
        rho_emb = pairwise_rho(projected, emb, n_pairs=a.n_pairs)
        rho_stsb = stsb_rho(projected, stsb_pairs, stsb_scores)
        print(f"  rho vs embedding: {rho_emb:.4f}", flush=True)
        print(f"  rho vs STS-B:     {rho_stsb:.4f}", flush=True)
        print(f"  ({time.time()-t0:.1f}s)", flush=True)
        results[arm_id] = {
            "name": name, "rho_embedding": round(rho_emb, 4),
            "rho_stsb": round(rho_stsb, 4), "dims": projected.shape[1],
        }

    # 4. Gates
    print("\n" + "=" * 60)
    print("V0 — Can chromavox space hold semantic structure?")
    print("=" * 60)
    for arm_id, r in results.items():
        print(f"  arm {arm_id} ({r['name']:<16}) rho_emb={r['rho_embedding']:+.4f}  "
              f"rho_stsb={r['rho_stsb']:+.4f}  dims={r['dims']}")
    print("-" * 60)

    if "A" in results and "D" in results:
        g1 = results["A"]["rho_embedding"] > GATES["V0-G1"][1]
        g2 = (results["A"]["rho_embedding"] - results["D"]["rho_embedding"]) > GATES["V0-G2"][1]
        print(f"V0-G1  A rho > 0.50:           {'PASS' if g1 else 'FAIL'}  ({results['A']['rho_embedding']:.4f})")
        print(f"V0-G2  A - D > 0.20:           {'PASS' if g2 else 'FAIL'}  ({results['A']['rho_embedding']:.4f} vs {results['D']['rho_embedding']:.4f})")
    if "A" in results and "E" in results:
        g3 = results["A"]["rho_embedding"] >= GATES["V0-G3"][1] * results["E"]["rho_embedding"]
        print(f"V0-G3  A >= 0.70 × E:          {'PASS' if g3 else 'FAIL'}  ({results['A']['rho_embedding']:.4f} vs {0.70*results['E']['rho_embedding']:.4f})")
    if "A" in results and "B" in results:
        g4 = results["A"]["rho_embedding"] > results["B"]["rho_embedding"]
        print(f"V0-G4  A > B (color pays):     {'PASS' if g4 else 'FAIL'}  ({results['A']['rho_embedding']:.4f} vs {results['B']['rho_embedding']:.4f})")
    if "A" in results and "F" in results:
        g5 = (results["F"]["rho_embedding"] - results["A"]["rho_embedding"]) < GATES["V0-G5"][1]
        print(f"V0-G5  F - A < 0.10 (quant):   {'PASS' if g5 else 'FAIL'}  ({results['F']['rho_embedding']-results['A']['rho_embedding']:.4f})")

    print("-" * 60)
    # Overall verdict
    if "A" in results and "D" in results:
        if g1 and g2:
            print("VERDICT: PASS — chromavox space can hold semantic structure.")
            if "E" in results and g3:
                print("         Most structure preserved (G3).")
            if "B" in results and g4:
                print("         Color channels earn their keep (G4).")
            print("         Next: train a model to PRODUCE chromavox fields from text.")
        else:
            print("VERDICT: KILL — chromavox constraints destroy semantic structure.")
            print("         The representation is fundamentally broken for semantics.")

    # Save results
    path = os.path.join(OUT, "v0_results.json")
    with open(path, "w") as f:
        json.dump({"arms": results, "gates": {
            k: v[0] for k, v in GATES.items()
        }}, f, indent=2)
    print(f"\nresults saved to {path}")


if __name__ == "__main__":
    main()