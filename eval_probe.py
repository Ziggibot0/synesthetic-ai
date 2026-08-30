"""eval_probe.py — cheap tests on the trained voxel model.
For each setup's checkpoint (results/model_<setup>.pt):

  A. Color-structure probe (the core cheap test):
     rep = (top-2 slots' hues/sat/lum/act + slot amplitudes) -> logistic
  B. Full 64-dim rep probe
  C. Baseline: raw token-count vector, NO learned rep
  All three under leave-one-hue-out logistic regression (AUC per class).
  Saves results/probe_<setup>.json + reps_<setup>.npy for the visualizer.

Run: py -3.12 eval_probe.py [setup ...]   (default: all three setups)
"""
import json, os, sys
import numpy as np
import torch

import voxel_model as V

OUT = V.OUT
DEV = os.environ.get("VOXEL_DEVICE") or (
    "cuda" if torch.cuda.is_available() else "cpu")

texts, masked, labels = V.load_data()
tok = V.train_bpe(texts)
MAXLEN = 128
_wc = {}

def enc(s):
    out = []
    for w in tok.words(s):
        if w not in _wc:
            _wc[w] = tok.encode(w)
        out += _wc[w]
    return out

starts = []
for t, m in zip(texts, masked):
    a, b = enc(t), enc(m)
    d = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), 0)
    starts.append(max(0, d - 32))

def encw(ids, start):
    if len(ids) <= MAXLEN:
        return ids
    s0 = max(0, min(start, len(ids) - MAXLEN))
    return ids[s0:s0 + MAXLEN]

def P(ids):
    a = np.zeros(MAXLEN, dtype=np.int64)
    a[:len(ids)] = ids
    return torch.tensor(a, device=DEV)

X = [P(encw(enc(t), s)) for t, s in zip(texts, starts)]

def extract(setup):
    ckpt = os.path.join(OUT, f"model_{setup}.pt")
    if not os.path.exists(ckpt):
        return None
    net = V.VoxelNet(V.VOCAB).to(DEV)
    net.load_state_dict(torch.load(ckpt, map_location=DEV))
    net.eval()
    reps, feats = [], []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            o = net(torch.stack(X[i:i + 256]))
            reps.append(o["rep"].cpu().numpy())
            sal = (o["bri"] * o["act"]).cpu().numpy()      # [B,K]
            top2 = np.argsort(-sal, axis=1)[:, :2]          # top-2 slots
            f = []
            for b in range(o["rep"].size(0)):
                row = []
                for k in top2[b]:
                    row += [float(o["hue"][b, k]) / 360.0,
                            float(o["sat"][b, k]),
                            float(o["bri"][b, k]),
                            float(o["act"][b, k])]
                f.append(row)
            feats.append(np.array(f, dtype=np.float32))
    return np.vstack(reps), np.vstack(feats)

def split_auc(F, y_labels, classes, seed=1337):
    """70/30 row split; per-class one-vs-rest logistic AUC on the
    untouched 30%."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(F))
    tr, te = idx[:int(0.7 * len(F))], idx[int(0.7 * len(F)):]
    y = np.array(y_labels)
    aucs = {}
    for cls in classes:
        ytr, yte = y[tr] == cls, y[te] == cls
        if yte.sum() < 5 or ytr.sum() < 5 or (~ytr).sum() < 5:
            aucs[str(cls)] = None
            continue
        sc = StandardScaler().fit(F[tr])
        lr = LogisticRegression(max_iter=2000)
        lr.fit(sc.transform(F[tr]), ytr)
        p = lr.predict_proba(sc.transform(F[te]))[:, 1]
        aucs[str(cls)] = round(float(roc_auc_score(yte, p)), 3)
    return aucs

def token_baseline():
    """Baseline: bag-of-token-counts, no learned rep."""
    F = np.zeros((len(texts), V.VOCAB + 3), dtype=np.float32)
    for i, s in enumerate(texts):
        for t in enc(s):
            F[i, t] += 1.0
    return F

def main(setups):
    cls_counts = {}
    for l in labels:
        cls_counts[l] = cls_counts.get(l, 0) + 1
    classes = [c for c, n in cls_counts.items() if n >= 30]
    print("classes (n>=30):", classes, flush=True)
    base_aucs = split_auc(token_baseline(), np.array(labels), classes)
    print("BASELINE token-count 70/30 AUC:", base_aucs, flush=True)
    results = {"classes": classes, "baseline": base_aucs}
    for s in setups:
        r = extract(s)
        if r is None:
            print(f"[{s}] no checkpoint, skipping"); continue
        reps, feats = r
        np.save(os.path.join(OUT, f"reps_{s}.npy"), reps)
        a = split_auc(feats, np.array(labels), classes)
        b = split_auc(reps, np.array(labels), classes)
        results[s] = {"color_top2": a, "rep64": b}
        print(f"[{s}] color-structure AUC:", a, flush=True)
        print(f"[{s}] rep-64 AUC      :", b, flush=True)
    with open(os.path.join(OUT, "probe_results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("saved -> results/probe_results.json", flush=True)

if __name__ == "__main__":
    which = sys.argv[1:] or ["free", "distill", "anchor"]
    main(which)