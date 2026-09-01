"""s5a_slot_audit.py — post-S5 audit: how much color info lives in slot-0?

S5's superpose() carried only each stream's SLOT-0 color (2 streams x 3
slots = 6 > the 3-slot cell, so 4 of 6 slot-contents were dropped).
If the S4 scene encoder spreads meaning across all 3 slots, S5's scene
arm was handicapped and the KILL is confounded. If slot-0 carries ~all
the color signal, the truncation was immaterial.

No gates — a measurement on the FROZEN S4 encoder (results/s4_encoders.pt).
"""
import numpy as np, torch, importlib.util
import s2_disentangle as s2
import s4_scene_encoder as s4

spec = importlib.util.spec_from_file_location("s5", "s5_superposition.py")
s5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s5)

SEED = 1337
DEVICE = s2.DEVICE

ckpt = torch.load("results/s4_encoders.pt", map_location="cpu")
enc = s4.Encoder(k=32)
enc.load_state_dict(ckpt["D"]); enc.to(DEVICE).eval()

rng = np.random.default_rng(SEED)
uniq = s2.load_unique_sentences()
perm = rng.permutation(len(uniq))
hold_sents = [uniq[i] for i in perm[:400]]
hold = s2.matryoshka(s2.embed_sentences(hold_sents)).astype(np.float32)

with torch.no_grad():
    x = torch.tensor(hold, dtype=torch.float32, device=DEVICE)
    f = enc(x).reshape(400, 512, 10).cpu().numpy()

lit = f[:, :, 0] > 0.01

# per-slot occupancy inside lit cells
for s, (a, b) in enumerate([(1, 4), (4, 7), (7, 10)]):
    occ = (f[:, :, a:b] > 0.01).any(axis=-1) & lit
    print(f"slot-{s}: lit-cells carrying content: {occ.sum()}/{lit.sum()} "
          f"({100*occ.sum()/max(1,lit.sum()):.1f}%)")

# pairwise-structure rho per slot view
rng_p = np.random.default_rng(SEED)
pset = set()
while len(pset) < 2000:
    i, j = int(rng_p.integers(0, 400)), int(rng_p.integers(0, 400))
    if i != j:
        pset.add((min(i, j), max(i, j)))
pairs = sorted(pset)

def rho_view(view, pairs):
    v = view.reshape(view.shape[0], -1)          # flatten cells x feats
    ii = np.array([i for i, _ in pairs]); jj = np.array([j for _, j in pairs])
    dA = np.linalg.norm(v[ii] - v[jj], axis=1)
    dB = np.linalg.norm(hold[ii] - hold[jj], axis=1)
    from scipy.stats import spearmanr
    r, _ = spearmanr(dA, dB)
    return float(r)

views = {
    "salience-only": f[:, :, 0:1],
    "slot-0 only": f[:, :, 1:4],
    "slot-1 only": f[:, :, 4:7],
    "slot-2 only": f[:, :, 7:10],
    "all slots": f[:, :, 1:10],
}
print()
for name, v in views.items():
    print(f"{name:<14} rho vs true = {rho_view(v, pairs):+.4f}")