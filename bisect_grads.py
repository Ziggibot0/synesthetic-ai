"""Bisect the dead gradient. CPU, 64 rows, 5 steps, per-submodule grads."""
import os
os.environ["VOXEL_DEVICE"] = "cpu"
import numpy as np
import torch
import voxel_model as V

texts, masked, labels = V.load_data()
texts, masked = texts[:64], masked[:64]
tok = V.train_bpe(texts)
enc = lambda s: tok.encode(s)[:96]
L = max(len(enc(t)) for t in texts + masked)

def P(ids):
    a = np.zeros(L, dtype=np.int64)
    a[:len(ids)] = ids
    return torch.tensor(a).unsqueeze(0)

net = V.VoxelNet(V.VOCAB)
opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.01)

reps = [net(P(enc(t)))["rep"] for t in texts[:8]]
print("rep std across 8 texts:", float(torch.cat(reps).std()))

for step in range(5):
    o1 = net(P(enc(texts[step])))
    o2 = net(P(enc(masked[step])))
    loss, _ = V.barlow_twins(o1["rep"], o2["rep"])
    print("loss", step, float(loss), "requires_grad:", loss.requires_grad)
    opt.zero_grad()
    loss.backward()
    if step == 0:
        for name, p in net.named_parameters():
            g = float(p.grad.abs().sum()) if p.grad is not None else -1.0
            print("grad", name, g)
    opt.step()

w = net.pool[0].weight
w0name = [n for n, _ in net.named_parameters()][0]
print("first param delta after 5 steps:",
      float(next(net.parameters()) - torch.tensor(0)).__class__)