"""20-step probe on device: rep norms, C-diag, grad norm, loss per step."""
import os, sys
import numpy as np
import torch
import voxel_model as V

dev = sys.argv[1] if len(sys.argv) > 1 else "cuda"
texts, masked, labels = V.load_data()
texts, masked = texts[:128], masked[:128]
tok = V.train_bpe(texts)
MAX = 128
def enc(s):
    out = []
    for w in tok.words(s):
        out += tok.encode(w)
    return out[:MAX]
L = MAX
def P(ids):
    a = np.zeros(L, dtype=np.int64)
    a[:len(ids)] = ids
    return torch.tensor(a, device=dev).unsqueeze(0)

net = V.VoxelNet(V.VOCAB).to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.01)
print("device:", dev, flush=True)
for step in range(20):
    o1 = net(P(enc(texts[step % 128])))
    o2 = net(P(enc(masked[step % 128])))
    n1 = float(o1["rep"].norm()); n2 = float(o2["rep"].norm())
    loss, (d, o) = V.barlow_twins(o1["rep"], o2["rep"])
    opt.zero_grad(); loss.backward()
    gn = float(torch.sqrt(sum((p.grad**2).sum() for p in net.parameters()
                              if p.grad is not None)))
    opt.step()
    print(f"step {step:2d} loss {float(loss):8.4f} d {d:7.3f} o {o:7.4f} "
          f"repnorms {n1:8.3f}/{n2:8.3f} gn {gn:.3e}", flush=True)