"""
show_one_field.py — print one real sentence's journey through the pipeline,
so a human can see the shape of the data at every stage.

Stage 1: the sentence (text)
Stage 2: the 64-dim embedding (nomic, matryoshka-truncated)
Stage 3: the flat 4,608-dim chromavox field (what the model actually writes)
Stage 4: the same field read back as structure: cells with density + colors,
         hue decoded to compass names, grid drawn layer by layer
"""
import json, os
import numpy as np
import torch

import s2_disentangle as s2

HUE_NAMES = [(15, "red"), (45, "orange"), (70, "yellow"), (150, "green"),
             (210, "cyan/blue"), (260, "blue/violet"), (300, "purple"),
             (340, "magenta/pink"), (361, "red")]


def hue_name(cosv, sinv):
    ang = np.degrees(np.arctan2(sinv, cosv)) % 360
    for lim, name in HUE_NAMES:
        if ang < lim:
            return ang, name
    return ang, "?"


def main():
    # same seeded split as the S2 run → pick a held-out sentence we never trained on
    uniq = s2.load_unique_sentences()
    rng = np.random.default_rng(s2.SEED)
    perm = rng.permutation(len(uniq))
    hold = [uniq[i] for i in perm[:400]]

    sent = hold[3]  # arbitrary pick
    print("=" * 72)
    print("STAGE 1 — the sentence")
    print("=" * 72)
    print(f'  "{sent}"'.replace("{sent}", sent))

    print()
    print("=" * 72)
    print("STAGE 2 — its 64-dim embedding (nomic v1.5, matryoshka-truncated)")
    print("=" * 72)
    cache = json.load(open(os.path.join(s2.OUT, "s2_embs.json")))
    e768 = np.array([cache[sent]], dtype=np.float32)
    e64 = s2.matryoshka(e768 := e768 if False else e768)
    np.set_printoptions(precision=3, suppress=True, linewidth=95)
    print(f"  shape: (64,)  -> one list of 64 numbers")
    print(f"  first 16 values: {e64[0][:16]}")
    print(f"   ... (64 numbers total; this is the 'meaning coordinates' everyone\n"
          f"   else uses — it is a point in 64-D space, nothing to look at)")

    print()
    print("=" * 72)
    print("STAGE 3 — the chromavox field as the model writes it (flat)")
    print("=" * 72)
    enc = s2.Encoder().to(s2.DEVICE)
    enc.load_state_dict(torch.load(os.path.join(s2.OUT, "s2_encoder.pt"),
                                   map_location=s2.DEVICE)["encoder"])
    enc.eval()
    field = s2.encode_fields(enc, e64)[0]      # (4608,)
    F = field.view(s2.N_CELLS, s2.FEATS_PER_CELL).cpu()
    print(f"  shape: (4608,)  =  512 cells x 9 numbers per cell")
    print(f"  first 27 numbers (cells 0,1,2 raw):")
    print(f"    {field[:27].cpu().numpy()}")
    print(f"  cell layout [density | hue_cos hue_sin brightness alpha | (slot2 same 4)]:")
    print(f"    cell0: density={F[0,0]:.2f}  slot0=({F[0,1]:+.2f},{F[0,2]:+.2f},b={F[0,3]:.2f},a={F[0,4]:.2f})"
          f"  slot1=({F[0,5]:+.2f},{F[0,6]:+.2f},b={F[0,7]:.2f},a={F[0,8]:.2f})")

    print()
    print("=" * 72)
    print("STAGE 4 — the same field, human-readable: top 8 cells by density")
    print("=" * 72)
    order = torch.argsort(F[:, 0], descending=True)
    print(f"{'cell':>5} {'(x,y,z)':>9} {'density':>8}   {'slot0 hue':>18} {'bri':>4} {'alp':>4}"
          f"   {'slot1 hue':>18} {'bri':>4} {'alp':>4}")
    for c in order[:8]:
        x, y, z = int(c) % 8, (int(c) // 8) % 8, int(c) // 64
        a0, n0 = hue_name(F[c, 1], F[c, 2])
        a1, n1 = hue_name(F[c, 5], F[c, 6])
        print(f"{int(c):>5} ({x},{y},{z})  {F[c,0]:>8.3f}   {a0:>5.0f}deg {n0:<11} {F[c,3]:.2f} {F[c,4]:.2f}"
              f"   {a1:>5.0f}deg {n1:<11} {F[c,7]:.2f} {F[c,8]:.2f}")

    print()
    print("=" * 72)
    print("STAGE 5 — the grid drawn (z=0 layer), density shown 0-9, '.'  = empty")
    print("=" * 72)
    dens = F[:, 0].view(8, 8, 8)
    for z, layer in enumerate(dens[:2]):     # just first two layers
        print(f"  z = {z}")
        for yy in range(8):
            row = "".join(str(min(9, int(float(v) * 10))) if v > 0.05 else "." for v in layer[yy])
            print(f"    {row}")
        if z == 0:
            print("  ... (6 more layers)")

    print()
    lit = (F[:, 0] > 0.05).sum().item()
    print(f"cells meaningfully lit (>0.05): {lit}/512   (last night's pathology: avg 510.6/512)")


if __name__ == "__main__":
    main()