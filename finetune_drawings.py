"""
finetune_drawings.py - supervised fine-tune of the voxel model on HUMAN data.

STATUS: PARKED (2026-08-30, user decision). Sean: his personal color
mappings could CORRUPT a stable learned representation - the goal is to
see whether the architecture develops synesthesia-like structure on its
own, not a carbon copy of HIS colors (which are themselves one learned
instantiation). Fine-tune never run; kept because the head-supervision
machinery may be useful later, and drawings remain useful as EVAL data
(human 0-1 scores / painted comparisons), not training data.

Original design (if ever revived): uses results/drawings/*.json (probe
text + painted cells) as supervision:

  * pos_head: soft cross-entropy toward the drawing's dominant cell
    (brightest mean lux across its colors; the architecture currently
    emits ONE argmax cell per thought - multi-cell distribution is a
    model-side gap we bake in during the retraining round, not here)
  * slot_head: for that cell's drawn colors, match the most active slots
    (best of both slot assignments when 2 colors), losses = circular hue +
    sat/bri MSE + activation pull toward 1 for used slots

Collapse guard: VicReg-style batch regularizer (variance hinge + covariance)
on the reps of the drawing batches. The old corpus invariance term is NOT
wired here (v1 = heads only, batch-only regularizer) - the corpus retrain
is a separate round. Replaces, in spirit, the hardcoded ANCHOR_WORDS table
with real data from a person with synesthesia.

Run:  py -3.12 finetune_drawings.py            (from the repo folder)
      py -3.12 finetune_drawings.py --steps 800 --lr 1e-4 --setup free
Writes results/model_finetuned.pt + results/history_finetune.json and
prints before/after anchor probes. Nothing runs at import.
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

GATES = {  # pre-registered, printed with the summary
    "fit_mean_hue_deg": 25.0,      # final circular hue error, degrees
    "no_collapse_mean_cos": 0.90,  # rep mean-pairwise-cosine must stay below
}


def build_model(setup: str):
    import torch
    import voxel_model as VM, bpe_pure
    b = bpe_pure.Bpe(VM.VOCAB)
    with open(os.path.join(HERE, "bpe_cache.json")) as f:
        d = json.load(f)
    b.w2i = {k: int(v) for k, v in d["w2i"].items()}
    b.merges = [tuple(m) for m in d["merges"]]
    net = VM.VoxelNet(VM.VOCAB)
    ck = os.path.join(HERE, "results", f"model_{setup}.pt")
    net.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True))
    return b, net


def hcirc_deg(h1, h2):
    """Circular shortest arc between two hue angles, in degrees."""
    return abs((h1 - h2 + 180.0) % 360.0 - 180.0)


def load_drawings():
    recs = []
    for p in sorted(glob.glob(os.path.join(HERE, "results", "drawings", "*.json"))):
        with open(p, encoding="utf-8") as f:
            rec = json.load(f)
        cells = rec.get("cells") or rec          # tolerate pre-metadata format
        text = (rec.get("text") or "").strip()
        parsed = []
        for k, cols in cells.items():
            xyz = [int(v) for v in k.split(",")]
            cols = [[float(a), float(sb), float(c)] for a, sb, c in cols][:2]
            if cols:
                parsed.append((xyz, cols))
        if text and parsed:
            recs.append({"text": text, "cells": parsed,
                         "name": os.path.splitext(os.path.basename(p))[0]})
    return recs


def dominant_cell(cells):
    """One argmax cell per thought (v1 architecture limit)."""
    return max(cells, key=lambda c: float(np.mean([col[2] for col in c[1]])))


def encode(b, text, MAXLEN=128):
    import numpy as np
    ids = []
    for w in b.words(text):
        ids += b.encode(w or "MSK")
    ids = ids[:MAXLEN]
    arr = np.zeros(MAXLEN, dtype=np.int64)
    arr[:len(ids)] = ids
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", default="free", choices=["free", "distill", "anchor"])
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    recs = load_drawings()
    if not recs:
        print("No drawings found in results/drawings/ - paint some in the "
              "viewer first (draw mode -> save). Nothing to do.")
        return
    print(f"fine-tune corpus: {len(recs)} drawings")

    import torch
    import torch.nn.functional as F
    import voxel_model as VM
    b, net = build_model(args.setup)
    dev = os.environ.get("VOXEL_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)

    X = torch.stack([torch.tensor(encode(b, r["text"])) for r in recs]).to(dev)
    net = net.to(dev).train()
    targets = []                                  # (xyz, [(h,s,l), ...<=2])
    for r in recs:
        xyz, cols = dominant_cell(r["cells"])
        targets.append((xyz, cols))
    print("regime: batch-only VicReg regularizer (corpus retrain is a "
          "separate, later run)")

    def batch_reg(rep):
        if rep.size(0) < 2:     # tiny-N guard: var/std over 1 sample = NaN
            return torch.tensor(0.0, device=rep.device)
        std = torch.sqrt(rep.var(0) + 1e-4)
        var = F.relu(1.0 - std).mean()
        rn = (rep - rep.mean(0)) / (std + 1e-4)
        N, D = rep.shape
        c = (rn.T @ rn) / N
        off = (c.pow(2).sum() - c.diag().pow(2).sum()) / (D * (D - 1))
        return 8.0 * var + 3.0 * off

    def anchor_hues(words=("green", "red", "blue")):
        with torch.no_grad():
            hs = []
            for w in words:
                o = net(torch.tensor(encode(b, w)).unsqueeze(0).to(dev))
                top = int((o["bri"] * o["act"])[0].argmax())
                hs.append(float(o["hue"][0, top]))
        return hs

    def probe_report(tag):
        want = {"green": 120, "red": 0, "blue": 240}
        hs = anchor_hues()
        print(f"anchor probes {tag}: " +
              " · ".join(f"'{w}' {h:.0f} (want ~{want[w]})"
                         for w, h in zip(want, hs)))

    def slot_loss_bi(out, bi, cols):
        """All-tensor hue loss (keeps gradients); circular diff."""
        act = out["bri"][bi] * out["act"][bi]
        top = act.argsort(descending=True)[:len(cols)]
        hue_t = out["hue"][bi]                    # [K] degrees
        hue_sel = hue_t[top]
        tgt = torch.tensor([c[0] for c in cols], device=hue_sel.device)
        cd = ((hue_sel - tgt + 180.0) % 360.0 - 180.0) / 180.0
        l = (cd ** 2).sum()
        l = l + sum((out["sat"][bi, s] - c[1]) ** 2 +
                    (out["bri"][bi, s] - c[2]) ** 2 +
                    (1.0 - out["act"][bi, s]) ** 2
                    for s, c in zip(top.tolist(), cols))
        return l

    hist, t0 = [], time.time()
    probe_report("BEFORE")

    for step in range(1, args.steps + 1):
        idx = torch.randperm(len(recs))[:args.batch]
        ids = X[idx]
        out = net(ids)
        loss = batch_reg(out["rep"])
        with torch.no_grad():                     # collapse monitor
            rr = out["rep"]
            if rr.size(0) >= 2:
                rrn = (rr - rr.mean(0)) / (rr.std(0) + 1e-8)
                mpc = float((rrn @ rrn.T).mean())
            else:
                mpc = float("nan")                # undefined at batch=1

        p_loss = torch.tensor(0.0, device=dev)
        s_loss = torch.tensor(0.0, device=dev)
        for bi in range(len(idx)):
            xyz, cols = targets[idx[bi]]
            tgt = torch.zeros(3, VM.GRID, device=dev)
            for ax in range(3):
                tgt[ax, xyz[ax]] = 1.0
            p_loss = p_loss + sum(
                -(torch.log_softmax(out["pos_logits"][bi, ax], 0)
                  * tgt[ax]).sum() for ax in range(3))
            if cols:
                s_loss = s_loss + slot_loss_bi(out, bi, cols)
        loss = loss + p_loss / len(idx) + 0.5 * s_loss / len(idx)
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 25 == 0 or step == 1:
            hist.append(dict(step=step, loss=float(loss.detach()),
                             pos=float(p_loss.detach()),
                             slot=float(s_loss.detach()), rep_mpc=mpc,
                             elapsed=time.time() - t0))
            print(f"step {step:4d} L={float(loss):.3f} pos={float(p_loss):.3f} "
                  f"slot={float(s_loss):.3f} mpc={mpc:.3f}")

    net.eval()
    probe_report("AFTER")
    # final fit: circular hue error vs each drawing's dominant color
    errs = []
    with torch.no_grad():
        for i in range(len(recs)):
            o = net(X[i:i + 1])
            xyz, cols = targets[i]
            top = int((o["bri"] * o["act"])[0].argmax())
            hue = float(o["hue"][0, top])
            if cols:
                errs.append(abs((hue - cols[0][0] + 180.0) % 360.0 - 180.0))
    fit = float(np.mean(errs)) if errs else float("nan")
    print(f"\nfinal mean hue error vs drawings: {fit:.1f} deg "
          f"(gate <= {GATES['fit_mean_hue_deg']})")
    print(f"final rep mpc: {mpc:.3f} (gate < {GATES['no_collapse_mean_cos']})")
    torch.save(net.state_dict(),
               os.path.join(HERE, "results", "model_finetuned.pt"))
    with open(os.path.join(HERE, "results", "history_finetune.json"), "w") as f:
        json.dump({"setup": args.setup, "steps": args.steps, "lr": args.lr,
                   "n_drawings": len(recs), "hist": hist, "fit_hue_deg": fit,
                   "final_mpc": mpc}, f, indent=1)
    print("saved results/model_finetuned.pt + history_finetune.json")


if __name__ == "__main__":
    main()