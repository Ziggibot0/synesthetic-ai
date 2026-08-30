"""
calc_train.py — train + evaluate one arm of the calc gate.

Usage:
  py -3.12 calc_data.py                 # generate data (once)
  py -3.12 calc_train.py --arm A --epochs 60
  py -3.12 calc_train.py --arm D --epochs 60
  ... (repeat for B, C)

Writes per-arm: results/calc_<arm>.json  (metrics + config + params)
                results/calc_<arm>.pt    (checkpoint)

Metrics (all on the normalized derivative field, per calc_gate.md):
  rel_err   : mean |pred - true| / |true| over positions with |true|>0.05
              (G1 gate: arm A rel_err < 0.15 on test_in)
  exact     : fraction of positions where predicted bin == true bin
  extrap    : rel_err on test_extrap (disjoint families + wider domain)
              (G2 gate: arm A >= arm D here)

The model is deterministic given --seed; set it to reproduce.
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import calc_data as cd
import calc_model as cm

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def load_data():
    p = os.path.join(OUT, "calc_data.npz")
    d = np.load(p)
    def split(prefix):
        return {k[len(prefix):]: d[k] for k in d.files if k.startswith(prefix)}
    return split("tr_"), split("ti_"), split("tx_")


def field_batch(d, idx, arm):
    """Build a FieldModel batch from a data dict + row indices."""
    B = len(idx)
    x = {
        "value_bin": torch.tensor(d["value_bin"][idx], dtype=torch.long),
        "density": torch.tensor(d["density"][idx], dtype=torch.float32),
        "hue": torch.tensor(d["hue"][idx], dtype=torch.float32),
        "brightness": torch.tensor(d["brightness"][idx], dtype=torch.float32),
        "alpha": torch.tensor(d["alpha"][idx], dtype=torch.float32),
        "occupancy": torch.tensor((d["density"][idx] > 0).astype(np.float32)),
    }
    y_bin = torch.tensor(d["deriv_bin"][idx], dtype=torch.long)
    y_dens = torch.tensor(d["deriv_density"][idx], dtype=torch.float32)
    return x, y_bin, y_dens


def token_batch(d, idx):
    """Build a TokenModel batch: src = value tokens, tgt = deriv tokens."""
    B = len(idx)
    src = np.array([[cm._value_to_token(v) for v in row] for row in d["fine_value"][idx]])
    tgt = np.array([[cm._value_to_token(v) for v in row] for row in d["deriv_value"][idx]])
    return (torch.tensor(src, dtype=torch.long),
            torch.tensor(tgt, dtype=torch.long))


def field_metrics(model, d, idx, arm):
    """Return (rel_err, exact) for FieldModel on rows idx."""
    model.eval()
    rel_errs, exacts = [], []
    with torch.no_grad():
        for i in range(0, len(idx), 256):
            b = idx[i:i + 256]
            x, y_bin, _ = field_batch(d, b, arm)
            x = {k: v.to(DEVICE) for k, v in x.items()}
            y_bin = y_bin.to(DEVICE)
            logits, _ = model(x)
            pred = logits.argmax(-1)
            # predicted continuous value = bin center
            pred_v = (pred.float() + 0.5) / cm.N_VALUE_BINS * 2 - 1
            true_v = torch.tensor(d["deriv_value"][b], device=DEVICE)
            mask = true_v.abs() > 0.05
            if mask.any():
                rel_errs.append(((pred_v - true_v).abs() / true_v.abs())[mask].mean().item())
            exacts.append((pred == y_bin).float().mean().item())
    return float(np.mean(rel_errs)), float(np.mean(exacts))


def token_metrics(model, d, idx):
    """Return (rel_err, exact) for TokenModel on rows idx (teacher-forced)."""
    model.eval()
    rel_errs, exacts = [], []
    with torch.no_grad():
        for i in range(0, len(idx), 256):
            b = idx[i:i + 256]
            src, tgt = token_batch(d, b)
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            logits = model(src, tgt[:, :-1])
            pred = logits.argmax(-1)
            true = tgt[:, 1:]
            pred_v = cm._token_to_value(pred.cpu().numpy())
            true_v = cm._token_to_value(true.cpu().numpy())
            mask = np.abs(true_v) > 0.05
            if mask.any():
                rel_errs.append((np.abs(pred_v - true_v) / np.abs(true_v))[mask].mean())
            exacts.append((pred == true).float().mean().item())
    return float(np.mean(rel_errs)), float(np.mean(exacts))


def train_field(model, d, val, arm, epochs, lr, batch, seed):
    set_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = d["value_bin"].shape[0]
    idx = np.arange(n)
    for ep in range(1, epochs + 1):
        model.train()
        np.random.shuffle(idx)
        tot = 0.0; nb = 0
        for i in range(0, n, batch):
            b = idx[i:i + batch]
            x, y_bin, y_dens = field_batch(d, b, arm)
            x = {k: v.to(DEVICE) for k, v in x.items()}
            y_bin, y_dens = y_bin.to(DEVICE), y_dens.to(DEVICE)
            logits, dens = model(x)
            loss = F.cross_entropy(logits.reshape(-1, cm.N_VALUE_BINS), y_bin.reshape(-1))
            loss = loss + F.mse_loss(dens, y_dens)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if ep % 10 == 0 or ep == 1:
            re, ex = field_metrics(model, val, np.arange(val["value_bin"].shape[0]), arm)
            print(f"[{ep:3d}/{epochs}] loss={tot/nb:.4f} test_in rel_err={re:.4f} exact={ex:.4f}", flush=True)
    return model


def train_token(model, d, val, epochs, lr, batch, seed):
    set_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = d["fine_value"].shape[0]
    idx = np.arange(n)
    for ep in range(1, epochs + 1):
        model.train()
        np.random.shuffle(idx)
        tot = 0.0; nb = 0
        for i in range(0, n, batch):
            b = idx[i:i + batch]
            src, tgt = token_batch(d, b)
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            logits = model(src, tgt[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, cm.TOK_VOCAB), tgt[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if ep % 10 == 0 or ep == 1:
            re, ex = token_metrics(model, val, np.arange(val["fine_value"].shape[0]))
            print(f"[{ep:3d}/{epochs}] loss={tot/nb:.4f} test_in rel_err={re:.4f} exact={ex:.4f}", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["A", "B", "C", "D"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()

    tr, ti, tx = load_data()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    if a.arm in ("A", "B", "C"):
        model = cm.FieldModel(a.arm).to(DEVICE)
        model = train_field(model, tr, ti, a.arm, a.epochs, a.lr, a.batch, a.seed)
        re_in, ex_in = field_metrics(model, ti, np.arange(ti["value_bin"].shape[0]), a.arm)
        re_x, ex_x = field_metrics(model, tx, np.arange(tx["value_bin"].shape[0]), a.arm)
    else:
        model = cm.TokenModel().to(DEVICE)
        model = train_token(model, tr, ti, a.epochs, a.lr, a.batch, a.seed)
        re_in, ex_in = token_metrics(model, ti, np.arange(ti["fine_value"].shape[0]))
        re_x, ex_x = token_metrics(model, tx, np.arange(tx["fine_value"].shape[0]))

    result = {
        "arm": a.arm, "epochs": a.epochs, "lr": a.lr, "batch": a.batch, "seed": a.seed,
        "n_params": model.n_params(),
        "test_in_rel_err": re_in, "test_in_exact": ex_in,
        "test_extrap_rel_err": re_x, "test_extrap_exact": ex_x,
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(OUT, f"calc_{a.arm}.json"), "w") as f:
        json.dump(result, f, indent=2)
    torch.save(model.state_dict(), os.path.join(OUT, f"calc_{a.arm}.pt"))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
