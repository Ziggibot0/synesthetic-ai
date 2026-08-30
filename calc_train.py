"""
calc_train.py — train + evaluate one arm of the calc gate, with full
logging, checkpointing, and a live status file so a run is never a
black box.

Usage:
  py -3.12 calc_data.py                 # generate data (once)
  py -3.12 calc_train.py --arm A --epochs 60
  py -3.12 calc_train.py --arm D --epochs 60 --resume   # resume from ckpt
  ... (repeat for B, C)
  py -3.12 calc_monitor.py              # live table of all arms
  py -3.12 calc_monitor.py --watch      # refresh every 10s

Monitoring / checkpointing (the "no black box" contract):
  * results/calc_<arm>_status.json  — updated EVERY epoch with epoch,
    loss, val/extrap metrics, elapsed, and ETA. Read it any time:
        cat results/calc_A_status.json
    On crash it is written with status="failed" + the error, so a dead
    run is never silent.
  * results/calc_<arm>_last.pt       — resumable checkpoint (model +
    optimizer + epoch + best) saved every --ckpt-every epochs and at
    the end. Resume with --resume.
  * results/calc_<arm>.pt           — final checkpoint (written on
    successful completion).
  * Per-epoch console lines with loss, val/extrap metrics, and ETA;
    per-batch progress lines every 25% of an epoch.

Metrics (all on the normalized derivative field, per calc_gate.md):
  rel_err   : mean |pred - true| / |true| over positions with |true|>0.05
              (G1 gate: arm A rel_err < 0.15 on test_in)
  exact     : fraction of positions where predicted bin == true bin
  extrap    : rel_err on test_extrap (disjoint families + wider domain)
              (G2 gate: arm A >= arm D here)

The model is deterministic given --seed; set it to reproduce.
"""
from __future__ import annotations
import argparse, json, os, time, traceback
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


def load_data(path=None):
    p = path or os.path.join(OUT, "calc_data.npz")
    d = np.load(p)
    def split(prefix):
        return {k[len(prefix):]: d[k] for k in d.files if k.startswith(prefix)}
    return split("tr_"), split("ti_"), split("tx_")


def field_batch(d, idx, arm):
    """Build a FieldModel batch from a data dict + row indices."""
    x = {
        "value_bin": torch.tensor(d["value_bin"][idx], dtype=torch.long),
        "density": torch.tensor(d["density"][idx], dtype=torch.float32),
        "hue": torch.tensor(d["hue"][idx], dtype=torch.float32),
        "hue_cos": torch.tensor(d["hue_cos"][idx], dtype=torch.float32),
        "hue_sin": torch.tensor(d["hue_sin"][idx], dtype=torch.float32),
        "brightness": torch.tensor(d["brightness"][idx], dtype=torch.float32),
        "alpha": torch.tensor(d["alpha"][idx], dtype=torch.float32),
        "occupancy": torch.tensor((d["density"][idx] > 0).astype(np.float32)),
    }
    y_bin = torch.tensor(d["deriv_bin"][idx], dtype=torch.long)
    y_dens = torch.tensor(d["deriv_density"][idx], dtype=torch.float32)
    return x, y_bin, y_dens


def token_batch(d, idx):
    """Build a TokenModel batch: src = value tokens, tgt = deriv tokens."""
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


# ---- monitoring / checkpointing helpers ---------------------------------

def write_status(arm, **kw):
    """Atomically write the live status file (temp + rename)."""
    path = os.path.join(OUT, f"calc_{arm}_status.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(kw, f, indent=2)
    os.replace(tmp, path)


def save_checkpoint(model, opt, epoch, arm, best, path):
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "epoch": epoch,
        "best": best,
    }, path)


def load_checkpoint(model, opt, arm):
    """Load resumable ckpt if present. Returns (start_epoch, best)."""
    path = os.path.join(OUT, f"calc_{arm}_last.pt")
    if os.path.exists(path):
        ck = torch.load(path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        print(f"[resume] loaded {path} at epoch {ck['epoch']}")
        return ck["epoch"], ck["best"]
    return 0, None


# ---- training -----------------------------------------------------------

def train(model, tr, val, tx, arm, epochs, lr, batch, seed, ckpt_every, is_token, resume):
    set_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    start_epoch, best = (load_checkpoint(model, opt, arm) if resume else (0, None))
    if not resume and os.path.exists(os.path.join(OUT, f"calc_{arm}_last.pt")):
        print(f"[warn] a checkpoint exists for arm {arm} but --resume was not "
              f"passed; starting fresh (will overwrite). Use --resume to continue.")
    n = tr["fine_value"].shape[0] if is_token else tr["value_bin"].shape[0]
    idx = np.arange(n)
    t0 = time.time()
    epoch_times = []
    n_batches = (n + batch - 1) // batch
    write_status(arm, status="starting", epoch=start_epoch, epochs=epochs,
                 loss=None, val_rel_err=None, val_exact=None,
                 extrap_rel_err=None, best_val_rel_err=best,
                 elapsed_s=0, eta_s=None, device=DEVICE)

    for ep in range(start_epoch + 1, epochs + 1):
        model.train()
        np.random.shuffle(idx)
        ep_t0 = time.time()
        tot = 0.0; nb = 0
        for bi, i in enumerate(range(0, n, batch)):
            b = idx[i:i + batch]
            if is_token:
                src, tgt = token_batch(tr, b)
                src, tgt = src.to(DEVICE), tgt.to(DEVICE)
                logits = model(src, tgt[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, cm.TOK_VOCAB), tgt[:, 1:].reshape(-1))
            else:
                x, y_bin, y_dens = field_batch(tr, b, arm)
                x = {k: v.to(DEVICE) for k, v in x.items()}
                y_bin, y_dens = y_bin.to(DEVICE), y_dens.to(DEVICE)
                logits, dens = model(x)
                loss = F.cross_entropy(logits.reshape(-1, cm.N_VALUE_BINS), y_bin.reshape(-1))
                loss = loss + F.mse_loss(dens, y_dens)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
            if (bi + 1) % max(1, n_batches // 4) == 0:
                ep_el = time.time() - ep_t0
                ep_eta = ep_el / (bi + 1) * (n_batches - bi - 1)
                print(f"  arm {arm} ep {ep}/{epochs} batch {bi+1}/{n_batches} "
                      f"loss={tot/nb:.4f} ep_eta={ep_eta:.0f}s", flush=True)

        ep_el = time.time() - ep_t0
        epoch_times.append(ep_el)
        avg_ep = float(np.mean(epoch_times[-5:]))
        eta = avg_ep * (epochs - ep)

        if is_token:
            re, ex = token_metrics(model, val, np.arange(val["fine_value"].shape[0]))
            re_x, ex_x = token_metrics(model, tx, np.arange(tx["fine_value"].shape[0]))
        else:
            re, ex = field_metrics(model, val, np.arange(val["value_bin"].shape[0]), arm)
            re_x, ex_x = field_metrics(model, tx, np.arange(tx["value_bin"].shape[0]), arm)
        best = re if best is None else min(best, re)

        write_status(arm, status="running", epoch=ep, epochs=epochs,
                     loss=round(tot / nb, 4), val_rel_err=round(re, 4),
                     val_exact=round(ex, 4), extrap_rel_err=round(re_x, 4),
                     best_val_rel_err=round(best, 4),
                     elapsed_s=round(time.time() - t0, 1), eta_s=round(eta, 1),
                     device=DEVICE)
        print(f"[{ep:3d}/{epochs}] loss={tot/nb:.4f} val_rel_err={re:.4f} "
              f"exact={ex:.4f} extrap={re_x:.4f} eta={eta:.0f}s", flush=True)

        if ep % ckpt_every == 0 or ep == epochs:
            save_checkpoint(model, opt, ep, arm, best,
                            os.path.join(OUT, f"calc_{arm}_last.pt"))

    write_status(arm, status="done", epoch=epochs, epochs=epochs,
                 loss=round(tot / nb, 4), val_rel_err=round(re, 4),
                 val_exact=round(ex, 4), extrap_rel_err=round(re_x, 4),
                 best_val_rel_err=round(best, 4),
                 elapsed_s=round(time.time() - t0, 1), eta_s=0, device=DEVICE)
    return model, re, ex, re_x, ex_x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["A", "B", "C", "D", "E"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--ckpt-every", type=int, default=10,
                    help="save resumable checkpoint every N epochs")
    ap.add_argument("--resume", action="store_true",
                    help="resume from results/calc_<arm>_last.pt if present")
    ap.add_argument("--data", default=None,
                    help="path to a specific calc_data.npz (default: results/calc_data.npz)")
    a = ap.parse_args()

    tr, ti, tx = load_data(a.data)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    try:
        if a.arm in ("A", "B", "C", "E"):
            model = cm.FieldModel(a.arm).to(DEVICE)
            model, re_in, ex_in, re_x, ex_x = train(
                model, tr, ti, tx, a.arm, a.epochs, a.lr, a.batch, a.seed,
                a.ckpt_every, is_token=False, resume=a.resume)
        else:
            model = cm.TokenModel().to(DEVICE)
            model, re_in, ex_in, re_x, ex_x = train(
                model, tr, ti, tx, a.arm, a.epochs, a.lr, a.batch, a.seed,
                a.ckpt_every, is_token=True, resume=a.resume)

        result = {
            "arm": a.arm, "epochs": a.epochs, "lr": a.lr, "batch": a.batch,
            "seed": a.seed, "n_params": model.n_params(),
            "test_in_rel_err": re_in, "test_in_exact": ex_in,
            "test_extrap_rel_err": re_x, "test_extrap_exact": ex_x,
            "seconds": round(time.time() - t0, 1),
        }
        with open(os.path.join(OUT, f"calc_{a.arm}.json"), "w") as f:
            json.dump(result, f, indent=2)
        torch.save(model.state_dict(), os.path.join(OUT, f"calc_{a.arm}.pt"))
        print(json.dumps(result, indent=2))
    except Exception:
        write_status(a.arm, status="failed", epoch=None, epochs=a.epochs,
                     error=traceback.format_exc().splitlines()[-1],
                     elapsed_s=round(time.time() - t0, 1))
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
