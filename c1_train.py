"""
c1_train.py — train + evaluate one arm of the C1 probe (superposition /
integration), with full logging, checkpointing, and live status.

Usage:
  py -3.12 c1_data.py                    # generate data (once)
  py -3.12 c1_train.py --task sup --kind voxel --epochs 60
  py -3.12 c1_train.py --task sup --kind token --epochs 60
  py -3.12 c1_train.py --task int --kind voxel --epochs 60
  py -3.12 c1_train.py --task int --kind token --epochs 60
  py -3.12 c1_report.py                  # verdict (all 4 arms)

Monitoring / checkpointing (the "no black box" contract, same as C0):
  * results/c1_<task>_<kind>_status.json — updated EVERY epoch with
    epoch, loss, val/extrap metrics, elapsed, ETA. On crash it is written
    with status="failed" + the error.
  * results/c1_<task>_<kind>_last.pt — resumable checkpoint.
  * results/c1_<task>_<kind>.pt — final checkpoint.

Metrics (normalized field, per docs/calc_c1.md):
  rel_err : mean |pred - true| / |true| over positions with |true|>0.05
  exact   : fraction of positions where predicted bin == true bin
  extrap  : rel_err on the disjoint-family, wider-domain split
"""
from __future__ import annotations
import argparse, json, os, time, traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import c1_data as cd
import c1_model as cm

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def load_data(path=None):
    p = path or os.path.join(OUT, "c1_data.npz")
    d = np.load(p)
    def split(prefix):
        return {k[len(prefix):]: d[k] for k in d.files if k.startswith(prefix)}
    return split("sup_train_"), split("sup_test_in_"), split("sup_test_extrap_"), \
           split("int_train_"), split("int_test_in_"), split("int_test_extrap_")


def voxel_batch(d, idx, task, encoding="cossin"):
    """Build a VoxelModel batch. prefix = 'f'/'g' (sup) or 'in' (int)."""
    if encoding == "energy":
        color_keys = ["wavelength", "energy"]
    else:
        color_keys = ["hue_cos", "hue_sin"]
    if task == "sup":
        x = {
            "f_value_bin": torch.tensor(d["f_value_bin"][idx], dtype=torch.long),
            "f_density": torch.tensor(d["f_density"][idx], dtype=torch.float32),
            **{f"f_{k}": torch.tensor(d[f"f_{k}"][idx], dtype=torch.float32) for k in color_keys},
            "f_brightness": torch.tensor(d["f_brightness"][idx], dtype=torch.float32),
            "f_alpha": torch.tensor(d["f_alpha"][idx], dtype=torch.float32),
            "g_value_bin": torch.tensor(d["g_value_bin"][idx], dtype=torch.long),
            "g_density": torch.tensor(d["g_density"][idx], dtype=torch.float32),
            **{f"g_{k}": torch.tensor(d[f"g_{k}"][idx], dtype=torch.float32) for k in color_keys},
            "g_brightness": torch.tensor(d["g_brightness"][idx], dtype=torch.float32),
            "g_alpha": torch.tensor(d["g_alpha"][idx], dtype=torch.float32),
        }
    else:
        x = {
            "in_value_bin": torch.tensor(d["in_value_bin"][idx], dtype=torch.long),
            "in_density": torch.tensor(d["in_density"][idx], dtype=torch.float32),
            **{f"in_{k}": torch.tensor(d[f"in_{k}"][idx], dtype=torch.float32) for k in color_keys},
            "in_brightness": torch.tensor(d["in_brightness"][idx], dtype=torch.float32),
            "in_alpha": torch.tensor(d["in_alpha"][idx], dtype=torch.float32),
        }
    y_bin = torch.tensor(d["t_value_bin"][idx], dtype=torch.long)
    y_dens = torch.tensor(d["t_density"][idx], dtype=torch.float32)
    return x, y_bin, y_dens


def token_batch(d, idx, task, serial="interleave"):
    """Build a TokenModel batch. sup: interleave or concat f,g; int: derivative values."""
    if task == "sup":
        fv = d["f_fine_value"][idx]
        gv = d["g_fine_value"][idx]
        tv = d["t_fine_value"][idx]
        if serial == "concat":
            # [f_0,f_1,...,f_31,g_0,g_1,...,g_31] -> length 64
            src = np.concatenate([fv, gv], axis=1)
        else:
            # interleave [f_0,g_0,f_1,g_1,...] -> length 64
            src = np.stack([np.stack([f, g], axis=1).reshape(-1) for f, g in zip(fv, gv)])
        tgt = np.array([[cm._value_to_token(v) for v in row] for row in tv])
    else:
        src = np.array([[cm._value_to_token(v) for v in row] for row in d["in_fine_value"][idx]])
        tgt = np.array([[cm._value_to_token(v) for v in row] for row in d["t_fine_value"][idx]])
    return (torch.tensor(src, dtype=torch.long),
            torch.tensor(tgt, dtype=torch.long))


def voxel_metrics(model, d, idx, task, encoding="cossin"):
    model.eval()
    rel_errs, exacts = [], []
    with torch.no_grad():
        for i in range(0, len(idx), 256):
            b = idx[i:i + 256]
            x, y_bin, _ = voxel_batch(d, b, task, encoding=encoding)
            x = {k: v.to(DEVICE) for k, v in x.items()}
            y_bin = y_bin.to(DEVICE)
            logits, _ = model(x)
            pred = logits.argmax(-1)
            pred_v = (pred.float() + 0.5) / cm.N_VALUE_BINS * 2 - 1
            true_v = torch.tensor(d["t_fine_value"][b], device=DEVICE)
            mask = true_v.abs() > 0.05
            if mask.any():
                rel_errs.append(((pred_v - true_v).abs() / true_v.abs())[mask].mean().item())
            exacts.append((pred == y_bin).float().mean().item())
    return float(np.mean(rel_errs)), float(np.mean(exacts))


def token_metrics(model, d, idx, task, serial="interleave"):
    model.eval()
    rel_errs, exacts = [], []
    with torch.no_grad():
        for i in range(0, len(idx), 256):
            b = idx[i:i + 256]
            src, tgt = token_batch(d, b, task, serial=serial)
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


def write_status(tag, **kw):
    path = os.path.join(OUT, f"c1_{tag}_status.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(kw, f, indent=2)
    os.replace(tmp, path)


def save_ckpt(model, opt, epoch, tag, best, path):
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                "epoch": epoch, "best": best}, path)


def load_ckpt(model, opt, tag):
    path = os.path.join(OUT, f"c1_{tag}_last.pt")
    if os.path.exists(path):
        ck = torch.load(path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        print(f"[resume] loaded {path} at epoch {ck['epoch']}")
        return ck["epoch"], ck["best"]
    return 0, None


def train(model, tr, val, tx, task, kind, tag, epochs, lr, batch, seed, ckpt_every, resume, serial="interleave", encoding="cossin"):
    set_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    start_epoch, best = (load_ckpt(model, opt, tag) if resume else (0, None))
    n = tr["f_value_bin"].shape[0] if task == "sup" else tr["in_value_bin"].shape[0]
    idx = np.arange(n)
    t0 = time.time()
    epoch_times = []
    n_batches = (n + batch - 1) // batch
    write_status(tag, status="starting", epoch=start_epoch, epochs=epochs,
                 loss=None, val_rel_err=None, val_exact=None, extrap_rel_err=None,
                 best_val_rel_err=best, elapsed_s=0, eta_s=None, device=DEVICE)

    for ep in range(start_epoch + 1, epochs + 1):
        model.train()
        np.random.shuffle(idx)
        ep_t0 = time.time()
        tot = 0.0; nb = 0
        for bi, i in enumerate(range(0, n, batch)):
            b = idx[i:i + batch]
            if kind == "voxel":
                x, y_bin, y_dens = voxel_batch(tr, b, task, encoding=encoding)
                x = {k: v.to(DEVICE) for k, v in x.items()}
                y_bin, y_dens = y_bin.to(DEVICE), y_dens.to(DEVICE)
                logits, dens = model(x)
                loss = F.cross_entropy(logits.reshape(-1, cm.N_VALUE_BINS), y_bin.reshape(-1))
                loss = loss + F.mse_loss(dens, y_dens)
            else:
                src, tgt = token_batch(tr, b, task, serial=serial)
                src, tgt = src.to(DEVICE), tgt.to(DEVICE)
                logits = model(src, tgt[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, cm.TOK_VOCAB), tgt[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
            if (bi + 1) % max(1, n_batches // 4) == 0:
                ep_el = time.time() - ep_t0
                ep_eta = ep_el / (bi + 1) * (n_batches - bi - 1)
                print(f"  {tag} ep {ep}/{epochs} batch {bi+1}/{n_batches} "
                      f"loss={tot/nb:.4f} ep_eta={ep_eta:.0f}s", flush=True)

        ep_el = time.time() - ep_t0
        epoch_times.append(ep_el)
        avg_ep = float(np.mean(epoch_times[-5:]))
        eta = avg_ep * (epochs - ep)

        if kind == "voxel":
            re, ex = voxel_metrics(model, val, np.arange(val["f_value_bin"].shape[0] if task == "sup" else val["in_value_bin"].shape[0]), task, encoding=encoding)
            re_x, ex_x = voxel_metrics(model, tx, np.arange(tx["f_value_bin"].shape[0] if task == "sup" else tx["in_value_bin"].shape[0]), task, encoding=encoding)
        else:
            re, ex = token_metrics(model, val, np.arange(val["f_fine_value"].shape[0] if task == "sup" else val["in_fine_value"].shape[0]), task, serial=serial)
            re_x, ex_x = token_metrics(model, tx, np.arange(tx["f_fine_value"].shape[0] if task == "sup" else tx["in_fine_value"].shape[0]), task, serial=serial)
        best = re if best is None else min(best, re)

        write_status(tag, status="running", epoch=ep, epochs=epochs,
                     loss=round(tot / nb, 4), val_rel_err=round(re, 4),
                     val_exact=round(ex, 4), extrap_rel_err=round(re_x, 4),
                     best_val_rel_err=round(best, 4),
                     elapsed_s=round(time.time() - t0, 1), eta_s=round(eta, 1), device=DEVICE)
        print(f"[{ep:3d}/{epochs}] loss={tot/nb:.4f} val_rel_err={re:.4f} "
              f"exact={ex:.4f} extrap={re_x:.4f} eta={eta:.0f}s", flush=True)

        if ep % ckpt_every == 0 or ep == epochs:
            save_ckpt(model, opt, ep, tag, best, os.path.join(OUT, f"c1_{tag}_last.pt"))

    write_status(tag, status="done", epoch=epochs, epochs=epochs,
                 loss=round(tot / nb, 4), val_rel_err=round(re, 4),
                 val_exact=round(ex, 4), extrap_rel_err=round(re_x, 4),
                 best_val_rel_err=round(best, 4),
                 elapsed_s=round(time.time() - t0, 1), eta_s=0, device=DEVICE)
    return model, re, ex, re_x, ex_x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["sup", "int"])
    ap.add_argument("--kind", required=True, choices=["voxel", "token"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--ckpt-every", type=int, default=10)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--data", default=None)
    ap.add_argument("--serial", default="interleave", choices=["interleave", "concat"],
                    help="token serialization for sup task: interleave [f0,g0,...] or concat [f0..f31,g0..g31]")
    ap.add_argument("--encoding", default="cossin", choices=["cossin", "energy"],
                    help="color encoding for voxel arm: cossin (hue cos/sin) or energy (wavelength+energy)")
    a = ap.parse_args()

    sup_tr, sup_ti, sup_tx, int_tr, int_ti, int_tx = load_data(a.data)
    tr = sup_tr if a.task == "sup" else int_tr
    val = sup_ti if a.task == "sup" else int_ti
    tx = sup_tx if a.task == "sup" else int_tx
    tag = f"{a.task}_{a.kind}" if a.serial == "interleave" else f"{a.task}_{a.kind}_{a.serial}"
    if a.encoding != "cossin":
        tag = f"{tag}_{a.encoding}"
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    try:
        model = cm.build(a.task, a.kind, encoding=a.encoding).to(DEVICE)
        model, re_in, ex_in, re_x, ex_x = train(
            model, tr, val, tx, a.task, a.kind, tag, a.epochs, a.lr, a.batch,
            a.seed, a.ckpt_every, a.resume, serial=a.serial, encoding=a.encoding)
        result = {
            "task": a.task, "kind": a.kind, "epochs": a.epochs, "lr": a.lr,
            "batch": a.batch, "seed": a.seed, "n_params": model.n_params(),
            "test_in_rel_err": re_in, "test_in_exact": ex_in,
            "test_extrap_rel_err": re_x, "test_extrap_exact": ex_x,
            "seconds": round(time.time() - t0, 1),
        }
        with open(os.path.join(OUT, f"c1_{tag}.json"), "w") as f:
            json.dump(result, f, indent=2)
        torch.save(model.state_dict(), os.path.join(OUT, f"c1_{tag}.pt"))
        print(json.dumps(result, indent=2))
    except Exception:
        write_status(tag, status="failed", epoch=None, epochs=a.epochs,
                     error=traceback.format_exc().splitlines()[-1],
                     elapsed_s=round(time.time() - t0, 1))
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
