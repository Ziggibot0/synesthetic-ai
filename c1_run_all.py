"""
c1_run_all.py — run the full C1 matrix (2 tasks x 2 arms) sequentially,
then print the verdict. One command, one run.

Usage:
  py -3.12 c1_run_all.py [--epochs 60] [--batch 128] [--resume]

Runs sup_voxel, sup_token, int_voxel, int_token in order (~16s/epoch
each on the iGPU, ~65 min total), then c1_report.py. Each arm writes its
own status file + resumable checkpoint; watch with c1_monitor.py.
"""
from __future__ import annotations
import argparse, subprocess, sys, time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    for task, kind in [("sup", "voxel"), ("sup", "token"),
                       ("int", "voxel"), ("int", "token")]:
        print(f"\n{'='*60}\nRUNNING {task}_{kind}  ({a.epochs} epochs)\n{'='*60}", flush=True)
        cmd = [sys.executable, "c1_train.py", "--task", task, "--kind", kind,
               "--epochs", str(a.epochs), "--batch", str(a.batch)]
        if a.resume:
            cmd.append("--resume")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"\n{task}_{kind} FAILED (exit {r.returncode}). "
                  f"Check results/c1_{task}_{kind}_status.json. "
                  f"Re-run with --resume to continue.", flush=True)
            sys.exit(1)

    print(f"\n{'='*60}\nALL C1 ARMS DONE in {time.time()-t0:.0f}s. Verdict:\n{'='*60}", flush=True)
    subprocess.run([sys.executable, "c1_report.py"])

if __name__ == "__main__":
    main()
