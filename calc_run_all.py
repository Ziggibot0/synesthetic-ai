"""
calc_run_all.py — run the full 4-arm calc-gate matrix sequentially, then
print the pre-registered verdict. One command, one overnight.

Usage:
  py -3.12 calc_run_all.py [--epochs 60] [--batch 128] [--resume]

Runs arms A, B, C, D in order (each ~16s/epoch on the iGPU, so the full
matrix is ~60-70 min), then calc_report.py. Each arm writes its own
status file + resumable checkpoint, so you can watch progress with
calc_monitor.py --watch and resume a crashed run with --resume.
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
    for arm in "ABCD":
        print(f"\n{'='*60}\nRUNNING ARM {arm}  ({a.epochs} epochs)\n{'='*60}", flush=True)
        cmd = [sys.executable, "calc_train.py", "--arm", arm,
               "--epochs", str(a.epochs), "--batch", str(a.batch)]
        if a.resume:
            cmd.append("--resume")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"\nARM {arm} FAILED (exit {r.returncode}). "
                  f"Check results/calc_{arm}_status.json. "
                  f"Re-run with --resume to continue.", flush=True)
            sys.exit(1)

    print(f"\n{'='*60}\nALL ARMS DONE in {time.time()-t0:.0f}s. Verdict:\n{'='*60}", flush=True)
    subprocess.run([sys.executable, "calc_report.py"])

if __name__ == "__main__":
    main()
