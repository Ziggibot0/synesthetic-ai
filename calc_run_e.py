"""
calc_run_e.py — run arm E (corrected color) on the v2 dataset, then
print the full verdict including E1/E2/E3.

Waits for the A/B/C/D matrix to finish (arm D done) so it doesn't
compete for the iGPU, then trains arm E on calc_data_v2.npz.

Usage:
  py -3.12 calc_run_e.py [--epochs 60] [--batch 128]
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def wait_for_d():
    """Block until arm D's status file says done (or failed)."""
    p = os.path.join(OUT, "calc_D_status.json")
    while True:
        if os.path.exists(p):
            with open(p) as f:
                import json
                s = json.load(f)
            if s.get("status") in ("done", "failed"):
                return s.get("status")
        time.sleep(15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    a = ap.parse_args()

    print("waiting for arm D to finish before starting E...", flush=True)
    st = wait_for_d()
    if st == "failed":
        print("arm D failed; not starting E. Check results/calc_D_status.json.", flush=True)
        sys.exit(1)

    print(f"\n{'='*60}\nRUNNING ARM E (corrected color) on v2 data\n{'='*60}", flush=True)
    cmd = [sys.executable, "calc_train.py", "--arm", "E",
           "--epochs", str(a.epochs), "--batch", str(a.batch),
           "--data", os.path.join(OUT, "calc_data_v2.npz")]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("arm E failed.", flush=True)
        sys.exit(1)

    print(f"\n{'='*60}\nFULL VERDICT (A/B/C/D/E):\n{'='*60}", flush=True)
    subprocess.run([sys.executable, "calc_report.py"])


if __name__ == "__main__":
    main()
