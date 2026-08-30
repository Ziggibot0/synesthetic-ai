"""
calc_monitor.py — live status of all calc-gate arms, from the status
files calc_train.py writes every epoch. No black box: see where each
arm is, its loss, val/extrap metrics, elapsed time, and ETA at a glance.

Usage:
  py -3.12 calc_monitor.py          # one snapshot
  py -3.12 calc_monitor.py --watch  # refresh every 10s until Ctrl-C

Also prints the pre-registered gate verdicts (G1/G2/G3) once all four
arms have a final results/calc_<arm>.json (same logic as calc_report.py).
"""
from __future__ import annotations
import argparse, json, os, time

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def fmt_eta(s):
    if s is None:
        return "?"
    s = int(s)
    return f"{s//60}m{s%60:02d}s"


def fmt_elapsed(s):
    if s is None:
        return "?"
    s = int(s)
    return f"{s//60}m{s%60:02d}s"


def snapshot():
    print("=" * 78)
    print("CALC GATE — live status  (status files updated every epoch)")
    print("=" * 78)
    print(f"{'arm':<4}{'status':<10}{'epoch':<8}{'loss':<8}{'val_rel':<9}"
          f"{'val_ex':<8}{'extrap':<9}{'best':<9}{'elapsed':<10}{'eta':<8}")
    print("-" * 78)
    for a in "ABCD":
        p = os.path.join(OUT, f"calc_{a}_status.json")
        if not os.path.exists(p):
            print(f"{a:<4}{'not started':<10}")
            continue
        with open(p) as f:
            s = json.load(f)
        st = s.get("status", "?")
        if st == "failed":
            print(f"{a:<4}{'FAILED':<10}error: {s.get('error','?')}")
            continue
        ep = s.get("epoch")
        ep_s = f"{ep}/{s.get('epochs')}" if ep is not None else "-"
        print(f"{a:<4}{st:<10}{ep_s:<8}"
              f"{s.get('loss','-'):<8}{s.get('val_rel_err','-'):<9}"
              f"{s.get('val_exact','-'):<8}{s.get('extrap_rel_err','-'):<9}"
              f"{s.get('best_val_rel_err','-'):<9}"
              f"{fmt_elapsed(s.get('elapsed_s')):<10}{fmt_eta(s.get('eta_s')):<8}")
    print("-" * 78)


def gates():
    r = {}
    for a in "ABCD":
        p = os.path.join(OUT, f"calc_{a}.json")
        if os.path.exists(p):
            with open(p) as f:
                r[a] = json.load(f)
    if len(r) < 4:
        print(f"Gates: need all 4 final results (have {len(r)}/4). "
              f"Run calc_report.py when complete.")
        return
    A, B, D = r["A"], r["B"], r["D"]
    g1 = A["test_in_rel_err"] < 0.15
    g2 = (A["test_in_rel_err"] <= D["test_in_rel_err"]
          and A["test_extrap_rel_err"] <= D["test_extrap_rel_err"])
    g3 = A["test_in_rel_err"] < B["test_in_rel_err"]
    d_ok = D["test_in_rel_err"] < 0.15
    print(f"G1 compute (A<0.15): {'PASS' if g1 else 'FAIL'} ({A['test_in_rel_err']:.4f})")
    print(f"G2 beat control (A<=D): {'PASS' if g2 else 'FAIL'}")
    print(f"G3 color pays (A<B): {'PASS' if g3 else 'FAIL'}")
    if g1 and g2:
        print("VERDICT: PASS")
    elif not g1 and d_ok:
        print("VERDICT: KILL")
    else:
        print("VERDICT: INCONCLUSIVE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true", help="refresh every 10s")
    a = ap.parse_args()
    if a.watch:
        try:
            while True:
                os.system("cls" if os.name == "nt" else "clear")
                snapshot(); gates()
                time.sleep(10)
        except KeyboardInterrupt:
            print("\nstopped")
    else:
        snapshot(); gates()


if __name__ == "__main__":
    main()
