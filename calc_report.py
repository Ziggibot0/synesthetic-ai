"""
calc_report.py — evaluate the pre-registered gates from the 4 arm results.

Reads results/calc_{A,B,C,D}.json (written by calc_train.py) and prints
the G1/G2/G3 verdicts plus the kill condition, exactly as pre-registered
in docs/calc_gate.md.

Gates:
  G1  compute:  arm A test_in rel_err < 0.15
  G2  beat ctrl: arm A test_in rel_err <= arm D  AND
                 arm A test_extrap rel_err <= arm D
  G3  color:     arm A test_in rel_err < arm B  (color earns its keep)
  PASS = G1 + G2. G3 failing alone is a scoped negative, not a kill.
  KILL = arm A fails G1 AND arm D succeeds (rel_err < 0.15) -> the voxel
         substrate does nothing a plain token baseline doesn't.

Run:  py -3.12 calc_report.py
"""
from __future__ import annotations
import json, os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load(arm):
    p = os.path.join(OUT, f"calc_{arm}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def main():
    r = {a: load(a) for a in "ABCD"}
    missing = [a for a, v in r.items() if v is None]
    if missing:
        print(f"MISSING arms: {missing} — run calc_train.py for each first.")
        return

    A, B, C, D = r["A"], r["B"], r["C"], r["D"]
    print("=" * 60)
    print("CALC GATE — pre-registered verdicts (docs/calc_gate.md)")
    print("=" * 60)
    for a in "ABCD":
        v = r[a]
        print(f"  arm {a}: params={v['n_params']:,}  "
              f"test_in rel_err={v['test_in_rel_err']:.4f} exact={v['test_in_exact']:.4f}  "
              f"extrap rel_err={v['test_extrap_rel_err']:.4f}")

    g1 = A["test_in_rel_err"] < 0.15
    g2 = (A["test_in_rel_err"] <= D["test_in_rel_err"]
          and A["test_extrap_rel_err"] <= D["test_extrap_rel_err"])
    g3 = A["test_in_rel_err"] < B["test_in_rel_err"]
    d_ok = D["test_in_rel_err"] < 0.15

    print("-" * 60)
    print(f"G1 compute (A rel_err < 0.15):        {'PASS' if g1 else 'FAIL'}  ({A['test_in_rel_err']:.4f})")
    print(f"G2 beat control (A <= D, in+extrap):  {'PASS' if g2 else 'FAIL'}")
    print(f"G3 color pays (A < B):                {'PASS' if g3 else 'FAIL'}  ({A['test_in_rel_err']:.4f} vs {B['test_in_rel_err']:.4f})")
    print("-" * 60)
    if g1 and g2:
        print("VERDICT: PASS — the voxel substrate computes, and beats the")
        print("         token baseline. Proceed to the binding lane (AudioCaps).")
    elif not g1 and d_ok:
        print("VERDICT: KILL — voxel substrate fails while tokens succeed.")
        print("         'voxel/color substrate does not support computation a")
        print("         token baseline doesn't already get.' Documented negative.")
    else:
        print("VERDICT: INCONCLUSIVE — see gates. G3 alone failing is a scoped")
        print("         negative (color is decoration in the calc setting), not a kill.")


if __name__ == "__main__":
    main()
