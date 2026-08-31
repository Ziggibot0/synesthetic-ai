"""
c1_report.py — evaluate the pre-registered C1 gates from the 4 arm results.

Reads results/c1_{sup,int}_{voxel,token}.json and prints the verdicts
per docs/calc_c1.md.

Gates:
  C1a-G1  V beats T on superposition (test_in rel_err V < T)
  C1b-G1  V beats T on integration   (test_in rel_err V < T)
  C1a-G2  V computes on superposition (rel_err < 0.15)
  C1b-G2  V computes on integration   (rel_err < 0.15)

PASS (substrate has a real edge) = V beats T on at least one task AND
computes on it. KILL (blanket) = T beats V on both tasks.

Run:  py -3.12 c1_report.py
"""
from __future__ import annotations
import json, os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load(tag):
    p = os.path.join(OUT, f"c1_{tag}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def main():
    tags = ["sup_voxel", "sup_token", "int_voxel", "int_token"]
    r = {t: load(t) for t in tags}
    missing = [t for t, v in r.items() if v is None]
    if missing:
        print(f"MISSING: {missing} — run c1_train.py for each first.")
        return

    sv, st, iv, it = r["sup_voxel"], r["sup_token"], r["int_voxel"], r["int_token"]
    print("=" * 60)
    print("C1 — superposition + integration probe verdicts")
    print("=" * 60)
    for t in tags:
        v = r[t]
        print(f"  {t:<12} params={v['n_params']:,}  test_in rel_err={v['test_in_rel_err']:.4f} "
              f"exact={v['test_in_exact']:.4f}  extrap={v['test_extrap_rel_err']:.4f}")

    a1 = sv["test_in_rel_err"] < st["test_in_rel_err"]
    b1 = iv["test_in_rel_err"] < it["test_in_rel_err"]
    a2 = sv["test_in_rel_err"] < 0.15
    b2 = iv["test_in_rel_err"] < 0.15

    print("-" * 60)
    print(f"C1a-G1 V beats T on superposition: {'PASS' if a1 else 'FAIL'}  "
          f"({sv['test_in_rel_err']:.4f} vs {st['test_in_rel_err']:.4f})")
    print(f"C1b-G1 V beats T on integration:   {'PASS' if b1 else 'FAIL'}  "
          f"({iv['test_in_rel_err']:.4f} vs {it['test_in_rel_err']:.4f})")
    print(f"C1a-G2 V computes on superposition: {'PASS' if a2 else 'FAIL'}  ({sv['test_in_rel_err']:.4f})")
    print(f"C1b-G2 V computes on integration:   {'PASS' if b2 else 'FAIL'}  ({iv['test_in_rel_err']:.4f})")
    print("-" * 60)

    if (a1 and a2) or (b1 and b2):
        print("VERDICT: PASS — the voxel substrate has a real, task-specific")
        print("         advantage (superposition and/or integration). Proceed to")
        print("         Phase 2 (binding lane) with the substrate validated.")
    else:
        print("VERDICT: KILL — the token baseline beats the voxel substrate on")
        print("         both superposition and integration. The C0 kill is a")
        print("         blanket verdict; the substrate has no task-specific edge.")
        print("         Phase 2 binding is not justified on this substrate.")


if __name__ == "__main__":
    main()
