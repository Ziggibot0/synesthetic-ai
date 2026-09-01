# S6 — end-to-end unification: results (pre-registered)

Status: COMPLETE, 2026-09-01. Design pre-registered in docs/calc_s6.md
(commit dd697e5) BEFORE any code. Full run 17.8 min iGPU.
Artifacts: results/s6_results.json, results/s6_full.log, results/s6_encoder.pt.

## Verdict: INVALID per gate letter — KILL in substance (explained below)

| arm | colocated rho (mean over 3 pairings) | solo RT | M5 color | coherence (w−b) | palette~meaning |
|-----|-----|-----|-----|-----|-----|
| J joint scene   | 0.3039 ± 0.021 | 0.9155 | 0.5383 | −0.228 | 0.100 |
| F frozen scene  | 0.3064 ± 0.034 | 0.8724 | 0.5538 | −0.285 | 0.066 |
| D joint dense   | 0.2922 ± 0.029 | 0.9967 | 0.9028 | −0.263 | 0.118 |

  G0 validity (F ≈ S5 0.198 ± 0.08):  FAIL (F = 0.306)
  G1 unfold lifts (J ≥ 0.45):         FAIL (0.304)
  G2 gradient did it (J−F ≥ 0.15):    FAIL (−0.003)
  G3 substrate matters (J−D ≥ 0.10):  FAIL (+0.012)
  G4 coherence rose (M5 ≥ 0.64):      FAIL (0.538, unchanged from S4's 0.54)

## Why G0 failed, honestly

G0's frozen-reproduce clause was mis-specified in the pre-registration.
Arm F is NOT S5's setup: by design it shares S6's training schedule,
which resamples random pairings every epoch (fingerprint control F3).
S5's reader saw 4096 FIXED superposed fields; S6's frozen arm saw
~200 × 4096 distinct pairings of the same pool — a data-augmentation
effect that lifted the frozen reader from 0.198 to 0.306 on its own.
The smoke run (30 epochs, fewer pairings) hit 0.200, consistent with
this explanation: more pairing diversity, not encoder change, moved F.

This does not rescue the experiment's hypothesis — it strengthens the
null. F is a STRICTLY STRONGER baseline than S5, and the J-vs-F
comparison (the actual question: does encoder gradient flow help?) is
internally valid — identical schedule, identical reader, identical
pairings, the ONLY difference is whether gradients reach the encoder.

## The substantive result

1. Joint gradient bought NOTHING: J − F = −0.003. Two hundred epochs of
   direct unfold pressure backpropagating into the encoder produced an
   encoder no more separable than the frozen one.

2. Coherence did not rise: M5 0.538 vs S4's 0.54 (flat); within-vs-
   between palette coherence stayed negative (−0.23; a sentence's own
   cells are LESS mutually similar than palettes are across sentences);
   palette~meaning alignment ~0.10 (noise). The encoder did not invent
   per-stream color coherence when directly pressured to — it kept its
   solo round-trip high (0.92) and left superposition on the table.

3. Substrate irrelevant under joint training: dense-joint 0.292 ≈
   scene-joint 0.304. Consistent with S5: the sparse/slot machinery is
   not the binding constraint.

4. No fingerprinting occurred (the controls came back clean but empty:
   nothing was learned to fingerprint WITH).

5. Ceiling context: unfold rho plateaus around ~0.3 under every
   configuration tried across S2/S5/S6 (0.412 dense smears at S2's
   different eval, 0.198–0.306 here). The failure is stable and
   architecture-insensitive at this scale.

## Interpretation (per pre-registration's KILL meaning)

By the letter, verdict = INVALID (G0 failed). In substance the run
answers the pre-registered question with the KILL outcome's meaning:
gradient pressure could NOT make language streams separable (J < 0.35
threshold met: 0.304) and could NOT install palette coherence (G4 flat).
The mechanism story closes: C1a worked because coherence was structural
(hue = value by construction); language streams lack it, and end-to-end
unification at this scale does not create it. One honest residual: the
MLP reader caveat from S5a still stands (an attention reader over cells
was never tried), and "at this scale" is a real qualifier — 10k
sentences, 256-hidden encoder, 200 epochs.

Post-hoc gate arithmetic is on the record above; no gates were moved.
The G0 mis-specification is a pre-registration authoring error, dated
and owned here, not a science error: the internal comparison the
experiment was built for (J vs F) is untouched by it.

## What this means for the thesis

Thesis clause 2 ("hold multiple states at once") now has its final and
strongest negative clause: KILLed under frozen pipelines (S2, S5) AND
under end-to-end joint training (S6), with fingerprint controls clean.
The unification hypothesis — that separately-trained parts were the
missing ingredient — is falsified for this architecture and scale.

The paper's positive results (V0, V1, C1a, S4) all stand. The negative
arc is now complete: one validated mechanism (C1a, coherent-by-
construction streams), one validated scene encoder (S4), and a
three-experiment boundary (S2/S5/S6) showing the mechanism does not
transfer to real language — not with sparse scenes, not with faithful
slots, and not with gradient pressure to learn it.

## Reproducibility

  py -3.12 s6_unified.py --smoke   # ~1 min
  py -3.12 s6_unified.py           # ~18 min iGPU
