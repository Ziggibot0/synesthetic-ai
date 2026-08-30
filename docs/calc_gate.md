# Calc gate — experiment design (pre-registered)

STATUS: STAGED, NOT RUN. Built per Sean's 2026-08-30 decision: train
solely on calculus first. "I just wanna see if this style of embedding
CAN do calculus" - the substrate existence proof, before any real-world
dataset. Supersedes the AudioCaps caption-pair test as the FIRST
experiment (the caption-pair gate remains the entry exam for the
BINDING lane - induced synesthesia - and is not deleted, just sequenced
second).

## Why calc-first is the right first gate

1. Cleanest possible falsification. The map f -> f' is exactly defined;
   no dataset confounds, no surface-form leakage (the fallacy trap), no
   corpus dependencies. Fully self-contained in this repo - the final
   piece of the embedding-vibes decoupling.
2. Hard precedent, right shape: Abacus embeddings (McLeish et al. 2024)
   and Lample & Charton (2020) were both trained purely on synthetic
   formal data. "Synthetic-only first" is the established methodology,
   not a shortcut.
3. The claim is sharp and binary: does the voxel substrate SUPPORT
   computation at all? If no, the synesthetic-thoughts program was
   never going to work. If yes, the binding question (AudioCaps,
   co-occurrence) is asked on a substrate that can already compute.

## Task (T1 from voxel_spec_v2.md)

Input : 1-D occupancy strip encoding f (32 x-bins x 16 value-bins;
        lit cell at (x_i, v_j) ~ f(x_i) = v_j; density = |f| quantized;
        color-set on each lit cell: 2 sig-fig quantized)
Output: same encoding of f' (symbolic derivative, evaluated on grid)
Model: small encoder-decoder transformer (Lample-Charton class, ~10-20M
       params - deliberately NOT tiny; matched across all arms)
Data:  generated - polynomial(deg<=6), sin/cos/linear-combos, exp with
       bounded rate; x in [-5, 5]; sympy-generated ground truth;
       derivative targets quantized finer (4 sig figs) since
       differentiation amplifies error.

Function families are fixed at generation time and the test set uses
DISJOINT families (and longer |x|) - generalization, not memorization.

## Arms (pre-registered ablations)

A  full v2 voxel field  (density + color-sets)
B  -color               (density only; is color earning its keep?)
C  -density-grad        (binary occupancy; is amount-of-stuff needed?)
D  plain token baseline (Charton prefix-notation tokens, param-matched)

D is the load-bearing control: if D wins, the representation does
nothing that tokenization already does, and calculus-via-voxels is
dead as a novelty claim (still a result; the repo records it).

## Pre-registered gates

G1  compute:      arm A rel-error of f' < 0.15 on held-out functions
                  ( Charton got ~100% exact match on integration with
                  tokens; a substrate that can't reach usable accuracy
                  fails here )
G2  beat control: A >= D on derivative exact-match AND on
                  length extrapolation (train |x|<=5, test |x|<=8)
G3  color pays:   A > B on at least one split (else the color dims are
                  decoration in the calculus setting - say so honestly)
PASS = G1 + G2. G3 determines what the paper claims about WHICH dims
matter; failing G3 alone is a scoped negative, not a kill.

KILL (STORY.md discipline): A fails G1 after tuned attempts AND D
succeeds -> "voxel/color substrate does not support computation a
token baseline doesn't already get" - documented, repo stays honest.

## Non-goals for this experiment

* No symbolic output (no prefix-notation emission in arm A/B) - the
  output is the derivative FIELD; checking against sympy ground truth
  numerically. Symbolic decoding is a later, separate question.
* No real-world data, no language, no binding - later experiments.
* T2 (stencil-dynamics internal state) untouched. Do not start it.

## Schedule (honest ETAs)

Build (generator + model + train/eval + ablation flags): one work
session. Data gen is second-scale; training ~10-20M params on the iGPU:
first numbers within hours of code-complete, full 4-arm matrix
overnight. Runs AFTER exp9 finishes (do not compete for the iGPU);
zero dependency on embedding-vibes or Ollama.