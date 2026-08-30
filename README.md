# synesthetic-ai

A voxel semantic-space model: text mapped to a position in a small 3D grid
plus a LIST of HSL colors — not a mix. Two colors at once, never blended
(a set, not an average), which is the structure the synesthesia literature
calls simultaneous color experience and this project treats as an
uncertainty/branch representation.

Independent project. Related work (fallacy geometry, JEPA trajectories)
lives in Ziggibot0/embedding-vibes; that project's data is reused here
for evaluation only, and is being fully decoupled from training
(see Status).

**Read this first:** [STORY.md](STORY.md) — why this project exists, the
hypothesis in falsifiable form, the operational definition of machine
synesthesia this repo tests, and the literature grounding. Full
bibliography with verified sources: [REFERENCES.md](REFERENCES.md).

## The bet

Representation determines capability (Abacus embeddings for arithmetic,
Lample & Charton 2020 for symbolic math). This project asks whether
*structured* representations — hue as an ordinal axis, color-sets as
un-collapsed superpositions, brightness as salience, an ego-centered grid —
give a tiny model (4-layer transformer, 64-dim pooled rep, 1k BPE vocab)
an interpretable geometry that a 768-dim pretrained embedder cannot offer
at 1/100th the size.

## Files

- `world.py` — the playground: voxel world model (position + color-set
  cells, edit log = time), ASCII renderer. Run: `python world.py`
- `voxel_cost_estimate.py` — honest cost comparison voxel vs continuous
  embeddings (storage/distance/NN-search wins; expressiveness loses).
- `bpe_pure.py` — clean-room pure-Python BPE (the Rust `tokenizers`
  crate fails on this Windows box with os error 123).
- `voxel_model.py` — the model: BPE -> 4x256 transformer -> position head
  (8^3 grid logits) + 8 color slots (activation/hue/sat/lum each), pooled
  to 64 dims. Three setups:
  - `free`    — VicReg on (article, masked-article) views only
  - `distill` — free + cosine pull toward PCA-64 of nomic-embed-text
  - `anchor`  — free + soft hue anchoring on color words (circular loss).
    NOTE: the ANCHOR_WORDS color dictionary is a hardcoded human prior
    and is scheduled for removal on principle (see STORY.md — "no human
    prior"); the anchor setup is kept only as the documented collapse
    case.
  Collapse monitors logged every epoch (mean-pair-cos, per-dim std,
  slot entropy). Checkpoints + histories under `results/`.
- `serve.py` + `viewer.html` — the voxel viewer: three.js 8^3 grid,
  origin/axes marked, per-thought cell with hard-swap color flicker
  (superimposition in time, never a blend), 0-1 accuracy scoring.
  Run: `py -3.12 serve.py` -> http://localhost:8123
  Draw mode exists for *evaluation* painting only (saved drawings are
  held-out human eval data; they are never training data — see STORY.md).
- `rep_structure.py` — one-command structure-preservation verdict
  (rho vs reference space; the mission metric).
- `train_wrap.py`, `ab_test.py`, `bisect_grads.py`, `step_probe.py` —
  harness + the debugging probes that diagnosed the training failure
  below.
- `finetune_drawings.py` — PARKED, never run. Head-supervision on human
  drawings; rejected on 2026-08-30 as a design principle (a model that
  copies a person's colors proves memorization, not synesthesia). Kept
  as machinery; do not run without revisiting STORY.md first.
- `scripts/prepare_audiocaps.py` + `audiocaps_pairs.py` — THE GATE,
  staged not run: same-clip vs shuffled AudioCaps caption pairs;
  pre-registered PASS (rho >= 0.15 AND delta >= 0.10) / KILL (rho < 0.05)
  verdict written to `results/pairs_verdict.json`.
- `docs/voxel_spec_v2.md` — v2 representation design (density +
  translucency dims, occupancy-field output with variable-length
  "paint as many cells as the thought needs", and the proposed
  voxel-native differential operators for a calculus task). Design
  only; implemented after the gate passes.

## Status (honest)

- Barlow Twins starved on the weak (article, masked) pair: loss frozen at
  62.02, cross-view correlation at chance, on CPU and iGPU alike.
  Diagnosed via gradient-norm/param-delta probes: weights moved, loss
  couldn't. Replaced with VicReg (sim 10 / var 10 / cov 5): loss 11.5 ->
  4.4 within 5 epochs, cross-view C-diag 62.0 -> 0.03. The fix, the
  diagnosis scripts, and the failure narrative are kept on purpose.
- Overnight triple run (free/distill/anchor, 60 epochs) + probe results:
  `results/probe_results.json` (per-class AUC, model vs token baseline).
- Mission-level verdict from `reps_*.npy` + `rep_structure` analysis:
  free/distill spaces are non-collapsed but semantically **random**
  (structure-preservation rho vs nomic = -0.000/+0.002); the anchor
  variant **collapsed** (mean rep cosine 0.995). Diagnosis: invariance
  pairs define what structure emerges — weak pairs (article vs
  content-masked) yield arbitrary structure. Next iteration trains on
  re-rendering pairs (same claim, different serializations) and grades
  on structure preservation, not classification.
- Known surface-form confound: the fallacy dataset leaks rhetoric — a
  token-count baseline hits AUC 0.61-0.91. Any claim from this data needs
  the paraphrase-controlled version. This is stated up front on purpose.
  The classification eval itself (`eval_probe.py`) was **removed**
  (2026-08-30): it tested the wrong substrate for this project's thesis
  (single-claim fallacy classification, not representation structure),
  and its data dependency kept this repo coupled to embedding-vibes.
  The negative token-baseline finding above is kept as the recorded
  reason.
- PIVOT (2026-08-30): no human color supervision, ever. The goal is
  whether the architecture develops synesthesia-like binding on its
  own — its OWN colors, not a copy of the author's. Rationale and the
  operational criteria for claiming it live in STORY.md.
- Decoupling status: `voxel_model.py` still points at embedding-vibes
  paths for training data/teacher; the viewer is fully self-contained.
  Removal of the cross-repo paths is pending the neutral-corpus decision.

## Run

```
py -3.12 train_wrap.py 60 all    # free, distill, anchor
py -3.12 rep_structure.py        # structure-preservation verdict
py -3.12 serve.py                # viewer -> http://localhost:8123
py -3.12 scripts/prepare_audiocaps.py    # build caption pairs (gate prereq)
py -3.12 audiocaps_pairs.py --arm same   # then --arm shuffled, --report
```

Requires torch (ROCm build used here), pandas, numpy, scikit-learn;
the gate additionally needs `pip install laion-clap` (and scipy, already
present). Checkpoints (*.pt) and reps (*.npy) are gitignored — too big
for the repo.

## License

MIT (see LICENSE). Citation metadata in CITATION.cff.