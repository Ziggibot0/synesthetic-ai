# synesthetic-ai

A voxel semantic-space model: text mapped to a position in a small 3D grid
plus a LIST of HSL colors — not a mix. Two colors at once, never blended
(a set, not an average), which is the structure the synesthesia literature
calls simultaneous color experience and this project treats as an
uncertainty/branch representation.

Independent project. Related work (fallacy geometry, JEPA trajectories)
lives in Ziggibot0/embedding-vibes; that project's data is reused here for
evaluation only.

**Read this first:** [STORY.md](STORY.md) — why this project exists, the
hypothesis in falsifiable form, and the literature grounding.

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
  - `anchor`  — free + soft hue anchoring on color words (circular loss)
  Collapse monitors logged every epoch (mean-pair-cos, per-dim std,
  slot entropy). Checkpoints + histories under `results/`.
- `eval_probe.py` — per-fallacy-class logistic probes (70/30 split, AUC)
  on (a) color structure (top-2 slots), (b) the 64-dim rep, (c) a raw
  token-count baseline. Saves reps as .npy for visualization.
- `train_wrap.py`, `ab_test.py`, `bisect_grads.py`, `step_probe.py` —
  harness + the debugging probes that diagnosed the training failure
  below.

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

## Run

```
py -3.12 train_wrap.py 60 all    # free, distill, anchor
py -3.12 eval_probe.py           # probes + baseline
```

Requires torch (ROCm build used here), pandas, numpy, scikit-learn.
Checkpoints (*.pt) and reps (*.npy) are gitignored — too big for the repo.