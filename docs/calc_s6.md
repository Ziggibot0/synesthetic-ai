# S6 — End-to-end unification: can gradient pressure make language streams color-coherent?

Status: PRE-REGISTERED 2026-09-01, BEFORE any code is run.

## Question (Sean's reframe, 2026-09-01)

"We need to UNIFY the model and stop translating between a bunch of models
... my brain is unified and every part ties to the other mechanisms so they
all update on the same events the same way at the same time."

Every prior superposition test (S2, S5) used a pipeline of separately-trained
parts: encoder trained for round-trip fidelity only, superposition applied
after, readers trained post-hoc. The encoder never felt any pressure from the
unfold task. S5a's caveat #2 says it outright: "the scene encoder had no
pressure toward per-stream color coherence (that is the identified
mechanism, not a slip)."

S6 removes the translation gaps: encoder -> superpose -> reader trained
JOINTLY under one loss, so unfold error backpropagates into the encoder.
If the coherence mechanism story (S5 results, "Mechanism" section) is right,
the encoder should be FORCED to invent per-stream palette coherence — the
thing C1a had by construction and language lacks naturally. We do not
hand-design coherence (no-human-prior principle intact); we create the
pressure and measure whether it emerges.

This is the one experiment the S5 data itself asked for. It is also,
by prior agreement, possibly the last: a clean KILL here closes the
mechanism story airtight (coherence must be intrinsic to the modality;
gradient pressure cannot buy it for language) and the project stops with
a complete negative arc.

## Hypotheses

  H1 (primary): joint training lifts colocated readout well above the
      frozen-pipeline numbers (S5: scene 0.198, dense union 0.243).
  H2 (mechanism): if H1 holds, it holds BECAUSE per-stream palette
      coherence rose — measurable on the learned encodings, and the win
      survives the fingerprint controls below.
  H0 (kill): even with direct gradient pressure, the encoder cannot make
      superposed language streams separable without cheating.

## Architecture (one model, one loss)

Data: same STS-B sentence corpus and 64-dim matryoshka embeddings as
S4/S5. Same pool/held-out split discipline as S5 (train on pool ONLY,
evaluate on held-out sentences never seen in any role).

Field spec: unchanged from S4 scene arm — 8^3 grid, top-k=32 cells via
STE gate, K=3 frequency-color slots, no density. The substrate is not
redesigned; only the training signal changes.

Forward pass per training step:
  1. Sample pairs (A, B) of pool sentences (random pairing, resampled
     every epoch — no fixed partner, see fingerprint control F3).
  2. emb_A, emb_B -> encoder -> field_A, field_B
  3. superpose(field_A, field_B) with S5's faithful construction:
     salience = max, private cells keep owner's color, shared cells hold
     both colors in separate slots, slot order randomized per field.
  4. superposed field -> reader -> (rec_A, rec_B)

Reader: same MLP shape as S5 (field -> 1024 GELU -> 2 x 64), query-free,
content-addressed recovery at eval — IDENTICAL readout machinery so the
comparison to S5 is apples-to-apples. (The attention-reader caveat from
S5a is deliberately NOT addressed here; one variable at a time. If S6
passes, an attention reader is a follow-up; if S6 kills, S6b may try it.)

Loss = L_unfold + lambda_rt * L_roundtrip
  L_unfold   = MSE(rec_A, emb_A) + MSE(rec_B, emb_B)  [order-invariant:
               min over the two stream-to-target assignments, since slot
               order is randomized]
  L_roundtrip = MSE(dec(field_solo), emb) with the S4 decoder head,
               kept so the encoding stays meaning-preserving SOLO
               (guards against the encoder abandoning content to win
               unfold). lambda_rt = 1.0.

Everything trains together from S4's weights as init (warm start; a
cold-start arm is not pre-registered — warm start is the fair test of
"add pressure to what exists").

## Arms

  ARM J  (the test)     joint training as above, scene k=32 field.
  ARM F  (baseline)     IDENTICAL architecture and training schedule but
                        encoder FROZEN at S4 weights; only reader (and
                        decoder head) train. Isolates "joint gradient did
                        it" from "longer reader training did it."
                        Expected ~= S5's 0.198 if S5 was fair.
  ARM D  (control)      joint training on the DENSE (arm-B) encoder.
                        If dense-joint matches scene-joint, sparsity/slots
                        are irrelevant and the win (if any) is generic
                        capacity, not the substrate story.

## Fingerprint controls (make-or-break, all pre-registered)

The known cheat: the encoder stops encoding meaning-coherent palettes and
instead paints stream-identity tags (fixed "channel colors" = frequency-
division multiplexing) or per-sentence arbitrary IDs (fingerprints).
Controls:

  F1  Held-out generalization: all gate metrics computed on held-out
      sentences never seen in training in ANY pairing. A pure
      memorization fingerprint cannot transfer.
  F2  Solo readability: the jointly-trained encoder's SOLO fields must
      still decode (S4-style round-trip) at rho >= 0.80. If unfold wins
      but solo meaning collapsed, the encoder traded content for tags.
  F3  Pairing-independence: pairs are random and resampled every epoch at
      train; at eval, each held-out sentence appears in multiple distinct
      pairings. Metric variance across pairings reported. A channel-tag
      scheme (always paint stream-position-0 reddish) is killed by the
      existing per-field slot randomization + max-salience symmetry;
      report the A/B-position swap consistency explicitly.
  F4  Mechanism measurement: M5-style palette coherence (per-sentence
      color self-similarity vs cross-sentence) measured on S4 encoder
      (baseline, known 0.54) and on the S6 encoder. Also report
      palette-vs-meaning alignment: do sentences with similar embeddings
      have similar palettes (coherence tied to CONTENT), or is palette
      similarity uncorrelated with meaning (arbitrary ID)? The second
      pattern = fingerprint even if F1-F3 pass; it gets reported honestly
      and caps the claim (see gates).

## Gates (pre-registered)

  S6-G0  validity: ARM F reproduces S5 within noise
         (|rho_F - 0.198| <= 0.08); solo round-trip of ARM J >= 0.80 (F2).
  S6-G1  unfold lifts: rho_colocated(ARM J) >= 0.45
         (more than double S5's 0.198 and clearly above dense union 0.243).
  S6-G2  the gradient did it: rho(ARM J) - rho(ARM F) >= 0.15.
  S6-G3  substrate matters: rho(ARM J) - rho(ARM D) >= 0.10.
  S6-G4  coherence rose: M5-style palette coherence of ARM J encoder
         > S4 baseline (0.54) by >= 0.10 absolute.

  PASS  = G0 + G1 + G2 + G4  (thesis clause 2 reopens; coherence is
          learnable for language under unified pressure).
        G3 additionally passing = the win is substrate-specific.
        G4 failing while G1-G2 pass = SUSPECT-FINGERPRINT: readout
          improved without palette coherence — claim capped to
          "joint training separates streams by some mechanism that is
          not the coherence story"; F4's palette-vs-meaning analysis
          adjudicates and the honest label goes in the results doc.
  KILL  = G0 passes and rho(ARM J) < 0.35 (gradient pressure could not
          make language streams separable; mechanism story closes:
          coherence is intrinsic to modality, not trainable-in at this
          scale). This is the "stop with a clear conscience" outcome.
  GRAY  = anything else (e.g. 0.35 <= rho < 0.45, or G2 fails —
          reader-training artifact).

## Honesty guards

- Same S5 discipline: readers/encoders see pool only; ALL gate numbers
  from held-out fields. Smoke run first; any contamination found in
  smoke gets a dated amendment BEFORE the full run, gates unmoved.
- Reader capacity, epochs, lr identical across arms J/F/D — the only
  difference between J and F is whether encoder gradients flow.
- No hand-designed coherence anywhere: no palette loss term, no color
  anchors, no stream tags in the input. The ONLY new pressure is the
  unfold task loss. (A explicit coherence regularizer would be S6c,
  a different, weaker claim.)
- Joint training is the regime where leakage/cheating is easiest;
  that is exactly why F1-F4 are pre-registered and non-negotiable.
- Order-invariant loss (min over assignments) prevents the trivial
  "always output A first" degenerate solution from poisoning training.

## Success meaning

If PASS: unification was the missing ingredient — separately-trained
parts could never learn coherence because nothing asked for it; one
gradient through the whole stack could. Thesis clause 2 reopens at the
semantic layer, and the unified-model framing becomes the paper's second
act. Next steps (not pre-registered): attention reader, 3+ streams,
evidence-resolution (S2r) rebuilt on the S6 encoder.

If KILL: the mechanism story is airtight and complete — C1a worked
because coherence was structural; language streams lack it and direct
gradient pressure at this scale cannot install it. The paper's negative
arc gains its final, strongest clause, and the project stops clean.

## Cost

Joint training is heavier than S5 (encoder gradients + 3 arms):
estimate 2-3 h iGPU full run, ~10 min smoke. One evening.

## Reproducibility

  py -3.12 s6_unified.py --smoke   # end-to-end validation
  py -3.12 s6_unified.py           # full run
Artifacts: results/s6_results.json, results/s6_full.log,
results/s6_encoder.pt (ARM J weights, for post-hoc audits).
