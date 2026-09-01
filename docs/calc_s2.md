# S2 — semantic superposition disentanglement (pre-registered)

Status: PRE-REGISTERED 2026-08-31, BEFORE this run. Supersedes the earlier
S2 sketch in the roadmap: the reader-vs-token comparative moves to S2b;
this gate is capability-only, which makes it cheaper and unfakeable.

## Why this experiment exists

The project goal now has three load-bearing clauses:

1. embed text semantically into chromavox fields
2. hold MULTIPLE states at once (co-located, unblended)
3. resolve held states when new evidence arrives

Validated so far: clause 1 at structure level (V0 rho 0.984; V1 learned
encoder 0.954-0.973) and clause 2 only on SYNTHETIC calculus (C1a voxel
0.114 vs token 1.004). Clause 3 has never been tested anywhere (paper
scope-limitation #3). S2 is the gate for clause 2 on real language; S2r
is clause 3. If S2 fails, everything above it is unreachable and the
negative is documented cheaply, before any resolution machinery is built.

## The question, falsifiable

Can a LEARNED reader recover BOTH semantic streams from a field in which
two unrelated real sentences share the same cells — and what does
co-location cost vs reading a solo field?

This is C1a's claim transplanted from toy calculus to real sentences.
It is NOT set-vs-blend (the architecture never blends; blend is a
non-represented case and testing it would be a strawman). The claim under
test is superposition = co-location without mixing.

## Design

Corpus: STS-B unique sentences. Sentence-disjoint split 85/15 (seed 1337):
train pool trains the encoder and readers; a held-out pool of 400 test
sentences (never used in any training) forms 200 eval fields. Embeddings:
nomic v1.5, 64-d matryoshka (fresh cache keyed by sentence order — the v1
cache is NOT reused because its row order is unrecoverable).

Encoder: V1 arm-B autoencoder recipe retrained (64 -> 256 -> 4,608,
constrained field, straight-through quantization, ~10min CPU). Its solo
structure rho (~0.97 in V1) is the ceiling every reader is measured
against.

Stream record of a sentence = its encoded field restricted to
[density (512), slot-0 color attrs (hue_cos, hue_sin, brightness, alpha)
= 512 x 4] — a 2,560-d record. Single-entry streams: two sentences must
share a cell whose per-cell cap is K=2, so each stream contributes its
slot-0 record only. Simplification noted; multi-entry streams are the
occupancy-field design (spec v2), out of scope here.

Superposed field from sentences (S1, S2), S2 a random unrelated sentence:
  density = min(d_S1 + d_S2, 1)      (mass adds, per spec v2)
  slot0   = S1's slot-0 attrs
  slot1   = S2's slot-0 attrs
  which sentence takes slot0 is RANDOMIZED per field (kills the
  slot-index shortcut: the reader cannot know where a stream sits
  without reading its content — same class of confound as the fixed-
  projection disjoint-blocks trap caught on 2026-08-31).

Reader: MLP 4,608 -> 1,024 (GELU) -> 5,120 (two records of 2,560). The
reader must UNFOLD the superposition: output both stream records from the
mixed field alone. Assignment ambiguity at eval is resolved by best of the
two output<->stream matchings (2-slot Hungarian). Training loss is the
symmetric sum over both matchings (assignment-free by construction).

Arms (same architecture, same budget, deterministic seeds):
  colocated   reader(superposed field) -> both records     (the test)
  solo        reader(solo field record) -> same record     (ceiling; the
              input already contains the target, so this measures the
              no-interference ceiling, not a capability)

Reported, not gated: encoder solo structure rho (sanity), mean
lit-cell overlap of the paired sentences (test-difficulty stat), AE
training loss curves.

Metrics (held-out 400 sentences, 2,000 random pairs, seed 1337):
  rho_colocated   pairwise-distance Spearman rho of reconstructed stream
                  records vs true stream records (post-encoding fields,
                  i.e. the reader's job is to recover the ENCODING)
  rho_solo        same, solo arm
  interference    delta = rho_solo - rho_colocated (mean over streams)
  symmetry        |delta_S1 - delta_S2|, S1/S2 = designated slot roles

## Pre-registered gates

  S2-G1  capacity:       rho_colocated >= 0.85
  S2-G2  interference:   rho_solo - rho_colocated <= 0.05
  S2-G3  symmetry:       |delta_S1 - delta_S2| <= 0.05
  PASS = G1 and G2 and G3.
  KILL = rho_colocated < 0.60  (co-located real-semantic streams cannot be
         unfolded; "hold multiple states" unreachable at this layer —
         S2r and V2E are moot until the representation changes)
  GRAY = otherwise; delta in (0.05, 0.15] reads as "holds but corrupts",
         escalate to v2 isolation mechanisms (translucency/alpha channel,
         occupancy field) rather than reader tweaks.

Thresholds chosen against V1's measured numbers (solo AE 0.973): G1 at
0.85 demands ~88% of the learned ceiling under co-location; G2 at 0.05
bounds interference to noise level; G3 catches order-asymmetry the
randomization should have killed.

Honest risks, stated before the run:
- The reader could learn to ignore the field and act as a second encoder
  ONLY if it had other input — it does not (query-free by design). A
  degenerate constant output collapses rho to ~0 and fails G1 loudly.
- Density recovery is the hardest sub-target (sum must be split);
  reported per-target-type so a color-only pass cannot masquerade as a
  density result.
- If the AE collapses to near-uniform density, fields barely overlap in
  lit content; the overlap stat will show it and the run is invalid
  before gates are even read (re-run after encoder diagnosis, same gates).

## What each outcome buys toward the goal

- PASS -> S2r (evidence resolution): state field + evidence in, resolved
  field out. Gates to pre-register then: both-streams-recoverable
  pre-evidence (S2-G1 reused), compatible-stream preservation AND
  incompatible-stream suppression post-evidence, symmetric wrong-branch
  control, and the substrate clause: update-from-field >=
  re-encode-from-text (the held field must be the thing that resolved,
  or the architecture is just a seq2seq with extra steps).
- KILL -> the multi-state clause dies at the semantic layer with ~1h
  spent; documented negative, roadmap redirects to representation-side
  fixes (v2 occupancy field) before any reader work.
- Further out (V2E): EMERGENT superposition — ambiguous text produces a
  multi-entry field WITHOUT constructed co-location, resolved by context.
  Requires an ambiguity-structured corpus (bank/bass-style heads with
  resolving continuations). That is the standing corpus decision, not a
  code problem yet.

## Reproducibility

  py -3.12 s2_disentangle.py --smoke   # end-to-end validation, ~3-5 min
  py -3.12 s2_disentangle.py           # full run, ~35-45 min CPU, seed 1337

Artifacts: results/s2_embs.npz (embeddings cache), results/s2_encoder.pt,
results/s2_readers.pt, results/s2_results.json, results/s2_run.log