# S5 — Semantic superposition on SCENES: can a reader unfold two overlapping scenes?

Status: PRE-REGISTERED 2026-09-01, BEFORE this run.

## Question (Sean's reframe, 2026-09-01)

"if there's hope then i will… i just wanted to see if synesthesia would help a
model think better." S4 showed the crowding is real (paired-scene overlap
Jaccard ~0.44 at k=32) but argued that crowding ≠ drowning. Sean's follow-up:
**"crowding allows for superposition tho right?"**

This run answers it directly. It is the S2 unfold test rebuilt on S4's SCENES
instead of S2's dense smears:

- S2 failed with fields that painted ~all 512 cells (overlap 0.995). A reader
  could not pull two streams apart — nothing was private, the streams drowned.
- C1a (toy calculus) won because its fields were SPARSE and separable.
- S4 produced the missing middle: sparse, position-alive scenes with partial
  overlap. This run tests whether a reader can now unfold two co-located
  sentences — using each scene's private cells to identify the stream, and the
  un-blended color-set slots to handle the shared (crowded) cells.

## Frozen encoder

Reuse S4's trained encoders (no retrain). Scene field = top-k=32 cells lit,
K=3 frequency-color slots, no density. Smear control = S4 arm B dense (V1
arm-B replica). Both loaded from results/s4_encoders.pt.

## Superposition construction

Mirrors S2 semantics (per spec v2):
- salience = max(s_A, s_B) per cell (occupancy union)
- cell in A-only -> that cell holds A's slot-0 color (slots 1:4)
- cell in B-only -> holds B's slot-0 color
- SHARED cell -> field slot-0 holds A's color, field slot-1 holds B's color,
  never blended (the red-AND-blue case)
- which stream carries the slot-0 position is RANDOMIZED per field (kills the
  slot-index / disjoint-blocks shortcut caught in S2).

## Architecture

Reader (per arm): 4,608/5,120 -> 1,024 (GELU) -> two 64-dim embeddings.
Query-free: it can only act on the field, reconstructing BOTH streams' meaning
embeddings. Loss = MSE against the true matryoshka 64-dim embeddings.

- colocated reader: superposed field -> (A_emb, B_emb)   (the test)
- solo reader:     solo scene field -> (A_emb)           (ceiling)

Metric = rho (Spearman of pairwise distances) over fixed held-out pair set on
reconstructed vs true embeddings, exactly as S4 M1. Each held-out sentence's
embedding is recovered by best cosine-match to its fields' output records
(content-addressed readout).

## Gates (pre-registered)

  S5-G1  unfold works on scenes:    rho_colocated >= 0.60
         (the smear regime scored 0.412; scenes should beat it clearly)
  S5-G2  scenes beat smears *in-experiment*:
         rho_colocated_scenes - rho_colocated_smear >= 0.10
  S5-G3  we tested the crowd:        mean paired-scene Jaccard in [0.15, 0.70]

  PASS = G1 and G2 and G3.
  KILL = rho_colocated_scenes < 0.50.
  GRAY otherwise.

## Honesty guards

- Reader is query-free (content-addressed), so it cannot re-encode from text.
- Slot-position randomized per field (no disjoint-block shortcut).
- Scene (k=32) vs dense control run through the IDENTICAL reader setup, so
  scenes-beat-smears is a within-experiment comparison, not a historical quote.
- Encoder validity precondition: solo scene reconstruction rho >= 0.50 or run
  invalid (k=32 solo meaning was 0.89 in S4).

## AMENDMENT (2026-09-01, after smoke, BEFORE full run)

The smoke run exposed a train/eval contamination in the original draft
implementation: readers were trained AND evaluated on the same held-out
fields, so they could memorize field->embedding and saturate at rho 1.0
(smoke: scene 1.0, dense 1.0, G2 trivially uninformative). Fixed before
the full run, in the S2 discipline: ALL readers (solo + colocated, scene
+ dense arms) train on POOL superposed/solo fields only and are evaluated
on HELD-OUT fields they never saw. Gate numbers unchanged; this amendment
tightens the test, it does not move any bar. Also added recover_acc
(fraction of stream records whose nearest true sentence is the correct
one) as a diagnostic, not a gate.

## Success meaning

If PASS: the substrate's one validated advantage (un-blended color-set
superposition) is shown to survive on real language when the base encoding is
a sparse, position-alive scene — the thesis claim, made at the semantic layer
for the first time. If the overlap-sweep shows unfold degrades smoothly with
overlap, we learn where the drowning line is. Either way: the paper's scope
limitation #1 is directly resolved.

## Cost

Full run ~15-25 min iGPU (readers only; encoders frozen). Smoke ~3 min.

## Reproducibility

  py -3.12 s5_superposition.py --smoke   # end-to-end validation
  py -3.12 s5_superposition.py           # full run
Artifacts: results/s5_results.json, results/s5_full.log.
