# S2a — channel audit: where does the semantic signal actually live?

Date: 2026-09-01. NOT pre-registered (diagnostic, no gates fired; it
changes no verdicts — it looks inside the encoder the pre-registered S2
run already trained and committed).

## Sean's hypothesis (the reason this audit exists)

"Density is an escape hatch: the model dumps semantics into density
because it's allowed to, and colors coast. If density carried no
semantic meaning — color and location only — the encoder would have to
use colors. Have we tried color + space, no density?"

## Method (s2_channel_audit.py)

Saved S2 encoder (docs/s2_results.md run, untouched). Encode the 400
held-out sentences. Compute pairwise-distance Spearman rho vs the 64-d
nomic embedding using ONLY selected slices of each field:

| channel            | dims | rho vs embedding |
|--------------------|------|-----------------:|
| full field         | 4608 | 0.966 |
| density only       | 512  | 0.627 |
| color, both slots  | 2048 | **0.966** |
| color, slot 0 only | 1024 | 0.930 |
| hue only (cos/sin) | 1024 | **0.960** |
| brightness + alpha | 1024 | 0.664 |
| occupancy binary   | 512  | -0.164 |

Diagnostics: density mean 0.162, std 0.132 (not saturated — it uses
its range). Lit cells per sentence: 510.6/512. Paired-sentence lit-cell
Jaccard 0.9946.

## The answer to Sean's question: mostly no — and that's the surprise

**The hue channels are already carrying the semantics.** Hue-only
(1,024 dims: two 2-D hue angles per cell) preserves 0.960 of embedding
structure vs 0.966 for the whole field. All color dims together match
the full field exactly (0.9660 vs 0.9661). Density holds a real but
partial share (0.627), brightness/alpha a smaller one (0.664). The not
an "escape hatch": color+space alone already encodes essentially all of
the semantic structure this encoder emits.

## What actually killed S2 (revised, more precise than yesterday's writeup)

The problem is RESOLUTION, not allocation. Two facts sit side by side:

1. Hue carries the semantics (rho 0.96) — but distributed across ALL
   512 cells (occupancy 510.6/512 lit; paired Jaccard 0.995).
2. Density's contribution is redundant-with-hue to a large degree
   (full 0.966 vs hue 0.960 — density adds ~nothing on top), yet in the
   SUPERPOSITION construction it is the channel that gets ADDED
   (min(d1+d2,1)), while hue entries stay separate per slot.

So Sean's proposed fix (drop density) removes the destructive term from
the superposition, and my yesterday's fix (sparsify occupancy) restores
per-cell addressability. They attack the same failure from two sides:

- Drop density -> slots carry hue only -> the reader must split by slot,
  which randomization made content-addressed; plausible but the field
  is still painted everywhere, so "which cells matter" remains
  unresolved (occupancy binary rho -0.164: WHERE a cell is lit carries
  ~no semantic information — position itself is semantically DEAD).
- Sparsify -> each sentence paints few cells; the summed density in a
  co-lit cell concerns only the few streams that chose that cell;
  slot-randomized hue still needs content-based association but there
  is now space to resolve it.

The audit's sharpest NEW finding: **position is semantically dead**
(occupancy rho -0.16). The encoder uses the grid as 512 parallel slots
and never uses ADJACENCY. For a representation whose design premise is
"space means something" (the ego-centered grid, shape/field vocabulary),
this is the deepest problem — not density. Density is a symptom: with
no pressure to use position meaningfully, the model has no reason to
concentrate mass anywhere in particular.

## Consequence for the roadmap (proposal, not yet pre-registered)

The next S2-family experiment should carry BOTH fixes as ablation arms,
since each is cheap and they answer different questions:

  arm 1  no-density superposition (Sean's fix): slots carry hue only;
         tests whether the destructive additive channel was the whole story
  arm 2  occupancy prior (my fix): top-k lit cells, density kept;
         tests whether spatial concentration restores separability
  arm 3  both combined
  arm 4  (control) re-run S2 as-is to confirm reproducibility of the KILL

Same reader, same gates as docs/calc_s2.md (G1 0.85 / G2 0.05 / G3 sym;
KILL < 0.60), same seed.

If arm 3 passes alone, the thesis stands with a corrected encoder. If
only arm 1 passes, density was the poison. If only arm 2 passes,
position-use was. If none pass, the multi-state clause is dead at the
semantic layer and the documented negative stands.