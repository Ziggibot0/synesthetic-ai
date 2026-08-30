# The story: Why teach a computer to see in colors?

## The personal observation (the hypothesis generator)

I have synesthesia. Specifics, all perceptual rather than interpretive:

- **Sound is space.** I see a rough shape of the room I'm in from ambient
  sound alone — lights off, no problem. Echoes have outlines.
- **Feeling is color.** My own physical and emotional states appear as
  color/position structure. So do other people's — everyone carries an
  "aura" I read automatically. I'm not claiming anything mystical: my
  brain surfaces information it parses faster than my words can justify.
  People tell me more than they think they do, and my wiring hands me the
  summary before the conversation catches up.
- **Hard thought is 4-D.** When I think hard about a problem, I get
  entropic fractal visions — four-dimensional structure I can walk
  through, where time is one of the ways the shape extends.
- **Music is tiling.** I write music by making those shapes tessellate.

None of this is decoration on my thinking — it *is* my thinking, in part,
and it works: I reliably arrive at conclusions well before most people
around me reach them. Everything above is cross-checked against the
synesthesia literature (Galton 1880 documented spatial "number forms";
Ramachandran & Hubbard 2001; Eagleman's standardized battery; Thaler et
al. mapped human echolocation); my experiences fall inside known
variation, and where they don't, I say so rather than overclaim.

## The hypothesis this repo tests

**If a structured sensory overlay demonstrably speeds up one mammal's
reasoning, an artificial analogue might speed up a learner's.**

That's the whole bet, and it is deliberately stated as an experiment, not
a belief. Three claims, each falsifiable:

1. **C1 (antecedent).** Structured multi-channel representations help
   learning systems — *before* considering any analogy. This one is
   already supported by independent work: Abacus embeddings made plain
   transformers go from near-0% to near-perfect addition purely by
   changing the positional representation ["Transformers Can Do
   Arithmetic with the Right Embeddings," 2024]; Lample & Charton
   (ICLR 2020) got transformers beating Mathematica on symbolic
   integration by careful representation choice.
2. **C2 (analogue).** A *voxel + superimposed color-set* representation
   — quantized position with a SET of colors (never mixed), brightness
   as salience, two-color superposition for uncertainty — induces
   *geometric* structure that survives better than an equivalent-sized
   continuous embedding measured by structure-preservation (distance
   rank-correlation to a reference embedding space, grid-usage, cell
   purity, superposition rate).
3. **C3 (machine benefit).** If C2 holds, a model forced to reason
   through this geometry inherits *baked-in* operations humans get from
   synesthesia-like binding: salience by brightness, uncertainty by
   superposition, association by adjacency — structure by construction
   rather than by statistical accident.

Falsification: if C2's metrics stay at chance (structure-preservation
rho ~ 0) under fair training, the analogue does nothing and this repo
becomes a documented negative result. That is acceptable science. The
first run (2026-08-30) is exactly that: a stable, non-collapsed,
**semantically random** space — see `results/` and the lessons below.

## Grounding in the literature (why this isn't an aesthetic fancy)

- **Standardized measurement exists.** Eagleman et al. (2006) built a
  134-item synesthesia battery precisely so self-reported synesthesia
  could be *tested* rather than taken on faith. This project applies the
  same posture: a private experience becomes a measurable claim.
- **Cross modal structure is real computation.** Ramachandran &
  Hubbard (2001): ~10% of people have explicit spatial number forms, and
  grapheme-color synesthesia correlates with *measurably* better
  arithmetic and discrimination performance — the overlay is
  functional, not decorative.
- **Human echolocation works.** Thaler, Arnold, Goodale & Kish (2011)
  showed blind humans parse room geometry from mouth-click echoes, with
  visual-cortex activation. A "sensory shape of the room from sound" is
  literally documented, not a metaphor. (My sound-shape sense is the
  congenital analogue.)
- **Superposition is a real ML concern.** A neural net cannot usefully
  pack 5 features into 2 dimensions without interference (Elhage et
  al., "Toy Models of Superposition," Anthropic 2022). A color-SET per
  cell — features that co-exist unblended rather than average — is a
  hand-built escape from exactly that interference. Nobody, to our
  knowledge, represents uncertainty as multiple co-located hues rather
  than a mixture.

## What we are NOT claiming

- We do not claim the model is conscious, or that this is consciousness,
  or that machine synesthesia is genuinely subjective. The claim is
  narrow: constrained multi-modal representations may carry reasoning-
  usable structure that unstructured vectors of the same size do not.
- We do not conflate this repo with our related fallacy-geometry work
  (Ziggibot0/embedding-vibes). That project borrows *from* the same
  mechanism zoo; they are separate projects with separate claims.

## Why this matters if it works

Frontier models burn billions of parameters to rediscover structure we
get for free from a restless brain. If a cheap, structured, inspectable
representation — one you can literally *look* at (every cell colored,
every superposition honest) — matches an unstructured vector on the
same data with far fewer parameters, that is a small, real step toward
models whose internals a human can read directly. Not AGI. Not magic.
A better abacus for a mind that already thinks in color.