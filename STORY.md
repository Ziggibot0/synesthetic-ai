# The story: Why teach a computer to see in colors?

We call the representation **chromavox** (chroma + vox): a colored voxel
semantic space. This document is the human story of why that representation
exists.

## The personal observation (the hypothesis generator)

I have synesthesia. Specifics, all perceptual rather than interpretive:

- **Sound is seen as colors in space.** I see a rough shape of the room I'm in from ambient
  sound alone — lights off, no problem. Echoes have outlines.
- **Feeling is color in space.** My own physical and emotional states appear as
  color/position structure. So do other people's — everyone carries an
  "aura" I read automatically. I'm not claiming anything mystical: my
  brain surfaces information it parses faster than my words can justify.
  People tell me more than they think they do, and my weird brain gives me the
  summary before the conversation catches up.
- **Hard thought is 4-D.** When I think hard about a problem, I get
  entropic fractal visions — four-dimensional structure I can walk
  through, where time is one of the ways the shape extends.
- **Music is tiling.** I write music by making those shapes tessellate. Colors that look nice together sound nice together.
- **Colors can be, but are not always, superimposed.** For instance, I'll see a shape from a sound that is both red and blue at the same time but not purple.
- **Some of the colors DNE in the physical world.** Some of the colors I "see" are not real colors. There are no words to describe them.
- **Time has shape.** Time (to me) looks like a road made out of this viscous fluid, and it's colors are superimposed gray and clear waves.
- **Shape texture/density is as important as color.** It's not "just colors". It's shapes that merge, diverge, pass through each other, or swirl together without mixing.

None of this is decoration on my thinking — it *is* my thinking, in part,
and it works: I reliably arrive at conclusions often before peers. Everything above is cross-checked against the
synesthesia literature (Galton 1880 documented spatial "number forms";
Ramachandran & Hubbard 2001; Eagleman's standardized battery; Thaler et
al. mapped human echolocation); my experiences fall inside known
variation, and where they don't, I say so rather than overclaim.

My "squishy", untestable claim is that I may not be as good at problem solving if I didn't have these synesthetic shortcuts.

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
**semantically random** space — see the README's Status section and
`results/`.

A second, quieter risk is the "so what": colors that emerge but do
nothing. Supervised color emergence was already demonstrated in 2018;
emergence alone is not a contribution. The claim this repo must
eventually earn is C3 — that the binding *does something* — via a
downstream comparison against an architecturally identical model
without the color substrate. That experiment is planned, not run, and
every interim claim here stays modest until it exists.

## Grounding in the literature

- **Standardized measurement exists.** Eagleman, Kagan, Nelson, Sagaram &
  Sarma (2007) built a freely accessible synesthesia test battery
  precisely so self-reported synesthesia could be *tested* rather than
  taken on faith. This project applies the same posture: a private
  experience becomes a measurable claim.
- **Cross modal structure is real computation.** Ramachandran &
  Hubbard (2001): ~10% of people have explicit spatial number forms, and
  grapheme-color synesthesia correlates with *measurably* better
  arithmetic and discrimination performance — the overlay is
  functional, not decorative.
- **Human echolocation works.** Thaler, Arnott & Goodale (2011) showed
  blind humans parse room geometry from mouth-click echoes, with
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

## Related computational work — and the gap this repo aims at

*(Full bibliographic details for every work cited in this document live
in [REFERENCES.md](REFERENCES.md), verified against primary sources on
2026-08-30.)*

- **Supervised imitation.** "A Deep Learning Model of Perception in
  Color-Letter Synesthesia" (MDPI, 2018) trained a GAN to colorize
  achromatic letters using aggregate letter-to-color statistics
  reported by a large synesthete cohort. It reproduces human-reported
  mappings; it does not discover binding, and every color it produces
  is supervised by human report.
- **Art systems.** SynVAE and other "artificial synesthete" projects
  (e.g. arXiv 2112.02953) translate between music and images through
  hand-coded note-color maps. Evocative work, but the correspondence is
  largely dictated by the designers and evaluated by none.
- **Position papers.** SyneState (Zenodo, 2025) proposes inducing
  "machine synesthesia" via constrained cross-modal mixing and states
  outright that the required components are integration work; it
  reports no empirical evaluation.
- **Cross-modal correspondence psychology.** Parise & Spence (2012) and
  Palmer & Schloss document that typical, non-synesthete perceivers
  learn audio-visual and music-color correspondences from statistical
  co-occurrence. This is precisely the mechanism this project bets on —
  but that literature offers no computational model that *induces* such
  binding in a learner and then tests it.

The gap: to our knowledge, no work (1) induces color binding with zero
human color priors, (2) represents the binding as an unblended SET
(superposition, never mixture), and (3) evaluates the result against
clinical-style criteria rather than impressions. That is this repo's
lane. If a reviewer knows of a counterexample, the claims here are
narrow enough to survive it.

## What would count as machine synesthesia here (operational definition)

Clinical synesthesia is not diagnosed by introspection; it is diagnosed
by measurable criteria (Eagleman et al. 2007). We port that playbook:

1. **Consistency (test-retest analog).** Same input yields the same
   position + color-set across retraining runs with different seeds and
   across paraphrases of the input.
2. **Structure.** The binding tracks a reference semantic geometry —
   structure-preservation rho above chance, with a shuffled-pair
   control (`audiocaps_pairs.py`) as the null hypothesis.
3. **Automaticity analog.** Binding lives in the forward pass, not in
   a classifier head bolted on afterward.
4. **No human prior.** No color is ever dictated. The hardcoded
   `ANCHOR_WORDS` dictionary is scheduled for removal on this principle;
   hand-painted human colors are EVAL data, never training data. A
   model that replays a person's reported colors demonstrates
   memorization, not synesthesia.

A system meeting 1-2 with 3-4 in place earns the phrase *induced
synesthesia-like binding*. It earns nothing about subjective
experience — see below.

## What we are NOT claiming

- We do not claim the model is conscious, or that this is consciousness,
  or that machine synesthesia is genuinely subjective. The claim is
  narrow: constrained multi-modal representations may carry reasoning-
  usable structure that unstructured vectors of the same size do not.
- We do not aim to replicate the author's personal colors. His mapping
  is one learned instantiation among many possible ones; a model that
  copied it would only prove memorization. If the model's binding later
  converges toward his, that is a finding to report — never a target
  to optimize. This is why human-painted supervision was considered
  and *rejected* (2026-08-30): the model gets its own synesthesia, or
  the claim is empty.
- We do not conflate this repo with our related fallacy-geometry work
  (Ziggibot0/embedding-vibes). That project borrows *from* the same
  mechanism zoo; they are separate projects with separate claims.

## Why this matters if it works

Frontier models burn billions of parameters to rediscover structure we
get "for free" from a restless brain. If a cheap, structured, inspectable
representation — one you can literally *look* at (every cell colored,
every superposition honest) — matches an unstructured vector on the
same data with far fewer parameters, that is a small, real step toward
models whose internals a human can read directly. Not AGI. Not magic.
A better abacus for a mind that already thinks in color.
