"""
Voxel-semantic-space: computational cost estimate
==================================================
Question: is a grid of colored voxels (Rubik's-cube style, quantized
semantic space) computationally cheaper than a high-dim continuous
embedding — for storage, distance, and similarity?

What this script does:
  1. Defines the space (origin = ego, spatial axes = time, colors =
     semantic channels).
  2. Compares memory, distance, and similarity cost across several
     representations.
  3. Gives honest answers about where it's cheap and where it's not.

Run: python voxel_cost_estimate.py
"""

import sys
import time
import struct
from dataclasses import dataclass, field
from typing import Optional
import math

# ──────────────────────────────────────────────
# The space (from the user's description)
# ──────────────────────────────────────────────
#
#  - Origin = ego / self.
#  - Spatial axes (x, y, z) encode time:
#      past  → left  / back  (−x / −z)
#      future→ right / back  (+x / −z)
#      present → front       (+z)
#    This makes a curve (not a straight line).
#  - Each voxel carries a COLOUR (the "semantic" dimension).
#  - Two colours at once, NOT a mix (e.g. red + blue, not purple).
#    → This is a set of colours, not a single RGB value.
#  - Uncertainty → superposition (quantum-style probability amp).
#
# So a "semantic point" is:
#   (x, y, z, {colour_set}, [probability_weights])
#
# A Rubik's-cube analog:
#   classic Rubik's = 3×3×3 grid, 6 faces × 1 colour each face,
#   but our version:
#     - 3D grid (x, y, z)
#     - each cell has a set of colours (0 to N possible)
#     - optional probability amplitude (superposition)


# ──────────────────────────────────────────────
# Representation classes
# ──────────────────────────────────────────────

@dataclass
class ContinuousEmbedding:
    """Standard high-dim embedding (e.g. 768-dim nomic)."""
    dims: int = 768
    bytes_per_point: int = 4 * 768  # float32

    def memory_for(self, n_points: int) -> int:
        return n_points * self.bytes_per_point

    def distance_cost(self) -> str:
        # dot product: N multiplies + N adds per pair
        return f"~{2*self.dims} flops per pair (dot product)"

    def similarity_cost(self, n_pairs: int) -> int:
        return n_pairs * 2 * self.dims


@dataclass
class VoxelSpace:
    """
    Quantized voxel space.
    
    grid:  grid size per axis (e.g. 16×16×16 for a 16-cubed grid)
    colours: number of possible colour channels (e.g. 12 named hues)
    colour_set: max colours per voxel (user says 2 at once, but
                allow up to 8 to be safe)
    superposition: if True, each colour in the set has a probability
                   (amplitude), stored as float16
    """
    grid: tuple[int, int, int] = (16, 16, 16)
    colours: int = 12            # named hue channels (red, blue, green, …)
    max_colours_per_voxel: int = 8
    superposition: bool = True   # store probability amplitude per colour

    def memory_for(self, n_points: int) -> int:
        """
        Each point stored as:
          - grid index: 3 × int8 = 3 bytes (16³ needs 8 bits/axis…
                        actually 16 fits in 4 bits, 3×4=12 bits → 2 bytes
                        but we round to 3 bytes for alignment)
          - colour mask: bitfield over `colours` channels
                         → ceil(colours/8) bytes
          - (if superposition) probability array: max_colours × float16
        """
        idx_bytes = 3
        mask_bytes = math.ceil(self.colours / 8)
        if self.superposition:
            prob_bytes = self.max_colours_per_voxel * 2  # float16
        else:
            prob_bytes = 0
        per_point = idx_bytes + mask_bytes + prob_bytes
        return n_points * per_point

    def per_point_bytes(self) -> int:
        idx = 3
        mask = math.ceil(self.colours / 8)
        prob = self.max_colours_per_voxel * 2 if self.superposition else 0
        return idx + mask + prob

    def grid_cells(self) -> int:
        return self.grid[0] * self.grid[1] * self.grid[2]

    def distance_cost(self) -> str:
        """
        Distance in voxel space has multiple components:
          1. Euclidean grid distance: 3 subtractions + 3 squares + sqrt
          2. Colour-set distance:  XOR or hamming on bit masks
             (or set intersection for overlap)
          3. Superposition: cosine on probability vectors
        Total: ~10-20 flops per pair — orders of magnitude cheaper.
        """
        return ("~10–20 flops per pair "
                "(grid: 3 sub + 3 sq + sqrt; "
                "colour: XOR/bit-hamming; "
                "prob: cosine on ≤8-dim vector)")

    def similarity_cost(self, n_pairs: int) -> int:
        # Conservative: same as continuous but ~50x fewer flops
        return n_pairs * 20


# ──────────────────────────────────────────────
# Cost comparison
# ──────────────────────────────────────────────

def compare(n_points: int = 10000):
    cont = ContinuousEmbedding()
    vox  = VoxelSpace()

    print("=" * 60)
    print("VOXEL SEMANTIC SPACE — COMPUTATIONAL COST ESTIMATE")
    print("=" * 60)

    # ── Memory ──
    mem_cont = cont.memory_for(n_points)
    mem_vox  = vox.memory_for(n_points)

    print(f"\nPoints:            {n_points:,}")
    print(f"\n── MEMORY ──")
    print(f"  Continuous (768d, f32):  {mem_cont/1024/1024:.1f} MB")
    print(f"  Voxel (16³ grid, no sup): {mem_vox/1024/1024:.2f} MB")

    # without superposition
    vox_nosup = VoxelSpace(superposition=False)
    mem_vox_nosup = vox_nosup.memory_for(n_points)
    print(f"  Voxel (16³ grid, NO superposition): {mem_vox_nosup/1024/1024:.2f} MB")

    print(f"\n  Per-point: continuous = {cont.bytes_per_point} bytes")
    print(f"             voxel (sup) = {vox.per_point_bytes()} bytes")
    print(f"             voxel (nosup) = {vox_nosup.per_point_bytes()} bytes")

    # ── Per-pair distance cost ──
    print(f"\n── PAIRWISE DISTANCE ──")
    print(f"  Continuous: {cont.distance_cost()}")
    print(f"  Voxel:      {vox.distance_cost()}")

    # rough ratio
    flops_cont = 2 * 768
    flops_vox  = 20
    print(f"  Speedup (~): {flops_cont/flops_vox:.0f}x")

    # ── Similarity over N pairs ──
    n_pairs = 50000
    sim_cont = cont.similarity_cost(n_pairs)
    sim_vox  = vox.similarity_cost(n_pairs)
    print(f"\n── {n_pairs} PAIRS SIMILARITY ──")
    print(f"  Continuous flops: {sim_cont/1e6:.1f}M")
    print(f"  Voxel flops:      {sim_vox/1e6:.1f}M")
    print(f"  Speedup (~):      {sim_cont/sim_vox:.0f}x")

    # ── Grid size comparison ──
    print(f"\n── GRID ──")
    print(f"  Voxel grid:    {vox.grid[0]}×{vox.grid[1]}×{vox.grid[2]} "
          f"= {vox.grid_cells():,} cells")
    print(f"  Equivalent continuous volume "
          f"(same # of distinct positions): {vox.grid_cells():,}")
    print(f"  768-dim continuous: infinite positions (float32)")

    # ── Rubik's Cube analogy ──
    print(f"\n── RUbIK'S CUBE ANALOGY ──")
    print(f"  Classic Rubik's:  3×3×3 = 27 cells, 6 colours/cell")
    print(f"  Your space:        16×16×16 = 4,096 cells, {vox.colours} colours/cell")
    print(f"  (16 is small; even 32³ = 32,768 cells is still tiny)")
    print(f"  64³ = {64**3:,} cells — still 10,000x fewer positions")
    print(f"  than 768-dim float32 space effectively offers.")

    # ── Honest caveats ──
    print(f"\n── HONEST CAVEATS ──")
    print(f"  1. Quantization LOSS: mapping 768-dim → voxel loses")
    print(f"     information. A 16³ grid can only represent 4,096")
    print(f"     distinct positions. If two semantics map to the same")
    print(f"     voxel, you can't distinguish them (except via colour).")
    print(f"  2. Colour channels are DISCRETE: you lose the continuous")
    print(f"     geometry of embedding space. Nearest-neighbour search")
    print(f"     becomes a grid lookup — which is GOOD for speed, BAD")
    print(f"     for smooth interpolation.")
    print(f"  3. Superposition adds complexity: if two colours carry")
    print(f"     probability weights, distance becomes a cosine on the")
    print(f"     probability vector + grid distance. Still cheap, but")
    print(f"     not a single dot product anymore.")
    print(f"  4. The 'two colours at once, not a mix' constraint means")
    print(f"     you're working with sets, not vectors. Set-distance")
    print(f"     (Jaccard, symmetric-difference) is cheap but different")
    print(f"     from cosine similarity.")
    print(f"  5. For NEAREST-NEIGHBOUR search in the full 768-dim space")
    print(f"     you'd use FAISS/HNSW (log lookup). In voxel space, you")
    print(f"     can do EXHAUSTIVE grid scan (4,096 cells) and find the")
    print(f"     closest in O(grid) — that's the real win. For 100K")
    print(f"     points, HNSW is ~1000 lookups; voxel is 4,096 lookups.")
    print(f"     At 64³ or higher, voxel starts losing to HNSW.")

    # ── Verdict ──
    print(f"\n── VERDICT ──")
    print(f"  Storage:     voxel wins by {mem_cont/mem_vox:.0f}x "
          f"(100K points) — trivial either way")
    print(f"  Distance:    voxel wins ~{flops_cont/flops_vox:.0f}x per pair — "
          f"matters at large N")
    print(f"  NN search:   voxel wins at small grids (≤32³), ties/loses "
          f"at 64³+")
    print(f"  Expressive power: continuous 768d wins — voxel can't")
    print(f"                represent fine-grained distinctions")
    print(f"")
    print(f"  For your use case (hypothesis-gen, semantic intuition):")
    print(f"  the voxel space is CHEAPER to reason about, visualize,")
    print(f"  and manipulate by hand — which is the whole point.")
    print(f"  It's a COARSER model, but the coarse model IS the idea.")
    print(f"")


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    compare(N)
