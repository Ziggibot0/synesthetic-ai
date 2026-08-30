"""
world.py — a playground voxel-semantic space.

Model (loose, for playing — not a spec):
  * A fixed 3D grid. Each cell holds a LIST of colours (HSL), not a mix.
  * The whole grid, read at once, is the "present" moment.
  * Time = sequence of snapshots. Updating the world = a list of edits.
    The edit log IS the history; the diff between two "presents" is change.
  * L (brightness) in each HSL entry = salience of that colour in that cell.
  * Optional: a weight per colour entry = superposition / uncertainty.
    Weighted but un-collapsed: the list never merges into a mixed hue.

Run:  python world.py
"""

from __future__ import annotations
import math, dataclasses, json
from dataclasses import dataclass, field
from typing import Optional

# ── primitives ──────────────────────────────────────────────
Hue = float          # 0–360
Sat = float          # 0–1
Lum = float          # 0–1  (brightness = salience)

@dataclass
class Color:
    h: float
    s: float
    l: float
    w: float = 1.0   # optional amplitude / uncertainty weight (1.0 = definite)

    @classmethod
    def named(cls, name: str, salience: float = 0.8, w: float = 1.0) -> "Color":
        named = {
            "red":  0, "orange": 30, "yellow": 60, "green": 120,
            "cyan": 180, "blue": 240, "magenta": 300,
        }
        return cls(named[name], 0.9, salience, w)

    def hsl_str(self) -> str:
        w = "" if self.w == 1.0 else f"  w={self.w:.2f}"
        return f"hsl({self.h:.0f},{self.s*100:.0f}%,{self.l*100:.0f}%){w}"

    def hex_str(self) -> str:
        # HSL -> hex (for the 'or hex' option; lossless round-trip via HSL)
        h, s, l = self.h / 360.0, self.s, self.l
        def f2(p, q, t):
            if t < 0: t += 1
            if t > 1: t -= 1
            if t < 1/6: return p + (q - p) * 6 * t
            if t < 1/2: return q
            if t < 2/3: return p + (q - p) * (2/3 - t) * 6
            return p
        q = l + s if l <= .5 else l + s - l*s
        p = 2 * l - q
        r = f2(p, q, h + 1/3); g = f2(p, q, h); b = f2(p, q, h - 1/3)
        return "#%02x%02x%02x" % (round(r*255), round(g*255), round(b*255))

@dataclass
class Cell:
    """One voxel: a position + a LIST of colours (not a mix)."""
    colors: list[Color] = field(default_factory=list)

    def add(self, color: Color) -> None:
        self.colors.append(color)

    def clear(self) -> None:
        self.colors = []

    @property
    def salient(self) -> Optional[Color]:
        # brightest colour = most salient thing in this cell
        return max(self.colors, key=lambda c: c.l, default=None)

# grid size (playground: small on purpose)
GRID = 8

@dataclass
class World:
    """The present: the whole grid, read at once."""
    g: int = GRID
    cells: dict[tuple[int,int,int], Cell] = field(default_factory=lambda: {})
    origin: tuple[int,int,int] = (4, 4, 4)    # ego / self, roughly central
    log: list[dict] = field(default_factory=list)

    def _key(self, x, y, z): return (x % self.g, y % self.g, z % self.g)

    def paint(self, x, y, z, *colors: Color, tag: str = "") -> None:
        """Add colours at a position (never merges them)."""
        c = self.cells.setdefault(self._key(x, y, z), Cell())
        for col in colors:
            c.add(col)
        self.log.append({"action": "paint", "pos": [x, y, z],
                         "colors": [dataclasses.asdict(c) for c in colors],
                         "tag": tag})

    def fade(self, x, y, z, tag: str = "") -> None:
        """Clear a position (something left the present)."""
        k = self._key(x, y, z)
        if k in self.cells:
            self.cells[k].clear()
        self.log.append({"action": "fade", "pos": [x, y, z], "tag": tag})

    # ── the whole space, read at once ──
    def present(self) -> dict:
        out = {"origin": list(self.origin), "occupied": {}}
        for (x, y, z), cell in sorted(self.cells.items()):
            if not cell.colors:
                continue
            out["occupied"][f"{x},{y},{z}"] = {
                "colors": [c.hsl_str() for c in cell.colors],
                "salient": cell.salient.hsl_str(),
            }
        return out

    # ── ASCII render: front (z=max) at bottom, back (z=0) at top ──
    _HUE_GLYPH = [(20, "R"), (50, "O"), (75, "Y"), (165, "G"),
                  (205, "C"), (260, "B"), (335, "M"), (361, "R")]
    def glyph(self, h: float) -> str:
        for limit, g in self._HUE_GLYPH:
            if h < limit: return g
        return "R"

    def render(self) -> str:
        lines = ["  front (present)  v",
                 "  back (past/future)  ^",
                 ""]
        for z in reversed(range(self.g)):
            row = []
            for x in range(self.g):
                glyphs = set()
                for (xx, y, zz), cell in self.cells.items():
                    if xx == x and zz == z and cell.colors:
                        glyphs.update(self.glyph(c.h) for c in cell.colors)
                row.append("".join(sorted(glyphs)) if glyphs else "·")
            lines.append("  " + " ".join(row))
        lines += ["", "  x ->  left(R) ......... right(B)"]
        lines += ["  origin(ego) = " + str(self.origin)]
        return "\n".join(lines)

# ── demo: a little "present" then a new moment ─────────────
def main():
    w = World()

    # ego at origin
    w.paint(*w.origin, Color.named("green", 1.0), tag="ego/self")
    # something in the past (back-left), something expected forward (present/front)
    w.paint(1, 2, 1, Color.named("blue", 0.5), tag="old memory (dim)")
    w.paint(3, 4, 7, Color.named("red", 0.9), tag="active idea (bright, front)")

    # the two-colours-not-a-mix shape: red AND blue in one cell, never purple
    w.paint(6, 3, 5, Color.named("red", 0.8), Color.named("blue", 0.8),
            tag="dual-hue shape (red+blue, NOT purple)")

    # uncertainty: a superposition — same cell, colours carry amplitude
    w.paint(2, 5, 3, Color.named("yellow", 0.6, w=0.7),
                          Color.named("cyan",   0.6, w=0.3),
            tag="uncertain (y .7 / c .3, un-collapsed)")

    print("=" * 62)
    print("  MOMENT 1 — the present, read at once")
    print("=" * 62)
    print(w.render())
    print()
    p = w.present()
    for pos, info in p["occupied"].items():
        print(f"  ({pos})  {'  +  '.join(info['colors'])}")

    # ── new info comes in → a new moment; the world updates ──
    print()
    print("  ... new info arrives: the yellow/cyan thing resolves to cyan ...")
    w.fade(2, 5, 3, tag="uncertainty resolved")
    w.paint(2, 5, 3, Color.named("cyan", 0.9), tag="now definite")

    print()
    print("=" * 62)
    print("  MOMENT 2 — updated present")
    print("=" * 62)
    print(w.render())
    print()
    print("  edit log (this is the history / time):")
    for e in w.log:
        print(f"    {e['action']:6} {e['pos']}  {e['tag']}")

    # ── cost of a 'thought' ──
    print()
    occ = len([c for c in w.cells.values() if c.colors])
    print(f"  grid = {w.g}^3 = {w.g**3} cells, {occ} occupied")
    print(f"  one 'read of the present' = O(grid) = {w.g**3} cells  -> constant,")
    print(f"    independent of how much total history exists.")
    print("  that's the real cheapness: bounded working space, time in the log.")

if __name__ == "__main__":
    main()
