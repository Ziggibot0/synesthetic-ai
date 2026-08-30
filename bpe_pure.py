"""Canonical BPE (byte-level-ish: whitespace-word, char start,
segmented recount per round), pure Python.
The `tokenizers` Rust crate misbehaves on this Windows box
(tok.train -> os error 123/2), so a clean-room version.

v2: exact train via word-dedup + pair->word index (weighted by word
counts), so 1024 merges on a few hundred articles takes seconds, not
hours. Same canonical algorithm, same merge semantics.
"""
from __future__ import annotations
import re
from collections import Counter


class Bpe:
    def __init__(self, vocab_size=1024):
        self.vocab_size = vocab_size
        self.w2i = {"[PAD]": 0, "[UNK]": 1, "MSK": 2}
        self.merges = []  # [(a, b, name)] in learning order

    @staticmethod
    def words(s):
        s = str(s).replace("MSK", " ")
        s = re.sub(r"\s+", " ", s).strip()
        return [w for w in s.split(" ") if w]

    # ── training: canonical loop, fast bookkeeping ──
    def train(self, texts):
        # dedupe words with counts (exact, frequency-weighted)
        wcount = Counter()
        for t in texts:
            wcount.update(self.words(t))
        segs = {w: list(w) for w in wcount}

        # live pair totals + pair -> words that contain it
        pair_tot = Counter()
        p2words = {}
        for w, seg in segs.items():
            for p in zip(seg, seg[1:]):
                pair_tot[p] += wcount[w]
                p2words.setdefault(p, set()).add(w)

        while len(self.w2i) < self.vocab_size:
            if not pair_tot:
                break
            (a, b), _ = max(pair_tot.items(),
                            key=lambda kv: (kv[1], kv[0]))
            name = f"<{len(self.merges):03d}>"
            self.merges.append((a, b, name))
            self.w2i[name] = len(self.w2i)

            touched = p2words.pop((a, b), set())
            for w in touched:
                seg = segs[w]
                c = wcount[w]
                for p in zip(seg, seg[1:]):
                    pair_tot[p] -= c
                    if pair_tot[p] <= 0:
                        del pair_tot[p]
                new = self._merge_in(seg, a, b, name)
                segs[w] = new
                for p in zip(new, new[1:]):
                    pair_tot[p] += c
                    p2words.setdefault(p, set()).add(w)
        return self

    @staticmethod
    def _merge_in(word, a, b, name):
        out, i = [], 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                out.append(name); i += 2
            else:
                out.append(word[i]); i += 1
        return out

    # ── inference: replay the merge table ──
    def encode(self, s, max_len=None):
        out = []
        for w in self.words(s):
            word = list(w)
            for a, b, name in self.merges:
                if a not in word or b not in word:
                    continue
                word = self._merge_in(word, a, b, name)
            for tok in word:
                out.append(self.w2i.get(tok, 1))   # 1 = [UNK]
        if max_len:
            out = out[:max_len]
        return out

    def decode(self, ids):
        i2w = {i: w for w, i in self.w2i.items()}
        return " ".join(i2w.get(i, "") for i in ids if i and i2w.get(i) != "[PAD]")