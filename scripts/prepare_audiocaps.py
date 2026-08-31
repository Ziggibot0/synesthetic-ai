"""
prepare_audiocaps.py - build the caption-pair corpus for gate test (b).

Fetches the public AudioCaps caption table (10-second clips, ~5 human
captions each) and reshapes to:  [{"clip_id": ..., "caption": ...}, ...]
saved to data/audiocaps_pairs.json. AUDIO IS NEVER DOWNLOADED: this gate
test uses captions only (the invariance pairs come from caption
multiplicity; audio embedding via CLAP-audio is a later arm).

Run:   py -3.12 scripts/prepare_audiocaps.py
Falls back to a manual path if the public CSV moves: put the CSV at
data/audiocaps.csv (columns must include a clip id + caption) and re-run.
"""
from __future__ import annotations
import csv, io, json, os, sys, urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "data")
OUT_JSON = os.path.join(OUT_DIR, "audiocaps_pairs.json")

CSV_URLS = [
    # official AudioCaps dataset repo (verified live 2026-08-30); three
    # splits concatenated. Columns: audiocap_id, youtube_id, start_time,
    # caption. NOTE: audiocap_id is per-caption, NOT per-clip - the clip
    # identity is youtube_id + start_time.
    "https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/train.csv",
    "https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/val.csv",
    "https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/test.csv",
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    local = os.path.join(OUT_DIR, "audiocaps.csv")
    if os.path.exists(local):
        print(f"using local {local}")
        raw = open(local, encoding="utf-8").read()
        rows = list(csv.DictReader(io.StringIO(raw)))
        if rows and "caption" in rows[0]:
            pass
        else:
            rows, raw = [], None
    else:
        rows, raw = [], None
    if not rows:
        print("fetching AudioCaps caption CSVs (train/val/test) ...")
        lines = []
        for f in ["train.csv", "val.csv", "test.csv"]:
            try:
                lines.append(urllib.request.urlopen(
                    f"https://raw.githubusercontent.com/cdjkim/audiocaps"
                    f"/master/dataset/{f}", timeout=60).read().decode("utf-8"))
                print(f"  fetched {f}")
            except Exception as e:
                sys.exit(f"could not fetch {f}: {e}\n"
                         f"manual alternative: download train/val/test.csv "
                         f"from github.com/cdjkim/audiocaps (dataset/) into "
                         f"data/ as audiocaps.csv (one combined file) "
                         f"and re-run.")
        raw = "\n".join(["audiocap_id,youtube_id,start_time,caption"]
                        + [ln for ln in "\n".join(lines).splitlines()
                           if ln.strip() and not ln.startswith("audiocap_id")])
        rows = list(csv.DictReader(io.StringIO(raw)))
    out = []
    for r in rows:
        ytid = (r.get("youtube_id") or r.get("clip_id") or "").strip()
        start = (r.get("start_time") or "").strip()
        # clip identity = youtube segment, not the whole video: segments
        # from the same video can be completely different sounds
        cid = f"{ytid}_{start}" if ytid and start else ytid
        cap = (r.get("caption") or "").strip()
        if cid and cap:
            out.append({"clip_id": cid, "caption": cap})
    if not out:
        sys.exit("CSV parsed but zero usable rows - inspect the format.")
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    grouped = len({r["clip_id"] for r in out})
    print(f"wrote {len(out)} captions across {grouped} clips -> {OUT_JSON}")
    print("next: py -3.12 audiocaps_pairs.py --arm same   (then --arm shuffled,"
          " then --report)")


if __name__ == "__main__":
    main()