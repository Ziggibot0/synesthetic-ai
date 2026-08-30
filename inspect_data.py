"""Inspect available training data for the voxel-space model."""
import numpy as np, json, os

d = r"C:\Users\skell\Desktop\embedding-vibes\experiments\exp1_linear_probe\results"
for f in ["embeddings_nomic_embed_text.npy", "labels.npy", "labels.json",
          "binary_labels.npy"]:
    p = os.path.join(d, f)
    if f.endswith(".npy"):
        a = np.load(p)
        print(f, a.shape, a.dtype)
    else:
        j = json.load(open(p))
        print(f, type(j).__name__, str(j)[:300])

# fallacy repo text
fr = r"C:\Users\skell\Desktop\embedding-vibes\data\logical-fallacy-repo"
for root, _, files in os.walk(fr):
    for fn in files:
        if fn.endswith((".jsonl", ".json", ".csv")) and "data" in root.lower():
            p = os.path.join(root, fn)
            print("---", p, os.path.getsize(p))
            with open(p, encoding="utf-8", errors="replace") as fh:
                print(fh.read(500))
            break
    else:
        continue
    break
