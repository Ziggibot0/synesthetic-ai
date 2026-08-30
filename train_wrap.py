import time, sys
t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

import voxel_model as v
v.EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
for s in sys.argv[2:] or ["free"]:
    log("=== setup:", s, "===")
    v.run(s, log=log)
