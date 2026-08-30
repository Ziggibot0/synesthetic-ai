"""
serve.py - interactive voxel visualizer for synesthetic-ai.

Runs the trained VoxelNet live (no embedding-vibes dependency: tokenizer
cache + .pt checkpoint only) and serves a three.js page that shows:

  * the 8^3 voxel grid (mostly empty space, as you describe)
  * the probe thought's cell, with its colors FLICKERING in time
    (superimposition - colors coexist, they never blend)
  * a 0..1 accuracy slider per probe -> saved to results/scores.json
    as human ground-truth (the paraphrase control the eval was missing)

Run:   py -3.12 serve.py [port]     (default 8123)
Then:  open http://localhost:8123
"""
from __future__ import annotations
import json, math, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from functools import partial
from urllib.parse import urlparse, parse_qs

import numpy as np
import torch
import importlib.util, os

import bpe_pure

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")

MAXLEN = 128

# ----------------------------------------------------------------------------
# model load (once; both setup variants share the network class)
# ----------------------------------------------------------------------------
spec = importlib.util.spec_from_file_location("vm", os.path.join(HERE, "voxel_model.py"))
VM = importlib.util.module_from_spec(spec)
spec.loader.exec_module(VM)


def load_net(setup: str):
    b = bpe_pure.Bpe(VM.VOCAB)
    d = json.load(open(os.path.join(HERE, "bpe_cache.json")))
    b.w2i = {k: int(v) for k, v in d["w2i"].items()}
    b.merges = [tuple(m) for m in d["merges"]]
    net = VM.VoxelNet(VM.VOCAB)
    net.eval()
    net.load_state_dict(
        torch.load(os.path.join(OUT, f"model_{setup}.pt"), map_location="cpu",
                   weights_only=True))
    return b, net


NETS = {}
for s in ["free", "distill", "anchor"]:
    try:
        NETS[s] = load_net(s)
        print(f"loaded model[{s}]")
    except Exception as e:                      # pragma: no cover
        print(f"model[{s}] unavailable: {e}")


def tokenize(b, text):
    ids = []
    for w in b.words(text):
        ids += b.encode(w or "MSK")
    return ids[:MAXLEN]


def forward(setup, text):
    b, net = NETS[setup]
    ids = tokenize(b, text)
    arr = np.zeros(MAXLEN, dtype=np.int64)
    arr[:len(ids)] = ids
    with torch.no_grad():
        o = net(torch.tensor(arr).unsqueeze(0))
    pos = o["pos_logits"][0].argmax(-1).tolist()
    cell = pos[0] * 64 + pos[1] * 8 + pos[2]
    sal = (o["bri"] * o["act"])[0].numpy()
    order = sal.argsort()[::-1][:8]
    slots = [{
        "hue": round(float(o["hue"][0, i].item()), 1),
        "sat": round(float(o["sat"][0, i].item()), 3),
        "lux": round(float(o["bri"][0, i].item()), 3),
        "act": round(float(o["act"][0, i].item()), 3),
    } for i in order if sal[i] > 0.01]
    return {"cell": pos, "index": cell, "slots": slots}


# ----------------------------------------------------------------------------
# HTTP layer
# ----------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):                  # quieter
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_index(self):
        with open(os.path.join(HERE, "viewer.html"), encoding="utf-8") as f:
            return f.read().encode()

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            return self._send(200, self._serve_index(), "text/html")
        if u.path == "/query":
            q = parse_qs(u.query)
            setup = q.get("setup", ["free"])[0]
            text = q.get("text", [""])[0]
            if setup not in NETS or not text.strip():
                return self._send(400, json.dumps({"error": "bad request"}))
            return self._send(200, json.dumps(forward(setup, text)))
        if u.path == "/models":
            return self._send(200, json.dumps(["free", "distill", "anchor"]))
        if u.path == "/scores":
            p = os.path.join(OUT, "scores.json")
            if os.path.exists(p):
                return self._send(200, open(p, encoding="utf-8").read())
            return self._send(200, "{}")
        if u.path == "/drawings":                      # list saved drawings
            d = os.path.join(OUT, "drawings")
            names = sorted(os.path.splitext(f)[0] for f in os.listdir(d)
                           if f.endswith(".json")) if os.path.isdir(d) else []
            return self._send(200, json.dumps(names))
        if u.path == "/drawing":                       # one drawing by name
            q = parse_qs(u.query)
            name = q.get("name", [""])[0]
            safe = os.path.basename(name)
            p = os.path.join(OUT, "drawings", safe + ".json")
            if safe and os.path.exists(p):
                return self._send(200, open(p, encoding="utf-8").read())
            return self._send(404, json.dumps({"error": "no such drawing"}))
        return self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/save":
            p = os.path.join(OUT, "scores.json")
            store = {}
            if os.path.exists(p):
                store = json.load(open(p, encoding="utf-8"))
            store.update(payload)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(store, f, indent=1)
            return self._send(200, json.dumps({"ok": True, "n": len(store)}))
        if self.path == "/draw":                       # save a hand drawing
            name = os.path.basename(str(payload.get("name", "")).strip())
            cells = payload.get("cells", {})
            if not name or not isinstance(cells, dict) or not cells:
                return self._send(400, json.dumps({"error": "need name + cells"}))
            d = os.path.join(OUT, "drawings")
            os.makedirs(d, exist_ok=True)
            rec = {"text": str(payload.get("text", "")),
                   "cells": cells,
                   "saved_at": time.time()}
            with open(os.path.join(d, name + ".json"), "w", encoding="utf-8") as f:
                json.dump(rec, f, indent=1)
            return self._send(200, json.dumps({"ok": True, "cells": len(cells)}))
        return self._send(404, json.dumps({"error": "not found"}))


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8123
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"voxel viewer -> http://localhost:{port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
