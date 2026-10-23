"""End-to-end dry run on the pod: teacher dump -> parse -> replay check -> train ->
evaluate -> one DAgger round. Fails loudly at the first broken link."""
import glob, os, subprocess, sys, time
import numpy as np
import torch

sys.path.insert(0, "/root/reflex2048")
from reflex.env import all_afterstates
from reflex.train import REC, TEACHER_TO_OURS, read_games

D = "/root/dry"
raw = np.fromfile(f"{D}/t.bin", dtype=REC)
assert REC.itemsize == 32 and len(raw) > 100
# 1. replay: our move applied to record i must equal record i+1 minus exactly one spawned 2/4
bad, n = 0, 0
for i in range(len(raw) - 1):
    if raw["move"][i] == 255 or raw["move"][i + 1] == 255:
        continue
    b = torch.from_numpy(raw["ranks"][i].astype(np.int64)).view(1, 4, 4)
    nxt = torch.from_numpy(raw["ranks"][i + 1].astype(np.int64)).view(4, 4)
    after, rew, legal = all_afterstates(b)
    a = TEACHER_TO_OURS[raw["move"][i]]
    diff = (after[0, a] != nxt)
    ok = legal[0, a] and diff.sum() == 1 and after[0, a][diff].item() == 0 and nxt[diff].item() in (1, 2)
    ok = ok and raw["score"][i + 1] - raw["score"][i] == rew[0, a].item()
    bad += not ok; n += 1
print(f"replay check: {n - bad}/{n} transitions match (moves, spawns, rewards)")
assert bad == 0
b, a, v, h = read_games(f"{D}/t.bin")
assert len(b) == len(a) == len(v) and (v >= 0).all() and v[0] > v[-1]
print(f"parse ok: {len(b)} states, first rtg {v[0]:.0f}, held-out {h.sum()}")

def run(cmd, timeout):
    t = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, cwd="/root/reflex2048")
    print(f"$ {cmd}\n  exit {r.returncode} in {time.time()-t:.0f}s\n  " + "\n  ".join((r.stdout + r.stderr).strip().splitlines()[-6:]))
    assert r.returncode == 0

os.makedirs(f"{D}/runs", exist_ok=True)
run(f"python3 -m reflex.train --data '{D}/t.bin' --out {D}/runs/reflex --steps 1000 --bs 16384", 600)
run(f"python3 -c \"import json;rows=[json.loads(l) for l in open('{D}/runs/reflex/train.jsonl')];print(rows[0]['ce'], rows[-1]['ce'], rows[-1].get('val_acc'))\"", 60)
run(f"python3 -m reflex.evaluate --ckpt {D}/runs/reflex/latest.pt --games 64 --out {D}/runs/eval.jsonl", 900)
print("DRY RUN OK")
