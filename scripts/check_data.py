"""Verify teacher files after the seed-sort demux: every within-game transition must replay
exactly in our env (move, one spawned 2/4, reward), and print teacher score stats.
python3 scripts/check_data.py "/root/data/teacher/*.bin" [max_files]"""
import glob, json, sys
import numpy as np
import torch
sys.path.insert(0, "/root/reflex2048")
from reflex.env import all_afterstates
from reflex.train import REC, TEACHER_TO_OURS, split_games

files = sorted(glob.glob(sys.argv[1]))[: int(sys.argv[2]) if len(sys.argv) > 2 else None]
tot = bad = 0
finals, maxt = [], []
for p in files:
    s, gid, final = split_games(np.fromfile(p, dtype=REC))
    end = s["move"] == 255
    i = np.flatnonzero((gid[:-1] == gid[1:]) & ~end[:-1] & ~end[1:])
    for c in np.array_split(i, max(1, len(i) // 2_000_000)):
        b = torch.from_numpy(s["ranks"][c].astype(np.int64)).cuda().view(-1, 4, 4)
        nxt = torch.from_numpy(s["ranks"][c + 1].astype(np.int64)).cuda().view(-1, 16)
        a = torch.from_numpy(TEACHER_TO_OURS[s["move"][c]].astype(np.int64)).cuda()
        after, rew, legal = all_afterstates(b)
        ar = torch.arange(len(c), device="cuda")
        aft = after[ar, a].view(-1, 16)
        diff = aft != nxt
        ok = legal[ar, a] & (diff.sum(1) == 1) & ((aft * diff).sum(1) == 0) & \
             ((nxt * diff).sum(1) >= 1) & ((nxt * diff).sum(1) <= 2)
        ok &= torch.from_numpy(s["score"][c + 1] - s["score"][c]).cuda() == rew[ar, a]
        tot += len(c); bad += int((~ok).sum())
    finals.append(s["score"][end]); maxt.append(s["ranks"][end].max(1))
    print(f"{p}: {len(s)} records, {end.sum()} games, mismatches so far {bad}/{tot}", flush=True)
f, m = np.concatenate(finals), np.concatenate(maxt)
res = dict(files=len(files), transitions=tot, mismatches=bad, games=len(f), mean=float(f.mean()),
           median=float(np.median(f)), max=int(f.max()), **{f"rate_{2**k}": float((m >= k).mean()) for k in range(11, 17)})
print(json.dumps(res))
