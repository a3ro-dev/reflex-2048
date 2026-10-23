"""Dry run of the teacher's -B label mode and one DAgger round. Fails loudly.
python3 scripts/dryrun_dagger.py <ckpt> <depth>"""
import glob, os, subprocess, sys, time
import numpy as np
import torch
sys.path.insert(0, "/root/reflex2048")
from reflex.env import all_afterstates
from reflex.train import REC, TEACHER_TO_OURS, read_labels

ckpt, depth = sys.argv[1], sys.argv[2]
D = "/root/dry2/dagger"
os.makedirs(D, exist_ok=True)
for f in glob.glob(f"{D}/*"):
    os.remove(f)
# 1. label boards the teacher already played in self-play: boards must round-trip, labels must
#    be legal in our env and mostly agree with the self-play decision (same depth, fresh cache)
raw = np.fromfile(sorted(glob.glob("/root/data/teacher4/*.bin"))[0], dtype=REC, count=200_000)
raw = raw[raw["move"] != 255][::20][:3000]
raw["ranks"].tofile(f"{D}/probe.u8")
t = time.time()
subprocess.run(["/root/teacher/2048bB", "-d", "4", "-B", f"{D}/probe.u8", "-o", f"{D}/probe.bin"],
               cwd="/root/teacher", check=True, stdout=subprocess.DEVNULL)
dt = time.time() - t
out = np.fromfile(f"{D}/probe.bin", dtype=REC)
assert len(out) == len(raw), (len(out), len(raw))
assert (out["ranks"] == raw["ranks"]).all(), "boards do not round-trip through -B"
b = torch.from_numpy(out["ranks"].astype(np.int64)).view(-1, 4, 4)
_, _, legal = all_afterstates(b)
a = torch.from_numpy(TEACHER_TO_OURS[out["move"]].astype(np.int64))
assert legal[torch.arange(len(a)), a].all(), "teacher label is illegal in our env"
agree = (out["move"] == raw["move"]).mean()
print(f"-B d4: {len(out)} boards in {dt:.1f}s ({len(out)/dt:.0f}/s), round-trip ok, labels legal, "
      f"agreement with self-play d4 {agree:.3f}")
assert agree > 0.9
# 2. one DAgger round end to end at the real labelling depth
r = subprocess.run(f"python3 -m reflex.dagger --ckpt {ckpt} --games 64 --q 0.05 --tail 50 --workers 4 "
                   f"--depth {depth} --bin 2048bB --rounds 1 --out {D} --log {D}/dagger.jsonl",
                   shell=True, cwd="/root/reflex2048", capture_output=True, text=True, timeout=900)
print(r.stdout[-800:], r.stderr[-1500:])
assert r.returncode == 0
bins = sorted(glob.glob(f"{D}/r*_w*.bin"))
assert len(bins) == 4 and not glob.glob(f"{D}/*.part") and not glob.glob(f"{D}/r*.u8"), os.listdir(D)
n = sum(len(read_labels(p)[0]) for p in bins)
print(f"DAgger round ok: {len(bins)} label files, {n} labelled states")
print("DAGGER DRY RUN OK")
