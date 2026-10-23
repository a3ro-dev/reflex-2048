"""Self-correction loop (DAgger): the student plays on GPU, the teacher relabels the
states the student actually reaches, and those labels join the training stream.
This fixes the compounding-error problem of pure imitation: the student learns what
to do in the positions its own mistakes lead to.

python -m reflex.dagger --teacher-dir /root/teacher --bin 2048b --depth 3 --workers 24
"""
import argparse, os, subprocess, time
import numpy as np
import torch

from .env import action_features, reset, spawn
from .evaluate import policy
from .model import load


@torch.no_grad()
def collect(model, games, every, cap, seed):
    """Student plays greedily; returns visited boards (n,16) uint8, subsampled."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    b = reset(games, "cuda", gen)
    alive = torch.ones(games, dtype=torch.bool, device="cuda")
    out, n, step = [], 0, 0
    while alive.any() and n < cap:
        idx = alive.nonzero().squeeze(1)
        p, legal, after, _ = policy(model, b[idx])
        dead = ~legal.any(1)
        if step % every == 0:
            keep = b[idx[~dead]].flatten(1).to(torch.uint8).cpu().numpy()
            out.append(keep); n += len(keep)
        a = p.argmax(1)
        nb = after[torch.arange(len(idx)), a]
        b[idx[~dead]] = spawn(nb[~dead], gen)
        alive[idx[dead]] = False
        step += 1
    return np.concatenate(out)[:cap]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/root/runs/reflex/latest.pt")
    ap.add_argument("--teacher-dir", default="/root/teacher")
    ap.add_argument("--bin", default="2048b")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--games", type=int, default=512)
    ap.add_argument("--per-round", type=int, default=400_000)
    ap.add_argument("--out", default="/root/data/dagger")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rnd = 0
    while not os.path.exists("/root/runs/WRAP_UP"):
        if not os.path.exists(args.ckpt):
            time.sleep(30); continue
        model = load(args.ckpt, "cuda")
        boards = collect(model, args.games, every=3, cap=args.per_round, seed=1000 + rnd)
        chunks = np.array_split(boards, args.workers)
        procs = []
        for w, c in enumerate(chunks):
            src = f"{args.out}/r{rnd}_w{w}.u8"
            c.tofile(src)
            dst = f"{args.out}/r{rnd}_w{w}.part"
            procs.append((subprocess.Popen([f"./{args.bin}", "-d", str(args.depth), "-B", src, "-o", dst],
                                           cwd=args.teacher_dir, stdout=subprocess.DEVNULL), src, dst))
        for p, src, dst in procs:
            p.wait()
            os.remove(src)
            os.replace(dst, dst[:-5] + ".bin")   # trainer only picks up finished files
        print(f"round {rnd}: labelled {len(boards)} student states", flush=True)
        rnd += 1


if __name__ == "__main__":
    main()
