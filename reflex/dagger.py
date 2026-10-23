"""Self-correction loop (DAgger): the student plays on GPU, the teacher relabels the
states the student actually reaches, and those labels join the training stream.
This fixes the compounding-error problem of pure imitation: the student learns what
to do in the positions its own mistakes lead to. States are sampled across whole games,
plus every board in the last `tail` moves before each death (where the mistakes are).

python -m reflex.dagger --teacher-dir /root/teacher --bin 2048b --depth 3 --workers 10
"""
import argparse, os, subprocess, time
import numpy as np
import torch

from .env import reset, spawn
from .evaluate import policy
from .model import load


@torch.no_grad()
def collect(model, games, q, tail, seed, max_steps=60000):
    """Student plays greedily. Returns visited boards (n,16) uint8 and the games' mean score."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    b = reset(games, "cuda", gen)
    alive = torch.ones(games, dtype=torch.bool, device="cuda")
    ring = torch.zeros(games, tail, 16, dtype=torch.uint8, device="cuda")
    score = torch.zeros(games, dtype=torch.int64, device="cuda")
    out, step = [], 0
    while alive.any() and step < max_steps:
        idx = alive.nonzero().squeeze(1)
        p, legal, after, reward = policy(model, b[idx])
        dead = ~legal.any(1)
        cur = b[idx].flatten(1).to(torch.uint8)
        ring[idx, step % tail] = cur
        take = (torch.rand(len(idx), device="cuda", generator=gen) < q) & ~dead
        out.append(cur[take])
        if dead.any():
            out.append(ring[idx[dead]].flatten(0, 1))
        a = p.argmax(1)
        ar = torch.arange(len(idx), device="cuda")
        live = idx[~dead]
        b[live] = spawn(after[ar, a][~dead], gen)
        score[live] += reward[ar, a][~dead]
        alive[idx[dead]] = False
        step += 1
    x = torch.cat(out)
    x = x[x.any(1)]                                   # ring slots never written are all-zero
    return x.cpu().numpy(), score.float().mean().item()


def label(boards, teacher_dir, binary, depth, workers, prefix):
    """Teacher labels boards through the patched binary's -B mode, in parallel workers.
    Output files appear atomically as <prefix>_w<k>.bin (the trainer only reads .bin)."""
    procs = []
    for w, c in enumerate(np.array_split(boards, workers)):
        src, dst = f"{prefix}_w{w}.u8", f"{prefix}_w{w}.part"
        c.tofile(src)
        if os.path.exists(dst):
            os.remove(dst)                            # -o appends
        procs.append((subprocess.Popen(["nice", "-n", "5", f"./{binary}", "-d", str(depth), "-B", src, "-o", dst],
                                       cwd=teacher_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL), src, dst))
    for p, src, dst in procs:
        if p.wait() != 0:
            raise RuntimeError(f"teacher exited {p.returncode} on {src}")
        os.remove(src)
        os.replace(dst, dst[:-5] + ".bin")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/root/runs/reflex/latest.pt")
    ap.add_argument("--teacher-dir", default="/root/teacher")
    ap.add_argument("--bin", default="2048bB", help="teacher build with the -B label mode")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--games", type=int, default=1024)
    ap.add_argument("--q", type=float, default=0.03, help="sampling rate of visited states")
    ap.add_argument("--tail", type=int, default=200, help="boards kept before each death")
    ap.add_argument("--cap", type=int, default=200_000, help="max states labelled per round")
    ap.add_argument("--rounds", type=int, default=10**9)
    ap.add_argument("--seed", type=int, default=1000, help="game seed of round 0 (use fresh seeds for eval sets)")
    ap.add_argument("--out", default="/root/data/dagger")
    ap.add_argument("--log", default="/root/runs/dagger.jsonl")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rnd = len([f for f in os.listdir(args.out) if f.endswith("_w0.bin")])   # resume numbering
    for _ in range(args.rounds):
        if os.path.exists("/root/runs/WRAP_UP"):
            break
        if not os.path.exists(args.ckpt):
            time.sleep(30); continue
        t0 = time.time()
        model = load(args.ckpt, "cuda")
        step = torch.load(args.ckpt, weights_only=False, map_location="cpu").get("step", 0)
        boards, mean = collect(model, args.games, args.q, args.tail, seed=args.seed + rnd)
        if len(boards) > args.cap:
            boards = boards[np.random.default_rng(rnd).choice(len(boards), args.cap, replace=False)]
        t1 = time.time()
        label(boards, args.teacher_dir, args.bin, args.depth, args.workers, f"{args.out}/r{rnd:04d}")
        rec = dict(t=time.time(), round=rnd, ckpt_step=step, states=len(boards), student_mean=mean,
                   collect_s=t1 - t0, label_s=time.time() - t1)
        with open(args.log, "a") as f:
            f.write(__import__("json").dumps(rec) + "\n")
        print(rec, flush=True)
        rnd += 1


if __name__ == "__main__":
    main()
