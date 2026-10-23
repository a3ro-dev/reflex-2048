"""Watch Reflex play one game in the terminal on a laptop CPU, one forward pass per move.

python -m reflex.play --ckpt reflex.pt [--seed 0] [--every 500]
"""
import argparse, time
import torch

from .env import ACTIONS, reset, spawn
from .evaluate import policy
from .model import load


def show(b, score, moves, p):
    tiles = [[(1 << int(x)) if x else 0 for x in row] for row in b.tolist()]
    print(f"\nmove {moves}  score {score}  " + "  ".join(f"{a} {q:.2f}" for a, q in zip(ACTIONS, p.tolist())))
    for row in tiles:
        print(" ".join(f"{t:>6}" if t else "     ." for t in row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="reflex.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=500, help="print the board every N moves")
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    model = load(args.ckpt, "cpu")
    gen = torch.Generator().manual_seed(args.seed)
    b, score, moves, t = reset(1, "cpu", gen), 0, 0, 0.0
    policy(model, b)                          # warm-up: builds the move lookup table once
    while True:
        s = time.perf_counter()
        p, legal, after, reward = policy(model, b)
        t += time.perf_counter() - s
        if not legal.any():
            break
        a = int(p.argmax(1))
        if moves % args.every == 0:
            show(b[0], score, moves, p[0])
        score += int(reward[0, a]); moves += 1
        b = spawn(after[:, a], gen)
    show(b[0], score, moves, p[0])
    print(f"\ngame over: score {score}, {moves} moves, max tile {1 << int(b.max())}, "
          f"{t / max(moves, 1) * 1000:.2f} ms per move on cpu (features + forward)")


if __name__ == "__main__":
    main()
