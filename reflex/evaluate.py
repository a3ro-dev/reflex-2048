"""Play many full games in parallel on GPU with the pure policy (one forward pass
per move, no search). Watches a checkpoint and re-evaluates whenever it changes.

python -m reflex.evaluate --ckpt runs/reflex/latest.pt --games 1024 [--watch] [--sym]
"""
import argparse, json, os, time
import torch
import torch.nn.functional as F

from .env import action_features, reset, score_of, spawn, sym, sym_perm
from .model import load

TILES = [2048, 4096, 8192, 16384, 32768, 65536]


@torch.no_grad()
def policy(model, b, use_sym=False):
    feats, legal, after, reward = action_features(b)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=b.is_cuda):
        logits, _ = model(b, feats, legal)
    p = F.softmax(logits, -1)
    if use_sym:  # average the distribution over the 8 symmetric views (still one batched pass)
        P = sym_perm().to(b.device)
        bs = torch.cat([sym(b, k) for k in range(1, 8)])
        f2, l2, _, _ = action_features(bs)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=b.is_cuda):
            lg, _ = model(bs, f2, l2)
        p2 = F.softmax(lg, -1).view(7, len(b), 4)
        for k in range(1, 8):
            p = p + p2[k - 1][:, P[k]]
        p = p / 8
    return p.nan_to_num(0), legal, after, reward


@torch.no_grad()
def lookahead(model, b, chunk=65536):
    """1-ply expectimax on the value head: Q(s,a) = r_a + E[V(s')] with the expectation exact
    over every spawn (a 2 w.p. .9 or a 4 w.p. .1 in each empty cell of the afterstate).
    Not search-free: about 8x(empty cells) forward passes per move. Returns q (n,4), -inf if illegal."""
    _, legal, after, reward = action_features(b)
    flat = after.flatten(2)                                        # (n,4,16)
    empty = flat == 0
    eye = torch.eye(16, dtype=flat.dtype, device=b.device)
    child = flat[:, :, None, None] + eye[:, None] * torch.tensor([1, 2], device=b.device)[:, None]  # (n,4,16,2,16)
    w = empty[..., None] * torch.tensor([.9, .1], device=b.device) / empty.sum(-1, keepdim=True)[..., None].clamp(min=1)
    valid = (w > 0) & legal[:, :, None, None]
    cb = child[valid].view(-1, 4, 4)
    vals = []
    for i in range(0, len(cb), chunk):
        f, l, _, _ = action_features(cb[i:i + chunk])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=b.is_cuda):
            _, v = model(cb[i:i + chunk], f, l)
        vals.append(torch.expm1(10 * v.float()).clamp(min=0))   # value head predicts log1p(return)/10
    ev = torch.zeros(valid.shape, device=b.device)
    ev[valid] = torch.cat(vals) if vals else ev[valid]
    q = reward.float() + (ev * w).sum((2, 3))
    return q.masked_fill(~legal, float("-inf")), legal, after, reward


@torch.no_grad()
def play(model, n, device="cuda", use_sym=False, live=None, seed=0, ply1=False):
    gen = torch.Generator(device=device).manual_seed(seed)
    b = reset(n, device, gen)
    score = torch.zeros(n, dtype=torch.int64, device=device)
    moves = torch.zeros(n, dtype=torch.int64, device=device)
    alive = torch.ones(n, dtype=torch.bool, device=device)
    t0, steps = time.time(), 0
    while alive.any():
        idx = alive.nonzero().squeeze(1)
        if ply1:
            p, legal, after, reward = lookahead(model, b[idx])
        else:
            p, legal, after, reward = policy(model, b[idx], use_sym)
        dead = ~legal.any(1)
        a = p.argmax(1)
        nb = after[torch.arange(len(idx)), a]
        r = reward[torch.arange(len(idx)), a]
        live_idx = idx[~dead]
        b[live_idx] = spawn(nb[~dead], gen)
        score[live_idx] += r[~dead]
        moves[live_idx] += 1
        alive[idx[dead]] = False
        steps += 1
        if live and steps % 25 == 0:
            j = int(score.argmax())
            json.dump(dict(board=b[j].tolist(), score=int(score[j]), moves=int(moves[j]),
                           alive=int(alive.sum()), games=n, step=steps), open(live + ".tmp", "w"))
            os.replace(live + ".tmp", live)
    dt = time.time() - t0
    mx = b.flatten(1).max(1).values
    res = dict(games=n, mean_score=score.float().mean().item(), median_score=score.float().median().item(),
               max_score=int(score.max()), mean_moves=moves.float().mean().item(),
               moves_per_s=moves.sum().item() / dt, seconds=dt, sym=use_sym, ply1=ply1)
    for t in TILES:
        res[f"rate_{t}"] = (mx >= t.bit_length() - 1).float().mean().item()
    best = int(score.argmax())
    res["best_board"] = b[best].tolist()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/root/runs/reflex/latest.pt")
    ap.add_argument("--games", type=int, default=1024)
    ap.add_argument("--sym", action="store_true")
    ap.add_argument("--ply1", action="store_true", help="1-ply expectimax on the value head (not search-free)")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--out", default="/root/runs/eval.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-interval", type=float, default=0, help="watch mode: seconds between evaluations "
                    "(evaluation shares the GPU with training)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    last, best = None, -1
    while True:
        if os.path.exists(args.ckpt) and os.path.getmtime(args.ckpt) != last:
            last, last_run = os.path.getmtime(args.ckpt), time.time()
            model = load(args.ckpt, args.device)
            step = torch.load(args.ckpt, weights_only=False, map_location="cpu").get("step", 0)
            res = play(model, args.games, device=args.device, use_sym=args.sym, seed=args.seed, ply1=args.ply1,
                       live=os.path.join(os.path.dirname(args.out), "live.json"))
            res.update(t=time.time(), step=step, ckpt=args.ckpt)
            if args.watch and res["mean_score"] > best:   # keep the best checkpoint seen so far
                best = res["mean_score"]
                torch.save(dict(cfg=model.cfg, model=model.state_dict(), step=step, eval=res),
                           os.path.join(os.path.dirname(args.ckpt), "best.pt"))
            with open(args.out, "a") as f:
                f.write(json.dumps(res) + "\n")
            print(json.dumps({k: v for k, v in res.items() if k != "best_board"}), flush=True)
        if not args.watch or os.path.exists("/root/runs/WRAP_UP"):
            break
        time.sleep(max(10, args.min_interval - (time.time() - last_run)) if last else 10)


if __name__ == "__main__":
    main()
