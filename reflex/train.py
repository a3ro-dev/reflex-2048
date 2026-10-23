"""Distill the teacher into Reflex on GPU. Streams in new teacher/DAgger files as they grow.

Loss = cross-entropy to the teacher's move (a strictly proper scoring rule, so the
optimum is the calibrated probability that the teacher picks each move)
+ value regression on log return-to-go. Random D4 symmetry per sample.
LR follows a cosine over wall-clock time (--hours), so the schedule fits the budget.
"""
import argparse, glob, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F

from .env import action_features, sym, sym_perm
from .model import Reflex

REC = np.dtype([("ranks", "u1", 16), ("move", "u1"), ("vtype", "u1"), ("pad", "<u2"),
                ("value", "<f4"), ("score", "<i4"), ("seed", "<u4")])
TEACHER_TO_OURS = np.array([0, 2, 3, 1], np.int8)   # teacher up,left,right,down -> ours UP,DOWN,LEFT,RIGHT


def split_games(raw):
    """Group records into games. Several writers may have appended to one file (each
    write is a whole number of records), so records of one game can be interleaved with
    another game. Every game has its own seed and same-seed games never run at the same
    time, so a stable sort by seed makes each game contiguous again.
    Returns (records sorted, game id per record, final score per record or -1)."""
    s = raw[np.argsort(raw["seed"], kind="stable")]
    end = s["move"] == 255
    start = np.r_[True, (s["seed"][1:] != s["seed"][:-1]) | end[:-1]]
    gid = np.cumsum(start) - 1
    final = np.full(gid[-1] + 1, -1, np.int64)
    final[gid[end]] = s["score"][end]
    # a game whose score ever decreases was interleaved with another: no value target for it
    dec = (np.diff(s["score"].astype(np.int64)) < 0) & (gid[1:] == gid[:-1])
    final[gid[1:][dec]] = -1
    return s, gid, final[gid]


def read_games(path):
    """Teacher self-play. Returns boards (n,16) u1, moves (n,) i1 ours, rtg (n,) f4 (nan if
    the game is unfinished), held-out mask (whole games, seed % 50 == 0)."""
    raw = np.fromfile(path, dtype=REC, count=os.path.getsize(path) // REC.itemsize)
    if len(raw) == 0:
        return None
    s, _, final = split_games(raw)
    keep = s["move"] != 255
    r, f = s[keep], final[keep]
    rtg = np.where(f >= 0, f - r["score"], np.nan).astype(np.float32)
    return r["ranks"], TEACHER_TO_OURS[r["move"]], rtg, r["seed"] % 50 == 0


def read_labels(path):
    """DAgger labels: teacher move for student-visited boards; no return known. 2% held out."""
    raw = np.fromfile(path, dtype=REC)
    r = raw[raw["move"] != 255]
    return (r["ranks"], TEACHER_TO_OURS[r["move"]], np.full(len(r), np.nan, np.float32),
            np.arange(len(r)) % 50 == 0)


class Pool:
    """One set of files resident on the GPU (boards as uint8); reloads files that changed."""
    def __init__(self, pattern, reader, device):
        self.pattern, self.reader, self.device, self.sizes, self.parts = pattern, reader, device, {}, {}
        self.n = 0

    def refresh(self):
        changed = False
        for p in sorted(sum((glob.glob(x) for x in self.pattern.split(",") if x), [])):
            s = os.path.getsize(p)
            if self.sizes.get(p) != s:
                g = self.reader(p)
                if g is not None:
                    self.parts[p], self.sizes[p], changed = g, s, True
        if changed:
            cat = lambda i: torch.from_numpy(np.concatenate([v[i] for v in self.parts.values()]))
            self.b = None
            torch.cuda.empty_cache()
            self.b = cat(0).to(self.device)
            self.a = cat(1).to(self.device).long()
            self.v = cat(2).to(self.device)
            h = cat(3).to(self.device)
            self.tr, self.va = (~h).nonzero().squeeze(1), h.nonzero().squeeze(1)
            self.n = len(self.a)
        return self.n

    def sample(self, n, val=False):
        pool = self.va if val else self.tr
        i = pool[torch.randint(0, len(pool), (n,), device=self.device)]
        return self.b[i].long().view(-1, 4, 4), self.a[i], self.v[i]


_CELLS = {}


def augment(b, a):
    """Random D4 symmetry per sample, one gather."""
    dev = b.device
    if dev not in _CELLS:
        idx = torch.arange(16).view(1, 4, 4)
        _CELLS[dev] = (torch.stack([sym(idx, k).flatten() for k in range(8)]).to(dev), sym_perm().to(dev))
    cells, P = _CELLS[dev]
    k = torch.randint(0, 8, (len(b),), device=dev)
    return b.flatten(1).gather(1, cells[k]).view(-1, 4, 4), P[k, a]


def calib(probs, correct, bins=15):
    """Expected calibration error of top-1 confidence vs. agreement with teacher."""
    conf = probs.max(-1).values
    ece = torch.zeros((), device=probs.device)
    for i in range(bins):
        m = (conf > i / bins) & (conf <= (i + 1) / bins)
        if m.any():
            ece += m.float().mean() * (conf[m].mean() - correct[m].float().mean()).abs()
    return ece.item()


@torch.no_grad()
def heldout(model, pool, n=32768):
    b, a, _ = pool.sample(n, val=True)
    f, l, _, _ = action_features(b)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lg, _ = model(b, f, l)
    p = F.softmax(lg, -1).nan_to_num(0)
    ok = p.argmax(-1) == a
    return dict(acc=ok.float().mean().item(), ece=calib(p, ok),
                brier=((p - F.one_hot(a, 4).float()) ** 2).sum(-1).mean().item(),
                nll=-(p.gather(1, a[:, None]).clamp(min=1e-9).log()).mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", default="d3:/root/data/teacher/*.bin:0.4,d4:/root/data/teacher4/*.bin:0.3,"
                    "dag:/root/data/dagger/*.bin:0.3",
                    help="name:glob:weight,... batch shares, renormalised over pools that have data; "
                         "pools named dag* hold DAgger labels")
    ap.add_argument("--out", default="/root/runs/reflex")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--bs", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hours", type=float, default=1.0, help="wall-clock length of the cosine schedule")
    ap.add_argument("--steps", type=int, default=10**9)
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--refresh", type=int, default=4000, help="steps between rescans of growing files")
    ap.add_argument("--init", default="")
    args = ap.parse_args()
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(args.out, exist_ok=True)
    model = Reflex(d=args.d, layers=args.layers, heads=args.d // 32).to(dev)
    if args.init:
        model.load_state_dict(torch.load(args.init, weights_only=False)["model"])
    fwd = torch.compile(model) if args.compile else model
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05, betas=(0.9, 0.98), fused=True)
    pools = []
    for item in args.mix.split(","):
        name, rest = item.split(":", 1)
        pattern, w = rest.rsplit(":", 1)
        pools.append((name, float(w), Pool(pattern, read_labels if name.startswith("dag") else read_games, dev)))
    for _, _, p in pools:
        p.refresh()
    if not any(p.n for _, _, p in pools):
        raise SystemExit(f"no data in {args.mix}")
    log = open(f"{args.out}/train.jsonl", "a")
    log.write(json.dumps(dict(t=time.time(), event="start", cfg=model.cfg, args=vars(args),
                              params=sum(p.numel() for p in model.parameters()))) + "\n")
    t0, seen = time.time(), 0
    for step in range(1, args.steps + 1):
        prog = (time.time() - t0) / 3600 / args.hours
        if prog >= 1:
            break
        for g in opt.param_groups:
            g["lr"] = args.lr * min(1, step / 1000) * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * prog)))
        live = [(w, p) for _, w, p in pools if p.n and len(p.tr)]
        tot = sum(w for w, _ in live)
        counts = [int(args.bs * w / tot) for w, _ in live]
        counts[0] += args.bs - sum(counts)
        parts = [p.sample(c) for c, (_, p) in zip(counts, live)]
        b, a, v = (torch.cat(x) for x in zip(*parts))
        b, a = augment(b, a)
        feats, legal, _, _ = action_features(b)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, val = fwd(b, feats, legal)
        lp = F.log_softmax(logits, -1).nan_to_num(0)          # -inf*0 on illegal moves
        ce = -(lp.gather(1, a[:, None])).mean()
        vm = torch.isfinite(v)
        vloss = F.smooth_l1_loss(val[vm], torch.log1p(v[vm]) / 10)
        loss = ce + 0.25 * vloss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        seen += len(b)
        if step % 100 == 0:
            ent = -(lp.exp() * lp).sum(-1).mean()
            rec = dict(t=time.time(), step=step, loss=loss.item(), ce=ce.item(), vloss=vloss.item(),
                       entropy=ent.item(), grad_norm=gn.item(), lr=opt.param_groups[0]["lr"],
                       samples=seen, samples_per_s=seen / (time.time() - t0), progress=prog,
                       train_acc=(logits.argmax(-1) == a).float().mean().item(),
                       **{f"n_{name}": p.n for name, _, p in pools})
            if step % 1000 == 0:
                model.eval()
                for name, _, p in pools:
                    if p.n and len(p.va):
                        rec.update({f"{name}_val_{k}": x for k, x in heldout(model, p).items()})
                model.train()
                torch.save(dict(cfg=model.cfg, model=model.state_dict(), step=step, samples=seen),
                           f"{args.out}/latest.pt.tmp")
                os.replace(f"{args.out}/latest.pt.tmp", f"{args.out}/latest.pt")
            if step % args.refresh == 0:
                for name, _, p in pools:
                    rec[f"n_{name}"] = p.refresh()
            log.write(json.dumps(rec) + "\n"); log.flush()
            if os.path.exists("/root/runs/WRAP_UP"):
                break
    torch.save(dict(cfg=model.cfg, model=model.state_dict(), step=step, samples=seen), f"{args.out}/latest.pt")
    log.write(json.dumps(dict(t=time.time(), event="end", step=step, samples=seen)) + "\n")


if __name__ == "__main__":
    main()
