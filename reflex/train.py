"""Distill the teacher into Reflex on GPU. Streams in new teacher files as they grow.

Loss = cross-entropy to the teacher's move (a strictly proper scoring rule, so the
optimum is the calibrated probability that the teacher picks each move)
+ value regression on log return-to-go. Random D4 symmetry per sample.
"""
import argparse, glob, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F

from .env import action_features, sym, sym_perm
from .model import Reflex

REC = np.dtype([("ranks", "u1", 16), ("move", "u1"), ("vtype", "u1"), ("pad", "<u2"),
                ("value", "<f4"), ("score", "<i4"), ("seed", "<u4")])
TEACHER_TO_OURS = np.array([0, 2, 3, 1])   # teacher up,left,right,down -> ours UP,DOWN,LEFT,RIGHT


def read_games(path):
    """Complete games only. Returns boards (n,16) u1, moves (n,) ours, rtg (n,) f4."""
    raw = np.fromfile(path, dtype=REC, count=os.path.getsize(path) // REC.itemsize)
    ends = np.flatnonzero(raw["move"] == 255)
    if len(ends) == 0:
        return None
    raw = raw[: ends[-1] + 1]
    game = np.cumsum(np.r_[0, (raw["move"] == 255)[:-1]])          # game index per row
    final = raw["score"][raw["move"] == 255]
    keep = raw["move"] != 255
    r = raw[keep]
    rtg = (final[game[keep]] - r["score"]).astype(np.float32)
    return r["ranks"], TEACHER_TO_OURS[r["move"]], rtg, r["seed"] % 50 == 0   # whole games held out


def read_labels(path):
    """DAgger labels: teacher move for student-visited boards; no return known."""
    raw = np.fromfile(path, dtype=REC)
    r = raw[raw["move"] != 255]
    return r["ranks"], TEACHER_TO_OURS[r["move"]], np.full(len(r), np.nan, np.float32), np.zeros(len(r), bool)


class Stream:
    """All teacher data on the GPU; reloads files that grew."""
    def __init__(self, pattern, device):
        self.pattern, self.device, self.sizes, self.parts = pattern, device, {}, {}
        self.refresh()

    def refresh(self):
        changed = False
        for p in sorted(sum((glob.glob(x) for x in self.pattern.split(",")), [])):
            s = os.path.getsize(p)
            if self.sizes.get(p) != s:
                g = read_labels(p) if "/dagger/" in p else read_games(p)
                if g is not None:
                    self.parts[p] = g
                    self.sizes[p] = s
                    changed = True
        if changed or not hasattr(self, "b"):
            if not self.parts:
                raise SystemExit(f"no complete teacher games in {self.pattern}")
            cat = lambda i: np.concatenate([v[i] for v in self.parts.values()])
            self.b = torch.from_numpy(cat(0).astype(np.int64)).to(self.device).view(-1, 4, 4)
            self.a = torch.from_numpy(cat(1).astype(np.int64)).to(self.device)
            self.v = torch.from_numpy(cat(2)).to(self.device)
            h = torch.from_numpy(cat(3)).to(self.device)
            self.tr, self.va = (~h).nonzero().squeeze(1), h.nonzero().squeeze(1)
        return len(self.a)

    def batch(self, n, val=False):
        pool = self.va if val and len(self.va) else self.tr
        i = pool[torch.randint(0, len(pool), (n,), device=self.device)]
        return self.b[i], self.a[i], self.v[i]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/root/data/teacher/*.bin,/root/data/dagger/*.bin")
    ap.add_argument("--out", default="/root/runs/reflex")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--init", default="")
    args = ap.parse_args()
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(args.out, exist_ok=True)
    model = Reflex(d=args.d, layers=args.layers, heads=args.d // 32).to(dev)
    if args.init:
        model.load_state_dict(torch.load(args.init, weights_only=False)["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05, betas=(0.9, 0.98))
    sched = lambda s: min(1, s / 2000) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1, s / args.steps))))
    data = Stream(args.data, dev)
    log = open(f"{args.out}/train.jsonl", "a")
    t0, seen = time.time(), 0
    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = args.lr * sched(step)
        b, a, v = data.batch(args.bs)
        b, a = augment(b, a)
        feats, legal, _, _ = action_features(b)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, val = model(b, feats, legal)
        lp = F.log_softmax(logits, -1).nan_to_num(0)          # -inf*0 on illegal moves
        ce = -(lp.gather(1, a[:, None])).mean()
        ent = -(lp.exp() * lp).sum(-1).mean()
        vm = torch.isfinite(v)
        vloss = F.smooth_l1_loss(val[vm], torch.log1p(v[vm]) / 10) if vm.any() else val.sum() * 0
        loss = ce + 0.25 * vloss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        seen += len(b)
        if step % 100 == 0:
            rec = dict(t=time.time(), step=step, loss=loss.item(), ce=ce.item(), vloss=vloss.item(),
                       entropy=ent.item(), grad_norm=gn.item(), lr=opt.param_groups[0]["lr"],
                       samples_per_s=seen / (time.time() - t0), dataset=len(data.a))
            if step % 1000 == 0:
                model.eval()
                with torch.no_grad():
                    hb, ha, _ = data.batch(32768, val=True)
                    f2, l2, _, _ = action_features(hb)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        hl, _ = model(hb, f2, l2)
                    p = F.softmax(hl, -1).nan_to_num(0)
                    ok = p.argmax(-1) == ha
                    onehot = F.one_hot(ha, 4).float()
                    rec.update(val_acc=ok.float().mean().item(), val_ece=calib(p, ok),
                               val_brier=((p - onehot) ** 2).sum(-1).mean().item(),
                               val_nll=-(p.gather(1, ha[:, None]).clamp(min=1e-9).log()).mean().item())
                model.train()
                torch.save(dict(cfg=model.cfg, model=model.state_dict(), step=step),
                           f"{args.out}/latest.pt.tmp")
                os.replace(f"{args.out}/latest.pt.tmp", f"{args.out}/latest.pt")
            if step % 5000 == 0:
                rec["dataset"] = data.refresh()
            log.write(json.dumps(rec) + "\n"); log.flush()
            if os.path.exists("/root/runs/WRAP_UP"):
                break
    torch.save(dict(cfg=model.cfg, model=model.state_dict(), step=step), f"{args.out}/latest.pt")


if __name__ == "__main__":
    main()
