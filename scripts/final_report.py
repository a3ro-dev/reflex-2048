"""Calibration and agreement of a checkpoint on held-out teacher labels (never trained on):
top-label ECE, classwise ECE, NLL, Brier, reliability bins, and accuracy by game stage.
python3 scripts/final_report.py <ckpt> <out.json> [n_per_pool] [pools=d3,d4,dag]"""
import json, sys
import torch
import torch.nn.functional as F
sys.path.insert(0, "/root/reflex2048")
from reflex.env import action_features
from reflex.model import load
from reflex.train import Pool, read_games, read_labels

ckpt, out = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 200_000
POOLS = {"d3": ("/root/data/teacher/*.bin", read_games), "d4": ("/root/data/teacher4/*.bin", read_games),
         "dag": ("/root/data/dagger/*.bin", read_labels)}
BINS = 15
model = load(ckpt, "cuda")
res = dict(ckpt=ckpt, step=torch.load(ckpt, weights_only=False, map_location="cpu").get("step"))
torch.manual_seed(0)
for name in (sys.argv[4] if len(sys.argv) > 4 else ",".join(POOLS)).split(","):
    pattern, reader = POOLS[name]
    pool = Pool(pattern, reader, "cuda")
    if not pool.refresh() or not len(pool.va):
        continue
    idx = pool.va[torch.randperm(len(pool.va), device="cuda")[:n]]
    b, a = pool.b[idx].long().view(-1, 4, 4), pool.a[idx]
    probs = []
    with torch.no_grad():
        for i in range(0, len(b), 32768):
            f, l, _, _ = action_features(b[i:i + 32768])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg, _ = model(b[i:i + 32768], f, l)
            probs.append(F.softmax(lg, -1).nan_to_num(0))
    p = torch.cat(probs)
    y = F.one_hot(a, 4).float()
    conf, pred = p.max(-1)
    ok = (pred == a).float()
    edges = torch.linspace(0, 1, BINS + 1, device="cuda")
    rel, ece = [], 0.0
    for i in range(BINS):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.any():
            rel.append(dict(lo=edges[i].item(), conf=conf[m].mean().item(), acc=ok[m].mean().item(), n=int(m.sum())))
            ece += m.float().mean().item() * abs(rel[-1]["conf"] - rel[-1]["acc"])
    cece = 0.0                                         # classwise ECE (Kull et al. 2019)
    for k in range(4):
        pk = p[:, k]
        for i in range(BINS):
            m = (pk > edges[i]) & (pk <= edges[i + 1])
            if m.any():
                cece += m.float().mean().item() * abs(pk[m].mean().item() - y[m, k].mean().item()) / 4
    stage = b.flatten(1).max(1).values                 # max-tile exponent
    by_stage = {}
    for lo, hi, label in [(0, 10, "<2048"), (11, 12, "2048-4096"), (13, 13, "8192"), (14, 14, "16384"), (15, 17, ">=32768")]:
        m = (stage >= lo) & (stage <= hi)
        if m.any():
            by_stage[label] = dict(acc=ok[m].mean().item(), n=int(m.sum()))
    res[name] = dict(n=len(a), acc=ok.mean().item(), ece=ece, classwise_ece=cece,
                     nll=-(p.gather(1, a[:, None]).clamp(min=1e-9).log()).mean().item(),
                     brier=((p - y) ** 2).sum(-1).mean().item(), mean_conf=conf.mean().item(),
                     reliability=rel, acc_by_stage=by_stage)
    print(name, {k: v for k, v in res[name].items() if k not in ("reliability", "acc_by_stage")}, flush=True)
    del pool
    torch.cuda.empty_cache()
json.dump(res, open(out, "w"), indent=1)
