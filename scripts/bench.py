"""Where does training time go? python3 scripts/bench.py [batch] [d:layers ...]"""
import sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, "/root/reflex2048")
from reflex.env import action_features
from reflex.model import Reflex
from reflex.train import augment

dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
B = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
cfgs = [tuple(map(int, c.split(":"))) for c in sys.argv[2:]] or [(256, 8)]
b = torch.randint(0, 12, (B, 4, 4), device=dev); a = torch.randint(0, 4, (B,), device=dev)


def t(f, n=20):
    for _ in range(3): f()
    torch.cuda.synchronize(); s = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time() - s) / n * 1000


print(f"bs {B}  augment {t(lambda: augment(b, a)):.1f} ms  features {t(lambda: action_features(b)):.1f} ms")
for d, L in cfgs:
    for name in ("eager", "compiled"):
        m = Reflex(d=d, layers=L, heads=d // 32).to(dev)
        f = torch.compile(m) if name == "compiled" else m
        opt = torch.optim.AdamW(m.parameters(), 1e-4, fused=True)
        def step():
            bb, aa = augment(b, a)
            feats, legal, _, _ = action_features(bb)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg, v = f(bb, feats, legal)
            loss = -(F.log_softmax(lg, -1).nan_to_num(0).gather(1, aa[:, None])).mean() + v.mean() * 0
            opt.zero_grad(); loss.backward(); opt.step()
        ms = t(step)
        print(f"d{d} L{L} {sum(p.numel() for p in m.parameters())/1e6:.1f}M {name:9s} full step {ms:.1f} ms "
              f"-> {B/ms*1000:,.0f} samples/s  mem {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)
