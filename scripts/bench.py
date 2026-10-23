"""Where does training time go? python3 scripts/bench.py"""
import sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, "/root/reflex2048")
from reflex.env import action_features
from reflex.model import Reflex
from reflex.train import augment

dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
B = 16384
b = torch.randint(0, 12, (B, 4, 4), device=dev); a = torch.randint(0, 4, (B,), device=dev)


def t(f, n=20):
    f(); torch.cuda.synchronize(); s = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time() - s) / n * 1000


print(f"augment   {t(lambda: augment(b, a)):.1f} ms")
print(f"features  {t(lambda: action_features(b)):.1f} ms")
feats, legal, _, _ = action_features(b)
for name, m in [("eager", Reflex().to(dev)), ("compiled", torch.compile(Reflex().to(dev)))]:
    opt = torch.optim.AdamW(m.parameters(), 1e-4)
    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg, v = m(b, feats, legal)
        loss = -(F.log_softmax(lg, -1).nan_to_num(0).gather(1, a[:, None])).mean() + v.mean() * 0
        opt.zero_grad(); loss.backward(); opt.step()
    ms = t(step)
    print(f"{name:9s} fwd+bwd {ms:.1f} ms -> {B/ms*1000:,.0f} samples/s")
