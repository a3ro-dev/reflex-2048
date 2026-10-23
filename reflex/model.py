"""Reflex: a non-autoregressive transformer that maps a 4x4 board to a calibrated
distribution over the 4 moves in one forward pass.

Tokens: 16 cell tokens (tile embedding + row/col embeddings) + 4 action tokens
(the enriched afterstate features of each move) + 1 value token. A bidirectional
pre-norm encoder (fused qkv + scaled_dot_product_attention) mixes them; each action
token emits one logit, illegal moves are masked to -inf, so the output is always a
valid distribution over legal moves only.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import ACTIONS, N_ACT_FEATS, V, action_features


class Block(nn.Module):
    def __init__(self, d, heads, ff):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ff * d), nn.GELU(), nn.Linear(ff * d, d))

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        x = x + self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, T, D))
        return x + self.ff(self.n2(x))


class Reflex(nn.Module):
    def __init__(self, d=256, layers=8, heads=8, ff=4):
        super().__init__()
        self.cfg = dict(d=d, layers=layers, heads=heads, ff=ff)
        self.tile = nn.Embedding(V, d)
        self.row = nn.Parameter(torch.randn(4, 1, d) * 0.02)
        self.col = nn.Parameter(torch.randn(1, 4, d) * 0.02)
        self.act_in = nn.Sequential(nn.Linear(N_ACT_FEATS, d), nn.GELU(), nn.Linear(d, d))
        self.act_id = nn.Parameter(torch.randn(4, d) * 0.02)
        self.val_tok = nn.Parameter(torch.randn(1, d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, heads, ff) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.pi = nn.Linear(d, 1)
        self.v = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, b, feats=None, legal=None):
        """b (B,4,4) exponents. Returns logits (B,4) with illegal = -inf, value (B,)."""
        if feats is None:
            feats, legal, _, _ = action_features(b)
        B = b.shape[0]
        cells = (self.tile(b) + self.row + self.col).flatten(1, 2)          # (B,16,d)
        acts = self.act_in(feats) + self.act_id                              # (B,4,d)
        x = torch.cat([cells, acts, self.val_tok.expand(B, 1, -1).to(cells.dtype)], 1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        logits = self.pi(x[:, 16:20]).squeeze(-1).float().masked_fill(~legal, float("-inf"))
        return logits, self.v(x[:, 20]).squeeze(-1).float()

    @torch.no_grad()
    def decide(self, board):
        """Type-safe single-board API: {'UP': p, 'DOWN': p, 'LEFT': p, 'RIGHT': p}."""
        b = torch.as_tensor(board, dtype=torch.int64).view(1, 4, 4)
        logits, _ = self(b.to(next(self.parameters()).device))
        p = F.softmax(logits, -1)[0].nan_to_num(0).tolist()
        return dict(zip(ACTIONS, p))


def load(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = Reflex(**ck["cfg"]).to(device)
    m.load_state_dict(ck["model"])
    return m.eval()


if __name__ == "__main__":
    import time
    m = Reflex().eval()
    print(f"params {sum(p.numel() for p in m.parameters())/1e6:.2f}M")
    b = torch.tensor([[1, 1, 2, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 3]])
    d = m.decide(b)
    assert abs(sum(d.values()) - 1) < 1e-5 and d["UP"] > 0
    t = time.perf_counter()
    for _ in range(50):
        m.decide(b)
    print("decision", d, f"{(time.perf_counter()-t)/50*1000:.2f} ms/forward on cpu")
