"""Batched 2048 on torch tensors. Boards are (B,4,4) int64 tile exponents (0 = empty).

Action order everywhere in this repo: 0=UP 1=DOWN 2=LEFT 3=RIGHT.
Every move is reduced to "slide the rows left" on a transformed view and one
row lookup table (18^4 rows) does the work, so thousands of games step in parallel.
"""
import torch

ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
V = 18                      # exponents 0..17 (131072 is the largest possible tile)
POW = torch.tensor([V**3, V**2, V, 1])


def _build_row_table():
    n = V**4
    new = torch.zeros(n, 4, dtype=torch.int64)
    rew = torch.zeros(n, dtype=torch.int64)
    for code in range(n):
        row = [(code // V**(3 - i)) % V for i in range(4)]
        tiles = [t for t in row if t]
        out, r, i = [], 0, 0
        while i < len(tiles):
            if i + 1 < len(tiles) and tiles[i] == tiles[i + 1] and tiles[i] < V - 1:
                out.append(tiles[i] + 1); r += 1 << (tiles[i] + 1); i += 2
            else:
                out.append(tiles[i]); i += 1
        new[code] = torch.tensor(out + [0] * (4 - len(out)))
        rew[code] = r
    return new, rew


_TABLES = {}


def tables(device):
    if device not in _TABLES:
        if "cpu" not in _TABLES:
            _TABLES["cpu"] = _build_row_table()
        _TABLES[device] = tuple(t.to(device) for t in _TABLES["cpu"])
    return _TABLES[device]


def _views(b):  # (B,4,4) -> (B,4,4,4): each action as a left-slide view
    t = b.transpose(-1, -2)
    return torch.stack([t, t.flip(-1), b, b.flip(-1)], 1)


def _unview(v):  # inverse of _views, v: (B,4,4,4)
    return torch.stack([v[:, 0].transpose(-1, -2), v[:, 1].flip(-1).transpose(-1, -2),
                        v[:, 2], v[:, 3].flip(-1)], 1)


def all_afterstates(b):
    """b (B,4,4) -> after (B,4,4,4), reward (B,4), legal (B,4) bool."""
    new_t, rew_t = tables(str(b.device))
    idx = (_views(b) * POW.to(b.device)).sum(-1)          # (B,4,4)
    after = _unview(new_t[idx])
    reward = rew_t[idx].sum(-1)
    legal = (after != b[:, None]).flatten(2).any(-1)
    return after, reward, legal


def spawn(b, gen=None):
    """Put a 2 (p=.9) or 4 at a random empty cell of every board that has one."""
    flat = b.flatten(1).clone()
    empty = flat == 0
    r = torch.rand(flat.shape, device=b.device, generator=gen).masked_fill(~empty, -1)
    pos = r.argmax(1)
    val = 1 + (torch.rand(len(flat), device=b.device, generator=gen) < 0.1).long()
    has = empty.any(1)
    flat[has, pos[has]] = val[has]
    return flat.view_as(b)


def reset(n, device="cpu", gen=None):
    return spawn(spawn(torch.zeros(n, 4, 4, dtype=torch.int64, device=device), gen), gen)


# ---------------------------------------------------------------- features
def board_stats(b):
    """Hand-made geometric features of boards (...,4,4) -> (...,8), roughly in [0,1]."""
    f = b.float()
    empties = (b == 0).flatten(-2).sum(-1).float() / 16
    mx = f.flatten(-2).max(-1).values
    corners = torch.stack([f[..., 0, 0], f[..., 0, 3], f[..., 3, 0], f[..., 3, 3]], -1)
    max_in_corner = (corners.max(-1).values == mx).float()
    dr, dc = f[..., :, 1:] - f[..., :, :-1], f[..., 1:, :] - f[..., :-1, :]
    # monotonicity: per line, how much of the variation goes one way
    def mono(d):
        inc, dec = d.clamp(min=0).sum(-1), (-d).clamp(min=0).sum(-1)
        tot = inc + dec
        return torch.where(tot > 0, torch.maximum(inc, dec) / tot.clamp(min=1e-6), torch.ones_like(tot)).mean(-1)
    mono_r, mono_c = mono(dr), mono(dc.transpose(-1, -2))
    nz_r, nz_c = (b[..., :, 1:] > 0) & (b[..., :, :-1] > 0), (b[..., 1:, :] > 0) & (b[..., :-1, :] > 0)
    smooth = -((dr.abs() * nz_r).sum((-1, -2)) + (dc.abs() * nz_c).sum((-1, -2))) / (24 * 17)
    merges = (((dr == 0) & nz_r).sum((-1, -2)) + ((dc == 0) & nz_c).sum((-1, -2))).float() / 24
    return torch.stack([empties, mx / 17, max_in_corner, mono_r, mono_c, smooth, merges,
                        f.flatten(-2).sum(-1) / (16 * 17)], -1)


N_ACT_FEATS = 2 + 8 + 4   # legal, reward, afterstate stats, "can merge again" per next action


def action_features(b):
    """The 'enriched state': for each of the 4 moves, what the afterstate looks like.
    Returns feats (B,4,N_ACT_FEATS), legal (B,4), after (B,4,4,4), reward (B,4)."""
    after, reward, legal = all_afterstates(b)
    B = b.shape[0]
    stats = board_stats(after)                                      # (B,4,8)
    _, r2, l2 = all_afterstates(after.reshape(B * 4, 4, 4))        # one more ply, no spawn
    look = (r2.float().log1p() / 17 * l2).view(B, 4, 4)
    feats = torch.cat([legal.float()[..., None], (reward.float().log1p() / 17)[..., None], stats, look], -1)
    return feats * legal[..., None], legal, after, reward


# ---------------------------------------------------------------- symmetry (D4)
# Each transform maps a board to a board; PERM[k][a] = the action that a becomes.
def sym(b, k):
    if k >= 4:
        b = b.flip(-1)
    return torch.rot90(b, k % 4, (-2, -1))


def _action_perm():
    perms = []
    probe = torch.arange(16).view(1, 4, 4)
    for k in range(8):
        # find where each action maps by checking the transformed "direction" of a gradient board
        p = []
        for a in range(4):
            g = torch.zeros(1, 4, 4, dtype=torch.int64)
            # board whose only legal move is direction a: a lone tile at the far side
            if a == 0: g[0, 3, 1] = 1
            if a == 1: g[0, 0, 1] = 1
            if a == 2: g[0, 1, 3] = 1
            if a == 3: g[0, 1, 0] = 1
            after, _, _ = all_afterstates(g)
            target = sym(after[:, a], k)
            sb = sym(g, k)
            after2, _, legal2 = all_afterstates(sb)
            hits = [c for c in range(4) if legal2[0, c] and torch.equal(after2[0, c], target[0])]
            p.append(hits[0])
        perms.append(p)
    del probe
    return torch.tensor(perms)


SYM_PERM = None


def sym_perm():
    global SYM_PERM
    if SYM_PERM is None:
        SYM_PERM = _action_perm()
    return SYM_PERM


def score_of(b):
    """Lower bound score implied by a board (assumes every spawn was a 2)."""
    f = b.float()
    return torch.where(b > 1, (f - 1) * torch.exp2(f), torch.zeros_like(f)).flatten(1).sum(1)


if __name__ == "__main__":
    b = torch.tensor([[[1, 1, 2, 2], [0, 0, 0, 0], [3, 0, 3, 0], [1, 2, 1, 2]]])
    after, rew, legal = all_afterstates(b)
    assert after[0, 2].tolist() == [[2, 3, 0, 0], [0, 0, 0, 0], [4, 0, 0, 0], [1, 2, 1, 2]], after[0, 2]
    assert after[0, 3].tolist() == [[0, 0, 2, 3], [0, 0, 0, 0], [0, 0, 0, 4], [1, 2, 1, 2]]
    assert rew[0, 2] == 4 + 8 + 16 and rew[0, 3] == 4 + 8 + 16
    assert after[0, 0, :, 0].tolist() == [1, 3, 1, 0] and legal.all()
    P = sym_perm()
    assert all(sorted(p.tolist()) == [0, 1, 2, 3] for p in P)
    # symmetry consistency on random boards
    g = torch.Generator().manual_seed(0)
    bs = torch.randint(0, 6, (64, 4, 4), generator=g)
    a0, r0, _ = all_afterstates(bs)
    for k in range(8):
        ak, rk, _ = all_afterstates(sym(bs, k))
        for a in range(4):
            assert torch.equal(ak[:, P[k, a]], sym(a0[:, a], k)) and torch.equal(rk[:, P[k, a]], r0[:, a])
    f, l, _, _ = action_features(bs)
    assert f.shape == (64, 4, N_ACT_FEATS) and torch.isfinite(f).all()
    print("env ok", P.tolist())
