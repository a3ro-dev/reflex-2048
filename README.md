# reflex-2048

A 14.5M-parameter transformer that plays 2048 with **one forward pass per move and no search**, distilled from an expectimax + lookup-table teacher ([macroxue/2048-ai](https://github.com/macroxue/2048-ai)) on one rented A100 for under $10.

It is not a record. It's a small, cheap, honest data point, plus two negative results that were more interesting than the score.

## Results

10,000 fresh games, greedy policy, end-of-schedule checkpoint (no picking the best eval):

| mean | median | max | 2048 | 4096 | 8192 | 16384 | 32768 | 65536 |
|---|---|---|---|---|---|---|---|---|
| **164,395** | 131,064 | 835,640 | 96.6% | 86.4% | 63.3% | 30.0% | 3.9% | 0% |

A second eval on 2,048 games from other seeds gave 166,767 mean (within 1.5%). Per-game scores weren't logged, so there are no standard errors.

How that compares (others as reported):

| agent | params | search | mean | 32768 |
|---|---|---|---|---|
| macroxue expectimax, large tables (our teacher) | tables | depth 4 | 625,898 | 70.7% |
| 2048former v0.1 (self-reported) | 53.2M | none | 605,491 | 66.1% |
| Guei et al. 2021, n-tuple (best published 1-ply) | 268.4M | none | 412,785 | 30.2% |
| **reflex-2048** | **14.5M** | **none** | **164,395** | **3.9%** |
| reflex-2048 + 1-ply over its value head (128 games) | 14.5M | 1-ply | 4,189 | 0% |

So: about a quarter of its teacher and 40% of the best n-tuple agent. 2048former is 3.7× bigger, learned from a depth-6 teacher, and used ~30× the training compute (72 vs 2.4 GPU-hours).

Speed: **9.1 ms per move** median on a laptop CPU (Intel, 8 threads, batch 1, features included), and 65k moves/s batched on the A100.

### What went wrong (the useful part)

- **DAgger collapsed play.** Relabelling the student's own states with the teacher and mixing them into 30% of each batch took greedy play from 56,652 to 4,188 mean. Held-out agreement with the teacher stayed at ~87% the whole time, so every training metric looked fine. The labels check out: they match the teacher's self-play moves on 99.95% of boards, and the game code is unchanged. The final model restarts from the last checkpoint before DAgger data arrived and trains on teacher data only. Cause not isolated.
- **1-ply over the value head is useless.** The auxiliary value head tracks the return across a game (0.88 log-space correlation on held-out teacher boards) but can't rank the four afterstates of a single move. Search on it scores 4,189.
- **Calibrated where it doesn't matter.** On teacher boards, ECE is 0.006 (93.6% agreement at 0.93 mean confidence). On a clean set of boards the student reaches itself (fresh seeds, never trained on), it agrees with the teacher only 61% of the time at 0.88 confidence: ECE 0.27.

Full write-up: [paper/main.pdf](paper/main.pdf).

## Play it on your laptop

```bash
pip install torch numpy
```

Download `reflex.pt` from the [latest release](https://github.com/a3ro-dev/reflex-2048/releases) or from [Hugging Face](https://huggingface.co/a3ro-dev/reflex-2048) (the model card there has a `snapshot_download` snippet), then:

```bash
python -m reflex.play --ckpt reflex.pt --seed 0 --every 500
```

From Python:

```python
from reflex.model import load
m = load("reflex.pt")
m.decide([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])   # tile exponents, 0 = empty
# {'UP': ..., 'DOWN': ..., 'LEFT': ..., 'RIGHT': ...}
```

## How it works

- **Input:** 16 tile tokens (tile-exponent + row + column embeddings), plus 4 action tokens built from each move's afterstate (legality, reward, empties, max tile and whether it's in a corner, monotonicity, smoothness, merges, and the follow-up move rewards), plus 1 value token.
- **Model:** 8 pre-norm bidirectional transformer blocks, d=384, 12 heads. One logit per action token, illegal moves masked.
- **Loss:** cross-entropy to the teacher's move plus a 0.25× Huber value loss on log return-to-go. Random D4 symmetry per sample.
- **Data:**
  - 360M depth-3 teacher decisions (21.9k games, mean 455k).
  - 100M depth-4 decisions.
  - Teacher transitions replay exactly in the batched GPU env (`reflex/env.py`).
- **Training:** 261M samples total, bf16, ~38k samples/s ceiling on an A100 (compute-bound).

## Repo map

- `reflex/`
  - `env.py`: batched GPU 2048.
  - `model.py`: the transformer.
  - `train.py`: distillation.
  - `evaluate.py`: parallel games; `--sym` for symmetry averaging, `--ply1` for the 1-ply row.
  - `dagger.py`: the DAgger loop.
  - `play.py`: watch a game on CPU.
- `teacher/patch_teacher.py`: patches macroxue/2048-ai to dump every decision (`-o`) and to label arbitrary boards (`-B`). That code is GPL-3.0, so it ships here as a patch script, not as a copy.
- `scripts/`: launcher, resume, final eval, calibration report, benchmarks, budget watchdog.
- `paper/`: NeurIPS-format write-up.

Budget: $8.72 of A100 time in total, including one pod migration when the original host's GPU got taken.
