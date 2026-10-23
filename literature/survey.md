# Literature survey: 2048 agents (as of 2026-10-23)

Numbers are as reported by each source. "No search" = 1-ply greedy on the learned function.

## Records to beat

| Who | Method | Mean score | 32768 rate | 65536 rate | Search | Source |
|---|---|---|---|---|---|---|
| 2048EndgameTablebase (game-difficulty, 2024-26) | expectimax d1-9 + endgame tablebases + distilled 2.5MB table | 772,353 | 86.1% | 8.4% | yes, heavy | github.com/game-difficulty/2048EndgameTablebase |
| macroxue/2048-ai (2022 big tables) | expectimax + DP lookup tables | 711,769 (d8), 493,058 (d3) | 80.5% (d8) | 3.5% | yes | github.com/macroxue/2048-ai |
| Guei, Chen, Wu (2021, IEEE ToG) | optimistic TD + TC, 2-stage 8x6-tuple, 6-ply tile-downgrading expectimax | 625,377 | 72% | 0.02% | yes | arXiv:2111.11090 |
| Jaśkowski (2016) | delayed TC, multi-stage weight promotion, carousel shaping, 1s/move | 609,104 | 70% | 0 | yes | arXiv:1604.05085 |
| Yeh et al. (2016) | multi-stage TD, 3-ply | — | 31.75% | first 65536 ever | yes | arXiv:1606.07374 |
| Guei et al. 8x6-tuple OTD, 1-ply | n-tuple lookup table | ~309,208 | 0% | 0 | **no** | arXiv:2111.11090 Table III |
| Szubert & Jaśkowski (2014) | TD(0) afterstate n-tuple | 100,178 | 0 | 0 | no | IEEE CIG 2014 |
| Deep RL (DQN/CNN policies, many student reports) | | < 60k typical | 0 | 0 | no | e.g. CS224R 2025, arXiv:2110.10374 |
| Gumbel/Stochastic MuZero for 2048 (Kao et al. 2022; Antonoglou et al. 2022) | learned model + MCTS | reported in the low-to-mid 100ks to ~500k range depending on simulations; check | | | yes | IEEE 10056566; Stochastic MuZero ICLR 2022 |

**Human no-undo record:** 1,314,064 with the 65536 tile (2048verse.com, 2023-12-31, video). The "~58k world record" sometimes quoted is a RecordSetter video entry, not the real ceiling.
**Theoretical maximum:** 3,932,100 points, 131,072 tile.

## The gap this project targets

No published neural network (CNN/transformer) policy plays 2048 at a strong level **without search**. The best no-search agents are n-tuple lookup tables (~300k mean, 0% 32768). Search-based programs are at 600-770k but cost 50-6000 moves/s per core.

Reflex asks: can a small transformer, one forward pass per move, absorb a search+tablebase teacher well enough to match or beat the best search-free agents, and approach search agents at a fraction of the cost per move?

## Method sources

- Teacher: macroxue/2048-ai (GPL-3.0), patched to dump every decision and to label arbitrary boards (`teacher/patch_teacher.py`).
- DAgger: Ross, Gordon, Bagnell (2011), "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning". The student's own state distribution is relabelled by the expert.
- Proper scoring rules (Gneiting & Raftery 2007): cross-entropy and Brier are strictly proper, so minimising them against teacher labels drives calibrated move probabilities. This is the "RLCD" calibration idea expressed as a standard objective.
- Calibration metric: ECE (Guo et al. 2017).
