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
| 2048former-50m-v0.1 (Keisling, 2025) | 53.2M encoder transformer distilled from macroxue d6, 120k games (~3B frames), 3 days on one 4090 | 605,491 | 66.1% | — | **no** | HF model card jkeisling/2048former-50m-v0.1 (self-reported, no paper; verified the card 2026-10-23) |
| Guei, Chen, Wu (2021), 2-stage 8x6-tuple OTD+TC + tile-downgrading, 1-ply | n-tuple lookup table, 268.4M weights | 412,785 ± 2,208 | 30.16% | — | **no** | arXiv:2111.11090 Table X, 1M games (verified in paper text) |
| Jaśkowski (2016), 16-stage 5x6-tuple MS-TC, 1-ply | n-tuple lookup table, 1342.2M weights | 324,710 ± 11,043 | 19% | 0 | **no** | arXiv:1604.05085, via arXiv:2111.11090 Table X |
| Szubert & Jaśkowski (2014) | TD(0) afterstate n-tuple | 100,178 | 0 | 0 | no | IEEE CIG 2014 |
| Deep RL (DQN/CNN policies, many student reports) | | < 60k typical | 0 | 0 | no | e.g. CS224R 2025, arXiv:2110.10374 |
| Gumbel/Stochastic MuZero for 2048 (Kao et al. 2022; Antonoglou et al. 2022) | learned model + MCTS | reported in the low-to-mid 100ks to ~500k range depending on simulations; check | | | yes | IEEE 10056566; Stochastic MuZero ICLR 2022 |

**Human no-undo record:** 1,314,064 with the 65536 tile (2048verse.com, 2023-12-31, video). The "~58k world record" sometimes quoted is a RecordSetter video entry, not the real ceiling.
**Theoretical maximum:** 3,932,100 points, 131,072 tile.

## The gap this project targets

**Correction (2026-10-23):** an earlier version of this file said the best no-search agent was an n-tuple table at ~309k with 0% 32768. That was wrong. The best published 1-ply n-tuple agent is 412,785 mean (Guei et al., Table X), and a 53M-parameter transformer, 2048former, self-reports 605,491 mean with no search. So "a transformer that plays 2048 well without search" is already done; the open questions are how small and how cheap it can be.

Reflex asks: how close does a much smaller transformer (6-15M params, one forward pass per move) get to its teacher, and to 2048former, on a hard $10 budget (about 4 A100-hours, teacher data included), and do its move probabilities stay calibrated? Our teacher is macroxue d3/d4 (published 493k / 626k mean). 2048former used d6 and about 72 GPU-hours, so we expect to land below it, and we report the gap.

## Method sources

- Teacher: macroxue/2048-ai (GPL-3.0), patched to dump every decision and to label arbitrary boards (`teacher/patch_teacher.py`).
- DAgger: Ross, Gordon, Bagnell (2011), "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning". The student's own state distribution is relabelled by the expert.
- Proper scoring rules (Gneiting & Raftery 2007): cross-entropy and Brier are strictly proper, so minimising them against teacher labels drives calibrated move probabilities. This is the "RLCD" calibration idea expressed as a standard objective.
- Calibration metric: ECE (Guo et al. 2017).
