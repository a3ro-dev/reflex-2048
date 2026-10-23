# Deep-research prompts (paste one at a time)

Each prompt is self-contained. Ask for primary sources (papers, repos, READMEs with
numbers) and for a table at the end.

---

## 1. The exact SOTA table for 2048 agents

Build the most complete, source-verified table of 2048 (4x4, standard spawn 90% 2 / 10% 4, no undo) AI results published up to October 2026. For every entry give: method, whether it searches at decision time (and how deep / how many ms per move), mean score, median score, max score, and reach rates for the 8192, 16384, 32768 and 65536 tiles, number of evaluation games, hardware, and the exact URL of the number. Include n-tuple TD lines (Szubert & Jaśkowski 2014, Wu/Yeh multi-stage TD, Jaśkowski 2016, Matsuzaki, Guei et al. 2021 optimistic TD), expectimax programs (nneonneo, macroxue/2048-ai, 2048EndgameTablebase by game-difficulty), MCTS / Stochastic MuZero / Gumbel MuZero, and any deep-network policy or value network (CNN, ResNet, transformer). Separate "with search" from "no search (1-ply)". Flag any number you could not trace to a primary source.

## 2. Neural networks that play 2048 without search

Find every published attempt (papers, theses, Kaggle, GitHub, blog posts, 2015-2026) to play 2048 with a deep neural network policy or value function evaluated greedily with no lookahead. I need: architecture, input encoding (one-hot exponents, embeddings, features), training method (DQN, PPO, TD, supervised imitation of an expectimax or n-tuple teacher, distillation), data scale, and the best mean score and tile reach rates. I especially want imitation/distillation from strong search agents and anything transformer-based. What did authors say limits neural nets on 2048 (credit assignment over 20k+ move games, rare late-game states, value scale)?

## 3. Distilling search into a fast policy

Survey methods for distilling a search-based expert (expectimax, MCTS, tablebases) into a single-forward-pass policy network in stochastic, long-horizon games: DAgger and its variants, expert iteration (ExIt), AlphaZero-style policy targets from visit counts, soft Q-value targets vs hard argmax labels, "Grandmaster-level chess without search" (Ruoss et al. 2024, DeepMind) and its follow-ups, Stockfish NNUE distillation, and anything on 2048 or Threes. For each: what targets they used (action-value bins, win probability, best move), model sizes, data scale, and how close the student got to the teacher. What design choices mattered most?

## 4. Calibration of discrete action policies

What does the literature say about calibrating the action probabilities of a policy network (not just a classifier)? Cover: proper scoring rules (log loss, Brier) as training objectives, temperature scaling, label smoothing effects on calibration, ECE vs classwise ECE for 4-way choices, how calibration of "probability that the expert picks move a" relates to decision quality, and any RL work that trains for calibration explicitly. Include any work named or resembling "Reinforcement Learning for Calibrated Decisions" or "Jev" from TypeSafe AI, and say plainly if those sources cannot be found or verified.

## 5. Late-game 2048: where agents die

Explain, with sources, why 2048 agents fail at the 32768 and 65536 stages: the board-space bottleneck when 15 tiles are large, the "snake chain" vs "perimeter defense formation" strategies, how macroxue's and game-difficulty's lookup tables and tablebases handle these endgames, and how often a perfect player would still fail (published success probabilities from tablebase positions). What features or training curricula (e.g. starting games from late-game positions, multi-stage networks, carousel shaping, tile-downgrading) have been shown to help learned agents in the late game?

## 6. Verifying a human record

What is the highest verified no-undo human score in standard 2048 as of 2026, who holds it, on what platform, and how was it verified (video, replay file)? List other claimed records and why they are or aren't accepted. Is there a community replay format (2048league / 2048verse) I can use to publish an AI game so it can be independently replayed and verified?
