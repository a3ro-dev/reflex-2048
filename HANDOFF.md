# HANDOFF: Reflex-2048 (paste this whole file as the first message of the new chat)

You are taking over a half-built research project from another Claude Code session. Read this fully, then work autonomously. The user (Akshat, GitHub `a3ro-dev`) wants results, not questions. They write lower-case and want the job done.

## 1. Goal

Build a **transformer that plays 2048 in one forward pass per move (no search at inference)**, fast enough to run on a laptop, and push it as far as possible toward beating the best published numbers. Final deliverables: (a) a trained model, (b) a public GitHub repo with clean commits, (c) a paper draft (use the `ml-paper-writing` skill), (d) a live dashboard while training runs. The user explicitly said the RLCD/Jev framing is optional; the real requirement is "transformer, super fast, plays 2048 best". Report honestly. Do not claim a record you did not measure.

**Records to be honest about** (details in `literature/survey.md`):
- Best no-search agent: n-tuple table, ~309k mean, 0% reach 32768 (Guei et al. 2021).
- Search programs: 625k to 772k mean, 72% to 86% reach 32768, 3.5% to 8.4% reach 65536 (macroxue/2048-ai; game-difficulty/2048EndgameTablebase).
- Human no-undo record: 1,314,064 with a 65536 tile (2048verse.com, 2023-12-31). The "~60k" the user remembers is a RecordSetter video entry and is not the real record. Tell the user this plainly.
- Realistic target: beat 309k mean with pure policy (a real, publishable gap), then approach teacher level. A single lucky game over 1.3M is possible only by sampling many games; report mean/median and rates, never cherry-pick.

## 2. Hard constraints from the user

- **RunPod budget: $10 absolute ceiling.** Pod `hiln5i7ijmhp7d`, 1x A100 80GB at $1.79/hr, started 2026-10-23 09:16:43 UTC. `scripts/watchdog.py` runs on the pod: writes `/root/runs/budget.json`, touches `/root/runs/WRAP_UP` at spend > $9.50, copies runs to `/workspace/runs` and runs `runpodctl stop pod` at spend > $9.85. Verify it is still running (`pgrep -af watchdog`). Spend was $0.95 at handoff. Do not create more pods. Never exceed the cap.
- **Use every resource on the pod:** 32 vCPU (the container sees 256 host cores via nproc but only 32 are yours), 80GB VRAM, ~251GB RAM (hosts report 2TB; do not trust). Nothing single-threaded, nothing idle. If something is not needed, downscale it explicitly and say why.
- **Always do a proper dry run before launching anything big.** `scripts/dryrun.py` is the pattern (teacher dump -> replay-verify -> train -> eval).
- The user dislikes many parallel subagents (laptop struggles) and popping console windows. Prefer doing work inline.
- Small atomic commits, push to remote when done. Public repo already exists: https://github.com/a3ro-dev/reflex-2048 (branch `master`).
- Dashboard design: the user said the current one "is functional but has no design sense or taste". They asked for `/web-designer` and `/impeccable`. Aesthetic they originally asked for: **dither graphs/elements and algorithmic ASCII art**. Redo the visual design properly with those skills (see section 6).
- `qenlo` (a3ro-dev/qenlo) is a Rust vector-search engine; judged irrelevant to this pipeline. Skip unless a real need appears.

## 3. Where everything lives

Local: `D:\2048-nn-` (Windows, Git Bash, Python 3.14 with CPU torch 2.11). Pod: SSH.

```bash
ssh -i ~/.runpod/ssh/runpodctl-ssh-key -p 15409 root@216.81.248.127     # (IP/port also in scripts/pod.env)
scripts/sync.sh      # tars the local repo to /root/reflex2048 on the pod (ignore the "Cannot change ownership" noise; --no-same-owner is set)
```

- Pod code: `/root/reflex2048`. Teacher builds: `/root/teacher` (depth 3 workers' tables are in `/root/teacher_s`, binary `2048`, cwd must hold `tuple_moves.10a/11a`). `/root/teacher` is the BIG-tables build (`2048b`), its table build was still running at handoff (single-threaded, minutes to tens of minutes, 27GB RAM): check `ls -la /root/teacher | grep tuple_moves`.
- Data: `/root/data/teacher/*.bin` (teacher self-play), `/root/data/dagger/*.bin` (student-state relabels). Logs `/root/logs`. Run outputs `/root/runs` (train.jsonl, eval.jsonl, live.json, budget.json, latest.pt).
- Dashboard: `dashboard/server.py` (stdlib, port 8000, started with `RUNS=/root/runs python3 dashboard/server.py`) behind nginx on 8080. nginx's stock config on this image did not include sites-enabled; I added `include /etc/nginx/sites-enabled/*;` to `/etc/nginx/nginx.conf`. **Public URL: https://hiln5i7ijmhp7d-8080.proxy.runpod.net/** (RunPod HTTPS proxy; the raw IP only exposes SSH). Locally preview with `cd dashboard && python -m http.server 8765` then `http://localhost:8765/index.html?demo` (demo data).
- The pod has a repo-scoped GitHub deploy key (`/root/.ssh/reflex_deploy`, write access to a3ro-dev/reflex-2048 only) so it can push results. Auth verified. `git clone` the repo on the pod if you want to push from there.

## 4. Code map (all in the repo)

- `reflex/env.py`: batched GPU 2048. Boards `(B,4,4)` int64 exponents. **Action order everywhere: 0=UP 1=DOWN 2=LEFT 3=RIGHT.** One 18^4 row lookup table does all moves. `action_features` = the "enriched state" (per move: legal, reward, afterstate stats: empties, max, max-in-corner, monotonicity r/c, smoothness, merges, tile sum, plus 1-ply lookahead). D4 symmetry helpers `sym`, `sym_perm`. Self-test: `python -m reflex.env` (passes; moves, rewards, legality, all 8 symmetries verified).
- `reflex/model.py`: `Reflex(d=256, layers=8, heads=8)`, 6.46M params. 16 cell tokens (tile emb + row + col emb) + 4 move tokens (features) + 1 value token, bidirectional encoder, one logit per move token, illegal moves masked to -inf, softmax. `decide(board)` returns `{'UP':p,...}`. Laptop CPU forward ~5 ms (batch 1, unoptimised).
- `reflex/train.py`: distillation. Cross-entropy to the teacher's move (strictly proper scoring rule -> calibrated probabilities) + 0.25 x value huber on log return-to-go (masked for DAgger rows that have no return). Random D4 augmentation (vectorised gather; verified equal to `sym`). Held-out = whole games with `seed % 50 == 0`, never trained on. Logs ECE/Brier/NLL/accuracy every 1000 steps; saves `latest.pt` atomically; honours `/root/runs/WRAP_UP`. Reloads grown data files every 5000 steps (`--data` accepts comma-separated globs). Record format: 32 bytes (`REC` dtype).
- `reflex/evaluate.py`: plays N games in parallel on GPU, pure argmax policy, optional `--sym` (8-fold symmetry-averaged distribution), `--watch` re-evaluates on every new checkpoint, writes `eval.jsonl` and `live.json`.
- `reflex/dagger.py`: student plays, teacher relabels visited boards through the patched binary's `-B boards.u8` mode. **Not yet run end to end. Dry-run it before trusting it** (the `-B` mode and the `.part` -> `.bin` rename are untested).
- `teacher/patch_teacher.py`: patches macroxue/2048-ai (GPL-3.0; keep the licence note when redistributing) to add `-o file` (dump every decision), `-B file` (label arbitrary boards). Teacher move order is 0=up 1=left 2=right 3=down; `TEACHER_TO_OURS = [0,2,3,1]` in train.py.
- `scripts/`: `gen_teacher.sh`, `dryrun.py`, `bench.py`, `sync.sh`, `watchdog.py`, `pod.env`.
- `dashboard/`: `index.html` (single file, polls `/api/state` every 3 s; `?demo` for fake data), `server.py`, `nginx.conf`.
- `literature/survey.md`, `prompts/gemini_deep_research.md` (six deep-research prompts for the user to run in Gemini; the user asked for these), `shots/` is gitignored.

## 5. Verified state at handoff

Verified:
- Env self-test passes. Teacher dump replay check: **17,619 of 17,619 transitions match** (moves, spawns, rewards) against our env.
- Dry run (`scripts/dryrun.py`): train 1000 steps bs=16384 on 17k states OK (held-out acc 99% is overfitting, meaningless), eval of 64 games OK. The dry-run model (trained on 2 games) scored mean 5.9k, as expected; that is **not** a result.
- Teacher: 30 workers of `./2048 -d 3 -i 400` were launched (30 x 400 = 12,000 games), writing `/root/data/teacher/d3_s*.bin`; about 7.4 GB and 4k+ complete games at handoff. Teacher d3 ran ~17k moves/s per worker on the small tables. Mean teacher score at d3 is roughly 490k (macroxue's published d3 big-table figure) but the dashboard's number is computed from logs, check it.
- Benchmark (A100, bs 16384): feature+augment ~6 ms, fwd+bwd eager 369 ms (44k samples/s), `torch.compile` 285 ms (58k samples/s). **That is slow for a 6.5M-param model.** Suspects: the CPU is oversubscribed (load average ~96 on 32 cores because 30 teacher workers + trainer + dashboard compete), the encoder with 21 tokens at bs 16384 is attention/launch-bound in fp32 softmax paths, and `nn.TransformerEncoder` is not fused. Fix by measuring: pin the trainer to dedicated cores (`taskset`), reduce teacher workers while training or give them lower nice priority, use `torch.nn.functional.scaled_dot_product_attention`-based custom blocks, bigger model or bigger batch to actually fill 80GB, CUDA graphs via `torch.compile(mode="max-autotune")`. A 6.5M model on an A100 should do several hundred thousand samples/s.

**NOT done / NOT verified (be skeptical of everything here):**
- No real training run has started. No real evaluation exists. Dashboard has only ever shown demo data.
- GPU is mostly idle waiting for data in the real flow (teacher on CPU). The user explicitly wants all 32 cores, the 80GB VRAM and the RAM used. Design the resource split: e.g. ~24 cores teacher, ~6 cores trainer/evaluator/DAgger labeller, whole dataset resident on GPU (60M states x 16 bytes is only ~1GB, so VRAM is mostly for bigger model/batch/many parallel eval games), RAM for the big lookup tables.
- DAgger loop untested. Value head is trained but unused at inference (fine; it is an auxiliary loss, and could be used later for 1-step lookahead over afterstates).
- Dashboard `teacher()` parses `score ` lines from logs (the log files may include binary progress characters; I saw `grep: binary file matches`; use `errors="ignore"`/`grep -a`, already `errors="ignore"` in server.py but verify counts).
- The Windows to Linux line-ending issue: after `sync.sh`, run `sed -i 's/\r$//' scripts/*.sh scripts/*.py` on the pod (already done for existing files; redo for new ones).
- The "RLCD / Jev / TypeSafe AI" sources: I could not verify they exist. The honest framing for the paper is: cross-entropy and Brier are strictly proper scoring rules, so distilling a teacher with them yields calibrated move probabilities; measure ECE/Brier against held-out teacher moves and, better, against **rollout outcomes** (does confidence correlate with actual value of the move?). Do not cite RLCD/Jev as established work.

## 6. Plan for the new session (suggested order)

1. `ssh` in; check `nvidia-smi`, `uptime`, `/root/runs/budget.json`, `pgrep -af "watchdog|server.py|2048 -d"`, and teacher table status. Run `qenlo-memory recall "2048 reflex runpod"`; save new durable facts with `qenlo-memory remember`.
2. **Fix the resource split and trainer throughput** (section 5). Re-benchmark with `scripts/bench.py` while teachers are running AND with them paused, to see the CPU contention cost.
3. Dry-run DAgger end to end on a tiny scale (`-B` label mode, `.part` rename, trainer pickup).
4. Launch for real, with a single launcher script under `scripts/` that starts: teacher workers (more games, then the big-table `2048b` if tables finished and RAM allows), trainer, evaluator `--watch`, DAgger loop, dashboard. Everything logs to `/root/runs`. Plan the $ budget: ~5 hours left at handoff; reserve the final 20 minutes for the final big evaluation (e.g. 10,000 games with `--sym`) and for pushing results.
5. Monitor via the dashboard URL and SSH; adjust. Expect the real loss to plateau well above the dry run's numbers. Track ECE and held-out accuracy, and **evaluation mean score and 8192/16384/32768 rates** as the real metrics.
6. Redesign the dashboard (use the `web-designer` and `impeccable` skills; keep dither graphs + algorithmic ASCII). Problems the user sees: no design sense or taste, functional but poor UX. Earlier critique to address: the ASCII "terrain" is small and sparse in its box, charts are uniform and undifferentiated, hierarchy is flat, the first screen shows a headline number that is not the most interesting thing. Make sure every metric the user asked for is visible (all training metrics, cost structure and burn, GPU/CPU/RAM, teacher stats, architecture diagram, live game). Add CPU and RAM panels (currently only GPU). Verify with the web-designer `shoot.mjs` scan before shipping; deploy with `scripts/sync.sh` (the dashboard is static, no restart needed for `index.html`).
7. Final: copy `/root/runs` summary + best checkpoint (25MB; commit via Git LFS or a GitHub release, not plain git bloat), write `README.md` (results table with honest numbers, how to run on a laptop: CPU inference script), `ml-paper-writing` draft in `paper/`, push. Give the user the Gemini prompts file path and a short report.
8. Before ending, stop spending: confirm the pod is stopped (`runpodctl pod list`) or the watchdog stopped it.

## 7. Gotchas learned

- `runpodctl get pod` is deprecated; use `runpodctl pod list/get`, `runpodctl me` for balance (balance was $254, account spend limit $80).
- Pod has no pip numpy by default: `pip install --break-system-packages numpy` (done). PyTorch 2.9.1+cu128 present.
- Pod image's PEP 668 blocks plain pip. nginx lives at /usr/sbin/nginx (installed via apt in this session).
- macroxue's tables: first run builds `tuple_moves.10a/11a` (1.2 to 1.3 GB small build; much larger for `2048b`), each worker mmaps them so launch workers only after the build finishes.
- Windows local: `python` is 3.14 with CPU torch; use it only for self-tests. Use Git Bash syntax.
- The user wants the final repo public, commits atomic, and any X/LinkedIn launch text drafted in their `a3roxyz-voice` skill style (lower-case, first-principles, honest tradeoffs) as files, never published without asking.
- Ask the user nothing that you can default. Surface only: final numbers, anything that blocks, and anything that costs money beyond the $10 cap (do not spend it).
