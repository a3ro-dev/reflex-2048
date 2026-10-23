#!/usr/bin/env bash
# Final numbers after training stops, all on the end-of-schedule checkpoint (no selection by
# eval score): 10k fresh games, a clean DAgger eval set from fresh seeds (CPU, in parallel),
# calibration report, then the 1-ply row last (not search-free) under a time limit.
# usage: final.sh [ply1_games=256] [ply1_seconds=900]. Outputs in /root/runs/final/.
set -euo pipefail
cd /root/reflex2048
O=/root/runs/final; mkdir -p $O; rm -f /root/runs/WRAP_UP
cp /root/runs/reflex/latest.pt $O/reflex.pt
python3 -m reflex.dagger --ckpt $O/reflex.pt --rounds 1 --seed 910000 --games 512 --depth 4 --workers 12 \
  --out /root/data/dagger_eval --log $O/dagger_eval.jsonl > $O/dagger_eval.log 2>&1 &
python3 -m reflex.evaluate --ckpt $O/reflex.pt --games 10000 --seed 900000 --out $O/eval_10k.jsonl
wait
python3 scripts/final_report.py $O/reflex.pt $O/report.json 200000 d3,d4,dag,dageval
timeout ${2:-900} python3 -m reflex.evaluate --ply1 --ckpt $O/reflex.pt --games ${1:-256} --seed 920000 \
  --out $O/eval_ply1.jsonl || echo "ply1 hit the time limit" | tee $O/ply1_timeout.txt
echo FINAL_DONE
