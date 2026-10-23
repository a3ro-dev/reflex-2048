#!/usr/bin/env bash
# Final numbers after training stops: latest vs best on 2048 games, winner on 10k fresh
# games, a clean DAgger eval set from fresh seeds, calibration report. Outputs in /root/runs/final/.
set -euo pipefail
cd /root/reflex2048
O=/root/runs/final; mkdir -p $O; rm -f /root/runs/WRAP_UP
R=/root/runs/reflex
for c in latest best best_run1; do
  python3 -m reflex.evaluate --ckpt $R/$c.pt --games 2048 --seed 1 --out $O/select_$c.jsonl
done
W=$(python3 -c "import json;m={c:json.loads(open('$O/select_'+c+'.jsonl').readlines()[-1])['mean_score'] for c in ('latest','best','best_run1')};print(max(m,key=m.get))")
cp $R/$W.pt $O/reflex.pt; echo "winner: $W" | tee $O/winner.txt
python3 -m reflex.evaluate --ckpt $O/reflex.pt --games 10000 --seed 900000 --out $O/eval_10k.jsonl
python3 -m reflex.dagger --ckpt $O/reflex.pt --rounds 1 --seed 910000 --games 512 --depth 4 --workers 12 \
  --out /root/data/dagger_eval --log $O/dagger_eval.jsonl
python3 scripts/final_report.py $O/reflex.pt $O/report.json 200000 d3,d4,dag,dageval
