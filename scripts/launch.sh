#!/usr/bin/env bash
# Start the real run on the pod. Self-play teachers (d4) are started separately and keep running.
# usage: scripts/launch.sh <train_hours>
set -euo pipefail
H=${1:-3.5}
cd /root/reflex2048
mkdir -p /root/runs/reflex /root/data/dagger /root/logs
rm -f /root/runs/WRAP_UP
nohup python3 -m reflex.train ${INIT:+--init $INIT} --d 384 --layers 8 --bs 8192 --lr 8e-4 --hours $H --refresh 6000 \
  --out /root/runs/reflex > /root/runs/train.log 2>&1 &
nohup nice -n 15 python3 -m reflex.evaluate --watch --min-interval 600 --games 512 --ckpt /root/runs/reflex/latest.pt \
  --out /root/runs/eval.jsonl > /root/runs/eval.log 2>&1 &
nohup nice -n 15 python3 -m reflex.dagger --ckpt /root/runs/reflex/latest.pt --depth 4 --workers 10 \
  --games 512 --q 0.02 --tail 150 > /root/runs/dagger.log 2>&1 &
sleep 2
pgrep -af "reflex\.(train|evaluate|dagger)"
