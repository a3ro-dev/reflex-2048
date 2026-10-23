#!/usr/bin/env bash
# After a pod stop (container disk wiped): link the volume snapshot back, restart the
# budget watchdog with the spend so far, resume training from latest.pt.
# usage: scripts/resume.sh <train_hours> <spent_usd_before_this_start> [cap=10]
set -euo pipefail
H=$1; SPENT=$2; CAP=${3:-10}
SNAP=/workspace/reflex2048-resume-20261023
for d in teacher data runs logs; do [ -e /root/$d ] || ln -s $SNAP/$d /root/$d; done
python3 -c "import numpy" 2>/dev/null || pip install -q --break-system-packages numpy
cd /root/reflex2048
sed -i 's/\r$//' scripts/*.sh scripts/*.py reflex/*.py
# watchdog charges from a start time backdated by what was already spent
START=$(python3 -c "import time,datetime as d;print(d.datetime.utcfromtimestamp(time.time()-$SPENT/1.79*3600).isoformat())")
pgrep -f "^python3 scripts/watchdog.py" >/dev/null || \
  nohup python3 scripts/watchdog.py $START $CAP > /root/logs/watchdog.log 2>&1 &
cp /root/runs/reflex/latest.pt /root/runs/reflex/resume_from.pt
[ -f /root/runs/reflex/best.pt ] && cp -n /root/runs/reflex/best.pt /root/runs/reflex/best_run1.pt || true   # evaluator restarts its best from scratch
INIT=/root/runs/reflex/resume_from.pt bash scripts/launch.sh $H
