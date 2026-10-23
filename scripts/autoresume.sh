#!/usr/bin/env bash
# local: retry starting the stopped pod until its host frees a GPU, then sync and resume training.
# usage: scripts/autoresume.sh <train_hours> <spent_usd>
set -uo pipefail
cd "$(dirname "$0")/.."
ID=$(grep POD_ID scripts/pod.env | cut -d= -f2)
until runpodctl pod get $ID 2>/dev/null | grep -q "\"desiredStatus\": \"RUNNING\"" || { r=$(runpodctl pod start $ID 2>&1); ! echo "$r" | grep -q error; }; do sleep 20; done
echo "$(date -u +%T) started"
until info=$(runpodctl ssh info $ID 2>&1) && echo "$info" | grep -q "root@"; do sleep 10; done
read -r HOST PORT < <(echo "$info" | python -c "import re,sys;m=re.search(r'root@([\d.]+)\D+?(\d+)|-p (\d+) root@([\d.]+)',sys.stdin.read());print(*(m.group(1,2) if m.group(1) else m.group(4,3)))")
printf 'POD_ID=%s\nPOD_HOST=%s\nPOD_PORT=%s\n' $ID $HOST $PORT > scripts/pod.env
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i $HOME/.runpod/ssh/runpodctl-ssh-key -p $PORT root@$HOST"
until $SSH true 2>/dev/null; do sleep 10; done
echo "$(date -u +%T) ssh up at $HOST:$PORT"
bash scripts/sync.sh 2>&1 | grep -v ownership
$SSH "bash /root/reflex2048/scripts/resume.sh $1 $2" 2>&1 | tail -5
