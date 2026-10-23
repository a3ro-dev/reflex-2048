#!/usr/bin/env bash
# Build the patched expectimax+lookup teacher and run N parallel self-play workers.
# usage: gen_teacher.sh <workers> <depth> <games_per_worker> <seed_base> [binary=2048b]
set -euo pipefail
W=${1:-28}; D=${2:-3}; G=${3:-200}; S=${4:-100000}; BIN=${5:-2048b}
ROOT=/root/reflex2048
T=/root/teacher
if [ ! -x $T/$BIN ]; then
  git clone -q --depth 1 https://github.com/macroxue/2048-ai.git $T
  python3 $ROOT/teacher/patch_teacher.py $T/2048.cc
  (cd $T && make -j4 2048 2048b >/dev/null)
fi
mkdir -p /root/data/teacher /root/logs
cd $T  # lookup tables are created/loaded relative to cwd
# first worker builds the tables; others wait so they mmap the same files
if [ ! -f $T/.tables_ready_$BIN ]; then
  ./$BIN -d $D -i 1 -o /dev/null 1 > /root/logs/teacher_warmup.log 2>&1
  touch $T/.tables_ready_$BIN
fi
for w in $(seq 0 $((W-1))); do
  seed=$((S + w * G))
  nohup ./$BIN -d $D -i $G -o /root/data/teacher/d${D}_s${seed}.bin $seed \
    > /root/logs/teacher_d${D}_w$w.log 2>&1 &
done
echo "launched $W workers depth $D"
