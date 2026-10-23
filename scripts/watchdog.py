"""Hard budget guard: tracks spend for this pod, saves artifacts to the volume and
stops the pod before the cap. usage: python watchdog.py <start_utc_iso> <cap_usd>"""
import json, os, shutil, subprocess, sys, time
from datetime import datetime, timezone

start = datetime.fromisoformat(sys.argv[1]).replace(tzinfo=timezone.utc).timestamp()
cap = float(sys.argv[2])
rate = float(os.environ.get("COST_PER_HR", "1.79"))
os.makedirs("/root/runs", exist_ok=True)
while True:
    hours = (time.time() - start) / 3600
    spent = hours * rate
    state = dict(rate_per_hr=rate, hours=hours, spent=spent, cap=cap,
                 remaining_hr=max(0, (cap - spent) / rate), gpu=os.environ.get("RUNPOD_GPU_NAME", ""),
                 status="wrapping_up" if spent > cap - 0.5 else "running")
    json.dump(state, open("/root/runs/budget.json.tmp", "w"))
    os.replace("/root/runs/budget.json.tmp", "/root/runs/budget.json")
    if spent > cap - 0.5:
        open("/root/runs/WRAP_UP", "w").close()           # trainers save + exit on this flag
    if spent > cap - 0.15:
        shutil.copytree("/root/runs", "/workspace/runs", dirs_exist_ok=True)
        subprocess.run(["runpodctl", "stop", "pod", os.environ["RUNPOD_POD_ID"]])
        break
    time.sleep(30)
