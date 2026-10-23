"""Tiny stdlib dashboard backend: GET /api/state aggregates every log on the pod."""
import glob, json, os, subprocess, time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

RUNS = os.environ.get("RUNS", "/root/runs")
HERE = os.path.dirname(os.path.abspath(__file__))


def jsonl(path, every=1, tail=4000):
    try:
        lines = open(path).read().splitlines()[-tail:]
    except FileNotFoundError:
        return []
    out = [json.loads(l) for l in lines[::every] if l.strip()]
    return out


def jfile(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def gpu():
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.strip()
        n, u, mu, mt, p, t = [x.strip() for x in q.split(",")]
        return dict(name=n, util=float(u), mem_used=float(mu), mem_total=float(mt), power=float(p), temp=float(t))
    except Exception:
        return None


def teacher():
    files = glob.glob("/root/data/teacher/*.bin")
    size = sum(os.path.getsize(f) for f in files)
    logs = sorted(glob.glob("/root/logs/teacher_*_w*.log"))
    games, scores = 0, []
    for l in logs:
        for line in open(l, errors="ignore"):
            if line.startswith("score "):
                scores.append(int(line.split()[1]))
    load = os.getloadavg()[0]
    return dict(files=len(files), states=size // 32, games=len(scores),
                mean_score=sum(scores) / len(scores) if scores else 0, max_score=max(scores, default=0),
                cpu_load=load, cpus=os.cpu_count(), workers=len(logs))


class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=HERE, **k)

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/api/state"):
            train = []
            for f in sorted(glob.glob(f"{RUNS}/*/train.jsonl")):
                rows = jsonl(f)
                step = max(1, len(rows) // 600)
                train.append(dict(run=f.split("/")[-2], rows=rows[::step] + rows[-1:]))
            body = json.dumps(dict(now=time.time(), budget=jfile(f"{RUNS}/budget.json"), gpu=gpu(),
                                   teacher=teacher(), train=train, eval=jsonl(f"{RUNS}/eval.jsonl"),
                                   live=jfile(f"{RUNS}/live.json"), notes=jfile(f"{RUNS}/notes.json"))).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            super().do_GET()


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 8000), H).serve_forever()
