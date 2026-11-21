"""Record the real app (stage) from the start page to the reveal, add the results cards and a voiceover.

  PYTHONPATH=backend:phase0 python docs/img/record_app.py [--url http://localhost:5174/] [--out ~/Desktop/foresight-demo.mp4]

Headless Chrome is driven over the DevTools protocol; Page.startScreencast captures what the app actually draws.
The voiceover (macOS `say`) uses the numbers from the run that was just recorded and from eval/results.json.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import glob
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import websockets

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "docs" / "img"), str(ROOT / "phase0")]
from build import CHROME  # noqa: E402
from demo_video import card  # noqa: E402
from demo_voiceover import LEAD, RATE, TAIL, VOICE, script  # noqa: E402

W, H, FPS = 1920, 1080, 24


class CDP:
    def __init__(self, ws):
        self.ws, self.n, self.pending, self.frames = ws, 0, {}, []

    async def reader(self):
        async for raw in self.ws:
            m = json.loads(raw)
            if "id" in m and m["id"] in self.pending:
                self.pending.pop(m["id"]).set_result(m.get("result", m.get("error")))
            elif m.get("method") == "Page.screencastFrame":
                p = m["params"]
                self.frames.append((time.time(), p["data"]))
                asyncio.create_task(self.send("Page.screencastFrameAck", {"sessionId": p["sessionId"]}))

    async def send(self, method, params=None):
        self.n += 1
        fut = asyncio.get_event_loop().create_future()
        self.pending[self.n] = fut
        await self.ws.send(json.dumps({"id": self.n, "method": method, "params": params or {}}))
        return await fut

    async def js(self, expr):
        r = await self.send("Runtime.evaluate", {"expression": expr, "returnByValue": True, "awaitPromise": True})
        return (r or {}).get("result", {}).get("value")


async def record(url: str) -> tuple[list, dict]:
    tmp = tempfile.mkdtemp()
    proc = subprocess.Popen([CHROME, "--headless=new", "--remote-debugging-port=9333", f"--user-data-dir={tmp}",
                             f"--window-size={W},{H}", "--hide-scrollbars", "--no-first-run",
                             "--autoplay-policy=no-user-gesture-required", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                targets = json.loads(urllib.request.urlopen("http://127.0.0.1:9333/json/list", timeout=1).read())
                page = next(t for t in targets if t["type"] == "page")
                break
            except Exception:  # noqa: BLE001
                await asyncio.sleep(0.2)
        async with websockets.connect(page["webSocketDebuggerUrl"], max_size=100_000_000) as ws:
            c = CDP(ws)
            rd = asyncio.create_task(c.reader())
            await c.send("Emulation.setDeviceMetricsOverride", {"width": W, "height": H, "deviceScaleFactor": 1, "mobile": False})
            await c.send("Page.enable")
            await c.send("Page.navigate", {"url": url})
            await asyncio.sleep(4)
            await c.send("Page.startScreencast", {"format": "jpeg", "quality": 85, "maxWidth": W, "maxHeight": H, "everyNthFrame": 1})
            marks = {"start": time.time()}
            await asyncio.sleep(3.5)
            marks["click"] = time.time()
            await c.js('document.querySelector(\'[aria-label="Run Ball rolls off"]\').click()')
            t0 = time.time()
            while time.time() - t0 < 300:
                await asyncio.sleep(0.25)
                if "reveal" not in marks and await c.js("!!document.querySelector('figure.lane-foresight video')"):
                    marks["reveal"] = time.time()
                    await asyncio.sleep(0.5)
                    await c.js("[...document.querySelectorAll('button')].find(b => b.textContent.trim() === 'Half speed')?.click()")
                if "reveal" in marks and await c.js("!!document.querySelector('figure.lane-omni video')"):
                    marks["omni"] = time.time()
                    await c.js("(() => { for (const v of document.querySelectorAll('video')) v.currentTime = 0; })()")
                    break
            await asyncio.sleep(11.5)
            marks["end"] = time.time()
            await c.send("Page.stopScreencast")
            rd.cancel()
            return c.frames, marks
    finally:
        proc.terminate()


def timeline(frames: list, s0: float, s1: float, dur: float) -> list[np.ndarray]:
    """Map source time [s0, s1] onto `dur` seconds at FPS, holding the latest captured frame."""
    times = [t for t, _ in frames]
    cache: dict[int, np.ndarray] = {}
    out, n = [], max(1, int(round(dur * FPS)))
    for k in range(n):
        ts = s0 + (s1 - s0) * k / n
        i = max(0, int(np.searchsorted(times, ts, side="right")) - 1)
        if i not in cache:
            img = cv2.imdecode(np.frombuffer(base64.b64decode(frames[i][1]), np.uint8), cv2.IMREAD_COLOR)
            cache = {i: cv2.resize(img, (W, H)) if img.shape[:2] != (H, W) else img}
        out.append(cache[i])
    return out


def speak(text: str, path: Path) -> float:
    subprocess.run(["say", "-v", VOICE, "-r", str(RATE), "-o", str(path), text], check=True)
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                                capture_output=True, text=True).stdout)


def main(url: str, out: Path) -> Path:
    frames, marks = asyncio.run(record(url))
    if "omni" not in marks:
        raise SystemExit(f"recording did not reach the full reveal: {sorted(marks)}")
    job = Path(max(glob.glob(str(ROOT / "outputs" / "j_*")), key=os.path.getmtime))
    lines = script(job)
    tmp = Path(tempfile.mkdtemp())
    vo = {k: (tmp / f"{k}.aiff", speak(v, tmp / f"{k}.aiff")) for k, v in lines.items()}
    d = {k: v[1] + LEAD + TAIL for k, v in vo.items()}

    seq: list[np.ndarray] = []
    starts: dict[str, float] = {}

    def add(key, imgs):
        starts[key] = len(seq) / FPS
        seq.extend(imgs)

    add("title", timeline(frames, marks["start"], marks["click"], d["title"]))
    run_len = marks["reveal"] - marks["click"]
    add("cond", timeline(frames, marks["click"], marks["click"] + run_len * d["cond"] / (d["cond"] + d["steps"]), d["cond"]))
    add("steps", timeline(frames, marks["click"] + run_len * d["cond"] / (d["cond"] + d["steps"]), marks["reveal"], d["steps"]))
    add("wait", timeline(frames, marks["reveal"], marks["omni"], 2.5))  # Omni still generating: sped up
    add("reveal", timeline(frames, marks["omni"], marks["end"], max(d["reveal"], 10.5)))

    evr = json.loads((ROOT / "eval" / "results.json").read_text())
    S = evr["systems"]
    names = {"foresight": "Foresight", "foresight_carbon": "Foresight (Carbon reads)", "genie": "Genie 4", "omni": "Gemini Omni 1.2",
             "const_velocity": "Constant velocity", "static": "Last frame held"}
    line = ", ".join(f"{names.get(k, k)} {v['physics_iq_verified_pct']:.1f}" for k, v in sorted(S.items(), key=lambda kv: -kv[1]["physics_iq_verified_pct"]))
    bg = card(f"""<div style="padding:70px 140px"><div class="d" style="font-size:66px;font-weight:800">Physics-IQ, {evr['n_clips']} real clips, official metric code</div>
      <div style="font-size:26px;color:var(--muted);margin-top:18px;max-width:1640px">{line}. Foresight beats both video models; a no-physics
      baseline with the same Gemini scene reading still scores higher, because the physics fit breaks on collisions and falls.</div></div>""", tmp / "e.png")
    chart = cv2.imread(str(ROOT / "eval" / "chart.png"))
    ch = cv2.resize(chart, (1400, int(chart.shape[0] * 1400 / chart.shape[1])))
    bg[300:300 + ch.shape[0], 260:260 + ch.shape[1]] = ch[: H - 300]
    add("chart", [bg] * int(d["chart"] * FPS))
    end = card("""<div style="padding:220px 140px"><div class="d" style="font-size:150px;font-weight:800">Foresight</div>
      <div style="font-size:44px;margin-top:40px">github.com/amaan784/foresight-worlds</div>
      <div style="font-size:30px;color:var(--muted);margin-top:24px">Built during the Gemini 4 NYC Hackathon, November 21, 2026.</div></div>""", tmp / "f.png")
    add("end", [end] * int(d["end"] * FPS))

    silent = tmp / "silent.mp4"
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS),
                          "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", str(silent)],
                         stdin=subprocess.PIPE)
    for f in seq:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    p.wait()
    keys = list(lines)
    inputs, filters = [], []
    for i, k in enumerate(keys):
        inputs += ["-i", str(vo[k][0])]
        ms = int((starts[k] + LEAD) * 1000)
        filters.append(f"[{i + 1}:a]adelay={ms}|{ms}[a{i}]")
    mix = "".join(f"[a{i}]" for i in range(len(keys))) + f"amix=inputs={len(keys)}:normalize=0[aout]"
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(silent), *inputs, "-filter_complex", ";".join(filters + [mix]),
                    "-map", "0:v", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out)],
                   check=True)
    print(out, f"{len(seq) / FPS:.1f} s", "job", job.name, {k: round(v - marks['start'], 1) for k, v in marks.items()})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:5174/")
    ap.add_argument("--out", type=Path, default=Path.home() / "Desktop" / "foresight-demo.mp4")
    a = ap.parse_args()
    main(a.url, a.out)
