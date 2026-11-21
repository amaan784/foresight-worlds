FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/backend \
    MPLCONFIGDIR=/tmp/foresight_mplconfig \
    PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      "fastapi>=0.115" \
      "uvicorn[standard]>=0.32" \
      "python-multipart>=0.0.12" \
      "google-genai>=3.6.1" \
      "pymunk>=7.0" \
      "scipy>=1.14" \
      "opencv-contrib-python-headless>=4.10" \
      "numpy>=2.0" \
      "pillow>=11.0" \
      "pydantic>=2.10" \
      "python-dotenv>=1.0" \
      "matplotlib>=3.9" \
      "pyyaml>=6.0"

COPY backend ./backend
COPY phase0/vidio.py ./phase0/vidio.py
COPY eval/results.json eval/chart.png ./eval/

RUN mkdir -p /app/outputs /app/logs /tmp/foresight_mplconfig

# Fetch 6 showcase Physics-IQ clips + metadata at build time (never committed to git)
RUN printf '%s\n' \
    'import pathlib, urllib.request, yaml' \
    'root = pathlib.Path("/app")' \
    'desc = root / "third_party/physics-IQ-benchmark/descriptions"' \
    '(desc / "best_practice").mkdir(parents=True, exist_ok=True)' \
    'raw_base = "https://raw.githubusercontent.com/google-deepmind/physics-iq-benchmark/main/descriptions"' \
    'urllib.request.urlretrieve(f"{raw_base}/data.yaml", desc / "data.yaml")' \
    'urllib.request.urlretrieve(f"{raw_base}/best_practice/descriptions_base.csv", desc / "best_practice/descriptions_base.csv")' \
    'scenes = {s["scene"]: s for s in yaml.safe_load((desc / "data.yaml").read_text())["scenes"]}' \
    'picks = ["ball-rolls-off", "ball-rolls-on-glass", "balls-collide", "ball-ramp", "roll-front-box", "ball-hits-nothing"]' \
    'gcs = "https://storage.googleapis.com/physics-iq-benchmark"' \
    'dest = root / "data/physics-IQ-benchmark"' \
    'for name in picks:' \
    '    t1_num = next(t["center"] for t in scenes[name]["takes"] if t["take"] == 1)' \
    '    t1 = f"{int(t1_num):04d}"' \
    '    for rel in (' \
    '        f"split-videos/conditioning/24FPS/{t1}_conditioning-videos_24FPS_perspective-center_take-1_trimmed-{name}.mp4",' \
    '        f"split-videos/conditioning/30FPS/{t1}_conditioning-videos_30FPS_perspective-center_take-1_trimmed-{name}.mp4",' \
    '        f"split-videos/testing/24FPS/{t1}_testing-videos_24FPS_perspective-center_take-1_trimmed-{name}.mp4",' \
    '    ):' \
    '        out = dest / rel' \
    '        out.parent.mkdir(parents=True, exist_ok=True)' \
    '        urllib.request.urlretrieve(f"{gcs}/{rel}", out)' \
    '    print("fetched", name, t1, flush=True)' \
    'from foresight.api import _list_samples' \
    'print("samples ready:", [s["id"] for s in _list_samples()], flush=True)' \
    > /tmp/fetch_samples.py && python /tmp/fetch_samples.py && rm -f /tmp/fetch_samples.py

EXPOSE 8000

CMD ["sh", "-c", "uvicorn foresight.api:app --app-dir backend --host 0.0.0.0 --port ${PORT:-8000}"]
