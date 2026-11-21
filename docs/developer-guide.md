# Developer guide

## Setup (about 2 minutes)

```bash
brew install python@3.12 ffmpeg node
uv venv .venv -p 3.12 && source .venv/bin/activate
uv pip install "google-genai>=3.2.0" opencv-contrib-python numpy scipy pymunk python-dotenv \
  "huggingface_hub[cli]" pandas matplotlib fastapi uvicorn python-multipart pyyaml tqdm joblib
cp .env.example .env            # GEMINI_API_KEY, and HF_TOKEN for Physics-IQ Verified
git clone --depth 1 https://github.com/google-deepmind/physics-IQ-benchmark third_party/physics-IQ-benchmark
cd frontend && npm install && cd ..
```

Physics-IQ Verified is gated: accept its terms on Hugging Face once. Without it, use `--source gcs` (the
original Physics-IQ files).

## Run the demo

```bash
python phase0/physiq_data.py ball-rolls-off ball-hits-nothing ball-in-basket --source gcs
python phase0/physiq_data.py ball-rolls-off ball-hits-nothing ball-in-basket --source gcs --fps 30
PYTHONPATH=backend uvicorn foresight.api:app --app-dir backend --host 0.0.0.0 --port 8000
npm --prefix frontend run dev       # open https://<lan-ip>:5173 and scan the QR code with a phone
```

## Run the eval

```bash
PYTHONPATH=backend:phase0 python eval/run_eval.py --systems static const_velocity foresight --clean
PYTHONPATH=backend:phase0 python eval/run_eval.py --systems omni --clean      # about $1 a clip, paced
PYTHONPATH=backend:phase0 python eval/readme_results.py                       # rewrites the README results
```

## Phase 0

`phase0/` holds the go/no-go checks run before building: [phase0/REPORT.md](../phase0/REPORT.md) has the
results (scoring a subset, Hydrogen perception, CSRT against Robotics-ER 2, Omni continuation).

## Rules for contributors and agents

[../CLAUDE.md](../CLAUDE.md) and [../AGENTS.md](../AGENTS.md). Model IDs live only in
`backend/foresight/config.py`; every Gemini call goes through `foresight.gemini.call`.
