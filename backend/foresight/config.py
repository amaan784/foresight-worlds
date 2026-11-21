"""Single source of truth for model IDs and pipeline constants.

Model IDs verified against google-genai 3.6.1 (`_gaos/types/interactions/model.py`, agentoption.py).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "logs"

# --- Gemini 4 family ---
HYDROGEN = "gemini-4-hydrogen"  # scene parsing (perception); Managed Agent default
CARBON = "gemini-4-carbon"  # repair: choose a different physics model / initial guesses
NEON = "gemini-4-neon"  # 503 fallback for scene parsing
DECISION = "gemini-decision-1"  # verification (noul / choice with "other"; check .confidence)

# --- Other models ---
MANAGED_AGENT = "antigravity-preview-09-2026"  # runs batch eval fits in its sandbox
OMNI = "gemini-omni-1.2-flash"  # video-continuation baseline
ROBOTICS_ER = "gemini-robotics-er-2-preview"  # point tracking candidate (not a Gemini 4 model)

# Scene-parsing fallback pairs on repeated 503 (Hydrogen <-> Neon).
FALLBACK = {HYDROGEN: NEON, NEON: HYDROGEN}

# --- Benchmark constants (Physics-IQ) ---
BENCH_FPS = 24
COND_SECONDS = 3.0
PRED_SECONDS = 5.0
