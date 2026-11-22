"""Single source of truth for model IDs and pipeline constants.

Every model ID and baseline toggle is configurable via environment variables,
with defaults that run out of the box on a standard personal Gemini API key.
Model IDs verified against the official Gemini API docs and google-genai 3.6.1.
"""
import os as _os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "logs"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# --- Public GA defaults + historical Gemini 4 identifiers ---
FLASH_38 = "gemini-3.8-flash"
FLASH_36 = "gemini-3.6-flash"
ARGON = "gemini-4-argon"
HYDROGEN = "gemini-4-hydrogen"  # Nov 21-22 eval scene parser
NEON = "gemini-4-neon"  # Nov 21-22 503 fallback / variant

# Scene reading: gemini-3.8-flash by default (GA, Free + Paid tier, agentic video + structured output).
PERCEPTION = _os.environ.get("FORESIGHT_PERCEPTION_MODEL", FLASH_38)
PERCEPTION_THINKING = _os.environ.get("FORESIGHT_PERCEPTION_THINKING", "low")
FALLBACK_MODEL = _os.environ.get("FORESIGHT_FALLBACK_MODEL", FLASH_36)

# Verification: structured output on gemini-3.8-flash by default; set FORESIGHT_VERIFY_MODE=decisions for Decision 1.
VERIFY = _os.environ.get("FORESIGHT_VERIFY_MODEL", FLASH_38)
VERIFY_MODE = _os.environ.get("FORESIGHT_VERIFY_MODE", "structured").strip().lower()
VERIFY_THINKING = _os.environ.get("FORESIGHT_VERIFY_THINKING", "low")
DECISION = _os.environ.get("FORESIGHT_DECISION_MODEL", "gemini-decision-1")

# Repair: local deterministic multi-start by default; set FORESIGHT_REPAIR_MODE=llm to query FORESIGHT_REPAIR_MODEL.
REPAIR_MODE = _os.environ.get("FORESIGHT_REPAIR_MODE", "local").strip().lower()
REPAIR = _os.environ.get("FORESIGHT_REPAIR_MODEL", FLASH_38)
CARBON = REPAIR
REPAIR_THINKING = _os.environ.get("FORESIGHT_REPAIR_THINKING", "low")
REPAIR_BUDGET_S = float(_os.environ.get("FORESIGHT_REPAIR_BUDGET_S", "90"))  # whole check-and-repair loop

# --- Optional / paid-tier baselines & agents ---
OMNI = _os.environ.get("FORESIGHT_OMNI_MODEL", "gemini-omni-1.1-flash")
OMNI_ENABLED = _env_bool("FORESIGHT_OMNI_ENABLED", False)
GENIE = _os.environ.get("FORESIGHT_GENIE_MODEL", "genie-4")
GENIE_ENABLED = _env_bool("FORESIGHT_GENIE_ENABLED", False)
MANAGED_AGENT = _os.environ.get("FORESIGHT_MANAGED_AGENT", "antigravity-preview-09-2026")
ROBOTICS_ER = _os.environ.get("FORESIGHT_ROBOTICS_ER_MODEL", "gemini-robotics-er-2-preview")

# Models served only on the v1beta API.
V1BETA_MODELS = {
    ROBOTICS_ER,
    OMNI,
    "gemini-omni-1.1-flash",
    "gemini-omni-1.2-flash",
    DECISION,
    MANAGED_AGENT,
    GENIE,
}

# Scene-parsing / verification fallback pairs on repeated 503.
_RAW_FALLBACK = {
    FLASH_38: FLASH_36,
    FLASH_36: FLASH_38,
    ARGON: FLASH_38,
    HYDROGEN: NEON,
    NEON: HYDROGEN,
    FALLBACK_MODEL: PERCEPTION,
    PERCEPTION: FALLBACK_MODEL,
}
FALLBACK = {k: v for k, v in _RAW_FALLBACK.items() if k and v and k != v}

# --- Benchmark constants (Physics-IQ) ---
BENCH_FPS = 24
COND_SECONDS = 3.0
PRED_SECONDS = 5.0
