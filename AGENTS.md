# AGENTS.md — read before doing anything

Project: **Foresight** (Gemini 4 NYC Hackathon, Nov 21 2026, submissions due 5:00 PM ET, demo recorded 4:30).
"For robotics teams: watch 3 seconds of real video, simulate the physics, and predict the next 5 — where video models get physics wrong."

Four agents work in this repo at the same time. Stay inside your directory.

| Agent | Owns | Brief |
|---|---|---|
| Claude Code | `backend/foresight/`, `phase0/`, `CONTRACT.md`, `README.md`, root `.gitignore` | — |
| Physics agent | `backend/physics/` | `docs/BRIEF_PHYSICS.md` |
| Frontend agent | `frontend/` | `docs/BRIEF_FRONTEND.md` |
| Eval agent | `eval/` | `docs/BRIEF_EVAL.md` |

Read in this order: `CLAUDE.md` (rules) → `CONTRACT.md` (interfaces) → your brief → `PROGRESS.md` → `phase0/REPORT.md` (what is already proven).

Non-negotiable rules (full list in CLAUDE.md):
- Commit as the configured git user. NEVER add `Co-Authored-By`, "Generated with", or any AI identity line to commits or PRs. 150–300 changed lines per commit.
- Before every push: `git pull --rebase`, then secret scan `git grep -nE "AIza" $(git rev-list --all)` (must print nothing). If push/pull asks for credentials: `git -c credential.helper= -c 'credential.helper=!gh auth git-credential' push`.
- Secrets only in `.env` (exists, gitignored). Never print keys. Never commit `data/` or any Physics-IQ dataset file.
- No mocks, no hardcoded or cached outputs presented as live. Every number in README/demo comes from an eval output file.
- Model IDs only from `backend/foresight/config.py`; every Gemini call through `foresight.gemini.call(...)`; preview models via `foresight.gemini.client_for(model)` (v1beta). 429 = project spend cap → stop and tell the user. Ask the user before any batch likely to cost > $5. Cap is $10 per rolling 10 min shared by ALL agents.
- Never pass temperature/top_p/top_k/thinking_budget; never thinking_level "minimal" on Hydrogen/Neon; native structured outputs only (no "reply in JSON" prompting). Confirm Gemini field names in the SDK source or with `python phase0/docs_mcp.py search "..."`.
- No Streamlit. Update only your own section of `PROGRESS.md` after each milestone.
