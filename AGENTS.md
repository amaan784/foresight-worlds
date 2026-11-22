# AGENTS.md — read before doing anything

Project: **Foresight**. "For robotics teams: watch 3 seconds of real video, simulate the physics, and predict the
next 5 — where video models get physics wrong." Built at the Gemini 4 NYC Hackathon (Nov 21 2026); the submitted
version is the branch `hackathon-submission-branch-archive` (commit `b3b3f79`) and must not change. Work since then
is on `main`.

Read in this order: this file → `HANDOVER.md` (current state and the plan) → `PROGRESS.md` → `eval/FAILURES.md`
(every failure so far, cause and fix) → `CONTRACT.md` (interfaces) → `docs/architecture.md`.

| Area | Code |
|---|---|
| Clip to scene: ingest, Gemini scene reading, tracking, scale, assets, API | `backend/foresight/` |
| Physics: depth layers, world, fit, verify/repair, simulate, render | `backend/physics/` (tests in `backend/physics/tests/`) |
| Stage and phone capture (Vite, React, Tailwind) | `frontend/` |
| Evals, results, failures | `eval/`, scoring helpers in `phase0/` |

Non-negotiable rules:
- Commit as the configured git user, about 450–600 changed lines per commit. NEVER add `Co-Authored-By`,
  "Generated with", or any AI identity line to commits or PRs. Push only when the user says so.
- Before every push: `git pull --rebase`, then a secret scan, `git grep -nE "AIza[0-9A-Za-z_-]{35}" $(git rev-list --all)`
  (must print nothing). If push/pull asks for credentials:
  `git -c credential.helper= -c 'credential.helper=!gh auth git-credential' push`.
- Secrets only in `.env` (gitignored; `.env.example` lists the names). Never print keys. Never commit `data/` or any
  Physics-IQ dataset file; credit Physics-IQ (code Apache-2.0, materials CC BY 4.0).
- No mocks, no hardcoded or cached outputs presented as live. Every number in the README or the stage comes from an
  eval output file (`eval/readme_results.py` writes the README's results section).
- Model IDs only from `backend/foresight/config.py`; every Gemini call through `foresight.gemini.call(...)` (retry,
  429 stop, logging to `logs/`). Look up model names and API fields in the current official Gemini API docs before
  writing Gemini code. Native structured outputs only (no "reply in JSON" prompting). Never pass temperature,
  top_p, top_k or thinking_budget.
- Ask the user before any batch of model calls likely to cost over $5. On HTTP 429, stop and tell the user.
- Judge physics changes on the same cached scene readings (`eval/offline_fit.py --perception DIR`): Gemini's
  readings vary between runs by about a point of score. The 15 center-view clips were used while developing; the
  wider eval's left/right cameras and added scenes were not.
- Update `PROGRESS.md` and add a row to `eval/FAILURES.md` after each milestone. No Streamlit.
