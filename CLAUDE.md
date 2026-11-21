# Foresight — agent rules

Foresight: for robotics teams. Watch 3 s of real video, simulate the physics, predict the next 5 s — where video models get physics wrong.
Built during the Gemini 4 NYC Hackathon (Nov 21 2026, submissions due 5:00 PM ET). Track: Element 119. Everything built today, run live (no mocks), public repo.

## Gemini API rules
- Gemini 4 APIs are newer than model training data. Before writing ANY Gemini / Decisions / Managed Agents / Omni / Robotics-ER code, confirm field names via the Gemini docs MCP (`https://gemini-api-docs-mcp.dev`; skill: `google-gemini/gemini-skills --skill gemini-api-dev`). Use `google-genai>=3.2.0`.
- Model IDs: `gemini-4-hydrogen`, `gemini-4-carbon`, `gemini-4-neon`, `gemini-decision-1`, Managed Agent `antigravity-preview-09-2026` (defaults to Hydrogen), video baseline `gemini-omni-1.2-flash`, tracker candidate `gemini-robotics-er-2-preview` (via `client.interactions.create`; NOT a Gemini 4 model).
- Model jobs: Hydrogen = scene parsing + Managed Agent; Carbon = repair; Decision 1 = verification; ER 2 = point tracking (if accurate), CSRT fills frames between ER 2 samples.
- Clients: `client = genai.Client()` for model calls; `beta_client = genai.Client(http_options={"api_version": "v1beta"})` for Decisions + Managed Agents.
- Never pass temperature / top_p / top_k / thinking_budget. Never use thinking_level "minimal" on Neon/Hydrogen. No manual context caching.
- Native structured outputs only — never "please reply in JSON" prompting.
- Decision 1: choice questions ALWAYS include an "other" key; always check `.confidence` in code.
- Video inputs to Gemini 4: `{"type": "video", "uri": ..., "processing": "agentic"}` via `client.files.upload`.
- Managed Agent accepts text+images (verify in docs): send trajectories/keyframes, not raw video. Reuse one `environment_id` so packages persist.

## Spend
- Tier 1 cap: $10 per rolling 10 min. Pace batch calls. Ask the user before any batch likely to exceed $5.

## Secrets
- Secrets in `.env` only (commit `.env.example`). Never print keys.

## Git
- Commit with the user's configured git identity. NEVER add Co-Authored-By, "Generated with", or any AI identity lines to commits or PRs.
- Commit in batches of roughly 150–300 changed lines; author is the user (configured identity).
- Always `git pull --rebase` before pushing (other agents work in `frontend/` and `eval/`).

## Product rules
- No Streamlit (banned).
- No hardcoded or cached outputs presented as live.
- Every number in README/demo must come from an eval script output file.
- Do not commit dataset files (Physics-IQ: code Apache-2.0, materials CC BY 4.0 — attribute in README).
- Keep `PROGRESS.md` (done / next / blockers) updated after each milestone.

## Layout
- `backend/foresight/` — pipeline package (ingest, perceive, track, scale, fit, simulate, render, verify) + FastAPI.
- `eval/` — Physics-IQ subset eval (results.json, tables, chart, FAILURES.md).
- `frontend/` — Vite + React + Tailwind demo.
- `phase0/` — go/no-go scripts.
- `data/` — local dataset files (gitignored).
