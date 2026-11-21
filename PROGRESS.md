# PROGRESS

Each agent edits ONLY its own section. Format: Done / Next / Blockers.

## Backend (Claude Code)

### Done
- CLAUDE.md rules, .env.example, gitignore (data/, eval/cache/, dataset-derived videos)
- `backend/foresight/config.py` (all model IDs) + `gemini.py` (retry/backoff, 503 Hydrogen<->Neon fallback, 429 stop, JSONL call log in logs/)
- Model IDs verified in google-genai 3.6.1 SDK source (public guides only list gemini-4-argon; SDK lists hydrogen/carbon/neon, omni-1.2-flash, antigravity-preview-09-2026, Decisions API)
- 0a scoring path: `phase0/score_subset.py` scores a single-perspective subset with the official Physics-IQ functions (run_physics_iq.py itself rejects anything but all 198 videos). Smoke-tested on 3 center-view scenes from the public original-GT bucket.
- Rigid-body scene list: 12 single-object + 3 multi-ball = 15 scenes; center perspective is the side view.
- Phase 0 run (see phase0/REPORT.md): 0b PASS (Hydrogen 12.5 s, CSRT 12/12 frames), 0d ER 2 not accurate enough (CSRT primary), 0c PASS (Omni extends, 39 s), 0a scoring PASS on original GT.

### Next
- Await go for Phase 1. Re-run 0a on Verified GT once HF terms are accepted.

### Blockers
- HF token works but Physics-IQ-Verified files return 403: user must click "Agree and access" on the dataset page
- Risk: in Physics-IQ conditioning clips the object moves only in the last 5-14 frames (0.2-0.6 s) of the 3 s window

## Physics
_(physics agent)_

## Frontend
_(frontend agent)_

## Eval
_(eval agent)_
