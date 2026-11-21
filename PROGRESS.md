# PROGRESS

## Done
- CLAUDE.md rules, .env.example, gitignore (data/, eval/cache/, dataset-derived videos)
- `backend/foresight/config.py` (all model IDs) + `gemini.py` (retry/backoff, 503 Hydrogen<->Neon fallback, 429 stop, JSONL call log in logs/)
- Model IDs verified in google-genai 3.6.1 SDK source (public guides only list gemini-4-argon; SDK lists hydrogen/carbon/neon, omni-1.2-flash, antigravity-preview-09-2026, Decisions API)
- 0a scoring path: `phase0/score_subset.py` scores a single-perspective subset with the official Physics-IQ functions (run_physics_iq.py itself rejects anything but all 198 videos). Smoke-tested on 3 center-view scenes from the public original-GT bucket.
- Rigid-body scene list: 12 single-object + 3 multi-ball = 15 scenes; center perspective is the side view.
- 0b/0c/0d scripts written; CSRT path smoke-tested offline (appearance-gated, forward+backward merge).

## Next
- Run 0a on Physics-IQ Verified (needs HF access), 0b Hydrogen perception, 0d ER 2 vs CSRT, 0c Omni extend; write Phase 0 report.

## Blockers
- GEMINI_API_KEY not in .env
- HF dataset Anates-Labs-Research/Physics-IQ-Verified is gated (auto-approve): user must accept terms + `hf auth login` or HF_TOKEN in .env
- Risk: in Physics-IQ conditioning clips the object moves only in the last 5-14 frames (0.2-0.6 s) of the 3 s window
