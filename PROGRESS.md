# PROGRESS

Each agent edits ONLY its own section. Format: Done / Next / Blockers.

## Backend (Claude Code)

### Done (15:22)
- Phase 0 complete (phase0/REPORT.md): Hydrogen perception PASS, CSRT PASS (ER 2 less accurate), Omni continuation PASS, subset scoring PASS (original GT).
- backend/foresight: perceive (Hydrogen, cached), track (gated CSRT, 30 FPS real timestamps), scale (size + gravity check), scene (CONTRACT §1 scene.json, temporal background fill, sprites, keyframes), pipeline.predict_clip (physics | const_velocity, endpoint error, overlay), omni (continuation lane), api (jobs, SSE with replay + `end`, samples, uploads).
- Verified live: sample job 18 s end to end (Hydrogen 15 s); phone-style VP9 webm upload → 3.0 s cond / 5.0 s reality / 5.0 s prediction in 22 s.
- Fixture for physics agent: phase0/out/scene_ball-rolls-off/scene.json.

### Next
- Integrate backend/physics as soon as `physics.predict` lands; run a physics job end to end; help eval wire predict_clip.

### Blockers
- backend/physics, frontend, eval not started yet (no commits as of 15:22).
- HF Physics-IQ-Verified still 403 until the user clicks "Agree and access".

## Physics

### Done
- Step 1 (`backend/physics/world.py`): pymunk world with y-down px -> y-up m conversion, finite static surface segments, static boxes with overlapping launch-pipe filtering, dynamic circle/box bodies, fixed `gravity_m_s2`, and per-body contact rolling resistance damping (7/7 tests passing).
- Step 2 (`backend/physics/fit.py`): two-stage `scipy.optimize.least_squares` fitter on real timestamps `t_s` with fixed gravity, strong physical priors, surface non-penetration bounds, and identifiability gating (7/7 synthetic tests passing: 5–13 pt tracks, <5% pos/vel error, <5% elasticity on bounce, <5% friction on sliding deceleration, <5% damping on rolling deceleration, noisy 4K roll keeps unidentifiable priors).
- Step 3 (`backend/physics/simulate.py`): exact 5.000 s forward rollout from `t_cond_end_s` (120 frames @ 24 FPS) + `evaluate_at_times` helper for conditioning keyframes.
- Step 4 (`backend/physics/render.py`): premultiplied alpha sprite compositing with rotation and off-screen skip, H.264 MP4 output via `phase0/vidio.write_mp4`, verified 5.000 s ±0.001 via `phase0/vidio.probe`.
- Step 5 (`backend/physics/verify.py`): `gemini-decision-1` verification on 3–4 conditioning keyframes with fitted trajectory outlines drawn at each keyframe time (`noul` outline check + `choice` motion check with `roll`/`slide`/`fall`/`bounce`/`other`), checking `.confidence` and noul truth against 0.7 threshold, routed via `foresight.gemini.call` + `client_for(config.DECISION)` (v1beta).
- Step 6 (`backend/physics/repair.py` + `physics.predict`): 2-panel diagnostic trajectory plot + keyframes -> `gemini-4-carbon` (`thinking_level: "medium"`, native structured output `RepairProposal`) -> bounded refit up to max 2 repairs; full `fit` -> `verify` -> `repair` -> `render` loop wired into `physics.predict` with CONTRACT.md §4 SSE events.
- Step 7 (`backend/physics/agent_runner.py`): optional Managed Agent (`antigravity-preview-09-2026`, v1beta) batch fitter reusing one `environment_id` (installs pymunk/scipy/opencv once) + local batch fallback.
- Real fixture verification: fitted Claude Code's `phase0/out/scene_ball-rolls-off/scene.json` (14 points @ 30 FPS, 4K) -> `fit_error_px = 7.88 px` (~2.2 mm, 0.065 ball radii), `vx0 = 1.265 m/s`, identified `damping = 0.424 s^-1`, unidentifiable friction/elasticity kept at priors (`0.50`, `0.65`), rolls cleanly off finite table at `t ~ 3.45 s`.
- All 21 unit & SDK-schema validation tests pass (`backend/physics/tests/`).

### Next
- Once `.env` with `GEMINI_API_KEY` and real `scene_ball-rolls-off` PNGs are present locally (or on Claude Code's machine), run live `verify_fit` / `propose_repair` / `predict` smoke test to confirm `gemini-decision-1` and `gemini-4-carbon` live responses and check `logs/genai_calls_*.jsonl`.
- Support integration / tuning across all 15 rigid-body Physics-IQ eval scenes if any edge cases surface.

### Blockers
- Local checkout has no `.env` (`GEMINI_API_KEY` not set) and `phase0/out/scene_ball-rolls-off/` PNGs are gitignored/not local — live Gemini API smoke test of Steps 5–7 requires `.env` with `GEMINI_API_KEY`.

## Frontend
_(frontend agent)_

## Eval
_(eval agent)_
