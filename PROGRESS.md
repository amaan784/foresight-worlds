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
- Step 4 (`backend/physics/render.py` + `physics.predict` export): premultiplied alpha sprite compositing with rotation and off-screen skip, H.264 MP4 output via `phase0/vidio.write_mp4`, verified 5.000 s ±0.001 via `phase0/vidio.probe` (16/16 total tests passing in 2.5 s).

### Next
- Step 5 (`backend/physics/verify.py`): confirm Decisions API types in SDK source and implement `gemini-decision-1` keyframe verification (fitted trajectory outline + motion choice with `other` + `.confidence`).
- Step 6 (`backend/physics/repair.py`): `gemini-4-carbon` repair loop (thinking_level `medium`, structured output, max 2 repairs) wired into `physics.predict`.
- Test end-to-end on real `phase0/out/scene_ball-rolls-off/scene.json` once committed by Claude Code.
- Step 7 (optional): `agent_runner.py` Managed Agent batch runner if time permits.

### Blockers
- `.env` with `GEMINI_API_KEY` is not yet present in this local checkout — needed to run live verification/repair tests for Steps 5–6 (no mocks).

## Frontend
_(frontend agent)_

## Eval
_(eval agent)_
