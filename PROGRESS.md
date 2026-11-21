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
_(physics agent)_

## Frontend
_(frontend agent)_

## Eval
_(eval agent)_
