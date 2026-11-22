# PROGRESS

Each agent edits ONLY its own section. Format: Done / Next / Blockers.

## Backend (Claude Code)

### Done (hackathon)
- Phase 0 (phase0/REPORT.md); backend/foresight pipeline + API; frontend stage and phone capture; eval harness; README, docs, images.
- Clean eval, 15 Physics-IQ clips, center view, original GT (eval/results.json): constant velocity 75.22, Foresight 58.97, Foresight with Carbon reading 58.8, Genie 4 42.23, last frame 34.96, Omni 1.2 27.51.

### Done after the hackathon (Nov 22)
- Hackathon state kept on local branch `hackathon-submission-branch-archive` (commit b3b3f79).
- Physics: track cleaning, one support line per depth, depth layers (block / hide / behind), bodies at rest that can be knocked, open containers, solid and level tables, soft surfaces, rolling resistance, energy-check fallback to constant velocity, Carbon repair at thinking low inside a 90 s budget (eval/FAILURES.md rows 10-24).
- Hydrogen now reports surface material (hard/soft) and object mass.
- Genie 4 on the stage: API lane next to Omni, toggle, fourth reveal panel.
- eval/offline_fit.py (free development loop), eval/run_wide.py (20 scenes x 3 cameras), scorer mask-cache bug fixed.
- Clean eval, 15 clips, center view (eval/results.json, Nov 22): constant velocity 77.06, Foresight with Carbon reading 76.8, Foresight 74.37, Genie 4 42.23 (Nov 21), last frame 34.96, Omni 27.51 (Nov 21). Median endpoint error (re-measured with the occlusion-aware real track): Foresight 9.9 cm, constant velocity 192.5 cm; Foresight closer on 12 of 14 clips.
- Wider eval, 20 scenes x 3 cameras (eval/results_wide.json): constant velocity 66.63, Foresight 64.73, Genie 4 46.52, last frame 36.48, Omni 30.11; median endpoint Foresight 18.2 cm, constant velocity 154.8 cm; Foresight closer on 36 of 51 scene-views. No failures, no fallbacks.
- Stage: drop a video from the laptop, recent runs, path plot, "what Foresight understood", trajectory JSON/CSV, /results page.
- README: "Post-hackathon changes" section; submitted version is branch hackathon-submission-branch-archive at b3b3f79.

### Next
- Push main and the archive branch when the user says so (the README links the branch).
- Physics-IQ Verified ground truth: needs the gated Hugging Face dataset accepted by the account owner.
- Glancing collisions in depth (balls-collide, ball-hits-duck) are out of reach for a 2D world.

### Blockers
- none

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
- Live Gemini API verification (all calls logged in `logs/genai_calls_20261121.jsonl`):
  - `gemini-decision-1` (`verify.py`, v1beta, 1.7 s): good rolling fit passed (`outline_on_object='yes'` 0.8275, `motion='roll'` 0.9599); corrupted bouncing fit failed (`outline_on_object='no'` P=0.07, `motion='fall'` 0.4004 < 0.7).
  - `gemini-4-carbon` (`repair.py`, `thinking_level='medium'`, structured output): diagnosed `'bounce'` from 2-panel plot + keyframes, estimated `vy0=-1.1 m/s` and `elasticity=0.82`; refit dropped RMSE `185.0 px -> 0.003 px` and passed re-verification (`outline='yes'` 0.8276, `motion='bounce'` 0.8957).
  - `antigravity-preview-09-2026` (`agent_runner.py`, v1beta): provisioned sandbox once (79 s), reused `environment_id` on `scene_ball-rolls-off/scene.json` (27 s), reproduced exact local fit (`fit_error_px=7.884`, `vx0=1.2649`, `damping=0.4242`); added fenced-JSON parser for Managed Agent markdown output.
- All 22 unit & SDK-schema validation tests pass (`backend/physics/tests/`).

### Next
- Support integration / tuning across all 15 rigid-body Physics-IQ eval scenes as `eval/` runs end-to-end.

### Blockers
- None.

## Frontend
_(frontend agent)_

## Eval
_(eval agent)_
