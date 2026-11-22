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
- C1 (public Gemini models): `config.py` configurable via `FORESIGHT_*` env vars defaulting to `gemini-3.8-flash` (`gemini-3.6-flash` 503 fallback), structured-output fit verification (`verify.py`, 0.7 gate; `FORESIGHT_VERIFY_MODE=decisions` keeps Decision 1), `genie.py` routed through `foresight.gemini.call`, Omni/Genie off by default and skipped cleanly on the stage when disabled or 403/404, 34/34 tests passing, offline 15-clip `c1_baseline` 74.52 and `c1_public` (`gemini-3.8-flash` readings) 76.15, live sample (`j_2ff3bfd0`) and upload (`j_d44dc973`) verified (eval/FAILURES.md row 28).
- C2 (phone capture Omni/Genie toggles): QR `captureUrl` carries `&omni=0|1&genie=0|1`, `Stage.tsx` syncs live toggles via `POST /api/sessions/{session}`, `Capture.tsx` fetches `GET /api/sessions/{session}` before upload (falling back to QR query params and `/api/info`), and `POST /api/predict` honors session toggles on `source=capture`; 38/38 tests passing, offline `c2_check` 74.52, live phone-capture upload `j_74a90a6b` verified with 0 Omni and 0 Genie calls in `logs/` (eval/FAILURES.md row 29).
- C3 (short-clip synced playback): `SyncedPanels` (`Stage.tsx` + `frontend/src/sync.ts`) runs on a shared 5.0 s monotonic timeline instead of `vids()[0]` + `<video loop>`, holding shorter `reality`/`overlay` clips on their final frame (`· ended`) while 5.0 s predictions play through to 5.0 s; 4/4 frontend sync tests + 39/39 backend/physics tests passing, offline `c3_check` 74.52, live 4.0 s upload `j_c60b6c13` verified in headless Chrome (eval/FAILURES.md row 30).
- C4 (gravity scale guard + late-release cut): `track.clean_track` trims held prefixes even when released 1 frame before `t=3.0` s (`moving_after == 1`), `scene.build_scene` prunes source-occluded points (`drop_source_statics`) before scale and support snapping, and `scale.choose_scale` gates gravity scale on clean free-fall residuals (`R^2 >= 0.98`, downward drop, no bounce) and `[0.5x, 2.0x]` agreement with size scale, recording `px_per_m_reason`; 44/44 backend/physics tests passing, 0 of 60 rebuilt wide scenes outside 2x of size scale (`ball-and-block-fall_right` `9.25 -> 4447.76` px/m, endpoint `1839.8 -> 3.4` cm; `marble-run-x_center` `124.67 -> 2160.0` px/m), offline 15-clip `--rebuild` improved `76.50 -> 77.64` (cached Hydrogen) and `76.15 -> 76.89` (cached `gemini-3.8-flash`) (eval/FAILURES.md row 31).
- C7 (packaging & secret-free CI): added PEP 621 `pyproject.toml` (`setuptools.build_meta`, `[tool.pytest.ini_options]`), `.github/workflows/ci.yml` (`GEMINI_API_KEY=""`, `GOOGLE_API_KEY=""`, backend `pytest` + frontend `npm test`, `tsc --noEmit`, `oxlint`, `vite build`), `conftest.py` zero-secret/live-call guard, `eval/results_wide.json` in `Dockerfile`, and updated `docs/developer-guide.md`; 58/58 zero-`PYTHONPATH` zero-secret `pytest` tests + 4/4 frontend tests + `vite build` pass, offline `c7_rebuild` 77.64 (eval/FAILURES.md row 34).
- A1 (`foresight audit` physicality auditor API, CLI & HTML report): added `foresight.audit` (`audit_video` / `audit_continuation`), `foresight.audit_html` (`render_audit_html` self-contained SVG trajectory + speed-vs-energy-bound report), and `foresight.cli` (`foresight audit ...` console script in `pyproject.toml`) evaluating occlusion-aware object permanence (`layers.py` hidden vs vanished, duplicate respawning, size stability), vanish/early-exit-penalized fixed-time position errors (`0.5, 1.0, 2.0, 3.0, 5.0` s + last real visible), and conditioning-initialized free-fall energy plausibility alongside Physics-IQ; verified on 63/63 offline `pytest` tests, 4/4 frontend tests, real `ball-rolls-off` (`foresight` PASS `3.2` cm vs `const_velocity` `5.7` cm vs `omni` FAIL `duplicated=True` at `+0.583` s), and offline `a1_check` 77.64 (eval/FAILURES.md row 35).
- A3 (interactive counterfactual what-if API & stage UI): added `foresight.whatif` + `GET/POST /api/jobs/{job_id}/whatif` (`api.py`) stepping fitted `PhysicsWorld` parameters from `t_start_s` to `t_cond_end_s` (`3.0` s) and simulating 5.0 s counterfactual rollouts in `4.2–18.9` ms (`0` Gemini calls, `0.0000` px identity diff across saved runs, strict `job_id` validation), plus `backend/physics/tests/test_a3_whatif.py` (`69/69` `pytest` tests passing, `a3_check` 77.64; eval/FAILURES.md row 37).

### Next (handed over to Antigravity on Nov 22, see HANDOVER.md)
- Option C hardening (`C1`–`C7`) and Option A (`A1`–`A3`: `foresight audit` CLI/API/HTML, batch audit across `eval/preds/` & `eval/preds_wide/`, and interactive what-if on the stage + `/results` auditor view) are all complete.
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
- All 69 backend/physics unit & SDK-schema validation tests pass (`backend/physics/tests/`).
- C5 (per-object start times): `BodyParams.t_start_s` + scheduled `PhysicsWorld` body activation (`world.py`), per-object `t_obj_start` initial-guess and active-body rollout fitting (`fit.py`), pre-activation overlay/plot filtering (`verify.py`, `repair.py`), and 4 staggered multi-object unit tests; offline 15-clip `--rebuild` `77.64 -> 77.64` (`two-balls-pass` `99.0 -> 99.1`, `balls-collide` `86.1 -> 86.1`, `ball-train` `77.9 -> 77.9`), `perception_public` `76.89 -> 76.90` (`two-balls-pass` `98.9 -> 99.1`, `balls-collide` `100.0`, `ball-train` `58.4`), legacy cached `74.52 -> 74.48` (eval/FAILURES.md row 32).
- C6 (local multi-start repair): `config.REPAIR_MODE` defaults to `"local"` (`FORESIGHT_REPAIR_MODE=llm` keeps LLM `propose_repair`), `local_multistart_repair` (`repair.py` + `physics.predict`) refits across motion models (`roll`, `slide`, `fall`, `bounce`) and apex-rebound / chord / ballistic seeds in **0.015 s min / 0.379 s median / 3.346 s max** across the 15 4K Physics-IQ clips (0 LLM repair calls), `verify.py` smooths `selection_score` with `-0.025 * fit_error_px`, `/api/info` + `Stage.tsx` surface `"local-multistart"` / `"Local multi-start"`, and `README.md` + `docs/architecture.md` document the flow; 51/51 backend/physics tests + 4/4 frontend tests + `vite build` pass, offline 15-clip `c6_rebuild` 77.64 and `c6_public` 76.90 (eval/FAILURES.md row 33).

### Next
- Option C (`C1`–`C7`) and Option A (`A1`–`A3`) complete; next open research direction is full 3D rigid-body geometry for glancing depth collisions.

### Blockers
- None.

## Frontend

### Done
- Stage (`Stage.tsx`), phone capture (`Capture.tsx`), path plot and explanation cards (`Insights.tsx`), eval results page (`Results.tsx`).
- C2: phone QR capture honors stage Omni/Genie toggles via both QR query params and live per-session server state (`/api/sessions/{session}`).
- C3: `SyncedPanels` (`Stage.tsx`, `sync.ts`, `sync.test.ts`) drives all panels from a shared 5.0 s timeline and holds shorter clips on their final frame instead of resetting predictions early.
- C7: GitHub Actions frontend CI job (`.github/workflows/ci.yml`: `npm ci`, `npm test`, `tsc --noEmit`, `oxlint`, `vite build`) + root `pyproject.toml`.
- A3: interactive counterfactual What-If controls (`Insights.tsx`, `whatif.ts`, `whatif.test.ts`, `api.ts`) with draggable 4K SVG start-position handle, `vx`/`vy`/friction/bounce sliders, live latency & counterfactual endpoint error badges (`5.0–5.8` ms in headless Chrome), plus the `Physics auditor` comparison section on `/results` (`Results.tsx`, `7/7` frontend Node tests passing).

### Next
- Option C (`C1`–`C7`) and Option A (`A1`–`A3`) complete.

### Blockers
- None.

## Eval
### Done
- Clean 15-clip center-view eval (`eval/results.json`, `eval/results_table.md`, `eval/chart.png`), 60-scene-view wider eval (`eval/results_wide.json`, `eval/results_wide_table.md`), Carbon & Neon variants (`eval/run_variant.py`), Genie 4 streaming baseline (`eval/run_genie.py`), offline fitter loop (`eval/offline_fit.py`), occlusion-aware endpoint remeasurement (`eval/remeasure_endpoints.py`), and automated README/failure writer (`eval/readme_results.py`).
- A2 (batch physics audit over all saved eval predictions): added `eval/run_audit.py` (`ProcessPoolExecutor`, `0` Gemini calls) auditing all `105` main-eval videos (`eval/preds/`, 7 systems x 15 clips on C4-clean `c7_rebuild` scenes) and all `300` wide-eval videos (`eval/preds_wide/`, 5 systems x 60 scene-views on C4-clean `c4_wide_scenes`), producing `eval/audit_results.json`, `eval/audit_wide.json`, `eval/audit_table.md`, `GET /api/eval/audit` & `GET /api/eval/audit/wide` (`backend/foresight/api.py`, `Dockerfile`), `README.md` `## Physics auditor`, and `backend/physics/tests/test_a2_eval_audit.py` (`66/66` offline `pytest` tests passing, `a2_check` `77.64`; `eval/FAILURES.md` row 36).
  - Resolves both metric inversions: on 15-clip, Foresight (`79.78%` audit score, `93.3%` pass, `6.3` cm penalized endpoint, `5.0` cm horizon median; Carbon `82.65%`, Neon `82.51%`) beats constant velocity (`65.44%` audit score, `60.0%` pass, `13.1` cm penalized endpoint, `10.3` cm horizon median despite `77.06%` Physics-IQ) and Omni (`49.08%`, `2/15` pass, `8` vanished, `11` duplicated, `39.2` cm penalized endpoint); on 60-clip wide, Foresight ranks #1 (`76.42%` audit score, `83.3%` pass, `5.5` cm penalized endpoint, `5.3` cm horizon median) ahead of Genie 4 (`72.59%`, `10.6` cm, `7` vanished, `11` duplicated, `1` energy violation), static (`72.46%`, `17.9` cm), constant velocity (`66.87%`, `12.5` cm), and Omni (`51.17%`, `23.3%` pass, `36/60` vanished, `33/60` duplicated, `37.2` cm penalized endpoint).

### Next
- Option C (`C1`–`C7`) and Option A (`A1`–`A3`) complete; run Physics-IQ Verified GT (`--gt hf`) once gated Hugging Face dataset access is enabled.

### Blockers
- None.
