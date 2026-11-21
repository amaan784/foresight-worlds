# Brief — Physics agent (`backend/physics/`)

You are building backend/physics/ for "Foresight" at the Gemini 4 NYC Hackathon (submissions due 5:00 PM ET; demo recording 4:30). Other agents work in parallel: Claude Code owns backend/foresight/ (perception, tracking, scale, scene assembly, API); other agents own frontend/ and eval/. Edit ONLY backend/physics/ and your own section of PROGRESS.md.

First: git pull --rebase, then read AGENTS.md, CLAUDE.md and CONTRACT.md and follow them:
- commit as the configured git user; NO Co-Authored-By / "Generated with" / AI lines; 150–300 lines per commit
- before every push: git pull --rebase, then secret scan (git grep -nE "AIza" $(git rev-list --all)); if push asks for credentials use: git -c credential.helper= -c 'credential.helper=!gh auth git-credential' push
- no mocks; secrets only in .env; model IDs only from backend/foresight/config.py
- every Gemini call through foresight.gemini.call (retry/backoff; 429 = spend cap → stop and tell the user); preview models (gemini-decision-1, Managed Agent) via foresight.gemini.client_for(model) (v1beta)
- confirm Gemini field names in the SDK source (.venv/lib/python3.12/site-packages/google/genai/_gaos/types/) or with: python phase0/docs_mcp.py search "..."
- run from repo root with: source .venv/bin/activate; PYTHONPATH=backend

Input: the scene JSON in CONTRACT.md §1 (original-resolution pixels, y DOWN → flip to y-up internally).
Output: physics.predict(scene, on_event=None) → the dict in CONTRACT.md §2 (prediction_mp4, fit_params, fit_error_px, motion_model, decision1_confidences, decision1_answers, repairs_used, trajectory, fit_trajectory). Also export physics.render.render_trajectory(scene, trajectory, out_path, out_size=None) — the const-velocity baseline reuses it. Call on_event("fit"|"verify"|"repair"|"render", payload) as in CONTRACT.md §4.

Build in order; commit after each step passes:
1. world.py — pymunk world: static segments from `surfaces` (finite — a ball must fall off the end of the table), static boxes from `static_objects`, dynamic circle/box bodies; metres via px_per_m; gravity FIXED at gravity_m_s2; per-body velocity damping (pymunk has no rolling resistance).
2. fit.py — scipy.optimize.least_squares on the track's real timestamps (t_s), not frame indices. Tracks are short (5–14 points, 0.2–0.6 s). FIRST write a synthetic test: simulate a ball with known params, sample 5–14 points like the real data, require initial position/velocity recovered within 5%. Require friction/elasticity within 5% only on a synthetic track that contains a bounce or clear deceleration — otherwise they are unidentifiable; use priors + bounds (tennis ball on wood: elasticity ~0.6–0.75, friction ~0.5).
3. simulate.py — run exactly 5.000 s from t_cond_end_s; one state per output frame (120 at 24 FPS).
4. render.py — background_png + sprite_png at simulated poses → MP4 via phase0/vidio.write_mp4; verify exactly 5.000 s ±0.001 with phase0/vidio.probe. Skip objects once off-screen.
5. verify.py — gemini-decision-1 on 3–4 `keyframes`, drawing the FITTED trajectory outline at each keyframe time (conditioning window, not the 5 s prediction): noul "Does the drawn outline sit on the real object?" + choice motion check with keys roll / slide / fall / bounce / other; read and check .confidence in code.
6. repair.py — if any confidence < 0.7: gemini-4-carbon (thinking_level "medium", native structured output) gets a trajectory plot + keyframes and proposes a motion model + initial guesses; refit; max 2 repairs.
7. agent_runner.py — OPTIONAL, last: batch fits inside the Managed Agent "antigravity-preview-09-2026" with one reused environment_id (install pymunk/scipy/opencv once). If its API blocks you for more than 15 minutes, stop and report; eval will run locally.

Real test data (ready now): phase0/out/scene_ball-rolls-off/scene.json (committed) with the cached Hydrogen perception in phase0/out/scene_ball-rolls-off/cache/. The PNGs it references (background, sprite, keyframes) are dataset frames and are NOT committed; if they are missing, regenerate them in ~5 s with no API key:
  python phase0/physiq_data.py ball-rolls-off --source gcs --fps 24 && python phase0/physiq_data.py ball-rolls-off --source gcs --fps 30
  PYTHONPATH=backend python -m foresight.scene data/physics-IQ-benchmark/split-videos/conditioning/24FPS/0023_conditioning-videos_24FPS_perspective-center_take-1_trimmed-ball-rolls-off.mp4 phase0/out/scene_ball-rolls-off --fit-video data/physics-IQ-benchmark/split-videos/conditioning/30FPS/0023_conditioning-videos_30FPS_perspective-center_take-1_trimmed-ball-rolls-off.mp4 --cache-dir phase0/out/scene_ball-rolls-off/cache
Ground truth for that clip (to eyeball your prediction): data/physics-IQ-benchmark/split-videos/testing/24FPS/0023_testing-videos_24FPS_perspective-center_take-1_trimmed-ball-rolls-off.mp4 (ball rolls right at ~1.2 m/s and falls off the table edge).
Report at 3:30 (expect steps 1–4 done) and keep your PROGRESS.md section updated.
