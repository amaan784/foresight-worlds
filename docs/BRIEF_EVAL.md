# Brief — Eval agent (`eval/`)

You are building the Phase 2 evaluation for **Foresight** at the Gemini 4 NYC Hackathon (submissions due 5:00 PM ET; eval numbers needed by **4:15**; the README and demo quote ONLY numbers from your output files). Three other agents work in this repo right now: Claude Code (`backend/foresight/`: perception, tracking, orchestrator `predict_clip`, API), a physics agent (`backend/physics/`), a frontend agent (`frontend/`). **Edit only `eval/`** and your own section of `PROGRESS.md`. Import from `phase0/` and `backend/`; never edit them (tell the user if they need a fix).

First: `git pull --rebase`, then read `AGENTS.md`, `CLAUDE.md`, `CONTRACT.md` (§3 orchestrator, §5 eval layout), `phase0/REPORT.md` (already-proven facts and numbers). Env: `source .venv/bin/activate`, run from repo root with `PYTHONPATH=backend:phase0`.

## Question the eval answers
On 15 real rigid-body clips (Physics-IQ, center/side view), does physics simulation from 3 s of video predict the next 5 s better than (a) Gemini's video model continuing the clip, (b) a physics-free constant-velocity extrapolation, (c) doing nothing (static last frame)? Scored with the **official Physics-IQ metric code**.

## Data
- Clips: `RIGID_SINGLE + RIGID_MULTI` in `phase0/physiq_data.py` (15 scenes), perspective `center`, benchmark FPS **24**.
- Ground truth: Physics-IQ **Verified** (`--source hf`) if the HF gate is open (test by fetching one file; the user had to click "Agree and access"), otherwise the public original GT (`--source gcs`). Record which in `results.json` (`gt_source`). Never commit `data/`.
- Fetch: `python phase0/physiq_data.py <scenes...> --source hf|gcs --fps 24` (conditioning + take-1/take-2 testing + masks + switch frame). Also fetch the **30 FPS conditioning** clip per scene (Foresight fits on real timestamps; the 24 FPS files are frame-dropped): either `--fps 30` (small extra downloads) or a tiny helper using `needed_files(...)[0]` with fps 30.

## Systems → `eval/preds/<system>/<id>_perspective-center_trimmed-<scene>.mp4` (exactly 5.000 s, 24 FPS)
1. **`foresight`** — `from foresight.pipeline import predict_clip`; `predict_clip(cond_video=<24FPS conditioning>, fit_video=<30FPS conditioning>, out_video=..., out_fps=24, out_size=(960, 540), dynamics="physics", cache_dir=eval/cache/foresight/<scene> (dev only))`. Claude lands `predict_clip` around **3:50**; build everything else first.
2. **`omni`** — `gemini-omni-1.2-flash` video extension. Reuse the proven code in `phase0/p0c_omni_extend.py` (import its helpers; do not edit it): upload the 3 s conditioning at 1280×720, prompt = the official Physics-IQ base prompt (`descriptions/best_practice/descriptions_base.csv`) + " Continue the scene.", `generation_config={"video_config": {"task": "extend"}}`, v1beta client via `foresight.gemini.client_for(config.OMNI)`, all calls through `foresight.gemini.call`. Omni returns input + continuation; take the 5 s starting at t = 3.0 s (`extract_5s`). If the continuation is shorter than 5 s, pad by holding the last frame AND record it in FAILURES.md.
   - **Cost**: ≈ $0.10 per output second of 720p video (docs pricing: 5,792 output tokens/s at $17.50 per 1M). Phase 0's call produced a 10 s continuation ≈ **$1.04/clip → ~$15 for 15 clips**. This is over the $5 limit: **ask the user for approval with this estimate before running the batch.** You may try appending "Continue the scene for 5 seconds." to cut cost — check the returned duration.
   - **Pacing**: the spend cap is $10 per rolling 10 min shared by all agents. Run Omni sequentially, at most **one call per 90 s**. On `SpendCapHit` (429) stop and tell the user.
   - Cache raw outputs in `eval/cache/omni/<scene>.mp4` during development (the cache dir is gitignored); see "Clean run" below.
3. **`const_velocity`** — physics-free baseline with the same perception, tracking and renderer as Foresight: `predict_clip(..., dynamics="const_velocity")` (Claude implements it). Only the dynamics differ.
4. **`static`** — `phase0/static_pred.py` `static_prediction(bench_dir, scene, "center", 24, out_dir)` (last conditioning frame held 120 frames). Phase 0 measured 33.9 on 3 clips: this baseline is strong because the MSE term rewards no change — report it honestly.

## Scoring
`from score_subset import score_folder` (`phase0/score_subset.py`): `score_folder(Path("eval/preds/<system>"), bench_dir, 24, Path("eval/results"))` → `eval/results/<system>_metrics.json` (pooled `physics_iq_verified_score_pct`, `physics_iq_original_score_pct`, per-scenario `final_arithmetic`). It validates 5.000 s ±0.001 at 24 FPS and raises otherwise. Primary metric: the **Verified-formula** score (`final_score_view`).

## Outputs (commit these)
- `eval/run_eval.py` — `python eval/run_eval.py --systems foresight omni const_velocity static --scenes all [--no-cache] [--gt hf|gcs]`; idempotent; logs progress; writes everything below.
- `eval/results.json`:
  ```jsonc
  {"generated_at": "ISO8601", "git_commit": "<sha>", "clean_run": true, "gt_source": "physics-IQ-benchmark-verified",
   "view": "center", "fps": 24, "scenes": ["ball-rolls-off", ...],
   "systems": {"foresight": {"physics_iq_verified_pct": 41.2, "physics_iq_original_pct": 40.1,
                              "per_scenario_verified_pct": {"ball-rolls-off": 52.3, ...},
                              "n_ok": 15, "n_failed": 0, "latency_s_median": 21.0}, ...},
   "gemini_calls": {"hydrogen": 15, "decision": 15, "carbon": 2, "omni": 15}, "est_cost_usd": 17.3}
  ```
- `eval/results_table.md` — per-scenario rows × system columns (verified %), a pooled row, best per row in bold, a one-line note on GT source and clean-run status.
- `eval/chart.png` — bar chart of pooled verified score per system (value labels on bars, y-axis 0–100 "Physics-IQ score (Verified formula, %)", subtitle "15 rigid-body side-view clips, official metric code"), readable at slide size, colour-blind-safe; optionally a second panel with per-scenario dots.
- `eval/FAILURES.md` — every failure or weak result: clip, system, what happened (crash / lost track / wrong motion model / Omni hallucination / padding), cause, and what was changed (or "not fixed"). Include notable Omni physics failures (e.g. Phase 0: ball vanishes and respawns from the pipe).
- `eval/.gitignore`: `cache/`, `preds/` (predictions contain dataset frames — do not commit them).

## Clean run
Iterate with the cache on 2–3 clips. The **final reported numbers must come from one clean, uncached run** (`--no-cache`) over all 15 clips and all systems, with `clean_run: true` in results.json. Budget the clean run's Omni cost (~$15) — ask the user once, before the final run, and do not re-run Omni after that unless they approve.

## Rules
- Total eval traffic: a few hundred requests max. Estimated per clip: Foresight 2–4 calls (Hydrogen, Decision 1, sometimes Carbon), Omni 1.
- Commit as the configured git user, NO Co-Authored-By / "Generated with" lines, 150–300 lines per commit, `git pull --rebase` + secret scan before each push (see AGENTS.md).
- No hardcoded or hand-edited numbers anywhere; tables and charts are generated from the metrics JSON.
- Update the `## Eval` section of `PROGRESS.md` after each milestone. Report to the user at 3:45 (static + omni + harness ready) and when the clean run finishes.
