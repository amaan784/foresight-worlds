# Handover: Foresight after the hackathon

Written Nov 22, 2026, when development moved from Claude Code to Antigravity. Read `AGENTS.md` (rules) first, then
this file, then `PROGRESS.md` and `eval/FAILURES.md` (every failure found so far, with the fix and the reason).

## What Foresight is

For robotics teams: watch 3 s of real side-view video, read the scene with Gemini, fit a 2D physics world to what
was seen (pymunk), and predict the next 5 s. It is compared with video models (Gemini Omni, Genie 4) and two
baselines (constant velocity, last frame held) on Physics-IQ clips with the official metric code.

- The version submitted to the hackathon is the branch `hackathon-submission-branch-archive`, commit `b3b3f79`.
  Do not change that branch.
- Everything after it is post-hackathon work on `main`; the README section "Post-hackathon changes" lists it.

## Where things stand (numbers come from the files named, never retype them by hand)

- 15-clip clean eval, center camera: `eval/results.json`, `eval/results_table.md`, `eval/chart.png`. Foresight is
  above both video models on the Physics-IQ score and below constant velocity; on endpoint error it is closer than
  constant velocity on most clips.
- Wider eval, 20 scenes x 3 cameras (60 scene-views, most never used while developing): `eval/results_wide.json`,
  `eval/results_wide_table.md`. Same Physics-IQ ordering; here the video models' median endpoint errors are lower
  than Foresight's, and Foresight is near the held last frame on the five added scenes (ball-in-sand, two marble
  runs, two block stacks).
- Scene-reading variants (Carbon, Neon) are rows in `eval/results.json`; the Managed Agent's remote fits are in
  `eval/agent_batch.json` (identical to the local fits).
- Every Gemini call is logged in `logs/genai_calls_YYYYMMDD.jsonl` (media stripped).

## How to run

Setup: `docs/developer-guide.md`. Then:

```bash
source .venv/bin/activate
PYTHONPATH=backend uvicorn foresight.api:app --app-dir backend --host 0.0.0.0 --port 8000
npm --prefix frontend run dev                                   # stage on https://<lan-ip>:5173 (http on 5174 with FORESIGHT_HTTP=1)
PYTHONPATH=backend:phase0 python -m pytest -q backend/physics/tests    # 29 tests, no model calls
PYTHONPATH=backend:phase0 python eval/offline_fit.py --tag dev --rebuild   # dynamics only, cached Gemini readings, free
PYTHONPATH=backend:phase0 python eval/run_eval.py --systems foresight const_velocity static --clean   # live, about $1-2
PYTHONPATH=backend:phase0 python eval/run_wide.py --systems foresight const_velocity static         # live, about $5
PYTHONPATH=backend:phase0 python eval/remeasure_endpoints.py main   # re-measure endpoints, no model calls
PYTHONPATH=backend:phase0 python eval/run_eval.py --merge-only && PYTHONPATH=backend:phase0 python eval/readme_results.py
```

`eval/offline_fit.py` is the development loop: it reruns fit, simulate, render and the official score on cached
scene readings in a few minutes. It runs on the same 15 clips the main eval reports, so it is not a held-out test;
judge changes on the wider eval's left/right cameras and added scenes too.

## The plan: Option C (hardening), then Option A (physics auditor)

Do C1 first and in full. Then the rest of C in order, then A. After each task: run the tests and the offline eval,
add a row to `eval/FAILURES.md` (what happened, cause, change, measured effect), update your section of
`PROGRESS.md`, commit.

### C1. Run on publicly available Gemini models (urgent)

The hackathon's temporary Gemini account is deleted on Nov 22, the day after the event. Its key and its access to
the Gemini 4 previews
(`gemini-4-hydrogen`, `gemini-4-carbon`, `gemini-4-neon`, `gemini-decision-1`, `gemini-omni-1.2-flash`, `genie-4`,
`antigravity-preview-09-2026`, `gemini-robotics-er-2-preview`) are gone, so nothing live runs until this is done.

- Look up the current publicly available model IDs and API surfaces in the official Gemini API docs before writing
  code. Do not trust model names from memory or from another model's suggestion.
- Make every model ID configurable through environment variables in `backend/foresight/config.py` (it is already
  the single source of model IDs), with defaults that work on a personal key.
- Scene reading (`perceive.py`) needs video input plus native structured output. Repair (`repair.py`) needs images
  plus structured output.
- Verification (`physics/verify.py`) uses the Decisions API (`client.decisions.create`, confidences, an "other"
  option). If no public equivalent exists, implement the same two questions (does the outline sit on the object;
  which motion: roll/slide/fall/bounce/other) with a standard model, native structured output and a stated
  confidence, keep the 0.7 gate, and say in the README that the check changed.
- Omni, Genie 4 and the Managed Agent may have no public equivalent. The stage must then skip those lanes with a
  clear message (`/api/info` already reports what is enabled), not crash. The eval keeps their Nov 21-22 rows and
  says when they were run.
- Keep every call going through `foresight.gemini.call` (retry, 429 stop, logging).
- Done when: with a personal key in `.env`, a live stage run of a Physics-IQ sample and a phone/laptop upload both
  finish; `eval/offline_fit.py --rebuild --perception eval/cache/perception_public` works end to end; the README's
  models table names the models actually used.

### C2. Phone captures ignore the stage's Omni/Genie toggles (spends money)

`frontend/src/Capture.tsx:70-71` always sends `omni=true` and `genie=true`. Carry the stage's choices in the QR URL
(`captureUrl` in `frontend/src/Stage.tsx`) or store them per session on the server. Done when: with both toggles off,
a phone run makes no Omni or Genie call (check `logs/`).

### C3. Short videos reset the synced playback

`SyncedPanels` in `frontend/src/Stage.tsx:450` uses the first video as the clock (`master`, line 458). A short
reality clip (an upload under 8 s) loops and drags the 5 s predictions back to the start. Drive the panels from a
5 s timeline or the longest video. Done when: a 4 s upload plays all predictions to 5 s.

### C4. Gravity scale can be wildly wrong

`backend/foresight/scale.py`, `choose_scale`: when the size-based and gravity-based scales disagree by over 30% with
6+ free-fall points, it takes the gravity one. In the wider eval's scene cache this produced 9 px/m against 4,448
from size (480x, `ball-and-block-fall`, right camera) and 17x on `marble-run-x` (center). Only accept a gravity scale
from a clean free-fall segment (small parabola residual, after the release cut in `track.clean_track`) and within a
plausible factor of the size scale; otherwise keep the size scale and record why. Done when: no scene in
`eval/cache/wide_*/work/foresight/*/scene.json`, rebuilt, ends more than 2x from its size scale without a stated
reason, and the offline score is not worse.

### C5. Multi-object fits share one start time

`backend/physics/fit.py:442` sets `t_start_s` to the earliest timestamp over all objects, so an object first seen
later (a second ball leaving its pipe) is extrapolated backwards and can collide before it exists. Give each body
its own start (hold it out of the world, or kinematic, until its first observation). Done when: a unit test with
two objects starting at different times passes, and two-balls-pass / balls-collide / ball-train are not worse
offline.

### C6. Replace Carbon repairs with a local multi-start fit

In both evals no repaired run ever ended passing Decision 1 (0 of 19, from the `repairs_used` and
`decision1_confidences` fields in `eval/results.json` and `eval/results_wide.json`), and a repair costs 45 s to
minutes. Replace the repair step (`backend/physics/__init__.py`, loop at line 87) with a local multi-start:
refit from several motion models (roll, slide, fall, bounce) and initial guesses, keep the best by fit error and the
verification check. Keep Carbon behind a flag if useful for a written explanation. Done when: the check-and-repair
loop takes seconds, offline scores are not worse, and the README's "How it works" describes it.

### C7. Packaging and CI

Add `pyproject.toml` (the dependencies in `docs/developer-guide.md`) and a GitHub Actions workflow running the
physics tests, `tsc --noEmit` and `vite build` for the frontend. CI must never call Gemini or need secrets.

### Then Option A: a physics auditor for video and world models

The finding to build on: the Physics-IQ pixel score ranks a straight line first (a still picture matches a scene
that has come to rest), and endpoint error, as measured now, cannot penalize a model that drops the object (its
track just stops). An auditor fixes both and is the most publishable direction.

- `foresight audit` (CLI and Python API): input a 3 s conditioning clip, any model's 5 s continuation, and optionally
  the real 5 s. Output JSON plus an HTML report with: object permanence (does the object vanish, duplicate or change
  size), position error at fixed times (counting a vanished object as an error, not a stopped track), and physical
  plausibility (speed against the free-fall bound, as in `physics.implausible`), next to the Physics-IQ score.
- Reuse what exists: tracking (`foresight/track.py`, `pipeline.track_in_video`), depth layers for occlusion
  (`physics/layers.py`: an object behind something is hidden, not vanished), the energy check, the scorer.
- Run it over the saved predictions of every system in both evals (`eval/preds/`, `eval/preds_wide/`, gitignored,
  on the original machine) and report where it and the pixel score disagree.
- Interactive what-if on the stage: in the reveal, change the start velocity, friction or bounce (or drag the start
  point) and re-simulate. The 2D world runs a 5 s rollout fast; serve it from an API endpoint and draw the path over
  the frame (`frontend/src/Insights.tsx`, `PathPlot`).

## Pitfalls already hit

- The shell is zsh: `$sha:r` is read as a modifier. Write `${sha}:refs/heads/main`, or use a bash script.
- Masks: the scorer regenerates a prediction's masks only when the video is newer (`phase0/score_subset.py`). If you
  rewrite predictions another way, delete their masks.
- CSRT latches onto background or onto whatever hides the object; real and predicted tracks end at loss, the frame
  edge, or an object in front.
- Gemini's scene readings vary run to run by about a point of Physics-IQ score; a single rerun cannot show a smaller
  improvement. Compare on the same cached readings (`--perception DIR`).
- Hydrogen's prompt examples must not name objects from the eval clips (that would tune perception to the test set).
- Data files (`data/`, Physics-IQ) are never committed; credit Physics-IQ (code Apache-2.0, materials CC BY 4.0).
