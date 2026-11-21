# Evaluation method

Every number in the README comes from `eval/run_eval.py` and the files it writes: [../eval/results.json](../eval/results.json),
[../eval/results_table.md](../eval/results_table.md) and [../eval/chart.png](../eval/chart.png). Nothing is typed in by hand.

## Clips

The 15 rigid-body scenes of Physics-IQ (one object, or two balls, rolling, falling or bouncing; no fluids, magnets,
light, cloth or domino chains), from the **center** camera, which is the side view. Each clip is 3 s of
conditioning video and 5 s of ground truth at 24 FPS, filmed twice (take 1 and take 2) so the benchmark knows how
much real outcomes vary. The list is in `phase0/physiq_data.py`.

## Systems

| System | What it does |
|---|---|
| Foresight | perception, tracking, scale, physics fit with Decision 1 and Carbon, render |
| Constant velocity | the same perception, tracking and renderer, but the object keeps its last velocity: no gravity, no surfaces, no friction |
| Gemini Omni 1.2 Flash | continues the 3 s clip as video, prompted with the benchmark's own scene description plus "Continue the scene." |
| Last frame held | the last conditioning frame for 5 s |

A system that fails on a clip is scored with the held last frame for that clip, and the failure is listed in
[../eval/FAILURES.md](../eval/FAILURES.md).

## Score

The official Physics-IQ code (`third_party/physics-IQ-benchmark`, imported by `phase0/score_subset.py`) compares
each prediction with both real takes: where motion happens (spatial IoU), where and when (spatiotemporal IoU),
how much (weighted spatial IoU) and pixel error (MSE), each divided by the difference between the two real takes.
The reported score is the Verified formula's mean over clips, in percent; higher is better. `run_physics_iq.py`
only accepts all 198 videos and all three cameras, so the subset is scored by calling the same functions on one
camera.

**Endpoint error** is simpler: the distance, in cm, between where a system puts the object and where the real
object is, at the last frame the real object is visible (tracked with CSRT), converted with the scene's pixels
per metre.

## What the score rewards

Holding the last frame scores well because the MSE term rewards a video that does not change; a system has to
beat that, not only Omni. Ground truth is Physics-IQ Verified when its gated files are available and the
original Physics-IQ files otherwise; `results.json` says which (`gt_source`).

## Clean run

Development runs reuse Gemini outputs from `eval/cache/`. The reported numbers come from one run with `--clean`:
fresh Gemini calls, perception computed once per clip and shared by Foresight and constant velocity so only the
dynamics differ.
