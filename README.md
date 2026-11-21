<p align="center"><img src="docs/img/banner.png" alt="Foresight: watch 3 seconds, predict the next 5. A film strip of the first 3 seconds of a Physics-IQ clip: a grey tennis ball rolls out of a pipe across a table." width="100%"></p>

<p align="center">
  <b>For robotics teams: watch 3 seconds of real video, simulate the physics, and predict the next 5, where video models get physics wrong.</b><br>
  <sub>Built during the Gemini 4 NYC Hackathon · November 21, 2026</sub>
</p>

<p align="center">
  <a href="#try-it-in-30-seconds"><b>Try it in 30 seconds</b></a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#results">Results</a> ·
  <a href="docs/developer-guide.md">Developer guide</a> ·
  <a href="phase0/REPORT.md">Phase 0 report</a>
</p>

A robot that pushes, drops or rolls things needs to know where they will be in a few seconds. Video models can
show it, but they make up physics: in our first test, Gemini Omni continued a clip of a rolling ball by making
the ball vanish and a new one roll out of the pipe. Foresight does not generate pixels. It reads the scene with
Gemini 4 Hydrogen, tracks the object, fits a 2D physics world to what it saw, lets Gemini Decision 1 check the
fit and Gemini 4 Carbon repair it, and then runs the world forward for 5 seconds. You get a video and the
trajectory behind it.

<p align="center"><img src="docs/img/compare.png" alt="The same 3 seconds, continued three ways: reality, where the ball rolls right and leaves the table within a second; and Gemini Omni, where the ball disappears and a new one emerges from the pipe" width="100%"></p>

## Try it in 30 seconds

With the [setup](docs/developer-guide.md#setup-about-2-minutes) done and a Gemini API key in `.env`:

```bash
PYTHONPATH=backend uvicorn foresight.api:app --app-dir backend --host 0.0.0.0 --port 8000
npm --prefix frontend run dev
```

Open `https://<your-lan-ip>:5173`. Pick a Physics-IQ clip, or scan the QR code with a phone, prop it up side-on
and roll a ball through the frame. The laptop shows only the first 3 seconds while each step reports in, then
plays reality, Foresight and Omni side by side with how far each prediction ends up from the real object.

## How it works

1. **Read the scene.** Gemini 4 Hydrogen watches the clip (agentic video) and returns, as structured output, each
   object as a circle or a box with its real size, the surfaces it moves on, and the kind of motion.
2. **Track.** OpenCV CSRT follows the object frame by frame, with real timestamps from the 30 FPS clip.
3. **Measure.** Pixels per metre come from the object's real size; falls are cross-checked against gravity.
4. **Fit physics.** A pymunk world is fitted to the tracked path with least squares: start position, velocity,
   and the friction and bounce the clip can reveal.
5. **Check and repair.** Gemini Decision 1 looks at the fitted outline on real frames and says whether it sits on
   the object and what the motion is, with a confidence. Under 0.7, Gemini 4 Carbon proposes another motion model
   and the fit runs again.
6. **Render.** The world runs for exactly 5.000 s and the object is drawn onto the frame it was cleaned from.

<p align="center"><img src="docs/img/how-it-works.svg" alt="The pipeline from a 3 second clip to a 5 second prediction, with Omni as the baseline and both compared with reality" width="100%"></p>

<sub>Details: [docs/architecture.md](docs/architecture.md). Interfaces between the parts: [CONTRACT.md](CONTRACT.md).</sub>

<!-- results: start (written by eval/readme_results.py from eval/results.json; edits here are lost) -->
## Results

Not run yet.
<!-- results: end -->

## Gemini models, each with a job

| Model | Its job |
|---|---|
| Gemini 4 Hydrogen (`gemini-4-hydrogen`) | Reads the clip: objects, sizes, surfaces, motion (agentic video, structured output, thinking level low). 12.5 s in [Phase 0](phase0/REPORT.md). |
| Gemini 4 Neon (`gemini-4-neon`) | Takes over scene reading if Hydrogen keeps returning 503. |
| Gemini Decision 1 (`gemini-decision-1`) | Checks the fit on real frames: does the outline sit on the object, which motion is it; confidences gate the repair. |
| Gemini 4 Carbon (`gemini-4-carbon`) | Repairs a fit Decision 1 doubts: proposes another motion model and starting guesses (thinking level medium). |
| Gemini Omni 1.2 Flash (`gemini-omni-1.2-flash`) | The baseline: continues the same 3 s as video. |
| Gemini Robotics-ER 2 (`gemini-robotics-er-2-preview`) | Tested as a tracker in Phase 0 and not used: 39 px median error against 20 px for CSRT. |

## Limitations, in plain words

- **Side view, rigid objects, still camera.** Foresight models balls and boxes moving in the image plane. A ball
  rolling toward the camera, a cloth or a liquid is out of scope.
- **Short motion.** In Physics-IQ the object often moves for only the last 0.2 to 0.6 s of the 3 s it is shown,
  so the fit leans on gravity and the object's known size.
- **Scale comes from Gemini's size estimate.** A wrong size (a 6.7 cm tennis ball called 10 cm) scales every
  distance by the same factor; falls are cross-checked against gravity, rolls are not.
- **A subset of Physics-IQ.** 15 of 66 scenes and one of three cameras, so the scores are not comparable with the
  public leaderboard.
- **The score likes stillness.** Holding the last frame scores well on Physics-IQ because the pixel-error term
  rewards a video that does not change. The table shows it so you can judge.
- **Omni is slow and paid.** About 40 s and about $1 a clip, so the stage asks before calling it.

## Links and docs

| | |
|---|---|
| **Architecture** | [docs/architecture.md](docs/architecture.md) |
| **Evaluation method** | [docs/evaluation.md](docs/evaluation.md), results in [eval/](eval/) |
| **Phase 0 checks** | [phase0/REPORT.md](phase0/REPORT.md): scoring a subset, Hydrogen perception, CSRT against Robotics-ER 2, Omni continuation |
| **Developer guide** | [docs/developer-guide.md](docs/developer-guide.md): setup, demo, eval |
| **Interfaces** | [CONTRACT.md](CONTRACT.md) |
| **Data and licenses** | [docs/data-and-licenses.md](docs/data-and-licenses.md) |
| **Progress and rules** | [PROGRESS.md](PROGRESS.md), [CLAUDE.md](CLAUDE.md), [AGENTS.md](AGENTS.md) |

## Data and credits

Evaluation clips and scoring code come from [Physics-IQ](https://github.com/google-deepmind/physics-IQ-benchmark)
(Motamed et al., Google DeepMind; code Apache-2.0, materials CC BY 4.0) and
[Physics-IQ Verified](https://huggingface.co/datasets/Anates-Labs-Research/Physics-IQ-Verified) (Anates Labs
Research). No dataset files are in this repository; the images above contain a few Physics-IQ frames, credited
in their captions.

Built during the Gemini 4 NYC Hackathon (November 21, 2026); all code was written during the event. MIT license,
see [LICENSE](LICENSE).
