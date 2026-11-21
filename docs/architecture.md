# Architecture

Foresight has three parts: the **backend** (`backend/foresight/`), which turns a 3 second clip into a scene a
physics engine can run; the **physics module** (`backend/physics/`), which fits and runs that scene for 5 seconds
and checks itself with Gemini; and the **stage** (`frontend/`), which records on a phone and shows the result next
to reality and next to Gemini Omni. The interfaces between them are in [../CONTRACT.md](../CONTRACT.md).

<p align="center"><img src="img/how-it-works.svg" alt="The pipeline: 3 s of video, read the scene with Gemini 4 Hydrogen, track with CSRT, measure pixels per metre, fit physics, check with Decision 1 and repair with Carbon, render 5 s; Omni continues the same clip; both are compared with reality" width="100%"></p>

## From a clip to a scene (`backend/foresight/`)

1. **Ingest** (`api.py`, `video.py`): a phone upload (webm, mp4 or mov) becomes a constant-frame-rate 3 s
   conditioning clip at 24 FPS, the same 3 s at 30 FPS for fitting, and the hidden next 5 s. Physics-IQ clips
   come split already. The 24 FPS Physics-IQ files drop every fifth frame of the 30 FPS originals, so timing for
   the fit always comes from the 30 FPS clip.
2. **Read the scene** (`perceive.py`): Gemini 4 Hydrogen gets the clip with agentic video processing plus its
   first and last frames, at thinking level low, and returns a structured scene: each object as a circle or a box
   with its real size in cm, the surfaces it rolls on as line segments, and the kind of motion.
3. **Track** (`track.py`): OpenCV CSRT follows each moving object forward from the first frame and back from the
   last. CSRT rarely admits it has lost the object, so a frame only counts when the tracked patch still looks
   like the object.
4. **Measure** (`scale.py`): pixels per metre come from the object's size in pixels and its real size. For falls,
   a parabola fitted to the track gives a second estimate from gravity.
5. **Assemble** (`scene.py`): the last frame with the object removed (filled from the first frame when the
   object was elsewhere, inpainted otherwise), a sprite of the object, keyframes for checking, and
   `scene.json` (CONTRACT.md §1).

## Physics (`backend/physics/`)

A 2D pymunk world is built from `scene.json`: static segments for the surfaces, dynamic circles and boxes for
the objects, gravity fixed at 9.81 m/s². `scipy.optimize.least_squares` fits the start position, velocity and
the material parameters the track can support; the rest keep priors. Gemini Decision 1 looks at keyframes with
the fitted outline drawn on them and answers whether it sits on the object, and which motion it is (roll,
slide, fall, bounce or other), each with a confidence. Under 0.7, Gemini 4 Carbon (thinking level medium) looks
at a plot of the track and the keyframes and proposes another motion model and starting guesses; the fit runs
again, at most twice. The world then runs for exactly 5.000 s and is rendered onto the cleaned frame.

## The baseline (`backend/foresight/omni.py`)

Gemini Omni 1.2 Flash gets the same 3 s and is asked to continue the scene (video extension). It returns the
input plus a continuation; the 5 s right after the input are kept.

## Serving (`backend/foresight/api.py`)

FastAPI. `POST /api/predict` starts a job and returns at once; `GET /api/jobs/{id}/events` streams every stage as
server-sent events (perception, tracking, scale, fit, Decision 1 confidences, repairs, render, Omni), replaying
past events to late listeners. The Omni lane runs in parallel. Every Gemini call goes through
`foresight/gemini.py`: retry with backoff on 503, a switch from Hydrogen to Neon if 503s continue, a hard stop on
429 (the project's spend cap), and a JSON line per call in `logs/` with the media stripped.

## The stage (`frontend/`)

Vite, React and Tailwind. The laptop shows a QR code; the phone records 8 s with the rear camera and uploads it;
the laptop shows only the first 3 s while the steps fill in, then plays Reality, Foresight and Omni in sync with
the endpoint error under each.
