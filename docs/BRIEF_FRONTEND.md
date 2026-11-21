# Brief — Frontend agent (`frontend/`)

You are building the demo UI for **Foresight** at the Gemini 4 NYC Hackathon (submissions due 5:00 PM ET; the user records a 1-minute demo video at **4:30**). Three other agents work in this repo right now: Claude Code (`backend/foresight/`, the HTTP API you call), a physics agent (`backend/physics/`), and an eval agent (`eval/`). **Edit only `frontend/`** and your own section of `PROGRESS.md`.

First: `git pull --rebase`, then read `AGENTS.md`, `CLAUDE.md`, `CONTRACT.md` (§4 is your API), `phase0/REPORT.md`.

## What the demo must show (in this order)
1. A phone (QR code on the laptop) records ~8 s of a side-view scene: a ball rolling / falling on a table.
2. The laptop shows **only the first 3 s** (looping) while the backend streams progress live: Hydrogen's scene parse, tracking, scale, physics fit, Decision 1 confidences, repairs.
3. Reveal: **three synced panels** — **Reality** (the hidden 5 s) | **Foresight** | **Omni** (Gemini's video model continuing the same 3 s) — with the **endpoint error** of Foresight and Omni in big numbers.
Big, simple, legible: it will be screen-recorded and watched on a projector.

## Stack and setup
- Vite + React + TypeScript + Tailwind CSS. `npm create vite@latest frontend -- --template react-ts`, then Tailwind.
- HTTPS dev server (phones require HTTPS for `getUserMedia`): `@vitejs/plugin-basic-ssl`, `server: { host: "0.0.0.0", port: 5173, https: true }`. The phone must accept the self-signed cert once; put a one-line hint on the capture page.
- Proxy in `vite.config.ts`: `/api` and `/outputs` → `http://localhost:8000` (so the phone only talks to the frontend origin; no CORS, no mixed content).
- Backend (Claude's): `source .venv/bin/activate && uvicorn foresight.api:app --app-dir backend --host 0.0.0.0 --port 8000` from the repo root. Expected: `/api/info`, `/api/samples`, `POST /api/predict` with `sample_id` by ~3:50; phone-upload path by ~4:05. Until then build against the types in CONTRACT.md §4.
- `frontend/.gitignore`: `node_modules/`, `dist/`. Commit `package-lock.json`.
- QR: `qrcode.react`. No other heavy deps needed.

## Routes
### `/` — Stage (laptop / projector)
- On load: `GET /api/info` → `lan_ip`. Generate `session = crypto.randomUUID().slice(0, 8)`.
- Idle screen: big title "Foresight", one-line pitch, QR code for `https://<lan_ip>:5173/capture?session=<session>` (use the current page's port), and a **"Use a benchmark clip"** picker from `GET /api/samples` (thumbnails). Picking one → `POST /api/predict` (FormData: `sample_id`, `session`, `omni=true`).
- Poll `GET /api/jobs?session=<session>` every 1 s; when a job appears, open `new EventSource(events_url)` and switch to the Job view. (Also allow `/?job=<id>` to open a job directly.)

### Job view (same page)
- Left/top, large: `conditioning.mp4` (from the `ingest` event's `conditioning_url`), muted, looping, label **"What the model sees: 3 seconds"**.
- Right/below: a live **stage timeline** (ingest → perceive → track → scale → fit → verify → (repair) → render; Omni as a parallel lane) driven by `stage` events with elapsed seconds.
- Cards that fill in as events arrive (only show values that came from events):
  - `perception`: model name + latency, each object (name, shape, est. size in cm), motion type.
  - `tracking`: "tracked N of M frames".
  - `scale`: px per metre (+ gravity check if present).
  - `fit`: motion model, fit error (px), attempt number; update on each `fit` event.
  - `verify`: Decision 1 confidences as horizontal bars (0–1) with a marker at the 0.7 pass threshold; answers as text.
  - `repair`: "Carbon proposed: <motion_model> — <reason>".
  - `omni`: status chip ("generating… ~40 s", done, error, skipped).
- `error` event → a red panel with stage + message and a "Try again" button (back to idle). Never leave a spinner forever: if no event for 90 s, show "backend not responding".

### Reveal (after `result`)
- Three equal panels, labels in large type: **Reality** (`urls.reality`), **Foresight (physics)** (`urls.foresight`), **Omni (video model)** (`urls.omni`). If `urls.omni` is null while the `omni` lane is still running, show a placeholder and swap the video in when the `omni` `done` event arrives (re-fetch `GET /api/jobs/{id}`).
- **One master clock**: a single play/pause, a scrubber, loop, and a 0.5× toggle drive all three `<video>` elements (keep them within ~1 frame: on every `requestAnimationFrame`, if `|v.currentTime - master| > 0.05` set it). All videos are exactly 5.0 s, 24 FPS, H.264.
- Under Foresight and Omni: **endpoint error** from `metrics.endpoint_error_cm` (e.g. "ends 3.2 cm from reality"), huge font; colour the smaller error green. If null, show "—" with a small "(no reality clip)" note. Optionally show `endpoint_error_px` in small text.
- Footer strip: total latency (`metrics.latency_s_total`), Decision 1 confidences, repairs used. Optional toggle to swap the Foresight panel to `urls.overlay` (trajectory overlay).

### `/capture?session=<s>` — Phone
- `getUserMedia({ video: { facingMode: "environment", width: { ideal: 1920 }, height: { ideal: 1080 }, frameRate: { ideal: 30 } }, audio: false })`. Full-screen preview, "Rotate to landscape" hint if portrait.
- Tips (3 short lines): phone on a stand / held still, side view, roll or drop ONE ball across the frame.
- Big record button → 3-2-1 countdown → record **8.0 s** with `MediaRecorder` (pick `mimeType` via `MediaRecorder.isTypeSupported`: `video/mp4` (iOS Safari), else `video/webm;codecs=vp9`, else `video/webm`), progress ring, auto-stop.
- Upload `FormData { video: Blob (filename "capture.mp4" or ".webm"), session, cond_seconds: "3", omni: "true" }` to `POST /api/predict` with an upload progress bar (XMLHttpRequest for progress). Then show "Sent — watch the big screen" + a "Record again" button. On error show the message and a retry button.

## Design
Projector-first: dark background, high contrast, headings ≥ 48 px, numbers ≥ 64 px, labels ≥ 20 px, generous spacing, no dense tables, no tiny text. Must also work at phone width (capture page) with no horizontal scroll. Use one accent colour for Foresight, a neutral for Reality, another for Omni, consistently.

## Rules
- No mock data, fixtures, or hardcoded numbers in the shipped app. Everything displayed comes from the API. (A dev-only fixture is not allowed in the committed build.)
- No Streamlit. Commit as the configured git user, NO Co-Authored-By / "Generated with" lines, 150–300 lines per commit, `git pull --rebase` + secret scan before each push (see AGENTS.md).
- If the API contract blocks you, tell the user exactly what you need; do not edit `backend/`.
- Update the `## Frontend` section of `PROGRESS.md` (done / next / blockers) after each milestone.

## Done means
`npm run dev` serves `https://<lan_ip>:5173`; picking a sample on the laptop runs a real job end to end and reveals three synced panels with endpoint errors; the phone capture page records 8 s and the laptop picks the job up automatically.
