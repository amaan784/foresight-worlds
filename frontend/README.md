# Foresight frontend

Vite + React + Tailwind. The stage (laptop or projector) and the phone capture page.

```bash
npm install
npm run dev          # https://<lan-ip>:5173, phones scan the QR code on the stage
npm run dev --http   # second server on :5174 for a laptop-only browser
```

The backend must be running on port 8000 (`uvicorn foresight.api:app --app-dir backend --port 8000` from the repo root);
Vite proxies `/api` and `/outputs` to it. The phone accepts the self-signed certificate once.

- `src/Stage.tsx`: idle (QR code, Physics-IQ clips), live progress from the SSE stream, and the reveal with three synced videos.
- `src/Capture.tsx`: records 8 s with the rear camera and uploads it; the first 3 s are what Foresight sees.
- `src/api.ts`: the API client and the event reducer (CONTRACT.md §4).
