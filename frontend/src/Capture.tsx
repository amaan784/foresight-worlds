import { useEffect, useRef, useState } from "react";
import { apiUrl, getInfo, getSessionPrefs } from "./api";

const RECORD_S = 8;
type Phase = "idle" | "countdown" | "recording" | "uploading" | "sent" | "error";

function parseBoolParam(v: string | null): boolean | null {
  if (v === null || v === "") return null;
  const s = v.trim().toLowerCase();
  if (s === "1" || s === "true" || s === "yes" || s === "on") return true;
  if (s === "0" || s === "false" || s === "no" || s === "off") return false;
  return null;
}

function pickMime(): string {
  for (const m of ["video/mp4", "video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"]) {
    if (typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported(m)) return m;
  }
  return "";
}

export default function Capture() {
  const params = new URLSearchParams(location.search);
  const session = params.get("session") ?? "";
  const urlOmni = parseBoolParam(params.get("omni"));
  const urlGenie = parseBoolParam(params.get("genie"));
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [count, setCount] = useState(3);
  const [elapsed, setElapsed] = useState(0);
  const [progress, setProgress] = useState(0);
  const [msg, setMsg] = useState<string | null>(null);
  const [portrait, setPortrait] = useState(false);
  const [physics, setPhysics] = useState(true);
  const [askOmni, setAskOmni] = useState<boolean>(() => urlOmni ?? false);
  const [askGenie, setAskGenie] = useState<boolean>(() => urlGenie ?? false);
  // State, not just the ref: the record button must re-render as enabled once the camera stream arrives.
  const [camReady, setCamReady] = useState(false);

  useEffect(() => {
    getInfo()
      .then((i) => {
        setPhysics(i.physics_available);
        if (urlOmni === null) setAskOmni(Boolean(i.omni_enabled));
        if (urlGenie === null) setAskGenie(Boolean(i.genie_enabled));
      })
      .catch(() => {});
    if (session) {
      getSessionPrefs(session)
        .then((p) => {
          if (!p) return;
          setAskOmni(p.omni);
          setAskGenie(p.genie);
          if (p.dynamics === "physics" || p.dynamics === "const_velocity") setPhysics(p.dynamics === "physics");
        })
        .catch(() => {});
    }
    const onRot = () => setPortrait(window.innerHeight > window.innerWidth);
    onRot();
    window.addEventListener("resize", onRot);
    navigator.mediaDevices
      ?.getUserMedia({ video: { facingMode: "environment", width: { ideal: 1920 }, height: { ideal: 1080 }, frameRate: { ideal: 30 } }, audio: false })
      .then((s) => { streamRef.current = s; if (videoRef.current) videoRef.current.srcObject = s; setCamReady(true); })
      .catch((e) => { setPhase("error"); setMsg(`Camera unavailable: ${e.message}. Open this page over https and allow camera access.`); });
    return () => { window.removeEventListener("resize", onRot); streamRef.current?.getTracks().forEach((t) => t.stop()); };
  }, [session, urlOmni, urlGenie]);

  async function record() {
    const stream = streamRef.current;
    if (!stream) return;
    setMsg(null);
    setPhase("countdown");
    for (let i = 3; i > 0; i--) { setCount(i); await new Promise((r) => setTimeout(r, 1000)); }
    const mime = pickMime();
    const rec = new MediaRecorder(stream, mime ? { mimeType: mime, videoBitsPerSecond: 8_000_000 } : undefined);
    const chunks: Blob[] = [];
    rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    const done = new Promise<void>((r) => (rec.onstop = () => r()));
    rec.start(250);
    setPhase("recording");
    const t0 = performance.now();
    const tick = setInterval(() => setElapsed((performance.now() - t0) / 1000), 50);
    await new Promise((r) => setTimeout(r, RECORD_S * 1000));
    rec.stop();
    clearInterval(tick);
    await done;
    const type = rec.mimeType || mime || "video/webm";
    await upload(new Blob(chunks, { type }), type.includes("mp4") ? "capture.mp4" : "capture.webm");
  }

  async function upload(blob: Blob, name: string) {
    setPhase("uploading");
    setProgress(0);
    let omniChoice = askOmni;
    let genieChoice = askGenie;
    let dynamicsChoice = physics ? "physics" : "const_velocity";
    if (session) {
      const latest = await getSessionPrefs(session);
      if (latest) {
        omniChoice = latest.omni;
        genieChoice = latest.genie;
        setAskOmni(latest.omni);
        setAskGenie(latest.genie);
        if (latest.dynamics === "physics" || latest.dynamics === "const_velocity") {
          dynamicsChoice = latest.dynamics;
        }
      }
    }
    const f = new FormData();
    f.set("video", blob, name);
    f.set("session", session);
    f.set("source", "capture");
    f.set("cond_seconds", "3");
    f.set("omni", String(omniChoice));
    f.set("genie", String(genieChoice));
    f.set("dynamics", dynamicsChoice);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", apiUrl("/api/predict"));
    xhr.upload.onprogress = (e) => e.lengthComputable && setProgress(e.loaded / e.total);
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) setPhase("sent");
      else { setPhase("error"); setMsg(`Upload refused (${xhr.status}): ${xhr.responseText.slice(0, 200)}`); }
    };
    xhr.onerror = () => { setPhase("error"); setMsg("Upload failed: could not reach the Foresight server. Check your connection."); };
    xhr.send(f);
  }

  const ring = phase === "recording" ? Math.min(1, elapsed / RECORD_S) : phase === "uploading" ? progress : 0;
  const C = 2 * Math.PI * 44;

  return (
    <div className="fixed inset-0 bg-black text-[#e9e9e6] overflow-hidden">
      <video ref={videoRef} autoPlay muted playsInline className="absolute inset-0 h-full w-full object-cover" />
      {/* 3 s / 5 s guide along the bottom while recording */}
      {phase === "recording" && (
        <div className="absolute left-4 right-28 bottom-6 h-2 rounded-full bg-white/25 overflow-hidden" aria-hidden>
          <div className="h-full bg-white" style={{ width: `${(elapsed / RECORD_S) * 100}%` }} />
          <div className="absolute top-0 h-full w-[3px] bg-[#2350d8]" style={{ left: `${(3 / RECORD_S) * 100}%` }} />
        </div>
      )}
      <div className="absolute top-0 inset-x-0 p-5 bg-gradient-to-b from-black/70 to-transparent">
        <div className="display text-2xl font-extrabold">Foresight</div>
        {phase === "idle" && (
          <p className="mt-1 text-base max-w-md opacity-90">
            Hold still or prop the phone up, side view. Tap record, then roll or drop one ball across the frame.
            The first 3 seconds are what Foresight sees; it predicts the other 5.
          </p>
        )}
        {phase === "recording" && <p className="mt-1 font-mono">{elapsed < 3 ? "seen: " : "hidden: "}{elapsed.toFixed(1)} of {RECORD_S} s</p>}
        {phase === "uploading" && <p className="mt-1">Sending to the laptop… {Math.round(progress * 100)}%</p>}
        {phase === "sent" && <p className="mt-1 text-xl font-semibold">Sent. Watch the big screen.</p>}
        {msg && <p role="alert" className="mt-2 text-[#ffb4a8]">{msg}</p>}
      </div>
      {portrait && phase === "idle" && (
        <div className="absolute inset-x-0 top-1/2 -translate-y-1/2 text-center text-xl font-semibold">Turn the phone sideways</div>
      )}
      {phase === "countdown" && (
        <div className="absolute inset-0 grid place-items-center display text-[9rem] font-extrabold drop-shadow-lg" aria-live="assertive">{count}</div>
      )}
      <div className="absolute right-5 top-1/2 -translate-y-1/2">
        {(phase === "idle" || phase === "sent" || phase === "error" || phase === "recording" || phase === "uploading" || phase === "countdown") && (
          <button onClick={record} disabled={phase === "recording" || phase === "uploading" || phase === "countdown" || !camReady}
                  aria-label={phase === "sent" ? "Record again" : "Record 8 seconds"} className="relative grid place-items-center size-24">
            <svg viewBox="0 0 100 100" className="absolute inset-0 -rotate-90">
              <circle cx="50" cy="50" r="44" fill="none" stroke="rgba(255,255,255,.35)" strokeWidth="6" />
              <circle cx="50" cy="50" r="44" fill="none" stroke="#fff" strokeWidth="6" strokeDasharray={C} strokeDashoffset={C * (1 - ring)} />
            </svg>
            <span className={`block rounded-full bg-[#e5392e] transition-all ${phase === "recording" ? "size-9 rounded-md" : "size-16"}`} />
          </button>
        )}
        {phase === "sent" && <div className="mt-2 text-center text-sm">Record again</div>}
      </div>
    </div>
  );
}
