// Client for the Foresight API (CONTRACT.md §4). Everything shown in the UI comes from these calls.
import { useEffect, useReducer, useRef } from "react";

export const API_BASE = ((import.meta.env.VITE_API_BASE as string | undefined) ?? "").replace(/\/+$/, "");

export function apiUrl(path: string): string;
export function apiUrl(path: null | undefined): null;
export function apiUrl(path: string | null | undefined): string | null;
export function apiUrl(path: string | null | undefined): string | null {
  if (!path) return null;
  if (/^https?:\/\//i.test(path)) return path;
  return `${API_BASE}${path.startsWith("/") ? path : `/${path}`}`;
}

export type Info = {
  lan_ip: string;
  physics_available: boolean;
  omni_enabled: boolean;
  genie_enabled?: boolean;
  verify_mode?: string;
  models: Record<string, string>;
};
export type Sample = { id: string; title: string; thumb_url: string };

export type Perception = {
  model: string;
  latency_s: number;
  cached?: boolean;
  objects: { name: string; shape: string; est_real_size_cm: number; moving: boolean }[];
  motion_type: string;
  motion_start_s: number;
};
export type Result = {
  job_id: string;
  status: string;
  dynamics: "physics" | "const_velocity";
  urls: { conditioning: string; reality: string | null; foresight: string; omni: string | null; genie?: string | null; overlay: string | null;
          trajectory_json?: string; trajectory_csv?: string };
  title?: string;
  source?: string;
  frame_url?: string;
  paths?: Paths;
  understanding?: Understanding;
  metrics: {
    latency_s_total: number;
    latency_s_by_stage: Record<string, number>;
    fit_error_px: number | null;
    decision1_confidences: Record<string, number> | null;
    repairs_used: number | null;
    endpoint_error_cm: { foresight: number | null; omni?: number | null; genie?: number | null };
    endpoint_error_px: { foresight: number | null; omni?: number | null; genie?: number | null };
    dynamics_used?: string;
    fallback_reason?: string | null;
  };
  scene: { objects: { name: string; shape: string; est_real_size_cm: number }[]; motion_type: string; px_per_m: number };
  models?: Record<string, string>;
  physics?: { motion_model?: string } | null;
  omni?: { status: string; latency_s: number | null; error: string | null; message?: string | null };
  genie?: { status: string; latency_s: number | null; error: string | null; message?: string | null };
};

/** Points are [t_s, x, y] in the clip's original pixels. */
export type Paths = { orig_size: [number, number]; seen: number[][]; predicted: number[][]; real: number[][] };

export type Understanding = {
  objects: { id: string; name: string; shape: string; size_cm: number | null; mass_g: number | null; role: string;
             fit: { speed_m_s: number | null; friction: number | null; bounce: number | null; damping_per_s: number | null } | null }[];
  surfaces: { name: string; kind: string; material: string }[];
  depth_notes: string[];
  px_per_m: number;
  scale_source?: string;
  motion_model?: string | null;
  checks: { answers: Record<string, string>; confidences: Record<string, number> };
  repairs_used?: number | null;
  dynamics_used?: string | null;
  fallback_reason?: string | null;
};

export type RecentRun = { job_id: string; created: string; title: string; poster: string | null; dynamics: string | null;
                          endpoint_error_cm: { foresight: number | null; omni: number | null; genie: number | null } };

export type Lane = { status: string; url: string | null; latency_s: number | null; error: string | null; message?: string | null; endpoint_error_cm?: number | null; t?: number };

export type JobState = {
  id: string;
  stages: Record<string, { status: string; message: string; t: number }>;
  ingest?: { conditioning_url: string; cond_frames: number; reality_frames: number; has_reality: boolean; width: number; height: number };
  perception?: Perception;
  tracking: { object: string; frames_ok: number; frames_total: number; points: number[][] }[];
  scale?: { px_per_m: number; source: string; gravity_check: { g_est_m_s2: number } | null };
  fits: { attempt: number; motion_model: string; fit_error_px: number; n_points: number; params?: Record<string, unknown> }[];
  verifies: { model?: string; mode?: string; attempt: number; confidences: Record<string, number>; answers?: Record<string, string>; passed: boolean }[];
  repairs: { model?: string; attempt: number; proposal: { motion_model?: string; reason?: string } }[];
  render?: { url: string | null; frames: number; duration_s: number };
  omni?: Lane;
  omniStartT?: number;
  genie?: Lane;
  fallback?: { to: string; reason: string };
  repairSkipped?: { reason: string };
  result?: Result;
  error?: { stage: string; message: string };
  ended: boolean;
  lastT: number;
};

const empty = (id: string): JobState => ({ id, stages: {}, tracking: [], fits: [], verifies: [], repairs: [], ended: false, lastT: 0 });

type Action = { type: "reset"; id: string } | { type: "event"; name: string; data: any } | { type: "result"; data: Result };

function reduce(s: JobState, a: Action): JobState {
  if (a.type === "reset") return empty(a.id);
  if (a.type === "result") return { ...s, result: a.data };
  const d = a.data;
  const next = { ...s, lastT: typeof d?.t === "number" ? d.t : s.lastT };
  switch (a.name) {
    case "stage": return { ...next, stages: { ...s.stages, [d.stage]: d } };
    case "ingest": return { ...next, ingest: d };
    case "perception": return { ...next, perception: d };
    case "tracking": return { ...next, tracking: [...s.tracking.filter((x) => x.object !== d.object), d] };
    case "scale": return { ...next, scale: d };
    case "fit": return { ...next, fits: [...s.fits, d] };
    case "verify": return { ...next, verifies: [...s.verifies, d] };
    case "repair": return { ...next, repairs: [...s.repairs, d] };
    case "render": return d.url ? { ...next, render: d } : next;
    case "omni": return { ...next, omni: d, omniStartT: d.status === "start" ? d.t : s.omniStartT };
    case "genie": return { ...next, genie: d };
    case "fallback": return { ...next, fallback: d };
    case "repair_skipped": return { ...next, repairSkipped: d };
    case "result": return { ...next, result: d };
    case "error": return { ...next, error: d, ended: true };
    case "end": return { ...next, ended: true };
    default: return next;
  }
}

const EVENTS = ["stage", "ingest", "perception", "tracking", "scale", "fit", "verify", "repair", "repair_skipped", "fallback",
  "render", "omni", "genie", "result", "end"];

/** Subscribe to a job's SSE stream. The server replays every past event on (re)connect, so state resets on open. */
export function useJob(jobId: string | null): JobState | null {
  const [state, dispatch] = useReducer(reduce, empty(jobId ?? ""));
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!jobId) return;
    dispatch({ type: "reset", id: jobId });
    const es = new EventSource(apiUrl(`/api/jobs/${jobId}/events`));
    esRef.current = es;
    es.onopen = () => dispatch({ type: "reset", id: jobId });
    const refresh = () =>
      fetch(apiUrl(`/api/jobs/${jobId}`)).then((r) => r.json()).then((r) => r.urls && dispatch({ type: "result", data: r })).catch(() => {});
    for (const name of EVENTS) {
      es.addEventListener(name, (e) => {
        const data = JSON.parse((e as MessageEvent).data);
        dispatch({ type: "event", name, data });
        if ((name === "omni" || name === "genie") && data.status !== "start") setTimeout(refresh, 300);
        if (name === "end") { es.close(); refresh(); }
      });
    }
    // "error" is both our event name and EventSource's connection-error event: only ours carries data.
    es.addEventListener("error", (e) => {
      const me = e as MessageEvent;
      if (me.data) { dispatch({ type: "event", name: "error", data: JSON.parse(me.data) }); es.close(); }
    });
    return () => es.close();
  }, [jobId]);

  return jobId ? state : null;
}

export async function getInfo(): Promise<Info> {
  const r = await fetch(apiUrl("/api/info"));
  if (!r.ok) {
    const hint = API_BASE ? `Check backend at ${API_BASE}.` : "Start the backend with uvicorn on port 8000 (or set VITE_API_BASE).";
    throw new Error(`The backend did not answer (${r.status}). ${hint}`);
  }
  return r.json();
}

export async function getSamples(): Promise<Sample[]> {
  const r = await fetch(apiUrl("/api/samples"));
  return r.ok ? r.json() : [];
}

export async function listSessionJobs(session: string): Promise<{ job_id: string; created: string; status: string }[]> {
  return fetch(apiUrl(`/api/jobs?session=${encodeURIComponent(session)}`))
    .then((x) => (x.ok ? x.json() : []))
    .catch(() => []);
}

export async function startJob(form: FormData): Promise<{ job_id: string }> {
  const r = await fetch(apiUrl("/api/predict"), { method: "POST", body: form });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail ?? `Request failed (${r.status})`);
  return r.json();
}

export async function cancelJob(id: string): Promise<void> {
  await fetch(apiUrl(`/api/jobs/${id}/cancel`), { method: "POST" }).catch(() => {});
}

export function fmtCm(v: number | null | undefined): string {
  if (v === null || v === undefined) return "—";
  return v >= 100 ? `${(v / 100).toFixed(2)} m` : `${v.toFixed(1)} cm`;
}

export async function getRecent(limit = 8): Promise<RecentRun[]> {
  return fetch(apiUrl(`/api/recent?limit=${limit}`)).then((r) => (r.ok ? r.json() : [])).catch(() => []);
}

export async function getEval(which: "main" | "wide"): Promise<any | null> {
  return fetch(apiUrl(which === "main" ? "/api/eval" : "/api/eval/wide")).then((r) => (r.ok ? r.json() : null)).catch(() => null);
}
