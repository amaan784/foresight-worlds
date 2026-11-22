import { useEffect, useMemo, useRef, useState } from "react";
import { QRCodeSVG } from "qrcode.react";
import { apiUrl, cancelJob, fmtCm, getInfo, getRecent, getSamples, listSessionJobs, saveSessionPrefs, startJob, useJob, type Info, type JobState, type Lane, type RecentRun, type Sample } from "./api";
import { Downloads, PathPlot, Understood } from "./Insights";
import { advanceClock, DEFAULT_TIMELINE_S, syncAllVideos, syncVideo, timelineDuration } from "./sync";

const newSession = () => Math.random().toString(36).slice(2, 10);

export default function Stage() {
  const [info, setInfo] = useState<Info | null>(null);
  const [infoErr, setInfoErr] = useState<string | null>(null);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [session, setSession] = useState(newSession);
  const [jobId, setJobId] = useState<string | null>(() => new URLSearchParams(location.search).get("job"));
  const [askOmni, setAskOmni] = useState(false);
  const [askGenie, setAskGenie] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [startErr, setStartErr] = useState<string | null>(null);
  const [recent, setRecent] = useState<RecentRun[]>([]);
  const job = useJob(jobId);

  useEffect(() => {
    getInfo()
      .then((inf) => {
        setInfo(inf);
        setAskOmni(Boolean(inf.omni_enabled));
        setAskGenie(Boolean(inf.genie_enabled));
      })
      .catch((e) => setInfoErr(String(e.message ?? e)));
    getSamples().then(setSamples);
  }, []);
  useEffect(() => { if (!jobId) getRecent(8).then(setRecent); }, [jobId]);

  // A phone that scanned the QR code posts with this session; pick its job up.
  useEffect(() => {
    if (jobId) return;
    const since = new Date().toISOString();
    const t = setInterval(async () => {
      const r = await listSessionJobs(session);
      const fresh = r.find((j) => j.created >= since.slice(0, 19)) ?? r[0];
      if (fresh) setJobId(fresh.job_id);
    }, 1000);
    return () => clearInterval(t);
  }, [jobId, session]);

  const dynamics = info?.physics_available ? "physics" : "const_velocity";

  useEffect(() => {
    if (jobId) return;
    void saveSessionPrefs(session, { omni: askOmni, genie: askGenie, dynamics });
  }, [jobId, session, askOmni, askGenie, dynamics]);

  async function runSample(s: Sample) {
    setBusy(s.id);
    setStartErr(null);
    const f = new FormData();
    f.set("sample_id", s.id);
    f.set("session", session);
    f.set("omni", String(askOmni));
    f.set("genie", String(askGenie));
    f.set("dynamics", dynamics);
    try {
      open((await startJob(f)).job_id);
    } catch (e) {
      setStartErr(String((e as Error).message));
    } finally {
      setBusy(null);
    }
  }

  async function runUpload(file: File) {
    setBusy("upload");
    setStartErr(null);
    const f = new FormData();
    f.set("video", file, file.name);
    f.set("session", session);
    f.set("cond_seconds", "3");
    f.set("omni", String(askOmni));
    f.set("genie", String(askGenie));
    f.set("dynamics", dynamics);
    try {
      open((await startJob(f)).job_id);
    } catch (e) {
      setStartErr(String((e as Error).message));
    } finally {
      setBusy(null);
    }
  }

  function open(id: string) {
    setJobId(id);
    history.replaceState(null, "", `/?job=${id}`);
  }

  function reset() {
    setJobId(null);
    setSession(newSession());
    history.replaceState(null, "", "/");
  }

  // Phones need https for the camera. The plain-HTTP stage (:5174) points them at the HTTPS dev server (:5173).
  const isLocal = location.hostname === "localhost" || location.hostname === "127.0.0.1";
  const httpsPort = location.protocol === "https:" ? location.port || "443" : "5173";
  const captureQuery = `session=${encodeURIComponent(session)}&omni=${askOmni ? "1" : "0"}&genie=${askGenie ? "1" : "0"}`;
  const captureUrl = isLocal
    ? (info ? `https://${info.lan_ip}:${httpsPort}/capture?${captureQuery}` : "")
    : location.protocol === "https:"
      ? `${location.origin}/capture?${captureQuery}`
      : `https://${location.hostname}:5173/capture?${captureQuery}`;

  return (
    <div className="min-h-full flex flex-col">
      <header className="flex flex-wrap items-baseline justify-between gap-x-6 px-4 sm:px-10 pt-8 pb-4">
        <button onClick={reset} className="display text-3xl font-extrabold tracking-tight" aria-label="Foresight, back to start">
          Foresight
        </button>
        {!job && (
          <nav className="flex items-center gap-6 text-lg">
            <a href="/results" className="font-medium underline decoration-line decoration-2 underline-offset-4 hover:decoration-ink">How well it predicts</a>
          </nav>
        )}
        {job && (
          <div className="flex items-center gap-4">
            <Elapsed job={job} />
            {!job.result && !job.ended && (
              <button onClick={async () => { await cancelJob(job.id); reset(); }}
                      className="rounded-full border border-bad/50 px-4 py-1.5 text-base font-medium text-bad hover:bg-bad/5">
                Stop this run
              </button>
            )}
            <button onClick={reset} className="rounded-full bg-ink px-4 py-1.5 text-base font-medium text-paper">New run</button>
          </div>
        )}
      </header>
      <main className="flex-1 px-4 sm:px-10 pb-10">
        {!job && (
          <Idle info={info} infoErr={infoErr} samples={samples} captureUrl={captureUrl} askOmni={askOmni}
                setAskOmni={setAskOmni} askGenie={askGenie} setAskGenie={setAskGenie} busy={busy} onSample={runSample} startErr={startErr}
                onUpload={runUpload} recent={recent} onOpen={open} />
        )}
        {job && !job.result && <Running job={job} models={info?.models} onReset={reset} />}
        {job && job.result && <Reveal job={job} models={info?.models} onReset={reset} />}
      </main>
    </div>
  );
}

function Elapsed({ job }: { job: JobState }) {
  const [, tick] = useState(0);
  const started = useRef(Date.now() - job.lastT * 1000);
  useEffect(() => {
    if (job.ended) return;
    const t = setInterval(() => tick((x) => x + 1), 100);
    return () => clearInterval(t);
  }, [job.ended]);
  const done = job.result?.metrics.latency_s_total;
  const s = done ?? (job.ended ? job.lastT : (Date.now() - started.current) / 1000);
  return (
    <div className="font-mono text-base text-muted flex items-center gap-2">
      {!job.ended && !done && <span className="live-dot inline-block size-2.5 rounded-full bg-cobalt" aria-hidden />}
      {done ? `predicted in ${s.toFixed(1)} s` : `${s.toFixed(1)} s`}
    </div>
  );
}

/* ----------------------------------------------------------------------------------------- idle */
function Idle(p: {
  info: Info | null; infoErr: string | null; samples: Sample[]; captureUrl: string; askOmni: boolean;
  setAskOmni: (v: boolean) => void; askGenie: boolean; setAskGenie: (v: boolean) => void; busy: string | null; onSample: (s: Sample) => void; startErr: string | null;
  onUpload: (f: File) => void; recent: RecentRun[]; onOpen: (id: string) => void;
}) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-[1fr_auto] gap-x-16 gap-y-10 max-w-[1400px]">
      <section className="pt-6">
        <h1 className="text-[clamp(2.75rem,8vw,7.5rem)] font-extrabold">Watch 3 seconds.<br />Predict the next 5.</h1>
        <p className="mt-6 max-w-[38rem] text-xl text-muted">
          For robotics teams. Foresight reads a short side-view clip, fits real physics to what it saw, and
          renders what happens next. Next to it, Gemini's video model continues the same clip, and both are
          checked against what really happened.
        </p>
        {p.infoErr && <p role="alert" className="mt-6 text-bad font-medium">{p.infoErr}</p>}
        {p.info && !p.info.physics_available && (
          <p className="mt-6 max-w-[38rem] rounded-lg border border-amber/40 bg-amber/10 px-4 py-3 text-base">
            The physics module is not loaded on this server yet, so runs use the constant-velocity baseline
            (no physics) and are labelled that way.
          </p>
        )}
      </section>

      <aside className="self-start lg:mt-6 rounded-2xl bg-card p-7 shadow-[0_1px_0_var(--color-line)] border border-line w-full max-w-[380px] lg:w-[340px]">
        <h2 className="text-2xl font-bold">Record on your phone</h2>
        <p className="mt-2 text-base text-muted">Scan, hold the phone still, side view, roll or drop one ball. It records 8 seconds.</p>
        <div className="mt-5 grid place-items-center rounded-xl bg-paper p-5">
          {p.captureUrl ? <QRCodeSVG value={p.captureUrl} size={220} bgColor="transparent" fgColor="#15171c" /> : <div className="size-[220px]" />}
        </div>
        <p className="mt-3 break-all font-mono text-xs text-muted">{p.captureUrl}</p>
        <label className={`mt-5 flex items-start gap-3 text-base ${p.info && !p.info.omni_enabled ? "opacity-60" : ""}`}>
          <input type="checkbox" className="mt-1.5 size-4 accent-[var(--color-amber)]" checked={p.askOmni}
                 disabled={p.info ? !p.info.omni_enabled : false}
                 onChange={(e) => p.setAskOmni(e.target.checked)} />
          <span>
            Also ask Omni ({p.info?.models.baseline_video ?? "Gemini Omni"}) to continue the clip{" "}
            <span className="text-muted">
              {p.info && !p.info.omni_enabled
                ? "(disabled on server; set FORESIGHT_OMNI_ENABLED=true on a paid-tier key)"
                : "(about 40 s, paid tier)"}
            </span>
          </span>
        </label>
        <label className={`mt-3 flex items-start gap-3 text-base ${p.info && !p.info.genie_enabled ? "opacity-60" : ""}`}>
          <input type="checkbox" className="mt-1.5 size-4 accent-[var(--color-plum)]" checked={p.askGenie}
                 disabled={p.info ? !p.info.genie_enabled : false}
                 onChange={(e) => p.setAskGenie(e.target.checked)} />
          <span>
            And {p.info?.models.world_model ?? "Genie 4"} to play the scene on from its last frame{" "}
            <span className="text-muted">
              {p.info && !p.info.genie_enabled
                ? "(disabled on server; set FORESIGHT_GENIE_ENABLED=true when available)"
                : "(about 15 s)"}
            </span>
          </span>
        </label>
        <DropVideo busy={p.busy === "upload"} onFile={p.onUpload} />
      </aside>

      <section className="lg:col-span-2 min-w-0">
        <h2 className="text-2xl font-bold">Or use a Physics-IQ clip</h2>
        <p className="mt-1 text-base text-muted">Real footage from the benchmark. The last 5 seconds stay hidden until the reveal.</p>
        <div className="mt-5 flex gap-4 overflow-x-auto pb-2">
          {p.samples.length === 0 && <p className="text-muted">No local clips found. Run phase0/physiq_data.py to fetch some.</p>}
          {p.samples.map((s) => (
            <button key={s.id} onClick={() => p.onSample(s)} disabled={!!p.busy} aria-label={`Run ${s.title}`}
                    className="monitor shrink-0 w-[280px] text-left disabled:opacity-60">
              <img src={apiUrl(s.thumb_url)} alt="" />
              <span className="tag"><span className="dot" />{p.busy === s.id ? "Starting…" : s.title}</span>
              <span className="tc">3 s seen</span>
            </button>
          ))}
        </div>
        {p.startErr && <p role="alert" className="mt-4 text-bad font-medium">{p.startErr}</p>}
      </section>

      {p.recent.length > 0 && (
        <section className="lg:col-span-2 min-w-0">
          <h2 className="text-2xl font-bold">Recent runs</h2>
          <p className="mt-1 text-base text-muted">Open one to replay the reveal. The number is how far Foresight's object ended from the real one.</p>
          <div className="mt-5 flex gap-4 overflow-x-auto pb-2">
            {p.recent.map((r) => (
              <button key={r.job_id} onClick={() => p.onOpen(r.job_id)} aria-label={`Open the ${r.title} run`}
                      className="monitor shrink-0 w-[240px] text-left">
                {r.poster ? <img src={apiUrl(r.poster)} alt="" /> : <div className="slate">{r.title}</div>}
                <span className="tag"><span className="dot" style={{ background: "var(--color-cobalt)" }} />{r.title}</span>
                <span className="tc">{r.endpoint_error_cm.foresight !== null ? fmtCm(r.endpoint_error_cm.foresight) : new Date(r.created).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>
              </button>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

/** A video file from this computer: for when the phone cannot reach the laptop (venue Wi-Fi often blocks it). */
function DropVideo({ busy, onFile }: { busy: boolean; onFile: (f: File) => void }) {
  const [over, setOver] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const take = (files: FileList | null) => { const f = files?.[0]; if (f && f.type.startsWith("video/")) onFile(f); };
  return (
    <div onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
         onDrop={(e) => { e.preventDefault(); setOver(false); take(e.dataTransfer.files); }}
         className={`mt-6 rounded-xl border-2 border-dashed p-5 text-base transition-colors ${over ? "border-cobalt bg-cobalt/5" : "border-line"}`}>
      <p className="font-semibold">No phone on this network?</p>
      <p className="mt-1 text-muted">Drop a side-view video here, at least 3 s long. Anything after 3 s is kept hidden as the answer.</p>
      <button onClick={() => input.current?.click()} disabled={busy}
              className="mt-3 rounded-full bg-ink px-4 py-1.5 font-medium text-paper disabled:opacity-60">
        {busy ? "Uploading…" : "Choose a video"}
      </button>
      <input ref={input} type="file" accept="video/*" className="hidden" onChange={(e) => take(e.target.files)} />
    </div>
  );
}

/* ----------------------------------------------------------------------------------------- running */
function Timeline({ condUrl }: { condUrl?: string }) {
  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
      <figure className="monitor lane-reality">
        {condUrl ? <video src={condUrl} poster={condUrl.replace(/\.mp4$/, ".jpg")} autoPlay muted loop playsInline />
                 : <div className="slate">Loading the clip</div>}
        <figcaption className="tag"><span className="dot" />What the model sees</figcaption>
        <span className="tc">0:00 – 0:03</span>
      </figure>
      <figure className="monitor">
        <div className="slate">The next 5 seconds stay hidden until every step below is done.</div>
        <figcaption className="tag"><span className="dot" style={{ background: "#5f6470" }} />Not seen yet</figcaption>
        <span className="tc">0:03 – 0:08</span>
      </figure>
    </div>
  );
}

function StepRow({ n, title, state, children }: { n: number; title: string; state: "wait" | "live" | "done" | "error"; children?: React.ReactNode }) {
  const tone = state === "done" ? "text-ink" : state === "live" ? "text-cobalt" : state === "error" ? "text-bad" : "text-muted/60";
  return (
    <li className="grid grid-cols-[3rem_1fr] gap-4 py-3.5 border-b border-line last:border-0">
      <span className={`display text-4xl font-extrabold ${tone}`}>{n}</span>
      <div>
        <div className={`text-xl font-semibold ${state === "wait" ? "text-muted/70" : ""}`}>
          {title} {state === "live" && <span className="live-dot ml-1 inline-block size-2 rounded-full bg-cobalt align-middle" aria-label="running" />}
        </div>
        {children && <div className="mt-1 text-base text-muted">{children}</div>}
      </div>
    </li>
  );
}

function stageState(job: JobState, name: string): "wait" | "live" | "done" | "error" {
  const s = job.stages[name];
  if (!s) return "wait";
  return s.status === "start" ? "live" : s.status === "error" ? "error" : "done";
}

function Bars({ conf }: { conf: Record<string, number> }) {
  return (
    <div className="mt-2 space-y-2 max-w-md">
      {Object.entries(conf).map(([k, v]) => (
        <div key={k}>
          <div className="flex justify-between text-sm"><span>{k.replaceAll("_", " ")}</span><span className="font-mono">{v.toFixed(2)}</span></div>
          <div className="relative mt-1 h-2.5 rounded-full bg-line">
            <div className={`h-full rounded-full ${v >= 0.7 ? "bg-cobalt" : "bg-bad"}`} style={{ width: `${Math.round(v * 100)}%` }} />
            <div className="absolute top-[-3px] h-[16px] w-[2px] bg-ink" style={{ left: "70%" }} title="pass threshold 0.7" />
          </div>
        </div>
      ))}
    </div>
  );
}

function Running({ job, models, onReset }: { job: JobState; models?: Record<string, string>; onReset: () => void }) {
  const p = job.perception;
  const tr = job.tracking[0];
  const fit = job.fits.at(-1);
  const ver = job.verifies.at(-1);
  const physicsStage = job.stages.fit ? "fit" : job.stages.simulate ? "simulate" : "fit";
  const perceptionModel = p?.model ?? models?.perception ?? "Gemini";
  const verifyModel = ver?.model ?? models?.verify ?? "Gemini";
  const omniModel = models?.baseline_video ?? "Gemini Omni";
  const genieModel = models?.world_model ?? "Genie 4";
  return (
    <div className="max-w-[1400px] space-y-8">
      <Timeline condUrl={apiUrl(job.ingest?.conditioning_url) ?? undefined} />
      {job.error && (
        <div role="alert" className="rounded-xl border border-bad/40 bg-bad/5 p-5">
          <p className="text-xl font-semibold text-bad">Stopped at {job.error.stage}</p>
          <p className="mt-1 text-base">{job.error.message}</p>
          <button onClick={onReset} className="mt-4 rounded-full bg-ink px-5 py-2 text-paper font-medium">Start over</button>
        </div>
      )}
      <div className="grid grid-cols-1 lg:grid-cols-[1.4fr_1fr] gap-8 lg:gap-12">
        <ol>
          <StepRow n={1} title={`${perceptionModel} reads the scene`} state={stageState(job, "perceive")}>
            {p && (
              <>
                {p.objects.filter((o) => o.moving).map((o) => `${o.name}, a ${o.shape} about ${o.est_real_size_cm} cm across`).join("; ")}
                . Motion: {p.motion_type}. Read in {p.latency_s.toFixed(1)} s{p.cached ? " (cached from an earlier run)" : ""}.
              </>
            )}
          </StepRow>
          <StepRow n={2} title="Track it frame by frame" state={stageState(job, "track")}>
            {tr && <>{tr.object}: found in {tr.frames_ok} of {tr.frames_total} frames</>}
          </StepRow>
          <StepRow n={3} title="Measure the scene" state={job.scale ? "done" : stageState(job, "track") === "done" ? "live" : "wait"}>
            {job.scale && <>{Math.round(job.scale.px_per_m).toLocaleString()} pixels per metre, from the object's size{job.scale.gravity_check ? `; gravity check ${job.scale.gravity_check.g_est_m_s2} m/s²` : ""}</>}
          </StepRow>
          <StepRow n={4} title="Fit physics to what it saw" state={stageState(job, physicsStage)}>
            {fit && <>{fit.motion_model}, {fit.fit_error_px.toFixed(1)} px from the tracked path over {fit.n_points} points (attempt {fit.attempt + 1})</>}
            {!fit && job.stages.simulate && <>{job.stages.simulate.message}</>}
            {job.fallback && <p className="mt-2 text-bad">Physics check failed ({job.fallback.reason}); this run shows {job.fallback.to} instead.</p>}
          </StepRow>
          <StepRow n={5} title={`${verifyModel} checks the fit`} state={job.verifies.length ? "done" : stageState(job, "verify")}>
            {ver && <Bars conf={ver.confidences} />}
            {job.repairs.map((r) => {
              const raw = r.model ?? models?.repair ?? "Local multi-start";
              const label = raw === "local-multistart" ? "Local multi-start" : raw;
              return (
                <p key={r.attempt} className="mt-2">{label} proposed {r.proposal.motion_model}: {r.proposal.reason}</p>
              );
            })}
            {job.repairSkipped && <p className="mt-2 text-muted">Stopped repairing: {job.repairSkipped.reason}.</p>}
          </StepRow>
          <StepRow n={6} title="Render the next 5 seconds" state={job.render ? "done" : stageState(job, "render")}>
            {job.render && <>{job.render.frames} frames, {job.render.duration_s.toFixed(2)} s</>}
          </StepRow>
        </ol>
        <div className="space-y-6 self-start">
          <BaselineLane title="Meanwhile, Omni" sub={`${omniModel} continues the same 3 seconds as video.`} lane={job.omni} />
          <BaselineLane title={`And ${genieModel}`} sub={`The ${genieModel} world model plays the scene on from its last frame.`} lane={job.genie} />
        </div>
      </div>
    </div>
  );
}

function BaselineLane({ title, sub, lane: o }: { title: string; sub: string; lane?: Lane }) {
  return (
    <section className="rounded-2xl border border-line bg-card p-6">
      <h3 className="text-2xl font-bold">{title}</h3>
      <p className="mt-1 text-base text-muted">{sub}</p>
      <p className="mt-4 text-lg">
        {!o && "Waiting to start"}
        {o?.status === "start" && <span className="text-amber font-medium">Generating <span className="live-dot inline-block size-2 rounded-full bg-amber align-middle" /></span>}
        {o?.status === "done" && <span className="font-medium">Done in {o.latency_s?.toFixed(0)} s</span>}
        {o?.status === "skipped" && <span className="text-muted">{o.message || "Not requested for this run"}</span>}
        {o?.status === "error" && <span className="text-bad">Failed: {o.error}</span>}
      </p>
    </section>
  );
}

/* ----------------------------------------------------------------------------------------- reveal */
function Reveal({ job, models, onReset }: { job: JobState; models?: Record<string, string>; onReset: () => void }) {
  const r = job.result!;
  const omniUrl = r.urls.omni ?? (job.omni?.status === "done" ? job.omni.url : null);
  const genieUrl = r.urls.genie ?? (job.genie?.status === "done" ? job.genie.url : null);
  const eeF = r.metrics.endpoint_error_cm.foresight;
  const eeO = r.metrics.endpoint_error_cm.omni ?? job.omni?.endpoint_error_cm ?? null;
  const eeG = r.metrics.endpoint_error_cm.genie ?? job.genie?.endpoint_error_cm ?? null;
  const [showPath, setShowPath] = useState(false);
  const fellBack = r.metrics.dynamics_used === "const_velocity_fallback";
  const foresightTitle = r.dynamics !== "physics" ? "Constant velocity (no physics)" : fellBack ? "Foresight (fell back to constant velocity)" : "Foresight (physics)";
  // The panel closest to where the object really ended up, among those with a measured endpoint.
  const scored = ([["foresight", eeF], ["omni", eeO], ["genie", eeG]] as const).filter(([, e]) => e !== null) as [string, number][];
  const better = scored.length >= 2 ? scored.reduce((a, b) => (b[1] < a[1] ? b : a))[0] : null;
  const laneNote = (l?: Lane, saved?: { status: string; message?: string | null }) =>
    l?.status === "start"
      ? "still generating…"
      : l?.status === "skipped" || saved?.status === "skipped"
        ? l?.message || saved?.message || "not requested"
        : l?.status === "error" || saved?.status === "error"
          ? "failed"
          : "";
  const verifyLabel = (job.verifies.at(-1)?.model ?? r.models?.verify ?? models?.verify) === "gemini-decision-1" ? "Decision 1" : "Fit check";

  const panels = [
    { key: "reality", title: "Reality", sub: "what actually happened", url: apiUrl(r.urls.reality), color: "text-ink", err: null as number | null, note: r.urls.reality ? "" : "no hidden clip for this run" },
    { key: "foresight", title: foresightTitle, sub: "simulated from 3 s of video", url: apiUrl(showPath && r.urls.overlay ? r.urls.overlay : r.urls.foresight), color: "text-cobalt", err: eeF, note: "" },
    { key: "omni", title: "Omni (video model)", sub: "generated from the same 3 s", url: apiUrl(omniUrl),
      color: "text-amber", err: eeO,
      note: laneNote(job.omni, r.omni) },
    ...(job.genie && job.genie.status !== "skipped" ? [{ key: "genie", title: "Genie 4 (world model)", sub: "played on from the last frame",
      url: apiUrl(genieUrl), color: "text-plum", err: eeG, note: laneNote(job.genie, r.genie) }] : []),
  ];

  return (
    <div className="max-w-[1600px] space-y-6">
      <SyncedPanels panels={panels} better={better} />
      <div className="flex flex-wrap items-center gap-x-8 gap-y-3 text-base text-muted">
        <span>Endpoint error: how far the predicted object ends up from the real one at the last moment it is visible.</span>
        {r.urls.overlay && (
          <label className="flex items-center gap-2 text-ink">
            <input type="checkbox" className="size-4 accent-[var(--color-cobalt)]" checked={showPath} onChange={(e) => setShowPath(e.target.checked)} />
            Draw the predicted path on reality
          </label>
        )}
      </div>
      <dl className="grid grid-cols-2 md:grid-cols-4 gap-6 border-t border-line pt-5">
        <Stat label="Total time" value={`${r.metrics.latency_s_total.toFixed(1)} s`} />
        <Stat label="Scene parse" value={`${(r.metrics.latency_s_by_stage.perceive ?? 0).toFixed(1)} s`} />
        <Stat label="Fit error" value={r.metrics.fit_error_px !== null ? `${r.metrics.fit_error_px.toFixed(1)} px` : "—"} />
        <Stat label={verifyLabel} value={r.metrics.decision1_confidences ? Object.values(r.metrics.decision1_confidences).map((v) => v.toFixed(2)).join(" / ") : "—"} />
      </dl>
      <Downloads r={r} />
      {(r.paths?.predicted.length || r.understanding) && (
        <div className="grid grid-cols-1 gap-10 border-t border-line pt-8 xl:grid-cols-[1.25fr_1fr]">
          {r.paths && r.paths.predicted.length > 0 && <PathPlot paths={r.paths} frameUrl={r.frame_url ?? null} endpointCm={eeF} />}
          {r.understanding && <Understood u={r.understanding} />}
        </div>
      )}
      <button onClick={onReset} className="rounded-full bg-ink px-6 py-3 text-paper text-lg font-medium">Run another clip</button>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-sm text-muted">{label}</dt>
      <dd className="display text-3xl font-bold">{value}</dd>
    </div>
  );
}

type Panel = { key: string; title: string; sub: string; url: string | null; color: string; err: number | null; note: string };

function SyncedPanels({ panels, better }: { panels: Panel[]; better: string | null }) {
  const refs = useRef<Record<string, HTMLVideoElement | null>>({});
  const [playing, setPlaying] = useState(true);
  const [slow, setSlow] = useState(false);
  const [t, setT] = useState(0);
  const [dur, setDur] = useState(DEFAULT_TIMELINE_S);
  const [held, setHeld] = useState<Record<string, number | null>>({});
  const clockRef = useRef({ t: 0, lastMs: null as number | null });
  const urls = useMemo(() => panels.map((p) => p.url ?? "").join("|"), [panels]);

  useEffect(() => {
    clockRef.current.lastMs = null;
    const rate = slow ? 0.5 : 1;
    let raf = 0;
    const loop = (nowMs: number) => {
      const map = refs.current;
      const targetDur = timelineDuration(Object.values(map).map((v) => (v && v.src ? v.duration : 0)));
      const dtMs = clockRef.current.lastMs === null ? 0 : nowMs - clockRef.current.lastMs;
      clockRef.current.lastMs = nowMs;
      const step = advanceClock(clockRef.current.t, dtMs, playing, rate, targetDur);
      clockRef.current.t = step.t;
      const statuses = syncAllVideos(map, step.t, { playing, rate, wrapped: step.wrapped });
      const nextHeld: Record<string, number | null> = {};
      for (const [k, st] of Object.entries(statuses)) {
        nextHeld[k] = st.heldAtEnd ? st.clipDuration : null;
      }
      setDur(targetDur);
      setT(step.t);
      setHeld((prev) => {
        const same =
          Object.keys(prev).length === Object.keys(nextHeld).length &&
          Object.entries(nextHeld).every(([k, v]) => prev[k] === v);
        return same ? prev : nextHeld;
      });
      raf = requestAnimationFrame(loop);
    };
    syncAllVideos(refs.current, clockRef.current.t, { playing, rate });
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [playing, slow, urls]);

  const seek = (s: number) => {
    clockRef.current = { t: s, lastMs: null };
    syncAllVideos(refs.current, s, { playing, rate: slow ? 0.5 : 1, forceSeek: true });
    setT(s);
  };

  return (
    <div>
      <div className={`grid grid-cols-1 gap-6 md:grid-cols-2 ${panels.length > 3 ? "xl:grid-cols-4" : "xl:grid-cols-3"}`}>
        {panels.map((p) => {
          const heldDur = held[p.key];
          const panelT = heldDur != null ? heldDur : t;
          return (
            <div key={p.key}>
              <figure className={`monitor lane-${p.key}`}>
                {p.url ? (
                  <video
                    ref={(el) => { refs.current[p.key] = el; }}
                    src={p.url}
                    poster={p.url.replace(/\.mp4$/, ".jpg")}
                    muted
                    playsInline
                    onLoadedData={(e) => {
                      syncVideo(e.currentTarget, clockRef.current.t, { playing, rate: slow ? 0.5 : 1, forceSeek: true });
                    }}
                  />
                ) : (
                  <div className="slate">{p.note || "No video"}</div>
                )}
                <figcaption className="tag"><span className="dot" />{p.title}</figcaption>
                <span className="tc">
                  0:{(3 + panelT).toFixed(2).padStart(5, "0")}{heldDur != null && heldDur < dur - 0.1 ? " · ended" : ""}
                </span>
              </figure>
              <div className="mt-4 text-base text-muted">{p.sub}</div>
              {p.key === "reality" ? (
                <div className="display text-6xl font-extrabold">Reference</div>
              ) : (
                <>
                  <div className={`display text-6xl font-extrabold ${p.key === better ? "text-good" : "text-ink"}`}>
                    {fmtCm(p.err)}
                  </div>
                  <div className="text-base text-muted">{p.err === null ? (p.note || "no endpoint measured") : "from where it really ended up"}</div>
                </>
              )}
            </div>
          );
        })}
      </div>
      <div className="mt-6 flex flex-wrap items-center gap-4">
        <button onClick={() => setPlaying((x) => !x)} className="rounded-full bg-ink px-5 py-2 text-paper font-medium w-24">{playing ? "Pause" : "Play"}</button>
        <button onClick={() => setSlow((x) => !x)} aria-pressed={slow} className={`rounded-full border px-4 py-2 font-medium ${slow ? "border-cobalt text-cobalt" : "border-line"}`}>Half speed</button>
        <input type="range" min={0} max={dur} step={1 / 24} value={Math.min(t, dur)} onChange={(e) => seek(Number(e.target.value))}
               className="flex-1 accent-[var(--color-cobalt)]" aria-label="Scrub all videos" />
        <span className="font-mono text-base sm:w-36 text-right">0:{(3 + t).toFixed(2).padStart(5, "0")} of 0:{String(Math.round(3 + dur)).padStart(2, "0")}</span>
      </div>
    </div>
  );
}
