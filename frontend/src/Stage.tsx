import { useEffect, useMemo, useRef, useState } from "react";
import { QRCodeSVG } from "qrcode.react";
import { cancelJob, fmtCm, getInfo, getSamples, startJob, useJob, type Info, type JobState, type Sample } from "./api";

const newSession = () => Math.random().toString(36).slice(2, 10);

export default function Stage() {
  const [info, setInfo] = useState<Info | null>(null);
  const [infoErr, setInfoErr] = useState<string | null>(null);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [session, setSession] = useState(newSession);
  const [jobId, setJobId] = useState<string | null>(() => new URLSearchParams(location.search).get("job"));
  const [askOmni, setAskOmni] = useState(true);
  const [busy, setBusy] = useState<string | null>(null);
  const [startErr, setStartErr] = useState<string | null>(null);
  const job = useJob(jobId);

  useEffect(() => {
    getInfo().then(setInfo).catch((e) => setInfoErr(String(e.message ?? e)));
    getSamples().then(setSamples);
  }, []);

  // A phone that scanned the QR code posts with this session; pick its job up.
  useEffect(() => {
    if (jobId) return;
    const since = new Date().toISOString();
    const t = setInterval(async () => {
      const r = await fetch(`/api/jobs?session=${session}`).then((x) => x.json()).catch(() => []);
      const fresh = r.find((j: { created: string }) => j.created >= since.slice(0, 19)) ?? r[0];
      if (fresh) setJobId(fresh.job_id);
    }, 1000);
    return () => clearInterval(t);
  }, [jobId, session]);

  const dynamics = info?.physics_available ? "physics" : "const_velocity";

  async function runSample(s: Sample) {
    setBusy(s.id);
    setStartErr(null);
    const f = new FormData();
    f.set("sample_id", s.id);
    f.set("session", session);
    f.set("omni", String(askOmni));
    f.set("dynamics", dynamics);
    try {
      setJobId((await startJob(f)).job_id);
    } catch (e) {
      setStartErr(String((e as Error).message));
    } finally {
      setBusy(null);
    }
  }

  function reset() {
    setJobId(null);
    setSession(newSession());
    history.replaceState(null, "", "/");
  }

  const captureUrl = info ? `https://${info.lan_ip}:${location.port || "5173"}/capture?session=${session}` : "";

  return (
    <div className="min-h-full flex flex-col">
      <header className="flex flex-wrap items-baseline justify-between gap-x-6 px-4 sm:px-10 pt-8 pb-4">
        <button onClick={reset} className="display text-3xl font-extrabold tracking-tight" aria-label="Foresight, back to start">
          Foresight
        </button>
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
                setAskOmni={setAskOmni} busy={busy} onSample={runSample} startErr={startErr} />
        )}
        {job && !job.result && <Running job={job} onReset={reset} />}
        {job && job.result && <Reveal job={job} onReset={reset} />}
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
  setAskOmni: (v: boolean) => void; busy: string | null; onSample: (s: Sample) => void; startErr: string | null;
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
        <label className="mt-5 flex items-start gap-3 text-base">
          <input type="checkbox" className="mt-1.5 size-4 accent-[var(--color-amber)]" checked={p.askOmni}
                 onChange={(e) => p.setAskOmni(e.target.checked)} />
          <span>Also ask Omni to continue the clip <span className="text-muted">(about 40 s, about $1)</span></span>
        </label>
      </aside>

      <section className="lg:col-span-2 min-w-0">
        <h2 className="text-2xl font-bold">Or use a Physics-IQ clip</h2>
        <p className="mt-1 text-base text-muted">Real footage from the benchmark. The last 5 seconds stay hidden until the reveal.</p>
        <div className="mt-5 flex gap-4 overflow-x-auto pb-2">
          {p.samples.length === 0 && <p className="text-muted">No local clips found. Run phase0/physiq_data.py to fetch some.</p>}
          {p.samples.map((s) => (
            <button key={s.id} onClick={() => p.onSample(s)} disabled={!!p.busy} aria-label={`Run ${s.title}`}
                    className="monitor shrink-0 w-[280px] text-left disabled:opacity-60">
              <img src={s.thumb_url} alt="" />
              <span className="tag"><span className="dot" />{p.busy === s.id ? "Starting…" : s.title}</span>
              <span className="tc">3 s seen</span>
            </button>
          ))}
        </div>
        {p.startErr && <p role="alert" className="mt-4 text-bad font-medium">{p.startErr}</p>}
      </section>
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

function Running({ job, onReset }: { job: JobState; onReset: () => void }) {
  const p = job.perception;
  const tr = job.tracking[0];
  const fit = job.fits.at(-1);
  const ver = job.verifies.at(-1);
  const physicsStage = job.stages.fit ? "fit" : job.stages.simulate ? "simulate" : "fit";
  return (
    <div className="max-w-[1400px] space-y-8">
      <Timeline condUrl={job.ingest?.conditioning_url} />
      {job.error && (
        <div role="alert" className="rounded-xl border border-bad/40 bg-bad/5 p-5">
          <p className="text-xl font-semibold text-bad">Stopped at {job.error.stage}</p>
          <p className="mt-1 text-base">{job.error.message}</p>
          <button onClick={onReset} className="mt-4 rounded-full bg-ink px-5 py-2 text-paper font-medium">Start over</button>
        </div>
      )}
      <div className="grid grid-cols-1 lg:grid-cols-[1.4fr_1fr] gap-8 lg:gap-12">
        <ol>
          <StepRow n={1} title="Gemini 4 Hydrogen reads the scene" state={stageState(job, "perceive")}>
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
          </StepRow>
          <StepRow n={5} title="Gemini Decision 1 checks the fit" state={job.verifies.length ? "done" : stageState(job, "verify")}>
            {ver && <Bars conf={ver.confidences} />}
            {job.repairs.map((r) => (
              <p key={r.attempt} className="mt-2">Gemini 4 Carbon proposed {r.proposal.motion_model}: {r.proposal.reason}</p>
            ))}
          </StepRow>
          <StepRow n={6} title="Render the next 5 seconds" state={job.render ? "done" : stageState(job, "render")}>
            {job.render && <>{job.render.frames} frames, {job.render.duration_s.toFixed(2)} s</>}
          </StepRow>
        </ol>
        <OmniLane job={job} />
      </div>
    </div>
  );
}

function OmniLane({ job }: { job: JobState }) {
  const o = job.omni;
  return (
    <section className="self-start rounded-2xl border border-line bg-card p-6">
      <h3 className="text-2xl font-bold">Meanwhile, Omni</h3>
      <p className="mt-1 text-base text-muted">Gemini Omni 1.2 Flash continues the same 3 seconds as video.</p>
      <p className="mt-4 text-lg">
        {!o && "Waiting to start"}
        {o?.status === "start" && <span className="text-amber font-medium">Generating <span className="live-dot inline-block size-2 rounded-full bg-amber align-middle" /></span>}
        {o?.status === "done" && <span className="font-medium">Done in {o.latency_s?.toFixed(0)} s</span>}
        {o?.status === "skipped" && <span className="text-muted">Not requested for this run</span>}
        {o?.status === "error" && <span className="text-bad">Failed: {o.error}</span>}
      </p>
    </section>
  );
}

/* ----------------------------------------------------------------------------------------- reveal */
function Reveal({ job, onReset }: { job: JobState; onReset: () => void }) {
  const r = job.result!;
  const omniUrl = r.urls.omni ?? (job.omni?.status === "done" ? job.omni.url : null);
  const eeF = r.metrics.endpoint_error_cm.foresight;
  const eeO = r.metrics.endpoint_error_cm.omni ?? job.omni?.endpoint_error_cm ?? null;
  const [showPath, setShowPath] = useState(false);
  const foresightTitle = r.dynamics === "physics" ? "Foresight (physics)" : "Constant velocity (no physics)";
  const better = eeF !== null && eeO !== null ? (eeF <= eeO ? "f" : "o") : null;

  const panels = [
    { key: "reality", title: "Reality", sub: "what actually happened", url: r.urls.reality, color: "text-ink", err: null as number | null, note: r.urls.reality ? "" : "no hidden clip for this run" },
    { key: "foresight", title: foresightTitle, sub: "simulated from 3 s of video", url: showPath && r.urls.overlay ? r.urls.overlay : r.urls.foresight, color: "text-cobalt", err: eeF, note: "" },
    { key: "omni", title: "Omni (video model)", sub: "generated from the same 3 s", url: omniUrl,
      color: "text-amber", err: eeO,
      note: job.omni?.status === "start" ? "still generating…" : job.omni?.status === "skipped" ? "not requested" : job.omni?.status === "error" ? "failed" : "" },
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
        <Stat label="Decision 1" value={r.metrics.decision1_confidences ? Object.values(r.metrics.decision1_confidences).map((v) => v.toFixed(2)).join(" / ") : "—"} />
      </dl>
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

function SyncedPanels({ panels, better }: { panels: Panel[]; better: "f" | "o" | null }) {
  const refs = useRef<Record<string, HTMLVideoElement | null>>({});
  const [playing, setPlaying] = useState(true);
  const [slow, setSlow] = useState(false);
  const [t, setT] = useState(0);
  const urls = useMemo(() => panels.map((p) => p.url ?? "").join("|"), [panels]);

  const vids = () => Object.values(refs.current).filter((v): v is HTMLVideoElement => !!v && !!v.src);
  const master = () => vids()[0];

  useEffect(() => {
    let raf = 0;
    const loop = () => {
      const m = master();
      if (m) {
        for (const v of vids()) if (v !== m && Math.abs(v.currentTime - m.currentTime) > 0.06) v.currentTime = m.currentTime;
        setT(m.currentTime);
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [urls]);

  useEffect(() => {
    for (const v of vids()) {
      v.playbackRate = slow ? 0.5 : 1;
      if (playing) v.play().catch(() => {}); else v.pause();
    }
  }, [playing, slow, urls]);

  const seek = (s: number) => { for (const v of vids()) v.currentTime = s; setT(s); };

  return (
    <div>
      <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
        {panels.map((p) => (
          <div key={p.key}>
            <figure className={`monitor lane-${p.key}`}>
              {p.url ? (
                <video ref={(el) => { refs.current[p.key] = el; }} src={p.url} poster={p.url.replace(/\.mp4$/, ".jpg")} muted playsInline loop
                       onLoadedData={(e) => { if (playing) e.currentTarget.play().catch(() => {}); }} />
              ) : (
                <div className="slate">{p.note || "No video"}</div>
              )}
              <figcaption className="tag"><span className="dot" />{p.title}</figcaption>
              <span className="tc">0:{(3 + t).toFixed(2).padStart(5, "0")}</span>
            </figure>
            <div className="mt-4 text-base text-muted">{p.sub}</div>
            {p.key === "reality" ? (
              <div className="display text-6xl font-extrabold">Reference</div>
            ) : (
              <>
                <div className={`display text-6xl font-extrabold ${(p.key === "foresight" && better === "f") || (p.key === "omni" && better === "o") ? "text-good" : "text-ink"}`}>
                  {fmtCm(p.err)}
                </div>
                <div className="text-base text-muted">{p.err === null ? (p.note || "no endpoint measured") : "from where it really ended up"}</div>
              </>
            )}
          </div>
        ))}
      </div>
      <div className="mt-6 flex flex-wrap items-center gap-4">
        <button onClick={() => setPlaying((x) => !x)} className="rounded-full bg-ink px-5 py-2 text-paper font-medium w-24">{playing ? "Pause" : "Play"}</button>
        <button onClick={() => setSlow((x) => !x)} aria-pressed={slow} className={`rounded-full border px-4 py-2 font-medium ${slow ? "border-cobalt text-cobalt" : "border-line"}`}>Half speed</button>
        <input type="range" min={0} max={5} step={1 / 24} value={t} onChange={(e) => seek(Number(e.target.value))}
               className="flex-1 accent-[var(--color-cobalt)]" aria-label="Scrub all three videos" />
        <span className="font-mono text-base sm:w-36 text-right">0:{(3 + t).toFixed(2).padStart(5, "0")} of 0:08</span>
      </div>
    </div>
  );
}
