// The reveal's second half: where the object went, what Foresight understood, and the numbers to take away.
import { apiUrl, fmtCm, type Paths, type Result, type Understanding } from "./api";

/** Predicted vs real path, drawn in the clip's own pixels over the last frame the model saw. */
export function PathPlot({ paths, frameUrl, endpointCm }: { paths: Paths; frameUrl: string | null; endpointCm: number | null }) {
  const [W, H] = paths.orig_size;
  const s = W / 1000; // stroke and marker sizes scale with the clip's resolution
  const pred = paths.predicted;
  const real = paths.real;
  const lastReal = real.at(-1);
  // Foresight's position at the moment of the last real point: the endpoint error is measured there.
  const atEnd = lastReal ? pred.reduce((a, b) => (Math.abs(b[0] - lastReal[0]) < Math.abs(a[0] - lastReal[0]) ? b : a), pred[0]) : null;
  const line = (pts: number[][]) => pts.map((p) => `${p[1].toFixed(1)},${p[2].toFixed(1)}`).join(" ");

  return (
    <figure className="m-0">
      <div className="monitor">
        <div className="relative" style={{ aspectRatio: `${W} / ${H}` }}>
          {frameUrl && <img src={apiUrl(frameUrl)} alt="" className="absolute inset-0 h-full w-full object-cover" />}
          <svg viewBox={`0 0 ${W} ${H}`} className="absolute inset-0 h-full w-full" role="img"
               aria-label="Paths: what the model saw, what Foresight predicted, and what really happened">
            <rect width={W} height={H} fill="rgb(13 14 17 / 0.28)" />
            {pred.length > 1 && <polyline points={line(pred)} fill="none" stroke="var(--color-cobalt)" strokeWidth={7 * s}
                                          strokeLinecap="round" strokeLinejoin="round" />}
            {real.map((p, i) => <circle key={`r${i}`} cx={p[1]} cy={p[2]} r={6 * s} fill="#f2f2ef" />)}
            {paths.seen.map((p, i) => <circle key={`s${i}`} cx={p[1]} cy={p[2]} r={7 * s} fill="none" stroke="#f2f2ef"
                                              strokeWidth={2.5 * s} opacity={0.85} />)}
            {lastReal && atEnd && (
              <g>
                <line x1={atEnd[1]} y1={atEnd[2]} x2={lastReal[1]} y2={lastReal[2]} stroke="#f2f2ef" strokeWidth={3 * s}
                      strokeDasharray={`${10 * s} ${8 * s}`} />
                <circle cx={atEnd[1]} cy={atEnd[2]} r={16 * s} fill="none" stroke="var(--color-cobalt)" strokeWidth={6 * s} />
                <circle cx={lastReal[1]} cy={lastReal[2]} r={16 * s} fill="none" stroke="#f2f2ef" strokeWidth={6 * s} />
              </g>
            )}
          </svg>
        </div>
        <figcaption className="tag"><span className="dot" style={{ background: "var(--color-cobalt)" }} />Where it went</figcaption>
      </div>
      <div className="mt-3 flex flex-wrap gap-x-6 gap-y-1 text-base text-muted">
        <span><span className="mr-2 inline-block size-3 rounded-full border-2 border-ink align-middle" />seen in the first 3 s</span>
        <span><span className="mr-2 inline-block h-1 w-6 rounded bg-cobalt align-middle" />Foresight's next 5 s</span>
        <span><span className="mr-2 inline-block size-3 rounded-full bg-ink align-middle" />what really happened</span>
        {endpointCm !== null && <span className="text-ink">rings: {fmtCm(endpointCm)} apart at the last real sighting</span>}
      </div>
    </figure>
  );
}

const ROLE: Record<string, string> = { moving: "moving", "can be knocked": "resting, can be knocked", container: "open container", static: "fixed" };

/** What Gemini read and what the fit found, in plain units. */
export function Understood({ u }: { u: Understanding }) {
  const fell = u.dynamics_used === "const_velocity_fallback";
  const checks = Object.entries(u.checks.confidences);
  return (
    <section aria-labelledby="understood" className="space-y-6">
      <h3 id="understood" className="text-2xl font-bold">What Foresight understood</h3>
      <dl className="divide-y divide-line border-y border-line">
        {u.objects.map((o) => (
          <div key={o.id} className="grid grid-cols-[minmax(0,1fr)_auto] gap-x-4 py-3">
            <dt className="font-semibold">{o.name}</dt>
            <dd className="text-right text-muted">{ROLE[o.role] ?? o.role}</dd>
            <dd className="col-span-2 text-base text-muted">
              {o.shape}{o.size_cm ? `, ${o.size_cm} cm` : ""}{o.mass_g ? `, about ${o.mass_g >= 1000 ? `${(o.mass_g / 1000).toFixed(1)} kg` : `${Math.round(o.mass_g)} g`}` : ""}
              {o.fit && o.fit.speed_m_s !== null && <>; leaves the 3 s at <span className="text-ink font-medium">{o.fit.speed_m_s.toFixed(2)} m/s</span>,
                bounce {o.fit.bounce?.toFixed(2)}, friction {o.fit.friction?.toFixed(2)}</>}
            </dd>
          </div>
        ))}
        {u.surfaces.map((s) => (
          <div key={s.name} className="grid grid-cols-[minmax(0,1fr)_auto] gap-x-4 py-3">
            <dt className="font-semibold">{s.name}</dt>
            <dd className="text-right text-muted">{s.material === "soft" ? "soft surface" : s.kind === "container" ? "container rim" : `${s.kind}, hard`}</dd>
          </div>
        ))}
      </dl>
      {u.depth_notes.length > 0 && (
        <ul className="space-y-1.5 text-base">
          {u.depth_notes.map((n) => <li key={n}>{n}</li>)}
        </ul>
      )}
      <p className="text-base text-muted">
        Motion: <span className="text-ink">{u.motion_model ?? "—"}</span>. Scale: {Math.round(u.px_per_m).toLocaleString()} pixels per metre,
        from {u.scale_source === "gravity" ? "the fall (gravity)" : "the object's real size"}.
        {checks.length > 0 && <> Fit check: {checks.map(([k, v]) => `${k.replaceAll("_", " ")} ${v.toFixed(2)}`).join(", ")}.</>}
        {u.repairs_used ? ` Repaired the fit ${u.repairs_used} time${u.repairs_used > 1 ? "s" : ""}.` : ""}
      </p>
      {fell && <p role="note" className="text-base text-bad">The physics failed its energy check ({u.fallback_reason}), so this run shows constant velocity.</p>}
    </section>
  );
}

/** The predicted trajectory as files a robotics team can load into its own simulator. */
export function Downloads({ r }: { r: Result }) {
  if (!r.urls.trajectory_json) return null;
  return (
    <div className="flex flex-wrap items-center gap-3">
      <span className="text-base text-muted">Take the prediction with you:</span>
      <a href={apiUrl(r.urls.trajectory_json)} download={`foresight-${r.job_id}.json`}
         className="rounded-full border border-ink px-4 py-1.5 font-medium hover:bg-ink hover:text-paper">Trajectory, JSON</a>
      {r.urls.trajectory_csv && (
        <a href={apiUrl(r.urls.trajectory_csv)} download={`foresight-${r.job_id}.csv`}
           className="rounded-full border border-ink px-4 py-1.5 font-medium hover:bg-ink hover:text-paper">CSV</a>
      )}
      <span className="text-sm text-muted">positions in metres and pixels at 24 FPS, with the fitted speed, friction and bounce</span>
    </div>
  );
}
