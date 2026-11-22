// The reveal's second half: where the object went, counterfactual what-if rollouts, what Foresight understood, and exports.
import { useEffect, useMemo, useRef, useState } from "react";
import { apiUrl, fmtCm, getWhatIf, type Paths, postWhatIf, type Result, type Understanding, type WhatIfBaseline, type WhatIfResponse } from "./api";
import { buildWhatIfRequest, clientToScenePx, draftFromBaseline, velocityBound, type WhatIfDraft } from "./whatif";

function WhatIfSlider({
  label, ariaLabel, display, min, max, step, value, onChange,
}: {
  label: string; ariaLabel: string; display: string; min: number; max: number; step: number; value: number; onChange: (v: number) => void;
}) {
  return (
    <label className="space-y-1">
      <span className="flex justify-between">
        <span className="font-medium">{label}</span>
        <span className="font-mono text-muted">{display}</span>
      </span>
      <input type="range" aria-label={ariaLabel} min={min} max={max} step={step} value={value}
             onChange={(e) => onChange(Number(e.target.value))} className="w-full accent-[var(--color-good)]" />
    </label>
  );
}

/** Predicted vs real path, drawn in the clip's own pixels over the last frame the model saw, with interactive What-If controls. */
export function PathPlot({
  jobId, paths, frameUrl, endpointCm,
}: {
  jobId?: string; paths: Paths; frameUrl: string | null; endpointCm: number | null;
}) {
  const [W, H] = paths.orig_size;
  const s = W / 1000; // stroke and marker sizes scale with the clip's resolution
  const pred = paths.predicted;
  const real = paths.real;
  const lastReal = real.at(-1);

  const svgRef = useRef<SVGSVGElement | null>(null);
  const draggingRef = useRef(false);
  const [isDragging, setIsDragging] = useState(false);
  const [baseline, setBaseline] = useState<WhatIfBaseline | null>(null);
  const [selectedObjId, setSelectedObjId] = useState<string>("");
  const [draft, setDraft] = useState<WhatIfDraft | null>(null);
  const [whatif, setWhatif] = useState<WhatIfResponse | null>(null);

  useEffect(() => {
    if (!jobId) return;
    let alive = true;
    getWhatIf(jobId).then((b) => {
      if (!alive || !b) return;
      setBaseline(b);
      setSelectedObjId(b.object_id);
      setDraft(draftFromBaseline(b));
      setWhatif(null);
    });
    return () => { alive = false; };
  }, [jobId]);

  const curBaseline = baseline && baseline.job_id === jobId ? baseline : null;
  const baseObj = useMemo(
    () => (curBaseline ? (curBaseline.objects.find((o) => o.object_id === selectedObjId) ?? curBaseline) : null),
    [curBaseline, selectedObjId],
  );
  const curDraft = curBaseline ? draft : null;
  const req = useMemo(() => (baseObj && curDraft ? buildWhatIfRequest(baseObj.object_id, curDraft, baseObj) : null), [baseObj, curDraft]);
  const reqJson = req ? JSON.stringify(req) : "";

  useEffect(() => {
    if (!jobId || !reqJson) return;
    const parsed = JSON.parse(reqJson);
    const ctrl = new AbortController();
    const timer = window.setTimeout(() => {
      postWhatIf(jobId, parsed, ctrl.signal).then((res) => {
        if (res && !ctrl.signal.aborted) setWhatif(res);
      });
    }, 45);
    return () => { window.clearTimeout(timer); ctrl.abort(); };
  }, [jobId, reqJson]);

  const activeWhatIf = req && whatif && whatif.job_id === jobId ? whatif : null;
  const cfPred = activeWhatIf?.predicted ?? null;
  const cfMain = activeWhatIf && activeWhatIf.object_id !== curBaseline?.object_id ? activeWhatIf.main_predicted : cfPred;

  // Foresight's position at the moment of the last real point: the endpoint error is measured there.
  const atEnd = lastReal ? pred.reduce((a, b) => (Math.abs(b[0] - lastReal[0]) < Math.abs(a[0] - lastReal[0]) ? b : a), pred[0]) : null;
  const cfAtEnd = lastReal && cfMain?.length ? cfMain.reduce((a, b) => (Math.abs(b[0] - lastReal[0]) < Math.abs(a[0] - lastReal[0]) ? b : a), cfMain[0]) : null;
  const line = (pts: number[][]) => pts.map((p) => `${p[1].toFixed(1)},${p[2].toFixed(1)}`).join(" ");
  const handlePos: [number, number] | null = curDraft ? curDraft.startPx : pred[0] ? [pred[0][1], pred[0][2]] : null;

  const updatePointerPos = (clientX: number, clientY: number) => {
    const rect = svgRef.current?.getBoundingClientRect();
    const pt = rect ? clientToScenePx(clientX, clientY, rect, [W, H]) : null;
    if (pt) setDraft((d) => (d ? { ...d, startPx: pt } : d));
  };

  const vxMax = baseObj ? velocityBound(baseObj.vx0) : 4;
  const vyMax = baseObj ? velocityBound(baseObj.vy0) : 4;
  const fmtSigned = (v: number) => `${v >= 0 ? "+" : ""}${v.toFixed(2)} m/s`;

  const selectObject = (oid: string) => {
    setSelectedObjId(oid);
    const target = curBaseline?.objects.find((o) => o.object_id === oid) ?? curBaseline;
    if (target) setDraft(draftFromBaseline(target));
    setWhatif(null);
  };

  return (
    <figure className="m-0 space-y-4">
      <div>
        <div className="monitor">
          <div className="relative" style={{ aspectRatio: `${W} / ${H}` }}>
            {frameUrl && <img src={apiUrl(frameUrl)} alt="" className="absolute inset-0 h-full w-full object-cover" />}
            <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} className="absolute inset-0 h-full w-full select-none" role="img"
                 aria-label="Paths: what the model saw, what Foresight predicted, counterfactual what-if path, and what really happened">
              <rect width={W} height={H} fill="rgb(13 14 17 / 0.28)" />
              {pred.length > 1 && <polyline points={line(pred)} fill="none" stroke="var(--color-cobalt)" strokeWidth={7 * s}
                                            strokeLinecap="round" strokeLinejoin="round" opacity={cfPred ? 0.65 : 1} />}
              {cfPred && cfPred.length > 1 && (
                <polyline data-testid="whatif-polyline" points={line(cfPred)} fill="none" stroke="#34d399" strokeWidth={7.5 * s}
                          strokeDasharray={`${16 * s} ${10 * s}`} strokeLinecap="round" strokeLinejoin="round" />
              )}
              {real.map((p, i) => <circle key={`r${i}`} cx={p[1]} cy={p[2]} r={6 * s} fill="#f2f2ef" />)}
              {paths.seen.map((p, i) => <circle key={`s${i}`} cx={p[1]} cy={p[2]} r={7 * s} fill="none" stroke="#f2f2ef"
                                                strokeWidth={2.5 * s} opacity={0.85} />)}
              {lastReal && atEnd && (
                <g>
                  <line x1={(cfAtEnd ?? atEnd)[1]} y1={(cfAtEnd ?? atEnd)[2]} x2={lastReal[1]} y2={lastReal[2]} stroke="#f2f2ef"
                        strokeWidth={3 * s} strokeDasharray={`${10 * s} ${8 * s}`} />
                  <circle cx={atEnd[1]} cy={atEnd[2]} r={16 * s} fill="none" stroke="var(--color-cobalt)" strokeWidth={6 * s} opacity={cfAtEnd ? 0.65 : 1} />
                  {cfAtEnd && <circle cx={cfAtEnd[1]} cy={cfAtEnd[2]} r={17 * s} fill="none" stroke="#34d399" strokeWidth={6 * s} />}
                  <circle cx={lastReal[1]} cy={lastReal[2]} r={16 * s} fill="none" stroke="#f2f2ef" strokeWidth={6 * s} />
                </g>
              )}
              {curBaseline && curDraft && handlePos && (
                <g data-testid="whatif-handle" role="slider" tabIndex={0}
                   aria-label="Drag start position at 3.0 seconds or nudge with arrow keys"
                   aria-valuemin={0} aria-valuemax={W} aria-valuenow={Math.round(handlePos[0])}
                   aria-valuetext={`${Math.round(handlePos[0])}, ${Math.round(handlePos[1])} px`}
                   style={{ cursor: isDragging ? "grabbing" : "grab", touchAction: "none" }}
                   onPointerDown={(e) => {
                     e.preventDefault();
                     e.currentTarget.setPointerCapture(e.pointerId);
                     draggingRef.current = true;
                     setIsDragging(true);
                     updatePointerPos(e.clientX, e.clientY);
                   }}
                   onPointerMove={(e) => { if (draggingRef.current) updatePointerPos(e.clientX, e.clientY); }}
                   onPointerUp={(e) => {
                     if (!draggingRef.current) return;
                     draggingRef.current = false;
                     setIsDragging(false);
                     if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId);
                   }}
                   onPointerCancel={(e) => {
                     draggingRef.current = false;
                     setIsDragging(false);
                     if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId);
                   }}
                   onKeyDown={(e) => {
                     const step = (e.shiftKey ? 4 : 1) * Math.max(4, Math.round(W * 0.01));
                     const deltas: Record<string, [number, number]> = {
                       ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step],
                     };
                     const d = deltas[e.key];
                     if (!d) return;
                     e.preventDefault();
                     setDraft((cur) => (cur ? {
                       ...cur,
                       startPx: [Math.max(0, Math.min(W, cur.startPx[0] + d[0])), Math.max(0, Math.min(H, cur.startPx[1] + d[1]))],
                     } : cur));
                   }}>
                  <circle cx={handlePos[0]} cy={handlePos[1]} r={28 * s} fill="transparent" />
                  <circle cx={handlePos[0]} cy={handlePos[1]} r={14 * s} fill="#10b981" fillOpacity={0.32} stroke="#34d399" strokeWidth={4.5 * s} />
                  <circle cx={handlePos[0]} cy={handlePos[1]} r={5 * s} fill="#f2f2ef" />
                </g>
              )}
            </svg>
          </div>
          <figcaption className="tag">
            <span className="dot" style={{ background: cfPred ? "#34d399" : "var(--color-cobalt)" }} />
            {cfPred ? "Where it goes (what-if)" : "Where it went"}
          </figcaption>
        </div>
        <div className="mt-3 flex flex-wrap gap-x-6 gap-y-1 text-base text-muted">
          <span><span className="mr-2 inline-block size-3 rounded-full border-2 border-ink align-middle" />seen in the first 3 s</span>
          <span><span className="mr-2 inline-block h-1 w-6 rounded bg-cobalt align-middle" />Foresight's next 5 s</span>
          {cfPred && <span className="text-good font-medium"><span className="mr-2 inline-block h-1 w-6 rounded border-b-2 border-dashed border-good align-middle" />what-if rollout</span>}
          <span><span className="mr-2 inline-block size-3 rounded-full bg-ink align-middle" />what really happened</span>
          {endpointCm !== null && !activeWhatIf && <span className="text-ink">rings: {fmtCm(endpointCm)} apart at the last real sighting</span>}
          {activeWhatIf && activeWhatIf.endpoint_error_cm !== null && (
            <span className="text-ink">what-if endpoint: <strong className="text-good">{fmtCm(activeWhatIf.endpoint_error_cm)}</strong>{endpointCm !== null ? ` (fitted: ${fmtCm(endpointCm)})` : ""}</span>
          )}
        </div>
      </div>

      {curBaseline && curDraft && baseObj && (
        <section aria-labelledby="whatif-heading" className="rounded-xl border border-line bg-card p-5 space-y-4">
          <div className="flex flex-wrap items-baseline justify-between gap-3">
            <div>
              <h3 id="whatif-heading" className="text-xl font-bold">What if…</h3>
              <p className="text-sm text-muted">
                Drag the green start handle on the frame or adjust 3.0 s velocity, friction and bounce — simulated locally in pymunk (0 Gemini calls).
              </p>
            </div>
            <div className="flex items-center gap-3">
              {activeWhatIf && (
                <span data-testid="whatif-latency" className="rounded-full bg-paper px-3 py-1 font-mono text-xs text-muted">
                  {activeWhatIf.latency_ms.toFixed(1)} ms
                </span>
              )}
              {req && (
                <button type="button" onClick={() => { setDraft(draftFromBaseline(baseObj)); setWhatif(null); }}
                        className="rounded-full border border-ink px-3.5 py-1 text-sm font-medium hover:bg-ink hover:text-paper">
                  Reset to fitted
                </button>
              )}
            </div>
          </div>

          {curBaseline.objects.length > 1 && (
            <div className="flex items-center gap-3 text-sm">
              <label htmlFor="whatif-obj" className="font-medium text-muted">Object:</label>
              <select id="whatif-obj" value={baseObj.object_id} onChange={(e) => selectObject(e.target.value)}
                      className="rounded-lg border border-line bg-paper px-3 py-1 font-medium text-ink">
                {curBaseline.objects.map((o) => <option key={o.object_id} value={o.object_id}>{o.name} ({o.object_id})</option>)}
              </select>
            </div>
          )}

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-3 text-sm">
            <WhatIfSlider label="Horizontal velocity vx (+ right)" ariaLabel="Horizontal velocity vx"
                          display={fmtSigned(curDraft.vx0)} min={-vxMax} max={vxMax} step={0.05} value={curDraft.vx0}
                          onChange={(vx0) => setDraft({ ...curDraft, vx0 })} />
            <WhatIfSlider label="Vertical velocity vy (+ up)" ariaLabel="Vertical velocity vy"
                          display={fmtSigned(curDraft.vy0)} min={-vyMax} max={vyMax} step={0.05} value={curDraft.vy0}
                          onChange={(vy0) => setDraft({ ...curDraft, vy0 })} />
            <WhatIfSlider label="Friction" ariaLabel="Friction"
                          display={curDraft.friction.toFixed(2)} min={0} max={1.5} step={0.02} value={curDraft.friction}
                          onChange={(friction) => setDraft({ ...curDraft, friction })} />
            <WhatIfSlider label="Bounce (restitution)" ariaLabel="Bounce"
                          display={curDraft.bounce.toFixed(2)} min={0} max={0.98} step={0.02} value={curDraft.bounce}
                          onChange={(bounce) => setDraft({ ...curDraft, bounce })} />
          </div>

          {activeWhatIf?.implausible_reason && (
            <p role="status" className="text-sm text-bad">Energy bound warning: {activeWhatIf.implausible_reason}</p>
          )}
        </section>
      )}
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
