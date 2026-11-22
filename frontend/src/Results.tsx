// How well it predicts: the eval and physics auditor, read straight from eval/*.json via the API.
import { useEffect, useState } from "react";
import { fmtCm, getAuditEval, getEval } from "./api";

const NAMES: Record<string, string> = {
  foresight: "Foresight", foresight_carbon: "Foresight, Carbon reads the scene", foresight_neon: "Foresight, Neon reads the scene",
  const_velocity: "Constant velocity",
  omni: "Gemini Omni 1.2 Flash", genie: "Genie 4", static: "Last frame held",
};
const ROLE: Record<string, string> = {
  foresight: "physics", foresight_carbon: "physics", foresight_neon: "physics", const_velocity: "straight line, no physics",
  omni: "video model", genie: "world model", static: "nothing moves",
};
const COLOR: Record<string, string> = {
  foresight: "var(--color-cobalt)", foresight_carbon: "#5b7ff0", foresight_neon: "#8fa6f5", const_velocity: "#7d8597",
  omni: "var(--color-amber)", genie: "var(--color-plum)", static: "#c4c6cc",
};

type Row = { key: string; score: number; endpoint: number | null; extra?: string };

function Bars({ rows, max = 100 }: { rows: Row[]; max?: number }) {
  return (
    <ol className="space-y-3">
      {rows.map((r) => (
        <li key={r.key} className="grid grid-cols-[minmax(9rem,15rem)_1fr_4.5rem] items-center gap-4">
          <div>
            <div className="font-semibold leading-tight">{NAMES[r.key] ?? r.key}</div>
            <div className="text-sm text-muted">{ROLE[r.key]}</div>
          </div>
          <div className="h-7 rounded bg-line/60" aria-hidden>
            <div className="h-full rounded" style={{ width: `${(r.score / max) * 100}%`, background: COLOR[r.key] ?? "#7d8597" }} />
          </div>
          <div className="display text-3xl font-extrabold text-right">{r.score.toFixed(1)}</div>
        </li>
      ))}
    </ol>
  );
}

function AuditTable({ systems }: { systems: Record<string, any> }) {
  const entries = Object.entries(systems).sort(([, a], [, b]) => b.audit_score_pct - a.audit_score_pct);
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-base">
        <thead>
          <tr className="text-left text-muted">
            <th className="py-2 pr-4 font-medium">System</th>
            <th className="py-2 px-2 text-right font-medium">Audit score</th>
            <th className="py-2 px-2 text-right font-medium">Physics-IQ</th>
            <th className="py-2 px-2 text-right font-medium">Passed</th>
            <th className="py-2 px-2 text-right font-medium">Penalized ep.</th>
            <th className="py-2 px-2 text-right font-medium">Stopped ep.</th>
            <th className="py-2 px-2 text-right font-medium">Vanished</th>
            <th className="py-2 pl-2 text-right font-medium">Duplicated</th>
          </tr>
        </thead>
        <tbody>
          {entries.map(([k, s]) => (
            <tr key={k} className="border-t border-line">
              <td className="py-2 pr-4 font-medium">{NAMES[k] ?? k}</td>
              <td className="py-2 px-2 text-right font-mono font-bold text-ink">{s.audit_score_pct.toFixed(1)}</td>
              <td className="py-2 px-2 text-right font-mono text-muted">{s.physics_iq_verified_pct?.toFixed(1) ?? "—"}</td>
              <td className="py-2 px-2 text-right font-mono">{s.n_passed}/{s.n_clips} ({s.pass_rate_pct.toFixed(0)}%)</td>
              <td className="py-2 px-2 text-right font-mono font-semibold">{fmtCm(s.penalized_endpoint_cm_median)}</td>
              <td className="py-2 px-2 text-right font-mono text-muted">{fmtCm(s.reported_endpoint_cm_median)}</td>
              <td className={`py-2 px-2 text-right font-mono ${s.vanished_count > 0 ? "text-bad font-semibold" : "text-muted"}`}>{s.vanished_count}</td>
              <td className={`py-2 pl-2 text-right font-mono ${s.duplicated_count > 0 ? "text-bad font-semibold" : "text-muted"}`}>{s.duplicated_count}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Results() {
  const [main, setMain] = useState<any | null | undefined>(undefined);
  const [wide, setWide] = useState<any | null | undefined>(undefined);
  const [auditMain, setAuditMain] = useState<any | null | undefined>(undefined);
  const [auditWide, setAuditWide] = useState<any | null | undefined>(undefined);

  useEffect(() => {
    getEval("main").then(setMain);
    getEval("wide").then(setWide);
    getAuditEval("main").then(setAuditMain);
    getAuditEval("wide").then(setAuditWide);
  }, []);

  const mainRows: Row[] = main ? Object.entries<any>(main.systems)
    .map(([k, v]) => ({ key: k, score: v.physics_iq_verified_pct, endpoint: v.endpoint_error_cm_median ?? null }))
    .sort((a, b) => b.score - a.score) : [];
  const wideRows: Row[] = wide ? Object.entries<any>(wide.systems).filter(([, v]) => v.pooled_verified_pct != null)
    .map(([k, v]) => ({ key: k, score: v.pooled_verified_pct, endpoint: v.endpoint_error_cm_median ?? null }))
    .sort((a, b) => b.score - a.score) : [];
  const auditRows: Row[] = auditMain ? Object.entries<any>(auditMain.systems)
    .map(([k, v]) => ({ key: k, score: v.audit_score_pct, endpoint: v.penalized_endpoint_cm_median ?? null }))
    .sort((a, b) => b.score - a.score) : [];
  const ends = mainRows.filter((r) => r.endpoint !== null);
  const systems = mainRows.map((r) => r.key);

  return (
    <div className="min-h-full">
      <header className="flex flex-wrap items-baseline justify-between gap-x-6 px-4 sm:px-10 pt-8 pb-4">
        <a href="/" className="display text-3xl font-extrabold tracking-tight">Foresight</a>
        <nav className="text-lg"><a href="/" className="font-medium underline decoration-line decoration-2 underline-offset-4 hover:decoration-ink">Run a prediction</a></nav>
      </header>
      <main className="px-4 sm:px-10 pb-16 max-w-[1200px] space-y-14">
        <section className="pt-4">
          <h1 className="text-[clamp(2.5rem,6vw,5rem)] font-extrabold">How well it predicts</h1>
          <p className="mt-5 max-w-[46rem] text-xl text-muted">
            Each system sees the first 3 seconds of a real Physics-IQ clip and produces the next 5. Three measures: the
            benchmark's pixel-change score, stopped-track endpoint error, and the physics auditor (object permanence,
            penalized position error, and energy bounds).
          </p>
        </section>

        {main === undefined && <p className="text-muted">Loading the results…</p>}
        {main === null && <p className="text-bad">No eval results on this server yet (eval/results.json).</p>}

        {main && (
          <section className="space-y-8" aria-labelledby="main-eval">
            <div>
              <h2 id="main-eval" className="text-3xl font-bold">{main.n_clips} clips, side camera</h2>
              <p className="mt-1 text-base text-muted">Physics-IQ score, higher is better. Official metric code, {main.gt_source === "hf" ? "Physics-IQ Verified" : "original"} ground truth.</p>
            </div>
            <Bars rows={mainRows} />
            {ends.length > 0 && (
              <div>
                <h3 className="text-xl font-semibold">Where the object ends up</h3>
                <p className="mt-1 text-base text-muted">Median endpoint error over the clips, lower is better. Measured where a system tracks an object.</p>
                <div className="mt-4 flex flex-wrap gap-x-12 gap-y-4">
                  {ends.map((r) => (
                    <div key={r.key}>
                      <div className="display text-5xl font-extrabold" style={{ color: COLOR[r.key] }}>{fmtCm(r.endpoint)}</div>
                      <div className="text-base text-muted">{NAMES[r.key]}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
            <details className="rounded-xl border border-line bg-card p-5">
              <summary className="cursor-pointer text-lg font-semibold">Every clip</summary>
              <div className="mt-4 overflow-x-auto">
                <table className="w-full text-base">
                  <thead>
                    <tr className="text-left text-muted">
                      <th className="py-2 pr-4 font-medium">Clip</th>
                      {systems.map((k) => <th key={k} className="py-2 px-2 font-medium text-right">{NAMES[k]}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {main.scenes.map((sc: string) => {
                      const vals = systems.map((k) => main.systems[k].per_scenario_verified_pct?.[sc] as number | undefined);
                      const best = Math.max(...vals.filter((v): v is number => v !== undefined));
                      return (
                        <tr key={sc} className="border-t border-line">
                          <td className="py-2 pr-4">{sc.replaceAll("-", " ")}</td>
                          {vals.map((v, i) => <td key={systems[i]} className={`py-2 px-2 text-right font-mono ${v === best ? "font-bold text-ink" : "text-muted"}`}>{v?.toFixed(1) ?? "—"}</td>)}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </details>
            <p className="text-sm text-muted">
              From eval/results.json, generated {String(main.generated_at).slice(0, 10)}, commit {main.git_commit}, clean run {main.clean_run ? "yes" : "no"}.
              Rows can come from different clean runs; the video models' rows do not depend on Foresight's code.
            </p>
          </section>
        )}

        {auditMain && (
          <section data-testid="physics-auditor-section" className="space-y-8" aria-labelledby="physics-auditor">
            <div>
              <h2 id="physics-auditor" className="text-3xl font-bold">Physics auditor</h2>
              <p className="mt-1 max-w-[46rem] text-base text-muted">
                Pixel masks reward an empty background after a straight-line sprite leaves the frame early, and stopped-track
                endpoint error stops measuring when a video model drops the object. <code className="font-mono text-sm">foresight audit</code> checks
                occlusion-aware object permanence (vanishes and duplicate respawns), penalized fixed-horizon position error, and
                free-fall energy bounds across all {auditMain.n_videos_audited} main-eval and {auditWide?.n_videos_audited ?? 300} wide-eval predictions.
              </p>
            </div>
            <Bars rows={auditRows} />
            <div className="space-y-3">
              <h3 className="text-xl font-semibold">15 clips, side camera ({auditMain.n_videos_audited} videos audited)</h3>
              <AuditTable systems={auditMain.systems} />
            </div>
            {auditWide && (
              <div className="space-y-3">
                <h3 className="text-xl font-semibold">20 scenes, all three cameras ({auditWide.n_videos_audited} videos audited)</h3>
                <AuditTable systems={auditWide.systems} />
              </div>
            )}
            <p className="text-sm text-muted">
              From eval/audit_results.json{auditWide ? " and eval/audit_wide.json" : ""}, generated {String(auditMain.generated_at).slice(0, 10)},
              commit {auditMain.git_commit}, {auditMain.gemini_calls} Gemini calls.
            </p>
          </section>
        )}

        {wideRows.length > 0 && (
          <section className="space-y-8" aria-labelledby="wide-eval">
            <div>
              <h2 id="wide-eval" className="text-3xl font-bold">{wide.scenes.length} scenes, all three cameras</h2>
              <p className="mt-1 text-base text-muted">
                The same systems on the left, center and right cameras, and five more scenes chosen by a rule fixed before
                the run. Most of these clips were never used while building Foresight.
              </p>
            </div>
            <Bars rows={wideRows} />
            <div className="overflow-x-auto">
              <table className="w-full max-w-[44rem] text-base">
                <thead><tr className="text-left text-muted"><th className="py-2 pr-4 font-medium">System</th>
                  {wide.views.map((v: string) => <th key={v} className="py-2 px-2 text-right font-medium">{v}</th>)}
                  <th className="py-2 px-2 text-right font-medium">endpoint</th></tr></thead>
                <tbody>
                  {wideRows.map((r) => (
                    <tr key={r.key} className="border-t border-line">
                      <td className="py-2 pr-4">{NAMES[r.key]}</td>
                      {wide.views.map((v: string) => <td key={v} className="py-2 px-2 text-right font-mono">{wide.systems[r.key].per_view_pct?.[v]?.toFixed(1) ?? "—"}</td>)}
                      <td className="py-2 px-2 text-right font-mono">{r.endpoint !== null ? fmtCm(r.endpoint) : "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="text-sm text-muted">From eval/results_wide.json, generated {String(wide.generated_at).slice(0, 10)}, commit {wide.git_commit}.</p>
          </section>
        )}
      </main>
    </div>
  );
}
