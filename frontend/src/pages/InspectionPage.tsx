import { Activity, CircleStop, Gauge, ListChecks, Play, RotateCcw, ScanSearch, Sparkles, SquareActivity } from 'lucide-react'
import { useState, type Dispatch, type SetStateAction } from 'react'
import { AuditDrawer } from '../components/AuditDrawer'
import { CandidateTable } from '../components/CandidateTable'
import { PredictionBars } from '../components/PredictionBars'
import { SensorPanels } from '../components/SensorPanels'
import { TraceTimeline } from '../components/TraceTimeline'
import type { Candidate, Health, Job, Manifest, Session, TraceEvent } from '../types'

export type SessionConfig = { budget: number; lambdaCost: number; policy: string; costs: Record<string, number> }

export function InspectionPage({ health, manifest, session, events, report, replayMode, config, setConfig, busy, job, onCreate, onRecommend, onAcquire, onRun, onStop, onExport }: {
  health: Health | null
  manifest: Manifest | null
  session: Session | null
  events: TraceEvent[]
  report: Record<string, unknown> | null
  replayMode: boolean
  config: SessionConfig
  setConfig: Dispatch<SetStateAction<SessionConfig>>
  busy: boolean
  job: Job | null
  onCreate: () => void
  onRecommend: () => void
  onAcquire: (group: string) => void
  onRun: () => void
  onStop: () => void
  onExport: () => void
}) {
  const [auditOpen, setAuditOpen] = useState(false)
  const [activeCandidate, setActiveCandidate] = useState<string | null>(null)
  const candidates = session?.recommendation?.candidates ?? []
  const selected = session?.recommendation?.selected_action ?? null
  const auditCandidate: Candidate | null = candidates.find((candidate) => candidate.group_id === activeCandidate) ?? candidates[0] ?? null
  const ready = health?.status === 'ready' && health.model_ready && health.data_ready
  const active = session?.status === 'active' && !replayMode
  const labels = manifest?.labels ?? []

  function inspectCandidate(id: string) { setActiveCandidate(id); setAuditOpen(true) }

  return <main className="page inspection-page">
    <section className="configuration-bar" aria-label="Session configuration">
      <div className="config-title"><span className="eyebrow">Live scenario</span><strong>{session ? `Session ${session.id.slice(0, 8)}` : 'Configure a new session'}</strong></div>
      <label>Budget<input type="number" min="0" max="100" step="1" value={config.budget} onChange={(event) => setConfig((current) => ({ ...current, budget: Number(event.target.value) }))} disabled={replayMode} /></label>
      <label>Cost weight<input type="number" min="0" step="0.005" value={config.lambdaCost} onChange={(event) => setConfig((current) => ({ ...current, lambdaCost: Number(event.target.value) }))} disabled={replayMode} /></label>
      <label>Policy<select value={config.policy} onChange={(event) => setConfig((current) => ({ ...current, policy: event.target.value }))} disabled={replayMode}><option value="information">Information</option><option value="value">Empirical value</option><option value="value_no_residual">Value without residual</option><option value="entropy_drop">Entropy drop</option><option value="raw_kl">Raw KL</option><option value="random">Random</option><option value="static">Static</option><option value="prior">Prior only</option><option value="all">All panels</option></select></label>
      <button className="primary-button" onClick={onCreate} disabled={!ready || busy || replayMode}><RotateCcw size={16} /> {session ? 'Start new session' : 'Start live session'}</button>
      {session && !replayMode ? <span className="config-note">Changes apply to a new session and never rewrite past charges.</span> : null}
    </section>

    <div className="console-grid">
      <section className="surface panels-column"><div className="section-heading"><div><span className="eyebrow">Available observations</span><h2>Sensor panels</h2></div><span className="count-chip">{session?.observed_groups.length ?? 0}/{manifest?.panels.length ?? 0}</span></div>
        {manifest ? <SensorPanels panels={manifest.panels} observedGroups={session?.observed_groups ?? []} visibleValues={session?.visible_values ?? {}} costs={config.costs} /> : <div className="skeleton-block">Waiting for the public panel manifest…</div>}
      </section>

      <section className="surface prediction-column"><div className="section-heading"><div><span className="eyebrow">Internal pump leakage target</span><h2>Model probabilities</h2></div><Activity className="section-icon" /></div>
        <PredictionBars labels={labels} probabilities={session?.probabilities ?? null} />
        <div className="stat-grid">
          <div><span>Simulated cost</span><strong>{session ? `${session.spent} / ${session.budget}` : '—'}</strong><small>units spent</small></div>
          <div><span>Panels acquired</span><strong>{session?.observed_groups.length ?? 0}</strong><small>including initial</small></div>
          <div><span>Inference time</span><strong>{session && Number.isFinite(session.inference_ms) ? `${session.inference_ms.toFixed(0)} ms` : '—'}</strong><small>backend measured</small></div>
          <div><span>Remaining budget</span><strong>{session?.remaining_budget ?? config.budget}</strong><small>simulated units</small></div>
        </div>
        <div className="model-strip"><Gauge size={17} /><div><span>Model reference</span><strong>{session?.model_ref || (health?.model_ready ? 'Ready; session not started' : 'TabPFN 3.5 not yet ready')}</strong></div></div>
        {report ? <div className="report-card"><span className="eyebrow">Completed report</span><h3>Retrospective outcome</h3><dl>{Object.entries(report).filter(([key, value]) => key !== 'session' && ['string', 'number', 'boolean'].includes(typeof value)).slice(0, 8).map(([key, value]) => <div key={key}><dt>{key.replaceAll('_', ' ')}</dt><dd>{String(value)}</dd></div>)}</dl></div> : null}
      </section>

      <section className="surface action-column"><div className="section-heading"><div><span className="eyebrow">Current decision</span><h2>Ranked checks</h2></div><button className="text-button" disabled={!auditCandidate} onClick={() => setAuditOpen(true)}><ScanSearch size={15} /> Audit</button></div>
        <CandidateTable candidates={candidates} panels={manifest?.panels ?? []} selectedAction={selected} activeCandidate={activeCandidate} onSelect={inspectCandidate} />
        <div className="action-stack">
          <button className="primary-button wide" onClick={() => selected && onAcquire(selected)} disabled={!active || busy || !selected}><Play size={16} /> Run recommended check</button>
          <div className="action-pair"><button className="secondary-button" onClick={onRecommend} disabled={!active || busy}><ListChecks size={16} /> Recommend</button><button className="secondary-button" onClick={onRun} disabled={!active || busy}><Sparkles size={16} /> Autopilot</button><button className="danger-button" onClick={onStop} disabled={!active || (busy && !job)}><CircleStop size={16} /> Stop</button></div>
        </div>
        {job ? <div className="job-progress" role="status"><div><SquareActivity size={16} /><span>{job.status === 'queued' ? 'Queued' : job.status === 'running' ? 'Numerical job running' : job.status}</span><strong>{Math.round(Math.max(0, Math.min(1, job.progress)) * 100)}%</strong></div><progress max="1" value={Math.max(0, Math.min(1, job.progress))} /></div> : null}
        {replayMode ? <div className="readonly-note">Recorded replay is immutable. Live recommendations, budget changes, and branch actions are disabled.</div> : null}
        {session?.stop_reason ? <div className="stop-note"><strong>Session stopped</strong><span>{session.stop_reason.replaceAll('_', ' ')}</span></div> : null}
      </section>
    </div>

    <section className="surface trace-section"><div className="section-heading"><div><span className="eyebrow">Sanitized append-only record</span><h2>Decision trace</h2></div><button className="secondary-button compact" disabled={events.length === 0} onClick={onExport}>Export trace</button></div><TraceTimeline events={events} /></section>
    <AuditDrawer candidate={auditCandidate} panels={manifest?.panels ?? []} open={auditOpen} onClose={() => setAuditOpen(false)} />
    {auditOpen ? <button className="drawer-backdrop" onClick={() => setAuditOpen(false)} aria-label="Close acquisition audit" /> : null}
  </main>
}
