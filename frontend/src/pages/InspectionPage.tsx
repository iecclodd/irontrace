import { Download, Play, RotateCcw, ScanSearch } from 'lucide-react'
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

  const reportEntries = report ? Object.entries(report).filter(([key, value]) => key !== 'session' && ['string', 'number', 'boolean'].includes(typeof value)).slice(0, 8) : []
  const progress = job ? Math.max(0, Math.min(1, job.progress)) : 0

  return <main className="page inspection-page">
    <section className="card toolbar" aria-label="Session configuration">
      <div className="toolbar-title">
        <strong>{session ? `Session ${session.id.slice(0, 8)}` : 'New session'}{replayMode ? <span className="tag">Recorded replay</span> : session?.status === 'active' ? <span className="tag accent">Live</span> : null}</strong>
        <span>{replayMode ? 'Read-only. Controls are disabled.' : 'Costs are simulated units.'}</span>
      </div>
      <label className="field">Budget<input type="number" min="0" max="100" step="1" value={config.budget} onChange={(event) => setConfig((current) => ({ ...current, budget: Number(event.target.value) }))} disabled={replayMode} /></label>
      <label className="field">Cost weight<input type="number" min="0" step="0.005" value={config.lambdaCost} onChange={(event) => setConfig((current) => ({ ...current, lambdaCost: Number(event.target.value) }))} disabled={replayMode} /></label>
      <label className="field wide">Policy<select value={config.policy} onChange={(event) => setConfig((current) => ({ ...current, policy: event.target.value }))} disabled={replayMode}><option value="default">Selected on development data (recommended)</option><option value="information">Information</option><option value="value" disabled={!health?.utility_ready}>Empirical value</option><option value="value_no_residual" disabled={!health?.utility_ready}>Value without residual</option><option value="entropy_drop">Entropy drop</option><option value="raw_kl">Raw KL</option><option value="random">Random</option><option value="static">Static</option><option value="prior">Prior only</option><option value="all">All panels</option></select></label>
      <div className="toolbar-end">
        {session && !replayMode ? <span className="note">Changes apply to a new session and never rewrite past charges.</span> : null}
        <button className="btn primary" onClick={onCreate} disabled={!ready || busy || replayMode}>{session ? <RotateCcw /> : <Play />} {session ? 'Start new session' : 'Start live session'}</button>
      </div>
    </section>

    <div className="console-grid">
      <section className="card panels-column">
        <div className="card-head"><div><h2>Sensor panels</h2><p>Readings unlock when a panel is acquired.</p></div><span className="count">{session?.observed_groups.length ?? 0} of {manifest?.panels.length ?? 0}</span></div>
        {manifest ? <SensorPanels panels={manifest.panels} observedGroups={session?.observed_groups ?? []} visibleValues={session?.visible_values ?? {}} costs={session?.costs ?? config.costs} /> : <div className="empty"><span>Waiting for the public panel manifest…</span></div>}
      </section>

      <section className="card prediction-column">
        <div className="card-head"><div><h2>Leakage probabilities</h2><p>Internal pump leakage, from the current model output.</p></div></div>
        <PredictionBars labels={labels} probabilities={session?.probabilities ?? null} />
        <div className="stat-grid">
          <div><span>Spent</span><strong>{session ? `${session.spent} / ${session.budget}` : '—'}</strong><small>simulated units</small></div>
          <div><span>Remaining</span><strong>{session?.remaining_budget ?? config.budget}</strong><small>simulated units</small></div>
          <div><span>Panels acquired</span><strong>{session?.observed_groups.length ?? 0}</strong><small>including initial</small></div>
          <div><span>Inference time</span><strong>{session && Number.isFinite(session.inference_ms) ? `${session.inference_ms.toFixed(0)} ms` : '—'}</strong><small>backend measured</small></div>
        </div>
        <dl className="meta-list">
          <div><dt>Model reference</dt><dd className={session?.model_ref ? 'mono' : ''} title={session?.model_ref || undefined}>{session?.model_ref || (health?.model_ready ? 'Ready; session not started' : 'TabPFN 3.5 not yet ready')}</dd></div>
          {session ? <div><dt>Resolved policy</dt><dd>{session.policy.replaceAll('_', ' ')}</dd></div> : null}
          {session?.selection_note ? <div><dt>Selection</dt><dd title={session.selection_note}>{session.selection_note}</dd></div> : null}
        </dl>
        {report ? <div className="report"><h3>Retrospective outcome</h3><dl>{reportEntries.map(([key, value]) => <div key={key}><dt>{key.replaceAll('_', ' ')}</dt><dd>{String(value)}</dd></div>)}</dl></div> : null}
      </section>

      <section className="card action-column">
        <div className="card-head"><div><h2>Ranked checks</h2><p>Click a check to see its audit details.</p></div><button className="btn ghost" disabled={!auditCandidate} onClick={() => setAuditOpen(true)}><ScanSearch /> Audit</button></div>
        <CandidateTable candidates={candidates} panels={manifest?.panels ?? []} selectedAction={selected} activeCandidate={activeCandidate} onSelect={inspectCandidate} />
        <div className="actions">
          <button className="btn primary block" onClick={() => selected && onAcquire(selected)} disabled={!active || busy || !selected}>Run recommended check</button>
          <div className="actions-row"><button className="btn" onClick={onRecommend} disabled={!active || busy}>Recommend</button><button className="btn" onClick={onRun} disabled={!active || busy}>Autopilot</button><button className="btn danger" onClick={onStop} disabled={!active || (busy && !job)}>Stop</button></div>
        </div>
        {job ? <div className="inline-status job" role="status"><div><span>{job.status === 'queued' ? 'Queued' : job.status === 'running' ? 'Numerical job running' : job.status === 'completed' ? 'Completed' : job.status}</span><span>{Math.round(progress * 100)}%</span></div><div className="job-bar"><span style={{ width: `${progress * 100}%` }} /></div></div> : null}
        {replayMode ? <div className="inline-status">Recorded replay is immutable. Live recommendations, budget changes, and branch actions are disabled.</div> : null}
        {session?.stop_reason ? <div className="inline-status"><strong>Session stopped</strong> · {session.stop_reason.replaceAll('_', ' ')}</div> : null}
      </section>
    </div>

    <section className="card trace-section">
      <div className="card-head"><div><h2>Decision trace</h2><p>Sanitized, append-only record of this session.</p></div><button className="btn" disabled={events.length === 0} onClick={onExport}><Download /> Export</button></div>
      <TraceTimeline events={events} />
    </section>
    {auditOpen ? <button className="drawer-backdrop" onClick={() => setAuditOpen(false)} aria-label="Close acquisition audit" /> : null}
    <AuditDrawer candidate={auditCandidate} candidates={candidates} panels={manifest?.panels ?? []} open={auditOpen} onClose={() => setAuditOpen(false)} />
  </main>
}
