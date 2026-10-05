import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AlertTriangle, Archive, Beaker, CheckCircle2, ChevronDown, DatabaseZap, FlaskConical, LoaderCircle, Radio, RefreshCw, ShieldCheck } from 'lucide-react'
import { api, ApiError, errorMessage } from './api'
import { StatusBadge } from './components/StatusBadge'
import { BenchmarkPage } from './pages/BenchmarkPage'
import { InspectionPage, type SessionConfig } from './pages/InspectionPage'
import type { BenchmarkResponse, Health, Job, Manifest, ReplayRecord, ReplaySummary, Session, TraceEvent } from './types'

type View = 'inspection' | 'benchmark'

function sessionFromReplay(record: ReplayRecord): Session | null {
  if (record.session && typeof record.session === 'object') return record.session
  const possible = record as Partial<Session>
  return typeof possible.id === 'string' && Array.isArray(possible.observed_groups) ? possible as Session : null
}

export default function App() {
  const [view, setView] = useState<View>('inspection')
  const [health, setHealth] = useState<Health | null>(null)
  const [modelInfo, setModelInfo] = useState<Record<string, unknown> | null>(null)
  const [manifest, setManifest] = useState<Manifest | null>(null)
  const [replays, setReplays] = useState<ReplaySummary[]>([])
  const [selectedReplay, setSelectedReplay] = useState('live')
  const [session, setSession] = useState<Session | null>(null)
  const [events, setEvents] = useState<TraceEvent[]>([])
  const [report, setReport] = useState<Record<string, unknown> | null>(null)
  const [job, setJob] = useState<Job | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [benchmarks, setBenchmarks] = useState<BenchmarkResponse | null>(null)
  const [benchmarkLoading, setBenchmarkLoading] = useState(false)
  const [benchmarkError, setBenchmarkError] = useState<string | null>(null)
  const abortRef = useRef(false)
  const pollTokenRef = useRef(0)
  const [config, setConfig] = useState<SessionConfig>({ budget: 6, lambdaCost: .02, policy: 'default', costs: {} })
  const replayMode = selectedReplay !== 'live'

  const loadBootstrap = useCallback(async () => {
    const [healthResult, manifestResult, modelResult, replayResult] = await Promise.allSettled([api.health(), api.manifest(), api.modelInfo(), api.replays()])
    if (healthResult.status === 'fulfilled') setHealth(healthResult.value)
    else setHealth({ status: 'blocked', model_ready: false, data_ready: false, blockers: [`API unavailable: ${errorMessage(healthResult.reason)}`], mode: 'live', utility_ready: false })
    if (manifestResult.status === 'fulfilled') {
      setManifest(manifestResult.value)
      setConfig((current) => ({ ...current, costs: Object.fromEntries(manifestResult.value.panels.map((panel) => [panel.id, panel.cost])) }))
    }
    if (modelResult.status === 'fulfilled') setModelInfo(modelResult.value)
    if (replayResult.status === 'fulfilled') setReplays(replayResult.value.replays)
  }, [])

  useEffect(() => { void loadBootstrap() }, [loadBootstrap])
  useEffect(() => {
    if (health?.status !== 'loading') return
    const timer = window.setTimeout(() => void loadBootstrap(), 2000)
    return () => window.clearTimeout(timer)
  }, [health?.status, health, loadBootstrap])
  useEffect(() => {
    abortRef.current = false
    return () => { abortRef.current = true }
  }, [])

  const refreshSession = useCallback(async (current: Session, shouldApply: () => boolean = () => true) => {
    const [nextSession, trace] = await Promise.all([api.sessions.get(current.id), api.sessions.trace(current.id)])
    if (!shouldApply()) return nextSession
    setSession(nextSession)
    setEvents(trace.events)
    if (nextSession.status === 'completed') {
      const completedReport = await api.sessions.report(nextSession.id).catch(() => null)
      if (shouldApply()) setReport(completedReport)
    }
    return nextSession
  }, [])

  async function pollJob(jobId: string, current: Session) {
    const token = ++pollTokenRef.current
    while (!abortRef.current && token === pollTokenRef.current) {
      const nextJob = await api.job(jobId)
      setJob(nextJob)
      if (nextJob.status === 'completed') { await refreshSession(current, () => token === pollTokenRef.current && !abortRef.current); return }
      if (nextJob.status === 'error') throw new Error(nextJob.error || 'The numerical job failed.')
      current = await refreshSession(current, () => token === pollTokenRef.current && !abortRef.current)
      await new Promise((resolve) => window.setTimeout(resolve, 600))
    }
  }

  async function runAction(action: () => Promise<void>) {
    setBusy(true); setError(null)
    try { await action() } catch (caught) { setError(errorMessage(caught)) } finally { setBusy(false) }
  }

  const createSession = () => runAction(async () => {
    const next = await api.sessions.create({ budget: config.budget, lambda_cost: config.lambdaCost, policy: config.policy, costs: config.costs })
    setSelectedReplay('live'); setSession(next); setEvents(next.trace ?? []); setReport(null); setJob(null)
  })

  const recommend = () => session && runAction(async () => {
    const queued = await api.sessions.recommend(session.id, session.version)
    setJob({ id: queued.job_id, status: 'queued', progress: 0, error: null, result: null })
    await pollJob(queued.job_id, session)
  })

  const autopilot = () => session && runAction(async () => {
    const queued = await api.sessions.run(session.id, session.version)
    setJob({ id: queued.job_id, status: 'queued', progress: 0, error: null, result: null })
    await pollJob(queued.job_id, session)
  })

  const acquire = (groupId: string) => session && runAction(async () => {
    const next = await api.sessions.acquire(session.id, session.version, groupId)
    setSession(next)
    const trace = await api.sessions.trace(next.id)
    setEvents(trace.events)
  })

  const stop = () => session && runAction(async () => {
    let latest = await api.sessions.get(session.id)
    let next: Session
    try {
      next = await api.sessions.stop(latest.id, latest.version)
    } catch (caught) {
      if (!(caught instanceof ApiError) || caught.status !== 409) throw caught
      latest = await api.sessions.get(session.id)
      next = await api.sessions.stop(latest.id, latest.version)
    }
    pollTokenRef.current += 1
    setSession(next)
    const [trace, completedReport] = await Promise.all([api.sessions.trace(next.id), api.sessions.report(next.id).catch(() => null)])
    setEvents(trace.events); setReport(completedReport)
  })

  async function chooseReplay(id: string) {
    setSelectedReplay(id); setError(null); setJob(null); setReport(null)
    if (id === 'live') { setSession(null); setEvents([]); return }
    setBusy(true)
    try {
      const record = await api.replay(id)
      const replaySession = sessionFromReplay(record)
      setSession(replaySession)
      if (replaySession) setConfig(current => ({budget:replaySession.budget,lambdaCost:replaySession.lambda_cost,policy:replaySession.policy,costs:replaySession.costs ?? current.costs}))
      setEvents(record.trace ?? record.session?.trace ?? [])
      setReport(record.report && typeof record.report === 'object' ? record.report as Record<string, unknown> : null)
    } catch (caught) { setError(errorMessage(caught)); setSession(null); setEvents([]) } finally { setBusy(false) }
  }

  const loadBenchmarks = useCallback(async () => {
    setBenchmarkLoading(true); setBenchmarkError(null)
    try { setBenchmarks(await api.benchmarks()) } catch (caught) { setBenchmarkError(errorMessage(caught)) } finally { setBenchmarkLoading(false) }
  }, [])
  useEffect(() => { if (view === 'benchmark' && !benchmarks) void loadBenchmarks() }, [view, benchmarks, loadBenchmarks])

  function exportTrace() {
    const blob = new Blob([JSON.stringify({ mode: replayMode ? 'recorded_replay' : 'live', session_id: session?.id ?? null, events }, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob); const link = document.createElement('a'); link.href = url; link.download = `nextcheck-trace-${session?.id ?? 'recorded'}.json`; link.click(); URL.revokeObjectURL(url)
  }

  const healthTone = health?.status === 'ready' ? 'teal' : health?.status === 'loading' ? 'warn' : 'danger'
  const provenanceReady = health?.model_ready && Object.keys(modelInfo ?? {}).length > 0
  const blockers = useMemo(() => health?.blockers ?? [], [health])

  return <div className="app-shell">
    <header className="app-header">
      <a className="brand" href="#top" aria-label="NextCheck home"><span className="brand-mark"><FlaskConical /></span><span><strong>NextCheck</strong><small>Know what to measure next</small></span></a>
      <nav aria-label="Primary navigation"><button className={view === 'inspection' ? 'active' : ''} onClick={() => setView('inspection')}>Inspection</button><button className={view === 'benchmark' ? 'active' : ''} onClick={() => setView('benchmark')}>Benchmarks</button></nav>
      <div className="header-actions">
        <StatusBadge tone={provenanceReady ? 'teal' : 'warn'} icon={provenanceReady ? CheckCircle2 : AlertTriangle}>TabPFN 3.5 {provenanceReady ? 'READY' : 'NOT READY'}</StatusBadge>
        <StatusBadge tone={replayMode ? 'neutral' : healthTone} icon={replayMode ? Archive : Radio}>{replayMode ? 'RECORDED REPLAY' : 'LIVE INFERENCE'}</StatusBadge>
        <StatusBadge icon={Beaker}>SIMULATED ACQUISITION COSTS</StatusBadge>
        <label className="replay-select"><span className="sr-only">Select live mode or a recorded replay</span><Archive size={15} /><select value={selectedReplay} onChange={(event) => void chooseReplay(event.target.value)}><option value="live">Live session</option>{replays.map((replay) => <option value={replay.id} key={replay.id}>{replay.title ?? replay.name ?? replay.id}</option>)}</select><ChevronDown size={14} /></label>
      </div>
    </header>

    {health?.utility_blockers?.length ? <section className="readiness-banner" role="status"><div><strong>Empirical value policies unavailable</strong>{health.utility_blockers.map((message) => <p key={message}>{message}</p>)}</div></section> : null}
    {health?.status !== 'ready' && view === 'inspection' ? <section className="readiness-banner" role="status"><span className="readiness-icon">{health?.status === 'loading' ? <LoaderCircle className="spin" /> : <DatabaseZap />}</span><div><strong>{health?.status === 'loading' ? 'Backend is preparing resources' : 'Live inference is unavailable'}</strong><p>{blockers[0] ?? 'Start the FastAPI backend on 127.0.0.1:8000, prepare the public dataset, and complete the TabPFN 3.5 model doctor. This console will not substitute another model.'}</p>{blockers.length > 1 ? <details><summary>{blockers.length - 1} more blockers</summary>{blockers.slice(1).map((blocker) => <p key={blocker}>{blocker}</p>)}</details> : null}</div><button className="secondary-button" onClick={() => void loadBootstrap()}><RefreshCw size={15} /> Recheck</button></section> : null}
    {error ? <div className="global-error" role="alert"><AlertTriangle size={18} /><div><strong>Action could not be completed</strong><span>{error}</span></div><button onClick={() => setError(null)} aria-label="Dismiss error">Dismiss</button></div> : null}

    {view === 'inspection' ? <InspectionPage health={health} manifest={manifest} session={session} events={events} report={report} replayMode={replayMode} config={config} setConfig={setConfig} busy={busy} job={job} onCreate={createSession} onRecommend={recommend} onAcquire={acquire} onRun={autopilot} onStop={stop} onExport={exportTrace} /> : <BenchmarkPage data={benchmarks} loading={benchmarkLoading} error={benchmarkError} onReload={() => void loadBenchmarks()} />}
    <footer><span><ShieldCheck size={14} /> Local software replay · no machinery control or repair guidance</span><span>{health ? `API status: ${health.status}` : 'Connecting to local API…'}</span></footer>
  </div>
}
