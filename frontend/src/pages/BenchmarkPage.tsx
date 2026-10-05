import { BarChart3, Database, FileCheck2, ShieldAlert } from 'lucide-react'
import type { BenchmarkResponse } from '../types'

type Point = { cost: number; loss: number; policy: string; source: Record<string, unknown> }
type ValuePoint = { predicted: number; realized: number; series: string }

function numeric(record: Record<string, unknown>, ...keys: string[]): number | null {
  for (const key of keys) if (typeof record[key] === 'number' && Number.isFinite(record[key])) return record[key] as number
  const metrics = record.metrics
  if (metrics && typeof metrics === 'object') for (const key of keys) {
    const value = (metrics as Record<string, unknown>)[key]
    if (typeof value === 'number' && Number.isFinite(value)) return value
  }
  return null
}

function text(record: Record<string, unknown>, ...keys: string[]): string {
  for (const key of keys) if (typeof record[key] === 'string') return record[key] as string
  return '—'
}

function pointsFor(runs: Record<string, unknown>[]): Point[] {
  return runs.flatMap((run) => {
    const summaries = Array.isArray(run.summaries) ? run.summaries : Array.isArray(run.results) ? run.results : [run]
    return summaries.flatMap((item) => {
      if (!item || typeof item !== 'object') return []
      const source = { ...run, ...(item as Record<string, unknown>) }
      const cost = numeric(source, 'mean_cost', 'cost')
      const loss = numeric(source, 'log_loss')
      return cost == null || loss == null ? [] : [{ cost, loss, policy: text(source, 'policy', 'name'), source }]
    })
  })
}

function valuePointsFor(runs: Record<string, unknown>[]): ValuePoint[] {
  const result: ValuePoint[] = []
  const add = (item: unknown, group?: string) => {
    if (!item || typeof item !== 'object') return
    const source = item as Record<string, unknown>
    const predicted = numeric(source, 'predicted_delta', 'predicted_value', 'predicted')
    const realized = numeric(source, 'realized_delta', 'realized_value', 'realized', 'actual_delta')
    if (predicted == null || realized == null) return
    result.push({ predicted, realized, series: group ?? text(source, 'variant', 'model', 'policy') })
  }
  for (const run of runs) {
    const scatter = run.value_scatter
    if (Array.isArray(scatter)) scatter.forEach((item) => add(item))
    else if (scatter && typeof scatter === 'object') {
      for (const [group, items] of Object.entries(scatter as Record<string, unknown>)) {
        if (Array.isArray(items)) items.forEach((item) => add(item, group))
      }
    }
  }
  return result
}

function BenchmarkChart({ points }: { points: Point[] }) {
  const width = 720, height = 280, left = 48, right = 18, top = 18, bottom = 42
  const maxCost = Math.max(11, ...points.map((point) => point.cost))
  const losses = points.map((point) => point.loss)
  const minLoss = Math.min(...losses), maxLoss = Math.max(...losses)
  const span = Math.max(.1, maxLoss - minLoss)
  const x = (value: number) => left + value / maxCost * (width - left - right)
  const y = (value: number) => top + (maxLoss + span * .1 - value) / (span * 1.2) * (height - top - bottom)
  return <div className="chart-wrap"><svg className="benchmark-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Predictive log loss against mean simulated measurement cost">
    {[0, .25, .5, .75, 1].map((ratio) => <g key={ratio}><line x1={left} y1={top + ratio * (height - top - bottom)} x2={width - right} y2={top + ratio * (height - top - bottom)} className="chart-grid" /><text x={left - 7} y={top + ratio * (height - top - bottom) + 3} textAnchor="end" className="chart-tick">{(maxLoss + span * .1 - ratio * span * 1.2).toFixed(2)}</text></g>)}
    <line x1={left} y1={height - bottom} x2={width - right} y2={height - bottom} className="chart-axis" />
    <line x1={left} y1={top} x2={left} y2={height - bottom} className="chart-axis" />
    {points.map((point, index) => <g key={`${point.policy}-${point.cost}-${index}`}><circle cx={x(point.cost)} cy={y(point.loss)} r="6" className="chart-point"><title>{`${point.policy}: loss ${point.loss.toFixed(4)}, cost ${point.cost.toFixed(2)}`}</title></circle></g>)}
    <text x={width / 2} y={height - 7} textAnchor="middle" className="chart-label">Mean simulated measurement cost</text>
    <text x={14} y={height / 2} textAnchor="middle" transform={`rotate(-90 14 ${height / 2})`} className="chart-label">Predictive log loss</text>
    {[0, maxCost / 2, maxCost].map((value) => <text key={value} x={x(value)} y={height - bottom + 20} textAnchor="middle" className="chart-tick">{value.toFixed(value % 1 ? 1 : 0)}</text>)}
  </svg></div>
}

function ValueScatter({ points }: { points: ValuePoint[] }) {
  const width = 720, height = 280, left = 52, right = 18, top = 18, bottom = 42
  const values = points.flatMap((point) => [point.predicted, point.realized, 0])
  const min = Math.min(...values), max = Math.max(...values)
  const span = Math.max(.001, max - min)
  const low = min - span * .08, high = max + span * .08
  const x = (value: number) => left + (value - low) / (high - low) * (width - left - right)
  const y = (value: number) => top + (high - value) / (high - low) * (height - top - bottom)
  const isAblation = (series: string) => series.toLowerCase().includes('no_residual') || series.toLowerCase().includes('ablation')
  return <div className="chart-wrap"><svg className="benchmark-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Predicted acquisition value against realized log loss improvement">
    {[0, .25, .5, .75, 1].map((ratio) => <g key={ratio}><line x1={left} y1={top + ratio * (height - top - bottom)} x2={width - right} y2={top + ratio * (height - top - bottom)} className="chart-grid" /><text x={left - 7} y={top + ratio * (height - top - bottom) + 3} textAnchor="end" className="chart-tick">{(high - ratio * (high - low)).toFixed(2)}</text></g>)}
    <line x1={x(low)} y1={y(low)} x2={x(high)} y2={y(high)} className="chart-reference" />
    <line x1={left} y1={y(0)} x2={width - right} y2={y(0)} className="chart-zero" />
    <line x1={x(0)} y1={top} x2={x(0)} y2={height - bottom} className="chart-zero" />
    {points.map((point, index) => <circle key={`${point.series}-${index}`} cx={x(point.predicted)} cy={y(point.realized)} r="5" className={isAblation(point.series) ? 'chart-point ablation' : 'chart-point'}><title>{`${point.series}: predicted ${point.predicted.toFixed(4)}, realized ${point.realized.toFixed(4)}`}</title></circle>)}
    <text x={width / 2} y={height - 7} textAnchor="middle" className="chart-label">Predicted log loss improvement</text>
    <text x={14} y={height / 2} textAnchor="middle" transform={`rotate(-90 14 ${height / 2})`} className="chart-label">Realized log loss improvement</text>
    {[low, (low + high) / 2, high].map((value) => <text key={value} x={x(value)} y={height - bottom + 20} textAnchor="middle" className="chart-tick">{value.toFixed(2)}</text>)}
  </svg></div>
}

export function BenchmarkPage({ data, loading, error, onReload }: { data: BenchmarkResponse | null; loading: boolean; error: string | null; onReload: () => void }) {
  const points = pointsFor(data?.runs ?? [])
  const valuePoints = valuePointsFor(data?.runs ?? [])
  return <main className="page benchmark-page">
    <section className="page-intro"><div><span className="eyebrow">Recorded evaluation</span><h1>Policy benchmark</h1><p>Actual persisted summaries from the frozen evaluation pipeline. No sample metrics are generated in this console.</p></div><button className="secondary-button" onClick={onReload} disabled={loading}>{loading ? 'Reading artifacts…' : 'Refresh results'}</button></section>
    {error ? <div className="error-banner" role="alert"><ShieldAlert size={18} /><div><strong>Benchmark results unavailable</strong><span>{error}</span></div></div> : null}
    {data && data.blockers.length > 0 ? <div className="blocker-panel"><ShieldAlert size={20} /><div><strong>Evaluation blockers</strong>{data.blockers.map((blocker) => <p key={blocker}>{blocker}</p>)}</div></div> : null}
    {!loading && data && data.runs.length === 0 ? <section className="empty-state large"><div className="empty-icon"><Database /></div><span className="eyebrow">No result files found</span><h2>Run the frozen benchmark to populate this screen</h2><p>The API did not return persisted run summaries. Prepare data and model access, then run the repository evaluation command documented by the backend. Results must include a status, exact split and model references, and actual metrics before a chart can appear.</p><div className="empty-checks"><span><FileCheck2 size={16} /> artifacts/run_*/summary.json</span><span><FileCheck2 size={16} /> execution_status.json</span></div></section> : null}
    {points.length > 0 ? <>
      <section className="surface chart-card"><div className="section-heading"><div><span className="eyebrow">Primary comparison</span><h2>Loss versus acquisition cost</h2></div><span className="data-provenance"><BarChart3 size={15} /> {points.length} persisted points</span></div><BenchmarkChart points={points} /><p className="microcopy">All-panels entries are full-information references and may be infeasible below their recorded cost. Lower log loss is better.</p></section>
      {valuePoints.length > 0 ? <section className="surface chart-card"><div className="section-heading"><div><span className="eyebrow">Empirical value audit</span><h2>Predicted versus realized value</h2></div><span className="data-provenance"><span className="legend-dot" /> Value <span className="legend-dot ablation" /> No-residual ablation</span></div><ValueScatter points={valuePoints} /><p className="microcopy">Points come from persisted out-of-context value examples. The diagonal marks agreement; disagreement does not identify a causal fault.</p></section> : null}
      <section className="surface result-table-card"><div className="section-heading"><div><span className="eyebrow">Run records</span><h2>Metrics and provenance</h2></div></div><div className="table-scroll"><table><thead><tr><th>Policy</th><th>Run / status</th><th>Budget</th><th>Cost weight</th><th>Feasible</th><th>Log loss</th><th>Mean cost</th><th>Accuracy</th><th>Balanced accuracy</th><th>Brier</th><th>Acquisitions</th><th>Wall time</th><th>Sample count</th><th>Split / model reference</th></tr></thead><tbody>{points.map((point, index) => <tr key={`${point.policy}-${index}`}><td><strong>{point.policy}</strong>{point.policy.toLowerCase().includes('all') ? <small className="table-note">Full-information reference</small> : null}</td><td><strong>{text(point.source, 'scope')}</strong><small className="table-note">{text(point.source, 'status')}</small></td><td>{numeric(point.source, 'budget') ?? '—'}</td><td>{numeric(point.source, 'lambda_cost')?.toFixed(3) ?? '—'}</td><td>{typeof point.source.feasible === 'boolean' ? (point.source.feasible ? 'Yes' : 'No') : '—'}</td><td>{point.loss.toFixed(4)}</td><td>{point.cost.toFixed(2)}</td><td>{numeric(point.source, 'accuracy')?.toFixed(3) ?? '—'}</td><td>{numeric(point.source, 'balanced_accuracy')?.toFixed(3) ?? '—'}</td><td>{numeric(point.source, 'brier_score', 'brier')?.toFixed(3) ?? '—'}</td><td>{numeric(point.source, 'mean_acquisitions', 'acquisition_count', 'purchases')?.toFixed(2) ?? '—'}</td><td>{numeric(point.source, 'wall_time_seconds', 'wall_time', 'wall_time_s')?.toFixed(2) ?? '—'}</td><td>{numeric(point.source, 'sample_count', 'n') ?? objectCount(point.source.evaluated_counts)}</td><td className="ref-cell">{compactRef(point.source.split_counts)}<br />{compactRef(point.source.model_provenance)}</td></tr>)}</tbody></table></div></section>
    </> : data && data.runs.length > 0 && !loading ? <section className="empty-state"><BarChart3 /><h2>Run records contain no plottable summaries</h2><p>The API returned run metadata, but no record contained both numeric log loss and mean simulated cost. Inspect the evaluation status and summary schema.</p></section> : null}
  </main>
}

function compactRef(value: unknown): string {
  if (typeof value === 'string') return value
  if (value && typeof value === 'object') {
    const record = value as Record<string, unknown>
    const preferred = ['hash', 'checkpoint', 'checkpoint_hash', 'model_ref']
    const preferredValue = preferred.map((key) => record[key]).find((item) => typeof item === 'string')
    if (typeof preferredValue === 'string') return preferredValue
    return Object.entries(record).slice(0, 3).map(([key, item]) => `${key}:${String(item)}`).join(' · ')
  }
  return '—'
}

function objectCount(value: unknown): string | number {
  if (typeof value === 'number') return value
  if (value && typeof value === 'object') return compactRef(value)
  return '—'
}
