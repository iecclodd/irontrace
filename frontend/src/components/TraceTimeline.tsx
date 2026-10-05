import { CheckCircle2, CircleDot, Flag, Gauge, Waypoints } from 'lucide-react'
import type { TraceEvent } from '../types'

function description(event: TraceEvent): string {
  const group = typeof event.group_id === 'string' ? event.group_id : typeof event.selected_action === 'string' ? event.selected_action : null
  const reason = typeof event.reason === 'string' ? event.reason : typeof event.stop_reason === 'string' ? event.stop_reason : null
  if (event.kind.includes('acquir')) return `${group ?? 'Panel'} readings were acquired and charged to the simulated budget.`
  if (event.kind.includes('recommend') || event.kind.includes('analysis')) return group ? `${group} was proposed from the returned acquisition scores.` : 'Candidate panels were ranked from the current visible state.'
  if (event.kind.includes('stop') || event.kind.includes('complete')) return `The replay ended${reason ? `: ${reason.replaceAll('_', ' ')}` : ''}.`
  if (event.kind.includes('predict')) return 'Model probabilities were updated from the currently visible panels.'
  return `Recorded ${event.kind.replaceAll('_', ' ')} event.`
}

const iconFor = (kind: string) => kind.includes('stop') ? Flag : kind.includes('acquir') ? CheckCircle2 : kind.includes('predict') ? Gauge : kind.includes('recommend') ? Waypoints : CircleDot

export function TraceTimeline({ events }: { events: TraceEvent[] }) {
  if (events.length === 0) return <div className="subtle-empty"><Waypoints size={22} /><strong>No trace events yet</strong><span>Session decisions and acquired values will appear here in chronological order.</span></div>
  return <ol className="timeline">
    {events.map((event, index) => {
      const Icon = iconFor(event.kind)
      const time = event.timestamp ? new Date(event.timestamp) : null
      const values = event.visible_values && typeof event.visible_values === 'object' ? Object.entries(event.visible_values) : []
      const probabilities = Array.isArray(event.probabilities) ? event.probabilities : []
      return <li key={`${event.timestamp}-${event.kind}-${index}`}><span className="timeline-icon"><Icon size={15} /></span><div><div className="timeline-title"><strong>{event.kind.replaceAll('_', ' ')}</strong><time>{time && !Number.isNaN(time.getTime()) ? time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : 'Recorded'}</time></div><p>{description(event)}</p>
        {typeof event.cost === 'number' ? <p>Charged {event.cost} simulated units. Total spent: {String(event.spent)}.</p> : null}
        {probabilities.length === 3 ? <p>{probabilities.map((value,index) => `${['No leakage','Weak leakage','Severe leakage'][index]}: ${(Number(value)*100).toFixed(1)}%`).join(' · ')}</p> : null}
        {values.length ? <details className="trace-values"><summary>{values.length} visible feature summaries</summary><dl>{values.map(([name,value]) => <div key={name}><dt>{name}</dt><dd>{typeof value === 'number' ? value.toPrecision(6) : String(value)}</dd></div>)}</dl></details> : null}
      </div></li>
    })}
  </ol>
}
