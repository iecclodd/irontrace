import { CheckCircle2, CircleDot, Flag, Gauge, Waypoints } from 'lucide-react'
import type { TraceEvent } from '../types'

function description(event: TraceEvent): string {
  const group = typeof event.group_id === 'string' ? event.group_id : null
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
      return <li key={`${event.timestamp}-${event.kind}-${index}`}><span className="timeline-icon"><Icon size={15} /></span><div><div className="timeline-title"><strong>{event.kind.replaceAll('_', ' ')}</strong><time>{time && !Number.isNaN(time.getTime()) ? time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : 'Recorded'}</time></div><p>{description(event)}</p></div></li>
    })}
  </ol>
}
