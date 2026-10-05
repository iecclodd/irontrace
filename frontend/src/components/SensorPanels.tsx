import { Check, LockKeyhole } from 'lucide-react'
import type { Panel } from '../types'

const formatValue = (value: number) => Number.isFinite(value) ? value.toLocaleString(undefined, { maximumFractionDigits: 3 }) : 'Unavailable'

export function SensorPanels({ panels, observedGroups, visibleValues, costs }: { panels: Panel[]; observedGroups: string[]; visibleValues: Record<string, number>; costs: Record<string, number> }) {
  const observed = new Set(observedGroups)
  return <ul className="panel-list">
    {panels.map((panel) => {
      const isObserved = observed.has(panel.id)
      const readings = panel.features.flatMap((feature) => feature in visibleValues ? [[feature, visibleValues[feature]] as const] : [])
      return <li className={`panel-row ${isObserved ? 'observed' : 'locked'}`} key={panel.id}>
        <div className="panel-head">
          <div className="panel-name"><strong>{panel.name}</strong><span>{panel.sensors.join(', ')}</span></div>
          <span className="panel-state">{isObserved ? <><Check /> Acquired</> : <><LockKeyhole /> {costs[panel.id] ?? panel.cost} units</>}</span>
        </div>
        {isObserved ? <div className="readings" aria-label={`${panel.name} visible readings`}>
          {readings.length > 0 ? readings.slice(0, 8).map(([key, value]) => <div key={key}><span>{key}</span><strong>{formatValue(value)}</strong></div>) : <p>Panel acquired. No summarized values were returned.</p>}
          {readings.length > 8 ? <small>+ {readings.length - 8} more visible values in the trace</small> : null}
        </div> : null}
      </li>
    })}
  </ul>
}
