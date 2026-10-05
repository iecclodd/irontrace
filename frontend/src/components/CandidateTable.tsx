import { ArrowUpRight, SearchX } from 'lucide-react'
import type { Candidate, Panel } from '../types'

const metric = (value: number | null | undefined, digits = 3) => value == null || !Number.isFinite(value) ? '—' : value.toFixed(digits)

export function CandidateTable({ candidates, panels, selectedAction, activeCandidate, onSelect }: { candidates: Candidate[]; panels: Panel[]; selectedAction: string | null; activeCandidate: string | null; onSelect: (id: string) => void }) {
  const names = new Map(panels.map((panel) => [panel.id, panel.name]))
  if (candidates.length === 0) return <div className="subtle-empty"><SearchX size={22} /><strong>No ranked checks</strong><span>Run a recommendation to compare affordable sensor panels.</span></div>
  return <div className="candidate-list" role="list" aria-label="Ranked candidate checks">
    {candidates.map((candidate, index) => <button type="button" className={`candidate-row ${activeCandidate === candidate.group_id ? 'active' : ''}`} onClick={() => onSelect(candidate.group_id)} key={candidate.group_id}>
      <span className="candidate-rank">{index + 1}</span>
      <span className="candidate-main"><strong>{names.get(candidate.group_id) ?? candidate.group_id}</strong><small>{candidate.affordable ? `${candidate.cost} simulated units` : 'Over remaining budget'}</small></span>
      <span className="candidate-metric"><small>Net value</small><strong>{metric(candidate.net_value)}</strong></span>
      {selectedAction === candidate.group_id ? <span className="recommended-label">Recommended <ArrowUpRight size={12} /></span> : null}
    </button>)}
  </div>
}
