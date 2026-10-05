import { X } from 'lucide-react'
import type { Candidate, Panel } from '../types'

const metric = (value: number | null | undefined) => value == null || !Number.isFinite(value) ? 'Not computed' : value.toFixed(5)

export function AuditDrawer({ candidate, panels, open, onClose }: { candidate: Candidate | null; panels: Panel[]; open: boolean; onClose: () => void }) {
  const panel = panels.find((item) => item.id === candidate?.group_id)
  const donorCount = candidate?.sampler && typeof candidate.sampler.donor_count === 'number' ? candidate.sampler.donor_count : null
  const ranksDiffer = candidate?.predicted_delta != null && candidate.information != null && candidate.information !== candidate.predicted_delta
  return <aside className={`audit-drawer ${open ? 'open' : ''}`} aria-hidden={!open} aria-label="Acquisition audit">
    <div className="drawer-head"><div><span className="eyebrow">Acquisition audit</span><h2>{panel?.name ?? candidate?.group_id ?? 'Candidate details'}</h2></div><button className="icon-button" onClick={onClose} aria-label="Close acquisition audit"><X /></button></div>
    {candidate ? <>
      <div className="audit-grid">
        <div><span>Raw KL</span><strong>{metric(candidate.raw_kl)}</strong></div>
        <div><span>Centered information</span><strong>{metric(candidate.information)}</strong></div>
        <div><span>Residual</span><strong>{metric(candidate.residual)}</strong></div>
        <div><span>Entropy drop</span><strong>{metric(candidate.entropy_drop)}</strong></div>
        <div><span>Predicted value</span><strong>{metric(candidate.predicted_delta)}</strong></div>
        <div><span>Net value</span><strong>{metric(candidate.net_value)}</strong></div>
        <div><span>Donor count</span><strong>{donorCount ?? 'Not reported'}</strong></div>
        <div><span>Sensitivity</span><strong>{metric(candidate.sensitivity)}</strong></div>
      </div>
      <div className="audit-note"><strong>{ranksDiffer ? 'Information and empirical value may rank this action differently.' : 'The returned scores do not establish a ranking conflict.'}</strong><p>Disagreement is a diagnostic, not proof of an incorrect prediction. These are model-implied acquisition quantities under the recorded sampler configuration.</p></div>
    </> : <p>Select a ranked check to inspect its returned diagnostics.</p>}
  </aside>
}
