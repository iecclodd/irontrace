import { X } from 'lucide-react'
import type { Candidate, Panel } from '../types'

const metric = (value: number | null | undefined) => value == null || !Number.isFinite(value) ? 'Not computed' : value.toFixed(5)

export function AuditDrawer({ candidate, candidates, panels, open, onClose }: { candidate: Candidate | null; candidates: Candidate[]; panels: Panel[]; open: boolean; onClose: () => void }) {
  const panel = panels.find((item) => item.id === candidate?.group_id)
  const donorCount = candidate?.sampler?.neighborhood_size ?? candidate?.sampler?.donor_count
  const uniqueDonors = candidate?.sampler?.unique_donor_count
  const comparable = candidates.filter((item) => item.affordable && item.predicted_delta != null && item.information != null)
  const informationOrder = [...comparable].sort((a,b) => b.information! - a.information!).map(item => item.group_id).join(',')
  const valueOrder = [...comparable].sort((a,b) => b.predicted_delta! - a.predicted_delta!).map(item => item.group_id).join(',')
  const ranksDiffer = comparable.length > 1 && informationOrder !== valueOrder
  if (!open) return null
  return <aside className="audit-drawer" aria-label="Acquisition audit">
    <div className="drawer-head"><div><p>Acquisition audit</p><h2>{panel?.name ?? candidate?.group_id ?? 'Candidate details'}</h2></div><button className="icon-button" onClick={onClose} aria-label="Close acquisition audit"><X /></button></div>
    {candidate ? <>
      <dl className="audit-list">
        <div><dt>Raw KL</dt><dd>{metric(candidate.raw_kl)}</dd></div>
        <div><dt>Centered information</dt><dd>{metric(candidate.information)}</dd></div>
        <div><dt>Residual</dt><dd>{metric(candidate.residual)}</dd></div>
        <div><dt>Entropy drop</dt><dd>{metric(candidate.entropy_drop)}</dd></div>
        <div><dt>Predicted value</dt><dd>{metric(candidate.predicted_delta)}</dd></div>
        <div><dt>{candidate.predicted_delta == null ? 'Heuristic after cost' : 'Net value'}</dt><dd>{metric(candidate.net_value)}</dd></div>
        <div><dt>Donor neighborhood / unique draws</dt><dd>{typeof donorCount === 'number' ? donorCount : 'Not reported'} / {typeof uniqueDonors === 'number' ? uniqueDonors : '—'}</dd></div>
        <div><dt>Sensitivity</dt><dd>{metric(candidate.sensitivity)}</dd></div>
      </dl>
      <div className="audit-note"><strong>{comparable.length < 2 ? 'Empirical ranking comparison is not available.' : ranksDiffer ? 'Centered information and predicted improvement rank the available panels differently.' : 'Centered information and predicted improvement give the same panel ordering.'}</strong><p>Disagreement is a diagnostic, not proof of an incorrect prediction. These are model-implied acquisition quantities under the recorded sampler configuration.</p></div>
    </> : <p className="note">Select a ranked check to inspect its returned diagnostics.</p>}
  </aside>
}
