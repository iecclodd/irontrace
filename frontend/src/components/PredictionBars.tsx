export function PredictionBars({ labels, probabilities }: { labels: string[]; probabilities: number[] | null }) {
  if (!probabilities) return <div className="empty"><strong>No probabilities yet</strong><span>Start a live session to see the model’s output.</span></div>
  const valid = probabilities.map((probability) => Number.isFinite(probability) ? Math.max(0, Math.min(1, probability)) : 0)
  const top = valid.indexOf(Math.max(...valid))
  return <div className="probability-list">
    {valid.map((probability, index) => <div className={`probability-row ${index === top ? 'top' : ''}`} key={labels[index] ?? index}>
      <div><span>{labels[index] ?? `Class ${index + 1}`}</span><strong>{(probability * 100).toFixed(1)}%</strong></div>
      <div className="probability-track" role="meter" aria-label={`${labels[index] ?? `Class ${index + 1}`} model probability`} aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(probability * 100)}>
        <span style={{ width: `${probability * 100}%` }} />
      </div>
    </div>)}
    <p className="note">These are this checkpoint’s raw outputs, not certified confidence or a safety assessment.</p>
  </div>
}
