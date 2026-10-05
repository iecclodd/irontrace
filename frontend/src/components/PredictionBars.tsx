export function PredictionBars({ labels, probabilities }: { labels: string[]; probabilities: number[] | null }) {
  if (!probabilities) return <div className="subtle-empty"><strong>No model probabilities yet</strong><span>Request a recommendation after starting a ready live session.</span></div>
  return <div className="probability-list">
    {probabilities.map((probability, index) => {
      const valid = Number.isFinite(probability) ? Math.max(0, Math.min(1, probability)) : 0
      return <div className="probability-row" key={labels[index] ?? index}>
        <div><span>{labels[index] ?? `Class ${index + 1}`}</span><strong>{(valid * 100).toFixed(1)}%</strong></div>
        <div className="probability-track" role="meter" aria-label={`${labels[index] ?? `Class ${index + 1}`} model probability`} aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(valid * 100)}>
          <span style={{ width: `${valid * 100}%` }} />
        </div>
      </div>
    })}
    <p className="microcopy">Model probabilities describe this checkpoint’s output. They are not certified confidence or a safety assessment.</p>
  </div>
}
