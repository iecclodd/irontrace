export type Health = {
  status: 'ready' | 'blocked' | 'loading'
  model_ready: boolean
  data_ready: boolean
  blockers: string[]
  mode: 'live'
  utility_ready: boolean
  utility_blockers?: string[]
  selected_default?: Record<string, unknown> | string | null
}

export type Panel = {
  id: string
  name: string
  sensors: string[]
  cost: number
  features: string[]
}

export type Manifest = { panels: Panel[]; labels: string[] }

export type Candidate = {
  group_id: string
  cost: number
  affordable: boolean
  raw_kl: number | null
  information: number | null
  residual: number | null
  entropy_drop: number | null
  predicted_delta: number | null
  net_value: number | null
  sampler?: Record<string, unknown>
  sensitivity?: number | null
}

export type Analysis = {
  probabilities?: number[]
  candidates: Candidate[]
  selected_action: string | null
  stop_reason: string | null
  inference_ms: number
  state_hash?: string
}

export type TraceEvent = {
  kind: string
  timestamp: string
  [key: string]: unknown
}

export type Session = {
  id: string
  version: number
  status: 'active' | 'completed' | 'error'
  mode: 'live'
  budget: number
  spent: number
  costs?: Record<string, number>
  remaining_budget: number
  policy: string
  lambda_cost: number
  observed_groups: string[]
  visible_values: Record<string, number>
  probabilities: number[] | null
  recommendation: Analysis | null
  trace: TraceEvent[]
  stop_reason: string | null
  inference_ms: number
  model_ref: string
  selection_note?: string | null
}

export type Job = {
  id: string
  status: 'queued' | 'running' | 'completed' | 'error'
  progress: number
  error: string | null
  result: unknown
}

export type ReplaySummary = { id: string; title?: string; name?: string; timestamp?: string; [key: string]: unknown }
export type ReplayRecord = Record<string, unknown> & {
  id?: string
  session?: Session
  trace?: TraceEvent[]
  probabilities?: number[]
  labels?: string[]
}

export type BenchmarkResponse = { runs: Record<string, unknown>[]; blockers: string[] }
