import type { BenchmarkResponse, Health, Job, Manifest, ReplayRecord, ReplaySummary, Session } from './types'

export class ApiError extends Error {
  constructor(message: string, public status?: number) {
    super(message)
    this.name = 'ApiError'
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })
  if (!response.ok) {
    const body = await response.json().catch(() => null) as { detail?: string; message?: string } | null
    throw new ApiError(body?.detail ?? body?.message ?? `${response.status} ${response.statusText}`, response.status)
  }
  return response.json() as Promise<T>
}

export const api = {
  health: () => request<Health>('/health'),
  modelInfo: () => request<Record<string, unknown>>('/model-info'),
  manifest: () => request<Manifest>('/demo-manifest'),
  sessions: {
    create: (body: { budget: number; lambda_cost: number; policy: string; costs?: Record<string, number> }) =>
      request<Session>('/sessions', { method: 'POST', body: JSON.stringify(body) }),
    get: (id: string) => request<Session>(`/sessions/${encodeURIComponent(id)}`),
    recommend: (id: string, version: number) => request<{ job_id: string }>(`/sessions/${encodeURIComponent(id)}/recommend`, { method: 'POST', body: JSON.stringify({ version }) }),
    acquire: (id: string, version: number, groupId: string) => request<Session>(`/sessions/${encodeURIComponent(id)}/acquire`, { method: 'POST', body: JSON.stringify({ version, group_id: groupId, idempotency_key: crypto.randomUUID() }) }),
    run: (id: string, version: number) => request<{ job_id: string }>(`/sessions/${encodeURIComponent(id)}/run`, { method: 'POST', body: JSON.stringify({ version }) }),
    stop: (id: string, version: number) => request<Session>(`/sessions/${encodeURIComponent(id)}/stop`, { method: 'POST', body: JSON.stringify({ version }) }),
    trace: (id: string) => request<{ events: import('./types').TraceEvent[] }>(`/sessions/${encodeURIComponent(id)}/trace`),
    report: (id: string) => request<Record<string, unknown>>(`/sessions/${encodeURIComponent(id)}/report`),
  },
  job: (id: string) => request<Job>(`/jobs/${encodeURIComponent(id)}`),
  benchmarks: () => request<BenchmarkResponse>('/benchmarks'),
  replays: () => request<{ replays: ReplaySummary[] }>('/replays'),
  replay: (id: string) => request<ReplayRecord>(`/replays/${encodeURIComponent(id)}`),
}

export function errorMessage(error: unknown): string {
  if (error instanceof Error) return error.message
  return 'The request failed unexpectedly.'
}
