import type { LucideIcon } from 'lucide-react'
import type { ReactNode } from 'react'

export function StatusBadge({ children, tone = 'neutral', icon: Icon }: { children: ReactNode; tone?: 'neutral' | 'teal' | 'warn' | 'danger'; icon?: LucideIcon }) {
  return <span className={`status-badge ${tone}`}>{Icon ? <Icon size={13} aria-hidden="true" /> : null}{children}</span>
}
