import { cn } from '@/lib/utils'

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn('rounded-2xl bg-white/[0.04] shimmer', className)} />
}
