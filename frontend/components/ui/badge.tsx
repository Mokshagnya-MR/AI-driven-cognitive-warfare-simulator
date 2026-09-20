import { cn } from '@/lib/utils'

type BadgeProps = {
  className?: string
  children: React.ReactNode
}

export function Badge({ className, children }: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center rounded-full border px-3 py-1 text-xs font-medium tracking-[0.16em] uppercase',
        className
      )}
    >
      {children}
    </span>
  )
}