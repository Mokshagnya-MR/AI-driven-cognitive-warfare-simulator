'use client'

import { motion } from 'framer-motion'
import { Skeleton } from '@/components/ui/skeleton'

export default function Loader() {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-background/65 backdrop-blur-sm">
      <div className="surface-strong w-full max-w-md p-6">
        <div className="mb-6 flex items-center justify-between">
          <div>
            <div className="text-sm text-slate-400">Analyzing propagation patterns...</div>
            <div className="text-lg font-medium text-foreground">Building the assessment</div>
          </div>
          <motion.div
            className="h-10 w-10 rounded-full border border-white/10 border-t-slate-100"
            animate={{ rotate: 360 }}
            transition={{ duration: 1.2, repeat: Infinity, ease: 'linear' }}
          />
        </div>
        <div className="space-y-3">
          <Skeleton className="h-4 w-3/4" />
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-5/6" />
          <Skeleton className="h-24 w-full rounded-2xl" />
        </div>
      </div>
    </div>
  )
}
