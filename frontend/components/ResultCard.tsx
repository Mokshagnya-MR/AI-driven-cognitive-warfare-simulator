'use client'

import { motion } from 'framer-motion'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { cn } from '@/lib/utils'

interface ResultCardProps {
  title: string
  value?: string | number
  description?: string
  children?: React.ReactNode
  className?: string
  delay?: number
}

export default function ResultCard({
  title,
  value,
  description,
  children,
  className,
  delay = 0,
}: ResultCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.45, ease: 'easeOut' }}
      whileHover={{ scale: 1.01 }}
      className={cn('h-full', className)}
    >
      <Card className="h-full">
        <CardHeader>
          <CardTitle>{title}</CardTitle>
          {description ? <CardDescription>{description}</CardDescription> : null}
        </CardHeader>
        <CardContent className="space-y-3">
          {value !== undefined ? <div className="metric-value">{value}</div> : null}
          {children}
        </CardContent>
      </Card>
    </motion.div>
  )
}
