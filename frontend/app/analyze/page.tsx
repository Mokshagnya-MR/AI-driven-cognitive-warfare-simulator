'use client'

import dynamic from 'next/dynamic'
import { useState } from 'react'
import { motion } from 'framer-motion'
import { analyze } from '@/lib/api'
import Loader from '@/components/Loader'
import ResultCard from '@/components/ResultCard'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { cn } from '@/lib/utils'

const GraphVisualization = dynamic(() => import('@/components/GraphVisualization'), {
  ssr: false,
  loading: () => <div className="h-full min-h-[360px] rounded-3xl border border-white/10 bg-white/[0.03] shimmer" />,
})

type AnalyzeResponse = Awaited<ReturnType<typeof analyze>>

function formatRiskLabel(level: AnalyzeResponse['explanation']['risk_level']) {
  if (level === 'high') return 'High Risk'
  if (level === 'medium') return 'Medium Risk'
  return 'Low Risk'
}

function getRiskStyles(level: AnalyzeResponse['explanation']['risk_level']) {
  if (level === 'high') {
    return {
      badge: 'border-rose-500/20 bg-rose-500/10 text-rose-200',
      fill: 'bg-rose-400',
      accent: 'text-rose-200',
    }
  }

  if (level === 'medium') {
    return {
      badge: 'border-amber-500/20 bg-amber-500/10 text-amber-100',
      fill: 'bg-amber-300',
      accent: 'text-amber-100',
    }
  }

  return {
    badge: 'border-emerald-500/20 bg-emerald-500/10 text-emerald-100',
    fill: 'bg-emerald-300',
    accent: 'text-emerald-100',
  }
}

function getAssessmentLabel(result: AnalyzeResponse | null) {
  if (!result) return 'Assessment pending'
  return result.prediction === 1 ? 'Likely Misinformation' : 'Likely Authentic'
}

function getAssessmentTone(result: AnalyzeResponse | null) {
  if (!result) return 'text-slate-400'
  return result.prediction === 1 ? 'text-rose-200' : 'text-emerald-200'
}

function describeMetric(label: string, value: number) {
  if (label === 'Velocity') {
    if (value >= 6) return 'High (rapid spread)'
    if (value >= 3) return 'Moderate (steady spread)'
    return 'Low (contained spread)'
  }

  if (label === 'Depth') {
    if (value >= 7) return 'Deep cascade'
    if (value >= 4) return 'Multi-step spread'
    return 'Shallow spread'
  }

  if (label === 'Reach') {
    if (value >= 80) return 'Broad exposure'
    if (value >= 35) return 'Mixed exposure'
    return 'Limited exposure'
  }

  if (label === 'Echo density') {
    if (value >= 0.65) return 'Strong closed-cluster activity'
    if (value >= 0.4) return 'Moderate cluster concentration'
    return 'Open distribution'
  }

  return 'Indicator value'
}

export default function AnalyzePage() {
  const [topic, setTopic] = useState('')
  const [steps, setSteps] = useState(10)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<AnalyzeResponse | null>(null)
  const [error, setError] = useState('')

  const handleAnalyze = async () => {
    if (!topic.trim()) {
      setError('Enter a topic before running analysis.')
      return
    }

    setLoading(true)
    setError('')

    try {
      setResult(await analyze(topic, steps))
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Analysis failed.')
    } finally {
      setLoading(false)
    }
  }

  const confidenceValue = result ? Math.min(Math.max(result.confidence, 0), 1) : 0
  const riskStyles = result ? getRiskStyles(result.explanation.risk_level) : getRiskStyles('low')
  const featureHighlights = result?.explanation.key_drivers ?? []
  const reasoning = result?.explanation.reasoning ?? []
  const metrics = result?.metrics.propagation_metrics
  const graphStats = result?.metrics.graph_stats

  const propagationSignals = result && metrics
    ? [
        { label: 'Reach', value: metrics.reach, max: Math.max(graphStats?.nodes ?? 100, 100) },
        { label: 'Depth', value: metrics.depth, max: 12 },
        { label: 'Velocity', value: metrics.velocity, max: 10 },
        { label: 'Echo density', value: metrics.echo_chamber_density, max: 1 },
      ]
    : []

  return (
    <main className="page-shell py-12 lg:py-16">
      {loading && <Loader />}

      <motion.section initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.45 }} className="space-y-8">
        <div className="max-w-3xl space-y-3">
          <div className="metric-label">Analysis report</div>
          <h1 className="section-title">Propagation behavior and credibility assessment</h1>
          <p className="section-copy">Structured review of narrative spread, bot amplification, cluster concentration, and classification confidence.</p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>New analysis</CardTitle>
            <CardDescription>Assess a topic through generation, propagation, detection, and explanation.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-5">
            <Input value={topic} onChange={(event) => setTopic(event.target.value)} placeholder="e.g. election misinformation, health rumor, supply chain panic" />

            <div className="space-y-3">
              <div className="flex items-center justify-between text-sm text-slate-400">
                <span>Simulation steps</span>
                <span>{steps}</span>
              </div>
              <Input type="range" min={1} max={50} value={steps} onChange={(event) => setSteps(Number(event.target.value))} className="h-2 px-0 py-0 accent-slate-200" />
            </div>

            <Button onClick={handleAnalyze}>Analyze propagation patterns</Button>

            {error ? <div className="rounded-2xl border border-rose-500/20 bg-rose-500/5 px-4 py-3 text-sm text-rose-200">{error}</div> : null}
          </CardContent>
        </Card>

        {result ? (
          <div className="space-y-8">
            <ResultCard title="Generated narrative" description="Narrative synthesized for propagation analysis.">
              <div className="mx-auto max-w-4xl">
                <p className="whitespace-pre-wrap text-sm leading-7 text-slate-300 sm:text-[15px] sm:leading-8">
                  {result.generated_text}
                </p>
              </div>
            </ResultCard>

            <div className="surface-strong h-[380px] overflow-hidden p-3">
              <GraphVisualization graphData={result.metrics.graph_stats} propagationMetrics={result.metrics.propagation_metrics} />
            </div>

            <Card>
              <CardHeader>
                <CardTitle>Assessment</CardTitle>
                <CardDescription>Credibility and risk status for the analyzed narrative.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-5">
                <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                  <div className="space-y-2">
                    <div className={cn('text-3xl font-semibold tracking-tight sm:text-4xl', getAssessmentTone(result))}>
                      {getAssessmentLabel(result)}
                    </div>
                    <p className="max-w-2xl text-sm leading-7 text-slate-400 sm:text-[15px]">
                      {result.prediction === 1
                        ? 'The pattern suggests coordinated amplification, closed-cluster circulation, and elevated manipulation risk.'
                        : 'The pattern appears comparatively organic, with spread characteristics closer to typical user-driven sharing.'}
                    </p>
                  </div>
                  <div className="flex flex-col gap-3 lg:items-end">
                    <Badge className={cn('text-xs uppercase tracking-[0.18em]', riskStyles.badge)}>{formatRiskLabel(result.explanation.risk_level)}</Badge>
                    <div className="rounded-2xl border border-white/10 bg-white/[0.03] px-4 py-3 text-right">
                      <div className="metric-label">Confidence</div>
                      <div className="mt-1 text-3xl font-semibold text-foreground">{(confidenceValue * 100).toFixed(0)}%</div>
                    </div>
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm text-slate-400">
                    <span>Confidence</span>
                    <span>{(confidenceValue * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-3 overflow-hidden rounded-full bg-white/[0.06]">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${confidenceValue * 100}%` }}
                      transition={{ duration: 0.7, ease: 'easeOut' }}
                      className={cn('h-full rounded-full', riskStyles.fill)}
                    />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Why this was flagged</CardTitle>
                <CardDescription>Human-readable signals behind the assessment.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                {reasoning.map((line, index) => (
                  <motion.div
                    key={line}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.35, delay: index * 0.05 }}
                    className="flex items-start gap-3 rounded-2xl border border-white/10 bg-white/[0.03] px-4 py-3 text-sm leading-7 text-slate-300"
                  >
                    <span className={cn('mt-0.5 h-2 w-2 rounded-full', result.prediction === 1 ? 'bg-rose-300' : 'bg-emerald-300')} />
                    <span>{line}</span>
                  </motion.div>
                ))}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Propagation metrics</CardTitle>
                <CardDescription>Interpreted signals with operational meaning.</CardDescription>
              </CardHeader>
              <CardContent className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                {propagationSignals.map((signal, index) => {
                  const ratio = Math.min(signal.value / signal.max, 1)
                  const displayValue = signal.label === 'Echo density' ? signal.value.toFixed(4) : signal.label === 'Velocity' ? signal.value.toFixed(2) : Number(signal.value).toFixed(0)

                  return (
                    <motion.div
                      key={signal.label}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.4, delay: index * 0.05 }}
                      className="rounded-2xl border border-white/10 bg-white/[0.03] p-4"
                    >
                      <div className="metric-label">{signal.label}</div>
                      <div className="mt-2 text-2xl font-semibold text-foreground">
                        {signal.label === 'Velocity'
                          ? describeMetric(signal.label, signal.value)
                          : signal.label === 'Depth'
                            ? describeMetric(signal.label, signal.value)
                            : signal.label === 'Reach'
                              ? describeMetric(signal.label, signal.value)
                              : describeMetric(signal.label, signal.value)}
                      </div>
                      <div className="mt-2 text-sm text-slate-400">{signal.label}: {displayValue}</div>
                      <div className="mt-4 h-2 overflow-hidden rounded-full bg-white/[0.06]">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${Math.max(6, ratio * 100)}%` }}
                          transition={{ duration: 0.7, ease: 'easeOut' }}
                          className={cn('h-full rounded-full', signal.label === 'Velocity' ? 'bg-rose-300' : signal.label === 'Echo density' ? 'bg-amber-300' : 'bg-slate-200')}
                        />
                      </div>
                      <p className="mt-3 text-xs leading-5 text-slate-500">{describeMetric(signal.label, signal.value)}</p>
                    </motion.div>
                  )
                })}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>What influenced this decision</CardTitle>
                <CardDescription>SHAP feature importance translated into ranked drivers.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {featureHighlights.map((driver, index) => (
                  <motion.div
                    key={`${driver.label}-${index}`}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.35, delay: index * 0.05 }}
                    className="space-y-2 rounded-2xl border border-white/10 bg-white/[0.03] p-4"
                  >
                    <div className="flex items-center justify-between gap-4">
                      <div className="space-y-1">
                        <div className="text-sm font-medium text-slate-100">{driver.label}</div>
                        <div className="text-xs leading-5 text-slate-400">{driver.explanation}</div>
                      </div>
                      <div className="rounded-full border border-white/10 bg-black/20 px-2.5 py-1 text-xs text-slate-200">
                        {(driver.score * 100).toFixed(0)}%
                      </div>
                    </div>
                    <div className="h-2 overflow-hidden rounded-full bg-white/[0.06]">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${Math.min(driver.score * 100, 100)}%` }}
                        transition={{ duration: 0.65, ease: 'easeOut', delay: index * 0.05 }}
                        className={cn('h-full rounded-full', index === 0 ? 'bg-slate-100' : 'bg-slate-400')}
                      />
                    </div>
                  </motion.div>
                ))}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Recommendation</CardTitle>
                <CardDescription>Actionable next step for an analyst or reviewer.</CardDescription>
              </CardHeader>
              <CardContent>
                <div className={cn('rounded-3xl border px-5 py-4 text-sm leading-7', riskStyles.badge)}>
                  {result.explanation.recommendation}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Network composition</CardTitle>
                <CardDescription>Distribution of agent types in the simulation.</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                  {Object.entries(result.metrics.graph_stats.agent_counts).map(([agentType, count]) => (
                    <div key={agentType} className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
                      <div className="metric-label">{agentType}</div>
                      <div className="mt-2 text-2xl font-semibold text-foreground">{count}</div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        ) : null}
      </motion.section>
    </main>
  )
}
