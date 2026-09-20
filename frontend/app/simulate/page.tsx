'use client'

import dynamic from 'next/dynamic'
import { useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { simulate } from '@/lib/api'
import Loader from '@/components/Loader'
import ResultCard from '@/components/ResultCard'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'

const GraphVisualization = dynamic(() => import('@/components/GraphVisualization'), {
  ssr: false,
  loading: () => <div className="h-full min-h-[360px] rounded-3xl border border-white/10 bg-white/[0.03] shimmer" />,
})

export default function SimulatePage() {
  const [text, setText] = useState('')
  const [steps, setSteps] = useState(10)
  const [botRatio, setBotRatio] = useState(0.18)
  const [seedNodes, setSeedNodes] = useState(3)
  const [influencerStrength, setInfluencerStrength] = useState(0.7)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<Awaited<ReturnType<typeof simulate>> | null>(null)
  const [error, setError] = useState('')
  const hasMountedRef = useRef(false)
  const hasRunSimulationRef = useRef(false)

  const runSimulation = async () => {
    if (!text.trim()) {
      setError('Enter narrative text before running the simulation.')
      return
    }

    setLoading(true)
    setError('')

    try {
      setResult(
        await simulate(text, steps, {
          bot_ratio_override: botRatio,
          seed_nodes: seedNodes,
          influencer_strength: influencerStrength,
        })
      )
      hasRunSimulationRef.current = true
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Simulation failed.')
    } finally {
      setLoading(false)
    }
  }

  const handleSimulate = async () => {
    await runSimulation()
  }

  useEffect(() => {
    if (!hasMountedRef.current) {
      hasMountedRef.current = true
      return
    }
    if (!hasRunSimulationRef.current || !text.trim()) {
      return
    }

    const timer = window.setTimeout(() => {
      void runSimulation()
    }, 300)

    return () => window.clearTimeout(timer)
  }, [botRatio, seedNodes, influencerStrength, steps, text])

  return (
    <main className="page-shell py-12 lg:py-16">
      {loading && <Loader />}

      <motion.section initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.45 }} className="space-y-8">
        <div className="max-w-2xl space-y-3">
          <div className="metric-label">Simulate</div>
          <h1 className="section-title">Propagation model</h1>
          <p className="section-copy">Run a clean social network simulation and inspect the resulting spread metrics.</p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Input</CardTitle>
            <CardDescription>Provide text and a step count to simulate propagation across the graph.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-5">
            <Textarea value={text} onChange={(event) => setText(event.target.value)} placeholder="Paste a narrative, statement, or message to simulate." />

            <div className="space-y-3">
              <div className="flex items-center justify-between text-sm text-slate-400">
                <span>Simulation steps</span>
                <span>{steps}</span>
              </div>
              <Input type="range" min={1} max={50} value={steps} onChange={(event) => setSteps(Number(event.target.value))} className="h-2 px-0 py-0 accent-slate-200" />
            </div>

            <div className="space-y-3">
              <div className="flex items-center justify-between text-sm text-slate-400">
                <span>Bot ratio</span>
                <span>{botRatio.toFixed(2)}</span>
              </div>
              <Input type="range" min={0} max={1} step={0.01} value={botRatio} onChange={(event) => setBotRatio(Number(event.target.value))} className="h-2 px-0 py-0 accent-slate-200" />
            </div>

            <div className="space-y-3">
              <div className="flex items-center justify-between text-sm text-slate-400">
                <span>Initial seeds</span>
                <span>{seedNodes}</span>
              </div>
              <Input type="range" min={1} max={50} step={1} value={seedNodes} onChange={(event) => setSeedNodes(Number(event.target.value))} className="h-2 px-0 py-0 accent-slate-200" />
            </div>

            <div className="space-y-3">
              <div className="flex items-center justify-between text-sm text-slate-400">
                <span>Influence strength</span>
                <span>{influencerStrength.toFixed(2)}</span>
              </div>
              <Input type="range" min={0} max={1} step={0.01} value={influencerStrength} onChange={(event) => setInfluencerStrength(Number(event.target.value))} className="h-2 px-0 py-0 accent-slate-200" />
            </div>

            <Button onClick={handleSimulate}>Run simulation</Button>

            {error ? <div className="rounded-2xl border border-red-500/20 bg-red-500/5 px-4 py-3 text-sm text-red-300">{error}</div> : null}
          </CardContent>
        </Card>

        {result ? (
          <div className="space-y-8">
            <div className="surface-strong h-[380px] overflow-hidden p-3">
              <GraphVisualization graphData={result.graph_stats} />
            </div>

            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
              <ResultCard title="Reach" value={result.propagation_metrics.reach} description="Total exposed nodes" />
              <ResultCard title="Depth" value={result.propagation_metrics.depth} description="Maximum hop distance" />
              <ResultCard title="Velocity" value={result.propagation_metrics.velocity.toFixed(2)} description="New nodes per step" />
              <ResultCard title="Echo density" value={result.propagation_metrics.echo_chamber_density.toFixed(4)} description="Intra-cluster edge ratio" />
            </div>

            <Card>
              <CardHeader>
                <CardTitle>Network composition</CardTitle>
                <CardDescription>Agent distribution across the simulated network.</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                  {Object.entries(result.graph_stats.agent_counts).map(([agentType, count]) => (
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
