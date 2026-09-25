'use client'

import { useEffect, useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import { analyzeNews, fetchTrendingNews, type NewsItem } from '@/lib/api'
import Loader from '@/components/Loader'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { cn } from '@/lib/utils'

type AnalyzeResponse = Awaited<ReturnType<typeof analyzeNews>>

export default function NewsAnalystPage() {
  const [newsItems, setNewsItems] = useState<NewsItem[]>([])
  const [selectedItem, setSelectedItem] = useState<NewsItem | null>(null)
  const [result, setResult] = useState<AnalyzeResponse | null>(null)
  const [loadingNews, setLoadingNews] = useState(false)
  const [analyzing, setAnalyzing] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    const loadNews = async () => {
      setLoadingNews(true)
      setError('')
      try {
        const items = await fetchTrendingNews()
        setNewsItems(items.slice(0, 10))
        setSelectedItem(items[0] ?? null)
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : 'Failed to load trending news.')
      } finally {
        setLoadingNews(false)
      }
    }
    void loadNews()
  }, [])

  const assessmentLabel = useMemo(() => {
    if (!result) return 'No assessment yet'
    return result.prediction === 1 ? 'Likely Misinformation' : 'Likely Authentic'
  }, [result])

  const handleAnalyzeSelected = async () => {
    if (!selectedItem) {
      setError('Select a news headline to analyze.')
      return
    }

    setAnalyzing(true)
    setError('')
    try {
      const analyzed = await analyzeNews(selectedItem.title, selectedItem.description, 10)
      setResult(analyzed)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'News analysis failed.')
    } finally {
      setAnalyzing(false)
    }
  }

  return (
    <main className="page-shell py-12 lg:py-16">
      {(loadingNews || analyzing) && <Loader />}

      <motion.section initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.45 }} className="space-y-8">
        <div className="max-w-3xl space-y-3">
          <div className="metric-label">Real-time analysts</div>
          <h1 className="section-title">Current trending news intelligence</h1>
          <p className="section-copy">Pick a live India headline and run propagation and credibility analysis with contextual enrichment.</p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Trending headlines</CardTitle>
            <CardDescription>Showing title-only list for quick analyst triage.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {newsItems.length > 0 ? (
              newsItems.map((item, index) => {
                const isActive = selectedItem?.url === item.url
                return (
                  <button
                    key={`${item.url}-${index}`}
                    type="button"
                    onClick={() => setSelectedItem(item)}
                    className={cn(
                      'w-full rounded-xl border px-4 py-3 text-left transition',
                      isActive
                        ? 'border-cyan-400/40 bg-cyan-500/10 text-slate-100'
                        : 'border-slate-700/60 bg-slate-900/50 text-slate-300 hover:border-cyan-400/30 hover:bg-slate-800/60'
                    )}
                  >
                    <p className="text-sm font-medium">{item.title}</p>
                  </button>
                )
              })
            ) : (
              <p className="text-sm text-slate-400">No live headlines available.</p>
            )}

            <Button onClick={handleAnalyzeSelected} disabled={!selectedItem || analyzing}>
              Analyze selected news
            </Button>

            {error ? <div className="rounded-2xl border border-rose-500/20 bg-rose-500/5 px-4 py-3 text-sm text-rose-200">{error}</div> : null}
          </CardContent>
        </Card>

        {result ? (
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Analysis summary</CardTitle>
                <CardDescription>Model output for the selected headline.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="text-sm text-slate-400">Assessment</div>
                <div className={cn('text-2xl font-semibold', result.prediction === 1 ? 'text-rose-200' : 'text-emerald-200')}>
                  {assessmentLabel}
                </div>
                <div className="font-mono text-sm tabular-nums text-slate-300">Confidence: {(result.confidence * 100).toFixed(1)}%</div>
                <p className="whitespace-pre-wrap text-sm leading-7 text-slate-300">{result.explanation.summary}</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Why this was flagged (XAI)</CardTitle>
                <CardDescription>Reasoning trace generated from model attributions and propagation signals.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                {result.explanation.reasoning.map((line, index) => (
                  <div
                    key={`${line}-${index}`}
                    className="flex items-start gap-3 rounded-2xl border border-white/10 bg-white/[0.03] px-4 py-3 text-sm leading-7 text-slate-300"
                  >
                    <span className={cn('mt-0.5 h-2 w-2 rounded-full', result.prediction === 1 ? 'bg-rose-300' : 'bg-emerald-300')} />
                    <span>{line}</span>
                  </div>
                ))}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Top feature drivers</CardTitle>
                <CardDescription>Most influential XAI features for this prediction.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                {result.explanation.key_drivers.map((driver, index) => (
                  <div key={`${driver.label}-${index}`} className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
                    <div className="flex items-center justify-between gap-3">
                      <p className="text-sm font-medium text-slate-100">{driver.label}</p>
                      <span className="rounded-full border border-white/10 bg-black/20 px-2.5 py-1 font-mono text-xs tabular-nums text-slate-200">
                        {(driver.score * 100).toFixed(0)}%
                      </span>
                    </div>
                    <p className="mt-2 text-xs leading-6 text-slate-400">{driver.explanation}</p>
                  </div>
                ))}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Recommendation</CardTitle>
                <CardDescription>Suggested next analyst action.</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="rounded-2xl border border-white/10 bg-white/[0.03] px-4 py-3 text-sm leading-7 text-slate-300">
                  {result.explanation.recommendation}
                </div>
              </CardContent>
            </Card>
          </div>
        ) : null}
      </motion.section>
    </main>
  )
}
