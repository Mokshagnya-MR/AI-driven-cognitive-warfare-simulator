'use client'

import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { fetchTrendingNews, generateCounter, generateMisinformation, generateNeutral, type NewsItem } from '@/lib/api'
import Loader from '@/components/Loader'
import ResultCard from '@/components/ResultCard'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'

type GenerationType = 'misinformation' | 'counter' | 'neutral' | null

export default function GeneratePage() {
  const [topic, setTopic] = useState('')
  const [loading, setLoading] = useState(false)
  const [generatedText, setGeneratedText] = useState('')
  const [generationType, setGenerationType] = useState<GenerationType>(null)
  const [error, setError] = useState<string>('')
  const [trendingItems, setTrendingItems] = useState<NewsItem[]>([])

  useEffect(() => {
    const loadTrending = async () => {
      try {
        const items = await fetchTrendingNews()
        setTrendingItems(items.slice(0, 5))
      } catch {
        setTrendingItems([])
      }
    }
    void loadTrending()
  }, [])

  const handleGenerate = async (type: GenerationType) => {
    if (!topic.trim()) {
      setError('Enter a topic before generating.')
      return
    }

    setLoading(true)
    setError('')

    try {
      const nextText =
        type === 'misinformation'
          ? await generateMisinformation(topic)
          : type === 'counter'
            ? await generateCounter(topic)
            : await generateNeutral(topic)
      setGeneratedText(nextText)
      setGenerationType(type)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Generation failed.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="page-shell py-12 lg:py-16">
      {loading && <Loader />}

      <motion.section initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.45 }} className="space-y-8">
        <div className="max-w-2xl space-y-3">
          <div className="metric-label">Generate</div>
          <h1 className="section-title">Narrative studio</h1>
          <p className="section-copy">Generate different narrative types from a single topic input.</p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Topic</CardTitle>
            <CardDescription>Enter a subject to synthesize into different narrative styles.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-5">
            <Input value={topic} onChange={(event) => setTopic(event.target.value)} placeholder="e.g. elections, public health, climate policy" />

            <div className="grid gap-3 sm:grid-cols-3">
              <Button type="button" onClick={() => handleGenerate('misinformation')}>Misinformation</Button>
              <Button type="button" variant="secondary" onClick={() => handleGenerate('counter')}>Counter narrative</Button>
              <Button type="button" variant="outline" onClick={() => handleGenerate('neutral')}>Neutral report</Button>
            </div>

            {error ? <div className="rounded-2xl border border-red-500/20 bg-red-500/5 px-4 py-3 text-sm text-red-300">{error}</div> : null}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Trending topics</CardTitle>
            <CardDescription>Click a headline to autofill the topic field.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {trendingItems.length > 0 ? (
              trendingItems.map((item, index) => (
                <button
                  key={`${item.url}-${index}`}
                  type="button"
                  onClick={() => setTopic(item.title)}
                  className="w-full rounded-xl border border-slate-700/60 bg-slate-900/50 px-4 py-3 text-left transition hover:border-cyan-400/40 hover:bg-slate-800/60"
                >
                  <p className="text-sm font-medium text-slate-100">{item.title}</p>
                  <p className="mt-1 text-xs text-slate-400">{item.source}</p>
                </button>
              ))
            ) : (
              <p className="text-sm text-slate-400">No trending headlines available right now.</p>
            )}
          </CardContent>
        </Card>

        {generatedText ? (
          <ResultCard
            title={generationType === 'misinformation' ? 'Generated misinformation' : generationType === 'counter' ? 'Counter narrative' : 'Neutral report'}
            description="Rendered with a subtle fade-in for readability."
          >
            <p className="whitespace-pre-wrap text-sm leading-7 text-slate-300">{generatedText}</p>
          </ResultCard>
        ) : null}
      </motion.section>
    </main>
  )
}
