import Hero from '@/components/Hero'

export default function Home() {
  return (
    <>
      <Hero />
      <section className="page-shell py-16 lg:py-20">
        <div className="grid gap-6 lg:grid-cols-3">
          {[
            ['Generation', 'Draft misinformation, counter-narratives, and neutral reporting from a single interface.'],
            ['Simulation', 'Inspect propagation metrics across a lightweight social network model.'],
            ['Analysis', 'Run the full pipeline and inspect model outputs and explanations.'],
          ].map(([title, copy]) => (
            <div key={title} className="surface p-6">
              <div className="metric-label">{title}</div>
              <p className="mt-3 text-sm leading-6 text-slate-400">{copy}</p>
            </div>
          ))}
        </div>
      </section>
    </>
  )
}
