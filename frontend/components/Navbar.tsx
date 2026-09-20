'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { motion } from 'framer-motion'

const links = [
  { href: '/', label: 'Home' },
  { href: '/generate', label: 'Generate' },
  { href: '/simulate', label: 'Simulate' },
  { href: '/analyze', label: 'Analyze' },
  { href: '/news-analyst', label: 'News Analyst' },
]

export default function Navbar() {
  const pathname = usePathname()

  return (
    <header className="sticky top-0 z-50 border-b border-white/8 bg-background/80 backdrop-blur-xl">
      <div className="page-shell flex h-16 items-center justify-between">
        <Link href="/" className="flex items-center gap-3">
          <div className="h-8 w-8 rounded-full border border-white/10 bg-white/[0.04]" />
          <div>
            <div className="text-sm font-medium tracking-[0.18em] text-foreground">ACWS</div>
            <div className="text-[11px] text-slate-500">Cognitive warfare simulator</div>
          </div>
        </Link>

        <nav className="hidden items-center gap-2 md:flex">
          {links.map((link) => {
            const active = pathname === link.href
            return (
              <Link
                key={link.href}
                href={link.href}
                className="relative px-3 py-2 text-sm text-slate-400 transition hover:text-foreground"
              >
                <span>{link.label}</span>
                {active ? (
                  <motion.span
                    layoutId="nav-indicator"
                    className="absolute inset-x-3 -bottom-0.5 h-px bg-slate-200"
                    transition={{ type: 'spring', stiffness: 380, damping: 30 }}
                  />
                ) : null}
              </Link>
            )
          })}
        </nav>
      </div>
    </header>
  )
}
