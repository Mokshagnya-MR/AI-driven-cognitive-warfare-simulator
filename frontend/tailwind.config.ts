import type { Config } from 'tailwindcss'
import defaultTheme from 'tailwindcss/defaultTheme'

const config: Config = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['var(--font-geist-sans)', ...defaultTheme.fontFamily.sans],
        mono: ['var(--font-geist-mono)', ...defaultTheme.fontFamily.mono],
      },
      colors: {
        background: '#0B0F14',
        surface: '#0F172A',
        foreground: '#E5E7EB',
        muted: '#94A3B8',
        accent: {
          blue: '#3B82F6',
          purple: '#8B5CF6',
        },
      },
      boxShadow: {
        soft: '0 20px 60px rgba(15, 23, 42, 0.28)',
      },
      backgroundImage: {
        'subtle-grid':
          'linear-gradient(rgba(148, 163, 184, 0.08) 1px, transparent 1px), linear-gradient(90deg, rgba(148, 163, 184, 0.08) 1px, transparent 1px)',
      },
      backgroundSize: {
        grid: '72px 72px',
      },
    },
  },
  plugins: [],
}
export default config
