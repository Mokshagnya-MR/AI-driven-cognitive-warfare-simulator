interface SparklineProps {
  data: number[]
  className?: string
  strokeColor?: string
  fillId?: string
}

const WIDTH = 240
const HEIGHT = 64
const PADDING = 6

export default function Sparkline({ data, className, strokeColor = '#e2e8f0', fillId = 'sparkline-fill' }: SparklineProps) {
  if (data.length === 0) {
    return null
  }

  const max = Math.max(...data, 1)
  const min = Math.min(...data, 0)
  const range = Math.max(max - min, 1)
  const usableWidth = WIDTH - PADDING * 2
  const usableHeight = HEIGHT - PADDING * 2

  const points = data.map((value, index) => {
    const x = data.length === 1 ? PADDING : PADDING + (index / (data.length - 1)) * usableWidth
    const y = PADDING + usableHeight - ((value - min) / range) * usableHeight
    return { x, y }
  })

  const linePath = points.map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`).join(' ')
  const areaPath = `${linePath} L ${points[points.length - 1].x.toFixed(2)} ${HEIGHT - PADDING} L ${points[0].x.toFixed(2)} ${HEIGHT - PADDING} Z`
  const endpoint = points[points.length - 1]
  const gridLines = [0.25, 0.5, 0.75].map((fraction) => PADDING + usableHeight * fraction)

  return (
    <svg
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      className={className}
      role="img"
      aria-label={`Trend across ${data.length} steps, from ${data[0]} to ${data[data.length - 1]}`}
    >
      <defs>
        <linearGradient id={fillId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={strokeColor} stopOpacity="0.28" />
          <stop offset="100%" stopColor={strokeColor} stopOpacity="0" />
        </linearGradient>
      </defs>

      {gridLines.map((y) => (
        <line key={y} x1={PADDING} x2={WIDTH - PADDING} y1={y} y2={y} stroke="currentColor" strokeOpacity="0.08" strokeWidth="1" />
      ))}

      <path d={areaPath} fill={`url(#${fillId})`} stroke="none" />
      <path d={linePath} fill="none" stroke={strokeColor} strokeWidth="1.75" strokeLinejoin="round" strokeLinecap="round" />
      <circle cx={endpoint.x} cy={endpoint.y} r="2.5" fill={strokeColor} />
    </svg>
  )
}
