'use client'

import { useEffect, useRef } from 'react'

type Particle = {
  x: number
  y: number
  vx: number
  vy: number
  radius: number
}

export default function AnimatedBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const context = canvas.getContext('2d')
    if (!context) return

    let width = 0
    let height = 0
    let animationFrame = 0
    const particles: Particle[] = []

    const resize = () => {
      width = window.innerWidth
      height = window.innerHeight
      canvas.width = width
      canvas.height = height
    }

    const init = () => {
      particles.length = 0
      const count = Math.min(26, Math.floor(window.innerWidth / 64))
      for (let index = 0; index < count; index += 1) {
        particles.push({
          x: Math.random() * width,
          y: Math.random() * height,
          vx: (Math.random() - 0.5) * 0.18,
          vy: (Math.random() - 0.5) * 0.18,
          radius: 1 + Math.random() * 1.5,
        })
      }
    }

    const drawGrid = () => {
      context.strokeStyle = 'rgba(255,255,255,0.04)'
      context.lineWidth = 1
      const step = 96
      for (let x = 0; x < width; x += step) {
        context.beginPath()
        context.moveTo(x, 0)
        context.lineTo(x, height)
        context.stroke()
      }
      for (let y = 0; y < height; y += step) {
        context.beginPath()
        context.moveTo(0, y)
        context.lineTo(width, y)
        context.stroke()
      }
    }

    const tick = () => {
      context.clearRect(0, 0, width, height)
      context.fillStyle = 'rgba(11,15,20,0.92)'
      context.fillRect(0, 0, width, height)
      drawGrid()

      particles.forEach((particle, leftIndex) => {
        particle.x += particle.vx
        particle.y += particle.vy

        if (particle.x < -20) particle.x = width + 20
        if (particle.x > width + 20) particle.x = -20
        if (particle.y < -20) particle.y = height + 20
        if (particle.y > height + 20) particle.y = -20

        context.beginPath()
        context.fillStyle = 'rgba(148,163,184,0.25)'
        context.arc(particle.x, particle.y, particle.radius, 0, Math.PI * 2)
        context.fill()

        for (let rightIndex = leftIndex + 1; rightIndex < particles.length; rightIndex += 1) {
          const other = particles[rightIndex]
          const dx = other.x - particle.x
          const dy = other.y - particle.y
          const distance = Math.hypot(dx, dy)

          if (distance < 130) {
            context.strokeStyle = `rgba(148,163,184,${0.08 * (1 - distance / 130)})`
            context.beginPath()
            context.moveTo(particle.x, particle.y)
            context.lineTo(other.x, other.y)
            context.stroke()
          }
        }
      })

      animationFrame = window.requestAnimationFrame(tick)
    }

    resize()
    init()
    tick()
    window.addEventListener('resize', resize)

    return () => {
      window.removeEventListener('resize', resize)
      window.cancelAnimationFrame(animationFrame)
    }
  }, [])

  return <canvas ref={canvasRef} className="absolute inset-0 h-full w-full opacity-80" aria-hidden />
}
