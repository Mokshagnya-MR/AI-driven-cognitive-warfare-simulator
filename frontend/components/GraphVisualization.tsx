'use client'

import { useEffect, useRef } from 'react'
import type { Mesh, MeshBasicMaterial, Material } from 'three'

interface GraphData {
  nodes: number
  edges: number
  agent_counts: Record<string, number>
}

interface PropagationMetrics {
  reach: number
  depth: number
  velocity: number
  echo_chamber_density: number
}

interface GraphVisualizationProps {
  graphData?: GraphData
  propagationMetrics?: PropagationMetrics
}

type NodeState = {
  mesh: Mesh
  phase: number
}

export default function GraphVisualization({ graphData, propagationMetrics }: GraphVisualizationProps) {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    let active = true
    let animationFrame = 0
    let frameTick = 0
    let teardown = () => undefined

    const mount = async () => {
      const THREE = await import('three')
      if (!active || !containerRef.current) return

      const container = containerRef.current
      const width = container.clientWidth
      const height = container.clientHeight

      const scene = new THREE.Scene()
      scene.background = new THREE.Color(0x0b0f14)

      const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100)
      camera.position.z = 7

      const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
      renderer.setSize(width, height)
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.1))
      container.appendChild(renderer.domElement)

      const group = new THREE.Group()
      scene.add(group)

      const totalNodes = Math.max(8, graphData?.nodes ?? 12)
      const agentCounts = graphData?.agent_counts ?? { normal: 10, influencer: 4, bot: 3, skeptic: 1 }
      const palette = {
        bot: new THREE.Color(0x7c8597),
        influencer: new THREE.Color(0x8b5cf6),
        skeptic: new THREE.Color(0x3b82f6),
        normal: new THREE.Color(0xd1d5db),
      }
      const activePalette = {
        bot: new THREE.Color(0xef4444),
        influencer: new THREE.Color(0x8b5cf6),
        skeptic: new THREE.Color(0x3b82f6),
        normal: new THREE.Color(0xe5e7eb),
      }

      const types = Object.entries(agentCounts).flatMap(([type, count]) => Array.from({ length: count }, () => type))
      while (types.length < totalNodes) types.push('normal')

      const nodes: NodeState[] = []
      const nodeGeometry = new THREE.SphereGeometry(0.08, 10, 10)

      const radius = 2.8
      const ringStep = 0.85
      for (let index = 0; index < totalNodes; index += 1) {
        const type = types[index] ?? 'normal'
        const material = new THREE.MeshBasicMaterial({
          color: palette[type as keyof typeof palette] ?? palette.normal,
          transparent: true,
          opacity: 0.65,
        })
        const node = new THREE.Mesh(nodeGeometry, material)
        const ring = Math.floor(index / Math.max(1, Math.floor(totalNodes / 3)))
        const angle = (index / totalNodes) * Math.PI * 2
        const elevation = Math.sin(index * 0.85) * 0.8
        const distance = radius + ring * ringStep
        node.position.set(Math.cos(angle) * distance, elevation, Math.sin(angle) * distance)
        nodes.push({ mesh: node, phase: index / Math.max(1, totalNodes - 1) })
        group.add(node)
      }

      const edgeCount = Math.min(graphData?.edges ?? 12, totalNodes)
      const lineGeometry = new THREE.BufferGeometry()
      const linePositions: number[] = []
      for (let index = 0; index < edgeCount; index += 1) {
        const start = nodes[index % nodes.length]
        const end = nodes[(index * 2 + 3) % nodes.length]
        linePositions.push(start.mesh.position.x, start.mesh.position.y, start.mesh.position.z)
        linePositions.push(end.mesh.position.x, end.mesh.position.y, end.mesh.position.z)
      }
      lineGeometry.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3))
      const lineMaterial = new THREE.LineBasicMaterial({ color: 0x64748b, transparent: true, opacity: 0.28 })
      const lines = new THREE.LineSegments(lineGeometry, lineMaterial)
      group.add(lines)

      const activeLineGeometry = new THREE.BufferGeometry()
      const activeLineMaterial = new THREE.LineBasicMaterial({ color: 0xe5e7eb, transparent: true, opacity: 0.5 })
      const activeLines = new THREE.LineSegments(activeLineGeometry, activeLineMaterial)
      group.add(activeLines)

      scene.add(new THREE.AmbientLight(0xffffff, 0.45))
      const directional = new THREE.DirectionalLight(0xffffff, 0.8)
      directional.position.set(2, 4, 5)
      scene.add(directional)

      const pulseRing = new THREE.Mesh(
        new THREE.TorusGeometry(3.0, 0.02, 4, 64),
        new THREE.MeshBasicMaterial({ color: 0xe5e7eb, transparent: true, opacity: 0.08 })
      )
      pulseRing.rotation.x = Math.PI / 2
      group.add(pulseRing)

      const resize = () => {
        const nextWidth = container.clientWidth
        const nextHeight = container.clientHeight
        camera.aspect = nextWidth / nextHeight
        camera.updateProjectionMatrix()
        renderer.setSize(nextWidth, nextHeight)
      }

      const velocity = propagationMetrics?.velocity ?? 3.5
      const depth = propagationMetrics?.depth ?? 5
      const cycleDuration = Math.max(4200 - velocity * 220, 2600)
      const activationWindow = Math.min(0.2 + depth * 0.015, 0.38)

      const updateActiveEdges = (activeLimit: number) => {
        const positions: number[] = []
        const visibleNodes = nodes.slice(0, activeLimit)

        for (let index = 0; index < visibleNodes.length - 1; index += 1) {
          const left = visibleNodes[index].mesh.position
          const right = visibleNodes[(index + 2) % visibleNodes.length].mesh.position
          positions.push(left.x, left.y, left.z, right.x, right.y, right.z)
        }

        activeLineGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
        activeLineGeometry.computeBoundingSphere()
      }

      const animate = () => {
        animationFrame = window.requestAnimationFrame(animate)
        frameTick += 1
        const now = Date.now()
        const phase = (now % cycleDuration) / cycleDuration
        const easedPhase = Math.min(1, phase / Math.max(activationWindow, 0.01))
        const activeLimit = Math.max(1, Math.floor(easedPhase * totalNodes))

        group.rotation.y += 0.0016
        group.rotation.x = Math.sin(now * 0.00012) * 0.04
        pulseRing.scale.setScalar(0.9 + phase * 1.15)
        pulseRing.material.opacity = 0.06 + (1 - phase) * 0.06

        if (frameTick % 2 === 0) {
          nodes.forEach(({ mesh, phase: nodePhase }, index) => {
            const threshold = nodePhase
            const activation = Math.max(0, Math.min(1, (phase - threshold) / 0.08))
            const type = types[index] ?? 'normal'
            const baseColor = palette[type as keyof typeof palette] ?? palette.normal
            const activeColor = activePalette[type as keyof typeof activePalette] ?? activePalette.normal
            const currentColor = baseColor.clone().lerp(activeColor, activation)

            const meshMaterial = mesh.material as MeshBasicMaterial
            meshMaterial.color = currentColor
            meshMaterial.opacity = 0.45 + activation * 0.55
            mesh.scale.setScalar(0.9 + activation * 0.9)
            mesh.rotation.x += 0.008
            mesh.rotation.y += 0.008
          })

          updateActiveEdges(activeLimit)
        }

        renderer.render(scene, camera)
      }

      window.addEventListener('resize', resize)
      resize()
      animate()

      teardown = () => {
        window.removeEventListener('resize', resize)
        window.cancelAnimationFrame(animationFrame)
        lineGeometry.dispose()
        activeLineGeometry.dispose()
        nodeGeometry.dispose()
        pulseRing.geometry.dispose()
        ;(pulseRing.material as Material).dispose()
        renderer.dispose()
        renderer.domElement.remove()
      }
    }

    mount()

    return () => {
      active = false
      teardown()
    }
  }, [graphData])

  return <div ref={containerRef} className="h-full w-full" />
}
