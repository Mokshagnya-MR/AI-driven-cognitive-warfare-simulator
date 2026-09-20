'use client'

import { useEffect, useRef } from 'react'

export default function OrbScene() {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    let mounted = true
    let frame = 0
    let cleanup = () => undefined

    const run = async () => {
      const THREE = await import('three')
      if (!mounted || !containerRef.current) return

      const container = containerRef.current
      const width = container.clientWidth
      const height = container.clientHeight

      const scene = new THREE.Scene()
      const camera = new THREE.PerspectiveCamera(40, width / height, 0.1, 100)
      camera.position.z = 3.2

      const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
      renderer.setSize(width, height)
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5))
      container.appendChild(renderer.domElement)

      const group = new THREE.Group()
      scene.add(group)

      const sphere = new THREE.Mesh(
        new THREE.SphereGeometry(0.9, 32, 32),
        new THREE.MeshStandardMaterial({
          color: 0x91a4c3,
          roughness: 0.55,
          metalness: 0.12,
          transparent: true,
          opacity: 0.92,
        })
      )
      group.add(sphere)

      const wire = new THREE.Mesh(
        new THREE.SphereGeometry(1.08, 20, 20),
        new THREE.MeshBasicMaterial({ color: 0x334155, wireframe: true, transparent: true, opacity: 0.22 })
      )
      group.add(wire)

      const nodes: Array<InstanceType<typeof THREE.Mesh>> = []
      const nodeGeometry = new THREE.SphereGeometry(0.04, 12, 12)
      const nodeMaterial = new THREE.MeshBasicMaterial({ color: 0xe2e8f0, transparent: true, opacity: 0.45 })

      const positions = [
        [0.9, 0.05, 0],
        [-0.7, 0.55, 0.3],
        [-0.2, -0.75, -0.1],
        [0.45, -0.4, 0.7],
        [0.15, 0.8, -0.4],
      ]

      positions.forEach(([x, y, z]) => {
        const node = new THREE.Mesh(nodeGeometry, nodeMaterial.clone())
        node.position.set(x, y, z)
        nodes.push(node)
        group.add(node)
      })

      const light = new THREE.DirectionalLight(0xffffff, 1.1)
      light.position.set(2, 3, 4)
      scene.add(light)
      scene.add(new THREE.AmbientLight(0xffffff, 0.35))

      const handleResize = () => {
        const nextWidth = container.clientWidth
        const nextHeight = container.clientHeight
        camera.aspect = nextWidth / nextHeight
        camera.updateProjectionMatrix()
        renderer.setSize(nextWidth, nextHeight)
      }

      const animate = () => {
        frame = window.requestAnimationFrame(animate)
        group.rotation.y += 0.0035
        group.rotation.x = Math.sin(Date.now() * 0.0002) * 0.08
        nodes.forEach((node, index) => {
          node.position.y += Math.sin(Date.now() * 0.001 + index) * 0.0004
        })
        renderer.render(scene, camera)
      }

      window.addEventListener('resize', handleResize)
      handleResize()
      animate()

      cleanup = () => {
        window.removeEventListener('resize', handleResize)
        window.cancelAnimationFrame(frame)
        renderer.dispose()
        renderer.domElement.remove()
      }
    }

    run()

    return () => {
      mounted = false
      cleanup()
    }
  }, [])

  return <div ref={containerRef} className="h-full w-full" />
}
