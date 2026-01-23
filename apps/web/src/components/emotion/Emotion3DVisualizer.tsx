import React, { useRef, useEffect, useState, useCallback } from 'react'
import * as THREE from 'three'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import {
  RotateCw,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Target,
  Palette,
  Settings,
} from 'lucide-react'

// 情感状态接口定义
export interface EmotionState {
  emotion: string
  intensity: number
  valence: number // 效价 (-1 到 1)
  arousal: number // 唤醒度 (0 到 1)
  dominance: number // 支配性 (-1 到 1)
  confidence: number // 置信度 (0 到 1)
  timestamp: Date
}

// 组件Props接口
interface Emotion3DVisualizerProps {
  emotionalState?: EmotionState
  history?: EmotionState[]
  className?: string
  width?: number
  height?: number
  showControls?: boolean
  showAxisLabels?: boolean
  showTrajectory?: boolean
  interactive?: boolean
  onEmotionClick?: (emotion: EmotionState) => void
}

// 情感颜色映射
const getEmotionColor = (emotion: string): number => {
  const colorMap: Record<string, number> = {
    happiness: 0xffd700, // 金黄色
    joy: 0xffd700,
    sadness: 0x4169e1, // 皇家蓝
    sorrow: 0x4169e1,
    anger: 0xff4500, // 橙红色
    rage: 0xff0000,
    fear: 0x800080, // 紫色
    anxiety: 0x9370db,
    surprise: 0xff69b4, // 热粉色
    amazement: 0xff1493,
    disgust: 0x228b22, // 森林绿
    contempt: 0x556b2f,
    neutral: 0x808080, // 灰色
    calm: 0x87ceeb,
    excited: 0xff6347, // 番茄红
    relaxed: 0x90ee90, // 淡绿色
  }
  return colorMap[emotion.toLowerCase()] || 0x808080
}

// 3D情感可视化器组件
export const Emotion3DVisualizer: React.FC<Emotion3DVisualizerProps> = ({
  emotionalState,
  history = [],
  className,
  width = 600,
  height = 400,
  showControls = true,
  showAxisLabels = true,
  showTrajectory = true,
  interactive = true,
  onEmotionClick,
}) => {
  const mountRef = useRef<HTMLDivElement>(null)
  const sceneRef = useRef<THREE.Scene>()
  const rendererRef = useRef<THREE.WebGLRenderer>()
  const cameraRef = useRef<THREE.PerspectiveCamera>()
  const animationIdRef = useRef<number>()
  const emotionSphereRef = useRef<THREE.Mesh>()
  const trajectoryLineRef = useRef<THREE.Line>()
  const controlsRef = useRef<any>()

  const [isPlaying, setIsPlaying] = useState(true)
  const [showWireframe, setShowWireframe] = useState(false)
  const [currentHistoryIndex, setCurrentHistoryIndex] = useState(-1)

  // 初始化3D场景
  const initializeScene = useCallback(() => {
    if (!mountRef.current) return

    // 创建场景
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0xf5f5f5)

    // 创建摄像机
    const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000)
    camera.position.set(10, 10, 10)
    camera.lookAt(0, 0, 0)

    // 创建渲染器
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
    })
    renderer.setSize(width, height)
    renderer.setClearColor(0x000000, 0)
    renderer.shadowMap.enabled = true
    renderer.shadowMap.type = THREE.PCFSoftShadowMap

    // 添加环境光
    const ambientLight = new THREE.AmbientLight(0x404040, 0.6)
    scene.add(ambientLight)

    // 添加方向光
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8)
    directionalLight.position.set(10, 10, 5)
    directionalLight.castShadow = true
    scene.add(directionalLight)

    // 创建坐标轴
    if (showAxisLabels) {
      createAxes(scene)
    }

    // 创建网格地面
    const gridHelper = new THREE.GridHelper(10, 10, 0xcccccc, 0xcccccc)
    gridHelper.position.y = -5
    scene.add(gridHelper)

    // 添加到DOM
    mountRef.current.appendChild(renderer.domElement)

    // 保存引用
    sceneRef.current = scene
    rendererRef.current = renderer
    cameraRef.current = camera

    // 添加轨道控制器 (简化版本，不依赖外部库)
    if (interactive) {
      addSimpleControls(camera, renderer.domElement)
    }

    return { scene, camera, renderer }
  }, [width, height, showAxisLabels, interactive])

  // 创建坐标轴
  const createAxes = (scene: THREE.Scene) => {
    const axisLength = 6

    // X轴: Valence (效价) - 红色
    const xGeometry = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-axisLength, 0, 0),
      new THREE.Vector3(axisLength, 0, 0),
    ])
    const xMaterial = new THREE.LineBasicMaterial({ color: 0xff0000 })
    const xAxis = new THREE.Line(xGeometry, xMaterial)
    scene.add(xAxis)

    // Y轴: Arousal (唤醒度) - 绿色
    const yGeometry = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, -axisLength, 0),
      new THREE.Vector3(0, axisLength, 0),
    ])
    const yMaterial = new THREE.LineBasicMaterial({ color: 0x00ff00 })
    const yAxis = new THREE.Line(yGeometry, yMaterial)
    scene.add(yAxis)

    // Z轴: Dominance (支配性) - 蓝色
    const zGeometry = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 0, -axisLength),
      new THREE.Vector3(0, 0, axisLength),
    ])
    const zMaterial = new THREE.LineBasicMaterial({ color: 0x0000ff })
    const zAxis = new THREE.Line(zGeometry, zMaterial)
    scene.add(zAxis)

    // 添加轴标签 (使用简单的几何体表示)
    const createAxisLabel = (position: THREE.Vector3, color: number) => {
      const geometry = new THREE.SphereGeometry(0.1)
      const material = new THREE.MeshBasicMaterial({ color })
      const sphere = new THREE.Mesh(geometry, material)
      sphere.position.copy(position)
      scene.add(sphere)
    }

    createAxisLabel(new THREE.Vector3(axisLength + 0.5, 0, 0), 0xff0000) // X轴标签
    createAxisLabel(new THREE.Vector3(0, axisLength + 0.5, 0), 0x00ff00) // Y轴标签
    createAxisLabel(new THREE.Vector3(0, 0, axisLength + 0.5), 0x0000ff) // Z轴标签
  }

  // 添加简单的鼠标控制
  const addSimpleControls = (
    camera: THREE.PerspectiveCamera,
    domElement: HTMLElement
  ) => {
    let isMouseDown = false
    let mouseX = 0
    let mouseY = 0

    const onMouseDown = (event: MouseEvent) => {
      isMouseDown = true
      mouseX = event.clientX
      mouseY = event.clientY
    }

    const onMouseUp = () => {
      isMouseDown = false
    }

    const onMouseMove = (event: MouseEvent) => {
      if (!isMouseDown) return

      const deltaX = event.clientX - mouseX
      const deltaY = event.clientY - mouseY

      // 简单的轨道控制
      const spherical = new THREE.Spherical()
      spherical.setFromVector3(camera.position)
      spherical.theta -= deltaX * 0.01
      spherical.phi += deltaY * 0.01
      spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi))

      camera.position.setFromSpherical(spherical)
      camera.lookAt(0, 0, 0)

      mouseX = event.clientX
      mouseY = event.clientY
    }

    domElement.addEventListener('mousedown', onMouseDown)
    domElement.addEventListener('mouseup', onMouseUp)
    domElement.addEventListener('mousemove', onMouseMove)

    // 保存清理函数
    controlsRef.current = () => {
      domElement.removeEventListener('mousedown', onMouseDown)
      domElement.removeEventListener('mouseup', onMouseUp)
      domElement.removeEventListener('mousemove', onMouseMove)
    }
  }

  // 更新情感点
  const updateEmotionSphere = useCallback(
    (state: EmotionState) => {
      if (!sceneRef.current) return

      // 移除旧的情感球
      if (emotionSphereRef.current) {
        sceneRef.current.remove(emotionSphereRef.current)
      }

      // 创建新的情感球
      const geometry = new THREE.SphereGeometry(
        0.3 + state.intensity * 0.5, // 根据强度调整大小
        32,
        32
      )

      const material = new THREE.MeshLambertMaterial({
        color: getEmotionColor(state.emotion),
        transparent: true,
        opacity: 0.7 + state.confidence * 0.3, // 根据置信度调整透明度
        wireframe: showWireframe,
      })

      const emotionSphere = new THREE.Mesh(geometry, material)

      // 设置位置 (映射到3D空间)
      emotionSphere.position.set(
        state.valence * 5, // X轴: 效价
        (state.arousal - 0.5) * 10, // Y轴: 唤醒度 (调整到-5到5范围)
        state.dominance * 5 // Z轴: 支配性
      )

      emotionSphere.castShadow = true
      emotionSphere.receiveShadow = true

      // 添加点击事件
      if (onEmotionClick) {
        emotionSphere.userData = { emotion: state }
      }

      sceneRef.current.add(emotionSphere)
      emotionSphereRef.current = emotionSphere
    },
    [showWireframe, onEmotionClick]
  )

  // 更新轨迹路径
  const updateTrajectory = useCallback(() => {
    if (!sceneRef.current || !showTrajectory || history.length < 2) return

    // 移除旧的轨迹线
    if (trajectoryLineRef.current) {
      sceneRef.current.remove(trajectoryLineRef.current)
    }

    // 创建轨迹点
    const points = history.map(
      state =>
        new THREE.Vector3(
          state.valence * 5,
          (state.arousal - 0.5) * 10,
          state.dominance * 5
        )
    )

    const geometry = new THREE.BufferGeometry().setFromPoints(points)
    const material = new THREE.LineBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.6,
    })

    const trajectoryLine = new THREE.Line(geometry, material)
    sceneRef.current.add(trajectoryLine)
    trajectoryLineRef.current = trajectoryLine
  }, [history, showTrajectory])

  // 渲染循环
  const animate = useCallback(() => {
    if (!rendererRef.current || !sceneRef.current || !cameraRef.current) return

    // 更新控制器 (如果有)
    if (
      controlsRef.current &&
      typeof controlsRef.current.update === 'function'
    ) {
      controlsRef.current.update()
    }

    // 渲染场景
    rendererRef.current.render(sceneRef.current, cameraRef.current)

    if (isPlaying) {
      animationIdRef.current = requestAnimationFrame(animate)
    }
  }, [isPlaying])

  // 初始化和清理
  useEffect(() => {
    const result = initializeScene()
    if (result) {
      animate()
    }

    return () => {
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current)
      }
      if (controlsRef.current && typeof controlsRef.current === 'function') {
        controlsRef.current()
      }
      if (mountRef.current && rendererRef.current) {
        mountRef.current.removeChild(rendererRef.current.domElement)
      }
    }
  }, [initializeScene, animate])

  // 更新情感状态
  useEffect(() => {
    if (emotionalState) {
      updateEmotionSphere(emotionalState)
    }
  }, [emotionalState, updateEmotionSphere])

  // 更新轨迹
  useEffect(() => {
    updateTrajectory()
  }, [history, updateTrajectory])

  // 控制按钮处理
  const handleResetCamera = () => {
    if (cameraRef.current) {
      cameraRef.current.position.set(10, 10, 10)
      cameraRef.current.lookAt(0, 0, 0)
    }
  }

  const handleToggleWireframe = () => {
    setShowWireframe(!showWireframe)
  }

  const handleToggleAnimation = () => {
    setIsPlaying(!isPlaying)
    if (!isPlaying) {
      animate()
    }
  }

  return (
    <Card className={cn('relative overflow-hidden', className)}>
      <CardContent className="p-0">
        {/* 3D渲染区域 */}
        <div
          ref={mountRef}
          className="relative bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800"
          style={{ width, height }}
        />

        {/* 情感信息覆盖层 */}
        {emotionalState && (
          <div className="absolute top-4 left-4 bg-white/90 dark:bg-gray-800/90 rounded-lg p-3 shadow-lg">
            <h3 className="font-semibold text-sm mb-2">当前情感状态</h3>
            <div className="space-y-1 text-xs">
              <div className="flex items-center gap-2">
                <span>情感:</span>
                <Badge variant="secondary">{emotionalState.emotion}</Badge>
              </div>
              <div>强度: {(emotionalState.intensity * 100).toFixed(1)}%</div>
              <div>效价: {emotionalState.valence.toFixed(2)}</div>
              <div>唤醒度: {emotionalState.arousal.toFixed(2)}</div>
              <div>支配性: {emotionalState.dominance.toFixed(2)}</div>
              <div>置信度: {(emotionalState.confidence * 100).toFixed(1)}%</div>
            </div>
          </div>
        )}

        {/* 控制面板 */}
        {showControls && (
          <div className="absolute top-4 right-4 bg-white/90 dark:bg-gray-800/90 rounded-lg p-2 shadow-lg">
            <div className="flex flex-col gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={handleResetCamera}
                className="justify-start"
              >
                <Target className="h-4 w-4 mr-1" />
                重置视角
              </Button>

              <Button
                variant="ghost"
                size="sm"
                onClick={handleToggleWireframe}
                className="justify-start"
              >
                <Palette className="h-4 w-4 mr-1" />
                {showWireframe ? '实体' : '线框'}
              </Button>

              <Button
                variant="ghost"
                size="sm"
                onClick={handleToggleAnimation}
                className="justify-start"
              >
                {isPlaying ? (
                  <>
                    <RotateCcw className="h-4 w-4 mr-1" />
                    暂停
                  </>
                ) : (
                  <>
                    <RotateCw className="h-4 w-4 mr-1" />
                    播放
                  </>
                )}
              </Button>
            </div>
          </div>
        )}

        {/* 坐标轴说明 */}
        {showAxisLabels && (
          <div className="absolute bottom-4 left-4 bg-white/90 dark:bg-gray-800/90 rounded-lg p-3 shadow-lg">
            <h4 className="font-semibold text-xs mb-2">坐标轴说明</h4>
            <div className="space-y-1 text-xs">
              <div className="flex items-center gap-2">
                <div className="w-3 h-0.5 bg-red-500"></div>
                <span>X轴: 效价 (负面 ↔ 正面)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-0.5 bg-green-500"></div>
                <span>Y轴: 唤醒度 (平静 ↔ 激动)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-0.5 bg-blue-500"></div>
                <span>Z轴: 支配性 (被动 ↔ 主动)</span>
              </div>
            </div>
          </div>
        )}

        {/* 性能统计 */}
        <div className="absolute bottom-4 right-4 text-xs text-muted-foreground bg-white/70 dark:bg-gray-800/70 rounded px-2 py-1">
          轨迹点: {history.length}
        </div>
      </CardContent>
    </Card>
  )
}

export default Emotion3DVisualizer
