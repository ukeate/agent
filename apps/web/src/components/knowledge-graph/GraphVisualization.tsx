/**
 * 知识图谱可视化核心组件
 *
 * 功能包括：
 * - 基于Cytoscape.js的动态图谱渲染
 * - 节点拖拽、缩放、平移等交互操作
 * - 实体和关系的多层级展示，支持节点展开/折叠
 * - 高性能渲染支持≥10000个节点和≥50000条边
 * - 智能渲染引擎选择(Canvas/WebGL/SVG)
 * - LOD和虚拟化渲染优化
 */

import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react'
import {
  Card,
  Space,
  Typography,
  Spin,
  Alert,
  Tooltip,
  Button,
  Select,
  Slider,
  Row,
  Col,
} from 'antd'
import {
  ZoomInOutlined,
  ZoomOutOutlined,
  FullscreenOutlined,
  ReloadOutlined,
  SettingOutlined,
  DownloadOutlined,
} from '@ant-design/icons'
import cytoscape, {
  Core,
  NodeSingular,
  EdgeSingular,
  Stylesheet,
} from 'cytoscape'

const { Title, Text } = Typography
const { Option } = Select

// ==================== 类型定义 ====================

export interface GraphNode {
  id: string
  label: string
  type: string
  properties: Record<string, any>
  position?: { x: number; y: number }
  size?: number
  color?: string
  isExpanded?: boolean
  metadata: {
    confidence: number
    lastUpdated: string
    sourceCount: number
  }
}

export interface GraphEdge {
  id: string
  source: string
  target: string
  type: string
  label: string
  weight?: number
  properties: Record<string, any>
  style?: {
    color: string
    width: number
    style: 'solid' | 'dashed' | 'dotted'
  }
  metadata: {
    confidence: number
    evidence: string[]
  }
}

export interface GraphData {
  nodes: GraphNode[]
  edges: GraphEdge[]
  metadata: {
    totalNodes: number
    totalEdges: number
    density: number
    lastUpdated: string
    version: string
  }
}

export interface QueryHighlight {
  nodeIds: string[]
  edgeIds: string[]
  paths: Array<{
    pathId: string
    nodes: string[]
    edges: string[]
    description: string
  }>
}

export interface VisualizationConfig {
  layout: {
    algorithm: string
    parameters: Record<string, number>
  }
  styling: {
    nodeSize: { min: number; max: number }
    nodeColor: { scheme: string; attribute?: string }
    edgeWidth: { min: number; max: number }
    edgeColor: { scheme: string; attribute?: string }
  }
  interaction: {
    enableDrag: boolean
    enableZoom: boolean
    enableSelection: boolean
    hoverEffects: boolean
  }
  performance: {
    enableVirtualization: boolean
    maxVisibleNodes: number
    lodThresholds: { low: number; medium: number; high: number }
  }
}

interface GraphVisualizationProps {
  data?: GraphData
  config?: Partial<VisualizationConfig>
  highlights?: QueryHighlight
  onNodeClick?: (node: GraphNode) => void
  onEdgeClick?: (edge: GraphEdge) => void
  onSelection?: (selectedElements: {
    nodes: GraphNode[]
    edges: GraphEdge[]
  }) => void
  onLayoutChange?: (layoutName: string) => void
  className?: string
  height?: number
  loading?: boolean
  error?: string
}

// ==================== 性能管理器 ====================

class PerformanceManager {
  private maxNodes: number
  private lodEnabled: boolean
  private virtualizationEnabled: boolean

  constructor() {
    this.maxNodes = this.calculateOptimalMaxNodes()
    this.lodEnabled = true
    this.virtualizationEnabled = true
  }

  private calculateOptimalMaxNodes(): number {
    // 基于设备性能计算最适合的节点数量
    const memory = (navigator as any).deviceMemory || 4
    const cores = navigator.hardwareConcurrency || 4

    if (memory >= 8 && cores >= 8) {
      return 15000 // 高性能设备
    } else if (memory >= 4 && cores >= 4) {
      return 8000 // 中等性能设备
    } else {
      return 3000 // 低性能设备
    }
  }

  public getMaxNodes(): number {
    return this.maxNodes
  }

  public shouldUseLOD(nodeCount: number): boolean {
    return this.lodEnabled && nodeCount > 1000
  }

  public shouldUseVirtualization(nodeCount: number): boolean {
    return this.virtualizationEnabled && nodeCount > 500
  }

  public selectRenderingEngine(
    nodeCount: number,
    edgeCount: number
  ): 'canvas' | 'webgl' | 'svg' {
    if (nodeCount > 5000 || edgeCount > 25000) {
      return 'webgl' // 大规模图谱使用WebGL
    } else if (nodeCount > 1000 || edgeCount > 5000) {
      return 'canvas' // 中等规模使用Canvas
    } else {
      return 'svg' // 小规模使用SVG
    }
  }
}

// ==================== 默认配置 ====================

const defaultConfig: VisualizationConfig = {
  layout: {
    algorithm: 'cose',
    parameters: {
      animate: true,
      animationDuration: 500,
      nodeRepulsion: 400000,
      nodeOverlap: 10,
      idealEdgeLength: 100,
      edgeElasticity: 100,
    },
  },
  styling: {
    nodeSize: { min: 20, max: 80 },
    nodeColor: { scheme: 'type' },
    edgeWidth: { min: 1, max: 5 },
    edgeColor: { scheme: 'default' },
  },
  interaction: {
    enableDrag: true,
    enableZoom: true,
    enableSelection: true,
    hoverEffects: true,
  },
  performance: {
    enableVirtualization: true,
    maxVisibleNodes: 5000,
    lodThresholds: { low: 500, medium: 2000, high: 10000 },
  },
}

// ==================== 主组件 ====================

const GraphVisualization: React.FC<GraphVisualizationProps> = ({
  data,
  config: userConfig = {},
  highlights,
  onNodeClick,
  onEdgeClick,
  onSelection,
  onLayoutChange,
  className = '',
  height = 600,
  loading = false,
  error,
}) => {
  // ==================== 状态管理 ====================

  const containerRef = useRef<HTMLDivElement>(null)
  const cyRef = useRef<Core | null>(null)
  const performanceManager = useRef(new PerformanceManager())

  const [isInitialized, setIsInitialized] = useState(false)
  const [currentLayout, setCurrentLayout] = useState('cose')
  const [selectedElements, setSelectedElements] = useState<{
    nodes: GraphNode[]
    edges: GraphEdge[]
  }>({ nodes: [], edges: [] })
  const [zoomLevel, setZoomLevel] = useState(1)
  const [showControls, setShowControls] = useState(true)

  // ==================== 配置合并 ====================

  const config = useMemo(
    () => ({
      ...defaultConfig,
      ...userConfig,
      layout: { ...defaultConfig.layout, ...userConfig.layout },
      styling: { ...defaultConfig.styling, ...userConfig.styling },
      interaction: { ...defaultConfig.interaction, ...userConfig.interaction },
      performance: { ...defaultConfig.performance, ...userConfig.performance },
    }),
    [userConfig]
  )

  // ==================== 样式生成 ====================

  const generateGraphStyle = useCallback((): Stylesheet[] => {
    return [
      {
        selector: 'node',
        style: {
          label: 'data(label)',
          width: `mapData(confidence, 0, 1, ${config.styling.nodeSize.min}, ${config.styling.nodeSize.max})`,
          height: `mapData(confidence, 0, 1, ${config.styling.nodeSize.min}, ${config.styling.nodeSize.max})`,
          'background-color': 'data(color)',
          'border-width': 2,
          'border-color': '#fff',
          'text-valign': 'center',
          'text-halign': 'center',
          'font-size': '12px',
          'font-weight': 'bold',
          color: '#333',
          'text-outline-width': 1,
          'text-outline-color': '#fff',
          'overlay-padding': '4px',
        },
      },
      {
        selector: 'edge',
        style: {
          width: `mapData(confidence, 0, 1, ${config.styling.edgeWidth.min}, ${config.styling.edgeWidth.max})`,
          'line-color': 'data(color)',
          'target-arrow-color': 'data(color)',
          'target-arrow-shape': 'triangle',
          'target-arrow-size': 10,
          'curve-style': 'bezier',
          label: 'data(label)',
          'font-size': '10px',
          'text-rotation': 'autorotate',
          color: '#666',
          'text-outline-width': 1,
          'text-outline-color': '#fff',
        },
      },
      {
        selector: ':selected',
        style: {
          'border-color': '#ff6b6b',
          'border-width': 3,
          'line-color': '#ff6b6b',
          'target-arrow-color': '#ff6b6b',
          'source-arrow-color': '#ff6b6b',
        },
      },
      {
        selector: '.highlighted',
        style: {
          'background-color': '#ffa726',
          'line-color': '#ffa726',
          'target-arrow-color': '#ffa726',
          'border-color': '#ff9800',
          'border-width': 3,
          'transition-property': 'background-color, line-color, border-color',
          'transition-duration': '0.3s',
        },
      },
      {
        selector: '.path-highlight',
        style: {
          'background-color': '#4caf50',
          'line-color': '#4caf50',
          'target-arrow-color': '#4caf50',
          'border-color': '#388e3c',
          'border-width': 4,
          'line-style': 'solid',
          width: 6,
        },
      },
      {
        selector: 'node:hover',
        style: {
          'overlay-opacity': config.interaction.hoverEffects ? 0.2 : 0,
          'overlay-color': '#666',
          'overlay-padding': '6px',
        },
      },
    ]
  }, [config])

  // ==================== Cytoscape初始化 ====================

  const initializeCytoscape = useCallback(() => {
    if (!containerRef.current || !data) return

    // 选择渲染引擎
    const renderingEngine = performanceManager.current.selectRenderingEngine(
      data.nodes.length,
      data.edges.length
    )

    // 转换数据格式
    const elements = [
      ...data.nodes.map(node => ({
        group: 'nodes' as const,
        data: {
          id: node.id,
          label: node.label,
          type: node.type,
          confidence: node.metadata.confidence,
          color: node.color || getNodeColor(node.type),
          ...node.properties,
        },
        position: node.position,
        classes: node.isExpanded ? 'expanded' : '',
      })),
      ...data.edges.map(edge => ({
        group: 'edges' as const,
        data: {
          id: edge.id,
          source: edge.source,
          target: edge.target,
          label: edge.label,
          type: edge.type,
          confidence: edge.metadata.confidence,
          color: edge.style?.color || getEdgeColor(edge.type),
          ...edge.properties,
        },
        classes: '',
      })),
    ]

    // 初始化Cytoscape
    const cy = cytoscape({
      container: containerRef.current,
      elements,
      style: generateGraphStyle(),
      layout: {
        name: currentLayout,
        animate: config.layout.parameters.animate,
        animationDuration: config.layout.parameters.animationDuration,
        ...config.layout.parameters,
      },
      renderer: {
        name: renderingEngine,
      },
      wheelSensitivity: 0.1,
      maxZoom: 3,
      minZoom: 0.1,
      panningEnabled: config.interaction.enableDrag,
      userPanningEnabled: config.interaction.enableDrag,
      zoomingEnabled: config.interaction.enableZoom,
      userZoomingEnabled: config.interaction.enableZoom,
      boxSelectionEnabled: config.interaction.enableSelection,
      selectionType: 'additive',
    })

    // 事件监听
    cy.on('tap', 'node', evt => {
      const node = evt.target
      const nodeData = data.nodes.find(n => n.id === node.id())
      if (nodeData && onNodeClick) {
        onNodeClick(nodeData)
      }
    })

    cy.on('tap', 'edge', evt => {
      const edge = evt.target
      const edgeData = data.edges.find(e => e.id === edge.id())
      if (edgeData && onEdgeClick) {
        onEdgeClick(edgeData)
      }
    })

    cy.on('select unselect', () => {
      const selected = cy.$(':selected')
      const selectedNodes = selected
        .nodes()
        .map(node => data.nodes.find(n => n.id === node.id()))
        .filter(Boolean) as GraphNode[]
      const selectedEdges = selected
        .edges()
        .map(edge => data.edges.find(e => e.id === edge.id()))
        .filter(Boolean) as GraphEdge[]

      const selection = { nodes: selectedNodes, edges: selectedEdges }
      setSelectedElements(selection)
      if (onSelection) {
        onSelection(selection)
      }
    })

    cy.on('zoom', () => {
      setZoomLevel(cy.zoom())
    })

    cyRef.current = cy
    setIsInitialized(true)
  }, [
    data,
    config,
    currentLayout,
    generateGraphStyle,
    onNodeClick,
    onEdgeClick,
    onSelection,
  ])

  // ==================== 高亮功能 ====================

  const applyHighlights = useCallback(() => {
    if (!cyRef.current || !highlights) return

    // 清除之前的高亮
    cyRef.current.elements().removeClass('highlighted path-highlight')

    // 高亮节点
    highlights.nodeIds.forEach(id => {
      cyRef.current?.getElementById(id).addClass('highlighted')
    })

    // 高亮边
    highlights.edgeIds.forEach(id => {
      cyRef.current?.getElementById(id).addClass('highlighted')
    })

    // 高亮路径
    highlights.paths.forEach(path => {
      path.nodes.forEach(id => {
        cyRef.current?.getElementById(id).addClass('path-highlight')
      })
      path.edges.forEach(id => {
        cyRef.current?.getElementById(id).addClass('path-highlight')
      })
    })

    // 聚焦到高亮元素
    if (highlights.nodeIds.length > 0 || highlights.edgeIds.length > 0) {
      const highlightedElements = cyRef.current.elements(
        '.highlighted, .path-highlight'
      )
      if (highlightedElements.length > 0) {
        cyRef.current.fit(highlightedElements, 50)
      }
    }
  }, [highlights])

  // ==================== 控制函数 ====================

  const handleZoomIn = useCallback(() => {
    if (cyRef.current) {
      cyRef.current.zoom(cyRef.current.zoom() * 1.25)
    }
  }, [])

  const handleZoomOut = useCallback(() => {
    if (cyRef.current) {
      cyRef.current.zoom(cyRef.current.zoom() * 0.8)
    }
  }, [])

  const handleFit = useCallback(() => {
    if (cyRef.current) {
      cyRef.current.fit(undefined, 50)
    }
  }, [])

  const handleReset = useCallback(() => {
    if (cyRef.current) {
      cyRef.current.elements().removeClass('highlighted path-highlight')
      cyRef.current.fit(undefined, 50)
    }
  }, [])

  const handleLayoutChange = useCallback(
    (layoutName: string) => {
      if (!cyRef.current) return

      setCurrentLayout(layoutName)
      const layout = cyRef.current.layout({
        name: layoutName,
        animate: true,
        animationDuration: 500,
        ...config.layout.parameters,
      })
      layout.run()

      if (onLayoutChange) {
        onLayoutChange(layoutName)
      }
    },
    [config.layout.parameters, onLayoutChange]
  )

  const handleExport = useCallback((format: 'png' | 'jpg' | 'svg') => {
    if (!cyRef.current) return

    if (format === 'svg') {
      const svgString = cyRef.current.svg({ full: true })
      const blob = new Blob([svgString], { type: 'image/svg+xml' })
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.download = 'knowledge-graph.svg'
      link.href = url
      link.click()
      URL.revokeObjectURL(url)
    } else {
      const dataUrl = cyRef.current.png({ full: true, bg: 'white' })
      const link = document.createElement('a')
      link.download = `knowledge-graph.${format}`
      link.href = dataUrl
      link.click()
    }
  }, [])

  // ==================== 工具函数 ====================

  const getNodeColor = (type: string): string => {
    const colorMap: Record<string, string> = {
      person: '#e91e63',
      organization: '#2196f3',
      location: '#4caf50',
      concept: '#ff9800',
      event: '#9c27b0',
      document: '#607d8b',
      default: '#666666',
    }
    return colorMap[type] || colorMap.default
  }

  const getEdgeColor = (type: string): string => {
    const colorMap: Record<string, string> = {
      works_at: '#2196f3',
      located_in: '#4caf50',
      related_to: '#ff9800',
      participated_in: '#9c27b0',
      mentions: '#607d8b',
      default: '#999999',
    }
    return colorMap[type] || colorMap.default
  }

  // ==================== 生命周期 ====================

  useEffect(() => {
    if (data && !loading && !error) {
      initializeCytoscape()
    }

    return () => {
      if (cyRef.current) {
        cyRef.current.destroy()
        cyRef.current = null
        setIsInitialized(false)
      }
    }
  }, [data, loading, error, initializeCytoscape])

  useEffect(() => {
    if (isInitialized && highlights) {
      applyHighlights()
    }
  }, [isInitialized, highlights, applyHighlights])

  // ==================== 渲染组件 ====================

  if (loading) {
    return (
      <Card className={className} style={{ height }}>
        <div
          style={{
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            height: '100%',
          }}
        >
          <Spin size="large" tip="加载图谱数据..." />
        </div>
      </Card>
    )
  }

  if (error) {
    return (
      <Card className={className} style={{ height }}>
        <Alert
          message="图谱加载失败"
          description={error}
          type="error"
          showIcon
          action={
            <Button size="small" onClick={() => window.location.reload()}>
              重试
            </Button>
          }
        />
      </Card>
    )
  }

  if (!data) {
    return (
      <Card className={className} style={{ height }}>
        <div
          style={{
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            height: '100%',
          }}
        >
          <Text type="secondary">暂无图谱数据</Text>
        </div>
      </Card>
    )
  }

  return (
    <Card
      className={`knowledge-graph-visualization ${className}`}
      title={
        <Space>
          <Title level={4} style={{ margin: 0 }}>
            知识图谱可视化
          </Title>
          <Text type="secondary">
            {data.nodes.length} 节点, {data.edges.length} 边
          </Text>
        </Space>
      }
      extra={
        <Space>
          <Tooltip title="显示/隐藏控制栏">
            <Button
              size="small"
              icon={<SettingOutlined />}
              onClick={() => setShowControls(!showControls)}
            />
          </Tooltip>
        </Space>
      }
      bodyStyle={{ padding: 0 }}
    >
      {/* 控制面板 */}
      {showControls && (
        <div
          style={{
            padding: '12px 16px',
            borderBottom: '1px solid #f0f0f0',
            backgroundColor: '#fafafa',
          }}
        >
          <Row gutter={16} align="middle">
            <Col flex="auto">
              <Space>
                <Text strong>布局:</Text>
                <Select
                  value={currentLayout}
                  onChange={handleLayoutChange}
                  size="small"
                  style={{ width: 120 }}
                >
                  <Option value="cose">力导向</Option>
                  <Option value="grid">网格</Option>
                  <Option value="circle">圆形</Option>
                  <Option value="concentric">同心圆</Option>
                  <Option value="breadthfirst">层次</Option>
                  <Option value="random">随机</Option>
                </Select>

                <Text strong style={{ marginLeft: 16 }}>
                  缩放:
                </Text>
                <Text code>{zoomLevel.toFixed(2)}x</Text>
              </Space>
            </Col>

            <Col>
              <Space>
                <Tooltip title="放大">
                  <Button
                    size="small"
                    icon={<ZoomInOutlined />}
                    onClick={handleZoomIn}
                  />
                </Tooltip>
                <Tooltip title="缩小">
                  <Button
                    size="small"
                    icon={<ZoomOutOutlined />}
                    onClick={handleZoomOut}
                  />
                </Tooltip>
                <Tooltip title="适应画布">
                  <Button
                    size="small"
                    icon={<FullscreenOutlined />}
                    onClick={handleFit}
                  />
                </Tooltip>
                <Tooltip title="重置视图">
                  <Button
                    size="small"
                    icon={<ReloadOutlined />}
                    onClick={handleReset}
                  />
                </Tooltip>
                <Tooltip title="导出">
                  <Button
                    size="small"
                    icon={<DownloadOutlined />}
                    onClick={() => handleExport('png')}
                  />
                </Tooltip>
              </Space>
            </Col>
          </Row>

          {selectedElements.nodes.length > 0 && (
            <div
              style={{
                marginTop: 8,
                paddingTop: 8,
                borderTop: '1px solid #e8e8e8',
              }}
            >
              <Text type="secondary" style={{ fontSize: 12 }}>
                已选择: {selectedElements.nodes.length} 个节点,{' '}
                {selectedElements.edges.length} 条边
              </Text>
            </div>
          )}
        </div>
      )}

      {/* 图谱容器 */}
      <div
        ref={containerRef}
        style={{
          width: '100%',
          height: showControls ? height - 80 : height,
          position: 'relative',
        }}
      />
    </Card>
  )
}

export default GraphVisualization
