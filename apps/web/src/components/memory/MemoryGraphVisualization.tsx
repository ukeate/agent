/**
 * 记忆关联图可视化组件
 * 使用D3.js展示记忆之间的网络关系
 */
import React, { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'
import { Card, Select, Slider, Space, Tag, Button } from 'antd'
import { 
  PartitionOutlined, 
  ZoomInOutlined, 
  ZoomOutOutlined,
  ReloadOutlined 
} from '@ant-design/icons'
import { Memory, MemoryType } from '@/types/memory'
import { memoryService } from '@/services/memoryService'

interface GraphNode extends d3.SimulationNodeDatum {
  id: string
  type: MemoryType
  importance: number
  label: string
  content?: string
  access_count?: number
}

interface GraphLink extends d3.SimulationLinkDatum<GraphNode> {
  weight: number
  type: string
}

interface Props {
  memories: Memory[]
  onNodeClick?: (memory: Memory) => void
}

const MemoryGraphVisualization: React.FC<Props> = ({ memories, onNodeClick }) => {
  const svgRef = useRef<SVGSVGElement>(null)
  const [zoom, setZoom] = useState(1)
  const [minWeight, setMinWeight] = useState(0.3)
  const [selectedType, setSelectedType] = useState<MemoryType | 'all'>('all')
  const [graphStats, setGraphStats] = useState<any>(null)

  useEffect(() => {
    if (memories.length > 0) {
      drawGraph()
      loadGraphStats()
    }
  }, [memories, minWeight, selectedType])

  const loadGraphStats = async () => {
    try {
      const stats = await memoryService.getGraphStatistics()
      setGraphStats(stats)
    } catch (error) {
      console.error('加载图统计失败:', error)
    }
  }

  const drawGraph = () => {
    if (!svgRef.current) return

    // 清空现有内容
    d3.select(svgRef.current).selectAll('*').remove()

    // 过滤记忆
    const filteredMemories = selectedType === 'all' 
      ? memories 
      : memories.filter(m => m.type === selectedType)

    // 构建节点和边
    const nodes: GraphNode[] = filteredMemories.map(m => ({
      id: m.id,
      type: m.type,
      importance: m.importance,
      label: m.content.substring(0, 30) + '...',
      content: m.content,
      access_count: m.access_count
    }))

    const links: GraphLink[] = []
    
    // 根据相关性创建边（示例：基于相似度）
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        // 简化的相似度计算
        const similarity = Math.random() * 0.8
        if (similarity > minWeight) {
          links.push({
            source: nodes[i].id,
            target: nodes[j].id,
            weight: similarity,
            type: 'similar'
          })
        }
      }
    }

    // 设置SVG尺寸
    const width = 800
    const height = 600
    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height)

    // 创建缩放容器
    const g = svg.append('g')

    // 设置缩放行为
    const zoomBehavior = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 10])
      .on('zoom', (event) => {
        g.attr('transform', event.transform.toString())
        setZoom(event.transform.k)
      })

    svg.call(zoomBehavior)

    // 创建力导向图
    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink<GraphNode, GraphLink>(links)
        .id(d => d.id)
        .distance(d => 100 * (1 - d.weight))
        .strength(d => d.weight))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(d => getNodeRadius(d as GraphNode) + 5))

    // 创建边
    const link = g.append('g')
      .selectAll('line')
      .data(links)
      .enter().append('line')
      .attr('stroke', '#999')
      .attr('stroke-opacity', d => d.weight)
      .attr('stroke-width', d => d.weight * 3)

    // 创建节点组
    const node = g.append('g')
      .selectAll('g')
      .data(nodes)
      .enter().append('g')
      .call(d3.drag<SVGGElement, GraphNode>()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended))

    // 添加节点圆形
    node.append('circle')
      .attr('r', d => getNodeRadius(d))
      .attr('fill', d => getNodeColor(d.type))
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .style('cursor', 'pointer')
      .on('click', (_, d) => {
        const memory = memories.find(m => m.id === d.id)
        if (memory && onNodeClick) {
          onNodeClick(memory)
        }
      })

    // 添加节点标签
    node.append('text')
      .text(d => d.label)
      .attr('x', 0)
      .attr('y', d => -getNodeRadius(d) - 5)
      .attr('text-anchor', 'middle')
      .style('font-size', '10px')
      .style('fill', '#333')

    // 添加工具提示
    node.append('title')
      .text(d => `${d.content}\n重要性: ${d.importance}\n访问次数: ${d.access_count}`)

    // 更新位置
    simulation.on('tick', () => {
      link
        .attr('x1', d => (d.source as GraphNode).x!)
        .attr('y1', d => (d.source as GraphNode).y!)
        .attr('x2', d => (d.target as GraphNode).x!)
        .attr('y2', d => (d.target as GraphNode).y!)

      node.attr('transform', d => `translate(${d.x},${d.y})`)
    })

    // 拖拽函数
    function dragstarted(event: d3.D3DragEvent<SVGGElement, GraphNode, GraphNode>) {
      if (!event.active) simulation.alphaTarget(0.3).restart()
      event.subject.fx = event.subject.x
      event.subject.fy = event.subject.y
    }

    function dragged(event: d3.D3DragEvent<SVGGElement, GraphNode, GraphNode>) {
      event.subject.fx = event.x
      event.subject.fy = event.y
    }

    function dragended(event: d3.D3DragEvent<SVGGElement, GraphNode, GraphNode>) {
      if (!event.active) simulation.alphaTarget(0)
      event.subject.fx = null
      event.subject.fy = null
    }
  }

  const getNodeRadius = (node: GraphNode) => {
    return 10 + node.importance * 20
  }

  const getNodeColor = (type: MemoryType) => {
    switch (type) {
      case MemoryType.WORKING:
        return '#52c41a'
      case MemoryType.EPISODIC:
        return '#1890ff'
      case MemoryType.SEMANTIC:
        return '#722ed1'
      default:
        return '#666'
    }
  }

  const handleZoomIn = () => {
    const svg = d3.select(svgRef.current)
    svg.transition().call(
      d3.zoom<SVGSVGElement, unknown>().scaleTo as any,
      zoom * 1.2
    )
  }

  const handleZoomOut = () => {
    const svg = d3.select(svgRef.current)
    svg.transition().call(
      d3.zoom<SVGSVGElement, unknown>().scaleTo as any,
      zoom * 0.8
    )
  }

  const handleReset = () => {
    const svg = d3.select(svgRef.current)
    svg.transition().call(
      d3.zoom<SVGSVGElement, unknown>().transform as any,
      d3.zoomIdentity
    )
    setZoom(1)
  }

  return (
    <Card 
      title={
        <span>
          <PartitionOutlined /> 记忆关联网络
        </span>
      }
      extra={
        <Space>
          <Button icon={<ZoomInOutlined />} onClick={handleZoomIn} />
          <Button icon={<ZoomOutOutlined />} onClick={handleZoomOut} />
          <Button icon={<ReloadOutlined />} onClick={handleReset} />
        </Space>
      }
    >
      {/* 控制面板 */}
      <div style={{ marginBottom: 16 }}>
        <Space>
          <span>记忆类型:</span>
          <Select
            value={selectedType}
            onChange={setSelectedType}
            style={{ width: 120 }}
          >
            <Select.Option value="all">全部</Select.Option>
            <Select.Option value={MemoryType.WORKING}>
              <Tag color="green">工作记忆</Tag>
            </Select.Option>
            <Select.Option value={MemoryType.EPISODIC}>
              <Tag color="blue">情景记忆</Tag>
            </Select.Option>
            <Select.Option value={MemoryType.SEMANTIC}>
              <Tag color="purple">语义记忆</Tag>
            </Select.Option>
          </Select>

          <span style={{ marginLeft: 24 }}>最小权重:</span>
          <Slider
            min={0}
            max={1}
            step={0.1}
            value={minWeight}
            onChange={setMinWeight}
            style={{ width: 200 }}
          />
          <span>{minWeight.toFixed(1)}</span>
        </Space>
      </div>

      {/* 图统计信息 */}
      {graphStats && (
        <div style={{ marginBottom: 16 }}>
          <Space>
            <Tag>节点数: {graphStats.node_count}</Tag>
            <Tag>边数: {graphStats.edge_count}</Tag>
            <Tag>平均度: {graphStats.avg_degree}</Tag>
            <Tag>图密度: {graphStats.density}</Tag>
            <Tag>连通分量: {graphStats.connected_components}</Tag>
          </Space>
        </div>
      )}

      {/* 图例 */}
      <div style={{ marginBottom: 16 }}>
        <Space>
          <span>图例:</span>
          <Tag color="green">工作记忆</Tag>
          <Tag color="blue">情景记忆</Tag>
          <Tag color="purple">语义记忆</Tag>
          <span style={{ marginLeft: 16 }}>节点大小=重要性 | 边粗细=关联强度</span>
        </Space>
      </div>

      {/* SVG画布 */}
      <div style={{ border: '1px solid #f0f0f0', borderRadius: 4, overflow: 'hidden' }}>
        <svg ref={svgRef} style={{ width: '100%', height: 600 }} />
      </div>

      <div style={{ marginTop: 8, color: '#666', fontSize: 12 }}>
        提示: 拖拽节点可调整位置，滚轮缩放视图，点击节点查看详情
      </div>
    </Card>
  )
}

export default MemoryGraphVisualization