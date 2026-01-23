import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useState, useEffect } from 'react'
import { logger } from '../utils/logger'
import {
  Card,
  Tabs,
  Typography,
  Row,
  Col,
  Button,
  Alert,
  Tag,
  Statistic,
  Space,
  Divider,
} from 'antd'
import {
  ClockCircleOutlined,
  BranchesOutlined,
  DatabaseOutlined,
  ReloadOutlined,
  LoadingOutlined,
  NodeIndexOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons'

const { Title, Text } = Typography
const { TabPane } = Tabs

interface VectorClockNode {
  node_id: string
  current_clock: Record<string, number>
  last_updated: string
  status: 'active' | 'inactive' | 'syncing'
  pending_operations: number
}

interface VectorClockEvent {
  event_id: string
  node_id: string
  timestamp: string
  event_type:
    | 'local_update'
    | 'remote_sync'
    | 'conflict_detected'
    | 'conflict_resolved'
  vector_clock: Record<string, number>
  data: Record<string, any>
}

const VectorClockVisualizationPage: React.FC = () => {
  const [nodes, setNodes] = useState<VectorClockNode[]>([])
  const [events, setEvents] = useState<VectorClockEvent[]>([])
  const [loading, setLoading] = useState(true)

  const fetchVectorClockData = async () => {
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/offline/vector-clocks'))
      const data = await res.json()
      setNodes(Array.isArray(data?.nodes) ? data.nodes : [])
      setEvents(Array.isArray(data?.events) ? data.events : [])
    } catch (error) {
      logger.error('获取向量时钟数据失败:', error)
      setNodes([])
      setEvents([])
    }
  }

  useEffect(() => {
    const loadData = async () => {
      setLoading(true)
      await fetchVectorClockData()
      setLoading(false)
    }

    loadData()
    const interval = setInterval(loadData, 5000)
    return () => clearInterval(interval)
  }, [])

  const getNodeStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'success'
      case 'syncing':
        return 'processing'
      case 'inactive':
        return 'error'
      default:
        return 'default'
    }
  }

  const getEventTypeColor = (type: string) => {
    switch (type) {
      case 'local_update':
        return 'blue'
      case 'remote_sync':
        return 'green'
      case 'conflict_detected':
        return 'error'
      case 'conflict_resolved':
        return 'success'
      default:
        return 'default'
    }
  }

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '50px' }}>
        <LoadingOutlined style={{ fontSize: 24 }} />
        <div style={{ marginTop: 16 }}>加载向量时钟数据中...</div>
      </div>
    )
  }

  return (
    <div style={{ padding: '24px' }}>
      <div
        style={{
          marginBottom: '24px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <Title level={2}>
          <ClockCircleOutlined style={{ marginRight: '12px' }} />
          向量时钟可视化
        </Title>
        <Space>
          <Button
            onClick={() => window.location.reload()}
            icon={<ReloadOutlined />}
          >
            刷新
          </Button>
        </Space>
      </div>

      {/* 节点概览 */}
      <Row gutter={24} style={{ marginBottom: '24px' }}>
        <Col span={8}>
          <Card>
            <Statistic
              title="活跃节点"
              value={nodes.filter(n => n.status === 'active').length}
              prefix={<NodeIndexOutlined />}
              suffix={`/ ${nodes.length}`}
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card>
            <Statistic
              title="待处理操作"
              value={nodes.reduce((sum, n) => sum + n.pending_operations, 0)}
              prefix={<ThunderboltOutlined />}
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card>
            <Statistic
              title="最近事件"
              value={events.length}
              prefix={<BranchesOutlined />}
            />
          </Card>
        </Col>
      </Row>

      <Tabs defaultActiveKey="nodes">
        <TabPane
          tab={
            <span>
              <NodeIndexOutlined />
              节点状态
            </span>
          }
          key="nodes"
        >
          <Row gutter={16}>
            {nodes.length === 0 ? (
              <Col span={24}>
                <Alert
                  type="info"
                  message="暂无节点数据，待离线操作产生后再查看。"
                />
              </Col>
            ) : (
              nodes.map(node => (
                <Col key={node.node_id} span={8}>
                  <Card
                    title={
                      <div
                        style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                        }}
                      >
                        <Text strong>{node.node_id}</Text>
                        <Tag color={getNodeStatusColor(node.status)}>
                          {node.status}
                        </Tag>
                      </div>
                    }
                    style={{ marginBottom: '16px' }}
                  >
                    <div style={{ marginBottom: '12px' }}>
                      <Text strong>向量时钟:</Text>
                      <div
                        style={{
                          marginTop: '4px',
                          fontFamily: 'monospace',
                          backgroundColor: '#f5f5f5',
                          padding: '8px',
                          borderRadius: '4px',
                        }}
                      >
                        {JSON.stringify(node.current_clock, null, 2)}
                      </div>
                    </div>

                    <div style={{ marginBottom: '12px' }}>
                      <Text strong>最后更新:</Text>
                      <div>{new Date(node.last_updated).toLocaleString()}</div>
                    </div>

                    <div>
                      <Text strong>待处理操作:</Text>
                      <div>{node.pending_operations}</div>
                    </div>
                  </Card>
                </Col>
              ))
            )}
          </Row>
        </TabPane>

        <TabPane
          tab={
            <span>
              <BranchesOutlined />
              事件历史
            </span>
          }
          key="events"
        >
          <div
            style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}
          >
            {events.map(event => (
              <Card key={event.event_id} size="small">
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    marginBottom: '8px',
                  }}
                >
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                    }}
                  >
                    <Text strong>{event.event_id}</Text>
                    <Tag>{event.node_id}</Tag>
                    <Tag color={getEventTypeColor(event.event_type)}>
                      {event.event_type}
                    </Tag>
                  </div>
                  <Text type="secondary" style={{ fontSize: '12px' }}>
                    {new Date(event.timestamp).toLocaleString()}
                  </Text>
                </div>

                <Row gutter={16}>
                  <Col span={12}>
                    <Text strong>向量时钟:</Text>
                    <div
                      style={{
                        marginTop: '4px',
                        fontFamily: 'monospace',
                        backgroundColor: '#f0f9ff',
                        padding: '6px',
                        borderRadius: '4px',
                        fontSize: '12px',
                      }}
                    >
                      {JSON.stringify(event.vector_clock)}
                    </div>
                  </Col>
                  <Col span={12}>
                    <Text strong>事件数据:</Text>
                    <div
                      style={{
                        marginTop: '4px',
                        fontFamily: 'monospace',
                        backgroundColor: '#f6ffed',
                        padding: '6px',
                        borderRadius: '4px',
                        fontSize: '12px',
                      }}
                    >
                      {JSON.stringify(event.data)}
                    </div>
                  </Col>
                </Row>
              </Card>
            ))}
          </div>
        </TabPane>

        <TabPane
          tab={
            <span>
              <DatabaseOutlined />
              时钟比较
            </span>
          }
          key="comparison"
        >
          <Alert
            message="向量时钟比较"
            description="向量时钟用于确定事件的因果关系。如果时钟A在所有维度上都小于等于时钟B，且至少有一个维度严格小于，则A发生在B之前。"
            type="info"
            showIcon
            style={{ marginBottom: '16px' }}
          />

          <Row gutter={16}>
            {nodes.map((node1, i) =>
              nodes.map((node2, j) => {
                if (i >= j) return null

                // 简单的向量时钟比较逻辑
                const clock1 = node1.current_clock
                const clock2 = node2.current_clock
                const allKeys = new Set([
                  ...Object.keys(clock1),
                  ...Object.keys(clock2),
                ])

                let relation = 'concurrent'
                let node1Before = true
                let node2Before = true

                for (const key of allKeys) {
                  const val1 = clock1[key] || 0
                  const val2 = clock2[key] || 0

                  if (val1 > val2) node2Before = false
                  if (val2 > val1) node1Before = false
                }

                if (node1Before && !node2Before)
                  relation = `${node1.node_id} → ${node2.node_id}`
                else if (node2Before && !node1Before)
                  relation = `${node2.node_id} → ${node1.node_id}`
                else if (!node1Before && !node2Before) relation = 'concurrent'
                else relation = 'equal'

                return (
                  <Col key={`${i}-${j}`} span={12}>
                    <Card size="small" style={{ marginBottom: '12px' }}>
                      <div
                        style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                        }}
                      >
                        <div>
                          <Text strong>{node1.node_id}</Text> vs{' '}
                          <Text strong>{node2.node_id}</Text>
                        </div>
                        <Tag
                          color={
                            relation === 'concurrent'
                              ? 'orange'
                              : relation === 'equal'
                                ? 'blue'
                                : 'green'
                          }
                        >
                          {relation}
                        </Tag>
                      </div>
                    </Card>
                  </Col>
                )
              })
            )}
          </Row>
        </TabPane>
      </Tabs>
    </div>
  )
}

export default VectorClockVisualizationPage
