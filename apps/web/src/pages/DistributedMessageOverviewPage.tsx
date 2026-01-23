import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'
import { Card, Row, Col, Statistic, Progress, Table, Tag, Space, Typography, Button, message } from 'antd'
import { ReloadOutlined, ShareAltOutlined as NetworkOutlined } from '@ant-design/icons'

interface SystemMetrics {
  total_messages: number
  active_connections: number
  cluster_nodes: number
  throughput: number
  avg_latency_ms: number
  error_rate: number
  health: 'healthy' | 'warning' | 'error'
}

interface NodeMetric {
  id: string
  name: string
  status: string
  cpu: number
  memory: number
  connections: number
  uptime: string
}

const DistributedMessageOverviewPage: React.FC = () => {
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null)
  const [nodes, setNodes] = useState<NodeMetric[]>([])
  const [loading, setLoading] = useState(false)

  const loadData = async () => {
    setLoading(true)
    try {
      const [sysRes, nodeRes] = await Promise.all([
        apiFetch(buildApiUrl('/api/v1/streaming/backpressure/status')),
        apiFetch(buildApiUrl('/api/v1/service-discovery/agents'))
      ])

      const sys = await sysRes.json()
      const agents = await nodeRes.json()

      setMetrics({
        total_messages: sys.backpressure_status?.total_messages ?? 0,
        active_connections: sys.backpressure_status?.active_connections ?? 0,
        cluster_nodes: agents.agents?.length ?? 0,
        throughput: sys.backpressure_status?.throughput ?? 0,
        avg_latency_ms: sys.backpressure_status?.avg_latency_ms ?? 0,
        error_rate: sys.backpressure_status?.error_rate ?? 0,
        health: sys.backpressure_enabled ? 'healthy' : 'warning'
      })

      setNodes((agents.agents || []).map((a: any) => {
        const agentId = a.agent_id || a.id || ''
        const resource = a.resource_usage || {}
        const createdAt = a.created_at ? new Date(a.created_at).getTime() : 0
        const lastHeartbeat = a.last_heartbeat ? new Date(a.last_heartbeat).getTime() : 0
        const endTime = lastHeartbeat || Date.now()
        const uptimeSeconds = createdAt ? Math.max(0, Math.floor((endTime - createdAt) / 1000)) : 0
        const hours = uptimeSeconds ? Math.floor(uptimeSeconds / 3600) : 0
        const minutes = uptimeSeconds ? Math.floor((uptimeSeconds % 3600) / 60) : 0

        return {
          id: agentId,
          name: a.name || agentId,
          status: a.status || 'unknown',
          cpu: a.cpu_usage ?? resource.cpu_usage_percent ?? 0,
          memory: a.memory_usage ?? resource.memory_usage_percent ?? 0,
          connections: a.connections ?? resource.active_tasks ?? 0,
          uptime: uptimeSeconds ? `${hours}h ${minutes}m` : ''
        }
      }))
    } catch (e: any) {
      message.error(e?.message || '加载分布式消息数据失败')
      setMetrics(null)
      setNodes([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
    const timer = setInterval(loadData, 5000)
    return () => clearInterval(timer)
  }, [])

  const columns = [
    { title: '节点', dataIndex: 'name', key: 'name' },
    { title: '状态', dataIndex: 'status', key: 'status', render: (s: string) => <Tag color={s === 'online' ? 'green' : 'red'}>{s}</Tag> },
    { title: 'CPU%', dataIndex: 'cpu', key: 'cpu', render: (v: number) => `${v}%` },
    { title: '内存%', dataIndex: 'memory', key: 'memory', render: (v: number) => `${v}%` },
    { title: '连接数', dataIndex: 'connections', key: 'connections' },
    { title: '运行时间', dataIndex: 'uptime', key: 'uptime' },
  ]

  return (
    <div style={{ padding: 24 }}>
      <Row justify="space-between" align="middle" style={{ marginBottom: 16 }}>
        <Col>
          <Space>
            <NetworkOutlined />
            <Typography.Title level={3} style={{ margin: 0 }}>分布式消息概览</Typography.Title>
          </Space>
          <Typography.Text type="secondary">数据来源 /api/v1/streaming/backpressure/status 与 /api/v1/service-discovery/agents</Typography.Text>
        </Col>
        <Col>
          <Button icon={<ReloadOutlined />} onClick={loadData} loading={loading}>刷新</Button>
        </Col>
      </Row>

      {metrics && (
        <Card style={{ marginBottom: 16 }}>
          <Row gutter={16}>
            <Col span={4}><Statistic title="消息总量" value={metrics.total_messages} /></Col>
            <Col span={4}><Statistic title="活跃连接" value={metrics.active_connections} /></Col>
            <Col span={4}><Statistic title="节点数" value={metrics.cluster_nodes} /></Col>
            <Col span={4}><Statistic title="吞吐 (msg/s)" value={metrics.throughput} /></Col>
            <Col span={4}><Statistic title="均延迟(ms)" value={metrics.avg_latency_ms} precision={2} /></Col>
            <Col span={4}><Statistic title="错误率" value={(metrics.error_rate || 0) * 100} precision={2} suffix="%" /></Col>
          </Row>
        </Card>
      )}

      <Card title="节点状态">
        <Table
          rowKey="id"
          dataSource={nodes}
          columns={columns}
          loading={loading}
          pagination={false}
        />
      </Card>
    </div>
  )
}

export default DistributedMessageOverviewPage
