import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'
import { Card, Row, Col, Statistic, Progress, Table, Button, Space, Typography, message } from 'antd'
import { ReloadOutlined, ThunderboltOutlined } from '@ant-design/icons'

interface PerfMetric {
  metric: string
  value: number
  target?: number
}

interface ResourceMetric {
  name: string
  value: number
}

const QLearningPerformancePage: React.FC = () => {
  const [perf, setPerf] = useState<PerfMetric[]>([])
  const [resources, setResources] = useState<ResourceMetric[]>([])
  const [loading, setLoading] = useState(false)

  const loadData = async () => {
    setLoading(true)
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/qlearning/performance'))
      const data = await res.json()
      setPerf(data.performance || [])
      setResources(data.resources || [])
    } catch (e: any) {
      message.error(e?.message || '加载性能数据失败')
      setPerf([])
      setResources([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
    const timer = setInterval(loadData, 5000)
    return () => clearInterval(timer)
  }, [])

  const perfColumns = [
    { title: '指标', dataIndex: 'metric', key: 'metric' },
    { title: '当前值', dataIndex: 'value', key: 'value', render: (v: number) => v.toFixed(4) },
    { title: '目标值', dataIndex: 'target', key: 'target', render: (v?: number) => v !== undefined ? v.toFixed(4) : '—' }
  ]

  const resourceColumns = [
    { title: '资源', dataIndex: 'name', key: 'name' },
    { title: '值', dataIndex: 'value', key: 'value', render: (v: number) => `${v.toFixed(2)}%` }
  ]

  return (
    <div style={{ padding: 24 }}>
      <Space style={{ marginBottom: 16 }}>
        <ThunderboltOutlined />
        <Typography.Title level={3} style={{ margin: 0 }}>Q-Learning 性能监控</Typography.Title>
        <Button icon={<ReloadOutlined />} onClick={loadData} loading={loading}>刷新</Button>
      </Space>
      <Typography.Text type="secondary">数据来自 /api/v1/qlearning/performance。</Typography.Text>

      <Row gutter={16} style={{ marginTop: 16 }}>
        {perf.slice(0, 4).map((p) => (
          <Col span={6} key={p.metric}>
            <Card>
              <Statistic title={p.metric} value={p.value} precision={4} />
              {p.target !== undefined && (
                <Progress
                  percent={p.target ? Math.min(100, (p.value / p.target) * 100) : 0}
                  size="small"
                  status="active"
                />
              )}
            </Card>
          </Col>
        ))}
      </Row>

      <Card title="性能指标" style={{ marginTop: 16 }}>
        <Table
          rowKey="metric"
          dataSource={perf}
          columns={perfColumns}
          loading={loading}
          pagination={false}
        />
      </Card>

      <Card title="资源使用" style={{ marginTop: 16 }}>
        <Table
          rowKey="name"
          dataSource={resources}
          columns={resourceColumns}
          loading={loading}
          pagination={false}
        />
      </Card>
    </div>
  )
}

export default QLearningPerformancePage
