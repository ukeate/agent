import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'
import { Card, Row, Col, Alert, Button, Statistic, Typography, List, Tag, Spin, message } from 'antd'
import { ReloadOutlined, DashboardOutlined } from '@ant-design/icons'

const { Title, Text } = Typography

interface MetricPoint {
  timestamp: string
  value: number
  metric: string
}

interface HealthSummary {
  overall: string
  latency: string
  throughput: string
  errors: string
  resources: string
}

interface AlertItem {
  id: string
  type: string
  message: string
  timestamp: string
  resolved: boolean
}

const PersonalizationMonitorPage: React.FC = () => {
  const [metrics, setMetrics] = useState<MetricPoint[]>([])
  const [health, setHealth] = useState<HealthSummary | null>(null)
  const [alerts, setAlerts] = useState<AlertItem[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadData = async () => {
    setLoading(true)
    setError(null)
    try {
      const [metricRes, healthRes, alertRes] = await Promise.all([
        apiFetch(buildApiUrl('/api/v1/personalization/metrics'),
        apiFetch(buildApiUrl('/api/v1/personalization/health'),
        apiFetch(buildApiUrl('/api/v1/personalization/alerts'))
      ])


      const metricData = await metricRes.json()
      const healthData = await healthRes.json()
      const alertData = await alertRes.json()

      setMetrics(metricData.metrics || [])
      setHealth(healthData.health || healthData)
      setAlerts(alertData.alerts || [])
    } catch (e: any) {
      setError(e?.message || '加载失败')
      message.error(e?.message || '加载个性化监控数据失败')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
    const timer = setInterval(loadData, 5000)
    return () => clearInterval(timer)
  }, [])

  return (
    <div style={{ padding: 24 }}>
      <Row justify="space-between" align="middle" style={{ marginBottom: 16 }}>
        <Col>
          <Title level={3}><DashboardOutlined /> 个性化监控</Title>
          <Text type="secondary">数据来自 /api/v1/personalization/metrics|health|alerts，无任何本地静态数据</Text>
        </Col>
        <Col>
          <Button icon={<ReloadOutlined />} onClick={loadData} loading={loading}>刷新</Button>
        </Col>
      </Row>

      {error && <Alert type="error" message={error} showIcon closable style={{ marginBottom: 16 }} />}
      {loading && <Spin style={{ marginBottom: 16 }} />}

      <Row gutter={16}>
        <Col span={6}>
          <Card>
            <Statistic title="指标数量" value={metrics.length} />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic title="告警数量" value={alerts.length} />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic title="系统状态" value={health?.overall || '未知'} />
          </Card>
        </Col>
      </Row>

      <Card title="实时指标" style={{ marginTop: 16 }}>
        <List
          dataSource={metrics.slice(0, 50)}
          renderItem={m => (
            <List.Item>
              <Space direction="vertical">
                <Text strong>{m.metric}</Text>
                <Text type="secondary">{new Date(m.timestamp).toLocaleTimeString()}</Text>
                <Tag color="blue">{m.value}</Tag>
              </Space>
            </List.Item>
          )}
        />
        {metrics.length === 0 && <Text type="secondary">暂无指标数据</Text>}
      </Card>

      <Card title="健康状态" style={{ marginTop: 16 }}>
        {health ? (
          <Space wrap>
            <Tag color="blue">整体: {health.overall}</Tag>
            <Tag color="green">延迟: {health.latency}</Tag>
            <Tag color="cyan">吞吐: {health.throughput}</Tag>
            <Tag color="gold">错误: {health.errors}</Tag>
            <Tag color="orange">资源: {health.resources}</Tag>
          </Space>
        ) : (
          <Text type="secondary">暂无健康数据</Text>
        )}
      </Card>

      <Card title="告警" style={{ marginTop: 16 }}>
        <List
          dataSource={alerts}
          renderItem={a => (
            <List.Item>
              <Space direction="vertical">
                <Tag color={a.type === 'error' ? 'red' : a.type === 'warning' ? 'orange' : 'blue'}>{a.type}</Tag>
                <Text>{a.message}</Text>
                <Text type="secondary">{a.timestamp}</Text>
                <Tag color={a.resolved ? 'green' : 'gold'}>{a.resolved ? '已解决' : '未解决'}</Tag>
              </Space>
            </List.Item>
          )}
        />
        {alerts.length === 0 && <Text type="secondary">暂无告警</Text>}
      </Card>
    </div>
  )
}

export default PersonalizationMonitorPage
