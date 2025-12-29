import React, { useEffect, useState } from 'react'
import { Card, Row, Col, Statistic, List, Alert, Button, Typography, Space, Tag, message } from 'antd'
import { ReloadOutlined, HeartOutlined } from '@ant-design/icons'
import { buildApiUrl, apiFetch } from '../utils/apiBase'

interface HealthStatus {
  status: string
  components?: Record<string, any>
  failed_components?: string[]
  last_check?: string
}

const ServiceHealthMonitorPage: React.FC = () => {
  const [health, setHealth] = useState<HealthStatus | null>(null)
  const [metrics, setMetrics] = useState<any>(null)
  const [alerts, setAlerts] = useState<any[]>([])
  const [loading, setLoading] = useState(false)

  const loadData = async () => {
    setLoading(true)
    try {
      const [h, m, a] = await Promise.all([
        fetchJson('/api/v1/health?detailed=true'),
        fetchJson('/api/v1/health/metrics'),
        fetchJson('/api/v1/health/alerts')
      ])
      setHealth(h)
      setMetrics(m)
      setAlerts(a.alerts || [])
    } catch (e: any) {
      message.error(e?.message || '加载健康数据失败')
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
      <Space style={{ marginBottom: 16 }}>
        <HeartOutlined />
        <Typography.Title level={3} style={{ margin: 0 }}>服务健康监控</Typography.Title>
        <Button icon={<ReloadOutlined />} onClick={loadData} loading={loading}>刷新</Button>
      </Space>
      <Typography.Text type="secondary">数据来自 /api/v1/health 系列真实接口</Typography.Text>

      {health && (
        <Card style={{ marginTop: 12 }}>
          <Row gutter={16}>
            <Col span={6}>
              <Statistic title="总体状态" value={health.status} valueStyle={{ color: health.status === 'healthy' ? '#52c41a' : '#faad14' }} />
            </Col>
            <Col span={6}>
              <Statistic title="失败组件" value={health.failed_components?.length || 0} />
            </Col>
            <Col span={6}>
              <Statistic title="组件数" value={Object.keys(health.components || {}).length} />
            </Col>
          </Row>
          {health.failed_components && health.failed_components.length > 0 && (
            <Alert type="error" style={{ marginTop: 12 }} message={`失败组件: ${health.failed_components.join(', ')}`} />
          )}
        </Card>
      )}

      <Card title="指标" style={{ marginTop: 16 }}>
        <pre style={{ background: '#fafafa', padding: 12, maxHeight: 300, overflow: 'auto' }}>
{JSON.stringify(metrics, null, 2)}
        </pre>
      </Card>

      <Card title="告警" style={{ marginTop: 16 }}>
        <List
          dataSource={alerts}
          renderItem={(item: any) => (
            <List.Item>
              <Space>
                <Tag color="red">{item.level || 'alert'}</Tag>
                <span>{item.message || JSON.stringify(item)}</span>
              </Space>
            </List.Item>
          )}
        />
        {alerts.length === 0 && <Typography.Text type="secondary">当前无告警</Typography.Text>}
      </Card>
    </div>
  )
}

async function fetchJson(url: string) {
  const res = await apiFetch(buildApiUrl(url))
  return res.json()
}

export default ServiceHealthMonitorPage
