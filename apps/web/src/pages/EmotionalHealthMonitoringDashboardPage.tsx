import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'
import {
  Card,
  Row,
  Col,
  Statistic,
  Table,
  Tag,
  Button,
  Space,
  Alert,
  Typography,
  Progress,
} from 'antd'
import { ReloadOutlined, HeartOutlined } from '@ant-design/icons'

type Dashboard = {
  metrics?: Record<string, any>
  goals?: any[]
  alerts?: any[]
  recommendations?: any[]
  score?: number
}

const EmotionalHealthMonitoringDashboardPage: React.FC = () => {
  const [dashboard, setDashboard] = useState<Dashboard | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const userId = 'user_001'

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await apiFetch(
        buildApiUrl(`/api/v1/emotional-intelligence/health-dashboard/${userId}`)
      )
      const data = await res.json()
      setDashboard(data?.dashboard || null)
    } catch (e: any) {
      setError(e?.message || '加载失败')
      setDashboard(null)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
  }, [])

  const metrics = dashboard?.metrics || {}
  const goals = dashboard?.goals || []
  const alerts = dashboard?.alerts || []
  const recommendations = dashboard?.recommendations || []

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Space
          align="center"
          style={{ justifyContent: 'space-between', width: '100%' }}
        >
          <Space>
            <HeartOutlined />
            <Typography.Title level={3} style={{ margin: 0 }}>
              情感健康监测
            </Typography.Title>
          </Space>
          <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>
            刷新
          </Button>
        </Space>

        {error && <Alert type="error" message={error} />}

        <Row gutter={16}>
          <Col span={6}>
            <Card>
              <Statistic title="健康评分" value={dashboard?.score ?? 0} />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic title="指标数" value={Object.keys(metrics).length} />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic title="目标数" value={goals.length} />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic title="告警数" value={alerts.length} />
            </Card>
          </Col>
        </Row>

        <Card title="指标">
          <Table
            rowKey="key"
            dataSource={Object.keys(metrics).map(k => ({
              key: k,
              name: k,
              value: metrics[k],
            }))}
            loading={loading}
            columns={[
              { title: '名称', dataIndex: 'name' },
              {
                title: '数值',
                dataIndex: 'value',
                render: v => (
                  <Progress
                    percent={Math.min(100, Math.round((v || 0) * 100))}
                  />
                ),
              },
            ]}
            locale={{ emptyText: '暂无指标' }}
          />
        </Card>

        <Card title="目标">
          <Table
            rowKey="goal_id"
            dataSource={goals}
            loading={loading}
            columns={[
              { title: '目标', dataIndex: 'goal_type' },
              {
                title: '状态',
                dataIndex: 'status',
                render: (s: string) => <Tag color="blue">{s}</Tag>,
              },
              { title: '截止', dataIndex: 'deadline' },
            ]}
            locale={{ emptyText: '暂无目标' }}
          />
        </Card>

        <Card title="告警">
          <Table
            rowKey="alert_id"
            dataSource={alerts}
            loading={loading}
            columns={[
              { title: '类型', dataIndex: 'alert_type' },
              { title: '严重性', dataIndex: 'severity' },
              { title: '时间', dataIndex: 'timestamp' },
            ]}
            locale={{ emptyText: '暂无告警' }}
          />
        </Card>

        <Card title="建议">
          <Table
            rowKey="recommendation_id"
            dataSource={recommendations}
            loading={loading}
            columns={[
              { title: '标题', dataIndex: 'title' },
              { title: '类别', dataIndex: 'category' },
              { title: '优先级', dataIndex: 'priority' },
            ]}
            locale={{ emptyText: '暂无建议' }}
          />
        </Card>
      </Space>
    </div>
  )
}

export default EmotionalHealthMonitoringDashboardPage
