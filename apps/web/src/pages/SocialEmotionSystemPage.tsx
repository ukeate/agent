import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'
import { Card, Row, Col, Button, Space, Typography, Table, Alert, Spin, Tag } from 'antd'
import { TeamOutlined, ReloadOutlined } from '@ant-design/icons'

type SystemStatus = { service: string; status: string; latency_ms?: number }
type ServiceMetrics = { active_users?: number; messages?: number; sentiment_score?: number }
type UserInteraction = { id: string; user_id: string; message: string; sentiment?: string; timestamp?: string }
type SystemAlert = { id: string; level: string; message: string; created_at?: string }

const SocialEmotionSystemPage: React.FC = () => {
  const [status, setStatus] = useState<SystemStatus[]>([])
  const [metrics, setMetrics] = useState<ServiceMetrics | null>(null)
  const [interactions, setInteractions] = useState<UserInteraction[]>([])
  const [alerts, setAlerts] = useState<SystemAlert[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const [statusRes, metricsRes, interactionsRes, alertsRes] = await Promise.all([
        apiFetch(buildApiUrl('/api/v1/emotion-intelligence/health'),
        apiFetch(buildApiUrl('/api/v1/emotion-intelligence/metrics'),
        apiFetch(buildApiUrl('/api/v1/emotion-intelligence/interactions'),
        apiFetch(buildApiUrl('/api/v1/emotion-intelligence/alerts'))
      ])
      setStatus((await statusRes.json())?.services || [])
      setMetrics(await metricsRes.json())
      setInteractions((await interactionsRes.json())?.interactions || [])
      setAlerts((await alertsRes.json())?.alerts || [])
    } catch (e: any) {
      setError(e?.message || '加载失败')
      setStatus([])
      setMetrics(null)
      setInteractions([])
      setAlerts([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
  }, [])

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Space align="center" style={{ justifyContent: 'space-between', width: '100%' }}>
          <Space>
            <TeamOutlined />
            <Typography.Title level={3} style={{ margin: 0 }}>
              社交情感系统
            </Typography.Title>
          </Space>
          <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>
            刷新
          </Button>
        </Space>

        {error && <Alert type="error" message="加载失败" description={error} />}

        <Card title="服务状态">
          {loading ? (
            <Spin />
          ) : (
            <Table
              rowKey="service"
              dataSource={status}
              columns={[
                { title: '服务', dataIndex: 'service' },
                { title: '状态', dataIndex: 'status', render: (v) => <Tag color={v === 'healthy' ? 'green' : 'red'}>{v}</Tag> },
                { title: '延迟(ms)', dataIndex: 'latency_ms' },
              ]}
              locale={{ emptyText: '暂无状态数据。' }}
            />
          )}
        </Card>

        <Row gutter={16}>
          <Col span={12}>
            <Card title="核心指标">
              {loading ? (
                <Spin />
              ) : metrics ? (
                <Space size="large">
                  <Typography.Text>活跃用户: {metrics.active_users ?? '-'}</Typography.Text>
                  <Typography.Text>消息数: {metrics.messages ?? '-'}</Typography.Text>
                  <Typography.Text>情感得分: {metrics.sentiment_score ?? '-'}</Typography.Text>
                </Space>
              ) : (
                <Alert type="info" message="暂无指标数据。" />
              )}
            </Card>
          </Col>
          <Col span={12}>
            <Card title="告警">
              {loading ? (
                <Spin />
              ) : (
                <Table
                  rowKey="id"
                  dataSource={alerts}
                  columns={[
                    { title: '级别', dataIndex: 'level', render: (v) => <Tag color={v === 'critical' ? 'red' : 'orange'}>{v}</Tag> },
                    { title: '消息', dataIndex: 'message' },
                    { title: '时间', dataIndex: 'created_at' },
                  ]}
                  pagination={false}
                  locale={{ emptyText: '暂无告警。' }}
                />
              )}
            </Card>
          </Col>
        </Row>

        <Card title="最近交互">
          {loading ? (
            <Spin />
          ) : (
            <Table
              rowKey="id"
              dataSource={interactions}
              columns={[
                { title: '用户', dataIndex: 'user_id' },
                { title: '消息', dataIndex: 'message' },
                { title: '情感', dataIndex: 'sentiment' },
                { title: '时间', dataIndex: 'timestamp' },
              ]}
              locale={{ emptyText: '暂无交互数据。' }}
            />
          )}
        </Card>
      </Space>
    </div>
  )
}

export default SocialEmotionSystemPage
