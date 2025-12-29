import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'
import { Card, Space, Typography, Button, Table, message, Descriptions, Tag } from 'antd'
import { ReloadOutlined } from '@ant-design/icons'

type Health = {
  status?: string
  reasoner_initialized?: boolean
  timestamp?: string
  test_reasoning?: string
  test_confidence?: number
  test_error?: string
}

type StrategyStats = {
  total_queries?: number
  success_rate?: number
  avg_confidence?: number
  avg_execution_time?: number
  accuracy_score?: number
  last_updated?: string
}

type Performance = {
  summary?: Record<string, any>
  strategies?: Record<string, StrategyStats>
}

const KGReasoningOptimizationPage: React.FC = () => {
  const [loading, setLoading] = useState(false)
  const [health, setHealth] = useState<Health | null>(null)
  const [performance, setPerformance] = useState<Performance | null>(null)

  const load = async () => {
    setLoading(true)
    try {
      const [healthRes, perfRes] = await Promise.all([
        apiFetch(buildApiUrl('/api/v1/kg-reasoning/health'),
        apiFetch(buildApiUrl('/api/v1/kg-reasoning/strategies/performance'),
      ])
      const healthData = await healthRes.json().catch(() => null)
      const perfData = await perfRes.json().catch(() => null)


      setHealth(healthData)
      setPerformance(perfData)
    } catch (e: any) {
      message.error(e?.message || '加载失败')
      setHealth(null)
      setPerformance(null)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
  }, [])

  const rows = Object.entries(performance?.strategies || {}).map(([name, s]) => ({ name, ...s }))

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Space align="center" style={{ justifyContent: 'space-between', width: '100%' }}>
          <Typography.Title level={3} style={{ margin: 0 }}>
            知识图推理性能
          </Typography.Title>
          <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>
            刷新
          </Button>
        </Space>

        <Card title="健康检查" loading={loading}>
          {health ? (
            <Descriptions column={2} size="small">
              <Descriptions.Item label="状态">
                <Tag color={health.status === 'healthy' ? 'green' : 'red'}>{health.status || '-'}</Tag>
              </Descriptions.Item>
              <Descriptions.Item label="已初始化">{String(!!health.reasoner_initialized)}</Descriptions.Item>
              <Descriptions.Item label="时间">{health.timestamp || '-'}</Descriptions.Item>
              <Descriptions.Item label="自检">{health.test_reasoning || '-'}</Descriptions.Item>
              {health.test_confidence !== undefined ? (
                <Descriptions.Item label="自检置信度">{health.test_confidence ?? '-'}</Descriptions.Item>
              ) : null}
              {health.test_error ? <Descriptions.Item label="自检错误">{health.test_error}</Descriptions.Item> : null}
            </Descriptions>
          ) : (
            <Typography.Text type="secondary">暂无数据</Typography.Text>
          )}
        </Card>

        <Card title="策略性能" loading={loading}>
          <Space direction="vertical" style={{ width: '100%' }} size="middle">
            {performance?.summary ? (
              <Descriptions column={2} size="small">
                <Descriptions.Item label="策略数">{performance.summary.total_strategies ?? '-'}</Descriptions.Item>
                <Descriptions.Item label="总查询">{performance.summary.total_queries ?? '-'}</Descriptions.Item>
                <Descriptions.Item label="平均成功率">{performance.summary.avg_success_rate ?? '-'}</Descriptions.Item>
                <Descriptions.Item label="平均置信度">{performance.summary.avg_confidence ?? '-'}</Descriptions.Item>
                <Descriptions.Item label="平均耗时(s)">{performance.summary.avg_execution_time ?? '-'}</Descriptions.Item>
              </Descriptions>
            ) : (
              <Typography.Text type="secondary">暂无数据</Typography.Text>
            )}

            <Table
              rowKey="name"
              size="small"
              pagination={false}
              dataSource={rows}
              columns={[
                { title: '策略', dataIndex: 'name' },
                { title: '总查询', dataIndex: 'total_queries' },
                { title: '成功率', dataIndex: 'success_rate' },
                { title: '平均置信度', dataIndex: 'avg_confidence' },
                { title: '平均耗时(s)', dataIndex: 'avg_execution_time' },
                { title: '准确率', dataIndex: 'accuracy_score' },
                { title: '更新时间', dataIndex: 'last_updated' },
              ]}
            />
          </Space>
        </Card>
      </Space>
    </div>
  )
}

export default KGReasoningOptimizationPage
