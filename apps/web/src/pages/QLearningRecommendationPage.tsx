import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'
import { Card, Row, Col, Button, Table, Tag, Typography, Space, message } from 'antd'
import { ReloadOutlined, ThunderboltOutlined, BulbOutlined } from '@ant-design/icons'

interface RecommendationRecord {
  decision_id: string
  action: string
  source: string
  confidence: number
  reward?: number
  inference_time_ms?: number
  timestamp: string
}

interface StrategyPerformance {
  strategy: string
  total_decisions: number
  successful_decisions: number
  average_reward: number
  average_confidence: number
  average_inference_time: number
}

const QLearningRecommendationPage: React.FC = () => {
  const [recs, setRecs] = useState<RecommendationRecord[]>([])
  const [perf, setPerf] = useState<StrategyPerformance[]>([])
  const [loading, setLoading] = useState(false)

  const loadData = async () => {
    setLoading(true)
    try {
      const [recRes, perfRes] = await Promise.all([
        apiFetch(buildApiUrl('/api/v1/qlearning/recommendations'),
        apiFetch(buildApiUrl('/api/v1/qlearning/strategy/performance'))
      ])
      const recData = await recRes.json()
      const perfData = await perfRes.json()
      setRecs(recData.recommendations || [])
      setPerf(perfData.performance || [])
    } catch (e: any) {
      message.error(e?.message || '加载推荐数据失败')
      setRecs([])
      setPerf([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
    const timer = setInterval(loadData, 5000)
    return () => clearInterval(timer)
  }, [])

  const recColumns = [
    { title: '决策ID', dataIndex: 'decision_id', key: 'decision_id' },
    { title: '动作', dataIndex: 'action', key: 'action' },
    { title: '来源', dataIndex: 'source', key: 'source', render: (s: string) => <Tag color="blue">{s}</Tag> },
    { title: '置信度', dataIndex: 'confidence', key: 'confidence', render: (v: number) => `${v.toFixed(2)}%` },
    { title: '奖励', dataIndex: 'reward', key: 'reward', render: (v?: number) => v !== undefined ? v.toFixed(2) : '—' },
    { title: '推理耗时(ms)', dataIndex: 'inference_time_ms', key: 'inference_time_ms', render: (v?: number) => v ?? '—' },
    { title: '时间', dataIndex: 'timestamp', key: 'timestamp', render: (t: string) => new Date(t).toLocaleString() }
  ]

  const perfColumns = [
    { title: '策略', dataIndex: 'strategy', key: 'strategy' },
    { title: '决策总数', dataIndex: 'total_decisions', key: 'total_decisions' },
    { title: '成功决策', dataIndex: 'successful_decisions', key: 'successful_decisions' },
    { title: '平均奖励', dataIndex: 'average_reward', key: 'average_reward', render: (v: number) => v.toFixed(2) },
    { title: '平均置信度', dataIndex: 'average_confidence', key: 'average_confidence', render: (v: number) => `${v.toFixed(2)}%` },
    { title: '平均耗时(ms)', dataIndex: 'average_inference_time', key: 'average_inference_time', render: (v: number) => v.toFixed(2) }
  ]

  return (
    <div style={{ padding: 24 }}>
      <Space style={{ marginBottom: 16 }}>
        <ThunderboltOutlined />
        <Typography.Title level={3} style={{ margin: 0 }}>Q-Learning 推荐</Typography.Title>
        <Button icon={<ReloadOutlined />} onClick={loadData} loading={loading}>刷新</Button>
      </Space>
      <Typography.Text type="secondary">数据来源 /api/v1/qlearning/recommendations 与 /strategy/performance，无本地模拟。</Typography.Text>

      <Card title="最新推荐" style={{ marginTop: 12 }}>
        <Table
          rowKey="decision_id"
          dataSource={recs}
          columns={recColumns}
          loading={loading}
          pagination={{ pageSize: 20 }}
        />
      </Card>

      <Card title="策略性能" style={{ marginTop: 12 }}>
        <Table
          rowKey="strategy"
          dataSource={perf}
          columns={perfColumns}
          loading={loading}
          pagination={false}
        />
      </Card>
    </div>
  )
}

export default QLearningRecommendationPage
