import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'
import { Card, Table, Button, Space, Typography, Tag, Alert, Progress, message } from 'antd'
import { ReloadOutlined, DashboardOutlined } from '@ant-design/icons'

type QualityScore = {
  feedback_id: string
  quality_score: number
  quality_factors: Record<string, number>
  risk_flags?: string[]
}

type QualityMetrics = {
  total_feedbacks?: number
  average_quality_score?: number
  quality_score_distribution?: Record<string, number>
}

const FeedbackQualityMonitorPage: React.FC = () => {
  const [scores, setScores] = useState<QualityScore[]>([])
  const [metrics, setMetrics] = useState<QualityMetrics | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const [mRes, sRes] = await Promise.all([
        apiFetch(buildApiUrl('/api/v1/feedback/overview')),
        apiFetch(buildApiUrl('/api/v1/feedback/quality/score'), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify([]) // 后端若需要ID列表可在页面增加选择，这里传空获取默认评估
        })
      ])
      const mData = await mRes.json()
      const sData = await sRes.json()
      setMetrics(mData)
      setScores(Array.isArray(sData?.quality_scores) ? sData.quality_scores : [])
    } catch (e: any) {
      setError(e?.message || '加载失败')
      setMetrics(null)
      setScores([])
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
            <DashboardOutlined />
            <Typography.Title level={3} style={{ margin: 0 }}>
              反馈质量监控
            </Typography.Title>
          </Space>
          <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>
            刷新
          </Button>
        </Space>

        {error && <Alert type="error" message={error} />}

        <Card>
          <Space size="large">
            <Typography.Text>总反馈: {metrics?.total_feedbacks ?? 0}</Typography.Text>
            <Typography.Text>平均质量分: {metrics?.average_quality_score ?? '-'}</Typography.Text>
          </Space>
        </Card>

        <Card title="质量评分">
          <Table
            rowKey="feedback_id"
            loading={loading}
            dataSource={scores}
            locale={{ emptyText: '暂无数据' }}
            columns={[
              { title: '反馈ID', dataIndex: 'feedback_id' },
              {
                title: '质量分',
                dataIndex: 'quality_score',
                render: (v) => <Progress percent={Math.round((v || 0) * 100)} />
              },
              {
                title: '风险',
                dataIndex: 'risk_flags',
                render: (flags: string[] | undefined) =>
                  (flags || []).map((f) => (
                    <Tag key={f} color="red">
                      {f}
                    </Tag>
                  ))
              }
            ]}
          />
        </Card>
      </Space>
    </div>
  )
}

export default FeedbackQualityMonitorPage
