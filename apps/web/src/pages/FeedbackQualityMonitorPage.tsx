import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Alert,
  Button,
  Card,
  Input,
  InputNumber,
  Space,
  Tag,
  Typography,
} from 'antd'
import {
  apiFetch,
  buildApiUrl,
  extractApiErrorMessage,
  normalizeHttpErrorMessage,
} from '../utils/apiBase'

const { Title, Text } = Typography

type QualityScore = {
  feedback_id: string
  quality_score: number
  quality_factors?: Record<string, number>
  is_valid?: boolean
  reasons?: string[]
}

type OverviewData = {
  total_feedbacks?: number
  unique_users?: number
  average_rating?: number | null
  positive_ratio?: number | null
}

const FeedbackQualityMonitorPage: React.FC = () => {
  const [userId, setUserId] = useState('demo-user')
  const [limit, setLimit] = useState(10)
  const [scores, setScores] = useState<QualityScore[]>([])
  const [overview, setOverview] = useState<OverviewData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const initialUserId = useRef(userId)
  const initialLimit = useRef(limit)

  const loadOverview = useCallback(async () => {
    const res = await apiFetch(buildApiUrl('/api/v1/feedback/overview'))
    const payload = await res.json()
    if (!res.ok || payload?.success === false) {
      const message = extractApiErrorMessage(payload, '加载失败')
      throw new Error(normalizeHttpErrorMessage(res.status, message))
    }
    setOverview(payload?.data || null)
  }, [])

  const loadScores = useCallback(
    async (targetUserId: string, targetLimit: number) => {
      if (!targetUserId.trim()) {
        setScores([])
        return
      }
      const url = buildApiUrl(
        `/api/v1/feedback/user/${encodeURIComponent(
          targetUserId.trim()
        )}?limit=${targetLimit}`
      )
      const historyRes = await apiFetch(url)
      const historyPayload = await historyRes.json()
      if (!historyRes.ok || historyPayload?.success === false) {
        const message = extractApiErrorMessage(historyPayload, '获取反馈失败')
        throw new Error(normalizeHttpErrorMessage(historyRes.status, message))
      }
      const items = Array.isArray(historyPayload?.data?.items)
        ? historyPayload.data.items
        : []
      const feedbackIds = items
        .map((item: any) => item?.event_id)
        .filter(Boolean)
      if (feedbackIds.length === 0) {
        setScores([])
        return
      }
      const scoreRes = await apiFetch(
        buildApiUrl('/api/v1/feedback/quality/score'),
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(feedbackIds),
        }
      )
      const scorePayload = await scoreRes.json()
      if (!scoreRes.ok || scorePayload?.success === false) {
        const message = extractApiErrorMessage(scorePayload, '获取质量评分失败')
        throw new Error(normalizeHttpErrorMessage(scoreRes.status, message))
      }
      setScores(Array.isArray(scorePayload?.data) ? scorePayload.data : [])
    },
    []
  )

  const loadWith = useCallback(
    async (targetUserId: string, targetLimit: number) => {
      setLoading(true)
      setError(null)
      try {
        await Promise.all([
          loadOverview(),
          loadScores(targetUserId, targetLimit),
        ])
      } catch (err: any) {
        setError(err?.message || '加载失败')
        setScores([])
      } finally {
        setLoading(false)
      }
    },
    [loadOverview, loadScores]
  )

  const handleRefresh = () => {
    loadWith(userId, limit)
  }

  useEffect(() => {
    loadWith(initialUserId.current, initialLimit.current)
  }, [loadWith])

  const averageQuality = useMemo(() => {
    if (!scores.length) return '-'
    const total = scores.reduce((sum, item) => sum + (item.quality_score || 0), 0)
    return (total / scores.length).toFixed(2)
  }, [scores])

  const invalidScores = scores.filter(item => item.is_valid === false)

  return (
    <div style={{ padding: 24 }} data-testid="quality-monitor-page">
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <div>
          <Title level={3} style={{ marginBottom: 4 }}>
            反馈质量监控
          </Title>
          <Text type="secondary">聚合质量评分，识别异常反馈。</Text>
        </div>

        <Card>
          <Space wrap>
            <div>
              <Text type="secondary">用户ID</Text>
              <Input
                value={userId}
                onChange={event => setUserId(event.target.value)}
                placeholder="输入用户ID"
                style={{ width: 220 }}
              />
            </div>
            <div>
              <Text type="secondary">样本数量</Text>
              <InputNumber
                min={1}
                max={50}
                value={limit}
                onChange={value => setLimit(value || 10)}
                style={{ width: 120 }}
              />
            </div>
            <Button onClick={handleRefresh} loading={loading}>
              刷新
            </Button>
          </Space>
        </Card>

        {error && <Alert type="error" message={error} showIcon />}

        <Card>
          <Space wrap>
            <Text>
              总反馈：{overview?.total_feedbacks ?? 0}
            </Text>
            <Text>活跃用户：{overview?.unique_users ?? 0}</Text>
            <Text data-testid="quality-score">平均质量：{averageQuality}</Text>
            <Text data-testid="confidence-score">
              评分稳定度：{scores.length > 0 ? '稳定' : '-'}
            </Text>
            <Text data-testid="spam-detection-rate">
              异常比例：{scores.length > 0
                ? `${Math.round((invalidScores.length / scores.length) * 100)}%`
                : '-'}
            </Text>
          </Space>
        </Card>

        <Card title="质量评分列表">
          {scores.length > 0 ? (
            <Space direction="vertical" style={{ width: '100%' }}>
              {scores.map(item => (
                <div
                  key={item.feedback_id}
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    gap: 16,
                    borderBottom: '1px solid #f0f0f0',
                    paddingBottom: 8,
                  }}
                >
                  <div>
                    <Text strong>{item.feedback_id}</Text>
                    <div style={{ marginTop: 4 }}>
                      {(item.reasons || []).map(reason => (
                        <Tag key={reason} color={item.is_valid ? 'green' : 'red'}>
                          {reason}
                        </Tag>
                      ))}
                    </div>
                  </div>
                  <Tag color={item.is_valid ? 'green' : 'volcano'}>
                    {item.quality_score}
                  </Tag>
                </div>
              ))}
            </Space>
          ) : (
            <Text type="secondary">暂无质量评分数据</Text>
          )}
        </Card>

        <Card title="异常反馈提醒" data-testid="quality-alerts">
          {invalidScores.length > 0 ? (
            <Space wrap>
              {invalidScores.map(item => (
                <Tag key={item.feedback_id} color="red">
                  {item.feedback_id}
                </Tag>
              ))}
            </Space>
          ) : (
            <Text type="secondary">暂无异常反馈</Text>
          )}
        </Card>
      </Space>
    </div>
  )
}

export default FeedbackQualityMonitorPage
