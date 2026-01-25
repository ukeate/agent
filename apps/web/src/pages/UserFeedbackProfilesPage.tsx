import React, { useCallback, useState } from 'react'
import {
  Alert,
  Button,
  Card,
  Input,
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

type UserAnalytics = {
  user_id: string
  total_feedbacks: number
  feedback_distribution: Record<string, number>
  average_rating?: number | null
  engagement_score: number
  last_activity?: string | null
  preference_vector?: number[]
  trust_score?: number
}

type HistoryItem = {
  event_id: string
  feedback_type: string
  value: any
  item_id?: string
  timestamp?: string
}

const UserFeedbackProfilesPage: React.FC = () => {
  const [userId, setUserId] = useState('demo-user')
  const [analytics, setAnalytics] = useState<UserAnalytics | null>(null)
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showDetail, setShowDetail] = useState(false)

  const load = useCallback(async () => {
    if (!userId.trim()) {
      setError('请输入用户ID')
      return
    }
    setLoading(true)
    setError(null)
    try {
      const analyticsRes = await apiFetch(
        buildApiUrl(`/api/v1/feedback/analytics/user/${encodeURIComponent(userId.trim())}`)
      )
      const analyticsPayload = await analyticsRes.json()
      if (!analyticsRes.ok || analyticsPayload?.success === false) {
        const message = extractApiErrorMessage(analyticsPayload, '加载失败')
        throw new Error(normalizeHttpErrorMessage(analyticsRes.status, message))
      }

      const historyRes = await apiFetch(
        buildApiUrl(`/api/v1/feedback/user/${encodeURIComponent(userId.trim())}?limit=20`)
      )
      const historyPayload = await historyRes.json()
      if (!historyRes.ok || historyPayload?.success === false) {
        const message = extractApiErrorMessage(historyPayload, '加载失败')
        throw new Error(normalizeHttpErrorMessage(historyRes.status, message))
      }

      setAnalytics(analyticsPayload?.data || null)
      setHistory(
        Array.isArray(historyPayload?.data?.items)
          ? historyPayload.data.items
          : []
      )
    } catch (err: any) {
      setError(err?.message || '加载失败')
      setAnalytics(null)
      setHistory([])
    } finally {
      setLoading(false)
    }
  }, [userId])

  return (
    <div style={{ padding: 24 }} data-testid="user-profiles-page">
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <div>
          <Title level={3} style={{ marginBottom: 4 }}>
            用户反馈档案
          </Title>
          <Text type="secondary">输入用户ID查看反馈画像与历史记录。</Text>
        </div>

        <Space wrap>
          <Input
            value={userId}
            onChange={event => setUserId(event.target.value)}
            placeholder="输入用户ID"
            style={{ width: 240 }}
            data-testid="user-search"
          />
          <Button
            type="primary"
            onClick={load}
            loading={loading}
            data-testid="search-button"
          >
            查询
          </Button>
        </Space>

        {error && <Alert type="error" message={error} />}

        {analytics && (
          <Card
            data-testid="user-profile-item"
            onClick={() => setShowDetail(prev => !prev)}
            style={{ cursor: 'pointer' }}
          >
            <Space wrap>
              <Text strong>{analytics.user_id}</Text>
              <Text>总反馈：{analytics.total_feedbacks}</Text>
              <Text>平均评分：{analytics.average_rating ?? '-'}</Text>
              <Text>参与度：{analytics.engagement_score}</Text>
              <Text>信任度：{analytics.trust_score ?? '-'}</Text>
            </Space>
          </Card>
        )}

        {analytics && showDetail && (
          <Card title="反馈偏好分析" data-testid="preference-analysis">
            <Space direction="vertical" style={{ width: '100%' }}>
              <div data-testid="category-preferences">
                <Text strong>类型偏好：</Text>
                <Space wrap>
                  {Object.entries(analytics.feedback_distribution || {}).map(
                    ([key, value]) => (
                      <Tag key={key}>
                        {key}: {value}
                      </Tag>
                    )
                  )}
                </Space>
              </div>
              <div data-testid="feedback-patterns">
                <Text strong>偏好向量：</Text>
                <Text type="secondary">
                  {(analytics.preference_vector || []).join(', ') || '暂无数据'}
                </Text>
              </div>
              <div data-testid="engagement-metrics">
                <Text strong>最近活动：</Text>
                <Text type="secondary">{analytics.last_activity || '--'}</Text>
              </div>
            </Space>
          </Card>
        )}

        <Card title="反馈历史" data-testid="user-feedback-history">
          {history.length > 0 ? (
            <Space direction="vertical" style={{ width: '100%' }}>
              {history.map(item => (
                <div
                  key={item.event_id}
                  data-testid="feedback-item"
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    borderBottom: '1px solid #f0f0f0',
                    paddingBottom: 8,
                  }}
                >
                  <div>
                    <Text strong>{item.feedback_type}</Text>
                    <div>
                      <Text type="secondary">
                        {item.item_id || '未绑定物品'}
                      </Text>
                    </div>
                  </div>
                  <Text type="secondary">{item.timestamp || '--'}</Text>
                </div>
              ))}
            </Space>
          ) : (
            <Text type="secondary">暂无反馈历史</Text>
          )}
        </Card>
      </Space>
    </div>
  )
}

export default UserFeedbackProfilesPage
