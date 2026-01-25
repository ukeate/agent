import React, { useEffect, useMemo, useState } from 'react'
import { Alert, Button, Card, Space, Tag, Typography } from 'antd'
import {
  buildApiUrl,
  extractApiErrorMessage,
  normalizeHttpErrorMessage,
  apiFetch,
} from '../utils/apiBase'

const { Title, Text } = Typography

type OverviewData = {
  total_feedbacks?: number
  unique_users?: number
  average_rating?: number | null
  positive_ratio?: number | null
  feedback_types?: Record<string, number>
  top_items?: Array<{ item_id: string; feedback_count: number }>
}

type TimeFilter = 'all' | '7d' | '30d'

const TIME_FILTERS: Array<{ key: TimeFilter; label: string; days: number }> = [
  { key: 'all', label: '全部', days: 0 },
  { key: '7d', label: '最近7天', days: 7 },
  { key: '30d', label: '最近30天', days: 30 },
]

const FeedbackAnalyticsPage: React.FC = () => {
  const [overview, setOverview] = useState<OverviewData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [timeFilter, setTimeFilter] = useState<TimeFilter>('7d')
  const [typeFilter, setTypeFilter] = useState<string>('all')

  const load = async (filter: TimeFilter) => {
    setLoading(true)
    setError(null)
    try {
      const params = new URLSearchParams()
      const range = TIME_FILTERS.find(item => item.key === filter)
      if (range && range.days > 0) {
        const end = new Date()
        const start = new Date()
        start.setDate(end.getDate() - range.days)
        params.set('start_date', start.toISOString())
        params.set('end_date', end.toISOString())
      }
      const url = buildApiUrl(
        `/api/v1/feedback/overview${params.toString() ? `?${params.toString()}` : ''}`
      )
      const res = await apiFetch(url)
      const payload = await res.json()
      if (!res.ok || payload?.success === false) {
        const message = extractApiErrorMessage(payload, '加载失败')
        throw new Error(normalizeHttpErrorMessage(res.status, message))
      }
      setOverview(payload?.data || null)
    } catch (err: any) {
      setError(err?.message || '加载失败')
      setOverview(null)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load(timeFilter)
  }, [timeFilter])

  const typeRows = useMemo(() => {
    const entries = Object.entries(overview?.feedback_types || {})
    return entries.map(([label, value]) => ({ label, value }))
  }, [overview?.feedback_types])

  const topItems = overview?.top_items || []
  const averageRating =
    typeof overview?.average_rating === 'number'
      ? overview?.average_rating.toFixed(2)
      : '-'

  const positiveRatio = useMemo(() => {
    if (typeof overview?.positive_ratio === 'number') {
      return `${Math.round(overview.positive_ratio * 100)}%`
    }
    const likeCount = overview?.feedback_types?.like || 0
    const dislikeCount = overview?.feedback_types?.dislike || 0
    const total = likeCount + dislikeCount
    if (!total) return '-'
    return `${Math.round((likeCount / total) * 100)}%`
  }, [overview?.feedback_types, overview?.positive_ratio])

  const filteredTypes = useMemo(() => {
    if (typeFilter === 'all') return typeRows
    return typeRows.filter(item => item.label === typeFilter)
  }, [typeFilter, typeRows])

  const chartTitle =
    typeFilter === 'all' ? '反馈类型分布' : `${typeFilter} 反馈`

  return (
    <div style={{ padding: 24 }} data-testid="feedback-analytics-page">
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <div>
          <Title level={3} style={{ marginBottom: 4 }}>
            反馈数据分析
          </Title>
          <Text type="secondary">
            聚合反馈类型、评分趋势与高频反馈对象。
          </Text>
        </div>

        <Space wrap>
          <div data-testid="time-filter">
            <Space>
              {TIME_FILTERS.map(item => (
                <Button
                  key={item.key}
                  size="small"
                  type={timeFilter === item.key ? 'primary' : 'default'}
                  onClick={() => setTimeFilter(item.key)}
                  data-testid={
                    item.key === '7d'
                      ? 'time-filter-7days'
                      : item.key === '30d'
                        ? 'time-filter-30days'
                        : 'time-filter-all'
                  }
                >
                  {item.label}
                </Button>
              ))}
            </Space>
          </div>
          <div data-testid="feedback-type-filter">
            <Space>
              <Button
                size="small"
                type={typeFilter === 'all' ? 'primary' : 'default'}
                onClick={() => setTypeFilter('all')}
              >
                全部
              </Button>
              {typeRows.map(item => (
                <Button
                  key={item.label}
                  size="small"
                  type={typeFilter === item.label ? 'primary' : 'default'}
                  onClick={() => setTypeFilter(item.label)}
                  data-testid={
                    item.label === 'rating' ? 'filter-rating' : undefined
                  }
                >
                  {item.label}
                </Button>
              ))}
            </Space>
          </div>
          <Button onClick={() => load(timeFilter)} loading={loading}>
            刷新
          </Button>
        </Space>

        {error && (
          <Alert type="error" message={error} data-testid="data-load-error" />
        )}

        <Card>
          <Space wrap>
            <Text>
              总反馈：
              <span data-testid="total-feedback-count">
                {overview?.total_feedbacks ?? 0}
              </span>
            </Text>
            <Text>活跃用户：{overview?.unique_users ?? 0}</Text>
            <Text data-testid="average-rating">平均评分：{averageRating}</Text>
            <Text data-testid="positive-feedback-ratio">
              正向比例：{positiveRatio}
            </Text>
            <Text data-testid="active-filter">
              当前筛选：
              {TIME_FILTERS.find(item => item.key === timeFilter)?.label || '全部'}
            </Text>
          </Space>
        </Card>

        <Card
          title={<span data-testid="chart-title">{chartTitle}</span>}
          data-testid="feedback-chart"
        >
          {loading ? (
            <div data-testid="chart-loading">加载中...</div>
          ) : (
            <Space wrap data-testid="chart-loaded">
              {filteredTypes.length > 0 ? (
                filteredTypes.map(item => (
                  <Tag key={item.label} data-testid="chart-data-point">
                    {item.label}: {item.value}
                  </Tag>
                ))
              ) : (
                <Text type="secondary">暂无数据</Text>
              )}
            </Space>
          )}
        </Card>

        <Card title="高频反馈对象">
          <Space direction="vertical" style={{ width: '100%' }}>
            {topItems.length > 0 ? (
              topItems.map(item => (
                <div key={item.item_id} style={{ display: 'flex', gap: 12 }}>
                  <Text strong>{item.item_id}</Text>
                  <Text type="secondary">
                    反馈 {item.feedback_count}
                  </Text>
                </div>
              ))
            ) : (
              <Text type="secondary">暂无高频反馈对象</Text>
            )}
          </Space>
        </Card>
      </Space>
    </div>
  )
}

export default FeedbackAnalyticsPage
