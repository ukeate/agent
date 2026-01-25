import React, { useCallback, useEffect, useState } from 'react'
import {
  Alert,
  Button,
  Card,
  Input,
  Space,
  Tag,
  Typography,
} from 'antd'
import { ReloadOutlined } from '@ant-design/icons'
import {
  apiFetch,
  buildApiUrl,
  extractApiErrorMessage,
  normalizeHttpErrorMessage,
} from '../utils/apiBase'

const { Title, Text } = Typography

type ItemAnalytics = {
  item_id: string
  total_feedbacks: number
  average_rating?: number | null
  like_ratio: number
  engagement_metrics: Record<string, number>
  feedback_distribution: Record<string, number>
}

type TopItem = {
  item_id: string
  feedback_count: number
}

const ItemFeedbackAnalysisPage: React.FC = () => {
  const [items, setItems] = useState<TopItem[]>([])
  const [selected, setSelected] = useState<ItemAnalytics | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [searchId, setSearchId] = useState('demo-item')

  const loadTopItems = useCallback(async () => {
    const res = await apiFetch(buildApiUrl('/api/v1/feedback/overview'))
    const payload = await res.json()
    if (!res.ok || payload?.success === false) {
      const message = extractApiErrorMessage(payload, '加载失败')
      throw new Error(normalizeHttpErrorMessage(res.status, message))
    }
    const list = Array.isArray(payload?.data?.top_items)
      ? payload.data.top_items.map((item: any) => ({
          item_id: item.item_id,
          feedback_count: item.feedback_count,
        }))
      : []
    setItems(list)
  }, [])

  const loadItem = useCallback(async (itemId: string) => {
    const res = await apiFetch(
      buildApiUrl(`/api/v1/feedback/analytics/item/${encodeURIComponent(itemId)}`)
    )
    const payload = await res.json()
    if (!res.ok || payload?.success === false) {
      const message = extractApiErrorMessage(payload, '加载失败')
      throw new Error(normalizeHttpErrorMessage(res.status, message))
    }
    setSelected(payload?.data || null)
  }, [])

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      await loadTopItems()
    } catch (err: any) {
      setError(err?.message || '加载失败')
      setItems([])
    } finally {
      setLoading(false)
    }
  }, [loadTopItems])

  useEffect(() => {
    load()
  }, [load])

  const handleSearch = async () => {
    if (!searchId.trim()) return
    setLoading(true)
    setError(null)
    try {
      await loadItem(searchId.trim())
    } catch (err: any) {
      setError(err?.message || '加载失败')
      setSelected(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ padding: 24 }} data-testid="item-feedback-page">
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <Title level={3} style={{ margin: 0 }}>
            推荐项反馈分析
          </Title>
          <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>
            刷新
          </Button>
        </div>

        <Space wrap>
          <Input
            value={searchId}
            onChange={event => setSearchId(event.target.value)}
            placeholder="输入物品ID"
            style={{ width: 240 }}
            data-testid="item-search"
          />
          <Button type="primary" onClick={handleSearch} data-testid="search-button">
            查询
          </Button>
        </Space>

        {error && <Alert type="error" message={error} />}

        {selected && (
          <Card title="物品反馈汇总" data-testid="item-feedback-summary">
            <Space wrap>
              <Text>
                物品：<strong>{selected.item_id}</strong>
              </Text>
              <Text data-testid="feedback-count">
                反馈数：{selected.total_feedbacks}
              </Text>
              <Text data-testid="item-rating">
                平均评分：{selected.average_rating ?? '-'}
              </Text>
              <Text>
                好评率：{Math.round((selected.like_ratio || 0) * 100)}%
              </Text>
            </Space>
          </Card>
        )}

        <Card title="高频反馈物品">
          {items.length > 0 ? (
            <Space wrap>
              {items.map(item => (
                <Tag
                  key={item.item_id}
                  color="blue"
                  data-testid="item-feedback-item"
                  onClick={() => loadItem(item.item_id)}
                  style={{ cursor: 'pointer' }}
                >
                  {item.item_id} · {item.feedback_count}
                </Tag>
              ))}
            </Space>
          ) : (
            <Text type="secondary">暂无数据</Text>
          )}
        </Card>

        {selected && (
          <Card title="反馈类型分布">
            <Space wrap>
              {Object.entries(selected.feedback_distribution || {}).map(
                ([key, value]) => (
                  <Tag key={key}>
                    {key}: {value}
                  </Tag>
                )
              )}
            </Space>
          </Card>
        )}
      </Space>
    </div>
  )
}

export default ItemFeedbackAnalysisPage
