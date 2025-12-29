import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'
import { Card, Table, Button, Space, Typography, Tag, Drawer, Descriptions, message } from 'antd'
import { ReloadOutlined, SearchOutlined } from '@ant-design/icons'

type ItemAnalytics = {
  item_id: string
  feedback_count?: number
  avg_rating?: number
  like_ratio?: number
  last_feedback_time?: string
  top_feedbacks?: any[]
}

const ItemFeedbackAnalysisPage: React.FC = () => {
  const [items, setItems] = useState<ItemAnalytics[]>([])
  const [selected, setSelected] = useState<ItemAnalytics | null>(null)
  const [drawerVisible, setDrawerVisible] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/feedback/overview'))
      const data = await res.json()
      const list = Array.isArray(data?.top_items)
        ? data.top_items.map((t: any) => ({ item_id: t.item_id, feedback_count: t.feedback_count }))
        : []
      setItems(list)
    } catch (e: any) {
      setError(e?.message || '加载失败')
      setItems([])
    } finally {
      setLoading(false)
    }
  }

  const loadItem = async (itemId: string) => {
    try {
      const res = await apiFetch(buildApiUrl(`/api/v1/feedback/analytics/item/${itemId}`))
      const data = await res.json()
      setSelected(data)
      setDrawerVisible(true)
    } catch (e: any) {
      message.error(e?.message || '加载详情失败')
    }
  }

  useEffect(() => {
    load()
  }, [])

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Space align="center" style={{ justifyContent: 'space-between', width: '100%' }}>
          <Typography.Title level={3} style={{ margin: 0 }}>
            推荐项反馈分析
          </Typography.Title>
          <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>
            刷新
          </Button>
        </Space>

        {error && <Typography.Text type="danger">{error}</Typography.Text>}

        <Card>
          <Table
            rowKey="item_id"
            loading={loading}
            dataSource={items}
            locale={{ emptyText: '暂无数据' }}
            columns={[
              { title: 'Item', dataIndex: 'item_id' },
              { title: '反馈数', dataIndex: 'feedback_count' },
              {
                title: '操作',
                render: (_, record) => (
                  <Button icon={<SearchOutlined />} size="small" onClick={() => loadItem(record.item_id)}>
                    查看
                  </Button>
                )
              }
            ]}
          />
        </Card>

        <Drawer open={drawerVisible} onClose={() => setDrawerVisible(false)} width={420} title="详情">
          {selected ? (
            <Descriptions column={1} bordered>
              <Descriptions.Item label="Item">{selected.item_id}</Descriptions.Item>
              <Descriptions.Item label="反馈数">{selected.feedback_count}</Descriptions.Item>
              <Descriptions.Item label="平均评分">{selected.avg_rating}</Descriptions.Item>
              <Descriptions.Item label="好评率">
                <Tag color="green">{Math.round((selected.like_ratio || 0) * 100)}%</Tag>
              </Descriptions.Item>
              <Descriptions.Item label="最近反馈">{selected.last_feedback_time}</Descriptions.Item>
            </Descriptions>
          ) : (
            <Typography.Text>未选择</Typography.Text>
          )}
        </Drawer>
      </Space>
    </div>
  )
}

export default ItemFeedbackAnalysisPage
