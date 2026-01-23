import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'
import { Card, Table, Alert, Button, Space } from 'antd'

type QAItem = {
  id: string
  task_id?: string
  issue?: string
  severity?: string
  created_at?: string
}

const AnnotationQualityControlPage: React.FC = () => {
  const [items, setItems] = useState<QAItem[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await apiFetch(
        buildApiUrl('/api/v1/training-data/annotation-tasks-issues')
      )
      const data = await res.json()
      setItems(data?.issues || [])
    } catch (e: any) {
      setError(e?.message || '加载失败')
      setItems([])
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
        <Button onClick={load} loading={loading}>
          刷新
        </Button>
        {error && <Alert type="error" message={error} />}
        <Card title="质检问题">
          <Table
            rowKey="id"
            loading={loading}
            dataSource={items}
            locale={{ emptyText: '暂无质检记录，请先提交标注或质检结果。' }}
            columns={[
              { title: 'ID', dataIndex: 'id' },
              { title: '任务', dataIndex: 'task_id' },
              { title: '问题', dataIndex: 'issue' },
              { title: '严重级别', dataIndex: 'severity' },
              { title: '时间', dataIndex: 'created_at' },
            ]}
          />
        </Card>
      </Space>
    </div>
  )
}

export default AnnotationQualityControlPage
