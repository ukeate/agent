import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'
import { Card, Table, Button, Alert, Space } from 'antd'

type Workflow = {
  workflow_id: string
  workflow_type?: string
  status?: string
  started_at?: string
  completed_at?: string
  current_step?: string
  error?: string
}

const WorkflowOrchestrationPage: React.FC = () => {
  const [items, setItems] = useState<Workflow[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await apiFetch(
        buildApiUrl('/api/v1/platform/workflows?limit=50')
      )
      const data = await res.json()
      setItems(Array.isArray(data?.workflows) ? data.workflows : [])
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
        <Card title="工作流列表">
          <Table
            rowKey="workflow_id"
            loading={loading}
            dataSource={items}
            locale={{ emptyText: '暂无工作流，请先通过后端创建。' }}
            columns={[
              { title: 'ID', dataIndex: 'workflow_id' },
              { title: '类型', dataIndex: 'workflow_type' },
              { title: '状态', dataIndex: 'status' },
              { title: '当前步骤', dataIndex: 'current_step' },
              { title: '开始时间', dataIndex: 'started_at' },
              { title: '结束时间', dataIndex: 'completed_at' },
              { title: '错误', dataIndex: 'error' },
            ]}
          />
        </Card>
      </Space>
    </div>
  )
}

export default WorkflowOrchestrationPage
