import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'
import { Card, Table, Button, Alert, Space } from 'antd'

type Dag = { id: string; name?: string; status?: string; created_at?: string }

const DagOrchestratorPage: React.FC = () => {
  const [items, setItems] = useState<Dag[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/workflows'))
      const data = await res.json()
      setItems(data || [])
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
        <Card title="DAG 工作流">
          <Table
            rowKey="id"
            loading={loading}
            dataSource={items}
            locale={{ emptyText: '暂无工作流。' }}
            columns={[
              { title: 'ID', dataIndex: 'id' },
              { title: '名称', dataIndex: 'name' },
              { title: '状态', dataIndex: 'status' },
              { title: '创建时间', dataIndex: 'created_at' },
            ]}
          />
        </Card>
      </Space>
    </div>
  )
}

export default DagOrchestratorPage
