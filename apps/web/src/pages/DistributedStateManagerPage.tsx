import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'
import { Card, Table, Alert, Button, Space } from 'antd'

type StateNode = {
  node_id: string
  role?: string
  version?: string
  last_heartbeat?: string
}

const DistributedStateManagerPage: React.FC = () => {
  const [nodes, setNodes] = useState<StateNode[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/distributed-state/nodes'))
      const data = await res.json()
      setNodes(data?.nodes || [])
    } catch (e: any) {
      setError(e?.message || '加载失败')
      setNodes([])
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
        <Card title="分布式状态节点">
          <Table
            rowKey="node_id"
            loading={loading}
            dataSource={nodes}
            locale={{ emptyText: '暂无节点，请在后端注册状态节点。' }}
            columns={[
              { title: '节点', dataIndex: 'node_id' },
              { title: '角色', dataIndex: 'role' },
              { title: '版本', dataIndex: 'version' },
              { title: '最近心跳', dataIndex: 'last_heartbeat' },
            ]}
          />
        </Card>
      </Space>
    </div>
  )
}

export default DistributedStateManagerPage
