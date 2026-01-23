import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'
import { Card, Table, Space, Typography, Button, Alert, Spin } from 'antd'
import { LinkOutlined, ReloadOutlined } from '@ant-design/icons'

type Relation = {
  id: string
  source: string
  target: string
  type: string
  weight?: number
  created_at?: string
}

const KnowledgeGraphRelationManagement: React.FC = () => {
  const [relations, setRelations] = useState<Relation[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await apiFetch(
        buildApiUrl('/api/v1/knowledge-graph/relations')
      )
      const data = await res.json()
      setRelations(Array.isArray(data?.relations) ? data.relations : [])
    } catch (e: any) {
      setError(e?.message || '加载失败')
      setRelations([])
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
        <Space
          align="center"
          style={{ justifyContent: 'space-between', width: '100%' }}
        >
          <Space>
            <LinkOutlined />
            <Typography.Title level={3} style={{ margin: 0 }}>
              关系管理
            </Typography.Title>
          </Space>
          <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>
            刷新
          </Button>
        </Space>

        {error && <Alert type="error" message="加载失败" description={error} />}

        <Card>
          {loading ? (
            <Spin />
          ) : (
            <Table
              rowKey="id"
              dataSource={relations}
              columns={[
                { title: 'ID', dataIndex: 'id' },
                { title: '源', dataIndex: 'source' },
                { title: '目标', dataIndex: 'target' },
                { title: '类型', dataIndex: 'type' },
                { title: '权重', dataIndex: 'weight' },
                { title: '创建时间', dataIndex: 'created_at' },
              ]}
              locale={{ emptyText: '暂无关系数据。' }}
            />
          )}
        </Card>
      </Space>
    </div>
  )
}

export default KnowledgeGraphRelationManagement
