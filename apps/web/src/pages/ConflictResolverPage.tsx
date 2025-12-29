import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'
import { Card, Table, Tag, Space, Button, Typography, message } from 'antd'
import { ReloadOutlined, SafetyOutlined } from '@ant-design/icons'

interface ConflictRecord {
  conflict_id: string
  conflict_type: string
  severity: string
  status: string
  detected_at: string
  involved_tasks?: string[]
  involved_agents?: string[]
  resolution_strategy?: string
}

const ConflictResolverPage: React.FC = () => {
  const [conflicts, setConflicts] = useState<ConflictRecord[]>([])
  const [loading, setLoading] = useState(false)

  const loadData = async () => {
    setLoading(true)
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/distributed-task/conflicts'))
      const data = await res.json()
      setConflicts(data.conflicts || [])
    } catch (e: any) {
      message.error(e?.message || '加载冲突数据失败')
      setConflicts([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
    const timer = setInterval(loadData, 5000)
    return () => clearInterval(timer)
  }, [])

  const columns = [
    { title: 'ID', dataIndex: 'conflict_id', key: 'conflict_id' },
    { title: '类型', dataIndex: 'conflict_type', key: 'conflict_type' },
    { title: '严重度', dataIndex: 'severity', key: 'severity', render: (s: string) => <Tag color={s === 'critical' ? 'red' : s === 'high' ? 'orange' : 'blue'}>{s}</Tag> },
    { title: '状态', dataIndex: 'status', key: 'status', render: (s: string) => <Tag color={s === 'resolved' ? 'green' : 'gold'}>{s}</Tag> },
    { title: '检测时间', dataIndex: 'detected_at', key: 'detected_at', render: (t: string) => new Date(t).toLocaleString() },
    { title: '涉及任务', dataIndex: 'involved_tasks', key: 'involved_tasks', render: (arr?: string[]) => arr?.join(', ') || '—' },
    { title: '涉及代理', dataIndex: 'involved_agents', key: 'involved_agents', render: (arr?: string[]) => arr?.join(', ') || '—' },
    { title: '解决策略', dataIndex: 'resolution_strategy', key: 'resolution_strategy', render: (s?: string) => s || '—' },
  ]

  return (
    <div style={{ padding: 24 }}>
      <Space style={{ marginBottom: 16 }}>
        <SafetyOutlined />
        <Typography.Title level={3} style={{ margin: 0 }}>冲突检测与解决</Typography.Title>
        <Button icon={<ReloadOutlined />} onClick={loadData} loading={loading}>刷新</Button>
      </Space>
      <Typography.Text type="secondary">数据来自 /api/v1/distributed-task/conflicts，无本地模拟。</Typography.Text>

      <Card style={{ marginTop: 12 }}>
        <Table
          rowKey="conflict_id"
          dataSource={conflicts}
          columns={columns}
          loading={loading}
          pagination={{ pageSize: 20 }}
        />
      </Card>
    </div>
  )
}

export default ConflictResolverPage
