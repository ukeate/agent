import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'
import { Card, Table, Button, Space, Tag, Typography, message } from 'antd'
import { ReloadOutlined, PlayCircleOutlined } from '@ant-design/icons'

interface RLTestRecord {
  test_id: string
  suite: string
  status: string
  duration_ms: number
  error?: string
  timestamp: string
}

const RLIntegrationTestPage: React.FC = () => {
  const [tests, setTests] = useState<RLTestRecord[]>([])
  const [loading, setLoading] = useState(false)

  const loadData = async () => {
    setLoading(true)
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/qlearning/tests'))
      const data = await res.json()
      setTests(data.tests || [])
    } catch (e: any) {
      message.error(e?.message || '加载测试记录失败')
      setTests([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
  }, [])

  const runAll = async () => {
    setLoading(true)
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/qlearning/tests/run'), { method: 'POST' })
      message.success('已触发测试')
      loadData()
    } catch (e: any) {
      message.error(e?.message || '触发测试失败')
    } finally {
      setLoading(false)
    }
  }

  const columns = [
    { title: 'ID', dataIndex: 'test_id', key: 'test_id' },
    { title: '套件', dataIndex: 'suite', key: 'suite' },
    { title: '状态', dataIndex: 'status', key: 'status', render: (s: string) => <Tag color={s === 'passed' ? 'green' : s === 'running' ? 'blue' : 'red'}>{s}</Tag> },
    { title: '耗时(ms)', dataIndex: 'duration_ms', key: 'duration_ms' },
    { title: '时间', dataIndex: 'timestamp', key: 'timestamp', render: (t: string) => new Date(t).toLocaleString() },
    { title: '错误', dataIndex: 'error', key: 'error', render: (e?: string) => e || '—' }
  ]

  return (
    <div style={{ padding: 24 }}>
      <Space style={{ marginBottom: 16 }}>
        <Typography.Title level={3} style={{ margin: 0 }}>RL 集成测试</Typography.Title>
        <Button icon={<ReloadOutlined />} onClick={loadData} loading={loading}>刷新</Button>
        <Button type="primary" icon={<PlayCircleOutlined />} onClick={runAll} loading={loading}>运行全部</Button>
      </Space>
      <Card>
        <Table
          rowKey="test_id"
          dataSource={tests}
          columns={columns}
          loading={loading}
          pagination={{ pageSize: 20 }}
        />
      </Card>
    </div>
  )
}

export default RLIntegrationTestPage
