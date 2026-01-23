import { buildApiUrl, apiFetch } from '../../utils/apiBase'
import React, { useEffect, useState } from 'react'
import { Card, Table, Typography, Space, Button, Alert, Spin } from 'antd'
import { LineChartOutlined, ReloadOutlined } from '@ant-design/icons'

const { Title } = Typography

interface Summary {
  sessions: any[]
}

const UCBStrategiesPage: React.FC = () => {
  const [summary, setSummary] = useState<Summary>({ sessions: [] })
  const [loading, setLoading] = useState(false)

  const load = async () => {
    setLoading(true)
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/qlearning/summary'))
      const data = await res.json()
      setSummary(data)
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
          <Title level={3} style={{ margin: 0 }}>
            <LineChartOutlined /> UCB 多臂老虎机策略状态
          </Title>
          <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>
            刷新
          </Button>
        </Space>

        <Card title="会话列表">
          {loading ? (
            <Spin />
          ) : summary.sessions.length === 0 ? (
            <Alert
              type="info"
              message="暂无Q-Learning会话，请先通过 /api/v1/qlearning 接口创建会话。"
            />
          ) : (
            <Table
              rowKey={r => r.session_id}
              dataSource={summary.sessions}
              columns={[
                { title: '会话ID', dataIndex: 'session_id' },
                { title: '类型', dataIndex: 'agent_type' },
                { title: '创建时间', dataIndex: 'created_at' },
                {
                  title: '状态',
                  render: (_, row) => (row.is_training ? '训练中' : '空闲'),
                },
              ]}
            />
          )}
        </Card>
      </Space>
    </div>
  )
}

export default UCBStrategiesPage
