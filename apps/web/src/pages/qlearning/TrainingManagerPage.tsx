import { buildApiUrl, apiFetch } from '../../utils/apiBase'
import React, { useEffect, useState } from 'react'
import { Card, Space, Typography, Button, Table, Alert, Spin } from 'antd'
import { ControlOutlined, ReloadOutlined } from '@ant-design/icons'

const { Title } = Typography

const TrainingManagerPage: React.FC = () => {
  const [jobs, setJobs] = useState<any[]>([])
  const [checkpoints, setCheckpoints] = useState<any[]>([])
  const [loading, setLoading] = useState(false)

  const load = async () => {
    setLoading(true)
    try {
      const [sessionsRes, checkpointsRes] = await Promise.all([
        apiFetch(buildApiUrl('/api/v1/qlearning/sessions')),
        apiFetch(buildApiUrl('/api/v1/qlearning/checkpoints'))
      ])
      const sessionsData = await sessionsRes.json()
      const checkpointsData = await checkpointsRes.json()
      setJobs(sessionsData.sessions || [])
      setCheckpoints(checkpointsData.checkpoints || [])
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
        <Space align="center" style={{ justifyContent: 'space-between', width: '100%' }}>
          <Title level={3} style={{ margin: 0 }}>
            <ControlOutlined /> Q-Learning 训练管理
          </Title>
          <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>
            刷新
          </Button>
        </Space>

        {loading ? (
          <Spin />
        ) : (
          <>
            <Card title="训练作业">
              {jobs.length === 0 ? (
                <Alert type="info" message="暂无训练作业，先调用 /api/v1/qlearning 创建会话并启动训练。" />
              ) : (
                <Table
                  rowKey="session_id"
                  dataSource={jobs}
                  columns={[
                    { title: '会话ID', dataIndex: 'session_id' },
                    { title: '类型', dataIndex: 'agent_type' },
                    {
                      title: '状态',
                      render: (_, row) => (row.is_training ? '训练中' : '空闲'),
                    },
                    { title: '创建时间', dataIndex: 'created_at' },
                  ]}
                />
              )}
            </Card>

            <Card title="检查点">
              {checkpoints.length === 0 ? (
                <Alert type="info" message="暂无检查点。" />
              ) : (
                <Table
                  rowKey="id"
                  dataSource={checkpoints}
                  columns={[
                    { title: 'ID', dataIndex: 'id' },
                    { title: '会话ID', dataIndex: 'session_id' },
                    { title: '回合数', dataIndex: 'episode' },
                    { title: '时间', dataIndex: 'timestamp' },
                  ]}
                />
              )}
            </Card>
          </>
        )}
      </Space>
    </div>
  )
}

export default TrainingManagerPage
