import { buildApiUrl, apiFetch } from '../../utils/apiBase'
import React, { useEffect, useState } from 'react'
import { Card, Space, Typography, Table, Alert, Button, Tag, Spin } from 'antd'
import { ReloadOutlined } from '@ant-design/icons'

type SessionRow = {
  session_id: string
  agent_type?: string
  is_training?: boolean
  created_at?: string
  last_updated?: string
}

type AlgorithmInfo = {
  supported_algorithms?: Array<{
    type: string
    name?: string
    description?: string
  }>
  exploration_strategies?: string[]
  reward_functions?: string[]
}

type Props = {
  title: string
  subtitle?: string
}

const QLearningLiveView: React.FC<Props> = ({ title, subtitle }) => {
  const [sessions, setSessions] = useState<SessionRow[]>([])
  const [info, setInfo] = useState<AlgorithmInfo>({})
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const [summaryRes, infoRes] = await Promise.all([
        apiFetch(buildApiUrl('/api/v1/qlearning/summary')),
        apiFetch(buildApiUrl('/api/v1/qlearning/info')),
      ])
      const summaryData = await summaryRes.json()
      setSessions(summaryData?.sessions || [])

      const infoData = await infoRes.json()
      setInfo(infoData || {})
    } catch (e: any) {
      setError(e?.message || '加载失败')
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
          <div>
            <Typography.Title level={3} style={{ margin: 0 }}>
              {title}
            </Typography.Title>
            {subtitle && (
              <Typography.Text type="secondary">{subtitle}</Typography.Text>
            )}
          </div>
          <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>
            刷新
          </Button>
        </Space>

        {error && (
          <Alert type="error" message="加载失败" description={error} showIcon />
        )}

        <Card title="会话列表">
          {loading ? (
            <Spin />
          ) : sessions.length === 0 ? (
            <Alert
              type="info"
              message="暂无实时数据"
              description="调用 /api/v1/qlearning/agents 创建会话并启动训练后再查看。"
            />
          ) : (
            <Table
              rowKey="session_id"
              size="small"
              pagination={false}
              dataSource={sessions}
              columns={[
                { title: '会话ID', dataIndex: 'session_id' },
                { title: '类型', dataIndex: 'agent_type' },
                {
                  title: '状态',
                  render: (_, row) => (row.is_training ? '训练中' : '空闲'),
                },
                { title: '创建时间', dataIndex: 'created_at' },
                { title: '更新时间', dataIndex: 'last_updated' },
              ]}
            />
          )}
        </Card>

        <Card title="算法支持">
          {loading ? (
            <Spin />
          ) : (
            <Space direction="vertical" style={{ width: '100%' }}>
              <div>
                <Typography.Text strong>算法</Typography.Text>
                <Space wrap style={{ marginTop: 8 }}>
                  {(info.supported_algorithms || []).map(item => (
                    <Tag key={item.type}>{item.name || item.type}</Tag>
                  ))}
                  {(info.supported_algorithms || []).length === 0 && (
                    <Typography.Text type="secondary">暂无</Typography.Text>
                  )}
                </Space>
              </div>
              <div>
                <Typography.Text strong>探索策略</Typography.Text>
                <Space wrap style={{ marginTop: 8 }}>
                  {(info.exploration_strategies || []).map(s => (
                    <Tag key={s}>{s}</Tag>
                  ))}
                  {(info.exploration_strategies || []).length === 0 && (
                    <Typography.Text type="secondary">暂无</Typography.Text>
                  )}
                </Space>
              </div>
              <div>
                <Typography.Text strong>奖励函数</Typography.Text>
                <Space wrap style={{ marginTop: 8 }}>
                  {(info.reward_functions || []).map(r => (
                    <Tag key={r}>{r}</Tag>
                  ))}
                  {(info.reward_functions || []).length === 0 && (
                    <Typography.Text type="secondary">暂无</Typography.Text>
                  )}
                </Space>
              </div>
            </Space>
          )}
        </Card>
      </Space>
    </div>
  )
}

export default QLearningLiveView
