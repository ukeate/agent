import React, { useState } from 'react'
import {
  Card,
  Row,
  Col,
  Progress,
  Tag,
  Space,
  Button,
  Table,
  Statistic,
  Timeline,
  Alert,
  Tooltip,
} from 'antd'
import {
  PlayCircleOutlined,
  PauseCircleOutlined,
  StopOutlined,
  ClockCircleOutlined,
  ThunderboltOutlined,
  CheckCircleOutlined,
  LoadingOutlined,
  LineChartOutlined,
} from '@ant-design/icons'

interface AgentStats {
  agent_id: string
  algorithm_type: string
  episode_count: number
  step_count: number
  current_epsilon: number
  total_reward: number
  average_reward: number
  recent_rewards: number[]
}

interface TrainingSession {
  session_id: string
  agent_id: string
  status: 'idle' | 'running' | 'paused' | 'completed'
  total_episodes: number
  current_episode: number
  average_reward: number
  convergence_achieved: boolean
  start_time: string
  elapsed_time?: number
}

interface QLearningTrainingPanelProps {
  trainingSessions: TrainingSession[]
  agents: AgentStats[]
  onStartTraining: (agentId: string, episodes: number) => void
  loading: boolean
}

const QLearningTrainingPanel: React.FC<QLearningTrainingPanelProps> = ({
  trainingSessions,
  agents,
  onStartTraining,
  loading,
}) => {
  const [selectedSession, setSelectedSession] = useState<string | null>(null)

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running':
        return <LoadingOutlined style={{ color: '#52c41a' }} />
      case 'paused':
        return <PauseCircleOutlined style={{ color: '#faad14' }} />
      case 'completed':
        return <CheckCircleOutlined style={{ color: '#1890ff' }} />
      default:
        return <ClockCircleOutlined style={{ color: '#d9d9d9' }} />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return 'green'
      case 'paused':
        return 'orange'
      case 'completed':
        return 'blue'
      default:
        return 'default'
    }
  }

  const getStatusText = (status: string) => {
    switch (status) {
      case 'running':
        return '训练中'
      case 'paused':
        return '已暂停'
      case 'completed':
        return '已完成'
      default:
        return '空闲'
    }
  }

  const formatElapsedTime = (seconds: number = 0) => {
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    const remainingSeconds = seconds % 60

    if (hours > 0) {
      return `${hours}h ${minutes}m ${remainingSeconds}s`
    } else if (minutes > 0) {
      return `${minutes}m ${remainingSeconds}s`
    } else {
      return `${remainingSeconds}s`
    }
  }

  const getAlgorithmName = (algorithm: string) => {
    switch (algorithm) {
      case 'q_learning':
        return 'Classic Q-Learning'
      case 'dqn':
        return 'Deep Q-Network'
      case 'double_dqn':
        return 'Double DQN'
      case 'dueling_dqn':
        return 'Dueling DQN'
      default:
        return algorithm
    }
  }

  const getAgent = (agentId: string) => {
    return agents.find(agent => agent.agent_id === agentId)
  }

  const columns = [
    {
      title: '会话ID',
      dataIndex: 'session_id',
      key: 'session_id',
      width: 120,
      render: (text: string) => (
        <span style={{ fontFamily: 'monospace', fontSize: '12px' }}>
          {text.slice(-8)}
        </span>
      ),
    },
    {
      title: '智能体',
      dataIndex: 'agent_id',
      key: 'agent_id',
      width: 140,
      render: (agentId: string) => {
        const agent = getAgent(agentId)
        return (
          <div>
            <div style={{ fontSize: '12px', fontFamily: 'monospace' }}>
              {agentId.slice(-8)}
            </div>
            {agent && (
              <Tag size="small" color="blue">
                {getAlgorithmName(agent.algorithm_type)}
              </Tag>
            )}
          </div>
        )
      },
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (status: string) => (
        <Space>
          {getStatusIcon(status)}
          <Tag color={getStatusColor(status)}>{getStatusText(status)}</Tag>
        </Space>
      ),
    },
    {
      title: '训练进度',
      key: 'progress',
      width: 200,
      render: (_, record: TrainingSession) => {
        const progress = (record.current_episode / record.total_episodes) * 100
        return (
          <div>
            <Progress
              percent={Math.round(progress)}
              size="small"
              status={record.status === 'running' ? 'active' : 'normal'}
              format={percent =>
                `${record.current_episode}/${record.total_episodes}`
              }
            />
            <div style={{ fontSize: '11px', color: '#999', marginTop: '2px' }}>
              进度: {progress.toFixed(1)}%
            </div>
          </div>
        )
      },
    },
    {
      title: '性能',
      key: 'performance',
      width: 120,
      render: (_, record: TrainingSession) => (
        <div>
          <Statistic
            title="平均奖励"
            value={record.average_reward}
            precision={1}
            valueStyle={{ fontSize: '14px' }}
          />
          {record.convergence_achieved && (
            <Tag color="green" size="small">
              已收敛
            </Tag>
          )}
        </div>
      ),
    },
    {
      title: '运行时间',
      key: 'elapsed_time',
      width: 100,
      render: (_, record: TrainingSession) => (
        <div>
          <ClockCircleOutlined style={{ color: '#999', marginRight: '4px' }} />
          <span style={{ fontSize: '12px' }}>
            {formatElapsedTime(record.elapsed_time)}
          </span>
        </div>
      ),
    },
    {
      title: '操作',
      key: 'actions',
      width: 120,
      render: (_, record: TrainingSession) => (
        <Space>
          {record.status === 'running' ? (
            <>
              <Tooltip title="暂停训练">
                <Button
                  type="text"
                  icon={<PauseCircleOutlined />}
                  size="small"
                />
              </Tooltip>
              <Tooltip title="停止训练">
                <Button
                  type="text"
                  danger
                  icon={<StopOutlined />}
                  size="small"
                />
              </Tooltip>
            </>
          ) : record.status === 'paused' ? (
            <Tooltip title="继续训练">
              <Button type="text" icon={<PlayCircleOutlined />} size="small" />
            </Tooltip>
          ) : (
            <Tooltip title="查看结果">
              <Button type="text" icon={<LineChartOutlined />} size="small" />
            </Tooltip>
          )}
        </Space>
      ),
    },
  ]

  const runningSessions = trainingSessions.filter(s => s.status === 'running')
  const completedSessions = trainingSessions.filter(
    s => s.status === 'completed'
  )

  return (
    <Row gutter={[16, 16]}>
      {/* 训练概览统计 */}
      <Col span={24}>
        <Row gutter={16}>
          <Col span={6}>
            <Card>
              <Statistic
                title="活跃训练会话"
                value={runningSessions.length}
                prefix={<ThunderboltOutlined style={{ color: '#52c41a' }} />}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="已完成会话"
                value={completedSessions.length}
                prefix={<CheckCircleOutlined style={{ color: '#1890ff' }} />}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="总训练时长"
                value={trainingSessions.reduce(
                  (sum, s) => sum + (s.elapsed_time || 0),
                  0
                )}
                formatter={value => formatElapsedTime(Number(value))}
                prefix={<ClockCircleOutlined style={{ color: '#722ed1' }} />}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="平均收敛率"
                value={
                  completedSessions.length > 0
                    ? (completedSessions.filter(s => s.convergence_achieved)
                        .length /
                        completedSessions.length) *
                      100
                    : 0
                }
                precision={1}
                suffix="%"
                prefix={<LineChartOutlined style={{ color: '#faad14' }} />}
              />
            </Card>
          </Col>
        </Row>
      </Col>

      {/* 活跃训练会话详情 */}
      {runningSessions.length > 0 && (
        <Col span={24}>
          <Card
            title={
              <Space>
                <ThunderboltOutlined />
                活跃训练会话详情
              </Space>
            }
          >
            <Row gutter={16}>
              {runningSessions.map(session => {
                const agent = getAgent(session.agent_id)
                const progress =
                  (session.current_episode / session.total_episodes) * 100

                return (
                  <Col span={8} key={session.session_id}>
                    <Card size="small" hoverable>
                      <Space direction="vertical" style={{ width: '100%' }}>
                        <div>
                          <Tag color="green">训练中</Tag>
                          <span style={{ fontSize: '12px', color: '#999' }}>
                            {formatElapsedTime(session.elapsed_time)}
                          </span>
                        </div>

                        <div>
                          <div
                            style={{ fontSize: '12px', marginBottom: '4px' }}
                          >
                            智能体: <code>{session.agent_id.slice(-8)}</code>
                          </div>
                          {agent && (
                            <Tag size="small" color="blue">
                              {getAlgorithmName(agent.algorithm_type)}
                            </Tag>
                          )}
                        </div>

                        <Progress
                          percent={Math.round(progress)}
                          strokeColor="#52c41a"
                          format={() =>
                            `${session.current_episode}/${session.total_episodes}`
                          }
                        />

                        <Row>
                          <Col span={12}>
                            <Statistic
                              title="平均奖励"
                              value={session.average_reward}
                              precision={1}
                              valueStyle={{ fontSize: '12px' }}
                            />
                          </Col>
                          <Col span={12}>
                            {session.convergence_achieved && (
                              <div
                                style={{
                                  textAlign: 'center',
                                  paddingTop: '8px',
                                }}
                              >
                                <Tag color="green" size="small">
                                  <CheckCircleOutlined /> 已收敛
                                </Tag>
                              </div>
                            )}
                          </Col>
                        </Row>
                      </Space>
                    </Card>
                  </Col>
                )
              })}
            </Row>
          </Card>
        </Col>
      )}

      {/* 训练会话列表 */}
      <Col span={24}>
        <Card
          title={
            <Space>
              <ClockCircleOutlined />
              训练会话历史
              <Tag>{trainingSessions.length} 个会话</Tag>
            </Space>
          }
          extra={
            <Button
              type="primary"
              icon={<PlayCircleOutlined />}
              disabled={agents.length === 0}
            >
              开始新训练
            </Button>
          }
        >
          {trainingSessions.length === 0 ? (
            <Alert
              message="暂无训练会话"
              description="创建智能体后即可开始训练"
              type="info"
              showIcon
              style={{ textAlign: 'center', padding: '40px' }}
            />
          ) : (
            <Table
              columns={columns}
              dataSource={trainingSessions}
              rowKey="session_id"
              loading={loading}
              pagination={{ pageSize: 10 }}
              size="middle"
              onRow={record => ({
                onClick: () => setSelectedSession(record.session_id),
                style: {
                  backgroundColor:
                    selectedSession === record.session_id
                      ? '#f0f8ff'
                      : undefined,
                  cursor: 'pointer',
                },
              })}
            />
          )}
        </Card>
      </Col>
    </Row>
  )
}

export { QLearningTrainingPanel }
