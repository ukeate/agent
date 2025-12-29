import React, { useState } from 'react'
import { 
import { logger } from '../../utils/logger'
  Card, 
  Row, 
  Col, 
  Button, 
  Table, 
  Tag, 
  Progress, 
  Space, 
  Modal, 
  Form, 
  Select, 
  Input, 
  Statistic, 
  Tooltip,
  Alert,
  Descriptions
} from 'antd'
import {
  RobotOutlined,
  PlayCircleOutlined,
  EyeOutlined,
  DeleteOutlined,
  PlusOutlined,
  ThunderboltOutlined,
  LineChartOutlined,
  SettingOutlined
} from '@ant-design/icons'

const { Option } = Select

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

interface QLearningAgentPanelProps {
  agents: AgentStats[]
  loading: boolean
  onCreateAgent: (algorithmType: string) => void
  onStartTraining: (agentId: string, episodes: number) => void
  onSelectAgent: (agentId: string | null) => void
  selectedAgent: string | null
}

const QLearningAgentPanel: React.FC<QLearningAgentPanelProps> = ({
  agents,
  loading,
  onCreateAgent,
  onStartTraining,
  onSelectAgent,
  selectedAgent
}) => {
  const [createModalVisible, setCreateModalVisible] = useState(false)
  const [detailModalVisible, setDetailModalVisible] = useState(false)
  const [trainModalVisible, setTrainModalVisible] = useState(false)
  const [selectedAgentDetail, setSelectedAgentDetail] = useState<AgentStats | null>(null)
  const [form] = Form.useForm()
  const [trainForm] = Form.useForm()

  const getAlgorithmColor = (algorithm: string) => {
    switch (algorithm) {
      case 'q_learning': return 'blue'
      case 'dqn': return 'green'
      case 'double_dqn': return 'orange'
      case 'dueling_dqn': return 'purple'
      default: return 'default'
    }
  }

  const getAlgorithmName = (algorithm: string) => {
    switch (algorithm) {
      case 'q_learning': return 'Classic Q-Learning'
      case 'dqn': return 'Deep Q-Network'
      case 'double_dqn': return 'Double DQN'  
      case 'dueling_dqn': return 'Dueling DQN'
      default: return algorithm
    }
  }

  const getExplorationStatus = (epsilon: number) => {
    if (epsilon > 0.5) return { status: 'active', color: 'orange', text: '高探索' }
    if (epsilon > 0.1) return { status: 'normal', color: 'blue', text: '平衡探索' }
    return { status: 'low', color: 'green', text: '低探索' }
  }

  const handleCreateAgent = async (values: any) => {
    try {
      await onCreateAgent(values.algorithmType)
      setCreateModalVisible(false)
      form.resetFields()
    } catch (error) {
      logger.error('创建智能体失败:', error)
    }
  }

  const handleStartTraining = async (values: any) => {
    try {
      if (selectedAgentDetail) {
        await onStartTraining(selectedAgentDetail.agent_id, values.episodes)
        setTrainModalVisible(false)
        trainForm.resetFields()
      }
    } catch (error) {
      logger.error('启动训练失败:', error)
    }
  }

  const showAgentDetail = (agent: AgentStats) => {
    setSelectedAgentDetail(agent)
    setDetailModalVisible(true)
  }

  const showTrainingModal = (agent: AgentStats) => {
    setSelectedAgentDetail(agent)
    setTrainModalVisible(true)
  }

  const columns = [
    {
      title: '智能体ID',
      dataIndex: 'agent_id',
      key: 'agent_id',
      width: 160,
      render: (text: string) => (
        <span style={{ fontFamily: 'monospace', fontSize: '12px' }}>
          {text}
        </span>
      )
    },
    {
      title: '算法类型',
      dataIndex: 'algorithm_type',
      key: 'algorithm_type',
      width: 140,
      render: (algorithm: string) => (
        <Tag color={getAlgorithmColor(algorithm)}>
          {getAlgorithmName(algorithm)}
        </Tag>
      )
    },
    {
      title: '训练进度',
      key: 'training_progress',
      width: 120,
      render: (_, record: AgentStats) => (
        <div>
          <div style={{ fontSize: '12px', marginBottom: '4px' }}>
            {record.episode_count} episodes
          </div>
          <div style={{ fontSize: '12px', color: '#999' }}>
            {record.step_count.toLocaleString()} steps
          </div>
        </div>
      )
    },
    {
      title: '探索率',
      dataIndex: 'current_epsilon',
      key: 'current_epsilon',
      width: 100,
      render: (epsilon: number) => {
        const status = getExplorationStatus(epsilon)
        return (
          <Tooltip title={`探索率: ${(epsilon * 100).toFixed(1)}%`}>
            <Progress
              percent={epsilon * 100}
              size="small"
              strokeColor={status.color}
              format={() => status.text}
            />
          </Tooltip>
        )
      }
    },
    {
      title: '性能指标',
      key: 'performance',
      width: 130,
      render: (_, record: AgentStats) => (
        <div>
          <Statistic
            title="平均奖励"
            value={record.average_reward}
            precision={1}
            valueStyle={{ fontSize: '14px' }}
          />
          <div style={{ fontSize: '10px', color: '#999', marginTop: '2px' }}>
            总奖励: {record.total_reward.toFixed(1)}
          </div>
        </div>
      )
    },
    {
      title: '操作',
      key: 'actions',
      width: 160,
      render: (_, record: AgentStats) => (
        <Space>
          <Tooltip title="查看详情">
            <Button 
              type="text" 
              icon={<EyeOutlined />} 
              size="small"
              onClick={() => showAgentDetail(record)}
            />
          </Tooltip>
          <Tooltip title="开始训练">
            <Button 
              type="text" 
              icon={<PlayCircleOutlined />} 
              size="small"
              onClick={() => showTrainingModal(record)}
            />
          </Tooltip>
          <Tooltip title="配置">
            <Button 
              type="text" 
              icon={<SettingOutlined />} 
              size="small"
            />
          </Tooltip>
          <Tooltip title="删除">
            <Button 
              type="text" 
              danger 
              icon={<DeleteOutlined />} 
              size="small"
            />
          </Tooltip>
        </Space>
      )
    }
  ]

  return (
    <>
      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Space style={{ marginBottom: 16, width: '100%', justifyContent: 'space-between' }}>
            <Space>
              <RobotOutlined style={{ fontSize: '16px', color: '#1890ff' }} />
              <span style={{ fontWeight: 'bold' }}>智能体管理</span>
              <Tag>{agents.length} 个智能体</Tag>
            </Space>
            
            <Button 
              type="primary" 
              icon={<PlusOutlined />}
              onClick={() => setCreateModalVisible(true)}
            >
              创建新智能体
            </Button>
          </Space>
        </Col>

        <Col span={24}>
          {agents.length === 0 ? (
            <Alert
              message="暂无智能体"
              description="点击'创建新智能体'开始您的强化学习之旅"
              variant="default"
              showIcon
              style={{ textAlign: 'center', padding: '40px' }}
            />
          ) : (
            <Table
              columns={columns}
              dataSource={agents}
              rowKey="agent_id"
              loading={loading}
              pagination={false}
              size="middle"
              onRow={(record) => ({
                onClick: () => onSelectAgent(record.agent_id),
                style: {
                  backgroundColor: selectedAgent === record.agent_id ? '#f0f8ff' : undefined,
                  cursor: 'pointer'
                }
              })}
            />
          )}
        </Col>
      </Row>

      {/* 创建智能体模态框 */}
      <Modal
        title={
          <Space>
            <RobotOutlined />
            创建新的Q-Learning智能体
          </Space>
        }
        open={createModalVisible}
        onCancel={() => {
          setCreateModalVisible(false)
          form.resetFields()
        }}
        onOk={() => form.submit()}
        confirmLoading={loading}
      >
        <Form form={form} onFinish={handleCreateAgent} layout="vertical">
          <Form.Item
            name="algorithmType"
            label="算法类型"
            rules={[{ required: true, message: '请选择算法类型' }]}
          >
            <Select placeholder="选择Q-Learning算法" size="large">
              <Option value="q_learning">
                <Space>
                  <Tag color="blue">Classic Q-Learning</Tag>
                  <span>表格型算法，适用于离散小状态空间</span>
                </Space>
              </Option>
              <Option value="dqn">
                <Space>
                  <Tag color="green">Deep Q-Network</Tag>
                  <span>神经网络逼近，处理高维状态</span>
                </Space>
              </Option>
              <Option value="double_dqn">
                <Space>
                  <Tag color="orange">Double DQN</Tag>
                  <span>减少Q值高估偏差</span>
                </Space>
              </Option>
              <Option value="dueling_dqn">
                <Space>
                  <Tag color="purple">Dueling DQN</Tag>
                  <span>分离状态价值和优势函数</span>
                </Space>
              </Option>
            </Select>
          </Form.Item>
          
          <Alert
            message="算法选择建议"
            description="初学者推荐从Classic Q-Learning开始，熟悉后可尝试DQN处理更复杂的环境"
            variant="default"
            showIcon
            style={{ marginTop: 16 }}
          />
        </Form>
      </Modal>

      {/* 智能体详情模态框 */}
      <Modal
        title={
          <Space>
            <EyeOutlined />
            智能体详情
          </Space>
        }
        open={detailModalVisible}
        onCancel={() => setDetailModalVisible(false)}
        footer={null}
        width={600}
      >
        {selectedAgentDetail && (
          <Descriptions column={2} bordered>
            <Descriptions.Item label="智能体ID" span={2}>
              <code>{selectedAgentDetail.agent_id}</code>
            </Descriptions.Item>
            <Descriptions.Item label="算法类型">
              <Tag color={getAlgorithmColor(selectedAgentDetail.algorithm_type)}>
                {getAlgorithmName(selectedAgentDetail.algorithm_type)}
              </Tag>
            </Descriptions.Item>
            <Descriptions.Item label="当前探索率">
              {(selectedAgentDetail.current_epsilon * 100).toFixed(1)}%
            </Descriptions.Item>
            <Descriptions.Item label="训练轮数">
              {selectedAgentDetail.episode_count}
            </Descriptions.Item>
            <Descriptions.Item label="总步数">
              {selectedAgentDetail.step_count.toLocaleString()}
            </Descriptions.Item>
            <Descriptions.Item label="总奖励">
              {selectedAgentDetail.total_reward.toFixed(2)}
            </Descriptions.Item>
            <Descriptions.Item label="平均奖励">
              {selectedAgentDetail.average_reward.toFixed(2)}
            </Descriptions.Item>
            <Descriptions.Item label="最近奖励" span={2}>
              <Space wrap>
                {selectedAgentDetail.recent_rewards.slice(-5).map((reward, index) => (
                  <Tag key={index} color={reward > selectedAgentDetail.average_reward ? 'green' : 'orange'}>
                    {reward.toFixed(1)}
                  </Tag>
                ))}
              </Space>
            </Descriptions.Item>
          </Descriptions>
        )}
      </Modal>

      {/* 训练配置模态框 */}
      <Modal
        title={
          <Space>
            <PlayCircleOutlined />
            配置训练会话
          </Space>
        }
        open={trainModalVisible}
        onCancel={() => {
          setTrainModalVisible(false)
          trainForm.resetFields()
        }}
        onOk={() => trainForm.submit()}
        confirmLoading={loading}
      >
        <Form form={trainForm} onFinish={handleStartTraining} layout="vertical">
          <Form.Item
            name="episodes"
            label="训练轮数"
            rules={[{ required: true, message: '请输入训练轮数' }]}
            initialValue={100}
          >
            <Input type="number" placeholder="输入训练轮数" min={1} max={10000} />
          </Form.Item>
          
          <Alert
            message={`即将开始训练智能体: ${selectedAgentDetail?.agent_id}`}
            description="训练过程将在后台进行，您可以在训练监控页面查看实时进度"
            variant="default"
            showIcon
          />
        </Form>
      </Modal>
    </>
  )
}

export { QLearningAgentPanel }
