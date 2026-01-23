import React, { useState, useEffect } from 'react'
import {
  Card,
  Button,
  Input,
  List,
  Tag,
  Alert,
  Tabs,
  Space,
  Progress,
  Timeline,
  Divider,
  Badge,
  Row,
  Col,
  Statistic,
  Select,
  Modal,
  Form,
  message,
} from 'antd'
import {
  SendOutlined,
  TeamOutlined,
  RobotOutlined,
  MessageOutlined,
  HistoryOutlined,
  SettingOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  StopOutlined,
  ReloadOutlined,
} from '@ant-design/icons'
import apiClient from '../services/apiClient'

import { logger } from '../utils/logger'
const { TextArea } = Input
const { TabPane } = Tabs

interface Agent {
  id: string
  name: string
  role: string
  status: 'active' | 'thinking' | 'idle' | 'error'
  avatar?: string
}

interface Message {
  id: string
  role: string
  content: string
  timestamp: string
  agent?: string
}

interface Conversation {
  conversation_id: string
  status: string
  created_at: string
  round_count: number
  message_count: number
  participants: Agent[]
  messages: Message[]
}

const MultiAgentEnhancedPage: React.FC = () => {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [activeConversation, setActiveConversation] =
    useState<Conversation | null>(null)
  const [inputMessage, setInputMessage] = useState('')
  const [loading, setLoading] = useState(false)
  const [selectedAgents, setSelectedAgents] = useState<string[]>([
    'assistant',
    'critic',
  ])
  const [maxRounds, setMaxRounds] = useState(10)
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [form] = Form.useForm()

  // 可用的智能体角色
  const availableAgents = [
    { value: 'assistant', label: '助手智能体', color: 'blue' },
    { value: 'critic', label: '评论智能体', color: 'green' },
    { value: 'coder', label: '编程智能体', color: 'purple' },
    { value: 'planner', label: '规划智能体', color: 'orange' },
    { value: 'executor', label: '执行智能体', color: 'red' },
  ]

  // 创建新对话
  const createConversation = async () => {
    try {
      setLoading(true)
      const response = await apiClient.post('/multi-agent/conversation', {
        message: inputMessage,
        agent_roles: selectedAgents,
        max_rounds: maxRounds,
      })

      const newConversation = response.data
      setConversations(prev => [newConversation, ...prev])
      setActiveConversation(newConversation)
      setInputMessage('')
      setShowCreateModal(false)
      message.success('对话创建成功')

      // 开始轮询状态
      pollConversationStatus(newConversation.conversation_id)
    } catch (error: any) {
      message.error('创建对话失败: ' + error.message)
    } finally {
      setLoading(false)
    }
  }

  // 轮询对话状态
  const pollConversationStatus = async (conversationId: string) => {
    const interval = setInterval(async () => {
      try {
        const response = await apiClient.get(
          `/multi-agent/conversation/${conversationId}/status`
        )
        const updatedConversation = response.data

        setConversations(prev =>
          prev.map(conv =>
            conv.conversation_id === conversationId ? updatedConversation : conv
          )
        )

        if (activeConversation?.conversation_id === conversationId) {
          setActiveConversation(updatedConversation)
        }

        // 获取消息
        const messagesResponse = await apiClient.get(
          `/multi-agent/conversation/${conversationId}/messages`
        )
        if (messagesResponse.data.messages) {
          setActiveConversation(prev =>
            prev ? { ...prev, messages: messagesResponse.data.messages } : null
          )
        }

        // 如果对话已完成，停止轮询
        if (
          updatedConversation.status === 'completed' ||
          updatedConversation.status === 'error'
        ) {
          clearInterval(interval)
        }
      } catch (error) {
        logger.error('轮询失败:', error)
      }
    }, 2000)

    // 30秒后自动停止轮询
    setTimeout(() => clearInterval(interval), 30000)
  }

  // 控制对话
  const controlConversation = async (
    action: 'pause' | 'resume' | 'terminate'
  ) => {
    if (!activeConversation) return

    try {
      await apiClient.post(
        `/multi-agent/conversation/${activeConversation.conversation_id}/${action}`
      )
      message.success(
        `对话已${action === 'pause' ? '暂停' : action === 'resume' ? '恢复' : '终止'}`
      )

      // 更新状态
      setActiveConversation(prev =>
        prev
          ? {
              ...prev,
              status: action === 'terminate' ? 'terminated' : prev.status,
            }
          : null
      )
    } catch (error: any) {
      message.error('操作失败: ' + error.message)
    }
  }

  // 渲染消息
  const renderMessage = (msg: Message) => {
    const agentInfo = availableAgents.find(a => a.value === msg.role)
    const isUserMessage = msg.role === 'user'

    return (
      <div
        style={{
          padding: '12px',
          borderRadius: '8px',
          marginBottom: '8px',
          backgroundColor: isUserMessage ? '#e6f7ff' : '#f6ffed',
          marginLeft: isUserMessage ? '20%' : '0',
          marginRight: isUserMessage ? '0' : '20%',
        }}
      >
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            marginBottom: '8px',
          }}
        >
          <Tag color={agentInfo?.color || 'default'}>
            {msg.agent || msg.role}
          </Tag>
          <span style={{ color: '#999', fontSize: '12px' }}>
            {new Date(msg.timestamp).toLocaleTimeString()}
          </span>
        </div>
        <div style={{ color: '#333' }}>{msg.content}</div>
      </div>
    )
  }

  return (
    <div style={{ padding: '24px' }}>
      <Card
        title={
          <Space>
            <TeamOutlined />
            <span>多智能体协作系统 - 增强版</span>
          </Space>
        }
        extra={
          <Space>
            <Button
              type="primary"
              icon={<PlayCircleOutlined />}
              onClick={() => setShowCreateModal(true)}
            >
              创建新对话
            </Button>
            <Button
              icon={<ReloadOutlined />}
              onClick={() => window.location.reload()}
            >
              刷新
            </Button>
          </Space>
        }
      >
        <Row gutter={16}>
          <Col span={8}>
            <Card title="对话列表" size="small">
              <List
                dataSource={conversations}
                renderItem={conv => (
                  <List.Item
                    key={conv.conversation_id}
                    onClick={() => setActiveConversation(conv)}
                    style={{
                      cursor: 'pointer',
                      background:
                        activeConversation?.conversation_id ===
                        conv.conversation_id
                          ? '#f0f0f0'
                          : 'transparent',
                    }}
                  >
                    <List.Item.Meta
                      avatar={
                        <Badge
                          status={
                            conv.status === 'active' ? 'processing' : 'default'
                          }
                        />
                      }
                      title={`对话 ${conv.conversation_id.slice(0, 8)}`}
                      description={
                        <Space direction="vertical" size={0}>
                          <span>
                            轮数: {conv.round_count}/{maxRounds}
                          </span>
                          <span>消息: {conv.message_count}</span>
                        </Space>
                      }
                    />
                  </List.Item>
                )}
              />
            </Card>
          </Col>

          <Col span={16}>
            {activeConversation ? (
              <Card
                title={`对话详情 - ${activeConversation.conversation_id.slice(0, 8)}`}
                extra={
                  <Space>
                    <Button
                      icon={<PauseCircleOutlined />}
                      onClick={() => controlConversation('pause')}
                      disabled={activeConversation.status !== 'active'}
                    >
                      暂停
                    </Button>
                    <Button
                      icon={<PlayCircleOutlined />}
                      onClick={() => controlConversation('resume')}
                      disabled={activeConversation.status !== 'paused'}
                    >
                      恢复
                    </Button>
                    <Button
                      danger
                      icon={<StopOutlined />}
                      onClick={() => controlConversation('terminate')}
                      disabled={activeConversation.status === 'terminated'}
                    >
                      终止
                    </Button>
                  </Space>
                }
              >
                <Tabs defaultActiveKey="messages">
                  <TabPane tab="消息流" key="messages">
                    <div style={{ maxHeight: '400px', overflow: 'auto' }}>
                      {activeConversation.messages?.map(msg => (
                        <div key={msg.id} style={{ marginBottom: '16px' }}>
                          {renderMessage(msg)}
                        </div>
                      ))}
                    </div>
                  </TabPane>

                  <TabPane tab="参与者" key="participants">
                    <List
                      dataSource={activeConversation.participants}
                      renderItem={agent => (
                        <List.Item>
                          <List.Item.Meta
                            avatar={<RobotOutlined />}
                            title={agent.name}
                            description={
                              <Space>
                                <Tag
                                  color={
                                    agent.status === 'active'
                                      ? 'green'
                                      : 'default'
                                  }
                                >
                                  {agent.status}
                                </Tag>
                                <span>角色: {agent.role}</span>
                              </Space>
                            }
                          />
                        </List.Item>
                      )}
                    />
                  </TabPane>

                  <TabPane tab="统计" key="stats">
                    <Row gutter={16}>
                      <Col span={8}>
                        <Statistic
                          title="总轮数"
                          value={activeConversation.round_count}
                        />
                      </Col>
                      <Col span={8}>
                        <Statistic
                          title="消息数"
                          value={activeConversation.message_count}
                        />
                      </Col>
                      <Col span={8}>
                        <Statistic
                          title="参与者"
                          value={activeConversation.participants.length}
                        />
                      </Col>
                    </Row>
                    <Divider />
                    <Progress
                      percent={
                        (activeConversation.round_count / maxRounds) * 100
                      }
                      status={
                        activeConversation.status === 'active'
                          ? 'active'
                          : 'normal'
                      }
                    />
                  </TabPane>
                </Tabs>
              </Card>
            ) : (
              <Card>
                <Alert
                  message="请选择或创建一个对话"
                  description="点击左侧对话列表中的项目，或点击「创建新对话」按钮开始"
                  type="info"
                  showIcon
                />
              </Card>
            )}
          </Col>
        </Row>
      </Card>

      <Modal
        title="创建新的多智能体对话"
        visible={showCreateModal}
        onOk={createConversation}
        onCancel={() => setShowCreateModal(false)}
        confirmLoading={loading}
        width={600}
      >
        <Form form={form} layout="vertical">
          <Form.Item label="初始消息" required>
            <TextArea
              rows={4}
              value={inputMessage}
              onChange={e => setInputMessage(e.target.value)}
              placeholder="输入要讨论的问题或任务..."
            />
          </Form.Item>

          <Form.Item label="选择智能体">
            <Select
              mode="multiple"
              value={selectedAgents}
              onChange={setSelectedAgents}
              options={availableAgents}
              placeholder="选择参与的智能体"
            />
          </Form.Item>

          <Form.Item label="最大轮数">
            <Input
              type="number"
              value={maxRounds}
              onChange={e => setMaxRounds(parseInt(e.target.value) || 10)}
              min={1}
              max={50}
            />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}

export default MultiAgentEnhancedPage
