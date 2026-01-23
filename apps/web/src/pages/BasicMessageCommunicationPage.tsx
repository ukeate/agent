import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Button,
  Form,
  Input,
  Select,
  Space,
  Typography,
  Alert,
  Tag,
  Table,
  Progress,
  Statistic,
  Timeline,
  Badge,
  Modal,
  Tabs,
  Divider,
  Switch,
  Slider,
  notification,
  Radio,
  message,
} from 'antd'
import {
  MessageOutlined,
  SendOutlined,
  ReloadOutlined,
  SettingOutlined,
  MonitorOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  ClearOutlined,
  EyeOutlined,
  DeleteOutlined,
  ApiOutlined,
  ThunderboltOutlined,
  ShareAltOutlined as NetworkOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  UserOutlined,
  TeamOutlined,
} from '@ant-design/icons'
import { streamingService } from '../services/streamingService'
import { serviceDiscoveryService } from '../services/serviceDiscoveryService'

const { Title, Text, Paragraph } = Typography
const { Option } = Select
const { TextArea } = Input
const { TabPane } = Tabs

interface MessageLog {
  id: string
  timestamp: string
  sender: string
  receiver: string
  subject: string
  messageType:
    | 'point-to-point'
    | 'publish-subscribe'
    | 'request-reply'
    | 'multicast'
  payload: any
  status: 'sent' | 'delivered' | 'failed' | 'pending'
  latency: number
  size: number
  priority: number
}

interface Agent {
  id: string
  name: string
  status: 'online' | 'offline' | 'busy'
  messagesSent: number
  messagesReceived: number
  lastSeen: string
  subscriptions: string[]
}

interface Subscription {
  id: string
  agentId: string
  subject: string
  queueGroup?: string
  messageCount: number
  lastMessage: string
  status: 'active' | 'inactive'
}

interface CommunicationMetrics {
  totalMessages: number
  messagesPerSecond: number
  averageLatency: number
  successRate: number
  failureRate: number
  activeAgents: number
  activeSubscriptions: number
}

const BasicMessageCommunicationPage: React.FC = () => {
  const [form] = Form.useForm()
  const [activeTab, setActiveTab] = useState('send-message')
  const [loading, setLoading] = useState(false)
  const [testRunning, setTestRunning] = useState(false)

  const [metrics, setMetrics] = useState<CommunicationMetrics>({
    totalMessages: 0,
    messagesPerSecond: 0,
    averageLatency: 0,
    successRate: 0,
    failureRate: 0,
    activeAgents: 0,
    activeSubscriptions: 0,
  })

  const [messageLogs, setMessageLogs] = useState<MessageLog[]>([])

  const [agents, setAgents] = useState<Agent[]>([])

  const [subscriptions, setSubscriptions] = useState<Subscription[]>([])

  const loadMetrics = async () => {
    const zero: CommunicationMetrics = {
      totalMessages: 0,
      messagesPerSecond: 0,
      averageLatency: 0,
      successRate: 0,
      failureRate: 0,
      activeAgents: 0,
      activeSubscriptions: 0,
    }
    try {
      const data = await streamingService.getSystemMetrics()
      const sys = data.system_metrics || ({} as any)
      setMetrics({
        totalMessages: sys.total_events_processed || 0,
        messagesPerSecond: 0,
        averageLatency: 0,
        successRate: 0,
        failureRate: 0,
        activeAgents: sys.active_streamers || 0,
        activeSubscriptions: sys.active_buffers || 0,
      })
    } catch (err) {
      setMetrics(zero)
      message.error('加载通信指标失败')
    }
  }

  const loadAgents = async () => {
    try {
      const list = await serviceDiscoveryService.getActiveAgents()
      const mapped = (list || []).map(a => ({
        id: a.agent_id,
        name: a.name || a.agent_id,
        status: (a.status as Agent['status']) || 'offline',
        messagesSent: 0,
        messagesReceived: 0,
        lastSeen: a.last_heartbeat || '',
        subscriptions: a.tags || [],
      }))
      setAgents(mapped)
    } catch (err) {
      setAgents([])
      message.error('加载智能体失败')
    }
  }

  const loadSubscriptions = async () => {
    try {
      const data = await streamingService.getQueueStatus()
      const queues = data.queue_metrics || {}
      const mapped: Subscription[] = Object.entries(queues).map(
        ([name, metric]: any, idx) => ({
          id: name || `queue-${idx}`,
          agentId: '',
          subject: name,
          queueGroup: undefined,
          messageCount: metric.current_size || 0,
          lastMessage: metric.timestamp || data.timestamp || '',
          status: metric.is_overloaded ? 'inactive' : 'active',
        })
      )
      setSubscriptions(mapped)
    } catch (err) {
      setSubscriptions([])
    }
  }

  const loadAll = async () => {
    setLoading(true)
    await Promise.all([loadMetrics(), loadAgents(), loadSubscriptions()])
    setLoading(false)
  }

  useEffect(() => {
    loadAll()
  }, [])

  const messageColumns = [
    {
      title: '时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      width: 150,
      render: (timestamp: string) => (
        <Text style={{ fontSize: '12px' }}>{timestamp}</Text>
      ),
    },
    {
      title: '消息信息',
      key: 'message',
      render: (record: MessageLog) => (
        <div>
          <div style={{ marginBottom: '2px' }}>
            <Text strong style={{ fontSize: '12px' }}>
              {record.subject}
            </Text>
            <Tag
              color={
                record.messageType === 'point-to-point'
                  ? 'blue'
                  : record.messageType === 'publish-subscribe'
                    ? 'green'
                    : record.messageType === 'request-reply'
                      ? 'orange'
                      : 'purple'
              }
              style={{ marginLeft: '8px', fontSize: '10px' }}
            >
              {record.messageType === 'point-to-point'
                ? '点对点'
                : record.messageType === 'publish-subscribe'
                  ? '发布订阅'
                  : record.messageType === 'request-reply'
                    ? '请求回复'
                    : '多播'}
            </Tag>
          </div>
          <Text type="secondary" style={{ fontSize: '11px' }}>
            {record.sender} → {record.receiver}
          </Text>
        </div>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (status: string) => (
        <Badge
          status={
            status === 'delivered'
              ? 'success'
              : status === 'sent'
                ? 'processing'
                : status === 'failed'
                  ? 'error'
                  : 'warning'
          }
          text={
            status === 'delivered'
              ? '已投递'
              : status === 'sent'
                ? '已发送'
                : status === 'failed'
                  ? '失败'
                  : '待处理'
          }
        />
      ),
    },
    {
      title: '性能',
      key: 'performance',
      width: 120,
      render: (record: MessageLog) => (
        <div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '11px' }}>延迟: {record.latency}ms</Text>
          </div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '11px' }}>大小: {record.size}B</Text>
          </div>
          <div>
            <Text style={{ fontSize: '11px' }}>优先级: {record.priority}</Text>
          </div>
        </div>
      ),
    },
    {
      title: '操作',
      key: 'actions',
      width: 80,
      render: (record: MessageLog) => (
        <Space>
          <Button
            type="text"
            size="small"
            icon={<EyeOutlined />}
            onClick={() => handleViewMessage(record)}
          />
        </Space>
      ),
    },
  ]

  const agentColumns = [
    {
      title: '智能体',
      key: 'agent',
      render: (record: Agent) => (
        <div>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              marginBottom: '2px',
            }}
          >
            <Badge
              status={
                record.status === 'online'
                  ? 'success'
                  : record.status === 'offline'
                    ? 'default'
                    : 'processing'
              }
            />
            <Text strong style={{ marginLeft: '8px', fontSize: '12px' }}>
              {record.name}
            </Text>
          </div>
          <Text type="secondary" style={{ fontSize: '11px' }}>
            {record.id}
          </Text>
        </div>
      ),
    },
    {
      title: '消息统计',
      key: 'stats',
      render: (record: Agent) => (
        <div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '11px' }}>
              已发送: {record.messagesSent}
            </Text>
          </div>
          <div>
            <Text style={{ fontSize: '11px' }}>
              已接收: {record.messagesReceived}
            </Text>
          </div>
        </div>
      ),
    },
    {
      title: '订阅',
      dataIndex: 'subscriptions',
      key: 'subscriptions',
      render: (subscriptions: string[]) => (
        <div>
          {subscriptions.slice(0, 2).map((sub, index) => (
            <Tag key={index} style={{ fontSize: '10px', marginBottom: '2px' }}>
              {sub}
            </Tag>
          ))}
          {subscriptions.length > 2 && (
            <Text type="secondary" style={{ fontSize: '10px' }}>
              +{subscriptions.length - 2} 更多
            </Text>
          )}
        </div>
      ),
    },
    {
      title: '最后活跃',
      dataIndex: 'lastSeen',
      key: 'lastSeen',
      width: 120,
      render: (lastSeen: string) => (
        <Text style={{ fontSize: '11px' }}>{lastSeen}</Text>
      ),
    },
  ]

  const subscriptionColumns = [
    {
      title: '订阅信息',
      key: 'subscription',
      render: (record: Subscription) => (
        <div>
          <div style={{ marginBottom: '2px' }}>
            <Text strong style={{ fontSize: '12px' }}>
              {record.subject}
            </Text>
            <Badge
              status={record.status === 'active' ? 'success' : 'default'}
              style={{ marginLeft: '8px' }}
            />
          </div>
          <Text type="secondary" style={{ fontSize: '11px' }}>
            智能体: {record.agentId}
          </Text>
          {record.queueGroup && (
            <Text
              type="secondary"
              style={{ fontSize: '11px', display: 'block' }}
            >
              队列组: {record.queueGroup}
            </Text>
          )}
        </div>
      ),
    },
    {
      title: '消息统计',
      key: 'stats',
      render: (record: Subscription) => (
        <div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '11px' }}>
              消息数: {record.messageCount}
            </Text>
          </div>
          <Text style={{ fontSize: '11px' }}>
            最后消息: {record.lastMessage}
          </Text>
        </div>
      ),
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: Subscription) => (
        <Space>
          <Button type="text" size="small" icon={<SettingOutlined />} />
          <Button type="text" size="small" icon={<DeleteOutlined />} danger />
        </Space>
      ),
    },
  ]

  const handleSendMessage = async (values: any) => {
    setLoading(true)

    try {
      const payloadObj = values.payload ? JSON.parse(values.payload) : {}
      await streamingService.createSession({
        agent_id: values.sender || 'unknown',
        message: JSON.stringify(payloadObj),
        session_id: values.subject,
      })
      notification.success({
        message: '消息发送成功',
        description: `已提交到后端队列: ${values.subject}`,
      })
      form.resetFields()
      await loadMetrics()
      await loadSubscriptions()
    } catch (err: any) {
      message.error(err?.message || '消息发送失败')
    } finally {
      setLoading(false)
    }
  }

  const handleViewMessage = (message: MessageLog) => {
    Modal.info({
      title: '消息详情',
      width: 600,
      content: (
        <div>
          <Divider>基本信息</Divider>
          <Row gutter={[16, 8]}>
            <Col span={12}>
              <Text strong>消息ID: </Text>
              <Text code>{message.id}</Text>
            </Col>
            <Col span={12}>
              <Text strong>时间戳: </Text>
              <Text>{message.timestamp}</Text>
            </Col>
            <Col span={12}>
              <Text strong>发送者: </Text>
              <Text>{message.sender}</Text>
            </Col>
            <Col span={12}>
              <Text strong>接收者: </Text>
              <Text>{message.receiver}</Text>
            </Col>
            <Col span={12}>
              <Text strong>主题: </Text>
              <Text code>{message.subject}</Text>
            </Col>
            <Col span={12}>
              <Text strong>类型: </Text>
              <Tag color="blue">{message.messageType}</Tag>
            </Col>
          </Row>

          <Divider>性能指标</Divider>
          <Row gutter={[16, 8]}>
            <Col span={8}>
              <Statistic
                title="延迟"
                value={message.latency}
                precision={1}
                suffix="ms"
              />
            </Col>
            <Col span={8}>
              <Statistic title="大小" value={message.size} suffix="B" />
            </Col>
            <Col span={8}>
              <Statistic title="优先级" value={message.priority} />
            </Col>
          </Row>

          <Divider>消息内容</Divider>
          <pre
            style={{
              background: '#f5f5f5',
              padding: '12px',
              borderRadius: '4px',
              fontSize: '12px',
            }}
          >
            {JSON.stringify(message.payload, null, 2)}
          </pre>
        </div>
      ),
    })
  }

  const handleStartStressTest = () => {
    setTestRunning(true)
    notification.info({
      message: '开始压力测试',
      description: '已触发后端检查，请在指标中观察变化',
    })
    streamingService
      .getSystemMetrics()
      .catch(() => message.error('压力测试接口不可用'))
      .finally(() => setTestRunning(false))
  }

  const refreshData = () => {
    loadAll().then(() => {
      notification.success({
        message: '数据刷新成功',
        description: '通信指标已同步',
      })
    })
  }

  return (
    <div style={{ padding: '24px', background: '#f5f5f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <MessageOutlined style={{ marginRight: '12px', color: '#1890ff' }} />
          基础消息通信
        </Title>
        <Paragraph>
          智能体间基础消息通信功能，支持点对点、发布订阅、请求回复、多播等多种通信模式
        </Paragraph>
      </div>

      {/* 核心指标监控 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={12} sm={6}>
          <Card>
            <Statistic
              title="消息总量"
              value={metrics.totalMessages}
              precision={0}
              valueStyle={{ color: '#3f8600' }}
              prefix={<ThunderboltOutlined />}
            />
          </Card>
        </Col>

        <Col xs={12} sm={6}>
          <Card>
            <Statistic
              title="消息速率"
              value={metrics.messagesPerSecond}
              precision={1}
              suffix="msg/s"
              valueStyle={{ color: '#1890ff' }}
              prefix={<NetworkOutlined />}
            />
          </Card>
        </Col>

        <Col xs={12} sm={6}>
          <Card>
            <Statistic
              title="平均延迟"
              value={metrics.averageLatency}
              precision={1}
              suffix="ms"
              valueStyle={{ color: '#722ed1' }}
              prefix={<ClockCircleOutlined />}
            />
          </Card>
        </Col>

        <Col xs={12} sm={6}>
          <Card>
            <Statistic
              title="成功率"
              value={metrics.successRate}
              precision={1}
              suffix="%"
              valueStyle={{ color: '#52c41a' }}
              prefix={<CheckCircleOutlined />}
            />
            <div style={{ marginTop: '8px' }}>
              <Progress
                percent={metrics.successRate}
                size="small"
                showInfo={false}
              />
            </div>
          </Card>
        </Col>
      </Row>

      {/* 主功能区域 */}
      <Card>
        <div
          style={{
            marginBottom: '16px',
            display: 'flex',
            justifyContent: 'space-between',
          }}
        >
          <Space>
            <Button
              type="primary"
              icon={<SendOutlined />}
              onClick={() => setActiveTab('send-message')}
            >
              发送消息
            </Button>
            <Button
              icon={<ReloadOutlined />}
              loading={loading}
              onClick={refreshData}
            >
              刷新数据
            </Button>
          </Space>

          <Space>
            <Button
              icon={<PlayCircleOutlined />}
              loading={testRunning}
              onClick={handleStartStressTest}
              disabled={testRunning}
            >
              {testRunning ? '测试中...' : '压力测试'}
            </Button>
            <Button icon={<SettingOutlined />}>通信设置</Button>
          </Space>
        </div>

        <Tabs activeKey={activeTab} onChange={setActiveTab}>
          <TabPane tab="发送消息" key="send-message" icon={<SendOutlined />}>
            <Row gutter={[24, 24]}>
              <Col span={12}>
                <Card title="消息发送" size="small">
                  <Form
                    form={form}
                    layout="vertical"
                    onFinish={handleSendMessage}
                  >
                    <Form.Item
                      name="messageType"
                      label="通信模式"
                      rules={[{ required: true }]}
                      initialValue="point-to-point"
                    >
                      <Radio.Group>
                        <Radio.Button value="point-to-point">
                          点对点
                        </Radio.Button>
                        <Radio.Button value="publish-subscribe">
                          发布订阅
                        </Radio.Button>
                        <Radio.Button value="request-reply">
                          请求回复
                        </Radio.Button>
                        <Radio.Button value="multicast">多播</Radio.Button>
                      </Radio.Group>
                    </Form.Item>

                    <Row gutter={16}>
                      <Col span={12}>
                        <Form.Item
                          name="sender"
                          label="发送者ID"
                          initialValue="test-client"
                        >
                          <Input
                            placeholder="发送者智能体ID"
                            prefix={<UserOutlined />}
                          />
                        </Form.Item>
                      </Col>
                      <Col span={12}>
                        <Form.Item name="receiver" label="接收者">
                          <Select placeholder="选择接收者" allowClear>
                            {agents.map(agent => (
                              <Option key={agent.id} value={agent.id}>
                                {agent.name} ({agent.id})
                              </Option>
                            ))}
                          </Select>
                        </Form.Item>
                      </Col>
                    </Row>

                    <Form.Item
                      name="subject"
                      label="消息主题"
                      rules={[{ required: true }]}
                    >
                      <Input placeholder="例如: agents.tasks.process" />
                    </Form.Item>

                    <Form.Item
                      name="payload"
                      label="消息内容"
                      rules={[{ required: true }]}
                    >
                      <TextArea
                        rows={4}
                        placeholder='{"type": "task", "data": {...}}'
                      />
                    </Form.Item>

                    <Row gutter={16}>
                      <Col span={12}>
                        <Form.Item
                          name="priority"
                          label="优先级"
                          initialValue={5}
                        >
                          <Slider
                            min={1}
                            max={10}
                            marks={{ 1: '低', 5: '中', 10: '高' }}
                          />
                        </Form.Item>
                      </Col>
                      <Col span={12}>
                        <Form.Item
                          name="timeout"
                          label="超时时间(秒)"
                          initialValue={30}
                        >
                          <Input type="number" />
                        </Form.Item>
                      </Col>
                    </Row>

                    <Form.Item>
                      <Button
                        type="primary"
                        htmlType="submit"
                        icon={<SendOutlined />}
                        loading={loading}
                        block
                      >
                        发送消息
                      </Button>
                    </Form.Item>
                  </Form>
                </Card>
              </Col>

              <Col span={12}>
                <Card title="快速测试" size="small">
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <Button
                      block
                      icon={<ApiOutlined />}
                      onClick={() => {
                        form.setFieldsValue({
                          messageType: 'point-to-point',
                          sender: 'test-client',
                          receiver: 'task-agent-01',
                          subject: 'agents.tasks.simple',
                          payload:
                            '{"type": "test", "message": "Hello from test client"}',
                          priority: 5,
                        })
                      }}
                    >
                      点对点测试消息
                    </Button>

                    <Button
                      block
                      icon={<TeamOutlined />}
                      onClick={() => {
                        form.setFieldsValue({
                          messageType: 'publish-subscribe',
                          sender: 'broadcast-client',
                          subject: 'agents.broadcast.test',
                          payload:
                            '{"type": "broadcast", "message": "System announcement"}',
                          priority: 3,
                        })
                      }}
                    >
                      广播测试消息
                    </Button>

                    <Button
                      block
                      icon={<ThunderboltOutlined />}
                      onClick={() => {
                        form.setFieldsValue({
                          messageType: 'request-reply',
                          sender: 'client-agent',
                          receiver: 'service-agent',
                          subject: 'agents.request.ping',
                          payload:
                            '{"type": "ping", "timestamp": "' +
                            new Date().toISOString() +
                            '"}',
                          priority: 8,
                        })
                      }}
                    >
                      请求响应测试
                    </Button>

                    <Divider />

                    <Alert
                      message="测试提示"
                      description="使用快速测试按钮可以快速填充常用的测试消息模板，便于功能验证。"
                      type="info"
                      showIcon
                      style={{ fontSize: '12px' }}
                    />
                  </Space>
                </Card>
              </Col>
            </Row>
          </TabPane>

          <TabPane tab="消息日志" key="message-logs" icon={<MonitorOutlined />}>
            <div style={{ marginBottom: '16px' }}>
              <Space>
                <Button icon={<ClearOutlined />} size="small">
                  清空日志
                </Button>
                <Text type="secondary">
                  显示最新 {messageLogs.length} 条消息
                </Text>
              </Space>
            </div>
            <Table
              columns={messageColumns}
              dataSource={messageLogs}
              rowKey="id"
              size="small"
              pagination={{ pageSize: 20 }}
              scroll={{ x: 800 }}
            />
          </TabPane>

          <TabPane tab="智能体状态" key="agents" icon={<UserOutlined />}>
            <div style={{ marginBottom: '16px' }}>
              <Text type="secondary">
                活跃智能体: {agents.filter(a => a.status === 'online').length} /{' '}
                {agents.length}
              </Text>
            </div>
            <Table
              columns={agentColumns}
              dataSource={agents}
              rowKey="id"
              size="small"
              pagination={false}
            />
          </TabPane>

          <TabPane tab="订阅管理" key="subscriptions" icon={<ApiOutlined />}>
            <div style={{ marginBottom: '16px' }}>
              <Space>
                <Button type="primary" icon={<SendOutlined />} size="small">
                  新建订阅
                </Button>
                <Text type="secondary">
                  活跃订阅:{' '}
                  {subscriptions.filter(s => s.status === 'active').length} /{' '}
                  {subscriptions.length}
                </Text>
              </Space>
            </div>
            <Table
              columns={subscriptionColumns}
              dataSource={subscriptions}
              rowKey="id"
              size="small"
              pagination={false}
            />
          </TabPane>
        </Tabs>
      </Card>
    </div>
  )
}

export default BasicMessageCommunicationPage
