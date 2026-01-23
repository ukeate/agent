import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Button,
  Table,
  Switch,
  Modal,
  Form,
  Input,
  Select,
  InputNumber,
  Tag,
  Space,
  Alert,
  Tabs,
} from 'antd'
import {
  BellOutlined,
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  SettingOutlined,
  ThunderboltOutlined,
  CheckCircleOutlined,
  ExclamationTriangleOutlined,
  StopOutlined,
} from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'

const { Option } = Select
const { TextArea } = Input
const { TabPane } = Tabs

interface AlertRule {
  id: string
  name: string
  description: string
  metric: string
  condition: 'greater_than' | 'less_than' | 'equals' | 'not_equals'
  threshold: number
  duration: number // 持续时间（秒）
  severity: 'low' | 'medium' | 'high' | 'critical'
  enabled: boolean
  channels: string[]
  lastTriggered?: string
  status: 'normal' | 'firing' | 'resolved'
}

interface NotificationChannel {
  id: string
  name: string
  type: 'email' | 'slack' | 'webhook' | 'sms'
  config: Record<string, any>
  enabled: boolean
}

interface AlertHistory {
  id: string
  ruleName: string
  severity: string
  message: string
  timestamp: string
  status: 'firing' | 'resolved'
  duration: number
}

const RLAlertConfigPage: React.FC = () => {
  const [alertRules, setAlertRules] = useState<AlertRule[]>([])
  const [channels, setChannels] = useState<NotificationChannel[]>([])
  const [alertHistory, setAlertHistory] = useState<AlertHistory[]>([])
  const [showRuleModal, setShowRuleModal] = useState(false)
  const [showChannelModal, setShowChannelModal] = useState(false)
  const [editingRule, setEditingRule] = useState<AlertRule | null>(null)
  const [editingChannel, setEditingChannel] =
    useState<NotificationChannel | null>(null)
  const [form] = Form.useForm()
  const [channelForm] = Form.useForm()

  // 初始化数据
  useEffect(() => {
    const rules: AlertRule[] = [
      {
        id: '1',
        name: '推荐延迟过高',
        description: '当推荐系统响应时间超过100ms时触发告警',
        metric: 'rl_recommendation_latency_p95',
        condition: 'greater_than',
        threshold: 100,
        duration: 300,
        severity: 'high',
        enabled: true,
        channels: ['email-dev', 'slack-alerts'],
        lastTriggered: '2025-08-22 13:45:30',
        status: 'normal',
      },
      {
        id: '2',
        name: 'QPS过低',
        description: '当系统QPS低于800时触发告警',
        metric: 'rl_requests_per_second',
        condition: 'less_than',
        threshold: 800,
        duration: 600,
        severity: 'medium',
        enabled: true,
        channels: ['email-dev'],
        status: 'normal',
      },
      {
        id: '3',
        name: '缓存命中率低',
        description: '当缓存命中率低于90%时触发告警',
        metric: 'rl_cache_hit_rate',
        condition: 'less_than',
        threshold: 90,
        duration: 300,
        severity: 'medium',
        enabled: true,
        channels: ['slack-alerts'],
        status: 'normal',
      },
      {
        id: '4',
        name: '错误率过高',
        description: '当系统错误率超过1%时触发告警',
        metric: 'rl_error_rate',
        condition: 'greater_than',
        threshold: 1,
        duration: 120,
        severity: 'critical',
        enabled: true,
        channels: ['email-dev', 'slack-alerts', 'webhook-pagerduty'],
        lastTriggered: '2025-08-22 11:20:15',
        status: 'firing',
      },
      {
        id: '5',
        name: '模型准确率下降',
        description: '当推荐模型准确率低于85%时触发告警',
        metric: 'rl_model_accuracy',
        condition: 'less_than',
        threshold: 85,
        duration: 900,
        severity: 'high',
        enabled: false,
        channels: ['email-ml-team'],
        status: 'normal',
      },
    ]

    const notificationChannels: NotificationChannel[] = [
      {
        id: 'email-dev',
        name: '开发团队邮件',
        type: 'email',
        config: { recipients: ['dev-team@company.com'] },
        enabled: true,
      },
      {
        id: 'email-ml-team',
        name: 'ML团队邮件',
        type: 'email',
        config: { recipients: ['ml-team@company.com'] },
        enabled: true,
      },
      {
        id: 'slack-alerts',
        name: 'Slack告警频道',
        type: 'slack',
        config: {
          webhook: 'https://hooks.slack.com/...',
          channel: '#rl-alerts',
        },
        enabled: true,
      },
      {
        id: 'webhook-pagerduty',
        name: 'PagerDuty集成',
        type: 'webhook',
        config: { url: 'https://events.pagerduty.com/...', service_key: 'xxx' },
        enabled: true,
      },
    ]

    const history: AlertHistory[] = [
      {
        id: '1',
        ruleName: '错误率过高',
        severity: 'critical',
        message: '系统错误率达到1.2%，超过阈值1%',
        timestamp: '2025-08-22 14:15:30',
        status: 'firing',
        duration: 1800,
      },
      {
        id: '2',
        ruleName: '推荐延迟过高',
        severity: 'high',
        message: 'P95延迟达到125ms，超过阈值100ms',
        timestamp: '2025-08-22 13:45:30',
        status: 'resolved',
        duration: 900,
      },
      {
        id: '3',
        ruleName: '缓存命中率低',
        severity: 'medium',
        message: '缓存命中率为88.5%，低于阈值90%',
        timestamp: '2025-08-22 12:30:15',
        status: 'resolved',
        duration: 600,
      },
    ]

    setAlertRules(rules)
    setChannels(notificationChannels)
    setAlertHistory(history)
  }, [])

  const handleCreateRule = () => {
    setEditingRule(null)
    form.resetFields()
    setShowRuleModal(true)
  }

  const handleEditRule = (rule: AlertRule) => {
    setEditingRule(rule)
    form.setFieldsValue(rule)
    setShowRuleModal(true)
  }

  const handleDeleteRule = (ruleId: string) => {
    Modal.confirm({
      title: '确认删除',
      content: '确定要删除这个告警规则吗？',
      onOk: () => {
        setAlertRules(prev => prev.filter(rule => rule.id !== ruleId))
      },
    })
  }

  const handleSaveRule = async (values: any) => {
    const newRule: AlertRule = {
      id: editingRule?.id || Date.now().toString(),
      ...values,
      enabled: values.enabled ?? true,
      status: 'normal',
      lastTriggered: undefined,
    }

    if (editingRule) {
      setAlertRules(prev =>
        prev.map(rule => (rule.id === editingRule.id ? newRule : rule))
      )
    } else {
      setAlertRules(prev => [...prev, newRule])
    }

    setShowRuleModal(false)
    form.resetFields()
  }

  const handleToggleRule = (ruleId: string, enabled: boolean) => {
    setAlertRules(prev =>
      prev.map(rule => (rule.id === ruleId ? { ...rule, enabled } : rule))
    )
  }

  const handleCreateChannel = () => {
    setEditingChannel(null)
    channelForm.resetFields()
    setShowChannelModal(true)
  }

  const handleEditChannel = (channel: NotificationChannel) => {
    setEditingChannel(channel)
    channelForm.setFieldsValue(channel)
    setShowChannelModal(true)
  }

  const handleSaveChannel = async (values: any) => {
    const newChannel: NotificationChannel = {
      id: editingChannel?.id || Date.now().toString(),
      ...values,
      enabled: values.enabled ?? true,
    }

    if (editingChannel) {
      setChannels(prev =>
        prev.map(channel =>
          channel.id === editingChannel.id ? newChannel : channel
        )
      )
    } else {
      setChannels(prev => [...prev, newChannel])
    }

    setShowChannelModal(false)
    channelForm.resetFields()
  }

  const ruleColumns: ColumnsType<AlertRule> = [
    {
      title: '规则名称',
      dataIndex: 'name',
      key: 'name',
      render: (text, record) => (
        <div>
          <strong>{text}</strong>
          <div style={{ fontSize: '12px', color: '#666' }}>
            {record.description}
          </div>
        </div>
      ),
    },
    {
      title: '监控指标',
      dataIndex: 'metric',
      key: 'metric',
      render: metric => <code>{metric}</code>,
    },
    {
      title: '条件',
      key: 'condition',
      render: (_, record) => (
        <span>
          {record.condition.replace('_', ' ')} {record.threshold}
          {record.metric.includes('rate')
            ? '%'
            : record.metric.includes('latency')
              ? 'ms'
              : record.metric.includes('qps')
                ? ' req/s'
                : ''}
        </span>
      ),
    },
    {
      title: '严重性',
      dataIndex: 'severity',
      key: 'severity',
      render: severity => {
        const config = {
          low: { color: 'green', text: '低' },
          medium: { color: 'orange', text: '中' },
          high: { color: 'red', text: '高' },
          critical: { color: 'purple', text: '严重' },
        }
        return <Tag color={config[severity].color}>{config[severity].text}</Tag>
      },
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: status => {
        const config = {
          normal: {
            color: 'green',
            icon: <CheckCircleOutlined />,
            text: '正常',
          },
          firing: {
            color: 'red',
            icon: <ExclamationTriangleOutlined />,
            text: '告警中',
          },
          resolved: {
            color: 'blue',
            icon: <CheckCircleOutlined />,
            text: '已解决',
          },
        }
        return (
          <Tag color={config[status].color} icon={config[status].icon}>
            {config[status].text}
          </Tag>
        )
      },
    },
    {
      title: '启用状态',
      dataIndex: 'enabled',
      key: 'enabled',
      render: (enabled, record) => (
        <Switch
          checked={enabled}
          onChange={checked => handleToggleRule(record.id, checked)}
        />
      ),
    },
    {
      title: '最后触发',
      dataIndex: 'lastTriggered',
      key: 'lastTriggered',
      render: time => time || '-',
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record) => (
        <Space>
          <Button
            size="small"
            icon={<EditOutlined />}
            onClick={() => handleEditRule(record)}
          >
            编辑
          </Button>
          <Button
            size="small"
            danger
            icon={<DeleteOutlined />}
            onClick={() => handleDeleteRule(record.id)}
          >
            删除
          </Button>
        </Space>
      ),
    },
  ]

  const channelColumns: ColumnsType<NotificationChannel> = [
    {
      title: '频道名称',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: type => {
        const config = {
          email: { color: 'blue', text: '邮件' },
          slack: { color: 'purple', text: 'Slack' },
          webhook: { color: 'orange', text: 'Webhook' },
          sms: { color: 'green', text: '短信' },
        }
        return <Tag color={config[type].color}>{config[type].text}</Tag>
      },
    },
    {
      title: '配置',
      dataIndex: 'config',
      key: 'config',
      render: (config, record) => (
        <div style={{ fontSize: '12px' }}>
          {record.type === 'email' &&
            `收件人: ${config.recipients?.join(', ')}`}
          {record.type === 'slack' && `频道: ${config.channel}`}
          {record.type === 'webhook' &&
            `URL: ${config.url?.substring(0, 30)}...`}
          {record.type === 'sms' && `号码: ${config.phone}`}
        </div>
      ),
    },
    {
      title: '启用状态',
      dataIndex: 'enabled',
      key: 'enabled',
      render: enabled => <Switch checked={enabled} />,
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record) => (
        <Space>
          <Button
            size="small"
            icon={<EditOutlined />}
            onClick={() => handleEditChannel(record)}
          >
            编辑
          </Button>
          <Button size="small" icon={<ThunderboltOutlined />}>
            测试
          </Button>
        </Space>
      ),
    },
  ]

  const historyColumns: ColumnsType<AlertHistory> = [
    {
      title: '时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      width: 150,
    },
    {
      title: '规则',
      dataIndex: 'ruleName',
      key: 'ruleName',
    },
    {
      title: '严重性',
      dataIndex: 'severity',
      key: 'severity',
      render: severity => {
        const config = {
          low: { color: 'green' },
          medium: { color: 'orange' },
          high: { color: 'red' },
          critical: { color: 'purple' },
        }
        return <Tag color={config[severity].color}>{severity}</Tag>
      },
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: status => (
        <Tag color={status === 'firing' ? 'red' : 'green'}>
          {status === 'firing' ? '告警中' : '已解决'}
        </Tag>
      ),
    },
    {
      title: '持续时间',
      dataIndex: 'duration',
      key: 'duration',
      render: duration => `${Math.floor(duration / 60)}分${duration % 60}秒`,
    },
    {
      title: '消息',
      dataIndex: 'message',
      key: 'message',
      ellipsis: true,
    },
  ]

  const firingAlerts = alertRules.filter(
    rule => rule.status === 'firing'
  ).length
  const enabledRules = alertRules.filter(rule => rule.enabled).length

  return (
    <div style={{ padding: '24px' }}>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '24px',
        }}
      >
        <h1 style={{ margin: 0, display: 'flex', alignItems: 'center' }}>
          <BellOutlined style={{ marginRight: '8px' }} />
          强化学习告警配置
        </h1>
        <Space>
          <Button
            type="primary"
            icon={<PlusOutlined />}
            onClick={handleCreateRule}
          >
            新建告警规则
          </Button>
          <Button icon={<SettingOutlined />} onClick={handleCreateChannel}>
            配置通知频道
          </Button>
        </Space>
      </div>

      {/* 告警概览 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} sm={8}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <div
                style={{
                  fontSize: '24px',
                  fontWeight: 'bold',
                  color: '#cf1322',
                }}
              >
                {firingAlerts}
              </div>
              <div>当前告警</div>
            </div>
          </Card>
        </Col>
        <Col xs={24} sm={8}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <div
                style={{
                  fontSize: '24px',
                  fontWeight: 'bold',
                  color: '#3f8600',
                }}
              >
                {enabledRules}
              </div>
              <div>启用规则</div>
            </div>
          </Card>
        </Col>
        <Col xs={24} sm={8}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <div
                style={{
                  fontSize: '24px',
                  fontWeight: 'bold',
                  color: '#1890ff',
                }}
              >
                {channels.filter(c => c.enabled).length}
              </div>
              <div>通知频道</div>
            </div>
          </Card>
        </Col>
      </Row>

      {/* 当前告警警告 */}
      {firingAlerts > 0 && (
        <Alert
          message="系统告警"
          description={`当前有 ${firingAlerts} 个告警规则正在触发，请及时处理`}
          type="error"
          showIcon
          action={
            <Button size="small" danger>
              查看详情
            </Button>
          }
          closable
          style={{ marginBottom: '24px' }}
        />
      )}

      {/* 标签页 */}
      <Tabs defaultActiveKey="rules">
        <TabPane tab="告警规则" key="rules">
          <Card>
            <Table
              dataSource={alertRules}
              columns={ruleColumns}
              rowKey="id"
              pagination={{ pageSize: 10 }}
              size="middle"
            />
          </Card>
        </TabPane>

        <TabPane tab="通知频道" key="channels">
          <Card>
            <Table
              dataSource={channels}
              columns={channelColumns}
              rowKey="id"
              pagination={false}
              size="middle"
            />
          </Card>
        </TabPane>

        <TabPane tab="告警历史" key="history">
          <Card>
            <Table
              dataSource={alertHistory}
              columns={historyColumns}
              rowKey="id"
              pagination={{ pageSize: 20 }}
              size="small"
            />
          </Card>
        </TabPane>
      </Tabs>

      {/* 告警规则编辑弹窗 */}
      <Modal
        title={editingRule ? '编辑告警规则' : '新建告警规则'}
        visible={showRuleModal}
        onCancel={() => setShowRuleModal(false)}
        onOk={() => form.submit()}
        width={600}
      >
        <Form form={form} layout="vertical" onFinish={handleSaveRule}>
          <Form.Item
            name="name"
            label="规则名称"
            rules={[{ required: true, message: '请输入规则名称' }]}
          >
            <Input placeholder="输入告警规则名称" />
          </Form.Item>

          <Form.Item
            name="description"
            label="描述"
            rules={[{ required: true, message: '请输入规则描述' }]}
          >
            <TextArea rows={2} placeholder="描述这个告警规则的用途" />
          </Form.Item>

          <Form.Item
            name="metric"
            label="监控指标"
            rules={[{ required: true, message: '请选择监控指标' }]}
          >
            <Select placeholder="选择要监控的指标">
              <Option value="rl_recommendation_latency_p95">
                推荐延迟 P95
              </Option>
              <Option value="rl_requests_per_second">每秒请求数</Option>
              <Option value="rl_cache_hit_rate">缓存命中率</Option>
              <Option value="rl_error_rate">错误率</Option>
              <Option value="rl_model_accuracy">模型准确率</Option>
              <Option value="rl_cpu_usage">CPU使用率</Option>
              <Option value="rl_memory_usage">内存使用率</Option>
            </Select>
          </Form.Item>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="condition"
                label="条件"
                rules={[{ required: true, message: '请选择条件' }]}
              >
                <Select placeholder="选择比较条件">
                  <Option value="greater_than">大于</Option>
                  <Option value="less_than">小于</Option>
                  <Option value="equals">等于</Option>
                  <Option value="not_equals">不等于</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="threshold"
                label="阈值"
                rules={[{ required: true, message: '请输入阈值' }]}
              >
                <InputNumber
                  placeholder="输入阈值"
                  style={{ width: '100%' }}
                  min={0}
                />
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="duration"
                label="持续时间(秒)"
                rules={[{ required: true, message: '请输入持续时间' }]}
              >
                <InputNumber
                  placeholder="持续多少秒后触发"
                  style={{ width: '100%' }}
                  min={0}
                />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="severity"
                label="严重性"
                rules={[{ required: true, message: '请选择严重性' }]}
              >
                <Select placeholder="选择严重性级别">
                  <Option value="low">低</Option>
                  <Option value="medium">中</Option>
                  <Option value="high">高</Option>
                  <Option value="critical">严重</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>

          <Form.Item
            name="channels"
            label="通知频道"
            rules={[{ required: true, message: '请选择通知频道' }]}
          >
            <Select
              mode="multiple"
              placeholder="选择通知频道"
              options={channels.map(c => ({ label: c.name, value: c.id }))}
            />
          </Form.Item>
        </Form>
      </Modal>

      {/* 通知频道编辑弹窗 */}
      <Modal
        title={editingChannel ? '编辑通知频道' : '新建通知频道'}
        visible={showChannelModal}
        onCancel={() => setShowChannelModal(false)}
        onOk={() => channelForm.submit()}
        width={500}
      >
        <Form form={channelForm} layout="vertical" onFinish={handleSaveChannel}>
          <Form.Item
            name="name"
            label="频道名称"
            rules={[{ required: true, message: '请输入频道名称' }]}
          >
            <Input placeholder="输入通知频道名称" />
          </Form.Item>

          <Form.Item
            name="type"
            label="类型"
            rules={[{ required: true, message: '请选择频道类型' }]}
          >
            <Select placeholder="选择通知方式">
              <Option value="email">邮件</Option>
              <Option value="slack">Slack</Option>
              <Option value="webhook">Webhook</Option>
              <Option value="sms">短信</Option>
            </Select>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}

export default RLAlertConfigPage
