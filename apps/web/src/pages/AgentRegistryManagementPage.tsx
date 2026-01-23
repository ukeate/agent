import React, { useState, useEffect } from 'react'
import {
  Card,
  Table,
  Button,
  Space,
  Tag,
  Badge,
  Modal,
  Form,
  Input,
  Select,
  Drawer,
  Descriptions,
  Typography,
  Row,
  Col,
  Statistic,
  Progress,
  Alert,
  Divider,
  Timeline,
  Tooltip,
  message,
} from 'antd'
import { logger } from '../utils/logger'
import {
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  EyeOutlined,
  ReloadOutlined,
  ExportOutlined,
  SearchOutlined,
  FilterOutlined,
  UserOutlined,
  RobotOutlined,
  ThunderboltOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  CloseCircleOutlined,
  SyncOutlined,
  GlobalOutlined,
  MonitorOutlined,
  SettingOutlined,
  ApiOutlined,
} from '@ant-design/icons'
import {
  agentRegistryService,
  type AgentInfo as AgentInfoType,
  type RegisterAgentRequest,
  type UpdateAgentRequest,
} from '../services/agentRegistryService'

const { Title, Paragraph, Text } = Typography
const { Option } = Select

interface AgentRegistryManagementPageProps {}

type AgentInfo = AgentInfoType

const AgentRegistryManagementPage: React.FC<
  AgentRegistryManagementPageProps
> = () => {
  const [agents, setAgents] = useState<AgentInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedAgent, setSelectedAgent] = useState<AgentInfo | null>(null)
  const [drawerVisible, setDrawerVisible] = useState(false)
  const [modalVisible, setModalVisible] = useState(false)
  const [filterStatus, setFilterStatus] = useState<string>('all')
  const [filterType, setFilterType] = useState<string>('all')
  const [searchText, setSearchText] = useState('')

  const [form] = Form.useForm()

  const resolveErrorMessage = (error: unknown, fallback: string) => {
    if (error instanceof Error && error.message) return error.message
    if (typeof error === 'string' && error.trim()) return error
    return fallback
  }

  // 加载智能体列表
  const loadAgents = async () => {
    try {
      setLoading(true)
      const data = await agentRegistryService.listAgents()
      setAgents(data)
    } catch (error) {
      logger.error('加载智能体列表失败:', error)
      message.error(resolveErrorMessage(error, '加载智能体列表失败'))
      setAgents([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadAgents()
  }, [])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online':
        return 'success'
      case 'offline':
        return 'default'
      case 'error':
        return 'error'
      case 'registering':
        return 'processing'
      default:
        return 'default'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'online':
        return <CheckCircleOutlined />
      case 'offline':
        return <CloseCircleOutlined />
      case 'error':
        return <ExclamationCircleOutlined />
      case 'registering':
        return <SyncOutlined spin />
      default:
        return <MonitorOutlined />
    }
  }

  const getHealthColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return '#52c41a'
      case 'unhealthy':
        return '#ff4d4f'
      case 'unknown':
        return '#d9d9d9'
      default:
        return '#d9d9d9'
    }
  }

  const formatTime = (value?: string) => {
    if (!value) return '-'
    const parsed = new Date(value)
    if (Number.isNaN(parsed.getTime())) return '-'
    return parsed.toLocaleString()
  }

  const filteredAgents = agents.filter(agent => {
    const matchesStatus =
      filterStatus === 'all' || agent.status === filterStatus
    const matchesType = filterType === 'all' || agent.type === filterType
    const matchesSearch =
      !searchText ||
      agent.name.toLowerCase().includes(searchText.toLowerCase()) ||
      agent.capabilities.some(cap =>
        cap.toLowerCase().includes(searchText.toLowerCase())
      )
    return matchesStatus && matchesType && matchesSearch
  })

  const handleExport = () => {
    if (typeof document === 'undefined') return
    if (filteredAgents.length === 0) {
      message.warning('暂无可导出的智能体')
      return
    }
    const payload = filteredAgents.map(agent => ({
      id: agent.id,
      name: agent.name,
      type: agent.type,
      status: agent.status,
      version: agent.version,
      endpoint: agent.endpoint,
      capabilities: agent.capabilities,
      metadata: agent.metadata,
      metrics: agent.metrics,
      healthCheck: agent.healthCheck,
    }))
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: 'application/json;charset=utf-8',
    })
    const url = URL.createObjectURL(blob)
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-')
    const link = document.createElement('a')
    link.href = url
    link.download = `agent-registry-${timestamp}.json`
    document.body.appendChild(link)
    link.click()
    link.remove()
    URL.revokeObjectURL(url)
    message.success('导出成功')
  }

  const handleRegisterAgent = async (values: any) => {
    try {
      setLoading(true)
      const requestData: RegisterAgentRequest = {
        name: values.name,
        type: values.type,
        endpoint: values.endpoint,
        version: values.version,
        capabilities: values.capabilities || [],
        description: values.description,
        tags: values.tags,
        owner: values.owner,
        environment: values.environment,
        region: values.region,
        healthCheckInterval: values.healthCheckInterval,
        healthCheckTimeout: values.healthCheckTimeout,
        healthCheckRetries: values.healthCheckRetries,
      }

      const newAgent = await agentRegistryService.registerAgent(requestData)
      setAgents(prev => [newAgent, ...prev])
      setModalVisible(false)
      form.resetFields()
      message.success('智能体注册成功')
    } catch (error) {
      logger.error('注册智能体失败:', error)
      message.error(resolveErrorMessage(error, '注册智能体失败'))
    } finally {
      setLoading(false)
    }
  }

  const handleViewAgent = (agent: AgentInfo) => {
    setSelectedAgent(agent)
    setDrawerVisible(true)
  }

  const handleDeleteAgent = (agentId: string) => {
    Modal.confirm({
      title: '确认删除',
      content: '确定要删除这个智能体吗？此操作不可撤销。',
      onOk: async () => {
        try {
          await agentRegistryService.deleteAgent(agentId)
          setAgents(prev => prev.filter(agent => agent.id !== agentId))
          message.success('智能体删除成功')
        } catch (error) {
          logger.error('删除智能体失败:', error)
          message.error(resolveErrorMessage(error, '删除智能体失败'))
        }
      },
    })
  }

  const handleRefresh = async () => {
    await loadAgents()
  }

  const columns = [
    {
      title: '智能体信息',
      key: 'agent',
      render: (_, agent: AgentInfo) => (
        <Space>
          <Badge color={getHealthColor(agent.healthCheck.status)} />
          <div>
            <div>
              <Text strong>{agent.name}</Text>
              <Tag
                color={getStatusColor(agent.status)}
                style={{ marginLeft: 8 }}
              >
                {getStatusIcon(agent.status)} {agent.status.toUpperCase()}
              </Tag>
            </div>
            <Text type="secondary" style={{ fontSize: '12px' }}>
              {agent.type} • v{agent.version}
            </Text>
          </div>
        </Space>
      ),
    },
    {
      title: '服务端点',
      dataIndex: 'endpoint',
      key: 'endpoint',
      render: (endpoint: string) => (
        <Text code style={{ fontSize: '12px' }}>
          {endpoint}
        </Text>
      ),
    },
    {
      title: '核心能力',
      dataIndex: 'capabilities',
      key: 'capabilities',
      render: (capabilities: string[]) => (
        <div>
          {capabilities.slice(0, 2).map(cap => (
            <Tag key={cap} size="small">
              {cap}
            </Tag>
          ))}
          {capabilities.length > 2 && (
            <Tag size="small">+{capabilities.length - 2}</Tag>
          )}
        </div>
      ),
    },
    {
      title: '性能指标',
      key: 'performance',
      render: (_, agent: AgentInfo) => (
        <div style={{ minWidth: '120px' }}>
          <div>
            <Text type="secondary" style={{ fontSize: '12px' }}>
              在线时长:
            </Text>
            <Text style={{ marginLeft: 4, fontSize: '12px' }}>
              {agent.metrics.uptime}小时
            </Text>
          </div>
          <div>
            <Text type="secondary" style={{ fontSize: '12px' }}>
              响应时间:
            </Text>
            <Text style={{ marginLeft: 4, fontSize: '12px' }}>
              {agent.metrics.responseTime}ms
            </Text>
          </div>
        </div>
      ),
    },
    {
      title: '分组信息',
      key: 'environment',
      render: (_, agent: AgentInfo) => (
        <div>
          <Tag color="blue">{agent.metadata.environment || '-'}</Tag>
          <br />
          <Text type="secondary" style={{ fontSize: '12px' }}>
            {agent.metadata.region || '-'}
          </Text>
        </div>
      ),
    },
    {
      title: '最后活跃',
      key: 'lastSeen',
      render: (_, agent: AgentInfo) => (
        <Text type="secondary" style={{ fontSize: '12px' }}>
          {formatTime(agent.metadata.lastSeen)}
        </Text>
      ),
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, agent: AgentInfo) => (
        <Space size="small">
          <Tooltip title="查看详情">
            <Button
              size="small"
              icon={<EyeOutlined />}
              onClick={() => handleViewAgent(agent)}
            />
          </Tooltip>
          <Tooltip title="编辑">
            <Button
              size="small"
              icon={<EditOutlined />}
              onClick={() => message.info('编辑功能开发中')}
            />
          </Tooltip>
          <Tooltip title="删除">
            <Button
              size="small"
              danger
              icon={<DeleteOutlined />}
              onClick={() => handleDeleteAgent(agent.id)}
            />
          </Tooltip>
        </Space>
      ),
    },
  ]

  const agentTypes = [
    { value: 'ML_PROCESSOR', label: '机器学习处理器' },
    { value: 'DATA_ANALYZER', label: '数据分析器' },
    { value: 'RECOMMENDER', label: '推荐引擎' },
    { value: 'CONVERSATIONAL', label: '对话助手' },
    { value: 'WORKFLOW_ENGINE', label: '工作流引擎' },
    { value: 'CUSTOM', label: '自定义智能体' },
  ]

  return (
    <div style={{ padding: '24px', background: '#f5f5f5', minHeight: '100vh' }}>
      <div style={{ maxWidth: '1400px', margin: '0 auto' }}>
        {/* 页面标题 */}
        <div style={{ marginBottom: '24px' }}>
          <Title level={2}>
            <RobotOutlined /> 智能体注册管理
          </Title>
          <Paragraph>
            管理所有注册在服务发现系统中的智能体，包括服务注册、健康检查、性能监控和元数据管理。
          </Paragraph>
        </div>

        {/* 统计概览 */}
        <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
          <Col xs={24} sm={6}>
            <Card>
              <Statistic
                title="总智能体数"
                value={agents.length}
                prefix={<UserOutlined />}
                valueStyle={{ color: '#1890ff' }}
              />
            </Card>
          </Col>
          <Col xs={24} sm={6}>
            <Card>
              <Statistic
                title="在线数量"
                value={agents.filter(a => a.status === 'online').length}
                prefix={<CheckCircleOutlined />}
                valueStyle={{ color: '#52c41a' }}
              />
            </Card>
          </Col>
          <Col xs={24} sm={6}>
            <Card>
              <Statistic
                title="异常数量"
                value={agents.filter(a => a.status === 'error').length}
                prefix={<ExclamationCircleOutlined />}
                valueStyle={{ color: '#ff4d4f' }}
              />
            </Card>
          </Col>
          <Col xs={24} sm={6}>
            <Card>
              <Statistic
                title="平均在线时长"
                value={
                  agents.reduce((sum, a) => sum + a.metrics.uptime, 0) /
                    agents.length || 0
                }
                suffix="小时"
                precision={1}
                prefix={<ThunderboltOutlined />}
                valueStyle={{ color: '#faad14' }}
              />
            </Card>
          </Col>
        </Row>

        {/* 操作栏 */}
        <Card style={{ marginBottom: '16px' }}>
          <Row justify="space-between" align="middle">
            <Col>
              <Space>
                <Button
                  type="primary"
                  icon={<PlusOutlined />}
                  onClick={() => setModalVisible(true)}
                >
                  注册智能体
                </Button>
                <Button
                  icon={<ReloadOutlined />}
                  loading={loading}
                  onClick={handleRefresh}
                >
                  刷新
                </Button>
                <Button
                  icon={<ExportOutlined />}
                  onClick={handleExport}
                  disabled={filteredAgents.length === 0}
                >
                  导出配置
                </Button>
              </Space>
            </Col>
            <Col>
              <Space>
                <Input
                  name="agentSearch"
                  placeholder="搜索智能体名称或能力"
                  prefix={<SearchOutlined />}
                  value={searchText}
                  onChange={e => setSearchText(e.target.value)}
                  style={{ width: 200 }}
                />
                <Select
                  value={filterStatus}
                  onChange={setFilterStatus}
                  style={{ width: 120 }}
                  placeholder="状态筛选"
                >
                  <Option value="all">全部状态</Option>
                  <Option value="online">在线</Option>
                  <Option value="offline">离线</Option>
                  <Option value="error">异常</Option>
                  <Option value="registering">注册中</Option>
                </Select>
                <Select
                  value={filterType}
                  onChange={setFilterType}
                  style={{ width: 150 }}
                  placeholder="类型筛选"
                >
                  <Option value="all">全部类型</Option>
                  {agentTypes.map(type => (
                    <Option key={type.value} value={type.value}>
                      {type.label}
                    </Option>
                  ))}
                </Select>
              </Space>
            </Col>
          </Row>
        </Card>

        {/* 智能体列表 */}
        <Card>
          <Table
            columns={columns}
            dataSource={filteredAgents}
            rowKey="id"
            loading={loading}
            pagination={{
              pageSize: 10,
              showSizeChanger: true,
              showQuickJumper: true,
              showTotal: (total, range) =>
                `第 ${range[0]}-${range[1]} 条，共 ${total} 条记录`,
            }}
          />
        </Card>

        {/* 注册智能体Modal */}
        <Modal
          title="注册新智能体"
          visible={modalVisible}
          onOk={form.submit}
          onCancel={() => {
            setModalVisible(false)
            form.resetFields()
          }}
          width={800}
          confirmLoading={loading}
        >
          <Form form={form} layout="vertical" onFinish={handleRegisterAgent}>
            <Row gutter={16}>
              <Col span={12}>
                <Form.Item
                  name="name"
                  label="智能体名称"
                  rules={[{ required: true, message: '请输入智能体名称' }]}
                >
                  <Input placeholder="例如: ml-processor-001" />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item
                  name="type"
                  label="智能体类型"
                  rules={[{ required: true, message: '请选择智能体类型' }]}
                >
                  <Select placeholder="选择智能体类型">
                    {agentTypes.map(type => (
                      <Option key={type.value} value={type.value}>
                        {type.label}
                      </Option>
                    ))}
                  </Select>
                </Form.Item>
              </Col>
            </Row>

            <Row gutter={16}>
              <Col span={12}>
                <Form.Item
                  name="endpoint"
                  label="服务端点"
                  rules={[
                    { required: true, message: '请输入服务端点' },
                    { type: 'url', message: '请输入有效的URL' },
                  ]}
                >
                  <Input placeholder="http://192.168.1.100:8080" />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item
                  name="version"
                  label="版本号"
                  rules={[{ required: true, message: '请输入版本号' }]}
                >
                  <Input placeholder="1.0.0" />
                </Form.Item>
              </Col>
            </Row>

            <Form.Item name="description" label="描述信息">
              <Input.TextArea
                rows={2}
                placeholder="简要描述智能体的功能和用途"
              />
            </Form.Item>

            <Row gutter={16}>
              <Col span={12}>
                <Form.Item
                  name="capabilities"
                  label="核心能力"
                  rules={[{ required: true, message: '至少添加一个能力' }]}
                >
                  <Select
                    mode="tags"
                    placeholder="添加智能体的核心能力标签"
                    tokenSeparators={[',']}
                  />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item name="tags" label="标签">
                  <Select
                    mode="tags"
                    placeholder="添加标签进行分类管理"
                    tokenSeparators={[',']}
                  />
                </Form.Item>
              </Col>
            </Row>

            <Row gutter={16}>
              <Col span={12}>
                <Form.Item name="environment" label="分组">
                  <Input placeholder="如: default" />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item name="region" label="区域">
                  <Input placeholder="如: us-east-1" />
                </Form.Item>
              </Col>
            </Row>

            <Form.Item name="owner" label="所有者">
              <Input placeholder="团队或个人名称" />
            </Form.Item>

            <Divider>健康检查配置</Divider>

            <Row gutter={16}>
              <Col span={8}>
                <Form.Item name="healthCheckInterval" label="检查间隔(秒)">
                  <Input type="number" placeholder="30" />
                </Form.Item>
              </Col>
              <Col span={8}>
                <Form.Item name="healthCheckTimeout" label="超时时间(秒)">
                  <Input type="number" placeholder="5" />
                </Form.Item>
              </Col>
              <Col span={8}>
                <Form.Item name="healthCheckRetries" label="重试次数">
                  <Input type="number" placeholder="3" />
                </Form.Item>
              </Col>
            </Row>
          </Form>
        </Modal>

        {/* 智能体详情Drawer */}
        <Drawer
          title="智能体详细信息"
          visible={drawerVisible}
          onClose={() => setDrawerVisible(false)}
          width={720}
        >
          {selectedAgent && (
            <div>
              <Alert
                message={`智能体状态: ${selectedAgent.status.toUpperCase()}`}
                description={`健康检查: ${selectedAgent.healthCheck.status}`}
                type={
                  selectedAgent.status === 'online'
                    ? 'success'
                    : selectedAgent.status === 'error'
                      ? 'error'
                      : 'info'
                }
                showIcon
                style={{ marginBottom: '24px' }}
              />

              <Descriptions title="基本信息" bordered column={2}>
                <Descriptions.Item label="名称">
                  {selectedAgent.name}
                </Descriptions.Item>
                <Descriptions.Item label="类型">
                  {selectedAgent.type}
                </Descriptions.Item>
                <Descriptions.Item label="版本">
                  {selectedAgent.version}
                </Descriptions.Item>
                <Descriptions.Item label="端点">
                  {selectedAgent.endpoint}
                </Descriptions.Item>
                <Descriptions.Item label="分组">
                  {selectedAgent.metadata.environment || '-'}
                </Descriptions.Item>
                <Descriptions.Item label="区域">
                  {selectedAgent.metadata.region || '-'}
                </Descriptions.Item>
                <Descriptions.Item label="创建时间" span={2}>
                  {formatTime(selectedAgent.metadata.created)}
                </Descriptions.Item>
              </Descriptions>

              <Divider />

              <Title level={4}>核心能力</Title>
              <div style={{ marginBottom: '16px' }}>
                {selectedAgent.capabilities.map(cap => (
                  <Tag key={cap} color="blue">
                    {cap}
                  </Tag>
                ))}
              </div>

              <Title level={4}>性能指标</Title>
              <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
                <Col span={12}>
                  <Card size="small">
                    <Statistic
                      title="在线时长"
                      value={selectedAgent.metrics.uptime}
                      suffix="小时"
                    />
                  </Card>
                </Col>
                <Col span={12}>
                  <Card size="small">
                    <Statistic
                      title="响应时间"
                      value={selectedAgent.metrics.responseTime}
                      suffix="ms"
                    />
                  </Card>
                </Col>
                <Col span={12}>
                  <Card size="small">
                    <Statistic
                      title="请求总数"
                      value={selectedAgent.metrics.requestCount}
                    />
                  </Card>
                </Col>
                <Col span={12}>
                  <Card size="small">
                    <Statistic
                      title="错误率"
                      value={selectedAgent.metrics.errorRate}
                      suffix="%"
                    />
                  </Card>
                </Col>
              </Row>

              <Title level={4}>系统资源</Title>
              <div style={{ marginBottom: '24px' }}>
                <div style={{ marginBottom: '12px' }}>
                  <Text>内存使用: {selectedAgent.metrics.memoryUsage}%</Text>
                  <Progress
                    percent={selectedAgent.metrics.memoryUsage}
                    size="small"
                  />
                </div>
                <div>
                  <Text>CPU使用: {selectedAgent.metrics.cpuUsage}%</Text>
                  <Progress
                    percent={selectedAgent.metrics.cpuUsage}
                    size="small"
                  />
                </div>
              </div>

              <Title level={4}>健康检查配置</Title>
              <Descriptions bordered size="small">
                <Descriptions.Item label="启用状态">
                  {selectedAgent.healthCheck.enabled ? '已启用' : '已禁用'}
                </Descriptions.Item>
                <Descriptions.Item label="检查间隔">
                  {selectedAgent.healthCheck.interval}秒
                </Descriptions.Item>
                <Descriptions.Item label="超时时间">
                  {selectedAgent.healthCheck.timeout}秒
                </Descriptions.Item>
                <Descriptions.Item label="重试次数">
                  {selectedAgent.healthCheck.retries}次
                </Descriptions.Item>
              </Descriptions>
            </div>
          )}
        </Drawer>
      </div>
    </div>
  )
}

export default AgentRegistryManagementPage
