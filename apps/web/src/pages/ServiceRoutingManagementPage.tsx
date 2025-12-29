import React, { useState, useEffect } from 'react'
import { Card, Table, Button, Space, Tag, Modal, Form, Input, Select, Row, Col, Typography, Divider, Switch, InputNumber, Drawer, Alert, Badge, Tooltip, Timeline, Progress } from 'antd'
import { 
import { logger } from '../utils/logger'
  PlusOutlined, 
  EditOutlined, 
  DeleteOutlined, 
  EyeOutlined, 
  ReloadOutlined, 
  BranchesOutlined,
  ApiOutlined,
  ThunderboltOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  CloseCircleOutlined,
  SettingOutlined,
  ShareAltOutlined,
  GlobalOutlined,
  NodeIndexOutlined,
  MonitorOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined
} from '@ant-design/icons'
import { serviceDiscoveryService } from '../services/serviceDiscoveryService'
import apiClient from '../services/apiClient'

const { Title, Paragraph, Text } = Typography
const { Option } = Select
const { TextArea } = Input

interface ServiceRoutingManagementPageProps {}

interface RoutingRule {
  id: string
  name: string
  description: string
  status: 'active' | 'inactive' | 'error'
  priority: number
  conditions: {
    capability: string[]
    tags: string[]
    version: string
    region: string[]
    environment: string[]
    customRules: string
  }
  targets: {
    agentTypes: string[]
    endpoints: string[]
    loadBalanceStrategy: string
    failoverEnabled: boolean
    circuitBreakerEnabled: boolean
  }
  metrics: {
    requestCount: number
    successRate: number
    avgResponseTime: number
    errorCount: number
    lastUsed: string
  }
  created: string
  updated: string
  owner: string
}

interface RouteTestResult {
  success: boolean
  selectedAgent: string
  responseTime: number
  route: string
  timestamp: string
}

const ServiceRoutingManagementPage: React.FC<ServiceRoutingManagementPageProps> = () => {
  const [routingRules, setRoutingRules] = useState<RoutingRule[]>([])

  const [loading, setLoading] = useState(false)
  const [modalVisible, setModalVisible] = useState(false)
  const [drawerVisible, setDrawerVisible] = useState(false)
  const [testDrawerVisible, setTestDrawerVisible] = useState(false)
  const [selectedRule, setSelectedRule] = useState<RoutingRule | null>(null)
  const [editingRule, setEditingRule] = useState<RoutingRule | null>(null)
  const [testResults, setTestResults] = useState<RouteTestResult[]>([])

  const [form] = Form.useForm()
  const [testForm] = Form.useForm()

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'success'
      case 'inactive': return 'default'
      case 'error': return 'error'
      default: return 'default'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active': return <CheckCircleOutlined />
      case 'inactive': return <PauseCircleOutlined />
      case 'error': return <ExclamationCircleOutlined />
      default: return <CloseCircleOutlined />
    }
  }

  const loadBalanceStrategies = [
    { value: 'round_robin', label: '轮询' },
    { value: 'least_connections', label: '最少连接' },
    { value: 'weighted_round_robin', label: '加权轮询' },
    { value: 'capability_based', label: '能力优先' },
    { value: 'geographic', label: '地理位置优先' },
    { value: 'response_time', label: '响应时间优先' }
  ]

  const agentTypes = [
    { value: 'ML_PROCESSOR', label: '机器学习处理器' },
    { value: 'DATA_ANALYZER', label: '数据分析器' },
    { value: 'RECOMMENDER', label: '推荐引擎' },
    { value: 'CONVERSATIONAL', label: '对话助手' },
    { value: 'WORKFLOW_ENGINE', label: '工作流引擎' }
  ]

  const environments = ['development', 'testing', 'staging', 'production']
  const regions = ['us-east-1', 'us-west-2', 'eu-central-1', 'ap-southeast-1', 'ap-northeast-1']

  const loadRoutingRules = async () => {
    try {
      setLoading(true)
      const response = await apiClient.get('/service-routing/rules')
      setRoutingRules(response.data?.rules || [])
    } catch (error) {
      logger.error('加载路由规则失败:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadRoutingRules()
  }, [])

  const handleCreateRule = async (values: any) => {
    try {
      setLoading(true)
      const payload = {
        name: values.name,
        description: values.description || '',
        priority: values.priority || 50,
        conditions: {
          capability: values.capabilities || [],
          tags: values.tags || [],
          version: values.version || '',
          region: values.regions || [],
          environment: values.environments || [],
          customRules: values.customRules || ''
        },
        targets: {
          agentTypes: values.agentTypes || [],
          endpoints: values.endpoints || [],
          loadBalanceStrategy: values.loadBalanceStrategy || 'round_robin',
          failoverEnabled: values.failoverEnabled || false,
          circuitBreakerEnabled: values.circuitBreakerEnabled || false
        },
        owner: 'system'
      }
      if (editingRule) {
        await apiClient.put(`/service-routing/rules/${editingRule.id}`, payload)
      } else {
        await apiClient.post('/service-routing/rules', payload)
      }
      await loadRoutingRules()
      setModalVisible(false)
      setEditingRule(null)
      form.resetFields()
    } catch (error) {
      logger.error('创建路由规则失败:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleEditRule = (rule: RoutingRule) => {
    setEditingRule(rule)
    form.setFieldsValue({
      name: rule.name,
      description: rule.description,
      priority: rule.priority,
      capabilities: rule.conditions.capability,
      tags: rule.conditions.tags,
      version: rule.conditions.version,
      regions: rule.conditions.region,
      environments: rule.conditions.environment,
      customRules: rule.conditions.customRules,
      agentTypes: rule.targets.agentTypes,
      loadBalanceStrategy: rule.targets.loadBalanceStrategy,
      failoverEnabled: rule.targets.failoverEnabled,
      circuitBreakerEnabled: rule.targets.circuitBreakerEnabled
    })
    setModalVisible(true)
  }

  const handleDeleteRule = (ruleId: string) => {
    Modal.confirm({
      title: '确认删除',
      content: '确定要删除这个路由规则吗？此操作不可撤销。',
      onOk: async () => {
        try {
          setLoading(true)
          await apiClient.delete(`/service-routing/rules/${ruleId}`)
          await loadRoutingRules()
        } catch (error) {
          logger.error('删除路由规则失败:', error)
        } finally {
          setLoading(false)
        }
      }
    })
  }

  const handleToggleStatus = (ruleId: string) => {
    const current = routingRules.find(rule => rule.id === ruleId)
    if (!current) return
    const nextStatus = current.status === 'active' ? 'inactive' : 'active'
    setLoading(true)
    apiClient
      .patch(`/service-routing/rules/${ruleId}/status`, { status: nextStatus })
      .then(loadRoutingRules)
      .catch(error => {
        logger.error('切换规则状态失败:', error)
      })
      .finally(() => setLoading(false))
  }

  const handleTestRoute = async (values: any) => {
    try {
      setLoading(true)
      const res = await serviceDiscoveryService.selectAgent({
        capability: values.testCapability,
        strategy: values.testStrategy || 'round_robin',
        tags: values.testTags
      })
      const testResult: RouteTestResult = {
        success: Boolean(res.selected_agent),
        selectedAgent: res.selected_agent ? res.selected_agent.agent_id : '',
        responseTime: res.selection_time || 0,
        route: values.testCapability,
        timestamp: new Date().toISOString()
      }
      setTestResults(prev => [testResult, ...prev.slice(0, 9)])
    } catch (error) {
      logger.error('路由测试失败:', error)
      Modal.error({ title: '路由测试失败', content: (error as Error).message || '' })
    } finally {
      setLoading(false)
    }
  }

  const columns = [
    {
      title: '规则信息',
      key: 'rule',
      render: (_, rule: RoutingRule) => (
        <div>
          <div>
            <Text strong>{rule.name}</Text>
            <Tag color={getStatusColor(rule.status)} style={{ marginLeft: 8 }}>
              {getStatusIcon(rule.status)} {rule.status.toUpperCase()}
            </Tag>
          </div>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            {rule.description}
          </Text>
          <br />
          <Text type="secondary" style={{ fontSize: '12px' }}>
            优先级: {rule.priority}
          </Text>
        </div>
      )
    },
    {
      title: '路由条件',
      key: 'conditions',
      render: (_, rule: RoutingRule) => (
        <div style={{ maxWidth: '200px' }}>
          {rule.conditions.capability.slice(0, 2).map(cap => (
            <Tag key={cap} size="small" color="blue">{cap}</Tag>
          ))}
          {rule.conditions.capability.length > 2 && (
            <Tag size="small">+{rule.conditions.capability.length - 2}</Tag>
          )}
          <br />
          <Text type="secondary" style={{ fontSize: '12px' }}>
            环境: {rule.conditions.environment.join(', ')}
          </Text>
        </div>
      )
    },
    {
      title: '目标服务',
      key: 'targets',
      render: (_, rule: RoutingRule) => (
        <div>
          <div>
            {rule.targets.agentTypes.map(type => (
              <Tag key={type} size="small" color="green">{type}</Tag>
            ))}
          </div>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            策略: {rule.targets.loadBalanceStrategy}
          </Text>
          <br />
          <div>
            {rule.targets.failoverEnabled && <Badge status="success" text="故障转移" />}
            {rule.targets.circuitBreakerEnabled && <Badge status="warning" text="熔断器" />}
          </div>
        </div>
      )
    },
    {
      title: '性能指标',
      key: 'metrics',
      render: (_, rule: RoutingRule) => (
        <div style={{ minWidth: '120px' }}>
          <div>
            <Text type="secondary" style={{ fontSize: '12px' }}>请求数:</Text>
            <Text style={{ marginLeft: 4, fontSize: '12px' }}>{rule.metrics.requestCount}</Text>
          </div>
          <div>
            <Text type="secondary" style={{ fontSize: '12px' }}>成功率:</Text>
            <Text style={{ marginLeft: 4, fontSize: '12px', color: rule.metrics.successRate > 95 ? '#52c41a' : '#faad14' }}>
              {rule.metrics.successRate}%
            </Text>
          </div>
          <div>
            <Text type="secondary" style={{ fontSize: '12px' }}>响应:</Text>
            <Text style={{ marginLeft: 4, fontSize: '12px' }}>{rule.metrics.avgResponseTime}ms</Text>
          </div>
        </div>
      )
    },
    {
      title: '最后更新',
      key: 'updated',
      render: (_, rule: RoutingRule) => (
        <Text type="secondary" style={{ fontSize: '12px' }}>
          {new Date(rule.updated).toLocaleString()}
        </Text>
      )
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, rule: RoutingRule) => (
        <Space size="small">
          <Tooltip title="查看详情">
            <Button size="small" icon={<EyeOutlined />} onClick={() => {
              setSelectedRule(rule)
              setDrawerVisible(true)
            }} />
          </Tooltip>
          <Tooltip title="编辑">
            <Button size="small" icon={<EditOutlined />} onClick={() => handleEditRule(rule)} />
          </Tooltip>
          <Tooltip title={rule.status === 'active' ? '暂停' : '激活'}>
            <Button 
              size="small" 
              icon={rule.status === 'active' ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
              onClick={() => handleToggleStatus(rule.id)}
            />
          </Tooltip>
          <Tooltip title="删除">
            <Button size="small" danger icon={<DeleteOutlined />} onClick={() => handleDeleteRule(rule.id)} />
          </Tooltip>
        </Space>
      )
    }
  ]

  return (
    <div style={{ padding: '24px', background: '#f5f5f5', minHeight: '100vh' }}>
      <div style={{ maxWidth: '1400px', margin: '0 auto' }}>
        {/* 页面标题 */}
        <div style={{ marginBottom: '24px' }}>
          <Title level={2}>
            <BranchesOutlined /> 服务路由管理
          </Title>
          <Paragraph>
            配置和管理智能服务发现的路由规则，包括条件匹配、负载均衡策略和故障处理机制。
          </Paragraph>
        </div>

        {/* 统计概览 */}
        <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
          <Col xs={24} sm={6}>
            <Card>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div>
                  <Text type="secondary">总路由规则</Text>
                  <div style={{ fontSize: '24px', fontWeight: 'bold' }}>{routingRules.length}</div>
                </div>
                <ShareAltOutlined style={{ fontSize: '32px', color: '#1890ff' }} />
              </div>
            </Card>
          </Col>
          <Col xs={24} sm={6}>
            <Card>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div>
                  <Text type="secondary">活跃规则</Text>
                  <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#52c41a' }}>
                    {routingRules.filter(r => r.status === 'active').length}
                  </div>
                </div>
                <CheckCircleOutlined style={{ fontSize: '32px', color: '#52c41a' }} />
              </div>
            </Card>
          </Col>
          <Col xs={24} sm={6}>
            <Card>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div>
                  <Text type="secondary">异常规则</Text>
                  <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#ff4d4f' }}>
                    {routingRules.filter(r => r.status === 'error').length}
                  </div>
                </div>
                <ExclamationCircleOutlined style={{ fontSize: '32px', color: '#ff4d4f' }} />
              </div>
            </Card>
          </Col>
          <Col xs={24} sm={6}>
            <Card>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div>
                  <Text type="secondary">平均成功率</Text>
                  <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#faad14' }}>
                    {(routingRules.reduce((sum, r) => sum + r.metrics.successRate, 0) / routingRules.length || 0).toFixed(1)}%
                  </div>
                </div>
                <ThunderboltOutlined style={{ fontSize: '32px', color: '#faad14' }} />
              </div>
            </Card>
          </Col>
        </Row>

        {/* 操作栏 */}
        <Card style={{ marginBottom: '16px' }}>
          <Row justify="space-between" align="middle">
            <Col>
              <Space>
                <Button type="primary" icon={<PlusOutlined />} onClick={() => {
                  setEditingRule(null)
                  form.resetFields()
                  setModalVisible(true)
                }}>
                  创建路由规则
                </Button>
                <Button icon={<ReloadOutlined />} loading={loading} onClick={loadRoutingRules}>
                  刷新
                </Button>
                <Button icon={<ApiOutlined />} onClick={() => setTestDrawerVisible(true)}>
                  路由测试
                </Button>
              </Space>
            </Col>
            <Col>
              <Space>
                <Button icon={<SettingOutlined />}>
                  全局配置
                </Button>
                <Button icon={<MonitorOutlined />}>
                  性能监控
                </Button>
              </Space>
            </Col>
          </Row>
        </Card>

        {/* 路由规则列表 */}
        <Card>
          <Table
            columns={columns}
            dataSource={routingRules}
            rowKey="id"
            loading={loading}
            pagination={{
              pageSize: 10,
              showSizeChanger: true,
              showTotal: (total, range) => `第 ${range[0]}-${range[1]} 条，共 ${total} 条记录`
            }}
          />
        </Card>

        {/* 创建/编辑路由规则Modal */}
        <Modal
          title={editingRule ? "编辑路由规则" : "创建路由规则"}
          visible={modalVisible}
          onOk={form.submit}
          onCancel={() => {
            setModalVisible(false)
            setEditingRule(null)
            form.resetFields()
          }}
          width={800}
          confirmLoading={loading}
        >
          <Form
            form={form}
            layout="vertical"
            onFinish={handleCreateRule}
          >
            <Row gutter={16}>
              <Col span={12}>
                <Form.Item
                  name="name"
                  label="规则名称"
                  rules={[{ required: true, message: '请输入规则名称' }]}
                >
                  <Input placeholder="例如: ML处理器路由" />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item
                  name="priority"
                  label="优先级"
                  rules={[{ required: true, message: '请输入优先级' }]}
                >
                  <InputNumber min={1} max={100} placeholder="数值越大优先级越高" style={{ width: '100%' }} />
                </Form.Item>
              </Col>
            </Row>

            <Form.Item name="description" label="描述">
              <TextArea rows={2} placeholder="简要描述路由规则的用途" />
            </Form.Item>

            <Divider>路由条件</Divider>

            <Row gutter={16}>
              <Col span={12}>
                <Form.Item name="capabilities" label="所需能力">
                  <Select
                    mode="tags"
                    placeholder="选择或输入所需的智能体能力"
                    tokenSeparators={[',']}
                  />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item name="tags" label="标签匹配">
                  <Select
                    mode="tags"
                    placeholder="选择或输入标签"
                    tokenSeparators={[',']}
                  />
                </Form.Item>
              </Col>
            </Row>

            <Row gutter={16}>
              <Col span={8}>
                <Form.Item name="version" label="版本要求">
                  <Input placeholder="例如: >=2.0.0" />
                </Form.Item>
              </Col>
              <Col span={8}>
                <Form.Item name="environments" label="环境">
                  <Select mode="multiple" placeholder="选择环境">
                    {environments.map(env => (
                      <Option key={env} value={env}>{env}</Option>
                    ))}
                  </Select>
                </Form.Item>
              </Col>
              <Col span={8}>
                <Form.Item name="regions" label="区域">
                  <Select mode="multiple" placeholder="选择区域">
                    {regions.map(region => (
                      <Option key={region} value={region}>{region}</Option>
                    ))}
                  </Select>
                </Form.Item>
              </Col>
            </Row>

            <Form.Item name="customRules" label="自定义规则">
              <TextArea 
                rows={2} 
                placeholder="例如: cpu_usage < 80 AND memory_usage < 70"
              />
            </Form.Item>

            <Divider>目标配置</Divider>

            <Row gutter={16}>
              <Col span={12}>
                <Form.Item name="agentTypes" label="目标智能体类型">
                  <Select mode="multiple" placeholder="选择目标智能体类型">
                    {agentTypes.map(type => (
                      <Option key={type.value} value={type.value}>{type.label}</Option>
                    ))}
                  </Select>
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item name="loadBalanceStrategy" label="负载均衡策略">
                  <Select placeholder="选择负载均衡策略">
                    {loadBalanceStrategies.map(strategy => (
                      <Option key={strategy.value} value={strategy.value}>{strategy.label}</Option>
                    ))}
                  </Select>
                </Form.Item>
              </Col>
            </Row>

            <Row gutter={16}>
              <Col span={12}>
                <Form.Item name="failoverEnabled" valuePropName="checked">
                  <Switch /> 启用故障转移
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item name="circuitBreakerEnabled" valuePropName="checked">
                  <Switch /> 启用熔断器
                </Form.Item>
              </Col>
            </Row>
          </Form>
        </Modal>

        {/* 路由规则详情Drawer */}
        <Drawer
          title="路由规则详情"
          visible={drawerVisible}
          onClose={() => setDrawerVisible(false)}
          width={600}
        >
          {selectedRule && (
            <div>
              <Alert
                message={`规则状态: ${selectedRule.status.toUpperCase()}`}
                type={selectedRule.status === 'active' ? 'success' : selectedRule.status === 'error' ? 'error' : 'info'}
                showIcon
                style={{ marginBottom: '16px' }}
              />

              <Title level={4}>基本信息</Title>
              <Row gutter={16} style={{ marginBottom: '16px' }}>
                <Col span={12}>
                  <Text type="secondary">规则名称</Text>
                  <div>{selectedRule.name}</div>
                </Col>
                <Col span={12}>
                  <Text type="secondary">优先级</Text>
                  <div>{selectedRule.priority}</div>
                </Col>
                <Col span={24}>
                  <Text type="secondary">描述</Text>
                  <div>{selectedRule.description}</div>
                </Col>
              </Row>

              <Title level={4}>路由条件</Title>
              <div style={{ marginBottom: '16px' }}>
                <div style={{ marginBottom: '8px' }}>
                  <Text type="secondary">所需能力: </Text>
                  {selectedRule.conditions.capability.map(cap => (
                    <Tag key={cap} color="blue">{cap}</Tag>
                  ))}
                </div>
                <div style={{ marginBottom: '8px' }}>
                  <Text type="secondary">环境: </Text>
                  {selectedRule.conditions.environment.map(env => (
                    <Tag key={env}>{env}</Tag>
                  ))}
                </div>
                <div style={{ marginBottom: '8px' }}>
                  <Text type="secondary">区域: </Text>
                  {selectedRule.conditions.region.map(region => (
                    <Tag key={region}>{region}</Tag>
                  ))}
                </div>
                <div>
                  <Text type="secondary">版本要求: </Text>
                  <Text code>{selectedRule.conditions.version}</Text>
                </div>
                {selectedRule.conditions.customRules && (
                  <div style={{ marginTop: '8px' }}>
                    <Text type="secondary">自定义规则: </Text>
                    <Text code>{selectedRule.conditions.customRules}</Text>
                  </div>
                )}
              </div>

              <Title level={4}>目标配置</Title>
              <div style={{ marginBottom: '16px' }}>
                <div style={{ marginBottom: '8px' }}>
                  <Text type="secondary">目标类型: </Text>
                  {selectedRule.targets.agentTypes.map(type => (
                    <Tag key={type} color="green">{type}</Tag>
                  ))}
                </div>
                <div style={{ marginBottom: '8px' }}>
                  <Text type="secondary">负载均衡: </Text>
                  <Tag color="orange">{selectedRule.targets.loadBalanceStrategy}</Tag>
                </div>
                <div>
                  <Badge status={selectedRule.targets.failoverEnabled ? 'success' : 'default'} text="故障转移" />
                  <br />
                  <Badge status={selectedRule.targets.circuitBreakerEnabled ? 'warning' : 'default'} text="熔断器" />
                </div>
              </div>

              <Title level={4}>性能统计</Title>
              <Row gutter={16}>
                <Col span={12}>
                  <Card size="small">
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: '24px', fontWeight: 'bold' }}>{selectedRule.metrics.requestCount}</div>
                      <Text type="secondary">总请求数</Text>
                    </div>
                  </Card>
                </Col>
                <Col span={12}>
                  <Card size="small">
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#52c41a' }}>
                        {selectedRule.metrics.successRate}%
                      </div>
                      <Text type="secondary">成功率</Text>
                    </div>
                  </Card>
                </Col>
                <Col span={12}>
                  <Card size="small">
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: '24px', fontWeight: 'bold' }}>{selectedRule.metrics.avgResponseTime}ms</div>
                      <Text type="secondary">平均响应时间</Text>
                    </div>
                  </Card>
                </Col>
                <Col span={12}>
                  <Card size="small">
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#ff4d4f' }}>
                        {selectedRule.metrics.errorCount}
                      </div>
                      <Text type="secondary">错误次数</Text>
                    </div>
                  </Card>
                </Col>
              </Row>
            </div>
          )}
        </Drawer>

        {/* 路由测试Drawer */}
        <Drawer
          title="路由测试"
          visible={testDrawerVisible}
          onClose={() => setTestDrawerVisible(false)}
          width={600}
        >
          <Form
            form={testForm}
            layout="vertical"
            onFinish={handleTestRoute}
          >
            <Form.Item
              name="testCapability"
              label="测试能力"
              rules={[{ required: true, message: '请选择要测试的能力' }]}
            >
              <Select placeholder="选择要测试路由的能力">
                <Option value="text_processing">文本处理</Option>
                <Option value="sentiment_analysis">情感分析</Option>
                <Option value="data_mining">数据挖掘</Option>
                <Option value="recommendation">推荐引擎</Option>
              </Select>
            </Form.Item>

            <Button type="primary" htmlType="submit" loading={loading}>
              执行路由测试
            </Button>
          </Form>

          {testResults.length > 0 && (
            <>
              <Divider>测试结果</Divider>
              <Timeline>
                {testResults.map((result, index) => (
                  <Timeline.Item
                    key={index}
                    color={result.success ? 'green' : 'red'}
                    dot={result.success ? <CheckCircleOutlined /> : <ExclamationCircleOutlined />}
                  >
                    <div>
                      <Text strong>{result.route}</Text>
                      <Tag color={result.success ? 'success' : 'error'} style={{ marginLeft: 8 }}>
                        {result.success ? '成功' : '失败'}
                      </Tag>
                      <br />
                      <Text type="secondary" style={{ fontSize: '12px' }}>
                        智能体: {result.selectedAgent} | 响应时间: {result.responseTime}ms
                      </Text>
                      <br />
                      <Text type="secondary" style={{ fontSize: '12px' }}>
                        {new Date(result.timestamp).toLocaleString()}
                      </Text>
                    </div>
                  </Timeline.Item>
                ))}
              </Timeline>
            </>
          )}
        </Drawer>
      </div>
    </div>
  )
}

export default ServiceRoutingManagementPage
