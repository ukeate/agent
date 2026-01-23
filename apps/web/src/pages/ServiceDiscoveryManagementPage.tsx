import React, { useState, useEffect } from 'react'
import {
  Card,
  Button,
  Input,
  Select,
  Alert,
  Badge,
  Tabs,
  Form,
  Table,
  Progress,
  Space,
  Typography,
  Row,
  Col,
  Statistic,
  message,
  Modal,
  Tag,
} from 'antd'
import {
  CheckCircleOutlined,
  CloseCircleOutlined,
  ExclamationCircleOutlined,
  ReloadOutlined,
} from '@ant-design/icons'
import { logger } from '../utils/logger'
import {
  serviceDiscoveryService,
  type AgentMetadataResponse,
  type AgentRegistrationRequest,
  type ServiceStats,
  type HealthCheckResponse,
} from '../services/serviceDiscoveryService'

const { TabPane } = Tabs
const { Option } = Select
const { Title, Text } = Typography

const ServiceDiscoveryManagementPage: React.FC = () => {
  const [agents, setAgents] = useState<AgentMetadataResponse[]>([])
  const [stats, setStats] = useState<ServiceStats | null>(null)
  const [health, setHealth] = useState<HealthCheckResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)

  // 发现智能体的筛选参数
  const [capability, setCapability] = useState('')
  const [tags, setTags] = useState('')
  const [statusFilter, setStatusFilter] = useState('')
  const [group, setGroup] = useState('')
  const [region, setRegion] = useState('')

  // 注册智能体表单
  const [registerForm, setRegisterForm] = useState<
    Partial<AgentRegistrationRequest>
  >({
    agent_id: '',
    agent_type: 'general',
    name: '',
    version: '1.0.0',
    host: 'localhost',
    port: 8080,
    endpoint: '/api',
    capabilities: [],
    tags: [],
  })

  // 负载均衡配置
  const [loadBalancerStrategy, setLoadBalancerStrategy] =
    useState('round_robin')
  const [loadBalancerCapability, setLoadBalancerCapability] = useState('')
  const [selectedAgent, setSelectedAgent] =
    useState<AgentMetadataResponse | null>(null)

  useEffect(() => {
    loadInitialData()
  }, [])

  const loadInitialData = async () => {
    await Promise.all([loadAgents(), loadStats(), loadHealth()])
  }

  const loadAgents = async () => {
    try {
      setLoading(true)
      setError(null)
      const response = await serviceDiscoveryService.discoverAgents({
        capability: capability || undefined,
        tags: tags || undefined,
        status_filter: statusFilter || undefined,
        group: group || undefined,
        region: region || undefined,
      })
      setAgents(response.agents)
    } catch (err) {
      setError('加载智能体失败: ' + (err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  const loadStats = async () => {
    try {
      const statsData = await serviceDiscoveryService.getSystemStats()
      setStats(statsData)
    } catch (err) {
      logger.warn('加载统计数据失败:', err)
    }
  }

  const loadHealth = async () => {
    try {
      const healthData = await serviceDiscoveryService.healthCheck()
      setHealth(healthData)
    } catch (err) {
      logger.warn('健康检查失败:', err)
    }
  }

  const handleRegisterAgent = async () => {
    if (!registerForm.agent_id || !registerForm.name) {
      setError('请填写必填字段')
      return
    }

    try {
      setLoading(true)
      setError(null)

      await serviceDiscoveryService.registerAgent(
        registerForm as AgentRegistrationRequest
      )
      setSuccess('智能体注册成功')

      // 清空表单
      setRegisterForm({
        agent_id: '',
        agent_type: 'general',
        name: '',
        version: '1.0.0',
        host: 'localhost',
        port: 8080,
        endpoint: '/api',
        capabilities: [],
        tags: [],
      })

      // 刷新智能体列表
      await loadAgents()
    } catch (err) {
      setError('注册智能体失败: ' + (err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  const handleDeregisterAgent = async (agentId: string) => {
    if (!confirm('确定要注销此智能体吗？')) return

    try {
      setLoading(true)
      setError(null)

      await serviceDiscoveryService.deregisterAgent(agentId)
      setSuccess('智能体注销成功')

      // 刷新智能体列表
      await loadAgents()
    } catch (err) {
      setError('注销智能体失败: ' + (err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  const handleUpdateAgentStatus = async (
    agentId: string,
    newStatus: string
  ) => {
    try {
      setLoading(true)
      setError(null)

      await serviceDiscoveryService.updateAgentStatus(agentId, newStatus)
      setSuccess('智能体状态更新成功')

      // 刷新智能体列表
      await loadAgents()
    } catch (err) {
      setError('更新状态失败: ' + (err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  const handleSelectAgent = async () => {
    if (!loadBalancerCapability) {
      setError('请指定能力要求')
      return
    }

    try {
      setLoading(true)
      setError(null)

      const response = await serviceDiscoveryService.selectAgent({
        capability: loadBalancerCapability,
        strategy: loadBalancerStrategy,
      })

      setSelectedAgent(response.selected_agent)
      if (response.selected_agent) {
        setSuccess(
          `使用${response.strategy_used}策略选择智能体成功，耗时${response.selection_time.toFixed(3)}ms`
        )
      } else {
        setError('未找到符合条件的智能体')
      }
    } catch (err) {
      setError('选择智能体失败: ' + (err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'active':
        return 'success'
      case 'inactive':
        return 'error'
      case 'maintenance':
        return 'warning'
      default:
        return 'default'
    }
  }

  const formatUptime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    return `${hours}h ${minutes}m`
  }

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
        <Title level={2}>服务发现管理</Title>
        <Button
          icon={<ReloadOutlined />}
          onClick={loadInitialData}
          loading={loading}
        >
          刷新数据
        </Button>
      </div>

      {error && (
        <Alert
          message="错误"
          description={error}
          type="error"
          closable
          style={{ marginBottom: 16 }}
        />
      )}

      {success && (
        <Alert
          message="成功"
          description={success}
          type="success"
          closable
          style={{ marginBottom: 16 }}
        />
      )}

      {/* 系统状态概览 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} md={8}>
          <Card>
            <Statistic
              title="系统健康状态"
              value={health?.status || '加载中'}
              prefix={
                health?.status === 'healthy' ? (
                  <CheckCircleOutlined style={{ color: '#52c41a' }} />
                ) : (
                  <CloseCircleOutlined style={{ color: '#ff4d4f' }} />
                )
              }
            />
            {health && (
              <div style={{ marginTop: 8, fontSize: '12px', color: '#666' }}>
                版本: {health.version} | 运行时间:{' '}
                {formatUptime(health.uptime_seconds)}
              </div>
            )}
          </Card>
        </Col>
        <Col xs={24} md={8}>
          <Card>
            <Statistic title="智能体统计" value={agents.length} suffix="个" />
            <div style={{ marginTop: 8, fontSize: '12px', color: '#666' }}>
              活跃: {agents.filter(a => a.status === 'active').length} | 离线:{' '}
              {agents.filter(a => a.status === 'inactive').length}
            </div>
          </Card>
        </Col>
        <Col xs={24} md={8}>
          <Card>
            <Statistic
              title="总请求数"
              value={agents.reduce((sum, a) => sum + a.request_count, 0)}
              suffix="次"
            />
            <div style={{ marginTop: 8, fontSize: '12px', color: '#666' }}>
              总错误: {agents.reduce((sum, a) => sum + a.error_count, 0)}
            </div>
          </Card>
        </Col>
      </Row>

      <Tabs defaultActiveKey="discovery" type="card">
        <TabPane tab="智能体发现" key="discovery">
          <Card>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Row gutter={[8, 8]}>
                <Col span={4}>
                  <Input
                    name="capability"
                    placeholder="能力"
                    value={capability}
                    onChange={e => setCapability(e.target.value)}
                    autoComplete="off"
                  />
                </Col>
                <Col span={4}>
                  <Input
                    name="tags"
                    placeholder="标签 (逗号分隔)"
                    value={tags}
                    onChange={e => setTags(e.target.value)}
                    autoComplete="off"
                  />
                </Col>
                <Col span={4}>
                  <Select
                    name="statusFilter"
                    placeholder="状态筛选"
                    value={statusFilter}
                    onChange={setStatusFilter}
                    style={{ width: '100%' }}
                    allowClear
                  >
                    <Option value="">全部</Option>
                    <Option value="active">活跃</Option>
                    <Option value="inactive">离线</Option>
                    <Option value="maintenance">维护</Option>
                  </Select>
                </Col>
                <Col span={4}>
                  <Input
                    name="group"
                    placeholder="分组"
                    value={group}
                    onChange={e => setGroup(e.target.value)}
                    autoComplete="off"
                  />
                </Col>
                <Col span={4}>
                  <Input
                    name="region"
                    placeholder="区域"
                    value={region}
                    onChange={e => setRegion(e.target.value)}
                    autoComplete="off"
                  />
                </Col>
                <Col span={4}>
                  <Button type="primary" onClick={loadAgents} loading={loading}>
                    搜索智能体
                  </Button>
                </Col>
              </Row>

              <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
                {agents.map(agent => (
                  <Card
                    key={agent.agent_id}
                    size="small"
                    style={{ marginBottom: 8 }}
                  >
                    <Row justify="space-between" align="top">
                      <Col span={18}>
                        <Space direction="vertical" size="small">
                          <Space>
                            <Text strong>{agent.name}</Text>
                            <Badge
                              status={getStatusColor(agent.status)}
                              text={agent.status}
                            />
                          </Space>
                          <Text type="secondary" style={{ fontSize: '12px' }}>
                            ID: {agent.agent_id}
                          </Text>
                          <Text style={{ fontSize: '12px' }}>
                            类型: {agent.agent_type} | 版本: {agent.version}
                          </Text>
                          <Text style={{ fontSize: '12px' }}>
                            地址: {agent.host}:{agent.port}
                            {agent.endpoint}
                          </Text>
                          <Text style={{ fontSize: '12px' }}>
                            请求数: {agent.request_count} | 错误数:{' '}
                            {agent.error_count}
                          </Text>
                          {agent.avg_response_time > 0 && (
                            <Text style={{ fontSize: '12px' }}>
                              平均响应时间: {agent.avg_response_time}ms
                            </Text>
                          )}
                          {agent.tags && agent.tags.length > 0 && (
                            <Space wrap>
                              {agent.tags.map((tag, index) => (
                                <Tag key={index} size="small">
                                  {tag}
                                </Tag>
                              ))}
                            </Space>
                          )}
                        </Space>
                      </Col>
                      <Col span={6}>
                        <Space direction="vertical">
                          <Select
                            name={`agentStatus-${agent.agent_id}`}
                            placeholder="更改状态"
                            onChange={value =>
                              handleUpdateAgentStatus(agent.agent_id, value)
                            }
                            style={{ width: '100%' }}
                            disabled={loading}
                          >
                            <Option value="active">激活</Option>
                            <Option value="inactive">停用</Option>
                            <Option value="maintenance">维护</Option>
                          </Select>
                          <Button
                            danger
                            size="small"
                            onClick={() =>
                              handleDeregisterAgent(agent.agent_id)
                            }
                            loading={loading}
                          >
                            注销
                          </Button>
                        </Space>
                      </Col>
                    </Row>
                  </Card>
                ))}
              </div>

              {agents.length === 0 && !loading && (
                <div
                  style={{
                    textAlign: 'center',
                    padding: '40px',
                    color: '#999',
                  }}
                >
                  没有找到符合条件的智能体
                </div>
              )}
            </Space>
          </Card>
        </TabPane>

        <TabPane tab="注册智能体" key="register">
          <Card>
            <Form layout="vertical">
              <Row gutter={[16, 16]}>
                <Col span={12}>
                  <Form.Item label="智能体ID" required>
                    <Input
                      name="registerAgentId"
                      placeholder="智能体ID *"
                      value={registerForm.agent_id}
                      onChange={e =>
                        setRegisterForm({
                          ...registerForm,
                          agent_id: e.target.value,
                        })
                      }
                    />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item label="智能体名称" required>
                    <Input
                      name="registerAgentName"
                      placeholder="智能体名称 *"
                      value={registerForm.name}
                      onChange={e =>
                        setRegisterForm({
                          ...registerForm,
                          name: e.target.value,
                        })
                      }
                    />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item label="智能体类型">
                    <Select
                      name="registerAgentType"
                      value={registerForm.agent_type}
                      onChange={value =>
                        setRegisterForm({ ...registerForm, agent_type: value })
                      }
                      style={{ width: '100%' }}
                    >
                      <Option value="general">通用</Option>
                      <Option value="nlp">自然语言处理</Option>
                      <Option value="cv">计算机视觉</Option>
                      <Option value="ml">机器学习</Option>
                      <Option value="reasoning">推理</Option>
                    </Select>
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item label="版本">
                    <Input
                      name="registerAgentVersion"
                      placeholder="版本"
                      value={registerForm.version}
                      onChange={e =>
                        setRegisterForm({
                          ...registerForm,
                          version: e.target.value,
                        })
                      }
                    />
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <Form.Item label="主机地址">
                    <Input
                      name="registerAgentHost"
                      placeholder="主机地址"
                      value={registerForm.host}
                      onChange={e =>
                        setRegisterForm({
                          ...registerForm,
                          host: e.target.value,
                        })
                      }
                    />
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <Form.Item label="端口">
                    <Input
                      name="registerAgentPort"
                      type="number"
                      placeholder="端口"
                      value={registerForm.port}
                      onChange={e =>
                        setRegisterForm({
                          ...registerForm,
                          port: parseInt(e.target.value),
                        })
                      }
                    />
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <Form.Item label="端点路径">
                    <Input
                      name="registerAgentEndpoint"
                      placeholder="端点路径"
                      value={registerForm.endpoint}
                      onChange={e =>
                        setRegisterForm({
                          ...registerForm,
                          endpoint: e.target.value,
                        })
                      }
                    />
                  </Form.Item>
                </Col>
                <Col span={24}>
                  <Form.Item label="标签">
                    <Input
                      name="registerAgentTags"
                      placeholder="标签 (逗号分隔)"
                      onChange={e =>
                        setRegisterForm({
                          ...registerForm,
                          tags: e.target.value
                            .split(',')
                            .map(t => t.trim())
                            .filter(t => t),
                        })
                      }
                    />
                  </Form.Item>
                </Col>
              </Row>

              <Button
                type="primary"
                onClick={handleRegisterAgent}
                loading={loading}
              >
                注册智能体
              </Button>
            </Form>
          </Card>
        </TabPane>

        <TabPane tab="负载均衡" key="loadbalancer">
          <Card>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Row gutter={[16, 16]}>
                <Col span={12}>
                  <Input
                    name="loadBalancerCapability"
                    placeholder="所需能力"
                    value={loadBalancerCapability}
                    onChange={e => setLoadBalancerCapability(e.target.value)}
                    autoComplete="off"
                  />
                </Col>
                <Col span={12}>
                  <Select
                    name="loadBalancerStrategy"
                    placeholder="负载均衡策略"
                    value={loadBalancerStrategy}
                    onChange={setLoadBalancerStrategy}
                    style={{ width: '100%' }}
                  >
                    <Option value="round_robin">轮询</Option>
                    <Option value="least_connections">最少连接</Option>
                    <Option value="weighted_round_robin">加权轮询</Option>
                    <Option value="random">随机</Option>
                    <Option value="least_response_time">最短响应时间</Option>
                  </Select>
                </Col>
              </Row>

              <Button
                type="primary"
                onClick={handleSelectAgent}
                loading={loading}
              >
                选择智能体
              </Button>

              {selectedAgent && (
                <Card size="small" style={{ border: '1px solid #52c41a' }}>
                  <Title level={5}>选择的智能体</Title>
                  <Space direction="vertical" size="small">
                    <Text>名称: {selectedAgent.name}</Text>
                    <Text>ID: {selectedAgent.agent_id}</Text>
                    <Text>
                      地址: {selectedAgent.host}:{selectedAgent.port}
                      {selectedAgent.endpoint}
                    </Text>
                    <Text>
                      状态:{' '}
                      <Badge
                        status={getStatusColor(selectedAgent.status)}
                        text={selectedAgent.status}
                      />
                    </Text>
                  </Space>
                </Card>
              )}
            </Space>
          </Card>
        </TabPane>

        <TabPane tab="智能体管理" key="management">
          <Card>
            <Table
              dataSource={agents}
              rowKey="agent_id"
              pagination={{ pageSize: 10 }}
              scroll={{ x: 800 }}
              columns={[
                {
                  title: '名称',
                  dataIndex: 'name',
                  key: 'name',
                  width: 150,
                },
                {
                  title: 'ID',
                  dataIndex: 'agent_id',
                  key: 'agent_id',
                  width: 200,
                  render: text => <Text code>{text}</Text>,
                },
                {
                  title: '状态',
                  dataIndex: 'status',
                  key: 'status',
                  width: 100,
                  render: status => (
                    <Badge status={getStatusColor(status)} text={status} />
                  ),
                },
                {
                  title: '地址',
                  key: 'address',
                  width: 200,
                  render: (_, record) => `${record.host}:${record.port}`,
                },
                {
                  title: '请求数',
                  dataIndex: 'request_count',
                  key: 'request_count',
                  width: 100,
                },
                {
                  title: '错误数',
                  dataIndex: 'error_count',
                  key: 'error_count',
                  width: 100,
                  render: count => (
                    <Text style={{ color: count > 0 ? '#ff4d4f' : undefined }}>
                      {count}
                    </Text>
                  ),
                },
                {
                  title: '操作',
                  key: 'action',
                  width: 200,
                  render: (_, record) => (
                    <Space>
                      <Button
                        size="small"
                        onClick={() =>
                          handleUpdateAgentStatus(
                            record.agent_id,
                            record.status === 'active' ? 'inactive' : 'active'
                          )
                        }
                      >
                        {record.status === 'active' ? '停用' : '启用'}
                      </Button>
                      <Button
                        size="small"
                        danger
                        onClick={() => handleDeregisterAgent(record.agent_id)}
                      >
                        注销
                      </Button>
                    </Space>
                  ),
                },
              ]}
            />
          </Card>
        </TabPane>
      </Tabs>
    </div>
  )
}

export default ServiceDiscoveryManagementPage
