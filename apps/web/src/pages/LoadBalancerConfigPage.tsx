import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Form,
  Select,
  Switch,
  InputNumber,
  Button,
  Space,
  Typography,
  Divider,
  Alert,
  Table,
  Tag,
  Tooltip,
  Modal,
  Progress,
  Statistic,
  Timeline,
  Spin,
} from 'antd'
import { logger } from '../utils/logger'
import {
  SettingOutlined,
  ThunderboltOutlined,
  SaveOutlined,
  ReloadOutlined,
  ExperimentOutlined,
  BarChartOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  InfoCircleOutlined,
  PlayCircleOutlined,
  GlobalOutlined,
  ApiOutlined,
  MonitorOutlined,
  ClockCircleOutlined,
} from '@ant-design/icons'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as ChartTooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts'

const { Title, Paragraph, Text } = Typography
const { Option } = Select

interface LoadBalancerConfigPageProps {}

interface LoadBalancerConfig {
  globalStrategy: string
  healthCheckInterval: number
  connectionTimeout: number
  retryAttempts: number
  circuitBreakerEnabled: boolean
  circuitBreakerThreshold: number
  circuitBreakerTimeout: number
  stickySession: boolean
  sessionTimeout: number
  adaptiveLoadBalancing: boolean
  geographicRouting: boolean
  capabilityWeighting: boolean
  responseTimeWeighting: number
  connectionWeighting: number
  cpuWeighting: number
  memoryWeighting: number
}

interface StrategyMetrics {
  strategy: string
  requestsHandled: number
  averageResponseTime: number
  successRate: number
  errorRate: number
  activeConnections: number
  throughputPerSecond: number
}

interface AgentLoad {
  agentId: string
  agentName: string
  currentLoad: number
  connectionCount: number
  responseTime: number
  cpuUsage: number
  memoryUsage: number
  healthScore: number
  region: string
  capabilities: string[]
}

const LoadBalancerConfigPage: React.FC<LoadBalancerConfigPageProps> = () => {
  const [config, setConfig] = useState<LoadBalancerConfig | null>(null)
  const [strategyMetrics, setStrategyMetrics] = useState<StrategyMetrics[]>([])
  const [agentLoads, setAgentLoads] = useState<AgentLoad[]>([])
  const [performanceData, setPerformanceData] = useState<any[]>([])

  const [saving, setSaving] = useState(false)
  const [testModalVisible, setTestModalVisible] = useState(false)
  const [testResults, setTestResults] = useState<any[]>([])

  const [form] = Form.useForm()

  useEffect(() => {
    if (config) form.setFieldsValue(config)
  }, [config, form])

  useEffect(() => {
    loadAll()
  }, [])

  const loadAll = async () => {
    await Promise.all([loadConfig(), loadMetrics(), loadAgents()])
  }

  const loadConfig = async () => {
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/load-balancer/config'))
      const data = await res.json()
      if (data.success) {
        setConfig(data.config)
      }
    } catch (e) {
      logger.error('加载负载均衡配置失败:', e)
    }
  }

  const loadMetrics = async () => {
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/load-balancer/metrics'))
      const data = await res.json()
      if (data.success) {
        setStrategyMetrics(data.metrics || [])
        setPerformanceData(data.performance || [])
      }
    } catch (e) {
      logger.error('加载负载均衡指标失败:', e)
    }
  }

  const loadAgents = async () => {
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/cluster/agents'))
      const data = await res.json()
      if (data.agents) {
        setAgentLoads(
          data.agents.map((a: any) => ({
            agentId: a.agent_id,
            agentName: a.name,
            currentLoad: a.current_load ?? a.resource_usage?.active_tasks ?? 0,
            connectionCount:
              a.resource_usage?.active_tasks ?? a.current_load ?? 0,
            responseTime: a.resource_usage?.avg_response_time ?? 0,
            cpuUsage: a.resource_usage?.cpu_usage_percent ?? 0,
            memoryUsage: a.resource_usage?.memory_usage_percent ?? 0,
            healthScore: Math.max(
              0,
              100 - (a.resource_usage?.error_rate ?? 0) * 100
            ),
            region: a.labels?.region || 'default',
            capabilities: a.capabilities || [],
          }))
        )
      }
    } catch (e) {
      logger.error('加载智能体列表失败:', e)
    }
  }

  const loadBalanceStrategies = [
    {
      value: 'round_robin',
      label: '轮询 (Round Robin)',
      description: '按顺序轮流分配请求到每个智能体',
    },
    {
      value: 'least_connections',
      label: '最少连接 (Least Connections)',
      description: '分配到当前连接数最少的智能体',
    },
    {
      value: 'weighted_round_robin',
      label: '加权轮询 (Weighted Round Robin)',
      description: '根据智能体权重进行轮询分配',
    },
    {
      value: 'capability_based',
      label: '能力优先 (Capability Based)',
      description: '根据智能体能力匹配度进行分配',
    },
    {
      value: 'geographic',
      label: '地理位置优先 (Geographic)',
      description: '优先分配到地理位置最近的智能体',
    },
    {
      value: 'response_time',
      label: '响应时间优先 (Response Time)',
      description: '优先分配到响应时间最短的智能体',
    },
  ]

  const handleSaveConfig = async (values: LoadBalancerConfig) => {
    try {
      setSaving(true)
      const res = await apiFetch(buildApiUrl('/api/v1/load-balancer/config'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(values),
      })
      const data = await res.json()
      if (!data.success) {
        throw new Error(data.message || '配置保存失败')
      }
      if (data.success) {
        setConfig(data.config)
      }
      Modal.success({
        title: '配置保存成功',
        content: '负载均衡器配置已更新，新配置将在下次请求时生效。',
      })
    } catch (error) {
      Modal.error({
        title: '配置保存失败',
        content: '保存负载均衡器配置时出现错误，请重试。',
      })
    } finally {
      setSaving(false)
    }
  }

  const handleTestStrategy = async (strategy: string) => {
    try {
      setTestModalVisible(true)

      const res = await apiFetch(
        buildApiUrl('/api/v1/load-balancer/strategy/test'),
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ strategy }),
        }
      )
      const data = await res.json()
      setTestResults(prev => [data, ...prev.slice(0, 4)])
    } catch (error) {
      logger.error('策略测试失败:', error)
    }
  }

  const getLoadColor = (load: number) => {
    if (load < 50) return '#52c41a'
    if (load < 80) return '#faad14'
    return '#ff4d4f'
  }

  const getHealthColor = (health: number) => {
    if (health >= 90) return '#52c41a'
    if (health >= 70) return '#faad14'
    return '#ff4d4f'
  }

  const strategyColumns = [
    {
      title: '策略',
      dataIndex: 'strategy',
      key: 'strategy',
      render: (strategy: string) => {
        const strategyInfo = loadBalanceStrategies.find(
          s => s.value === strategy
        )
        return (
          <div>
            <Text strong>{strategyInfo?.label || strategy}</Text>
            <br />
            <Text type="secondary" style={{ fontSize: '12px' }}>
              {strategyInfo?.description}
            </Text>
          </div>
        )
      },
    },
    {
      title: '处理请求数',
      dataIndex: 'requestsHandled',
      key: 'requestsHandled',
      render: (value: number) => (
        <Statistic value={value} valueStyle={{ fontSize: '14px' }} />
      ),
    },
    {
      title: '平均响应时间',
      dataIndex: 'averageResponseTime',
      key: 'averageResponseTime',
      render: (value: number) => (
        <Statistic
          value={value}
          suffix="ms"
          valueStyle={{
            fontSize: '14px',
            color: value < 50 ? '#52c41a' : value < 100 ? '#faad14' : '#ff4d4f',
          }}
        />
      ),
    },
    {
      title: '成功率',
      dataIndex: 'successRate',
      key: 'successRate',
      render: (value: number) => (
        <div>
          <Progress
            percent={value}
            size="small"
            status={
              value > 98 ? 'success' : value > 95 ? 'active' : 'exception'
            }
          />
          <Text style={{ fontSize: '12px' }}>{value}%</Text>
        </div>
      ),
    },
    {
      title: '吞吐量',
      dataIndex: 'throughputPerSecond',
      key: 'throughputPerSecond',
      render: (value: number) => (
        <Statistic
          value={value}
          suffix="/s"
          valueStyle={{ fontSize: '14px' }}
        />
      ),
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record: StrategyMetrics) => (
        <Space>
          <Button
            size="small"
            icon={<ExperimentOutlined />}
            onClick={() => handleTestStrategy(record.strategy)}
          >
            测试
          </Button>
          <Button
            size="small"
            type="primary"
            onClick={() => {
              form.setFieldValue('globalStrategy', record.strategy)
              setConfig(prev => ({ ...prev, globalStrategy: record.strategy }))
            }}
          >
            设为默认
          </Button>
        </Space>
      ),
    },
  ]

  const agentLoadColumns = [
    {
      title: '智能体',
      key: 'agent',
      render: (_, agent: AgentLoad) => (
        <div>
          <Text strong>{agent.agentName}</Text>
          <br />
          <Tag color="blue" size="small">
            {agent.region}
          </Tag>
          <div>
            {agent.capabilities.slice(0, 2).map(cap => (
              <Tag key={cap} size="small">
                {cap}
              </Tag>
            ))}
          </div>
        </div>
      ),
    },
    {
      title: '当前负载',
      dataIndex: 'currentLoad',
      key: 'currentLoad',
      render: (load: number) => (
        <div>
          <Progress
            percent={load}
            size="small"
            strokeColor={getLoadColor(load)}
          />
          <Text style={{ fontSize: '12px' }}>{load}%</Text>
        </div>
      ),
    },
    {
      title: '连接数',
      dataIndex: 'connectionCount',
      key: 'connectionCount',
      render: (value: number) => (
        <Statistic value={value} valueStyle={{ fontSize: '14px' }} />
      ),
    },
    {
      title: '响应时间',
      dataIndex: 'responseTime',
      key: 'responseTime',
      render: (value: number) => (
        <Statistic
          value={value}
          suffix="ms"
          valueStyle={{
            fontSize: '14px',
            color: value < 50 ? '#52c41a' : '#faad14',
          }}
        />
      ),
    },
    {
      title: '健康分数',
      dataIndex: 'healthScore',
      key: 'healthScore',
      render: (score: number) => (
        <div>
          <Progress
            percent={score}
            size="small"
            strokeColor={getHealthColor(score)}
          />
          <Text style={{ fontSize: '12px' }}>{score}</Text>
        </div>
      ),
    },
  ]

  const strategyDistribution = strategyMetrics.map(metric => ({
    name:
      loadBalanceStrategies.find(s => s.value === metric.strategy)?.label ||
      metric.strategy,
    value: metric.requestsHandled,
    color:
      config && metric.strategy === config.globalStrategy
        ? '#1890ff'
        : '#d9d9d9',
  }))

  if (!config) {
    return (
      <div style={{ padding: 24, textAlign: 'center' }}>
        <Spin />
      </div>
    )
  }

  return (
    <div style={{ padding: '24px', background: '#f5f5f5', minHeight: '100vh' }}>
      <div style={{ maxWidth: '1400px', margin: '0 auto' }}>
        {/* 页面标题 */}
        <div style={{ marginBottom: '24px' }}>
          <Title level={2}>
            <ThunderboltOutlined /> 负载均衡器配置
          </Title>
          <Paragraph>
            配置智能代理服务发现系统的负载均衡策略、健康检查和性能参数，优化服务分发效率。
          </Paragraph>
        </div>

        {/* 当前状态概览 */}
        <Alert
          message={`当前负载均衡策略: ${config.globalStrategy}`}
          description={`平均响应时间 ${strategyMetrics[0]?.averageResponseTime ?? 0}ms，成功率 ${strategyMetrics[0]?.successRate ?? 0}%`}
          type="success"
          icon={<CheckCircleOutlined />}
          showIcon
          style={{ marginBottom: '24px' }}
        />

        <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
          {/* 策略性能对比 */}
          <Col xs={24} lg={16}>
            <Card title="策略性能趋势" extra={<BarChartOutlined />}>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={performanceData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis />
                  <ChartTooltip />
                  <Line
                    type="monotone"
                    dataKey="capability_based"
                    stroke="#1890ff"
                    strokeWidth={2}
                    name="能力优先"
                  />
                  <Line
                    type="monotone"
                    dataKey="round_robin"
                    stroke="#52c41a"
                    strokeWidth={2}
                    name="轮询"
                  />
                  <Line
                    type="monotone"
                    dataKey="least_connections"
                    stroke="#faad14"
                    strokeWidth={2}
                    name="最少连接"
                  />
                  <Line
                    type="monotone"
                    dataKey="response_time"
                    stroke="#ff7300"
                    strokeWidth={2}
                    name="响应时间优先"
                  />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          </Col>

          {/* 策略分布 */}
          <Col xs={24} lg={8}>
            <Card title="请求分布统计" extra={<GlobalOutlined />}>
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie
                    data={strategyDistribution}
                    cx="50%"
                    cy="50%"
                    innerRadius={30}
                    outerRadius={80}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {strategyDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <ChartTooltip />
                </PieChart>
              </ResponsiveContainer>
              <div style={{ textAlign: 'center', marginTop: '16px' }}>
                <Text type="secondary">当前主要使用能力优先策略</Text>
              </div>
            </Card>
          </Col>
        </Row>

        <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
          {/* 配置表单 */}
          <Col xs={24} lg={12}>
            <Card title="全局配置" extra={<SettingOutlined />}>
              <Form
                form={form}
                layout="vertical"
                onFinish={handleSaveConfig}
                initialValues={config}
              >
                <Form.Item
                  name="globalStrategy"
                  label="默认负载均衡策略"
                  rules={[{ required: true, message: '请选择默认策略' }]}
                >
                  <Select>
                    {loadBalanceStrategies.map(strategy => (
                      <Option key={strategy.value} value={strategy.value}>
                        <div>
                          <div>{strategy.label}</div>
                          <Text type="secondary" style={{ fontSize: '12px' }}>
                            {strategy.description}
                          </Text>
                        </div>
                      </Option>
                    ))}
                  </Select>
                </Form.Item>

                <Row gutter={16}>
                  <Col span={12}>
                    <Form.Item
                      name="healthCheckInterval"
                      label="健康检查间隔 (秒)"
                      rules={[
                        {
                          required: true,
                          min: 10,
                          max: 300,
                          message: '间隔时间应在10-300秒之间',
                        },
                      ]}
                    >
                      <InputNumber
                        min={10}
                        max={300}
                        style={{ width: '100%' }}
                      />
                    </Form.Item>
                  </Col>
                  <Col span={12}>
                    <Form.Item
                      name="connectionTimeout"
                      label="连接超时 (毫秒)"
                      rules={[
                        {
                          required: true,
                          min: 1000,
                          max: 30000,
                          message: '超时时间应在1-30秒之间',
                        },
                      ]}
                    >
                      <InputNumber
                        min={1000}
                        max={30000}
                        style={{ width: '100%' }}
                      />
                    </Form.Item>
                  </Col>
                </Row>

                <Row gutter={16}>
                  <Col span={12}>
                    <Form.Item
                      name="retryAttempts"
                      label="重试次数"
                      rules={[
                        {
                          required: true,
                          min: 0,
                          max: 10,
                          message: '重试次数应在0-10之间',
                        },
                      ]}
                    >
                      <InputNumber min={0} max={10} style={{ width: '100%' }} />
                    </Form.Item>
                  </Col>
                  <Col span={12}>
                    <Form.Item
                      name="circuitBreakerEnabled"
                      valuePropName="checked"
                    >
                      <div style={{ marginTop: '30px' }}>
                        <Switch /> 启用熔断器
                      </div>
                    </Form.Item>
                  </Col>
                </Row>

                <Divider>高级选项</Divider>

                <Row gutter={16}>
                  <Col span={12}>
                    <Form.Item
                      name="adaptiveLoadBalancing"
                      valuePropName="checked"
                    >
                      <Switch /> 自适应负载均衡
                    </Form.Item>
                  </Col>
                  <Col span={12}>
                    <Form.Item name="geographicRouting" valuePropName="checked">
                      <Switch /> 地理位置路由
                    </Form.Item>
                  </Col>
                </Row>

                <Row gutter={16}>
                  <Col span={12}>
                    <Form.Item name="stickySession" valuePropName="checked">
                      <Switch /> 会话保持
                    </Form.Item>
                  </Col>
                  <Col span={12}>
                    <Form.Item
                      name="capabilityWeighting"
                      valuePropName="checked"
                    >
                      <Switch /> 能力权重评估
                    </Form.Item>
                  </Col>
                </Row>

                <Divider>权重配置 (%)</Divider>

                <Row gutter={16}>
                  <Col span={12}>
                    <Form.Item
                      name="responseTimeWeighting"
                      label="响应时间权重"
                      rules={[{ required: true, min: 0, max: 100 }]}
                    >
                      <InputNumber
                        min={0}
                        max={100}
                        style={{ width: '100%' }}
                      />
                    </Form.Item>
                  </Col>
                  <Col span={12}>
                    <Form.Item
                      name="connectionWeighting"
                      label="连接数权重"
                      rules={[{ required: true, min: 0, max: 100 }]}
                    >
                      <InputNumber
                        min={0}
                        max={100}
                        style={{ width: '100%' }}
                      />
                    </Form.Item>
                  </Col>
                </Row>

                <Row gutter={16}>
                  <Col span={12}>
                    <Form.Item
                      name="cpuWeighting"
                      label="CPU使用率权重"
                      rules={[{ required: true, min: 0, max: 100 }]}
                    >
                      <InputNumber
                        min={0}
                        max={100}
                        style={{ width: '100%' }}
                      />
                    </Form.Item>
                  </Col>
                  <Col span={12}>
                    <Form.Item
                      name="memoryWeighting"
                      label="内存使用率权重"
                      rules={[{ required: true, min: 0, max: 100 }]}
                    >
                      <InputNumber
                        min={0}
                        max={100}
                        style={{ width: '100%' }}
                      />
                    </Form.Item>
                  </Col>
                </Row>

                <Form.Item>
                  <Space>
                    <Button
                      type="primary"
                      htmlType="submit"
                      icon={<SaveOutlined />}
                      loading={saving}
                    >
                      保存配置
                    </Button>
                    <Button icon={<ReloadOutlined />}>重置为默认</Button>
                  </Space>
                </Form.Item>
              </Form>
            </Card>
          </Col>

          {/* 实时监控 */}
          <Col xs={24} lg={12}>
            <Card
              title="智能体负载监控"
              extra={<MonitorOutlined />}
              style={{ height: '100%' }}
            >
              <Table
                columns={agentLoadColumns}
                dataSource={agentLoads}
                rowKey="agentId"
                size="small"
                pagination={false}
              />
            </Card>
          </Col>
        </Row>

        {/* 策略性能对比表 */}
        <Card title="负载均衡策略对比" extra={<ApiOutlined />}>
          <Table
            columns={strategyColumns}
            dataSource={strategyMetrics}
            rowKey="strategy"
            pagination={false}
          />
        </Card>

        {/* 策略测试结果Modal */}
        <Modal
          title="策略测试结果"
          visible={testModalVisible}
          onCancel={() => setTestModalVisible(false)}
          footer={null}
          width={800}
        >
          {testResults.length > 0 && (
            <Timeline>
              {testResults.map((result, index) => (
                <Timeline.Item
                  key={index}
                  dot={<ExperimentOutlined />}
                  color="blue"
                >
                  <div>
                    <Text strong>
                      {
                        loadBalanceStrategies.find(
                          s => s.value === result.strategy
                        )?.label
                      }{' '}
                      测试
                    </Text>
                    <Tag color="blue" style={{ marginLeft: 8 }}>
                      {new Date(result.timestamp).toLocaleTimeString()}
                    </Tag>
                    <br />
                    <Row gutter={16} style={{ marginTop: '8px' }}>
                      <Col span={6}>
                        <Statistic
                          title="平均响应时间"
                          value={result.averageResponseTime}
                          suffix="ms"
                          valueStyle={{ fontSize: '14px' }}
                        />
                      </Col>
                      <Col span={6}>
                        <Statistic
                          title="成功率"
                          value={result.successRate}
                          suffix="%"
                          precision={1}
                          valueStyle={{ fontSize: '14px' }}
                        />
                      </Col>
                      <Col span={6}>
                        <Statistic
                          title="吞吐量"
                          value={result.throughput}
                          suffix="/s"
                          valueStyle={{ fontSize: '14px' }}
                        />
                      </Col>
                      <Col span={6}>
                        <Statistic
                          title="总请求数"
                          value={result.requests}
                          valueStyle={{ fontSize: '14px' }}
                        />
                      </Col>
                    </Row>
                  </div>
                </Timeline.Item>
              ))}
            </Timeline>
          )}
        </Modal>
      </div>
    </div>
  )
}

export default LoadBalancerConfigPage
