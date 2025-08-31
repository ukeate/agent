import React, { useState, useEffect } from 'react'
import { Card, Row, Col, Table, Badge, Progress, Statistic, Alert, Timeline, Space, Button, Select, Tag, Typography, Modal, Form, Input, Switch, Tooltip, Drawer } from 'antd'
import { 
  HeartOutlined, 
  CheckCircleOutlined, 
  ExclamationCircleOutlined, 
  CloseCircleOutlined, 
  ReloadOutlined, 
  SettingOutlined,
  BellOutlined,
  MonitorOutlined,
  ThunderboltOutlined,
  ApiOutlined,
  ClockCircleOutlined,
  WarningOutlined,
  InfoCircleOutlined,
  PauseCircleOutlined,
  PlayCircleOutlined,
  EditOutlined,
  EyeOutlined
} from '@ant-design/icons'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as ChartTooltip, ResponsiveContainer, AreaChart, Area } from 'recharts'

const { Title, Paragraph, Text } = Typography
const { Option } = Select

interface ServiceHealthMonitorPageProps {}

interface HealthCheck {
  id: string
  agentId: string
  agentName: string
  agentType: string
  endpoint: string
  status: 'healthy' | 'unhealthy' | 'warning' | 'unknown'
  lastCheck: string
  nextCheck: string
  responseTime: number
  uptime: number
  consecutiveFailures: number
  healthScore: number
  checks: {
    http: boolean
    tcp: boolean
    custom: boolean
  }
  config: {
    interval: number
    timeout: number
    retries: number
    healthEndpoint: string
    enabled: boolean
  }
  metrics: {
    avgResponseTime: number
    successRate: number
    totalChecks: number
    failedChecks: number
  }
  alerts: {
    enabled: boolean
    thresholds: {
      responseTime: number
      failureRate: number
      consecutiveFailures: number
    }
  }
}

interface HealthEvent {
  id: string
  agentName: string
  eventType: 'health_check_failed' | 'health_check_recovered' | 'timeout' | 'endpoint_changed' | 'config_updated'
  severity: 'info' | 'warning' | 'error' | 'success'
  message: string
  timestamp: string
  details?: any
}

const ServiceHealthMonitorPage: React.FC<ServiceHealthMonitorPageProps> = () => {
  const [healthChecks, setHealthChecks] = useState<HealthCheck[]>([
    {
      id: 'hc-001',
      agentId: 'agent-001',
      agentName: 'ml-processor-alpha',
      agentType: 'ML_PROCESSOR',
      endpoint: 'http://192.168.1.101:8080/health',
      status: 'healthy',
      lastCheck: '2024-08-26T14:25:00Z',
      nextCheck: '2024-08-26T14:25:30Z',
      responseTime: 42,
      uptime: 99.8,
      consecutiveFailures: 0,
      healthScore: 98,
      checks: { http: true, tcp: true, custom: true },
      config: {
        interval: 30,
        timeout: 5,
        retries: 3,
        healthEndpoint: '/health',
        enabled: true
      },
      metrics: {
        avgResponseTime: 45,
        successRate: 99.2,
        totalChecks: 2880,
        failedChecks: 23
      },
      alerts: {
        enabled: true,
        thresholds: {
          responseTime: 1000,
          failureRate: 5,
          consecutiveFailures: 3
        }
      }
    },
    {
      id: 'hc-002',
      agentId: 'agent-002',
      agentName: 'data-analyzer-beta',
      agentType: 'DATA_ANALYZER',
      endpoint: 'http://192.168.1.102:8081/health',
      status: 'healthy',
      lastCheck: '2024-08-26T14:24:30Z',
      nextCheck: '2024-08-26T14:25:00Z',
      responseTime: 125,
      uptime: 97.5,
      consecutiveFailures: 0,
      healthScore: 89,
      checks: { http: true, tcp: true, custom: false },
      config: {
        interval: 30,
        timeout: 10,
        retries: 3,
        healthEndpoint: '/health',
        enabled: true
      },
      metrics: {
        avgResponseTime: 132,
        successRate: 97.8,
        totalChecks: 2845,
        failedChecks: 63
      },
      alerts: {
        enabled: true,
        thresholds: {
          responseTime: 2000,
          failureRate: 5,
          consecutiveFailures: 3
        }
      }
    },
    {
      id: 'hc-003',
      agentId: 'agent-003',
      agentName: 'recommendation-engine',
      agentType: 'RECOMMENDER',
      endpoint: 'http://192.168.1.103:8082/health',
      status: 'unhealthy',
      lastCheck: '2024-08-26T14:20:00Z',
      nextCheck: '2024-08-26T14:21:00Z',
      responseTime: 5000,
      uptime: 85.3,
      consecutiveFailures: 5,
      healthScore: 45,
      checks: { http: false, tcp: true, custom: false },
      config: {
        interval: 60,
        timeout: 5,
        retries: 5,
        healthEndpoint: '/health',
        enabled: true
      },
      metrics: {
        avgResponseTime: 2456,
        successRate: 78.2,
        totalChecks: 1420,
        failedChecks: 310
      },
      alerts: {
        enabled: true,
        thresholds: {
          responseTime: 1000,
          failureRate: 10,
          consecutiveFailures: 3
        }
      }
    },
    {
      id: 'hc-004',
      agentId: 'agent-004',
      agentName: 'chat-assistant-gamma',
      agentType: 'CONVERSATIONAL',
      endpoint: 'http://192.168.1.104:8083/health',
      status: 'warning',
      lastCheck: '2024-08-26T14:24:45Z',
      nextCheck: '2024-08-26T14:25:15Z',
      responseTime: 890,
      uptime: 94.2,
      consecutiveFailures: 2,
      healthScore: 72,
      checks: { http: true, tcp: true, custom: true },
      config: {
        interval: 30,
        timeout: 10,
        retries: 3,
        healthEndpoint: '/api/health',
        enabled: true
      },
      metrics: {
        avgResponseTime: 756,
        successRate: 92.4,
        totalChecks: 2760,
        failedChecks: 210
      },
      alerts: {
        enabled: true,
        thresholds: {
          responseTime: 800,
          failureRate: 8,
          consecutiveFailures: 3
        }
      }
    }
  ])

  const [healthEvents] = useState<HealthEvent[]>([
    {
      id: 'evt-001',
      agentName: 'recommendation-engine',
      eventType: 'health_check_failed',
      severity: 'error',
      message: 'HTTP健康检查连续失败5次，智能体状态异常',
      timestamp: '2024-08-26T14:20:00Z',
      details: { responseCode: 500, errorMessage: 'Internal Server Error' }
    },
    {
      id: 'evt-002',
      agentName: 'chat-assistant-gamma',
      eventType: 'timeout',
      severity: 'warning',
      message: '健康检查响应时间超过阈值 (890ms > 800ms)',
      timestamp: '2024-08-26T14:18:30Z',
      details: { threshold: 800, actualTime: 890 }
    },
    {
      id: 'evt-003',
      agentName: 'ml-processor-alpha',
      eventType: 'health_check_recovered',
      severity: 'success',
      message: '智能体健康状态已恢复正常',
      timestamp: '2024-08-26T14:15:00Z'
    },
    {
      id: 'evt-004',
      agentName: 'data-analyzer-beta',
      eventType: 'config_updated',
      severity: 'info',
      message: '健康检查配置已更新：超时时间从5s调整为10s',
      timestamp: '2024-08-26T14:10:00Z',
      details: { oldTimeout: 5, newTimeout: 10 }
    },
    {
      id: 'evt-005',
      agentName: 'recommendation-engine',
      eventType: 'endpoint_changed',
      severity: 'info',
      message: '健康检查端点已更新',
      timestamp: '2024-08-26T14:05:00Z',
      details: { oldEndpoint: '/status', newEndpoint: '/health' }
    }
  ])

  const [healthTrends] = useState([
    { time: '14:00', healthy: 18, warning: 2, unhealthy: 1, unknown: 0 },
    { time: '14:05', healthy: 17, warning: 3, unhealthy: 1, unknown: 0 },
    { time: '14:10', healthy: 19, warning: 1, unhealthy: 1, unknown: 0 },
    { time: '14:15', healthy: 20, warning: 0, unhealthy: 1, unknown: 0 },
    { time: '14:20', healthy: 18, warning: 1, unhealthy: 2, unknown: 0 },
    { time: '14:25', healthy: 17, warning: 1, unhealthy: 1, unknown: 0 }
  ])

  const [responseTrends] = useState([
    { time: '14:00', avg_response: 78, max_response: 245, min_response: 23 },
    { time: '14:05', avg_response: 82, max_response: 289, min_response: 31 },
    { time: '14:10', avg_response: 65, max_response: 156, min_response: 28 },
    { time: '14:15', avg_response: 71, max_response: 203, min_response: 25 },
    { time: '14:20', avg_response: 95, max_response: 1256, min_response: 34 },
    { time: '14:25', avg_response: 89, max_response: 890, min_response: 42 }
  ])

  const [loading, setLoading] = useState(false)
  const [configModalVisible, setConfigModalVisible] = useState(false)
  const [detailDrawerVisible, setDetailDrawerVisible] = useState(false)
  const [selectedHealthCheck, setSelectedHealthCheck] = useState<HealthCheck | null>(null)
  const [filterStatus, setFilterStatus] = useState<string>('all')
  
  const [form] = Form.useForm()

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return '#52c41a'
      case 'unhealthy': return '#ff4d4f'
      case 'warning': return '#faad14'
      case 'unknown': return '#d9d9d9'
      default: return '#d9d9d9'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy': return <CheckCircleOutlined />
      case 'unhealthy': return <CloseCircleOutlined />
      case 'warning': return <ExclamationCircleOutlined />
      case 'unknown': return <WarningOutlined />
      default: return <InfoCircleOutlined />
    }
  }

  const getEventIcon = (eventType: string, severity: string) => {
    switch (severity) {
      case 'success': return <CheckCircleOutlined style={{ color: '#52c41a' }} />
      case 'warning': return <ExclamationCircleOutlined style={{ color: '#faad14' }} />
      case 'error': return <CloseCircleOutlined style={{ color: '#ff4d4f' }} />
      default: return <InfoCircleOutlined style={{ color: '#1890ff' }} />
    }
  }

  const filteredHealthChecks = healthChecks.filter(check => 
    filterStatus === 'all' || check.status === filterStatus
  )

  const healthStats = {
    total: healthChecks.length,
    healthy: healthChecks.filter(h => h.status === 'healthy').length,
    unhealthy: healthChecks.filter(h => h.status === 'unhealthy').length,
    warning: healthChecks.filter(h => h.status === 'warning').length,
    avgResponseTime: healthChecks.reduce((sum, h) => sum + h.responseTime, 0) / healthChecks.length || 0,
    avgUptime: healthChecks.reduce((sum, h) => sum + h.uptime, 0) / healthChecks.length || 0
  }

  const handleRefresh = async () => {
    setLoading(true)
    await new Promise(resolve => setTimeout(resolve, 1000))
    // 模拟数据更新
    setHealthChecks(prev => prev.map(check => ({
      ...check,
      lastCheck: new Date().toISOString(),
      responseTime: check.responseTime + Math.floor(Math.random() * 20) - 10
    })))
    setLoading(false)
  }

  const handleToggleHealthCheck = (checkId: string, enabled: boolean) => {
    setHealthChecks(prev => prev.map(check => 
      check.id === checkId 
        ? { ...check, config: { ...check.config, enabled } }
        : check
    ))
  }

  const handleViewDetails = (healthCheck: HealthCheck) => {
    setSelectedHealthCheck(healthCheck)
    setDetailDrawerVisible(true)
  }

  const columns = [
    {
      title: '服务信息',
      key: 'service',
      render: (_, check: HealthCheck) => (
        <Space>
          <Badge color={getStatusColor(check.status)} />
          <div>
            <div>
              <Text strong>{check.agentName}</Text>
              <Tag color="blue" size="small" style={{ marginLeft: 8 }}>
                {check.agentType}
              </Tag>
            </div>
            <Text type="secondary" style={{ fontSize: '12px' }}>
              {check.endpoint}
            </Text>
          </div>
        </Space>
      )
    },
    {
      title: '健康状态',
      key: 'status',
      render: (_, check: HealthCheck) => (
        <div>
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '4px' }}>
            {getStatusIcon(check.status)}
            <Text style={{ marginLeft: 4, textTransform: 'capitalize' }}>
              {check.status}
            </Text>
          </div>
          <Progress
            percent={check.healthScore}
            size="small"
            strokeColor={getStatusColor(check.status)}
            showInfo={false}
          />
          <Text type="secondary" style={{ fontSize: '12px' }}>
            健康分数: {check.healthScore}
          </Text>
        </div>
      )
    },
    {
      title: '检查类型',
      key: 'checks',
      render: (_, check: HealthCheck) => (
        <Space direction="vertical" size="small">
          <div>
            <Badge status={check.checks.http ? 'success' : 'error'} text="HTTP" />
          </div>
          <div>
            <Badge status={check.checks.tcp ? 'success' : 'error'} text="TCP" />
          </div>
          <div>
            <Badge status={check.checks.custom ? 'success' : 'default'} text="自定义" />
          </div>
        </Space>
      )
    },
    {
      title: '响应时间',
      key: 'responseTime',
      render: (_, check: HealthCheck) => (
        <div>
          <Statistic
            value={check.responseTime}
            suffix="ms"
            valueStyle={{
              fontSize: '14px',
              color: check.responseTime < 100 ? '#52c41a' : 
                    check.responseTime < 500 ? '#faad14' : '#ff4d4f'
            }}
          />
          <Text type="secondary" style={{ fontSize: '12px' }}>
            平均: {check.metrics.avgResponseTime}ms
          </Text>
        </div>
      )
    },
    {
      title: '可用性',
      key: 'uptime',
      render: (_, check: HealthCheck) => (
        <div>
          <Progress
            percent={check.uptime}
            size="small"
            status={check.uptime > 99 ? 'success' : check.uptime > 95 ? 'active' : 'exception'}
          />
          <Text type="secondary" style={{ fontSize: '12px' }}>
            成功率: {check.metrics.successRate}%
          </Text>
        </div>
      )
    },
    {
      title: '连续失败',
      dataIndex: 'consecutiveFailures',
      key: 'consecutiveFailures',
      render: (failures: number) => (
        <Tag color={failures === 0 ? 'success' : failures < 3 ? 'warning' : 'error'}>
          {failures} 次
        </Tag>
      )
    },
    {
      title: '最后检查',
      key: 'lastCheck',
      render: (_, check: HealthCheck) => (
        <div>
          <Text style={{ fontSize: '12px' }}>
            {new Date(check.lastCheck).toLocaleString()}
          </Text>
          <br />
          <Text type="secondary" style={{ fontSize: '11px' }}>
            下次: {new Date(check.nextCheck).toLocaleTimeString()}
          </Text>
        </div>
      )
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, check: HealthCheck) => (
        <Space size="small">
          <Tooltip title="查看详情">
            <Button size="small" icon={<EyeOutlined />} onClick={() => handleViewDetails(check)} />
          </Tooltip>
          <Tooltip title="编辑配置">
            <Button size="small" icon={<EditOutlined />} />
          </Tooltip>
          <Tooltip title={check.config.enabled ? '暂停检查' : '启动检查'}>
            <Button
              size="small"
              icon={check.config.enabled ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
              onClick={() => handleToggleHealthCheck(check.id, !check.config.enabled)}
            />
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
            <HeartOutlined /> 服务健康监控
          </Title>
          <Paragraph>
            实时监控所有注册智能体的健康状态，提供多维度健康检查和告警机制。
          </Paragraph>
        </div>

        {/* 健康状态统计 */}
        <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
          <Col xs={24} sm={6}>
            <Card>
              <Statistic
                title="总服务数"
                value={healthStats.total}
                prefix={<ApiOutlined />}
                valueStyle={{ color: '#1890ff' }}
              />
            </Card>
          </Col>
          <Col xs={24} sm={6}>
            <Card>
              <Statistic
                title="健康服务"
                value={healthStats.healthy}
                prefix={<CheckCircleOutlined />}
                valueStyle={{ color: '#52c41a' }}
              />
            </Card>
          </Col>
          <Col xs={24} sm={6}>
            <Card>
              <Statistic
                title="异常服务"
                value={healthStats.unhealthy}
                prefix={<CloseCircleOutlined />}
                valueStyle={{ color: '#ff4d4f' }}
              />
            </Card>
          </Col>
          <Col xs={24} sm={6}>
            <Card>
              <Statistic
                title="平均响应时间"
                value={healthStats.avgResponseTime}
                suffix="ms"
                precision={0}
                prefix={<ThunderboltOutlined />}
                valueStyle={{ color: '#faad14' }}
              />
            </Card>
          </Col>
        </Row>

        {/* 系统状态告警 */}
        <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
          <Col xs={24}>
            {healthStats.unhealthy > 0 && (
              <Alert
                message={`检测到 ${healthStats.unhealthy} 个服务异常`}
                description="recommendation-engine 连续健康检查失败，请及时处理。"
                type="error"
                icon={<ExclamationCircleOutlined />}
                showIcon
                closable
                style={{ marginBottom: '16px' }}
              />
            )}
            {healthStats.warning > 0 && (
              <Alert
                message={`${healthStats.warning} 个服务处于警告状态`}
                description="chat-assistant-gamma 响应时间超过阈值，建议检查系统资源。"
                type="warning"
                icon={<WarningOutlined />}
                showIcon
                closable
              />
            )}
          </Col>
        </Row>

        <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
          {/* 健康状态趋势 */}
          <Col xs={24} lg={12}>
            <Card title="健康状态趋势" extra={<MonitorOutlined />}>
              <ResponsiveContainer width="100%" height={250}>
                <AreaChart data={healthTrends}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis />
                  <ChartTooltip />
                  <Area type="monotone" dataKey="healthy" stackId="1" stroke="#52c41a" fill="#52c41a" fillOpacity={0.6} />
                  <Area type="monotone" dataKey="warning" stackId="1" stroke="#faad14" fill="#faad14" fillOpacity={0.6} />
                  <Area type="monotone" dataKey="unhealthy" stackId="1" stroke="#ff4d4f" fill="#ff4d4f" fillOpacity={0.6} />
                  <Area type="monotone" dataKey="unknown" stackId="1" stroke="#d9d9d9" fill="#d9d9d9" fillOpacity={0.6} />
                </AreaChart>
              </ResponsiveContainer>
            </Card>
          </Col>

          {/* 响应时间趋势 */}
          <Col xs={24} lg={12}>
            <Card title="响应时间趋势" extra={<ThunderboltOutlined />}>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={responseTrends}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis />
                  <ChartTooltip />
                  <Line type="monotone" dataKey="avg_response" stroke="#1890ff" strokeWidth={2} name="平均响应时间" />
                  <Line type="monotone" dataKey="max_response" stroke="#ff4d4f" strokeWidth={1} strokeDasharray="5 5" name="最大响应时间" />
                  <Line type="monotone" dataKey="min_response" stroke="#52c41a" strokeWidth={1} strokeDasharray="5 5" name="最小响应时间" />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          </Col>
        </Row>

        <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
          {/* 健康检查列表 */}
          <Col xs={24} lg={16}>
            <Card
              title="服务健康检查"
              extra={
                <Space>
                  <Select
                    value={filterStatus}
                    onChange={setFilterStatus}
                    style={{ width: 120 }}
                  >
                    <Option value="all">全部状态</Option>
                    <Option value="healthy">健康</Option>
                    <Option value="unhealthy">异常</Option>
                    <Option value="warning">警告</Option>
                    <Option value="unknown">未知</Option>
                  </Select>
                  <Button icon={<ReloadOutlined />} loading={loading} onClick={handleRefresh}>
                    刷新
                  </Button>
                  <Button icon={<SettingOutlined />} onClick={() => setConfigModalVisible(true)}>
                    配置
                  </Button>
                </Space>
              }
            >
              <Table
                columns={columns}
                dataSource={filteredHealthChecks}
                rowKey="id"
                size="small"
                pagination={{
                  pageSize: 10,
                  showSizeChanger: true,
                  showTotal: (total, range) => `第 ${range[0]}-${range[1]} 条，共 ${total} 条记录`
                }}
              />
            </Card>
          </Col>

          {/* 最近事件 */}
          <Col xs={24} lg={8}>
            <Card title="最近事件" extra={<ClockCircleOutlined />}>
              <Timeline size="small">
                {healthEvents.slice(0, 8).map((event) => (
                  <Timeline.Item
                    key={event.id}
                    dot={getEventIcon(event.eventType, event.severity)}
                  >
                    <div>
                      <div style={{ marginBottom: '4px' }}>
                        <Text strong style={{ fontSize: '13px' }}>
                          {event.agentName}
                        </Text>
                        <Tag size="small" color={event.severity === 'error' ? 'red' : event.severity === 'warning' ? 'orange' : event.severity === 'success' ? 'green' : 'blue'}>
                          {event.severity.toUpperCase()}
                        </Tag>
                      </div>
                      <div style={{ fontSize: '12px', marginBottom: '2px' }}>
                        {event.message}
                      </div>
                      <Text type="secondary" style={{ fontSize: '11px' }}>
                        {new Date(event.timestamp).toLocaleString()}
                      </Text>
                    </div>
                  </Timeline.Item>
                ))}
              </Timeline>
            </Card>
          </Col>
        </Row>

        {/* 健康检查配置Modal */}
        <Modal
          title="全局健康检查配置"
          visible={configModalVisible}
          onCancel={() => setConfigModalVisible(false)}
          footer={null}
          width={600}
        >
          <Form layout="vertical">
            <Row gutter={16}>
              <Col span={12}>
                <Form.Item label="默认检查间隔 (秒)">
                  <Input placeholder="30" />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item label="默认超时时间 (秒)">
                  <Input placeholder="5" />
                </Form.Item>
              </Col>
            </Row>
            <Row gutter={16}>
              <Col span={12}>
                <Form.Item label="默认重试次数">
                  <Input placeholder="3" />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item>
                  <div style={{ marginTop: '30px' }}>
                    <Switch /> 启用全局告警
                  </div>
                </Form.Item>
              </Col>
            </Row>
            <Form.Item>
              <Space>
                <Button type="primary" icon={<SaveOutlined />}>
                  保存配置
                </Button>
                <Button>
                  取消
                </Button>
              </Space>
            </Form.Item>
          </Form>
        </Modal>

        {/* 健康检查详情Drawer */}
        <Drawer
          title="健康检查详情"
          visible={detailDrawerVisible}
          onClose={() => setDetailDrawerVisible(false)}
          width={600}
        >
          {selectedHealthCheck && (
            <div>
              <Alert
                message={`当前状态: ${selectedHealthCheck.status.toUpperCase()}`}
                type={selectedHealthCheck.status === 'healthy' ? 'success' : 
                      selectedHealthCheck.status === 'warning' ? 'warning' : 'error'}
                showIcon
                style={{ marginBottom: '16px' }}
              />

              <Title level={4}>基本信息</Title>
              <Row gutter={16} style={{ marginBottom: '16px' }}>
                <Col span={12}>
                  <Text type="secondary">智能体名称</Text>
                  <div>{selectedHealthCheck.agentName}</div>
                </Col>
                <Col span={12}>
                  <Text type="secondary">智能体类型</Text>
                  <div>{selectedHealthCheck.agentType}</div>
                </Col>
                <Col span={24}>
                  <Text type="secondary">健康检查端点</Text>
                  <div><Text code>{selectedHealthCheck.endpoint}</Text></div>
                </Col>
              </Row>

              <Title level={4}>检查配置</Title>
              <Row gutter={16} style={{ marginBottom: '16px' }}>
                <Col span={8}>
                  <Text type="secondary">检查间隔</Text>
                  <div>{selectedHealthCheck.config.interval}秒</div>
                </Col>
                <Col span={8}>
                  <Text type="secondary">超时时间</Text>
                  <div>{selectedHealthCheck.config.timeout}秒</div>
                </Col>
                <Col span={8}>
                  <Text type="secondary">重试次数</Text>
                  <div>{selectedHealthCheck.config.retries}次</div>
                </Col>
              </Row>

              <Title level={4}>性能指标</Title>
              <Row gutter={16} style={{ marginBottom: '16px' }}>
                <Col span={12}>
                  <Card size="small">
                    <Statistic title="当前响应时间" value={selectedHealthCheck.responseTime} suffix="ms" />
                  </Card>
                </Col>
                <Col span={12}>
                  <Card size="small">
                    <Statistic title="平均响应时间" value={selectedHealthCheck.metrics.avgResponseTime} suffix="ms" />
                  </Card>
                </Col>
                <Col span={12}>
                  <Card size="small">
                    <Statistic title="可用性" value={selectedHealthCheck.uptime} suffix="%" precision={1} />
                  </Card>
                </Col>
                <Col span={12}>
                  <Card size="small">
                    <Statistic title="成功率" value={selectedHealthCheck.metrics.successRate} suffix="%" precision={1} />
                  </Card>
                </Col>
              </Row>

              <Title level={4}>检查统计</Title>
              <Row gutter={16}>
                <Col span={12}>
                  <Text type="secondary">总检查次数: </Text>
                  <Text>{selectedHealthCheck.metrics.totalChecks}</Text>
                </Col>
                <Col span={12}>
                  <Text type="secondary">失败次数: </Text>
                  <Text>{selectedHealthCheck.metrics.failedChecks}</Text>
                </Col>
                <Col span={12}>
                  <Text type="secondary">连续失败: </Text>
                  <Text>{selectedHealthCheck.consecutiveFailures}</Text>
                </Col>
                <Col span={12}>
                  <Text type="secondary">健康分数: </Text>
                  <Text>{selectedHealthCheck.healthScore}</Text>
                </Col>
              </Row>
            </div>
          )}
        </Drawer>
      </div>
    </div>
  )
}

export default ServiceHealthMonitorPage