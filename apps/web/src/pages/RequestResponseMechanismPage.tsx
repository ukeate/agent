import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Table,
  Button,
  Form,
  Input,
  Select,
  Space,
  Typography,
  Alert,
  Tag,
  Modal,
  Tabs,
  Badge,
  Progress,
  Statistic,
  Timeline,
  Switch,
  Slider,
  Tooltip,
  Divider,
  notification,
  Radio,
  Drawer,
  Steps,
  Empty,
  Descriptions
} from 'antd'
import {
  SwapOutlined,
  SendOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  LoadingOutlined,
  ReloadOutlined,
  SettingOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  EyeOutlined,
  DeleteOutlined,
  ExclamationCircleOutlined,
  ThunderboltOutlined,
  ApiOutlined,
  MonitorOutlined,
  NetworkOutlined,
  UserOutlined,
  TeamOutlined,
  SearchOutlined,
  FilterOutlined,
  LineChartOutlined,
  WarningOutlined,
  RocketOutlined,
  SyncOutlined
} from '@ant-design/icons'

const { Title, Text, Paragraph } = Typography
const { Option } = Select
const { TextArea } = Input
const { TabPane } = Tabs
const { Step } = Steps

interface RequestResponsePair {
  id: string
  correlationId: string
  requestId: string
  responseId?: string
  requester: string
  responder: string
  subject: string
  method: string
  requestPayload: any
  responsePayload?: any
  status: 'pending' | 'completed' | 'timeout' | 'failed'
  requestTime: string
  responseTime?: string
  latency?: number
  timeout: number
  retryCount: number
  maxRetries: number
  priority: number
  tags: string[]
}

interface RequestPattern {
  id: string
  name: string
  description: string
  subject: string
  method: string
  expectedResponseTime: number
  timeout: number
  retryPolicy: 'none' | 'fixed' | 'exponential'
  maxRetries: number
  circuitBreakerEnabled: boolean
  rateLimitEnabled: boolean
  requestCount: number
  successRate: number
  averageLatency: number
  enabled: boolean
}

interface ResponseHandler {
  id: string
  agentId: string
  agentName: string
  handlerSubjects: string[]
  status: 'online' | 'offline' | 'busy'
  pendingRequests: number
  processedRequests: number
  averageResponseTime: number
  successRate: number
  lastSeen: string
  queueSize: number
  maxQueueSize: number
}

interface RequestMetrics {
  totalRequests: number
  pendingRequests: number
  completedRequests: number
  timeoutRequests: number
  failedRequests: number
  averageLatency: number
  p95Latency: number
  p99Latency: number
  requestsPerSecond: number
  successRate: number
}

const RequestResponseMechanismPage: React.FC = () => {
  const [form] = Form.useForm()
  const [activeTab, setActiveTab] = useState('monitor')
  const [loading, setLoading] = useState(false)
  const [testModalVisible, setTestModalVisible] = useState(false)
  const [patternModalVisible, setPatternModalVisible] = useState(false)
  const [metricsDrawerVisible, setMetricsDrawerVisible] = useState(false)
  const [selectedRequest, setSelectedRequest] = useState<RequestResponsePair | null>(null)
  
  const [metrics, setMetrics] = useState<RequestMetrics>({
    totalRequests: 23847,
    pendingRequests: 15,
    completedRequests: 22963,
    timeoutRequests: 645,
    failedRequests: 239,
    averageLatency: 156.8,
    p95Latency: 450.2,
    p99Latency: 892.1,
    requestsPerSecond: 78.5,
    successRate: 96.3
  })

  const [requestPairs, setRequestPairs] = useState<RequestResponsePair[]>([
    {
      id: 'req-001',
      correlationId: 'corr-abc123',
      requestId: 'request-001',
      responseId: 'response-001',
      requester: 'client-agent-01',
      responder: 'service-agent-01',
      subject: 'agents.service.user_info',
      method: 'GET_USER_INFO',
      requestPayload: { userId: 'user123', includeDetails: true },
      responsePayload: { userId: 'user123', name: 'John Doe', email: 'john@example.com', details: {} },
      status: 'completed',
      requestTime: '2025-08-26 12:45:30',
      responseTime: '2025-08-26 12:45:30.156',
      latency: 156,
      timeout: 5000,
      retryCount: 0,
      maxRetries: 3,
      priority: 8,
      tags: ['user-service', 'api-call']
    },
    {
      id: 'req-002',
      correlationId: 'corr-def456',
      requestId: 'request-002',
      requester: 'task-agent-02',
      responder: 'worker-agent-03',
      subject: 'agents.tasks.process',
      method: 'PROCESS_DATA',
      requestPayload: { taskId: 'task456', data: { type: 'image', url: 'http://example.com/image.jpg' } },
      status: 'pending',
      requestTime: '2025-08-26 12:45:25',
      timeout: 10000,
      retryCount: 1,
      maxRetries: 3,
      priority: 9,
      tags: ['task-processing', 'high-priority']
    },
    {
      id: 'req-003',
      correlationId: 'corr-ghi789',
      requestId: 'request-003',
      requester: 'monitor-agent',
      responder: 'analytics-agent',
      subject: 'agents.analytics.metrics',
      method: 'GET_METRICS',
      requestPayload: { timeRange: '1h', metrics: ['cpu', 'memory', 'requests'] },
      status: 'timeout',
      requestTime: '2025-08-26 12:44:00',
      timeout: 3000,
      retryCount: 3,
      maxRetries: 3,
      priority: 5,
      tags: ['monitoring', 'metrics']
    },
    {
      id: 'req-004',
      correlationId: 'corr-jkl012',
      requestId: 'request-004',
      responseId: 'response-004',
      requester: 'auth-agent',
      responder: 'security-service',
      subject: 'agents.security.validate',
      method: 'VALIDATE_TOKEN',
      requestPayload: { token: 'eyJhbGciOiJIUzI1NiIs...', context: 'api_access' },
      responsePayload: { valid: false, reason: 'Token expired', expiresAt: '2025-08-26 12:00:00' },
      status: 'completed',
      requestTime: '2025-08-26 12:45:20',
      responseTime: '2025-08-26 12:45:20.089',
      latency: 89,
      timeout: 2000,
      retryCount: 0,
      maxRetries: 2,
      priority: 10,
      tags: ['security', 'authentication']
    }
  ])

  const [requestPatterns, setRequestPatterns] = useState<RequestPattern[]>([
    {
      id: 'pattern-001',
      name: '用户信息查询',
      description: '查询用户基本信息和详细资料',
      subject: 'agents.service.user_info',
      method: 'GET_USER_INFO',
      expectedResponseTime: 200,
      timeout: 5000,
      retryPolicy: 'fixed',
      maxRetries: 3,
      circuitBreakerEnabled: true,
      rateLimitEnabled: true,
      requestCount: 15642,
      successRate: 98.5,
      averageLatency: 187,
      enabled: true
    },
    {
      id: 'pattern-002',
      name: '任务数据处理',
      description: '处理各种类型的任务数据',
      subject: 'agents.tasks.process',
      method: 'PROCESS_DATA',
      expectedResponseTime: 2000,
      timeout: 10000,
      retryPolicy: 'exponential',
      maxRetries: 5,
      circuitBreakerEnabled: true,
      rateLimitEnabled: false,
      requestCount: 5234,
      successRate: 94.2,
      averageLatency: 1850,
      enabled: true
    },
    {
      id: 'pattern-003',
      name: '安全令牌验证',
      description: '验证访问令牌的有效性',
      subject: 'agents.security.validate',
      method: 'VALIDATE_TOKEN',
      expectedResponseTime: 100,
      timeout: 2000,
      retryPolicy: 'none',
      maxRetries: 0,
      circuitBreakerEnabled: false,
      rateLimitEnabled: true,
      requestCount: 28945,
      successRate: 99.1,
      averageLatency: 92,
      enabled: true
    }
  ])

  const [responseHandlers, setResponseHandlers] = useState<ResponseHandler[]>([
    {
      id: 'handler-001',
      agentId: 'service-agent-01',
      agentName: '用户服务智能体',
      handlerSubjects: ['agents.service.user_info', 'agents.service.user_update'],
      status: 'online',
      pendingRequests: 3,
      processedRequests: 8456,
      averageResponseTime: 187,
      successRate: 98.5,
      lastSeen: '2025-08-26 12:45:30',
      queueSize: 3,
      maxQueueSize: 50
    },
    {
      id: 'handler-002',
      agentId: 'worker-agent-03',
      agentName: '数据处理工作智能体',
      handlerSubjects: ['agents.tasks.process', 'agents.tasks.analyze'],
      status: 'busy',
      pendingRequests: 8,
      processedRequests: 3247,
      averageResponseTime: 1850,
      successRate: 94.2,
      lastSeen: '2025-08-26 12:45:28',
      queueSize: 8,
      maxQueueSize: 20
    },
    {
      id: 'handler-003',
      agentId: 'security-service',
      agentName: '安全服务智能体',
      handlerSubjects: ['agents.security.validate', 'agents.security.encrypt'],
      status: 'online',
      pendingRequests: 1,
      processedRequests: 15632,
      averageResponseTime: 92,
      successRate: 99.1,
      lastSeen: '2025-08-26 12:45:25',
      queueSize: 1,
      maxQueueSize: 100
    },
    {
      id: 'handler-004',
      agentId: 'analytics-agent',
      agentName: '分析服务智能体',
      handlerSubjects: ['agents.analytics.metrics', 'agents.analytics.report'],
      status: 'offline',
      pendingRequests: 0,
      processedRequests: 2156,
      averageResponseTime: 0,
      successRate: 87.3,
      lastSeen: '2025-08-26 12:30:00',
      queueSize: 0,
      maxQueueSize: 30
    }
  ])

  const requestColumns = [
    {
      title: '请求信息',
      key: 'request',
      width: 300,
      render: (record: RequestResponsePair) => (
        <div>
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '4px' }}>
            <Badge 
              status={
                record.status === 'completed' ? 'success' :
                record.status === 'pending' ? 'processing' :
                record.status === 'timeout' ? 'warning' : 'error'
              }
            />
            <Text strong style={{ marginLeft: '8px', fontSize: '12px' }}>{record.method}</Text>
            <Tag color="blue" style={{ marginLeft: '8px', fontSize: '10px' }}>
              优先级: {record.priority}
            </Tag>
          </div>
          <Text code style={{ fontSize: '11px' }}>{record.subject}</Text>
          <div style={{ marginTop: '4px' }}>
            {record.tags.map((tag, index) => (
              <Tag key={index} style={{ fontSize: '10px', marginBottom: '2px' }}>
                {tag}
              </Tag>
            ))}
          </div>
        </div>
      )
    },
    {
      title: '通信方',
      key: 'participants',
      width: 200,
      render: (record: RequestResponsePair) => (
        <div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '11px' }}>
              <UserOutlined style={{ marginRight: '4px' }} />
              {record.requester}
            </Text>
          </div>
          <div style={{ marginBottom: '2px' }}>
            <Text type="secondary" style={{ fontSize: '11px' }}>
              <SwapOutlined style={{ marginRight: '4px' }} />
            </Text>
          </div>
          <div>
            <Text style={{ fontSize: '11px' }}>
              <TeamOutlined style={{ marginRight: '4px' }} />
              {record.responder}
            </Text>
          </div>
        </div>
      )
    },
    {
      title: '时间信息',
      key: 'timing',
      width: 150,
      render: (record: RequestResponsePair) => (
        <div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '11px' }}>请求: {record.requestTime}</Text>
          </div>
          {record.responseTime && (
            <div style={{ marginBottom: '2px' }}>
              <Text style={{ fontSize: '11px' }}>响应: {record.responseTime}</Text>
            </div>
          )}
          {record.latency && (
            <div>
              <Text style={{ fontSize: '11px', color: record.latency > 1000 ? '#ff4d4f' : '#52c41a' }}>
                <ClockCircleOutlined style={{ marginRight: '4px' }} />
                {record.latency}ms
              </Text>
            </div>
          )}
        </div>
      )
    },
    {
      title: '重试状态',
      key: 'retry',
      width: 100,
      render: (record: RequestResponsePair) => (
        <div>
          <Progress 
            percent={(record.retryCount / record.maxRetries) * 100}
            size="small"
            format={() => `${record.retryCount}/${record.maxRetries}`}
            status={record.retryCount >= record.maxRetries ? 'exception' : 'active'}
          />
          <Text style={{ fontSize: '10px', marginTop: '4px' }}>
            超时: {record.timeout}ms
          </Text>
        </div>
      )
    },
    {
      title: '操作',
      key: 'actions',
      width: 100,
      render: (record: RequestResponsePair) => (
        <Space>
          <Tooltip title="查看详情">
            <Button 
              type="text" 
              size="small" 
              icon={<EyeOutlined />}
              onClick={() => handleViewRequest(record)}
            />
          </Tooltip>
          {record.status === 'pending' && (
            <Tooltip title="取消请求">
              <Button 
                type="text" 
                size="small" 
                icon={<CloseCircleOutlined />}
                danger
                onClick={() => handleCancelRequest(record)}
              />
            </Tooltip>
          )}
        </Space>
      )
    }
  ]

  const patternColumns = [
    {
      title: '请求模式',
      key: 'pattern',
      render: (record: RequestPattern) => (
        <div>
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '4px' }}>
            <Badge status={record.enabled ? 'success' : 'default'} />
            <Text strong style={{ marginLeft: '8px' }}>{record.name}</Text>
          </div>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            {record.description}
          </Text>
          <br />
          <Text code style={{ fontSize: '11px' }}>{record.subject}</Text>
        </div>
      )
    },
    {
      title: '超时配置',
      key: 'timeout',
      render: (record: RequestPattern) => (
        <div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '11px' }}>预期: {record.expectedResponseTime}ms</Text>
          </div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '11px' }}>超时: {record.timeout}ms</Text>
          </div>
          <div>
            <Text style={{ fontSize: '11px' }}>重试: {record.maxRetries}次</Text>
          </div>
        </div>
      )
    },
    {
      title: '性能指标',
      key: 'metrics',
      render: (record: RequestPattern) => (
        <div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '11px' }}>请求: {record.requestCount.toLocaleString()}</Text>
          </div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '11px', color: record.successRate > 95 ? '#52c41a' : '#ff4d4f' }}>
              成功率: {record.successRate}%
            </Text>
          </div>
          <div>
            <Text style={{ fontSize: '11px' }}>延迟: {record.averageLatency}ms</Text>
          </div>
        </div>
      )
    },
    {
      title: '高级特性',
      key: 'features',
      render: (record: RequestPattern) => (
        <div>
          <div style={{ marginBottom: '2px' }}>
            <Tag color={record.circuitBreakerEnabled ? 'green' : 'default'} style={{ fontSize: '10px' }}>
              {record.circuitBreakerEnabled ? '熔断器开启' : '熔断器关闭'}
            </Tag>
          </div>
          <div style={{ marginBottom: '2px' }}>
            <Tag color={record.rateLimitEnabled ? 'orange' : 'default'} style={{ fontSize: '10px' }}>
              {record.rateLimitEnabled ? '限流开启' : '限流关闭'}
            </Tag>
          </div>
          <div>
            <Tag color="blue" style={{ fontSize: '10px' }}>
              {record.retryPolicy}重试
            </Tag>
          </div>
        </div>
      )
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: RequestPattern) => (
        <Space>
          <Button type="text" size="small" icon={<SettingOutlined />} />
          <Button type="text" size="small" icon={<LineChartOutlined />} />
          <Switch 
            size="small" 
            checked={record.enabled}
            onChange={(checked) => handleTogglePattern(record.id, checked)}
          />
        </Space>
      )
    }
  ]

  const handlerColumns = [
    {
      title: '处理器信息',
      key: 'handler',
      render: (record: ResponseHandler) => (
        <div>
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '4px' }}>
            <Badge 
              status={
                record.status === 'online' ? 'success' :
                record.status === 'offline' ? 'default' : 'processing'
              }
            />
            <Text strong style={{ marginLeft: '8px' }}>{record.agentName}</Text>
          </div>
          <Text type="secondary" style={{ fontSize: '11px' }}>{record.agentId}</Text>
          <div style={{ marginTop: '4px' }}>
            {record.handlerSubjects.slice(0, 2).map((subject, index) => (
              <Tag key={index} style={{ fontSize: '10px', marginBottom: '2px' }}>
                {subject}
              </Tag>
            ))}
            {record.handlerSubjects.length > 2 && (
              <Text type="secondary" style={{ fontSize: '10px' }}>
                +{record.handlerSubjects.length - 2}
              </Text>
            )}
          </div>
        </div>
      )
    },
    {
      title: '队列状态',
      key: 'queue',
      render: (record: ResponseHandler) => (
        <div>
          <div style={{ marginBottom: '4px' }}>
            <Text style={{ fontSize: '11px' }}>
              待处理: {record.pendingRequests}/{record.queueSize}
            </Text>
          </div>
          <Progress 
            percent={(record.queueSize / record.maxQueueSize) * 100}
            size="small"
            status={record.queueSize > record.maxQueueSize * 0.8 ? 'exception' : 'success'}
            showInfo={false}
          />
          <Text style={{ fontSize: '10px' }}>
            容量: {record.maxQueueSize}
          </Text>
        </div>
      )
    },
    {
      title: '性能指标',
      key: 'performance',
      render: (record: ResponseHandler) => (
        <div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '11px' }}>
              已处理: {record.processedRequests.toLocaleString()}
            </Text>
          </div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '11px', color: record.successRate > 95 ? '#52c41a' : '#ff4d4f' }}>
              成功率: {record.successRate}%
            </Text>
          </div>
          <div>
            <Text style={{ fontSize: '11px' }}>
              平均响应: {record.averageResponseTime}ms
            </Text>
          </div>
        </div>
      )
    },
    {
      title: '最后活跃',
      dataIndex: 'lastSeen',
      key: 'lastSeen',
      render: (lastSeen: string) => (
        <Text style={{ fontSize: '11px' }}>{lastSeen}</Text>
      )
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: ResponseHandler) => (
        <Space>
          <Button type="text" size="small" icon={<MonitorOutlined />} />
          <Button type="text" size="small" icon={<SettingOutlined />} />
          {record.status === 'offline' && (
            <Button type="text" size="small" icon={<PlayCircleOutlined />} />
          )}
        </Space>
      )
    }
  ]

  const handleViewRequest = (request: RequestResponsePair) => {
    Modal.info({
      title: '请求响应详情',
      width: 1000,
      content: (
        <div>
          <Descriptions column={2} size="small">
            <Descriptions.Item label="关联ID">{request.correlationId}</Descriptions.Item>
            <Descriptions.Item label="请求ID">{request.requestId}</Descriptions.Item>
            <Descriptions.Item label="响应ID">{request.responseId || '未响应'}</Descriptions.Item>
            <Descriptions.Item label="状态">
              <Badge 
                status={
                  request.status === 'completed' ? 'success' :
                  request.status === 'pending' ? 'processing' :
                  request.status === 'timeout' ? 'warning' : 'error'
                }
                text={request.status}
              />
            </Descriptions.Item>
            <Descriptions.Item label="请求方">{request.requester}</Descriptions.Item>
            <Descriptions.Item label="响应方">{request.responder}</Descriptions.Item>
            <Descriptions.Item label="主题">{request.subject}</Descriptions.Item>
            <Descriptions.Item label="方法">{request.method}</Descriptions.Item>
            <Descriptions.Item label="请求时间">{request.requestTime}</Descriptions.Item>
            <Descriptions.Item label="响应时间">{request.responseTime || '未响应'}</Descriptions.Item>
            <Descriptions.Item label="响应延迟">{request.latency ? `${request.latency}ms` : '未知'}</Descriptions.Item>
            <Descriptions.Item label="超时设置">{request.timeout}ms</Descriptions.Item>
            <Descriptions.Item label="重试次数">{request.retryCount}/{request.maxRetries}</Descriptions.Item>
            <Descriptions.Item label="优先级">{request.priority}</Descriptions.Item>
          </Descriptions>

          <Divider>请求内容</Divider>
          <pre style={{ background: '#f5f5f5', padding: '12px', borderRadius: '4px', fontSize: '12px' }}>
            {JSON.stringify(request.requestPayload, null, 2)}
          </pre>

          {request.responsePayload && (
            <>
              <Divider>响应内容</Divider>
              <pre style={{ background: '#f5f5f5', padding: '12px', borderRadius: '4px', fontSize: '12px' }}>
                {JSON.stringify(request.responsePayload, null, 2)}
              </pre>
            </>
          )}
        </div>
      )
    })
  }

  const handleCancelRequest = (request: RequestResponsePair) => {
    Modal.confirm({
      title: '取消请求',
      content: `确定要取消请求 "${request.method}" 吗？`,
      onOk: () => {
        setRequestPairs(prev => prev.map(r => 
          r.id === request.id ? { ...r, status: 'failed' as const } : r
        ))
        notification.success({
          message: '请求已取消',
          description: `请求 ${request.method} 已被取消`
        })
      }
    })
  }

  const handleTogglePattern = (patternId: string, enabled: boolean) => {
    setRequestPatterns(prev => prev.map(p =>
      p.id === patternId ? { ...p, enabled } : p
    ))
    notification.success({
      message: enabled ? '模式已启用' : '模式已禁用',
      description: `请求模式状态已更新`
    })
  }

  const handleSendTestRequest = (values: any) => {
    setLoading(true)
    
    const newRequest: RequestResponsePair = {
      id: `req-${Date.now()}`,
      correlationId: `corr-${Date.now()}`,
      requestId: `request-${Date.now()}`,
      requester: values.requester || 'test-client',
      responder: values.responder,
      subject: values.subject,
      method: values.method,
      requestPayload: JSON.parse(values.payload || '{}'),
      status: 'pending',
      requestTime: new Date().toLocaleString('zh-CN'),
      timeout: values.timeout || 5000,
      retryCount: 0,
      maxRetries: values.maxRetries || 3,
      priority: values.priority || 5,
      tags: values.tags || []
    }

    setRequestPairs(prev => [newRequest, ...prev])
    setTestModalVisible(false)
    form.resetFields()

    // 模拟响应
    setTimeout(() => {
      setRequestPairs(prev => prev.map(r =>
        r.id === newRequest.id ? {
          ...r,
          status: 'completed',
          responseId: `response-${Date.now()}`,
          responseTime: new Date().toLocaleString('zh-CN'),
          latency: Math.random() * 500 + 50,
          responsePayload: { result: 'success', message: 'Test request completed' }
        } : r
      ))
      setLoading(false)
      notification.success({
        message: '测试请求完成',
        description: '测试请求已成功处理'
      })
    }, 2000)
  }

  const refreshData = () => {
    setLoading(true)
    setTimeout(() => {
      setMetrics(prev => ({
        ...prev,
        totalRequests: prev.totalRequests + Math.floor(Math.random() * 100),
        requestsPerSecond: Math.random() * 20 + 60
      }))
      notification.success({
        message: '刷新成功',
        description: '请求响应数据已更新'
      })
      setLoading(false)
    }, 1000)
  }

  const getStatusColor = (successRate: number) => {
    if (successRate >= 95) return '#52c41a'
    if (successRate >= 90) return '#faad14'
    return '#ff4d4f'
  }

  return (
    <div style={{ padding: '24px', background: '#f5f5f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <SwapOutlined style={{ marginRight: '12px', color: '#1890ff' }} />
          请求响应机制
        </Title>
        <Paragraph>
          智能体间请求响应通信机制，包括请求监控、响应时间分析、重试策略、超时控制等功能
        </Paragraph>
      </div>

      {/* 性能指标概览 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={12} sm={6}>
          <Card>
            <Statistic
              title="请求总量"
              value={metrics.totalRequests}
              precision={0}
              valueStyle={{ color: '#1890ff' }}
              prefix={<SendOutlined />}
            />
            <div style={{ marginTop: '8px' }}>
              <Text type="secondary" style={{ fontSize: '12px' }}>
                待处理: {metrics.pendingRequests}
              </Text>
            </div>
          </Card>
        </Col>
        
        <Col xs={12} sm={6}>
          <Card>
            <Statistic
              title="请求速率"
              value={metrics.requestsPerSecond}
              precision={1}
              suffix="req/s"
              valueStyle={{ color: '#52c41a' }}
              prefix={<ThunderboltOutlined />}
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
            <div style={{ marginTop: '8px' }}>
              <Text type="secondary" style={{ fontSize: '12px' }}>
                P99: {metrics.p99Latency.toFixed(1)}ms
              </Text>
            </div>
          </Card>
        </Col>
        
        <Col xs={12} sm={6}>
          <Card>
            <Statistic
              title="成功率"
              value={metrics.successRate}
              precision={1}
              suffix="%"
              valueStyle={{ color: getStatusColor(metrics.successRate) }}
              prefix={<CheckCircleOutlined />}
            />
            <div style={{ marginTop: '8px' }}>
              <Progress 
                percent={metrics.successRate} 
                size="small" 
                showInfo={false}
                status={metrics.successRate >= 95 ? 'success' : 'exception'}
              />
            </div>
          </Card>
        </Col>
      </Row>

      {/* 主管理界面 */}
      <Card>
        <div style={{ marginBottom: '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Space>
            <Button 
              type="primary" 
              icon={<RocketOutlined />}
              onClick={() => setTestModalVisible(true)}
            >
              发送测试请求
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
            <Button icon={<LineChartOutlined />} onClick={() => setMetricsDrawerVisible(true)}>
              性能分析
            </Button>
            <Button icon={<SettingOutlined />}>机制配置</Button>
          </Space>
        </div>

        <Tabs activeKey={activeTab} onChange={setActiveTab}>
          <TabPane tab="请求监控" key="monitor" icon={<MonitorOutlined />}>
            <div style={{ marginBottom: '16px' }}>
              <Alert
                message="实时监控"
                description={`当前有 ${metrics.pendingRequests} 个请求待处理，平均响应时间 ${metrics.averageLatency.toFixed(1)}ms`}
                type="info"
                showIcon
                style={{ marginBottom: '16px' }}
              />
            </div>
            <Table
              columns={requestColumns}
              dataSource={requestPairs}
              rowKey="id"
              size="small"
              pagination={{ pageSize: 15 }}
              scroll={{ x: 1000 }}
            />
          </TabPane>
          
          <TabPane tab="请求模式" key="patterns" icon={<ApiOutlined />}>
            <div style={{ marginBottom: '16px' }}>
              <Button 
                type="primary" 
                icon={<PlusOutlined />}
                onClick={() => setPatternModalVisible(true)}
              >
                创建请求模式
              </Button>
            </div>
            <Table
              columns={patternColumns}
              dataSource={requestPatterns}
              rowKey="id"
              size="small"
              pagination={false}
            />
          </TabPane>
          
          <TabPane tab="响应处理器" key="handlers" icon={<TeamOutlined />}>
            <div style={{ marginBottom: '16px' }}>
              <Text type="secondary">
                在线处理器: {responseHandlers.filter(h => h.status === 'online').length} / {responseHandlers.length}
              </Text>
            </div>
            <Table
              columns={handlerColumns}
              dataSource={responseHandlers}
              rowKey="id"
              size="small"
              pagination={false}
            />
          </TabPane>
          
          <TabPane tab="超时重试" key="retry" icon={<SyncOutlined />}>
            <Card title="重试策略配置" size="small">
              <Row gutter={[16, 16]}>
                <Col span={8}>
                  <Card size="small" title="固定间隔重试">
                    <Statistic value={65} suffix="%" title="使用率" />
                    <Text type="secondary" style={{ fontSize: '12px' }}>
                      固定时间间隔进行重试
                    </Text>
                  </Card>
                </Col>
                <Col span={8}>
                  <Card size="small" title="指数退避重试">
                    <Statistic value={28} suffix="%" title="使用率" />
                    <Text type="secondary" style={{ fontSize: '12px' }}>
                      逐步增加重试间隔时间
                    </Text>
                  </Card>
                </Col>
                <Col span={8}>
                  <Card size="small" title="无重试">
                    <Statistic value={7} suffix="%" title="使用率" />
                    <Text type="secondary" style={{ fontSize: '12px' }}>
                      不进行自动重试
                    </Text>
                  </Card>
                </Col>
              </Row>
              
              <Divider />
              
              <Title level={5}>超时配置分析</Title>
              <Row gutter={[16, 16]}>
                <Col span={12}>
                  <Text strong>超时请求分布：</Text>
                  <div style={{ marginTop: '8px' }}>
                    <Tag color="red">高频服务: {Math.floor(metrics.timeoutRequests * 0.6)} 次</Tag>
                    <Tag color="orange">中频服务: {Math.floor(metrics.timeoutRequests * 0.3)} 次</Tag>
                    <Tag color="yellow">低频服务: {Math.floor(metrics.timeoutRequests * 0.1)} 次</Tag>
                  </div>
                </Col>
                <Col span={12}>
                  <Text strong>推荐优化：</Text>
                  <div style={{ marginTop: '8px' }}>
                    <div style={{ marginBottom: '4px' }}>
                      <Text style={{ fontSize: '12px' }}>• 调整高频服务超时时间到 3000ms</Text>
                    </div>
                    <div style={{ marginBottom: '4px' }}>
                      <Text style={{ fontSize: '12px' }}>• 启用指数退避重试策略</Text>
                    </div>
                    <div>
                      <Text style={{ fontSize: '12px' }}>• 配置熔断器防止级联故障</Text>
                    </div>
                  </div>
                </Col>
              </Row>
            </Card>
          </TabPane>
        </Tabs>
      </Card>

      {/* 测试请求Modal */}
      <Modal
        title="发送测试请求"
        visible={testModalVisible}
        onCancel={() => setTestModalVisible(false)}
        width={700}
        footer={[
          <Button key="cancel" onClick={() => setTestModalVisible(false)}>
            取消
          </Button>,
          <Button key="submit" type="primary" onClick={() => form.submit()}>
            发送请求
          </Button>
        ]}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleSendTestRequest}
        >
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item name="requester" label="请求方ID" initialValue="test-client">
                <Input placeholder="请求方智能体ID" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="responder" label="响应方ID" rules={[{ required: true }]}>
                <Select placeholder="选择响应方">
                  {responseHandlers.map(handler => (
                    <Option key={handler.agentId} value={handler.agentId}>
                      {handler.agentName}
                    </Option>
                  ))}
                </Select>
              </Form.Item>
            </Col>
          </Row>
          
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item name="subject" label="请求主题" rules={[{ required: true }]}>
                <Input placeholder="例如: agents.service.test" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="method" label="请求方法" rules={[{ required: true }]}>
                <Input placeholder="例如: GET_STATUS" />
              </Form.Item>
            </Col>
          </Row>
          
          <Form.Item name="payload" label="请求内容" rules={[{ required: true }]}>
            <TextArea 
              rows={4} 
              placeholder='{"key": "value", "data": {...}}'
            />
          </Form.Item>
          
          <Row gutter={16}>
            <Col span={8}>
              <Form.Item name="timeout" label="超时时间(ms)" initialValue={5000}>
                <Input type="number" />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item name="maxRetries" label="最大重试次数" initialValue={3}>
                <Input type="number" />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item name="priority" label="优先级" initialValue={5}>
                <Slider min={1} max={10} />
              </Form.Item>
            </Col>
          </Row>
          
          <Form.Item name="tags" label="标签">
            <Select mode="tags" placeholder="添加标签">
              <Option value="test">test</Option>
              <Option value="api-call">api-call</Option>
              <Option value="high-priority">high-priority</Option>
            </Select>
          </Form.Item>
        </Form>
      </Modal>

      {/* 性能分析Drawer */}
      <Drawer
        title="请求响应性能分析"
        placement="right"
        width={800}
        visible={metricsDrawerVisible}
        onClose={() => setMetricsDrawerVisible(false)}
      >
        <Row gutter={[16, 16]}>
          <Col span={12}>
            <Card title="延迟分析" size="small">
              <div style={{ marginBottom: '8px' }}>
                <Text>平均延迟: {metrics.averageLatency.toFixed(1)}ms</Text>
              </div>
              <div style={{ marginBottom: '8px' }}>
                <Text>P95延迟: {metrics.p95Latency.toFixed(1)}ms</Text>
              </div>
              <div style={{ marginBottom: '8px' }}>
                <Text>P99延迟: {metrics.p99Latency.toFixed(1)}ms</Text>
              </div>
            </Card>
          </Col>
          
          <Col span={12}>
            <Card title="状态分布" size="small">
              <div style={{ marginBottom: '8px' }}>
                <Badge status="success" text={`完成: ${metrics.completedRequests}`} />
              </div>
              <div style={{ marginBottom: '8px' }}>
                <Badge status="processing" text={`处理中: ${metrics.pendingRequests}`} />
              </div>
              <div style={{ marginBottom: '8px' }}>
                <Badge status="warning" text={`超时: ${metrics.timeoutRequests}`} />
              </div>
              <div>
                <Badge status="error" text={`失败: ${metrics.failedRequests}`} />
              </div>
            </Card>
          </Col>
        </Row>
        
        <Divider />
        
        <Card title="性能趋势" size="small">
          <div style={{ textAlign: 'center', padding: '40px 0' }}>
            <Text type="secondary">性能趋势图表开发中...</Text>
          </div>
        </Card>
      </Drawer>
    </div>
  )
}

export default RequestResponseMechanismPage