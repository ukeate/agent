import { buildApiUrl, apiFetch } from '../utils/apiBase'
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
  Collapse,
  Tree,
  InputNumber,
  Descriptions,
  List,
  Avatar,
  Steps
} from 'antd'
import {
  MonitorOutlined,
  DashboardOutlined,
  LineChartOutlined,
  BarChartOutlined,
  PieChartOutlined,
  RadarChartOutlined,
  ThunderboltOutlined,
  RocketOutlined,
  FireOutlined,
  BugOutlined,
  TrophyOutlined,
  BellOutlined,
  AlertOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ExclamationCircleOutlined,
  ReloadOutlined,
  SettingOutlined,
  EyeOutlined,
  EditOutlined,
  DeleteOutlined,
  PlayCircleOutlined,
  StopOutlined,
  SyncOutlined,
  ClockCircleOutlined,
  ShareAltOutlined as NetworkOutlined,
  CloudServerOutlined,
  DatabaseOutlined,
  ApiOutlined,
  SecurityScanOutlined,
  TeamOutlined,
  UserOutlined,
  SearchOutlined,
  FilterOutlined,
  DownloadOutlined,
  UploadOutlined,
  HeartOutlined,
  LockOutlined,
  UnlockOutlined
} from '@ant-design/icons'

const { Title, Text, Paragraph } = Typography
const { Option } = Select
const { TextArea } = Input
const { TabPane } = Tabs
const { Panel } = Collapse
const { Step } = Steps

interface PerformanceMetrics {
  messageLatency: {
    average: number
    p50: number
    p95: number
    p99: number
  }
  throughput: {
    current: number
    peak: number
    average24h: number
  }
  errorRates: {
    connection: number
    timeout: number
    processing: number
    total: number
  }
  resourceUsage: {
    cpu: number
    memory: number
    network: number
    disk: number
  }
  systemHealth: {
    overallScore: number
    availability: number
    reliability: number
    performance: number
  }
}

interface AlertRule {
  id: string
  name: string
  description: string
  metric: string
  condition: 'greater_than' | 'less_than' | 'equals' | 'not_equals'
  threshold: number
  duration: number
  severity: 'info' | 'warning' | 'critical' | 'fatal'
  channels: string[]
  enabled: boolean
  triggerCount: number
  lastTriggered?: string
  status: 'normal' | 'firing' | 'silenced'
  createdAt: string
}

interface PerformanceAlert {
  id: string
  ruleId: string
  ruleName: string
  severity: 'info' | 'warning' | 'critical' | 'fatal'
  message: string
  metric: string
  currentValue: number
  threshold: number
  startTime: string
  endTime?: string
  status: 'active' | 'resolved' | 'silenced'
  acknowledgedBy?: string
  acknowledgedAt?: string
  tags: string[]
}

interface OptimizationSuggestion {
  id: string
  category: 'performance' | 'reliability' | 'security' | 'scalability'
  title: string
  description: string
  impact: 'low' | 'medium' | 'high'
  effort: 'low' | 'medium' | 'high'
  priority: number
  metrics: {
    currentValue: number
    expectedImprovement: string
    affectedComponents: string[]
  }
  implementation: {
    steps: string[]
    estimatedTime: string
    risks: string[]
  }
  status: 'pending' | 'in_progress' | 'completed' | 'dismissed'
  createdAt: string
}

interface MonitoringDashboard {
  id: string
  name: string
  description: string
  widgets: DashboardWidget[]
  layout: 'grid' | 'flow'
  refreshInterval: number
  public: boolean
  tags: string[]
  createdBy: string
  createdAt: string
  lastModified: string
}

interface DashboardWidget {
  id: string
  type: 'line_chart' | 'bar_chart' | 'pie_chart' | 'statistic' | 'table' | 'gauge'
  title: string
  position: { x: number; y: number; w: number; h: number }
  config: {
    metrics: string[]
    timeRange: string
    aggregation?: 'avg' | 'sum' | 'max' | 'min'
    groupBy?: string[]
  }
  thresholds?: {
    warning: number
    critical: number
  }
}

const MonitoringPerformanceOptimizationPage: React.FC = () => {
  const [form] = Form.useForm()
  const [activeTab, setActiveTab] = useState('dashboard')
  const [loading, setLoading] = useState(false)
  const [alertModalVisible, setAlertModalVisible] = useState(false)
  const [optimizationDrawerVisible, setOptimizationDrawerVisible] = useState(false)
  const [dashboardModalVisible, setDashboardModalVisible] = useState(false)
  const [selectedTimeRange, setSelectedTimeRange] = useState('1h')
  
  const [performanceMetrics, setPerformanceMetrics] = useState<PerformanceMetrics>({
    messageLatency: {
      average: 156.8,
      p50: 89.2,
      p95: 245.7,
      p99: 456.3
    },
    throughput: {
      current: 1250.5,
      peak: 2845.7,
      average24h: 856.2
    },
    errorRates: {
      connection: 0.05,
      timeout: 0.12,
      processing: 0.08,
      total: 0.25
    },
    resourceUsage: {
      cpu: 45.2,
      memory: 62.8,
      network: 78.5,
      disk: 34.7
    },
    systemHealth: {
      overallScore: 92.5,
      availability: 99.8,
      reliability: 97.2,
      performance: 89.7
    }
  })

  const [alertRules, setAlertRules] = useState<AlertRule[]>([
    {
      id: 'alert-001',
      name: '高延迟告警',
      description: '当消息平均延迟超过500ms时触发告警',
      metric: 'message.latency.average',
      condition: 'greater_than',
      threshold: 500,
      duration: 300,
      severity: 'warning',
      channels: ['email', 'slack', 'webhook'],
      enabled: true,
      triggerCount: 23,
      lastTriggered: '2025-08-26 11:45:00',
      status: 'normal',
      createdAt: '2025-08-20 10:30:00'
    },
    {
      id: 'alert-002',
      name: '系统可用性告警',
      description: '当系统可用性低于99%时触发严重告警',
      metric: 'system.availability',
      condition: 'less_than',
      threshold: 99,
      duration: 60,
      severity: 'critical',
      channels: ['email', 'sms', 'pagerduty'],
      enabled: true,
      triggerCount: 2,
      lastTriggered: '2025-08-25 03:15:00',
      status: 'normal',
      createdAt: '2025-08-18 14:20:00'
    },
    {
      id: 'alert-003',
      name: '错误率告警',
      description: '当总错误率超过1%时触发告警',
      metric: 'errors.rate.total',
      condition: 'greater_than',
      threshold: 1,
      duration: 180,
      severity: 'warning',
      channels: ['slack'],
      enabled: true,
      triggerCount: 45,
      lastTriggered: '2025-08-26 09:30:00',
      status: 'firing',
      createdAt: '2025-08-19 09:15:00'
    }
  ])

  const [performanceAlerts, setPerformanceAlerts] = useState<PerformanceAlert[]>([
    {
      id: 'palert-001',
      ruleId: 'alert-003',
      ruleName: '错误率告警',
      severity: 'warning',
      message: '系统总错误率 1.25% 超过阈值 1%，持续时间 12 分钟',
      metric: 'errors.rate.total',
      currentValue: 1.25,
      threshold: 1,
      startTime: '2025-08-26 12:30:00',
      status: 'active',
      tags: ['error-rate', 'system-health']
    },
    {
      id: 'palert-002',
      ruleId: 'alert-001',
      ruleName: '高延迟告警',
      severity: 'warning',
      message: '消息平均延迟 523ms 超过阈值 500ms',
      metric: 'message.latency.average',
      currentValue: 523,
      threshold: 500,
      startTime: '2025-08-26 11:45:00',
      endTime: '2025-08-26 12:15:00',
      status: 'resolved',
      tags: ['latency', 'performance']
    }
  ])

  const [optimizationSuggestions, setOptimizationSuggestions] = useState<OptimizationSuggestion[]>([
    {
      id: 'opt-001',
      category: 'performance',
      title: '启用消息批处理',
      description: '通过批量处理消息可显著提升吞吐量并降低延迟',
      impact: 'high',
      effort: 'medium',
      priority: 95,
      metrics: {
        currentValue: 1250.5,
        expectedImprovement: '+40% 吞吐量, -25% 延迟',
        affectedComponents: ['MessageBus', 'RequestResponseManager']
      },
      implementation: {
        steps: [
          '配置批处理参数 (batch_size: 50, timeout: 100ms)',
          '更新消息处理逻辑以支持批量操作',
          '调整消费者配置以处理批量消息',
          '监控性能指标并调优参数'
        ],
        estimatedTime: '3-5 工作日',
        risks: ['可能增加内存使用', '需要调整下游系统']
      },
      status: 'pending',
      createdAt: '2025-08-26 09:00:00'
    },
    {
      id: 'opt-002',
      category: 'reliability',
      title: '实施智能重试策略',
      description: '采用指数退避和熔断机制提升系统可靠性',
      impact: 'high',
      effort: 'medium',
      priority: 88,
      metrics: {
        currentValue: 97.2,
        expectedImprovement: '+2.5% 可靠性',
        affectedComponents: ['ReliabilityManager', 'CircuitBreaker']
      },
      implementation: {
        steps: [
          '实现指数退避重试算法',
          '集成熔断器模式',
          '配置智能故障检测',
          '添加重试统计和监控'
        ],
        estimatedTime: '4-6 工作日',
        risks: ['可能增加系统复杂性', '需要仔细调优参数']
      },
      status: 'in_progress',
      createdAt: '2025-08-25 14:30:00'
    },
    {
      id: 'opt-003',
      category: 'scalability',
      title: '实现智能负载均衡',
      description: '基于实时负载和健康状况的动态负载均衡',
      impact: 'medium',
      effort: 'high',
      priority: 75,
      metrics: {
        currentValue: 3,
        expectedImprovement: '支持 10+ 节点水平扩展',
        affectedComponents: ['LoadBalancer', 'HealthMonitor']
      },
      implementation: {
        steps: [
          '设计负载均衡算法',
          '实现健康检查机制',
          '开发动态路由功能',
          '集成监控和告警'
        ],
        estimatedTime: '2-3 周',
        risks: ['复杂性较高', '需要充分测试']
      },
      status: 'pending',
      createdAt: '2025-08-24 16:45:00'
    }
  ])

  const [dashboards, setDashboards] = useState<MonitoringDashboard[]>([
    {
      id: 'dashboard-001',
      name: '系统总览仪表板',
      description: '系统核心指标和健康状况总览',
      widgets: [
        {
          id: 'widget-001',
          type: 'statistic',
          title: '消息吞吐量',
          position: { x: 0, y: 0, w: 6, h: 3 },
          config: { metrics: ['throughput.current'], timeRange: '1h' }
        },
        {
          id: 'widget-002',
          type: 'line_chart',
          title: '延迟趋势',
          position: { x: 6, y: 0, w: 6, h: 6 },
          config: { metrics: ['latency.p95', 'latency.p99'], timeRange: '24h' }
        }
      ],
      layout: 'grid',
      refreshInterval: 30,
      public: false,
      tags: ['overview', 'system'],
      createdBy: 'admin',
      createdAt: '2025-08-20 10:00:00',
      lastModified: '2025-08-26 11:30:00'
    }
  ])

  const alertColumns = [
    {
      title: '告警规则',
      key: 'rule',
      width: 250,
      render: (record: AlertRule) => (
        <div>
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '4px' }}>
            <Badge 
              status={
                record.status === 'firing' ? 'error' :
                record.status === 'silenced' ? 'warning' : 'success'
              }
            />
            <Text strong style={{ marginLeft: '8px', fontSize: '13px' }}>{record.name}</Text>
            <Tag 
              color={
                record.severity === 'fatal' ? 'red' :
                record.severity === 'critical' ? 'volcano' :
                record.severity === 'warning' ? 'orange' : 'blue'
              }
              style={{ marginLeft: '8px', fontSize: '10px' }}
            >
              {record.severity}
            </Tag>
          </div>
          <Text type="secondary" style={{ fontSize: '11px' }}>
            {record.description}
          </Text>
        </div>
      )
    },
    {
      title: '监控指标',
      key: 'metric',
      width: 150,
      render: (record: AlertRule) => (
        <div>
          <Text code style={{ fontSize: '11px' }}>{record.metric}</Text>
          <div style={{ marginTop: '4px' }}>
            <Text style={{ fontSize: '11px' }}>
              {record.condition} {record.threshold}
            </Text>
          </div>
        </div>
      )
    },
    {
      title: '触发统计',
      key: 'stats',
      width: 120,
      render: (record: AlertRule) => (
        <div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '11px' }}>次数: {record.triggerCount}</Text>
          </div>
          {record.lastTriggered && (
            <Text type="secondary" style={{ fontSize: '10px' }}>
              最后: {record.lastTriggered}
            </Text>
          )}
        </div>
      )
    },
    {
      title: '通知渠道',
      dataIndex: 'channels',
      key: 'channels',
      width: 120,
      render: (channels: string[]) => (
        <div>
          {channels.map((channel, index) => (
            <Tag key={index} style={{ fontSize: '10px', marginBottom: '2px' }}>
              {channel}
            </Tag>
          ))}
        </div>
      )
    },
    {
      title: '状态',
      dataIndex: 'enabled',
      key: 'enabled',
      width: 80,
      render: (enabled: boolean) => (
        <Switch size="small" checked={enabled} />
      )
    },
    {
      title: '操作',
      key: 'actions',
      width: 120,
      render: (record: AlertRule) => (
        <Space>
          <Button type="text" size="small" icon={<EyeOutlined />} />
          <Button type="text" size="small" icon={<EditOutlined />} />
          {record.status === 'firing' && (
            <Button type="text" size="small" icon={<StopOutlined />} />
          )}
        </Space>
      )
    }
  ]

  const activeAlertColumns = [
    {
      title: '告警信息',
      key: 'alert',
      render: (record: PerformanceAlert) => (
        <div>
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '4px' }}>
            <Badge 
              status={
                record.severity === 'fatal' || record.severity === 'critical' ? 'error' :
                record.severity === 'warning' ? 'warning' : 'processing'
              }
            />
            <Text strong style={{ marginLeft: '8px', fontSize: '12px' }}>{record.ruleName}</Text>
            <Tag 
              color={
                record.status === 'active' ? 'red' :
                record.status === 'resolved' ? 'green' : 'default'
              }
              style={{ marginLeft: '8px', fontSize: '10px' }}
            >
              {record.status === 'active' ? '活跃' : 
               record.status === 'resolved' ? '已解决' : '已静默'}
            </Tag>
          </div>
          <Text style={{ fontSize: '11px' }}>{record.message}</Text>
        </div>
      )
    },
    {
      title: '指标值',
      key: 'value',
      render: (record: PerformanceAlert) => (
        <div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ 
              fontSize: '11px',
              color: record.currentValue > record.threshold ? '#ff4d4f' : '#52c41a'
            }}>
              当前: {record.currentValue}
            </Text>
          </div>
          <div>
            <Text style={{ fontSize: '11px' }}>阈值: {record.threshold}</Text>
          </div>
        </div>
      )
    },
    {
      title: '时间',
      key: 'time',
      render: (record: PerformanceAlert) => (
        <div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '11px' }}>开始: {record.startTime}</Text>
          </div>
          {record.endTime && (
            <Text style={{ fontSize: '11px' }}>结束: {record.endTime}</Text>
          )}
        </div>
      )
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: PerformanceAlert) => (
        <Space>
          {record.status === 'active' && !record.acknowledgedBy && (
            <Button type="text" size="small" icon={<CheckCircleOutlined />}>
              确认
            </Button>
          )}
          <Button type="text" size="small" icon={<EyeOutlined />} />
        </Space>
      )
    }
  ]

  const optimizationColumns = [
    {
      title: '优化建议',
      key: 'suggestion',
      render: (record: OptimizationSuggestion) => (
        <div>
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '4px' }}>
            <Badge 
              status={
                record.status === 'completed' ? 'success' :
                record.status === 'in_progress' ? 'processing' : 'default'
              }
            />
            <Text strong style={{ marginLeft: '8px' }}>{record.title}</Text>
            <Tag 
              color={
                record.category === 'performance' ? 'blue' :
                record.category === 'reliability' ? 'green' :
                record.category === 'security' ? 'red' : 'orange'
              }
              style={{ marginLeft: '8px', fontSize: '10px' }}
            >
              {record.category}
            </Tag>
          </div>
          <Text type="secondary" style={{ fontSize: '11px' }}>
            {record.description}
          </Text>
        </div>
      )
    },
    {
      title: '影响评估',
      key: 'impact',
      render: (record: OptimizationSuggestion) => (
        <div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '11px' }}>
              影响: <Tag color={record.impact === 'high' ? 'red' : record.impact === 'medium' ? 'orange' : 'blue'} style={{ fontSize: '10px' }}>
                {record.impact}
              </Tag>
            </Text>
          </div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '11px' }}>
              工作量: <Tag color={record.effort === 'high' ? 'red' : record.effort === 'medium' ? 'orange' : 'blue'} style={{ fontSize: '10px' }}>
                {record.effort}
              </Tag>
            </Text>
          </div>
          <div>
            <Text style={{ fontSize: '11px' }}>
              优先级: <Tag color="gold" style={{ fontSize: '10px' }}>{record.priority}</Tag>
            </Text>
          </div>
        </div>
      )
    },
    {
      title: '预期收益',
      dataIndex: ['metrics', 'expectedImprovement'],
      key: 'improvement',
      render: (text: string) => <Text style={{ fontSize: '11px' }}>{text}</Text>
    },
    {
      title: '预估时间',
      dataIndex: ['implementation', 'estimatedTime'],
      key: 'time',
      render: (text: string) => <Text style={{ fontSize: '11px' }}>{text}</Text>
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Badge 
          status={
            status === 'completed' ? 'success' :
            status === 'in_progress' ? 'processing' :
            status === 'dismissed' ? 'default' : 'warning'
          }
          text={
            status === 'completed' ? '已完成' :
            status === 'in_progress' ? '进行中' :
            status === 'dismissed' ? '已忽略' : '待处理'
          }
        />
      )
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: OptimizationSuggestion) => (
        <Space>
          <Button type="text" size="small" icon={<EyeOutlined />} />
          {record.status === 'pending' && (
            <Button type="text" size="small" icon={<PlayCircleOutlined />} />
          )}
          <Button type="text" size="small" icon={<CloseCircleOutlined />} />
        </Space>
      )
    }
  ]

  const refreshData = () => {
    setLoading(true)
    apiFetch(buildApiUrl('/api/v1/health/metrics'))
      .then(res => {
        return res.json()
      })
      .then(data => {
        setPerformanceMetrics(prev => ({
          ...prev,
          messageLatency: {
            ...prev.messageLatency,
            average: data.latency_ms ?? prev.messageLatency.average,
            p50: data.latency_p50 ?? prev.messageLatency.p50,
            p95: data.latency_p95 ?? prev.messageLatency.p95,
            p99: data.latency_p99 ?? prev.messageLatency.p99
          },
          throughput: {
            ...prev.throughput,
            current: data.throughput_qps ?? prev.throughput.current,
            peak: prev.throughput.peak,
            average24h: prev.throughput.average24h
          },
          resourceUsage: {
            cpu: data.cpu_usage ?? prev.resourceUsage.cpu,
            memory: data.memory_usage ?? prev.resourceUsage.memory,
            network: data.network_usage ?? prev.resourceUsage.network,
            disk: data.disk_usage ?? prev.resourceUsage.disk
          }
        }))
        notification.success({
          message: '数据刷新成功',
          description: '监控指标已更新'
        })
      })
      .catch(() => {
        notification.error({ message: '数据刷新失败', description: '无法获取最新指标' })
      })
      .finally(() => setLoading(false))
  }

  const getHealthColor = (score: number) => {
    if (score >= 90) return '#52c41a'
    if (score >= 75) return '#faad14'
    return '#ff4d4f'
  }

  const getHealthStatus = (score: number) => {
    if (score >= 90) return '优秀'
    if (score >= 75) return '良好'
    if (score >= 60) return '一般'
    return '差'
  }

  return (
    <div style={{ padding: '24px', background: '#f5f5f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <MonitorOutlined style={{ marginRight: '12px', color: '#1890ff' }} />
          监控和性能优化
        </Title>
        <Paragraph>
          智能体消息系统监控、性能分析、告警管理和优化建议，确保系统稳定高效运行
        </Paragraph>
      </div>

      {/* 系统健康状态总览 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={12} sm={6}>
          <Card>
            <Statistic
              title="系统健康度"
              value={performanceMetrics.systemHealth.overallScore}
              precision={1}
              suffix="%"
              valueStyle={{ color: getHealthColor(performanceMetrics.systemHealth.overallScore) }}
              prefix={<HeartOutlined />}
            />
            <div style={{ marginTop: '8px' }}>
              <Text type="secondary" style={{ fontSize: '12px' }}>
                状态: {getHealthStatus(performanceMetrics.systemHealth.overallScore)}
              </Text>
            </div>
          </Card>
        </Col>
        
        <Col xs={12} sm={6}>
          <Card>
            <Statistic
              title="消息吞吐量"
              value={performanceMetrics.throughput.current}
              precision={1}
              suffix="msg/s"
              valueStyle={{ color: '#1890ff' }}
              prefix={<ThunderboltOutlined />}
            />
            <div style={{ marginTop: '8px' }}>
              <Text type="secondary" style={{ fontSize: '12px' }}>
                峰值: {performanceMetrics.throughput.peak.toFixed(1)}
              </Text>
            </div>
          </Card>
        </Col>
        
        <Col xs={12} sm={6}>
          <Card>
            <Statistic
              title="平均延迟"
              value={performanceMetrics.messageLatency.average}
              precision={1}
              suffix="ms"
              valueStyle={{ 
                color: performanceMetrics.messageLatency.average > 200 ? '#ff4d4f' : '#52c41a' 
              }}
              prefix={<ClockCircleOutlined />}
            />
            <div style={{ marginTop: '8px' }}>
              <Text type="secondary" style={{ fontSize: '12px' }}>
                P99: {performanceMetrics.messageLatency.p99.toFixed(1)}ms
              </Text>
            </div>
          </Card>
        </Col>
        
        <Col xs={12} sm={6}>
          <Card>
            <Statistic
              title="错误率"
              value={performanceMetrics.errorRates.total}
              precision={2}
              suffix="%"
              valueStyle={{ 
                color: performanceMetrics.errorRates.total > 1 ? '#ff4d4f' : '#52c41a' 
              }}
              prefix={<BugOutlined />}
            />
            <div style={{ marginTop: '8px' }}>
              <Progress 
                percent={Math.min(100, performanceMetrics.errorRates.total * 10)} 
                size="small" 
                showInfo={false}
                status={performanceMetrics.errorRates.total > 1 ? 'exception' : 'success'}
              />
            </div>
          </Card>
        </Col>
      </Row>

      {/* 活跃告警显示 */}
      {performanceAlerts.filter(a => a.status === 'active').length > 0 && (
        <Alert
          message={`当前有 ${performanceAlerts.filter(a => a.status === 'active').length} 个活跃告警`}
          description="系统检测到性能异常，请及时处理"
          type="warning"
          showIcon
          action={
            <Button size="small" onClick={() => setActiveTab('alerts')}>
              查看详情
            </Button>
          }
          style={{ marginBottom: '24px' }}
        />
      )}

      {/* 主管理界面 */}
      <Card>
        <div style={{ marginBottom: '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Space>
            <Select 
              value={selectedTimeRange} 
              onChange={setSelectedTimeRange}
              style={{ width: 120 }}
            >
              <Option value="1h">最近1小时</Option>
              <Option value="6h">最近6小时</Option>
              <Option value="24h">最近24小时</Option>
              <Option value="7d">最近7天</Option>
            </Select>
            <Button 
              icon={<ReloadOutlined />}
              loading={loading}
              onClick={refreshData}
            >
              刷新数据
            </Button>
          </Space>
          
          <Space>
            <Button icon={<BellOutlined />} onClick={() => setAlertModalVisible(true)}>
              创建告警
            </Button>
            <Button icon={<TrophyOutlined />} onClick={() => setOptimizationDrawerVisible(true)}>
              优化建议
            </Button>
            <Button icon={<DashboardOutlined />} onClick={() => setDashboardModalVisible(true)}>
              自定义仪表板
            </Button>
          </Space>
        </div>

        <Tabs activeKey={activeTab} onChange={setActiveTab}>
          <TabPane tab="监控仪表板" key="dashboard" icon={<DashboardOutlined />}>
            <Row gutter={[16, 16]}>
              <Col xs={24} lg={12}>
                <Card title="延迟分析" size="small">
                  <Row gutter={[16, 8]}>
                    <Col span={12}>
                      <Statistic title="P50" value={performanceMetrics.messageLatency.p50} suffix="ms" />
                    </Col>
                    <Col span={12}>
                      <Statistic title="P95" value={performanceMetrics.messageLatency.p95} suffix="ms" />
                    </Col>
                    <Col span={12}>
                      <Statistic title="P99" value={performanceMetrics.messageLatency.p99} suffix="ms" />
                    </Col>
                    <Col span={12}>
                      <Statistic title="平均" value={performanceMetrics.messageLatency.average} suffix="ms" />
                    </Col>
                  </Row>
                </Card>
              </Col>
              
              <Col xs={24} lg={12}>
                <Card title="系统资源使用" size="small">
                  <div style={{ marginBottom: '12px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                      <Text style={{ fontSize: '12px' }}>CPU</Text>
                      <Text style={{ fontSize: '12px' }}>{performanceMetrics.resourceUsage.cpu.toFixed(1)}%</Text>
                    </div>
                    <Progress percent={performanceMetrics.resourceUsage.cpu} size="small" />
                  </div>
                  
                  <div style={{ marginBottom: '12px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                      <Text style={{ fontSize: '12px' }}>内存</Text>
                      <Text style={{ fontSize: '12px' }}>{performanceMetrics.resourceUsage.memory.toFixed(1)}%</Text>
                    </div>
                    <Progress percent={performanceMetrics.resourceUsage.memory} size="small" />
                  </div>
                  
                  <div style={{ marginBottom: '12px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                      <Text style={{ fontSize: '12px' }}>网络</Text>
                      <Text style={{ fontSize: '12px' }}>{performanceMetrics.resourceUsage.network.toFixed(1)}%</Text>
                    </div>
                    <Progress percent={performanceMetrics.resourceUsage.network} size="small" />
                  </div>
                  
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                      <Text style={{ fontSize: '12px' }}>磁盘</Text>
                      <Text style={{ fontSize: '12px' }}>{performanceMetrics.resourceUsage.disk.toFixed(1)}%</Text>
                    </div>
                    <Progress percent={performanceMetrics.resourceUsage.disk} size="small" />
                  </div>
                </Card>
              </Col>
              
              <Col span={24}>
                <Card title="性能趋势图" size="small">
                  <div style={{ textAlign: 'center', padding: '60px 0' }}>
                    <Text type="secondary">性能趋势图表组件开发中...</Text>
                  </div>
                </Card>
              </Col>
            </Row>
          </TabPane>
          
          <TabPane tab="告警管理" key="alerts" icon={<BellOutlined />}>
            <div style={{ marginBottom: '16px' }}>
              <Space>
                <Button type="primary" icon={<BellOutlined />} onClick={() => setAlertModalVisible(true)}>
                  创建告警规则
                </Button>
                <Text type="secondary">
                  活跃告警: {performanceAlerts.filter(a => a.status === 'active').length}
                </Text>
              </Space>
            </div>
            
            {performanceAlerts.filter(a => a.status === 'active').length > 0 && (
              <>
                <Title level={5}>活跃告警</Title>
                <Table
                  columns={activeAlertColumns}
                  dataSource={performanceAlerts.filter(a => a.status === 'active')}
                  rowKey="id"
                  size="small"
                  pagination={false}
                  style={{ marginBottom: '24px' }}
                />
              </>
            )}
            
            <Title level={5}>告警规则</Title>
            <Table
              columns={alertColumns}
              dataSource={alertRules}
              rowKey="id"
              size="small"
              pagination={{ pageSize: 10 }}
            />
          </TabPane>
          
          <TabPane tab="性能优化" key="optimization" icon={<RocketOutlined />}>
            <div style={{ marginBottom: '16px' }}>
              <Alert
                message="优化建议"
                description={`系统分析发现 ${optimizationSuggestions.filter(s => s.status === 'pending').length} 个待处理的优化建议，预计可提升系统性能 15-30%`}
                type="info"
                showIcon
                action={
                  <Button size="small" onClick={() => setOptimizationDrawerVisible(true)}>
                    查看详情
                  </Button>
                }
                style={{ marginBottom: '16px' }}
              />
            </div>
            
            <Table
              columns={optimizationColumns}
              dataSource={optimizationSuggestions}
              rowKey="id"
              size="small"
              pagination={{ pageSize: 8 }}
              scroll={{ x: 1000 }}
            />
          </TabPane>
          
          <TabPane tab="系统分析" key="analysis" icon={<BarChartOutlined />}>
            <Row gutter={[16, 16]}>
              <Col xs={24} lg={12}>
                <Card title="系统健康评分" size="small">
                  <Row gutter={[16, 16]}>
                    <Col span={12}>
                      <div style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: '32px', fontWeight: 'bold', color: getHealthColor(performanceMetrics.systemHealth.availability) }}>
                          {performanceMetrics.systemHealth.availability.toFixed(1)}%
                        </div>
                        <Text type="secondary">可用性</Text>
                      </div>
                    </Col>
                    <Col span={12}>
                      <div style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: '32px', fontWeight: 'bold', color: getHealthColor(performanceMetrics.systemHealth.reliability) }}>
                          {performanceMetrics.systemHealth.reliability.toFixed(1)}%
                        </div>
                        <Text type="secondary">可靠性</Text>
                      </div>
                    </Col>
                    <Col span={24}>
                      <div style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: '32px', fontWeight: 'bold', color: getHealthColor(performanceMetrics.systemHealth.performance) }}>
                          {performanceMetrics.systemHealth.performance.toFixed(1)}%
                        </div>
                        <Text type="secondary">性能评分</Text>
                      </div>
                    </Col>
                  </Row>
                </Card>
              </Col>
              
              <Col xs={24} lg={12}>
                <Card title="错误分析" size="small">
                  <div style={{ marginBottom: '12px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Text>连接错误</Text>
                      <Text>{performanceMetrics.errorRates.connection}%</Text>
                    </div>
                    <Progress 
                      percent={performanceMetrics.errorRates.connection * 20} 
                      size="small" 
                      showInfo={false}
                    />
                  </div>
                  
                  <div style={{ marginBottom: '12px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Text>超时错误</Text>
                      <Text>{performanceMetrics.errorRates.timeout}%</Text>
                    </div>
                    <Progress 
                      percent={performanceMetrics.errorRates.timeout * 20} 
                      size="small" 
                      showInfo={false}
                    />
                  </div>
                  
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Text>处理错误</Text>
                      <Text>{performanceMetrics.errorRates.processing}%</Text>
                    </div>
                    <Progress 
                      percent={performanceMetrics.errorRates.processing * 20} 
                      size="small" 
                      showInfo={false}
                    />
                  </div>
                </Card>
              </Col>
              
              <Col span={24}>
                <Card title="性能基准对比" size="small">
                  <Row gutter={[16, 16]}>
                    <Col span={8}>
                      <div style={{ textAlign: 'center', padding: '20px 0' }}>
                        <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#52c41a' }}>+25%</div>
                        <Text type="secondary">相比上月吞吐量提升</Text>
                      </div>
                    </Col>
                    <Col span={8}>
                      <div style={{ textAlign: 'center', padding: '20px 0' }}>
                        <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#52c41a' }}>-18%</div>
                        <Text type="secondary">相比上月延迟降低</Text>
                      </div>
                    </Col>
                    <Col span={8}>
                      <div style={{ textAlign: 'center', padding: '20px 0' }}>
                        <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#ff4d4f' }}>+0.05%</div>
                        <Text type="secondary">相比上月错误率变化</Text>
                      </div>
                    </Col>
                  </Row>
                </Card>
              </Col>
            </Row>
          </TabPane>
        </Tabs>
      </Card>

      {/* 优化建议抽屉 */}
      <Drawer
        title="性能优化建议详情"
        placement="right"
        width={800}
        visible={optimizationDrawerVisible}
        onClose={() => setOptimizationDrawerVisible(false)}
      >
        <div style={{ marginBottom: '16px' }}>
          <Alert
            message="优化建议总览"
            description={`共 ${optimizationSuggestions.length} 个建议，其中高优先级 ${optimizationSuggestions.filter(s => s.priority >= 80).length} 个`}
            type="info"
            showIcon
          />
        </div>
        
        <List
          dataSource={optimizationSuggestions.sort((a, b) => b.priority - a.priority)}
          renderItem={item => (
            <List.Item>
              <List.Item.Meta
                avatar={
                  <Avatar 
                    style={{ 
                      backgroundColor: item.impact === 'high' ? '#ff4d4f' : 
                                      item.impact === 'medium' ? '#faad14' : '#52c41a'
                    }}
                  >
                    {item.impact.charAt(0).toUpperCase()}
                  </Avatar>
                }
                title={
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Text strong>{item.title}</Text>
                    <Tag color="blue">{item.priority}</Tag>
                  </div>
                }
                description={
                  <div>
                    <Text style={{ fontSize: '12px' }}>{item.description}</Text>
                    <div style={{ marginTop: '8px' }}>
                      <Tag color="green">预期收益: {item.metrics.expectedImprovement}</Tag>
                      <Tag color="orange">预估时间: {item.implementation.estimatedTime}</Tag>
                    </div>
                  </div>
                }
              />
            </List.Item>
          )}
        />
      </Drawer>
    </div>
  )
}

export default MonitoringPerformanceOptimizationPage
