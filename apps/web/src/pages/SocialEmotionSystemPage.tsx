import React, { useState, useEffect, useRef } from 'react'
import {
  Card,
  Row,
  Col,
  Tabs,
  Button,
  Table,
  Tag,
  Alert,
  Modal,
  Form,
  Input,
  Select,
  Switch,
  Progress,
  Divider,
  Space,
  List,
  Timeline,
  Badge,
  Tooltip,
  Rate,
  Statistic,
  Steps,
  Tree,
  Collapse,
  message,
  Spin,
  Descriptions,
  Radio,
  Slider,
  Avatar,
  Empty,
  Drawer,
  Dropdown,
  Menu,
  Typography,
  DatePicker,
  RangePicker,
  CheckCard
} from 'antd'
import {
  DashboardOutlined,
  HeartOutlined,
  TeamOutlined,
  GlobalOutlined,
  BrainOutlined,
  ShieldOutlined,
  SettingOutlined,
  MonitorOutlined,
  ThunderboltOutlined,
  RobotOutlined,
  UserOutlined,
  FireOutlined,
  TrophyOutlined,
  ExperimentOutlined,
  LineChartOutlined,
  BarChartOutlined,
  PieChartOutlined,
  RadarChartOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  SyncOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  StopOutlined,
  ReloadOutlined,
  EyeOutlined,
  BellOutlined,
  MessageOutlined,
  BookOutlined,
  SafetyCertificateOutlined,
  AuditOutlined,
  ApiOutlined,
  DatabaseOutlined,
  CloudOutlined,
  SecurityScanOutlined,
  BugOutlined,
  ToolOutlined,
  NodeIndexOutlined,
  FunctionOutlined
} from '@ant-design/icons'
import * as d3 from 'd3'

const { TabPane } = Tabs
const { Option } = Select
const { TextArea } = Input
const { Panel } = Collapse
const { Step } = Steps
const { Title, Text, Paragraph } = Typography
const { RangePicker: DateRangePicker } = DatePicker

interface SystemStatus {
  module_name: string
  status: 'running' | 'stopped' | 'error' | 'starting' | 'stopping'
  health_score: number
  uptime: number
  cpu_usage: number
  memory_usage: number
  last_updated: string
  error_count: number
  warning_count: number
  performance_metrics: Record<string, number>
}

interface ServiceMetrics {
  emotion_recognition: {
    accuracy: number
    response_time: number
    throughput: number
    error_rate: number
  }
  empathy_response: {
    quality_score: number
    user_satisfaction: number
    cultural_adaptation: number
    response_latency: number
  }
  social_intelligence: {
    decision_accuracy: number
    context_understanding: number
    relationship_mapping: number
    conflict_resolution: number
  }
  privacy_protection: {
    compliance_score: number
    data_anonymization: number
    consent_coverage: number
    audit_completeness: number
  }
}

interface UserInteraction {
  id: string
  user_id: string
  session_id: string
  interaction_type: string
  timestamp: string
  emotion_detected: string[]
  empathy_score: number
  cultural_context: string
  privacy_level: string
  outcome_quality: number
  user_feedback: number
}

interface SystemAlert {
  id: string
  severity: 'info' | 'warning' | 'error' | 'critical'
  module: string
  title: string
  description: string
  timestamp: string
  resolved: boolean
  resolution_time?: string
  impact_level: number
}

interface ConfigurationItem {
  category: string
  key: string
  value: any
  description: string
  editable: boolean
  requires_restart: boolean
  validation_rules?: any
}

const SocialEmotionSystemPage: React.FC = () => {
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('dashboard')
  const [systemStatus, setSystemStatus] = useState<SystemStatus[]>([])
  const [serviceMetrics, setServiceMetrics] = useState<ServiceMetrics | null>(null)
  const [userInteractions, setUserInteractions] = useState<UserInteraction[]>([])
  const [systemAlerts, setSystemAlerts] = useState<SystemAlert[]>([])
  const [configurations, setConfigurations] = useState<ConfigurationItem[]>([])
  const [realTimeMode, setRealTimeMode] = useState(true)
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [selectedModule, setSelectedModule] = useState<string | null>(null)
  const [modalVisible, setModalVisible] = useState(false)
  const [drawerVisible, setDrawerVisible] = useState(false)
  const [selectedTimeRange, setSelectedTimeRange] = useState('24h')
  
  const [form] = Form.useForm()
  const overviewChartRef = useRef<HTMLDivElement>(null)
  const performanceChartRef = useRef<HTMLDivElement>(null)
  const userEngagementChartRef = useRef<HTMLDivElement>(null)
  const systemHealthChartRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    initializeData()
    
    if (autoRefresh) {
      const interval = setInterval(() => {
        refreshData()
      }, 30000) // 30秒刷新一次
      
      return () => clearInterval(interval)
    }
  }, [autoRefresh])

  useEffect(() => {
    if (activeTab === 'dashboard') {
      renderDashboardCharts()
    }
  }, [activeTab, systemStatus, serviceMetrics, userInteractions])

  const initializeData = async () => {
    setLoading(true)
    try {
      await Promise.all([
        loadSystemStatus(),
        loadServiceMetrics(),
        loadUserInteractions(),
        loadSystemAlerts(),
        loadConfigurations()
      ])
    } catch (error) {
      console.error('初始化数据失败:', error)
      message.error('数据加载失败')
    } finally {
      setLoading(false)
    }
  }

  const refreshData = async () => {
    try {
      await Promise.all([
        loadSystemStatus(),
        loadServiceMetrics(),
        loadUserInteractions()
      ])
    } catch (error) {
      console.error('刷新数据失败:', error)
    }
  }

  const loadSystemStatus = async () => {
    // 模拟系统状态数据
    const mockStatus: SystemStatus[] = [
      {
        module_name: '情感识别引擎',
        status: 'running',
        health_score: 95,
        uptime: 99.8,
        cpu_usage: 45,
        memory_usage: 67,
        last_updated: new Date().toISOString(),
        error_count: 2,
        warning_count: 5,
        performance_metrics: {
          accuracy: 0.94,
          latency: 120,
          throughput: 1500
        }
      },
      {
        module_name: '共情响应生成器',
        status: 'running',
        health_score: 88,
        uptime: 99.5,
        cpu_usage: 32,
        memory_usage: 54,
        last_updated: new Date().toISOString(),
        error_count: 1,
        warning_count: 3,
        performance_metrics: {
          quality_score: 0.89,
          satisfaction: 0.92,
          response_time: 200
        }
      },
      {
        module_name: '社交智能决策',
        status: 'running',
        health_score: 91,
        uptime: 99.7,
        cpu_usage: 28,
        memory_usage: 43,
        last_updated: new Date().toISOString(),
        error_count: 0,
        warning_count: 2,
        performance_metrics: {
          decision_accuracy: 0.87,
          context_score: 0.91
        }
      },
      {
        module_name: '隐私保护系统',
        status: 'running',
        health_score: 97,
        uptime: 99.9,
        cpu_usage: 15,
        memory_usage: 25,
        last_updated: new Date().toISOString(),
        error_count: 0,
        warning_count: 1,
        performance_metrics: {
          compliance_score: 0.98,
          anonymization_rate: 1.0
        }
      },
      {
        module_name: '文化适应引擎',
        status: 'running',
        health_score: 85,
        uptime: 99.3,
        cpu_usage: 38,
        memory_usage: 61,
        last_updated: new Date().toISOString(),
        error_count: 3,
        warning_count: 7,
        performance_metrics: {
          cultural_accuracy: 0.86,
          adaptation_speed: 0.91
        }
      }
    ]
    
    setSystemStatus(mockStatus)
  }

  const loadServiceMetrics = async () => {
    const mockMetrics: ServiceMetrics = {
      emotion_recognition: {
        accuracy: 94.2,
        response_time: 120,
        throughput: 1500,
        error_rate: 0.8
      },
      empathy_response: {
        quality_score: 89.5,
        user_satisfaction: 92.1,
        cultural_adaptation: 87.3,
        response_latency: 200
      },
      social_intelligence: {
        decision_accuracy: 87.8,
        context_understanding: 91.2,
        relationship_mapping: 85.6,
        conflict_resolution: 88.9
      },
      privacy_protection: {
        compliance_score: 98.2,
        data_anonymization: 100,
        consent_coverage: 95.7,
        audit_completeness: 96.8
      }
    }
    
    setServiceMetrics(mockMetrics)
  }

  const loadUserInteractions = async () => {
    const mockInteractions: UserInteraction[] = [
      {
        id: 'interaction_001',
        user_id: 'user_12345',
        session_id: 'session_abc',
        interaction_type: 'empathy_response',
        timestamp: new Date(Date.now() - 300000).toISOString(),
        emotion_detected: ['sadness', 'frustration'],
        empathy_score: 8.5,
        cultural_context: 'western_individualistic',
        privacy_level: 'high',
        outcome_quality: 9.2,
        user_feedback: 5
      },
      {
        id: 'interaction_002',
        user_id: 'user_67890',
        session_id: 'session_def',
        interaction_type: 'social_decision',
        timestamp: new Date(Date.now() - 600000).toISOString(),
        emotion_detected: ['anxiety', 'hope'],
        empathy_score: 7.8,
        cultural_context: 'eastern_collectivistic',
        privacy_level: 'medium',
        outcome_quality: 8.7,
        user_feedback: 4
      }
    ]
    
    setUserInteractions(mockInteractions)
  }

  const loadSystemAlerts = async () => {
    const mockAlerts: SystemAlert[] = [
      {
        id: 'alert_001',
        severity: 'warning',
        module: '情感识别引擎',
        title: 'CPU使用率偏高',
        description: '情感识别引擎CPU使用率持续超过80%，建议检查处理队列',
        timestamp: new Date(Date.now() - 3600000).toISOString(),
        resolved: false,
        impact_level: 3
      },
      {
        id: 'alert_002',
        severity: 'info',
        module: '隐私保护系统',
        title: '合规检查完成',
        description: 'GDPR合规性检查已完成，所有项目均通过审核',
        timestamp: new Date(Date.now() - 7200000).toISOString(),
        resolved: true,
        resolution_time: new Date(Date.now() - 7000000).toISOString(),
        impact_level: 1
      }
    ]
    
    setSystemAlerts(mockAlerts)
  }

  const loadConfigurations = async () => {
    const mockConfigs: ConfigurationItem[] = [
      {
        category: '情感识别',
        key: 'emotion_threshold',
        value: 0.75,
        description: '情感识别的置信度阈值',
        editable: true,
        requires_restart: false
      },
      {
        category: '隐私保护',
        key: 'data_retention_days',
        value: 365,
        description: '个人数据保留天数',
        editable: true,
        requires_restart: true
      },
      {
        category: '文化适应',
        key: 'default_culture',
        value: 'neutral',
        description: '默认文化背景设置',
        editable: true,
        requires_restart: false
      }
    ]
    
    setConfigurations(mockConfigs)
  }

  const renderDashboardCharts = () => {
    renderOverviewChart()
    renderPerformanceChart()
    renderUserEngagementChart()
    renderSystemHealthChart()
  }

  const renderOverviewChart = () => {
    if (!overviewChartRef.current || !systemStatus.length) return

    const container = d3.select(overviewChartRef.current)
    container.selectAll('*').remove()

    const data = systemStatus.map(status => ({
      module: status.module_name.slice(0, 4),
      health: status.health_score
    }))

    const margin = { top: 20, right: 30, bottom: 40, left: 40 }
    const width = 400 - margin.left - margin.right
    const height = 200 - margin.top - margin.bottom

    const svg = container
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    const xScale = d3.scaleBand()
      .domain(data.map(d => d.module))
      .range([0, width])
      .padding(0.1)

    const yScale = d3.scaleLinear()
      .domain([0, 100])
      .range([height, 0])

    svg.selectAll('.bar')
      .data(data)
      .enter()
      .append('rect')
      .attr('class', 'bar')
      .attr('x', d => xScale(d.module)!)
      .attr('y', d => yScale(d.health))
      .attr('width', xScale.bandwidth())
      .attr('height', d => height - yScale(d.health))
      .attr('fill', d => {
        if (d.health >= 90) return '#52c41a'
        if (d.health >= 80) return '#faad14'
        if (d.health >= 70) return '#fa8c16'
        return '#f5222d'
      })
      .attr('opacity', 0.8)

    svg.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale))

    svg.append('g')
      .call(d3.axisLeft(yScale))
  }

  const renderPerformanceChart = () => {
    if (!performanceChartRef.current || !serviceMetrics) return

    const container = d3.select(performanceChartRef.current)
    container.selectAll('*').remove()

    const data = [
      { service: '情感识别', value: serviceMetrics.emotion_recognition.accuracy },
      { service: '共情响应', value: serviceMetrics.empathy_response.quality_score },
      { service: '社交智能', value: serviceMetrics.social_intelligence.decision_accuracy },
      { service: '隐私保护', value: serviceMetrics.privacy_protection.compliance_score }
    ]

    const width = 300
    const height = 200
    const radius = Math.min(width, height) / 2

    const svg = container
      .append('svg')
      .attr('width', width)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${width / 2}, ${height / 2})`)

    const color = d3.scaleOrdinal(d3.schemeCategory10)

    const pie = d3.pie<any>()
      .value(d => d.value)

    const arc = d3.arc<any>()
      .innerRadius(0)
      .outerRadius(radius)

    svg.selectAll('path')
      .data(pie(data))
      .enter()
      .append('path')
      .attr('d', arc)
      .attr('fill', (d, i) => color(i.toString()))
      .attr('opacity', 0.8)

    svg.selectAll('text')
      .data(pie(data))
      .enter()
      .append('text')
      .attr('transform', d => `translate(${arc.centroid(d)})`)
      .attr('text-anchor', 'middle')
      .text(d => `${d.data.value.toFixed(1)}%`)
      .attr('font-size', '10px')
      .attr('fill', 'white')
  }

  const renderUserEngagementChart = () => {
    if (!userEngagementChartRef.current) return

    const container = d3.select(userEngagementChartRef.current)
    container.selectAll('*').remove()

    // 模拟用户参与度时序数据
    const timeSeriesData = Array.from({ length: 24 }, (_, i) => ({
      hour: i,
      interactions: Math.floor(Math.random() * 100) + 20,
      satisfaction: Math.random() * 2 + 8
    }))

    const margin = { top: 20, right: 30, bottom: 40, left: 40 }
    const width = 400 - margin.left - margin.right
    const height = 200 - margin.top - margin.bottom

    const svg = container
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    const xScale = d3.scaleLinear()
      .domain([0, 23])
      .range([0, width])

    const yScale = d3.scaleLinear()
      .domain([0, d3.max(timeSeriesData, d => d.interactions)!])
      .range([height, 0])

    const line = d3.line<any>()
      .x(d => xScale(d.hour))
      .y(d => yScale(d.interactions))
      .curve(d3.curveMonotoneX)

    svg.append('path')
      .datum(timeSeriesData)
      .attr('fill', 'none')
      .attr('stroke', '#1890ff')
      .attr('stroke-width', 2)
      .attr('d', line)

    svg.selectAll('.dot')
      .data(timeSeriesData)
      .enter()
      .append('circle')
      .attr('class', 'dot')
      .attr('cx', d => xScale(d.hour))
      .attr('cy', d => yScale(d.interactions))
      .attr('r', 3)
      .attr('fill', '#1890ff')

    svg.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale))

    svg.append('g')
      .call(d3.axisLeft(yScale))
  }

  const renderSystemHealthChart = () => {
    if (!systemHealthChartRef.current || !systemStatus.length) return

    const container = d3.select(systemHealthChartRef.current)
    container.selectAll('*').remove()

    const data = systemStatus.map(status => ({
      module: status.module_name,
      cpu: status.cpu_usage,
      memory: status.memory_usage,
      health: status.health_score
    }))

    const margin = { top: 20, right: 30, bottom: 60, left: 80 }
    const width = 400 - margin.left - margin.right
    const height = 250 - margin.top - margin.bottom

    const svg = container
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    const yScale = d3.scaleBand()
      .domain(data.map(d => d.module.slice(0, 8)))
      .range([0, height])
      .padding(0.1)

    const xScale = d3.scaleLinear()
      .domain([0, 100])
      .range([0, width])

    // CPU使用率条
    svg.selectAll('.cpu-bar')
      .data(data)
      .enter()
      .append('rect')
      .attr('class', 'cpu-bar')
      .attr('x', 0)
      .attr('y', d => yScale(d.module.slice(0, 8))! + yScale.bandwidth() / 4)
      .attr('width', d => xScale(d.cpu))
      .attr('height', yScale.bandwidth() / 4)
      .attr('fill', '#fa8c16')
      .attr('opacity', 0.7)

    // 内存使用率条
    svg.selectAll('.memory-bar')
      .data(data)
      .enter()
      .append('rect')
      .attr('class', 'memory-bar')
      .attr('x', 0)
      .attr('y', d => yScale(d.module.slice(0, 8))! + yScale.bandwidth() * 3/4)
      .attr('width', d => xScale(d.memory))
      .attr('height', yScale.bandwidth() / 4)
      .attr('fill', '#1890ff')
      .attr('opacity', 0.7)

    svg.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale))

    svg.append('g')
      .call(d3.axisLeft(yScale))
  }

  const handleModuleOperation = async (moduleName: string, operation: 'start' | 'stop' | 'restart') => {
    try {
      message.loading({ content: `${operation === 'start' ? '启动' : operation === 'stop' ? '停止' : '重启'}模块中...`, key: 'module-op' })
      
      // 模拟API调用
      await new Promise(resolve => setTimeout(resolve, 2000))
      
      message.success({ content: `${moduleName} ${operation === 'start' ? '启动' : operation === 'stop' ? '停止' : '重启'}成功`, key: 'module-op' })
      
      // 刷新系统状态
      await loadSystemStatus()
    } catch (error) {
      message.error({ content: '操作失败', key: 'module-op' })
    }
  }

  const handleConfigurationUpdate = async (values: any) => {
    try {
      message.loading({ content: '更新配置中...', key: 'config-update' })
      
      // 模拟API调用
      await new Promise(resolve => setTimeout(resolve, 1500))
      
      message.success({ content: '配置更新成功', key: 'config-update' })
      
      setModalVisible(false)
      form.resetFields()
      await loadConfigurations()
    } catch (error) {
      message.error({ content: '配置更新失败', key: 'config-update' })
    }
  }

  const systemStatusColumns = [
    {
      title: '模块名称',
      dataIndex: 'module_name',
      key: 'module_name',
      render: (text: string, record: SystemStatus) => (
        <div>
          <div style={{ fontWeight: 'bold' }}>{text}</div>
          <Tag color={record.status === 'running' ? 'green' : record.status === 'error' ? 'red' : 'orange'}>
            {record.status.toUpperCase()}
          </Tag>
        </div>
      )
    },
    {
      title: '健康度',
      dataIndex: 'health_score',
      key: 'health_score',
      render: (score: number) => (
        <div style={{ width: 100 }}>
          <Progress 
            percent={score} 
            size="small" 
            status={score >= 90 ? 'success' : score >= 80 ? 'normal' : 'exception'}
            format={() => `${score}%`}
          />
        </div>
      )
    },
    {
      title: '运行时间',
      dataIndex: 'uptime',
      key: 'uptime',
      render: (uptime: number) => `${uptime}%`
    },
    {
      title: 'CPU',
      dataIndex: 'cpu_usage',
      key: 'cpu_usage',
      render: (usage: number) => (
        <Progress 
          percent={usage} 
          size="small" 
          strokeColor={usage > 80 ? '#f5222d' : usage > 60 ? '#faad14' : '#52c41a'}
          format={() => `${usage}%`}
        />
      )
    },
    {
      title: '内存',
      dataIndex: 'memory_usage',
      key: 'memory_usage',
      render: (usage: number) => (
        <Progress 
          percent={usage} 
          size="small" 
          strokeColor={usage > 80 ? '#f5222d' : usage > 60 ? '#faad14' : '#52c41a'}
          format={() => `${usage}%`}
        />
      )
    },
    {
      title: '错误/警告',
      key: 'issues',
      render: (_: any, record: SystemStatus) => (
        <Space>
          <Badge count={record.error_count} style={{ backgroundColor: '#f5222d' }} />
          <Badge count={record.warning_count} style={{ backgroundColor: '#faad14' }} />
        </Space>
      )
    },
    {
      title: '操作',
      key: 'action',
      render: (_: any, record: SystemStatus) => (
        <Dropdown
          overlay={
            <Menu>
              <Menu.Item 
                key="restart" 
                icon={<ReloadOutlined />}
                onClick={() => handleModuleOperation(record.module_name, 'restart')}
              >
                重启
              </Menu.Item>
              <Menu.Item 
                key="stop" 
                icon={<StopOutlined />}
                onClick={() => handleModuleOperation(record.module_name, 'stop')}
                disabled={record.status === 'stopped'}
              >
                停止
              </Menu.Item>
              <Menu.Item 
                key="start" 
                icon={<PlayCircleOutlined />}
                onClick={() => handleModuleOperation(record.module_name, 'start')}
                disabled={record.status === 'running'}
              >
                启动
              </Menu.Item>
              <Menu.Divider />
              <Menu.Item 
                key="details" 
                icon={<EyeOutlined />}
                onClick={() => {
                  setSelectedModule(record.module_name)
                  setDrawerVisible(true)
                }}
              >
                查看详情
              </Menu.Item>
            </Menu>
          }
        >
          <Button size="small">
            操作 <BarChartOutlined />
          </Button>
        </Dropdown>
      )
    }
  ]

  const renderDashboard = () => (
    <div>
      {/* 系统总览 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="运行模块"
              value={systemStatus.filter(s => s.status === 'running').length}
              total={systemStatus.length}
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#3f8600' }}
              formatter={(value, total) => `${value}/${total}`}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="平均健康度"
              value={systemStatus.reduce((sum, s) => sum + s.health_score, 0) / systemStatus.length || 0}
              precision={1}
              suffix="%"
              prefix={<HeartOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="活跃用户"
              value={1247}
              prefix={<UserOutlined />}
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="今日交互"
              value={8936}
              prefix={<MessageOutlined />}
              valueStyle={{ color: '#fa8c16' }}
            />
          </Card>
        </Col>
      </Row>

      {/* 图表展示 */}
      <Row gutter={[16, 16]}>
        <Col span={12}>
          <Card title="模块健康度" size="small">
            <div ref={overviewChartRef}></div>
          </Card>
        </Col>
        <Col span={12}>
          <Card title="服务性能分布" size="small">
            <div ref={performanceChartRef}></div>
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginTop: '16px' }}>
        <Col span={12}>
          <Card title="用户参与度趋势" size="small">
            <div ref={userEngagementChartRef}></div>
          </Card>
        </Col>
        <Col span={12}>
          <Card title="系统资源使用" size="small">
            <div ref={systemHealthChartRef}></div>
          </Card>
        </Col>
      </Row>

      {/* 实时告警 */}
      <Row gutter={[16, 16]} style={{ marginTop: '16px' }}>
        <Col span={24}>
          <Card 
            title="实时告警" 
            size="small"
            extra={
              <Space>
                <Badge count={systemAlerts.filter(a => !a.resolved).length} size="small">
                  <BellOutlined />
                </Badge>
                <Button size="small" onClick={() => setActiveTab('alerts')}>查看全部</Button>
              </Space>
            }
          >
            <List
              size="small"
              dataSource={systemAlerts.filter(a => !a.resolved).slice(0, 5)}
              renderItem={(alert) => (
                <List.Item
                  actions={[
                    <Button type="link" size="small">处理</Button>,
                    <Button type="link" size="small">忽略</Button>
                  ]}
                >
                  <List.Item.Meta
                    avatar={
                      <Avatar 
                        icon={
                          alert.severity === 'critical' ? <WarningOutlined /> :
                          alert.severity === 'error' ? <CloseCircleOutlined /> :
                          alert.severity === 'warning' ? <ExclamationCircleOutlined /> :
                          <CheckCircleOutlined />
                        }
                        style={{
                          backgroundColor: 
                            alert.severity === 'critical' ? '#f5222d' :
                            alert.severity === 'error' ? '#fa541c' :
                            alert.severity === 'warning' ? '#faad14' :
                            '#52c41a'
                        }}
                      />
                    }
                    title={
                      <div>
                        <Tag color="blue">{alert.module}</Tag>
                        <span>{alert.title}</span>
                      </div>
                    }
                    description={
                      <div>
                        <div>{alert.description}</div>
                        <div style={{ fontSize: '12px', color: '#999', marginTop: '4px' }}>
                          {new Date(alert.timestamp).toLocaleString()}
                        </div>
                      </div>
                    }
                  />
                </List.Item>
              )}
            />
          </Card>
        </Col>
      </Row>
    </div>
  )

  return (
    <div style={{ padding: '24px', backgroundColor: '#f0f2f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <h1 style={{ fontSize: '28px', fontWeight: 'bold', color: '#1890ff', marginBottom: '8px' }}>
          <DashboardOutlined style={{ marginRight: '12px' }} />
          社交情感理解系统 - 统一控制台
        </h1>
        <p style={{ fontSize: '16px', color: '#666', margin: 0 }}>
          全面监控和管理社交情感AI系统的各个组件，确保系统稳定高效运行
        </p>
      </div>

      <div style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Space>
          <Select 
            value={selectedTimeRange} 
            onChange={setSelectedTimeRange}
            style={{ width: 120 }}
          >
            <Option value="1h">最近1小时</Option>
            <Option value="24h">最近24小时</Option>
            <Option value="7d">最近7天</Option>
            <Option value="30d">最近30天</Option>
          </Select>
          
          <Switch 
            checkedChildren="实时监控" 
            unCheckedChildren="手动刷新" 
            checked={realTimeMode}
            onChange={setRealTimeMode}
          />
          
          <Switch 
            checkedChildren="自动刷新" 
            unCheckedChildren="手动刷新" 
            checked={autoRefresh}
            onChange={setAutoRefresh}
          />
        </Space>
        
        <Space>
          <Button 
            type="primary" 
            icon={<SettingOutlined />}
            onClick={() => setModalVisible(true)}
          >
            系统配置
          </Button>
          <Button 
            icon={<SecurityScanOutlined />}
          >
            健康检查
          </Button>
          <Button 
            icon={<SyncOutlined />}
            loading={loading}
            onClick={refreshData}
          >
            刷新数据
          </Button>
        </Space>
      </div>

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab={
          <span>
            <DashboardOutlined />
            系统总览
          </span>
        } key="dashboard">
          {renderDashboard()}
        </TabPane>

        <TabPane tab={
          <span>
            <MonitorOutlined />
            模块状态
          </span>
        } key="modules">
          <Card title="系统模块监控">
            <Table
              columns={systemStatusColumns}
              dataSource={systemStatus}
              rowKey="module_name"
              loading={loading}
              pagination={false}
              size="small"
            />
          </Card>
        </TabPane>

        <TabPane tab={
          <span>
            <LineChartOutlined />
            性能监控
          </span>
        } key="performance">
          <Row gutter={[16, 16]}>
            <Col span={12}>
              <Card title="情感识别引擎" size="small">
                <div style={{ marginBottom: '16px' }}>
                  <Statistic 
                    title="准确率" 
                    value={serviceMetrics?.emotion_recognition.accuracy || 0} 
                    suffix="%" 
                    precision={1}
                  />
                </div>
                <div style={{ marginBottom: '8px' }}>
                  <Text>响应时间: {serviceMetrics?.emotion_recognition.response_time || 0}ms</Text>
                </div>
                <div style={{ marginBottom: '8px' }}>
                  <Text>吞吐量: {serviceMetrics?.emotion_recognition.throughput || 0}/min</Text>
                </div>
                <div>
                  <Text>错误率: {serviceMetrics?.emotion_recognition.error_rate || 0}%</Text>
                </div>
              </Card>
            </Col>
            
            <Col span={12}>
              <Card title="共情响应生成" size="small">
                <div style={{ marginBottom: '16px' }}>
                  <Statistic 
                    title="质量评分" 
                    value={serviceMetrics?.empathy_response.quality_score || 0} 
                    suffix="分" 
                    precision={1}
                  />
                </div>
                <div style={{ marginBottom: '8px' }}>
                  <Text>用户满意度: {serviceMetrics?.empathy_response.user_satisfaction || 0}%</Text>
                </div>
                <div style={{ marginBottom: '8px' }}>
                  <Text>文化适应: {serviceMetrics?.empathy_response.cultural_adaptation || 0}%</Text>
                </div>
                <div>
                  <Text>响应延迟: {serviceMetrics?.empathy_response.response_latency || 0}ms</Text>
                </div>
              </Card>
            </Col>
          </Row>
          
          <Row gutter={[16, 16]} style={{ marginTop: '16px' }}>
            <Col span={12}>
              <Card title="社交智能决策" size="small">
                <Space direction="vertical" style={{ width: '100%' }}>
                  <div>
                    <Text>决策准确率: </Text>
                    <Progress 
                      percent={serviceMetrics?.social_intelligence.decision_accuracy || 0} 
                      size="small" 
                    />
                  </div>
                  <div>
                    <Text>语境理解: </Text>
                    <Progress 
                      percent={serviceMetrics?.social_intelligence.context_understanding || 0} 
                      size="small" 
                    />
                  </div>
                  <div>
                    <Text>关系映射: </Text>
                    <Progress 
                      percent={serviceMetrics?.social_intelligence.relationship_mapping || 0} 
                      size="small" 
                    />
                  </div>
                </Space>
              </Card>
            </Col>
            
            <Col span={12}>
              <Card title="隐私保护系统" size="small">
                <Space direction="vertical" style={{ width: '100%' }}>
                  <div>
                    <Text>合规评分: </Text>
                    <Progress 
                      percent={serviceMetrics?.privacy_protection.compliance_score || 0} 
                      size="small" 
                      strokeColor="#52c41a"
                    />
                  </div>
                  <div>
                    <Text>数据匿名化: </Text>
                    <Progress 
                      percent={serviceMetrics?.privacy_protection.data_anonymization || 0} 
                      size="small" 
                      strokeColor="#1890ff"
                    />
                  </div>
                  <div>
                    <Text>同意覆盖率: </Text>
                    <Progress 
                      percent={serviceMetrics?.privacy_protection.consent_coverage || 0} 
                      size="small" 
                      strokeColor="#722ed1"
                    />
                  </div>
                </Space>
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab={
          <span>
            <BellOutlined />
            系统告警
          </span>
        } key="alerts">
          <Card title="系统告警管理">
            <List
              dataSource={systemAlerts}
              renderItem={(alert) => (
                <List.Item
                  actions={[
                    !alert.resolved && <Button type="primary" size="small">处理</Button>,
                    <Button type="link" size="small">详情</Button>
                  ].filter(Boolean)}
                >
                  <List.Item.Meta
                    avatar={
                      <Avatar 
                        icon={
                          alert.severity === 'critical' ? <WarningOutlined /> :
                          alert.severity === 'error' ? <CloseCircleOutlined /> :
                          alert.severity === 'warning' ? <ExclamationCircleOutlined /> :
                          <CheckCircleOutlined />
                        }
                        style={{
                          backgroundColor: 
                            alert.severity === 'critical' ? '#f5222d' :
                            alert.severity === 'error' ? '#fa541c' :
                            alert.severity === 'warning' ? '#faad14' :
                            '#52c41a'
                        }}
                      />
                    }
                    title={
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <div>
                          <Tag color="blue">{alert.module}</Tag>
                          <span>{alert.title}</span>
                          {alert.resolved && <Tag color="green" style={{ marginLeft: 8 }}>已解决</Tag>}
                        </div>
                        <div>
                          <Tag color={alert.severity === 'critical' ? 'red' : alert.severity === 'error' ? 'orange' : 'yellow'}>
                            {alert.severity.toUpperCase()}
                          </Tag>
                        </div>
                      </div>
                    }
                    description={
                      <div>
                        <div style={{ marginBottom: '8px' }}>{alert.description}</div>
                        <div style={{ fontSize: '12px', color: '#999' }}>
                          <span>发生时间: {new Date(alert.timestamp).toLocaleString()}</span>
                          {alert.resolution_time && (
                            <span style={{ marginLeft: '16px' }}>
                              解决时间: {new Date(alert.resolution_time).toLocaleString()}
                            </span>
                          )}
                        </div>
                      </div>
                    }
                  />
                </List.Item>
              )}
            />
          </Card>
        </TabPane>

        <TabPane tab={
          <span>
            <UserOutlined />
            用户活动
          </span>
        } key="users">
          <Card title="用户交互记录">
            <List
              dataSource={userInteractions}
              renderItem={(interaction) => (
                <List.Item
                  actions={[
                    <Button type="link" size="small">查看详情</Button>
                  ]}
                >
                  <List.Item.Meta
                    title={
                      <div>
                        <Tag color="geekblue">{interaction.interaction_type}</Tag>
                        <span>用户: {interaction.user_id}</span>
                      </div>
                    }
                    description={
                      <div>
                        <div style={{ marginBottom: '4px' }}>
                          检测情感: {interaction.emotion_detected.map(emotion => (
                            <Tag key={emotion} size="small" color="purple">{emotion}</Tag>
                          ))}
                        </div>
                        <div style={{ marginBottom: '4px' }}>
                          共情评分: <Rate disabled value={interaction.empathy_score / 2} style={{ fontSize: '12px' }} />
                          文化背景: <Tag size="small">{interaction.cultural_context}</Tag>
                        </div>
                        <div style={{ fontSize: '12px', color: '#999' }}>
                          交互时间: {new Date(interaction.timestamp).toLocaleString()}
                          | 结果质量: {interaction.outcome_quality}/10
                          | 用户反馈: <Rate disabled value={interaction.user_feedback} style={{ fontSize: '10px' }} />
                        </div>
                      </div>
                    }
                  />
                </List.Item>
              )}
            />
          </Card>
        </TabPane>

        <TabPane tab={
          <span>
            <SettingOutlined />
            系统配置
          </span>
        } key="settings">
          <Row gutter={[16, 16]}>
            {configurations.map((config, index) => (
              <Col span={8} key={index}>
                <Card size="small" title={config.category}>
                  <div style={{ marginBottom: '8px' }}>
                    <Text strong>{config.key}</Text>
                  </div>
                  <div style={{ marginBottom: '8px' }}>
                    <Text>当前值: </Text>
                    <Tag color="blue">{String(config.value)}</Tag>
                  </div>
                  <div style={{ marginBottom: '8px', fontSize: '12px', color: '#666' }}>
                    {config.description}
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                      {config.requires_restart && (
                        <Tag size="small" color="orange">需重启</Tag>
                      )}
                    </div>
                    <Button 
                      size="small" 
                      type="link" 
                      disabled={!config.editable}
                    >
                      编辑
                    </Button>
                  </div>
                </Card>
              </Col>
            ))}
          </Row>
        </TabPane>
      </Tabs>

      {/* 模块详情抽屉 */}
      <Drawer
        title={`${selectedModule} - 详细信息`}
        placement="right"
        width={600}
        onClose={() => setDrawerVisible(false)}
        visible={drawerVisible}
      >
        {selectedModule && systemStatus.find(s => s.module_name === selectedModule) && (
          <div>
            <Descriptions title="模块状态" bordered column={2} size="small">
              <Descriptions.Item label="运行状态">
                <Tag color={systemStatus.find(s => s.module_name === selectedModule)!.status === 'running' ? 'green' : 'red'}>
                  {systemStatus.find(s => s.module_name === selectedModule)!.status.toUpperCase()}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="健康评分">
                {systemStatus.find(s => s.module_name === selectedModule)!.health_score}%
              </Descriptions.Item>
              <Descriptions.Item label="运行时间">
                {systemStatus.find(s => s.module_name === selectedModule)!.uptime}%
              </Descriptions.Item>
              <Descriptions.Item label="CPU使用率">
                {systemStatus.find(s => s.module_name === selectedModule)!.cpu_usage}%
              </Descriptions.Item>
              <Descriptions.Item label="内存使用率">
                {systemStatus.find(s => s.module_name === selectedModule)!.memory_usage}%
              </Descriptions.Item>
              <Descriptions.Item label="最后更新">
                {new Date(systemStatus.find(s => s.module_name === selectedModule)!.last_updated).toLocaleString()}
              </Descriptions.Item>
            </Descriptions>
            
            <Divider />
            
            <Title level={5}>性能指标</Title>
            {Object.entries(systemStatus.find(s => s.module_name === selectedModule)!.performance_metrics).map(([key, value]) => (
              <div key={key} style={{ marginBottom: '8px' }}>
                <Text>{key}: </Text>
                <Tag color="cyan">{typeof value === 'number' ? value.toFixed(2) : value}</Tag>
              </div>
            ))}
            
            <Divider />
            
            <Space>
              <Button 
                type="primary" 
                icon={<ReloadOutlined />}
                onClick={() => handleModuleOperation(selectedModule, 'restart')}
              >
                重启模块
              </Button>
              <Button icon={<EyeOutlined />}>查看日志</Button>
              <Button icon={<SettingOutlined />}>模块配置</Button>
            </Space>
          </div>
        )}
      </Drawer>

      {/* 系统配置模态框 */}
      <Modal
        title="系统配置管理"
        visible={modalVisible}
        onCancel={() => setModalVisible(false)}
        footer={[
          <Button key="cancel" onClick={() => setModalVisible(false)}>
            取消
          </Button>,
          <Button key="submit" type="primary" onClick={() => form.submit()}>
            应用配置
          </Button>
        ]}
        width={800}
      >
        <Form form={form} layout="vertical" onFinish={handleConfigurationUpdate}>
          <Alert
            message="配置变更提醒"
            description="部分配置项修改后需要重启相关模块才能生效，请谨慎操作"
            type="warning"
            showIcon
            style={{ marginBottom: '16px' }}
          />
          
          <Tabs>
            <TabPane tab="情感识别" key="emotion">
              <Form.Item label="识别阈值" name="emotion_threshold">
                <Slider min={0.1} max={1} step={0.05} marks={{ 0.1: '0.1', 0.5: '0.5', 1: '1.0' }} />
              </Form.Item>
              <Form.Item label="模型版本" name="model_version">
                <Select placeholder="选择模型版本">
                  <Option value="v1.2.3">v1.2.3 (稳定版)</Option>
                  <Option value="v1.3.0-beta">v1.3.0-beta (测试版)</Option>
                </Select>
              </Form.Item>
            </TabPane>
            
            <TabPane tab="隐私保护" key="privacy">
              <Form.Item label="数据保留天数" name="retention_days">
                <InputNumber min={1} max={3650} style={{ width: '100%' }} />
              </Form.Item>
              <Form.Item label="自动匿名化" name="auto_anonymization">
                <Switch checkedChildren="开启" unCheckedChildren="关闭" />
              </Form.Item>
            </TabPane>
            
            <TabPane tab="文化适应" key="culture">
              <Form.Item label="默认文化背景" name="default_culture">
                <Select placeholder="选择默认文化背景">
                  <Option value="neutral">中性</Option>
                  <Option value="western">西方文化</Option>
                  <Option value="eastern">东方文化</Option>
                </Select>
              </Form.Item>
            </TabPane>
          </Tabs>
        </Form>
      </Modal>
    </div>
  )
}

export default SocialEmotionSystemPage