import React, { useState, useEffect } from 'react'
import { Card, Row, Col, Statistic, Progress, Select, Button, Space, Typography, Alert, Tag, Table, Tooltip } from 'antd'
import { 
  BarChartOutlined, 
  ThunderboltOutlined, 
  ReloadOutlined, 
  DownloadOutlined, 
  SettingOutlined,
  MonitorOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  ClockCircleOutlined,
  GlobalOutlined,
  ApiOutlined,
  DatabaseOutlined,
  NetworkOutlined,
  RiseOutlined,
  FallOutlined,
  SyncOutlined
} from '@ant-design/icons'
import { 
  LineChart, 
  Line, 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip as ChartTooltip, 
  ResponsiveContainer, 
  PieChart, 
  Pie, 
  Cell, 
  BarChart, 
  Bar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar
} from 'recharts'

const { Title, Paragraph, Text } = Typography
const { Option } = Select

interface ServicePerformanceDashboardPageProps {}

interface MetricData {
  timestamp: string
  requests: number
  responseTime: number
  errorRate: number
  throughput: number
  cpuUsage: number
  memoryUsage: number
  networkIO: number
  diskIO: number
  connections: number
  registrations: number
  discoveries: number
  healthChecks: number
}

interface ServiceMetric {
  serviceName: string
  serviceType: string
  requests: number
  responseTime: number
  errorRate: number
  uptime: number
  connections: number
  region: string
  status: 'healthy' | 'warning' | 'error'
}

interface LoadBalancerMetric {
  strategy: string
  requests: number
  successRate: number
  avgResponseTime: number
  distribution: { [key: string]: number }
}

const ServicePerformanceDashboardPage: React.FC<ServicePerformanceDashboardPageProps> = () => {
  const [timeRange, setTimeRange] = useState('1h')
  const [refreshInterval, setRefreshInterval] = useState(30)
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [loading, setLoading] = useState(false)

  const [overallMetrics] = useState({
    totalRequests: 2456789,
    avgResponseTime: 47,
    errorRate: 0.8,
    uptime: 99.7,
    activeServices: 125,
    healthyServices: 118,
    totalThroughput: 1247.5,
    peakThroughput: 2156.8
  })

  const [metricsData, setMetricsData] = useState<MetricData[]>([
    { 
      timestamp: '14:00', 
      requests: 2340, responseTime: 45, errorRate: 0.8, throughput: 234.5, 
      cpuUsage: 68, memoryUsage: 72, networkIO: 125, diskIO: 45, connections: 456, 
      registrations: 65, discoveries: 145, healthChecks: 289 
    },
    { 
      timestamp: '14:05', 
      requests: 2567, responseTime: 42, errorRate: 0.6, throughput: 256.7, 
      cpuUsage: 71, memoryUsage: 75, networkIO: 142, diskIO: 52, connections: 478, 
      registrations: 72, discoveries: 156, healthChecks: 298 
    },
    { 
      timestamp: '14:10', 
      requests: 2234, responseTime: 48, errorRate: 1.2, throughput: 223.4, 
      cpuUsage: 65, memoryUsage: 69, networkIO: 118, diskIO: 39, connections: 432, 
      registrations: 68, discoveries: 142, healthChecks: 305 
    },
    { 
      timestamp: '14:15', 
      requests: 2456, responseTime: 44, errorRate: 0.9, throughput: 245.6, 
      cpuUsage: 69, memoryUsage: 73, networkIO: 132, diskIO: 47, connections: 467, 
      registrations: 75, discoveries: 168, healthChecks: 312 
    },
    { 
      timestamp: '14:20', 
      requests: 2189, responseTime: 51, errorRate: 1.5, throughput: 218.9, 
      cpuUsage: 62, memoryUsage: 67, networkIO: 108, diskIO: 41, connections: 421, 
      registrations: 71, discoveries: 159, healthChecks: 295 
    },
    { 
      timestamp: '14:25', 
      requests: 2398, responseTime: 47, errorRate: 0.7, throughput: 239.8, 
      cpuUsage: 67, memoryUsage: 71, networkIO: 126, diskIO: 44, connections: 453, 
      registrations: 78, discoveries: 175, healthChecks: 321 
    }
  ])

  const [serviceMetrics] = useState<ServiceMetric[]>([
    {
      serviceName: 'ml-processor-alpha',
      serviceType: 'ML_PROCESSOR',
      requests: 15420,
      responseTime: 45,
      errorRate: 0.2,
      uptime: 99.8,
      connections: 234,
      region: 'us-east-1',
      status: 'healthy'
    },
    {
      serviceName: 'data-analyzer-beta',
      serviceType: 'DATA_ANALYZER',
      requests: 8960,
      responseTime: 125,
      errorRate: 2.5,
      uptime: 97.5,
      connections: 198,
      region: 'us-west-2',
      status: 'healthy'
    },
    {
      serviceName: 'recommendation-engine',
      serviceType: 'RECOMMENDER',
      requests: 23450,
      responseTime: 189,
      errorRate: 5.8,
      uptime: 85.3,
      connections: 305,
      region: 'eu-central-1',
      status: 'error'
    },
    {
      serviceName: 'chat-assistant-gamma',
      serviceType: 'CONVERSATIONAL',
      requests: 12340,
      responseTime: 89,
      errorRate: 3.2,
      uptime: 94.2,
      connections: 167,
      region: 'ap-southeast-1',
      status: 'warning'
    }
  ])

  const [loadBalancerMetrics] = useState<LoadBalancerMetric[]>([
    {
      strategy: 'capability_based',
      requests: 15420,
      successRate: 99.2,
      avgResponseTime: 45,
      distribution: { 'ml-processor': 35, 'data-analyzer': 25, 'recommender': 40 }
    },
    {
      strategy: 'round_robin',
      requests: 8960,
      successRate: 97.5,
      avgResponseTime: 125,
      distribution: { 'service-a': 33, 'service-b': 33, 'service-c': 34 }
    },
    {
      strategy: 'least_connections',
      requests: 23450,
      successRate: 95.3,
      avgResponseTime: 89,
      distribution: { 'low-load': 45, 'medium-load': 35, 'high-load': 20 }
    }
  ])

  const [radarData] = useState([
    { metric: '响应时间', current: 80, target: 90, fullMark: 100 },
    { metric: '吞吐量', current: 75, target: 85, fullMark: 100 },
    { metric: '可用性', current: 95, target: 99, fullMark: 100 },
    { metric: 'CPU利用率', current: 65, target: 70, fullMark: 100 },
    { metric: '内存使用', current: 70, target: 75, fullMark: 100 },
    { metric: '错误率', current: 98, target: 95, fullMark: 100 }
  ])

  const pieData = [
    { name: '健康服务', value: overallMetrics.healthyServices, color: '#52c41a' },
    { name: '警告服务', value: 4, color: '#faad14' },
    { name: '异常服务', value: 3, color: '#ff4d4f' }
  ]

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return '#52c41a'
      case 'warning': return '#faad14'
      case 'error': return '#ff4d4f'
      default: return '#d9d9d9'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy': return <CheckCircleOutlined />
      case 'warning': return <ExclamationCircleOutlined />
      case 'error': return <CloseCircleOutlined />
      default: return <MonitorOutlined />
    }
  }

  const getTrendIcon = (value: number, threshold: number = 0) => {
    if (value > threshold) {
      return <RiseOutlined style={{ color: '#52c41a' }} />
    } else if (value < -threshold) {
      return <FallOutlined style={{ color: '#ff4d4f' }} />
    }
    return null
  }

  const serviceColumns = [
    {
      title: '服务信息',
      key: 'service',
      render: (_, service: ServiceMetric) => (
        <div>
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '4px' }}>
            {getStatusIcon(service.status)}
            <Text strong style={{ marginLeft: 4 }}>{service.serviceName}</Text>
          </div>
          <Tag size="small" color="blue">{service.serviceType}</Tag>
          <Tag size="small" color="orange">{service.region}</Tag>
        </div>
      )
    },
    {
      title: '请求数',
      dataIndex: 'requests',
      key: 'requests',
      render: (value: number) => (
        <Statistic value={value} valueStyle={{ fontSize: '14px' }} />
      )
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
            color: value < 50 ? '#52c41a' : value < 100 ? '#faad14' : '#ff4d4f'
          }} 
        />
      )
    },
    {
      title: '错误率',
      dataIndex: 'errorRate',
      key: 'errorRate',
      render: (value: number) => (
        <div>
          <Tag color={value < 1 ? 'success' : value < 5 ? 'warning' : 'error'}>
            {value}%
          </Tag>
        </div>
      )
    },
    {
      title: '可用性',
      dataIndex: 'uptime',
      key: 'uptime',
      render: (value: number) => (
        <div>
          <Progress 
            percent={value} 
            size="small" 
            status={value > 99 ? 'success' : value > 95 ? 'active' : 'exception'} 
          />
          <Text style={{ fontSize: '12px' }}>{value}%</Text>
        </div>
      )
    },
    {
      title: '连接数',
      dataIndex: 'connections',
      key: 'connections',
      render: (value: number) => (
        <Statistic value={value} valueStyle={{ fontSize: '14px' }} />
      )
    }
  ]

  const handleRefresh = async () => {
    setLoading(true)
    await new Promise(resolve => setTimeout(resolve, 1000))
    
    // 模拟数据更新
    setMetricsData(prev => prev.map(item => ({
      ...item,
      requests: item.requests + Math.floor(Math.random() * 200) - 100,
      responseTime: Math.max(20, item.responseTime + Math.floor(Math.random() * 20) - 10),
      errorRate: Math.max(0, item.errorRate + (Math.random() - 0.5) * 0.5)
    })))
    
    setLoading(false)
  }

  const handleExportData = () => {
    // 模拟导出功能
    const data = {
      timestamp: new Date().toISOString(),
      metrics: metricsData,
      services: serviceMetrics,
      loadBalancer: loadBalancerMetrics
    }
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `performance-metrics-${Date.now()}.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  useEffect(() => {
    let interval: NodeJS.Timeout | null = null
    
    if (autoRefresh && refreshInterval > 0) {
      interval = setInterval(() => {
        handleRefresh()
      }, refreshInterval * 1000)
    }
    
    return () => {
      if (interval) clearInterval(interval)
    }
  }, [autoRefresh, refreshInterval])

  return (
    <div style={{ padding: '24px', background: '#f5f5f5', minHeight: '100vh' }}>
      <div style={{ maxWidth: '1400px', margin: '0 auto' }}>
        {/* 页面标题 */}
        <div style={{ marginBottom: '24px' }}>
          <Title level={2}>
            <BarChartOutlined /> 性能监控仪表板
          </Title>
          <Paragraph>
            实时监控服务发现系统的整体性能指标、服务状态和负载均衡效果。
          </Paragraph>
        </div>

        {/* 控制栏 */}
        <Card style={{ marginBottom: '24px' }}>
          <Row justify="space-between" align="middle">
            <Col>
              <Space>
                <Text>时间范围:</Text>
                <Select value={timeRange} onChange={setTimeRange} style={{ width: 120 }}>
                  <Option value="15m">15分钟</Option>
                  <Option value="1h">1小时</Option>
                  <Option value="6h">6小时</Option>
                  <Option value="24h">24小时</Option>
                  <Option value="7d">7天</Option>
                </Select>
                <Text>刷新间隔:</Text>
                <Select value={refreshInterval} onChange={setRefreshInterval} style={{ width: 100 }}>
                  <Option value={10}>10秒</Option>
                  <Option value={30}>30秒</Option>
                  <Option value={60}>1分钟</Option>
                  <Option value={300}>5分钟</Option>
                </Select>
                <Button
                  type={autoRefresh ? 'primary' : 'default'}
                  icon={<SyncOutlined spin={autoRefresh} />}
                  onClick={() => setAutoRefresh(!autoRefresh)}
                >
                  {autoRefresh ? '自动刷新' : '手动刷新'}
                </Button>
              </Space>
            </Col>
            <Col>
              <Space>
                <Button icon={<ReloadOutlined />} loading={loading} onClick={handleRefresh}>
                  立即刷新
                </Button>
                <Button icon={<DownloadOutlined />} onClick={handleExportData}>
                  导出数据
                </Button>
                <Button icon={<SettingOutlined />}>
                  设置告警
                </Button>
              </Space>
            </Col>
          </Row>
        </Card>

        {/* 核心指标概览 */}
        <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
          <Col xs={24} sm={6}>
            <Card>
              <Statistic
                title="总请求数"
                value={overallMetrics.totalRequests}
                prefix={<ApiOutlined />}
                valueStyle={{ color: '#1890ff' }}
                suffix={getTrendIcon(2.5, 1)}
              />
              <Text type="secondary" style={{ fontSize: '12px' }}>
                比昨日 +2.5%
              </Text>
            </Card>
          </Col>
          <Col xs={24} sm={6}>
            <Card>
              <Statistic
                title="平均响应时间"
                value={overallMetrics.avgResponseTime}
                suffix="ms"
                prefix={<ThunderboltOutlined />}
                valueStyle={{ color: overallMetrics.avgResponseTime < 50 ? '#52c41a' : '#faad14' }}
                suffix={getTrendIcon(-3.2, 1)}
              />
              <Text type="secondary" style={{ fontSize: '12px' }}>
                比昨日 -3.2%
              </Text>
            </Card>
          </Col>
          <Col xs={24} sm={6}>
            <Card>
              <Statistic
                title="错误率"
                value={overallMetrics.errorRate}
                suffix="%"
                precision={1}
                prefix={<ExclamationCircleOutlined />}
                valueStyle={{ color: overallMetrics.errorRate < 1 ? '#52c41a' : '#ff4d4f' }}
              />
              <Text type="secondary" style={{ fontSize: '12px' }}>
                目标: &lt; 1%
              </Text>
            </Card>
          </Col>
          <Col xs={24} sm={6}>
            <Card>
              <Statistic
                title="系统可用性"
                value={overallMetrics.uptime}
                suffix="%"
                precision={1}
                prefix={<CheckCircleOutlined />}
                valueStyle={{ color: overallMetrics.uptime > 99 ? '#52c41a' : '#faad14' }}
              />
              <Text type="secondary" style={{ fontSize: '12px' }}>
                SLA: &gt; 99.5%
              </Text>
            </Card>
          </Col>
        </Row>

        {/* 性能告警 */}
        {overallMetrics.errorRate > 1 && (
          <Alert
            message="错误率告警"
            description={`当前错误率 ${overallMetrics.errorRate}% 超过阈值，建议检查服务状态。`}
            type="warning"
            icon={<ExclamationCircleOutlined />}
            showIcon
            closable
            style={{ marginBottom: '24px' }}
          />
        )}

        <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
          {/* 请求量和响应时间趋势 */}
          <Col xs={24} lg={12}>
            <Card title="请求量和响应时间趋势" extra={<MonitorOutlined />}>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={metricsData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="timestamp" />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" />
                  <ChartTooltip />
                  <Line yAxisId="left" type="monotone" dataKey="requests" stroke="#1890ff" strokeWidth={2} name="请求数" />
                  <Line yAxisId="right" type="monotone" dataKey="responseTime" stroke="#52c41a" strokeWidth={2} name="响应时间(ms)" />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          </Col>

          {/* 系统资源使用趋势 */}
          <Col xs={24} lg={12}>
            <Card title="系统资源使用趋势" extra={<DatabaseOutlined />}>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={metricsData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="timestamp" />
                  <YAxis />
                  <ChartTooltip />
                  <Area type="monotone" dataKey="cpuUsage" stackId="1" stroke="#1890ff" fill="#1890ff" fillOpacity={0.6} name="CPU使用率%" />
                  <Area type="monotone" dataKey="memoryUsage" stackId="2" stroke="#52c41a" fill="#52c41a" fillOpacity={0.6} name="内存使用率%" />
                </AreaChart>
              </ResponsiveContainer>
            </Card>
          </Col>
        </Row>

        <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
          {/* 服务发现操作趋势 */}
          <Col xs={24} lg={12}>
            <Card title="服务发现操作趋势" extra={<GlobalOutlined />}>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={metricsData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="timestamp" />
                  <YAxis />
                  <ChartTooltip />
                  <Bar dataKey="registrations" fill="#1890ff" name="注册" />
                  <Bar dataKey="discoveries" fill="#52c41a" name="发现" />
                  <Bar dataKey="healthChecks" fill="#faad14" name="健康检查" />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </Col>

          {/* 服务状态分布 */}
          <Col xs={24} lg={12}>
            <Card title="服务状态分布" extra={<CheckCircleOutlined />}>
              <Row>
                <Col span={12}>
                  <ResponsiveContainer width="100%" height={200}>
                    <PieChart>
                      <Pie
                        data={pieData}
                        cx="50%"
                        cy="50%"
                        innerRadius={30}
                        outerRadius={80}
                        paddingAngle={5}
                        dataKey="value"
                      >
                        {pieData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <ChartTooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </Col>
                <Col span={12}>
                  <div style={{ padding: '20px 0' }}>
                    {pieData.map((item, index) => (
                      <div key={index} style={{ marginBottom: '12px', display: 'flex', alignItems: 'center' }}>
                        <div style={{ 
                          width: '12px', 
                          height: '12px', 
                          backgroundColor: item.color, 
                          borderRadius: '2px',
                          marginRight: '8px'
                        }} />
                        <Text>{item.name}: {item.value}</Text>
                      </div>
                    ))}
                  </div>
                </Col>
              </Row>
            </Card>
          </Col>
        </Row>

        <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
          {/* 性能雷达图 */}
          <Col xs={24} lg={8}>
            <Card title="性能指标雷达图" extra={<RadarChartOutlined />}>
              <ResponsiveContainer width="100%" height={300}>
                <RadarChart data={radarData}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="metric" />
                  <PolarRadiusAxis angle={90} domain={[0, 100]} />
                  <Radar name="当前值" dataKey="current" stroke="#1890ff" fill="#1890ff" fillOpacity={0.3} />
                  <Radar name="目标值" dataKey="target" stroke="#52c41a" fill="#52c41a" fillOpacity={0.1} />
                  <ChartTooltip />
                </RadarChart>
              </ResponsiveContainer>
            </Card>
          </Col>

          {/* 网络IO和磁盘IO */}
          <Col xs={24} lg={16}>
            <Card title="网络IO和磁盘IO趋势" extra={<NetworkOutlined />}>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={metricsData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="timestamp" />
                  <YAxis />
                  <ChartTooltip />
                  <Line type="monotone" dataKey="networkIO" stroke="#1890ff" strokeWidth={2} name="网络IO (MB/s)" />
                  <Line type="monotone" dataKey="diskIO" stroke="#ff7300" strokeWidth={2} name="磁盘IO (MB/s)" />
                  <Line type="monotone" dataKey="connections" stroke="#52c41a" strokeWidth={2} name="连接数" />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          </Col>
        </Row>

        {/* 服务性能详细表格 */}
        <Card title="服务性能详细统计" extra={<ApiOutlined />}>
          <Table
            columns={serviceColumns}
            dataSource={serviceMetrics}
            rowKey="serviceName"
            pagination={false}
            size="small"
          />
        </Card>
      </div>
    </div>
  )
}

export default ServicePerformanceDashboardPage