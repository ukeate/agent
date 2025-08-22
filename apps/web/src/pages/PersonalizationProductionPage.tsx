import React, { useState, useEffect } from 'react'
import { Card, Row, Col, Progress, Button, Space, Typography, Table, Tag, Statistic, Alert, Descriptions, Badge, Tooltip, Switch } from 'antd'
import { 
  DashboardOutlined,
  ThunderboltOutlined,
  SafetyOutlined,
  MonitorOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  WarningOutlined,
  ClockCircleOutlined,
  CloudServerOutlined,
  DatabaseOutlined,
  CloudOutlined,
  ApiOutlined,
  RocketOutlined,
  HeartOutlined,
  EyeOutlined,
  ReloadOutlined,
  SettingOutlined,
  AlertOutlined,
  LineChartOutlined
} from '@ant-design/icons'
import { Line, Column, Gauge, Area } from '@ant-design/plots'

const { Title, Text, Paragraph } = Typography

interface SystemStatus {
  overall_status: 'healthy' | 'degraded' | 'unhealthy' | 'critical'
  uptime_seconds: number
  timestamp: string
}

interface HealthCheck {
  service: string
  status: 'healthy' | 'degraded' | 'unhealthy'
  latency_ms: number
  last_check: string
  details?: any
}

interface SystemMetrics {
  cpu_percent: number
  memory_percent: number
  disk_percent: number
  process_memory_mb: number
}

interface KeyMetrics {
  recommendation_latency_p99: number
  feature_computation_latency_avg: number
  cache_hit_rate: number
  error_rate: number
  requests_per_second: number
}

const PersonalizationProductionPage: React.FC = () => {
  const [systemStatus, setSystemStatus] = useState<SystemStatus>({
    overall_status: 'healthy',
    uptime_seconds: 86400,
    timestamp: new Date().toISOString()
  })

  const [healthChecks, setHealthChecks] = useState<HealthCheck[]>([])
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics>({
    cpu_percent: 45.2,
    memory_percent: 62.8,
    disk_percent: 35.4,
    process_memory_mb: 1024.5
  })

  const [keyMetrics, setKeyMetrics] = useState<KeyMetrics>({
    recommendation_latency_p99: 82.5,
    feature_computation_latency_avg: 7.8,
    cache_hit_rate: 0.87,
    error_rate: 0.002,
    requests_per_second: 1250.3
  })

  const [metricHistory, setMetricHistory] = useState<any[]>([])
  const [activeAlerts, setActiveAlerts] = useState(2)
  const [autoRefresh, setAutoRefresh] = useState(true)

  // 模拟数据
  useEffect(() => {
    const generateHealthChecks = () => {
      setHealthChecks([
        {
          service: 'redis',
          status: 'healthy',
          latency_ms: 2.1,
          last_check: new Date().toISOString()
        },
        {
          service: 'personalization_engine',
          status: 'healthy',
          latency_ms: 45.6,
          last_check: new Date().toISOString()
        },
        {
          service: 'feature_engine',
          status: 'healthy',
          latency_ms: 8.9,
          last_check: new Date().toISOString()
        },
        {
          service: 'model_service',
          status: 'degraded',
          latency_ms: 95.2,
          last_check: new Date().toISOString()
        },
        {
          service: 'cache',
          status: 'healthy',
          latency_ms: 12.3,
          last_check: new Date().toISOString()
        }
      ])
    }

    const generateMetricHistory = () => {
      const now = Date.now()
      const history = []
      for (let i = 59; i >= 0; i--) {
        const timestamp = new Date(now - i * 60000).toLocaleTimeString()
        history.push({
          timestamp,
          latency: 80 + Math.random() * 40,
          qps: 1200 + Math.random() * 100,
          error_rate: 0.001 + Math.random() * 0.004,
          cache_hit_rate: 0.85 + Math.random() * 0.1
        })
      }
      setMetricHistory(history)
    }

    generateHealthChecks()
    generateMetricHistory()

    if (autoRefresh) {
      const interval = setInterval(() => {
        generateHealthChecks()
        generateMetricHistory()
        
        // 模拟指标变化
        setKeyMetrics(prev => ({
          recommendation_latency_p99: prev.recommendation_latency_p99 + (Math.random() - 0.5) * 10,
          feature_computation_latency_avg: Math.max(5, prev.feature_computation_latency_avg + (Math.random() - 0.5) * 2),
          cache_hit_rate: Math.min(1, Math.max(0.7, prev.cache_hit_rate + (Math.random() - 0.5) * 0.05)),
          error_rate: Math.max(0, prev.error_rate + (Math.random() - 0.5) * 0.001),
          requests_per_second: Math.max(1000, prev.requests_per_second + (Math.random() - 0.5) * 50)
        }))

        setSystemMetrics(prev => ({
          cpu_percent: Math.min(100, Math.max(20, prev.cpu_percent + (Math.random() - 0.5) * 5)),
          memory_percent: Math.min(100, Math.max(40, prev.memory_percent + (Math.random() - 0.5) * 3)),
          disk_percent: prev.disk_percent + (Math.random() - 0.5) * 0.1,
          process_memory_mb: Math.max(512, prev.process_memory_mb + (Math.random() - 0.5) * 50)
        }))
      }, 5000)

      return () => clearInterval(interval)
    }
  }, [autoRefresh])

  const getStatusColor = (status: string) => {
    const colors = {
      healthy: '#52c41a',
      degraded: '#faad14', 
      unhealthy: '#ff4d4f',
      critical: '#a8071a'
    }
    return colors[status] || '#d9d9d9'
  }

  const getStatusIcon = (status: string) => {
    const icons = {
      healthy: <CheckCircleOutlined style={{ color: '#52c41a' }} />,
      degraded: <WarningOutlined style={{ color: '#faad14' }} />,
      unhealthy: <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />,
      critical: <AlertOutlined style={{ color: '#a8071a' }} />
    }
    return icons[status] || <ClockCircleOutlined />
  }

  const formatUptime = (seconds: number) => {
    const days = Math.floor(seconds / 86400)
    const hours = Math.floor((seconds % 86400) / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    return `${days}天 ${hours}时 ${minutes}分`
  }

  // 延迟趋势图配置
  const latencyTrendConfig = {
    data: metricHistory,
    xField: 'timestamp',
    yField: 'latency',
    smooth: true,
    color: '#1890ff',
    point: { size: 3 },
    yAxis: {
      title: { text: '延迟 (ms)' }
    },
    height: 200
  }

  // QPS趋势图配置
  const qpsTrendConfig = {
    data: metricHistory,
    xField: 'timestamp',
    yField: 'qps',
    smooth: true,
    color: '#52c41a',
    point: { size: 3 },
    yAxis: {
      title: { text: 'QPS' }
    },
    height: 200
  }

  // 缓存命中率仪表盘
  const cacheHitGaugeConfig = {
    percent: keyMetrics.cache_hit_rate,
    range: {
      color: 'l(0) 0:#ff4d4f 0.7:#faad14 1:#52c41a',
    },
    statistic: {
      content: {
        formatter: ({ percent }: any) => `${(percent * 100).toFixed(1)}%`,
      },
    },
    height: 200
  }

  const healthCheckColumns = [
    {
      title: '服务',
      dataIndex: 'service',
      key: 'service',
      render: (service: string) => (
        <Space>
          {service === 'redis' && <DatabaseOutlined />}
          {service === 'personalization_engine' && <RocketOutlined />}
          {service === 'feature_engine' && <ThunderboltOutlined />}
          {service === 'model_service' && <CloudServerOutlined />}
          {service === 'cache' && <CloudOutlined />}
          <Text strong>{service}</Text>
        </Space>
      )
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Space>
          {getStatusIcon(status)}
          <Tag color={getStatusColor(status)}>
            {status.toUpperCase()}
          </Tag>
        </Space>
      )
    },
    {
      title: '延迟',
      dataIndex: 'latency_ms',
      key: 'latency_ms',
      render: (latency: number) => (
        <Text type={latency > 100 ? 'danger' : latency > 50 ? 'warning' : 'success'}>
          {latency.toFixed(1)}ms
        </Text>
      )
    },
    {
      title: '最后检查',
      dataIndex: 'last_check',
      key: 'last_check',
      render: (time: string) => (
        <Text type="secondary">
          {new Date(time).toLocaleTimeString()}
        </Text>
      )
    },
    {
      title: '操作',
      key: 'actions',
      render: (_: any, record: HealthCheck) => (
        <Space>
          <Button 
            type="link" 
            icon={<EyeOutlined />} 
            onClick={() => {/* 查看详情 */}}
          >
            详情
          </Button>
          <Button 
            type="link" 
            icon={<ReloadOutlined />}
            onClick={() => {/* 重新检查 */}}
          >
            重检
          </Button>
        </Space>
      )
    }
  ]

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>
        <DashboardOutlined /> 生产环境监控
      </Title>
      <Paragraph type="secondary">
        实时监控个性化引擎生产环境状态，包括系统健康、性能指标和告警信息
      </Paragraph>

      {/* 控制面板 */}
      <Card style={{ marginBottom: 24 }}>
        <Row justify="space-between" align="middle">
          <Col>
            <Space size="large">
              <Space>
                {getStatusIcon(systemStatus.overall_status)}
                <Text strong>系统状态: </Text>
                <Tag color={getStatusColor(systemStatus.overall_status)} style={{ fontSize: '14px', padding: '4px 8px' }}>
                  {systemStatus.overall_status.toUpperCase()}
                </Tag>
              </Space>
              <Space>
                <ClockCircleOutlined />
                <Text>运行时间: {formatUptime(systemStatus.uptime_seconds)}</Text>
              </Space>
              <Space>
                <AlertOutlined />
                <Text>活跃告警: </Text>
                <Badge count={activeAlerts} showZero />
              </Space>
            </Space>
          </Col>
          <Col>
            <Space>
              <Switch 
                checked={autoRefresh}
                onChange={setAutoRefresh}
                checkedChildren="自动刷新"
                unCheckedChildren="手动刷新"
              />
              <Button icon={<ReloadOutlined />} onClick={() => window.location.reload()}>
                刷新
              </Button>
              <Button icon={<SettingOutlined />}>
                配置
              </Button>
            </Space>
          </Col>
        </Row>
      </Card>

      {/* 关键指标 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="P99延迟"
              value={keyMetrics.recommendation_latency_p99}
              precision={1}
              suffix="ms"
              prefix={<ThunderboltOutlined />}
              valueStyle={{ 
                color: keyMetrics.recommendation_latency_p99 > 100 ? '#ff4d4f' : '#52c41a' 
              }}
            />
            <Progress 
              percent={Math.min(100, keyMetrics.recommendation_latency_p99)} 
              size="small"
              strokeColor={keyMetrics.recommendation_latency_p99 > 100 ? '#ff4d4f' : '#52c41a'}
              showInfo={false}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="特征计算延迟"
              value={keyMetrics.feature_computation_latency_avg}
              precision={1}
              suffix="ms"
              prefix={<ApiOutlined />}
              valueStyle={{ 
                color: keyMetrics.feature_computation_latency_avg > 10 ? '#ff4d4f' : '#52c41a' 
              }}
            />
            <Progress 
              percent={Math.min(100, keyMetrics.feature_computation_latency_avg * 10)} 
              size="small"
              strokeColor={keyMetrics.feature_computation_latency_avg > 10 ? '#ff4d4f' : '#52c41a'}
              showInfo={false}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="QPS"
              value={keyMetrics.requests_per_second}
              precision={1}
              prefix={<LineChartOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
            <Progress 
              percent={Math.min(100, keyMetrics.requests_per_second / 20)} 
              size="small"
              strokeColor="#1890ff"
              showInfo={false}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="错误率"
              value={keyMetrics.error_rate * 100}
              precision={3}
              suffix="%"
              prefix={<ExclamationCircleOutlined />}
              valueStyle={{ 
                color: keyMetrics.error_rate > 0.01 ? '#ff4d4f' : '#52c41a' 
              }}
            />
            <Progress 
              percent={Math.min(100, keyMetrics.error_rate * 10000)} 
              size="small"
              strokeColor={keyMetrics.error_rate > 0.01 ? '#ff4d4f' : '#52c41a'}
              showInfo={false}
            />
          </Card>
        </Col>
      </Row>

      {/* 系统资源 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card title="CPU使用率">
            <Progress
              type="circle"
              percent={systemMetrics.cpu_percent}
              format={percent => `${percent?.toFixed(1)}%`}
              strokeColor={systemMetrics.cpu_percent > 80 ? '#ff4d4f' : '#52c41a'}
              size={120}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card title="内存使用率">
            <Progress
              type="circle"
              percent={systemMetrics.memory_percent}
              format={percent => `${percent?.toFixed(1)}%`}
              strokeColor={systemMetrics.memory_percent > 85 ? '#ff4d4f' : '#52c41a'}
              size={120}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card title="磁盘使用率">
            <Progress
              type="circle"
              percent={systemMetrics.disk_percent}
              format={percent => `${percent?.toFixed(1)}%`}
              strokeColor={systemMetrics.disk_percent > 90 ? '#ff4d4f' : '#52c41a'}
              size={120}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card title="缓存命中率">
            <Gauge {...cacheHitGaugeConfig} />
          </Card>
        </Col>
      </Row>

      {/* 性能趋势 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={12}>
          <Card title="延迟趋势" extra={<Badge status="processing" text="实时" />}>
            <Line {...latencyTrendConfig} />
          </Card>
        </Col>
        <Col span={12}>
          <Card title="QPS趋势" extra={<Badge status="processing" text="实时" />}>
            <Line {...qpsTrendConfig} />
          </Card>
        </Col>
      </Row>

      {/* 健康检查 */}
      <Card 
        title="服务健康检查" 
        extra={
          <Space>
            <Badge status="processing" text="自动检查中" />
            <Button icon={<ReloadOutlined />} size="small">刷新</Button>
          </Space>
        }
      >
        <Table
          columns={healthCheckColumns}
          dataSource={healthChecks}
          rowKey="service"
          size="small"
          pagination={false}
        />
      </Card>

      {/* 系统信息 */}
      <Card title="系统信息" style={{ marginTop: 24 }}>
        <Row gutter={[16, 16]}>
          <Col span={12}>
            <Descriptions column={1} size="small">
              <Descriptions.Item label="进程内存">
                {systemMetrics.process_memory_mb.toFixed(1)} MB
              </Descriptions.Item>
              <Descriptions.Item label="最后更新">
                {new Date(systemStatus.timestamp).toLocaleString()}
              </Descriptions.Item>
              <Descriptions.Item label="版本信息">
                v1.2.3 (Build 20240115)
              </Descriptions.Item>
            </Descriptions>
          </Col>
          <Col span={12}>
            <Descriptions column={1} size="small">
              <Descriptions.Item label="部署环境">
                Production
              </Descriptions.Item>
              <Descriptions.Item label="负载均衡">
                Nginx + Upstream
              </Descriptions.Item>
              <Descriptions.Item label="监控状态">
                <Badge status="success" text="正常" />
              </Descriptions.Item>
            </Descriptions>
          </Col>
        </Row>
      </Card>
    </div>
  )
}

export default PersonalizationProductionPage