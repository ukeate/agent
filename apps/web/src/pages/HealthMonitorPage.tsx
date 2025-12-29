import React, { useState, useEffect } from 'react'
import {
import { logger } from '../utils/logger'
  Card,
  Row,
  Col,
  Statistic,
  Progress,
  Button,
  Space,
  Table,
  Tag,
  Alert,
  Typography,
  Timeline,
  List,
  Badge,
  Tabs,
  message,
  Spin
} from 'antd'
import {
  HeartOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  WarningOutlined,
  ReloadOutlined,
  ApiOutlined,
  ExclamationCircleOutlined
} from '@ant-design/icons'
import { healthService } from '../services/healthService'
import type { HealthCheckResult, HealthAlert } from '../services/healthService'

const { Title, Text } = Typography
const { TabPane } = Tabs

interface HealthCheck {
  service: string
  status: 'healthy' | 'unhealthy' | 'degraded'
  responseTime: number
  lastCheck: string
  uptime: number
  details?: string
}

interface SystemMetric {
  name: string
  value: number
  unit: string
  status: 'normal' | 'warning' | 'critical'
  threshold: number
}

interface Alert {
  id: string
  level: 'info' | 'warning' | 'error' | 'critical'
  message: string
  service: string
  timestamp: string
  resolved: boolean
}

const HealthMonitorPage: React.FC = () => {
  const [healthChecks, setHealthChecks] = useState<HealthCheck[]>([])
  const [systemMetrics, setSystemMetrics] = useState<SystemMetric[]>([])
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [loading, setLoading] = useState(true)

  // 加载健康状态数据
  const loadHealthData = async () => {
    try {
      setLoading(true)
      
      // 并行调用多个API
      const [healthStatus, systemMetrics, alerts] = await Promise.all([
        healthService.getDetailedHealth(),
        healthService.getSystemMetrics(),
        healthService.getHealthAlerts()
      ])

      // 转换健康状态数据
      if (healthStatus.components) {
        const checks: HealthCheck[] = Object.entries(healthStatus.components).map(([service, data]: [string, any]) => ({
          service: service,
          status: data.status === 'healthy' ? 'healthy' : data.status === 'degraded' ? 'degraded' : 'unhealthy',
          responseTime: data.response_time_ms || 0,
          lastCheck: new Date(healthStatus.timestamp || Date.now()).toLocaleString(),
          uptime: data.uptime_seconds || 0,
          details: data.error || data.note
        }))
        setHealthChecks(checks)
      }

      // 转换系统指标数据  
      if (systemMetrics) {
        const metrics: SystemMetric[] = []
        const processMetrics = (systemMetrics as any).process || {}
        const systemInfo = (systemMetrics as any).system || {}
        const cpuPercent = processMetrics.cpu_percent
        const memoryRss = processMetrics.memory_rss_mb
        const memoryTotal = systemInfo.memory_total_mb

        if (typeof cpuPercent === 'number') {
          metrics.push({
            name: 'CPU使用率',
            value: cpuPercent,
            unit: '%',
            status: cpuPercent >= 95 ? 'critical' : cpuPercent >= 80 ? 'warning' : 'normal',
            threshold: 100
          })
        }

        if (typeof memoryRss === 'number' && typeof memoryTotal === 'number' && memoryTotal > 0) {
          const percent = (memoryRss / memoryTotal) * 100
          metrics.push({
            name: '内存使用率',
            value: Number(percent.toFixed(1)),
            unit: '%',
            status: percent >= 95 ? 'critical' : percent >= 80 ? 'warning' : 'normal',
            threshold: 100
          })
        }

        setSystemMetrics(metrics)
      }

      // 转换告警数据
      if (alerts.alerts) {
        const alertList: Alert[] = alerts.alerts.map((alert: any) => ({
          id: alert.id,
          level: alert.level,
          message: alert.message,
          service: alert.service,
          timestamp: new Date(alert.timestamp).toLocaleString(),
          resolved: alert.resolved || false
        }))
        setAlerts(alertList)
      }

    } catch (error) {
      logger.error('加载健康数据失败:', error)
      message.error('获取健康数据失败')
      setHealthChecks([])
      setSystemMetrics([])
      setAlerts([])
    } finally {
      setLoading(false)
    }
  }

  // 手动刷新
  const handleRefresh = async () => {
    await loadHealthData()
    message.success('数据已刷新')
  }

  // 初始加载和自动刷新
  useEffect(() => {
    loadHealthData()
  }, [])

  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(loadHealthData, 30000) // 30秒刷新一次
      return () => clearInterval(interval)
    }
  }, [autoRefresh])

  const getStatusColor = (status: string) => {
    const colors = {
      healthy: 'success',
      degraded: 'warning', 
      unhealthy: 'error'
    }
    return colors[status as keyof typeof colors] || 'default'
  }

  const getStatusIcon = (status: string) => {
    const icons = {
      healthy: <CheckCircleOutlined />,
      degraded: <WarningOutlined />,
      unhealthy: <CloseCircleOutlined />
    }
    return icons[status as keyof typeof icons]
  }

  const getMetricStatusColor = (status: string) => {
    const colors = {
      normal: '#52c41a',
      warning: '#faad14',
      critical: '#ff4d4f'
    }
    return colors[status as keyof typeof colors]
  }

  const getAlertColor = (level: string) => {
    const colors = {
      info: 'blue',
      warning: 'orange',
      error: 'red',
      critical: 'red'
    }
    return colors[level as keyof typeof colors]
  }

  const getAlertStatus = (level: string): "success" | "processing" | "default" | "error" | "warning" => {
    const statusMap = {
      info: 'processing' as const,
      warning: 'warning' as const,
      error: 'error' as const,
      critical: 'error' as const
    }
    return statusMap[level as keyof typeof statusMap] || 'default'
  }

  const healthyServices = healthChecks.filter(s => s.status === 'healthy').length
  const totalServices = healthChecks.length
  const overallHealth = (healthyServices / totalServices) * 100

  const columns = [
    {
      title: '服务',
      dataIndex: 'service',
      key: 'service',
      render: (service: string) => (
        <Space>
          <ApiOutlined />
          <Text strong>{service}</Text>
        </Space>
      )
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={getStatusColor(status)} icon={getStatusIcon(status)}>
          {status === 'healthy' && '健康'}
          {status === 'degraded' && '降级'}
          {status === 'unhealthy' && '不健康'}
        </Tag>
      )
    },
    {
      title: '响应时间',
      dataIndex: 'responseTime',
      key: 'responseTime',
      render: (time: number) => (
        <Text style={{ 
          color: time > 500 ? '#ff4d4f' : time > 200 ? '#faad14' : '#52c41a' 
        }}>
          {time}ms
        </Text>
      )
    },
    {
      title: '正常运行时间',
      dataIndex: 'uptime',
      key: 'uptime',
      render: (uptime: number) => (
        <div>
          <Progress 
            percent={uptime} 
            size="small"
            strokeColor={uptime > 99 ? '#52c41a' : uptime > 95 ? '#faad14' : '#ff4d4f'}
          />
          <Text className="text-xs">{uptime}%</Text>
        </div>
      )
    },
    {
      title: '最后检查',
      dataIndex: 'lastCheck',
      key: 'lastCheck'
    },
    {
      title: '详情',
      dataIndex: 'details',
      key: 'details',
      render: (details?: string) => details || '-'
    }
  ]

  return (
    <div className="p-6">
      <div className="mb-6">
        <div className="flex justify-between items-center mb-4">
          <Title level={2}>系统健康监控</Title>
          <Space>
            <Button 
              icon={<ReloadOutlined />}
              onClick={() => setAutoRefresh(!autoRefresh)}
              type={autoRefresh ? 'primary' : 'default'}
            >
              {autoRefresh ? '停止' : '开始'}自动刷新
            </Button>
            <Button 
              icon={<HeartOutlined />}
              onClick={handleRefresh}
              loading={loading}
            >
              立即检查
            </Button>
          </Space>
        </div>

        {overallHealth < 80 && (
          <Alert
            message="系统健康状态异常"
            description={`当前有 ${totalServices - healthyServices} 个服务存在问题，建议立即检查系统状态`}
            variant="destructive"
            showIcon
            closable
            className="mb-4"
          />
        )}

        <Row gutter={16} className="mb-6">
          <Col span={8}>
            <Card>
              <Statistic
                title="系统整体健康度"
                value={overallHealth}
                precision={1}
                suffix="%"
                valueStyle={{ 
                  color: overallHealth > 90 ? '#3f8600' : overallHealth > 70 ? '#faad14' : '#cf1322' 
                }}
                prefix={<HeartOutlined />}
              />
              <Progress 
                percent={overallHealth} 
                strokeColor={overallHealth > 90 ? '#52c41a' : overallHealth > 70 ? '#faad14' : '#ff4d4f'}
                className="mt-2"
              />
            </Card>
          </Col>
          <Col span={8}>
            <Card>
              <Statistic
                title="健康服务"
                value={healthyServices}
                suffix={`/ ${totalServices}`}
                valueStyle={{ color: '#3f8600' }}
                prefix={<CheckCircleOutlined />}
              />
              <div className="mt-2 text-xs text-gray-500">
                {healthChecks.filter(s => s.status === 'degraded').length} 个降级, {' '}
                {healthChecks.filter(s => s.status === 'unhealthy').length} 个不健康
              </div>
            </Card>
          </Col>
          <Col span={8}>
            <Card>
              <Statistic
                title="未解决告警"
                value={alerts.filter(a => !a.resolved).length}
                valueStyle={{ color: '#cf1322' }}
                prefix={<ExclamationCircleOutlined />}
              />
              <div className="mt-2 text-xs text-gray-500">
                {alerts.filter(a => !a.resolved && a.level === 'critical').length} 个严重, {' '}
                {alerts.filter(a => !a.resolved && a.level === 'error').length} 个错误
              </div>
            </Card>
          </Col>
        </Row>
      </div>

      <Tabs defaultActiveKey="services">
        <TabPane tab="服务状态" key="services">
          <Card title="服务健康检查">
            <Table
              columns={columns}
              dataSource={healthChecks}
              rowKey="service"
              pagination={false}
              size="small"
            />
          </Card>
        </TabPane>

        <TabPane tab="系统指标" key="metrics">
          {loading ? (
            <div style={{ textAlign: 'center', padding: '50px' }}>
              <Spin size="large" />
              <p>加载系统指标...</p>
            </div>
          ) : (
          <Row gutter={16}>
            {systemMetrics.map((metric, index) => (
              <Col span={8} key={index} className="mb-4">
                <Card>
                  <div className="flex justify-between items-center mb-2">
                    <Text strong>{metric.name}</Text>
                    <Badge 
                      status={metric.status === 'normal' ? 'success' : metric.status === 'warning' ? 'warning' : 'error'}
                    />
                  </div>
                  <div className="text-2xl font-bold mb-2" style={{ color: getMetricStatusColor(metric.status) }}>
                    {metric.value}{metric.unit}
                  </div>
                  <Progress 
                    percent={(metric.value / metric.threshold) * 100}
                    strokeColor={getMetricStatusColor(metric.status)}
                    size="small"
                  />
                  <div className="text-xs text-gray-500 mt-1">
                    阈值: {metric.threshold}{metric.unit}
                  </div>
                </Card>
              </Col>
            ))}
          </Row>
          )}
        </TabPane>

        <TabPane tab={`告警 (${alerts.filter(a => !a.resolved).length})`} key="alerts">
          <Card title="系统告警">
            <List
              dataSource={alerts}
              renderItem={(alert) => (
                <List.Item
                  style={{ 
                    backgroundColor: alert.resolved ? '#f6f6f6' : 'white',
                    opacity: alert.resolved ? 0.7 : 1
                  }}
                >
                  <List.Item.Meta
                    avatar={
                      <Badge 
                        status={alert.resolved ? 'default' : getAlertStatus(alert.level)} 
                      />
                    }
                    title={
                      <div className="flex justify-between items-center">
                        <Space>
                          <Tag color={getAlertColor(alert.level)}>
                            {alert.level.toUpperCase()}
                          </Tag>
                          <Text strong={!alert.resolved}>{alert.message}</Text>
                        </Space>
                        <Text type="secondary" className="text-xs">{alert.timestamp}</Text>
                      </div>
                    }
                    description={
                      <div className="flex justify-between items-center">
                        <Text type="secondary">服务: {alert.service}</Text>
                        {alert.resolved && (
                          <Tag color="green">已解决</Tag>
                        )}
                      </div>
                    }
                  />
                </List.Item>
              )}
            />
          </Card>
        </TabPane>

        <TabPane tab="实时日志" key="logs">
          <Card title="健康检查日志">
            <Timeline
              items={[
                {
                  color: 'green',
                  children: (
                    <div>
                      <Text strong>所有服务健康检查完成</Text>
                      <br />
                      <Text type="secondary">检查了6个服务，5个健康，1个降级 - 刚刚</Text>
                    </div>
                  )
                },
                {
                  color: 'red',
                  children: (
                    <div>
                      <Text strong>文件存储服务检查失败</Text>
                      <br />
                      <Text type="secondary">连接超时，正在重试 - 2分钟前</Text>
                    </div>
                  )
                },
                {
                  color: 'orange',
                  children: (
                    <div>
                      <Text strong>OpenAI API响应延迟</Text>
                      <br />
                      <Text type="secondary">响应时间850ms，超过200ms阈值 - 5分钟前</Text>
                    </div>
                  )
                },
                {
                  color: 'blue',
                  children: (
                    <div>
                      <Text strong>系统指标收集完成</Text>
                      <br />
                      <Text type="secondary">CPU 45%, 内存 67%, 磁盘 52% - 10分钟前</Text>
                    </div>
                  )
                }
              ]}
            />
          </Card>
        </TabPane>
      </Tabs>
    </div>
  )
}

export default HealthMonitorPage
