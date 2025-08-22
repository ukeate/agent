import React, { useState, useEffect } from 'react'
import {
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
  Tabs
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
  const [healthChecks, setHealthChecks] = useState<HealthCheck[]>([
    {
      service: 'PostgreSQL数据库',
      status: 'healthy',
      responseTime: 15,
      lastCheck: '30秒前',
      uptime: 99.8
    },
    {
      service: 'Redis缓存',
      status: 'healthy',
      responseTime: 5,
      lastCheck: '30秒前',
      uptime: 99.9
    },
    {
      service: 'Qdrant向量数据库',
      status: 'healthy',
      responseTime: 25,
      lastCheck: '30秒前',
      uptime: 99.5
    },
    {
      service: 'OpenAI API',
      status: 'degraded',
      responseTime: 850,
      lastCheck: '30秒前',
      uptime: 98.2,
      details: '响应时间较慢'
    },
    {
      service: 'MCP服务',
      status: 'healthy',
      responseTime: 12,
      lastCheck: '30秒前',
      uptime: 99.7
    },
    {
      service: '文件存储',
      status: 'unhealthy',
      responseTime: 0,
      lastCheck: '2分钟前',
      uptime: 95.1,
      details: '连接超时'
    }
  ])

  const [systemMetrics] = useState<SystemMetric[]>([
    { name: 'CPU使用率', value: 45, unit: '%', status: 'normal', threshold: 80 },
    { name: '内存使用率', value: 67, unit: '%', status: 'normal', threshold: 85 },
    { name: '磁盘使用率', value: 52, unit: '%', status: 'normal', threshold: 90 },
    { name: '网络延迟', value: 25, unit: 'ms', status: 'normal', threshold: 100 },
    { name: '数据库连接数', value: 85, unit: '个', status: 'warning', threshold: 100 },
    { name: '缓存命中率', value: 92, unit: '%', status: 'normal', threshold: 80 }
  ])

  const [alerts] = useState<Alert[]>([
    {
      id: '1',
      level: 'error',
      message: '文件存储服务连接失败',
      service: '文件存储',
      timestamp: '2分钟前',
      resolved: false
    },
    {
      id: '2',
      level: 'warning',
      message: 'OpenAI API响应时间超过阈值',
      service: 'OpenAI API',
      timestamp: '5分钟前',
      resolved: false
    },
    {
      id: '3',
      level: 'warning',
      message: '数据库连接数接近上限',
      service: 'PostgreSQL数据库',
      timestamp: '10分钟前',
      resolved: false
    },
    {
      id: '4',
      level: 'info',
      message: 'Redis缓存自动清理完成',
      service: 'Redis缓存',
      timestamp: '1小时前',
      resolved: true
    }
  ])

  const [autoRefresh, setAutoRefresh] = useState(true)

  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(() => {
        setHealthChecks(prev => prev.map(check => ({
          ...check,
          responseTime: check.status === 'healthy' 
            ? Math.max(1, check.responseTime + (Math.random() - 0.5) * 10)
            : check.responseTime,
          lastCheck: '刚刚'
        })))
      }, 30000)
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
              onClick={() => console.log('执行全面健康检查')}
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