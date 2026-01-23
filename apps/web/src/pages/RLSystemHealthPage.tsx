import React, { useMemo, useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Badge,
  Statistic,
  Alert,
  Timeline,
  Table,
  Tag,
  Space,
  Button,
  Select,
} from 'antd'
import { Line, Gauge } from '@ant-design/plots'
import {
  HeartOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  CloseCircleOutlined,
  ThunderboltOutlined,
  DatabaseOutlined,
  CloudServerOutlined,
  ApiOutlined,
  SafetyCertificateOutlined,
  MonitorOutlined,
} from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'
import { healthService } from '../services/healthService'

const { Option } = Select

interface SystemComponent {
  id: string
  name: string
  type: 'service' | 'database' | 'cache' | 'queue'
  status: 'healthy' | 'warning' | 'error' | 'offline'
  uptime: number | null
  responseTime: number | null
  lastCheck: string
  details: string
  dependencies: string[]
}

interface HealthMetric {
  timestamp: string
  cpu: number
  memory: number
  disk: number
  responseTime: number
}

interface ServiceHealth {
  id: string
  service: string
  status: SystemComponent['status']
  responseTime: number | null
  uptime: number | null
}

interface HealthEvent {
  id: string
  timestamp: string
  type: 'info' | 'warning' | 'error' | 'recovery'
  component: string
  message: string
  duration?: number
}

const RLSystemHealthPage: React.FC = () => {
  const [timeRange, setTimeRange] = useState('1h')
  const [components, setComponents] = useState<SystemComponent[]>([])
  const [healthMetrics, setHealthMetrics] = useState<HealthMetric[]>([])
  const [serviceHealth, setServiceHealth] = useState<ServiceHealth[]>([])
  const [healthEvents, setHealthEvents] = useState<HealthEvent[]>([])
  const [loading, setLoading] = useState(false)

  const normalizeStatus = (status?: string): SystemComponent['status'] => {
    if (status === 'healthy') return 'healthy'
    if (status === 'degraded') return 'warning'
    if (status === 'unhealthy') return 'error'
    return 'offline'
  }

  const formatDuration = (seconds: number | null) => {
    if (!seconds || Number.isNaN(seconds)) return '-'
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    const secs = Math.floor(seconds % 60)
    if (hours > 0) return `${hours}小时 ${minutes}分钟`
    if (minutes > 0) return `${minutes}分钟 ${secs}秒`
    return `${secs}秒`
  }

  const appendMetrics = (metrics: any) => {
    if (!metrics || metrics.error) return
    const systemInfo = metrics?.system || {}
    const performanceInfo = metrics?.performance || {}
    const point: HealthMetric = {
      timestamp: metrics?.timestamp || new Date().toISOString(),
      cpu:
        typeof systemInfo.cpu_percent === 'number' ? systemInfo.cpu_percent : 0,
      memory:
        typeof systemInfo.memory_percent === 'number'
          ? systemInfo.memory_percent
          : 0,
      disk:
        typeof systemInfo.disk_percent === 'number' ? systemInfo.disk_percent : 0,
      responseTime:
        typeof performanceInfo.average_response_time_ms === 'number'
          ? performanceInfo.average_response_time_ms
          : 0,
    }
    setHealthMetrics(prev => {
      const next = [...prev, point]
      return next.length > 200 ? next.slice(next.length - 200) : next
    })
  }

  const loadData = async () => {
    setLoading(true)
    try {
      const [detailed, metrics, alerts] = await Promise.all([
        healthService.getDetailedHealth(),
        healthService.getSystemMetrics(),
        healthService.getHealthAlerts({ limit: 20 }),
      ])

      const comps: SystemComponent[] = Object.entries(
        detailed.components || {}
      ).map(([key, value]: any) => {
        const componentType =
          key.includes('database')
            ? 'database'
            : key.includes('redis')
              ? 'cache'
              : 'service'
        const responseTime =
          typeof value.response_time_ms === 'number' ? value.response_time_ms : null
        const uptime =
          typeof value.uptime_seconds === 'number' ? value.uptime_seconds : null
        const details =
          value.error ||
          value.note ||
          (Array.isArray(value.warnings) ? value.warnings.join('，') : '')
        return {
          id: key,
          name: key,
          type: componentType,
          status: normalizeStatus(value.status),
          uptime,
          responseTime,
          lastCheck: detailed.timestamp || new Date().toISOString(),
          details,
          dependencies: [],
        }
      })
      setComponents(comps)

      appendMetrics(metrics)

      const services: ServiceHealth[] = comps.map(c => ({
        id: c.id,
        service: c.name,
        status: c.status,
        responseTime: c.responseTime,
        uptime: c.uptime,
      }))
      setServiceHealth(services)

      const events: HealthEvent[] = (alerts.alerts || []).map((alert: any) => ({
        id: `${alert.name || 'alert'}-${alert.timestamp || Date.now()}`,
        timestamp: alert.timestamp || new Date().toISOString(),
        type:
          alert.severity === 'critical'
            ? 'error'
            : alert.severity === 'warning'
              ? 'warning'
              : alert.severity === 'error'
                ? 'error'
                : 'info',
        component: alert.name || 'system',
        message: alert.message,
        duration: undefined,
      }))
      setHealthEvents(events)
    } catch (e) {
      setComponents([])
      setServiceHealth([])
      setHealthEvents([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
  }, [])

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const metrics = await healthService.getSystemMetrics()
        appendMetrics(metrics)
      } catch {
        return
      }
    }, 30000)
    return () => clearInterval(interval)
  }, [])

  const filteredMetrics = useMemo(() => {
    const rangeMs =
      timeRange === '24h'
        ? 24 * 60 * 60 * 1000
        : timeRange === '7d'
          ? 7 * 24 * 60 * 60 * 1000
          : 60 * 60 * 1000
    const cutoff = Date.now() - rangeMs
    return healthMetrics.filter(metric => {
      const timestamp = Date.parse(metric.timestamp)
      return Number.isNaN(timestamp) ? false : timestamp >= cutoff
    })
  }, [healthMetrics, timeRange])

  const systemHealth = useMemo(() => {
    if (components.length === 0) return 0
    const score = components.reduce((sum, component) => {
      if (component.status === 'healthy') return sum + 1
      if (component.status === 'warning') return sum + 0.7
      if (component.status === 'error') return sum + 0.3
      return sum
    }, 0)
    return score / components.length
  }, [components])

  // 系统健康趋势图配置
  const healthTrendConfig = {
    data: filteredMetrics
      .map(m => [
        { timestamp: m.timestamp, metric: 'CPU使用率', value: m.cpu },
        { timestamp: m.timestamp, metric: '内存使用率', value: m.memory },
        { timestamp: m.timestamp, metric: '磁盘使用率', value: m.disk },
        { timestamp: m.timestamp, metric: '平均响应时间', value: m.responseTime },
      ])
      .flat(),
    xField: 'timestamp',
    yField: 'value',
    seriesField: 'metric',
    smooth: true,
    color: ['#1890ff', '#52c41a', '#faad14', '#f5222d'],
    legend: { position: 'top' },
  }

  // 系统健康仪表盘配置
  const healthGaugeConfig = {
    percent: systemHealth,
    range: {
      ticks: [0, 1 / 3, 2 / 3, 1],
      color: ['#F4664A', '#FAAD14', '#30BF78'],
    },
    indicator: {
      pointer: {
        style: {
          stroke: '#D0D0D0',
        },
      },
      pin: {
        style: {
          stroke: '#D0D0D0',
        },
      },
    },
    statistic: {
      content: {
        style: {
          fontSize: '36px',
          lineHeight: '36px',
        },
        formatter: () => (systemHealth * 100).toFixed(1) + '%',
      },
    },
  }

  const componentColumns: ColumnsType<SystemComponent> = [
    {
      title: '组件',
      dataIndex: 'name',
      key: 'name',
      render: (text, record) => (
        <div style={{ display: 'flex', alignItems: 'center' }}>
          {record.type === 'service' && (
            <ApiOutlined style={{ marginRight: '8px', color: '#1890ff' }} />
          )}
          {record.type === 'database' && (
            <DatabaseOutlined
              style={{ marginRight: '8px', color: '#52c41a' }}
            />
          )}
          {record.type === 'cache' && (
            <CloudServerOutlined
              style={{ marginRight: '8px', color: '#faad14' }}
            />
          )}
          {record.type === 'queue' && (
            <MonitorOutlined style={{ marginRight: '8px', color: '#722ed1' }} />
          )}
          <div>
            <strong>{text}</strong>
            <div style={{ fontSize: '12px', color: '#666' }}>
              {record.details}
            </div>
          </div>
        </div>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: status => {
        const config = {
          healthy: {
            color: 'success',
            icon: <CheckCircleOutlined />,
            text: '健康',
          },
          warning: {
            color: 'warning',
            icon: <ExclamationCircleOutlined />,
            text: '警告',
          },
          error: {
            color: 'error',
            icon: <CloseCircleOutlined />,
            text: '错误',
          },
          offline: {
            color: 'default',
            icon: <CloseCircleOutlined />,
            text: '离线',
          },
        }
        return (
          <Badge status={config[status].color} text={config[status].text} />
        )
      },
    },
    {
      title: '运行时长',
      dataIndex: 'uptime',
      key: 'uptime',
      render: uptime => (
        <span style={{ fontSize: '12px' }}>{formatDuration(uptime)}</span>
      ),
      sorter: (a, b) => (a.uptime ?? -1) - (b.uptime ?? -1),
    },
    {
      title: '响应时间',
      dataIndex: 'responseTime',
      key: 'responseTime',
      render: time =>
        typeof time === 'number' ? (
          <Tag color={time < 10 ? 'green' : time < 50 ? 'orange' : 'red'}>
            {time.toFixed(1)}ms
          </Tag>
        ) : (
          <Tag color="default">-</Tag>
        ),
      sorter: (a, b) => (a.responseTime ?? -1) - (b.responseTime ?? -1),
    },
    {
      title: '最后检查',
      dataIndex: 'lastCheck',
      key: 'lastCheck',
    },
    {
      title: '依赖',
      dataIndex: 'dependencies',
      key: 'dependencies',
      render: deps =>
        deps.length > 0 ? (
          <div>
            {deps.map((dep: string) => (
              <Tag key={dep} size="small">
                {dep}
              </Tag>
            ))}
          </div>
        ) : (
          <Tag color="default">-</Tag>
        ),
    },
  ]

  const serviceColumns: ColumnsType<ServiceHealth> = [
    {
      title: '服务',
      dataIndex: 'service',
      key: 'service',
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: status => {
        const config = {
          healthy: { color: 'green', text: '健康' },
          warning: { color: 'orange', text: '警告' },
          error: { color: 'red', text: '错误' },
          offline: { color: 'default', text: '离线' },
        }
        return <Tag color={config[status].color}>{config[status].text}</Tag>
      },
    },
    {
      title: '响应时间',
      dataIndex: 'responseTime',
      key: 'responseTime',
      render: value =>
        typeof value === 'number' ? `${value.toFixed(1)}ms` : '-',
      sorter: (a, b) => (a.responseTime ?? -1) - (b.responseTime ?? -1),
    },
    {
      title: '运行时长',
      dataIndex: 'uptime',
      key: 'uptime',
      render: value => formatDuration(value),
      sorter: (a, b) => (a.uptime ?? -1) - (b.uptime ?? -1),
    },
  ]

  const healthyComponents = components.filter(
    c => c.status === 'healthy'
  ).length
  const warningComponents = components.filter(
    c => c.status === 'warning'
  ).length
  const errorComponents = components.filter(c => c.status === 'error').length
  const responseTimes = components
    .map(c => c.responseTime)
    .filter((value): value is number => typeof value === 'number')
  const avgResponseTime = responseTimes.length
    ? responseTimes.reduce((sum, value) => sum + value, 0) / responseTimes.length
    : null

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
        <h1 style={{ margin: 0, display: 'flex', alignItems: 'center' }}>
          <HeartOutlined style={{ marginRight: '8px' }} />
          强化学习系统健康监控
        </h1>
        <Space>
          <Select
            value={timeRange}
            onChange={setTimeRange}
            style={{ width: 120 }}
          >
            <Option value="1h">最近1小时</Option>
            <Option value="24h">最近24小时</Option>
            <Option value="7d">最近7天</Option>
          </Select>
          <Button
            type="primary"
            icon={<ThunderboltOutlined />}
            loading={loading}
            onClick={loadData}
          >
            刷新状态
          </Button>
        </Space>
      </div>

      {/* 系统状态警告 */}
      {(errorComponents > 0 || warningComponents > 0) && (
        <Alert
          message="系统健康警告"
          description={`发现 ${errorComponents} 个错误组件和 ${warningComponents} 个警告组件，请及时处理`}
          type={errorComponents > 0 ? 'error' : 'warning'}
          showIcon
          closable
          style={{ marginBottom: '24px' }}
        />
      )}

      {/* 系统健康概览 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} sm={6} md={4}>
          <Card>
            <Statistic
              title="系统健康度"
              value={(systemHealth * 100).toFixed(1)}
              prefix={<HeartOutlined />}
              suffix="%"
              valueStyle={{ color: systemHealth > 0.9 ? '#3f8600' : '#cf1322' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6} md={4}>
          <Card>
            <Statistic
              title="健康组件"
              value={healthyComponents}
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6} md={4}>
          <Card>
            <Statistic
              title="警告组件"
              value={warningComponents}
              prefix={<ExclamationCircleOutlined />}
              valueStyle={{
                color: warningComponents > 0 ? '#faad14' : '#3f8600',
              }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6} md={4}>
          <Card>
            <Statistic
              title="错误组件"
              value={errorComponents}
              prefix={<CloseCircleOutlined />}
              valueStyle={{
                color: errorComponents > 0 ? '#cf1322' : '#3f8600',
              }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6} md={4}>
          <Card>
            <Statistic
              title="平均响应时间"
              value={avgResponseTime ?? 0}
              suffix={avgResponseTime !== null ? 'ms' : ''}
              formatter={value =>
                avgResponseTime !== null ? Number(value).toFixed(1) : '-'
              }
              valueStyle={{
                color:
                  avgResponseTime !== null
                    ? avgResponseTime < 20
                      ? '#3f8600'
                      : '#cf1322'
                    : '#999',
              }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6} md={4}>
          <Card>
            <Statistic
              title="在线服务"
              value={components.filter(c => c.status !== 'offline').length}
              suffix={`/${components.length}`}
              prefix={<SafetyCertificateOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
      </Row>

      {/* 系统健康仪表盘和趋势 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} lg={8}>
          <Card title="系统整体健康度" loading={loading}>
            <Gauge {...healthGaugeConfig} height={250} />
          </Card>
        </Col>
        <Col xs={24} lg={16}>
          <Card title="系统资源使用趋势" loading={loading}>
            <Line {...healthTrendConfig} height={250} />
          </Card>
        </Col>
      </Row>

      {/* 组件状态详情 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} lg={16}>
          <Card title="组件健康状态" loading={loading}>
            <Table
              dataSource={components}
              columns={componentColumns}
              rowKey="id"
              pagination={false}
              size="middle"
            />
          </Card>
        </Col>
        <Col xs={24} lg={8}>
          <Card title="健康事件时间线" loading={loading}>
            <Timeline mode="left" style={{ marginTop: '16px' }}>
              {healthEvents.map(event => (
                <Timeline.Item
                  key={event.id}
                  color={
                    event.type === 'error'
                      ? 'red'
                      : event.type === 'warning'
                        ? 'orange'
                        : event.type === 'recovery'
                          ? 'green'
                          : 'blue'
                  }
                  dot={
                    event.type === 'error' ? (
                      <CloseCircleOutlined />
                    ) : event.type === 'warning' ? (
                      <ExclamationCircleOutlined />
                    ) : event.type === 'recovery' ? (
                      <CheckCircleOutlined />
                    ) : (
                      <SafetyCertificateOutlined />
                    )
                  }
                >
                  <div>
                    <div style={{ fontSize: '12px', color: '#999' }}>
                      {event.timestamp}
                    </div>
                    <div style={{ fontWeight: 'bold' }}>{event.component}</div>
                    <div style={{ fontSize: '14px', marginTop: '4px' }}>
                      {event.message}
                    </div>
                    {event.duration && (
                      <div
                        style={{
                          fontSize: '12px',
                          color: '#666',
                          marginTop: '2px',
                        }}
                      >
                        持续时间: {Math.floor(event.duration / 60)}分
                        {event.duration % 60}秒
                      </div>
                    )}
                  </div>
                </Timeline.Item>
              ))}
            </Timeline>
          </Card>
        </Col>
      </Row>

      {/* 服务健康详情 */}
      <Card title="服务健康详情" loading={loading}>
        <Table
          dataSource={serviceHealth}
          columns={serviceColumns}
          rowKey="service"
          pagination={false}
          size="middle"
        />
      </Card>
    </div>
  )
}

export default RLSystemHealthPage
