import React, { useState, useEffect, useMemo, useRef, useCallback } from 'react'
import { logger } from '../utils/logger'
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
  Empty,
  Typography,
  Timeline,
  List,
  Badge,
  Tabs,
  message,
  Spin,
} from 'antd'
import {
  HeartOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  WarningOutlined,
  ReloadOutlined,
  ApiOutlined,
  ExclamationCircleOutlined,
} from '@ant-design/icons'
import {
  healthService,
  type HealthAlert,
  type HealthAlertsResponse,
  type HealthCheckResult,
  type SystemMetrics,
} from '../services/healthService'

const { Title, Text } = Typography
const { TabPane } = Tabs

interface HealthCheck {
  service: string
  status: 'healthy' | 'unhealthy' | 'degraded'
  responseTime: number | null
  lastCheck: string
  uptime: number | null
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
  timestampMs: number
  resolved: boolean
}

const resolveTimestampMs = (value?: string) => {
  if (!value) return Date.now()
  const parsed = new Date(value).getTime()
  return Number.isFinite(parsed) ? parsed : Date.now()
}

type HealthMonitorPageProps = {
  title?: string
}

const HealthMonitorPage: React.FC<HealthMonitorPageProps> = ({
  title = '系统健康监控',
}) => {
  const [healthChecks, setHealthChecks] = useState<HealthCheck[]>([])
  const [systemMetrics, setSystemMetrics] = useState<SystemMetric[]>([])
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [lastUpdateAt, setLastUpdateAt] = useState('')
  const [lastUpdateError, setLastUpdateError] = useState<string | null>(null)
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const loadingRef = useRef(false)
  const mountedRef = useRef(true)

  const formatDuration = (seconds: number | null) => {
    if (!seconds || Number.isNaN(seconds)) return '-'
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    const secs = Math.floor(seconds % 60)
    if (hours > 0) return `${hours}小时 ${minutes}分钟`
    if (minutes > 0) return `${minutes}分钟 ${secs}秒`
    return `${secs}秒`
  }

  // 加载健康状态数据
  const loadHealthData = useCallback(async (silent: boolean = false) => {
    if (loadingRef.current) return false
    loadingRef.current = true
    let success = true
    try {
      if (silent) {
        setRefreshing(true)
      } else {
        setLoading(true)
      }

      const [healthResult, metricsResult, alertsResult] =
        await Promise.allSettled([
          healthService.getDetailedHealth(),
          healthService.getSystemMetrics(),
          healthService.getHealthAlerts(),
        ])
      if (!mountedRef.current) return false

      const healthOk = healthResult.status === 'fulfilled'
      const metricsOk = metricsResult.status === 'fulfilled'
      const alertsOk = alertsResult.status === 'fulfilled'

      if (!healthOk) {
        logger.error('获取健康状态失败:', healthResult.reason)
      }
      if (!metricsOk) {
        logger.error('获取系统指标失败:', metricsResult.reason)
      }
      if (!alertsOk) {
        logger.error('获取健康告警失败:', alertsResult.reason)
      }

      if (healthOk) {
        const healthStatus = healthResult.value as HealthCheckResult
        const components = healthStatus.components ?? {}
        const lastCheckTime = new Date(
          healthStatus.timestamp || Date.now()
        ).toLocaleString()
        const checks: HealthCheck[] = Object.entries(components).map(
          ([service, data]) => {
            const payload = data as Record<string, unknown>
            const responseTime =
              typeof payload.response_time_ms === 'number'
                ? payload.response_time_ms
                : null
            const uptime =
              typeof payload.uptime_seconds === 'number'
                ? payload.uptime_seconds
                : null
            const details =
              (typeof payload.error === 'string' && payload.error) ||
              (typeof payload.note === 'string' && payload.note) ||
              (Array.isArray(payload.warnings)
                ? payload.warnings.join('，')
                : undefined)
            const rawStatus =
              typeof payload.status === 'string' ? payload.status : ''
            return {
              service,
              status:
                rawStatus === 'healthy'
                  ? 'healthy'
                  : rawStatus === 'degraded'
                    ? 'degraded'
                    : 'unhealthy',
              responseTime,
              lastCheck: lastCheckTime,
              uptime,
              details,
            }
          }
        )
        setHealthChecks(checks)
      }

      if (metricsOk) {
        const metricsPayload = metricsResult.value as SystemMetrics
        const metrics: SystemMetric[] = []
        const systemInfo = metricsPayload.system
        const performanceInfo = metricsPayload.performance
        const cpuPercent = systemInfo?.cpu_percent
        const memoryPercent = systemInfo?.memory_percent
        const diskPercent = systemInfo?.disk_percent
        const avgResponseTime = performanceInfo?.average_response_time_ms

        if (typeof cpuPercent === 'number') {
          metrics.push({
            name: 'CPU使用率',
            value: cpuPercent,
            unit: '%',
            status:
              cpuPercent >= 95
                ? 'critical'
                : cpuPercent >= 80
                  ? 'warning'
                  : 'normal',
            threshold: 100,
          })
        }

        if (typeof memoryPercent === 'number') {
          metrics.push({
            name: '内存使用率',
            value: Number(memoryPercent.toFixed(1)),
            unit: '%',
            status:
              memoryPercent >= 95
                ? 'critical'
                : memoryPercent >= 80
                  ? 'warning'
                  : 'normal',
            threshold: 100,
          })
        }

        if (typeof diskPercent === 'number') {
          metrics.push({
            name: '磁盘使用率',
            value: Number(diskPercent.toFixed(1)),
            unit: '%',
            status:
              diskPercent >= 95
                ? 'critical'
                : diskPercent >= 80
                  ? 'warning'
                  : 'normal',
            threshold: 100,
          })
        }

        if (typeof avgResponseTime === 'number') {
          metrics.push({
            name: '平均响应时间',
            value: Number(avgResponseTime.toFixed(1)),
            unit: 'ms',
            status:
              avgResponseTime >= 1000
                ? 'critical'
                : avgResponseTime >= 500
                  ? 'warning'
                  : 'normal',
            threshold: 1000,
          })
        }

        setSystemMetrics(metrics)
      }

      if (alertsOk) {
        const alertsPayload = alertsResult.value as HealthAlertsResponse
        const alertList: Alert[] = (alertsPayload.alerts || []).map(
          (alert: HealthAlert) => {
            const timestampMs = resolveTimestampMs(alert.timestamp)
            return {
              id: `${alert.name || 'alert'}-${alert.timestamp || Date.now()}`,
              level:
                alert.severity === 'critical'
                  ? 'critical'
                  : alert.severity === 'warning'
                    ? 'warning'
                    : alert.severity === 'error'
                      ? 'error'
                      : 'info',
              message: alert.message,
              service: alert.name || 'system',
              timestamp: new Date(timestampMs).toLocaleString(),
              timestampMs,
              resolved: Boolean(
                (alert as HealthAlert & { resolved?: boolean }).resolved
              ),
            }
          }
        )
        setAlerts(alertList)
      }
      success = healthOk && metricsOk && alertsOk
      if (mountedRef.current) {
        const failedParts = []
        if (!healthOk) failedParts.push('健康状态')
        if (!metricsOk) failedParts.push('系统指标')
        if (!alertsOk) failedParts.push('告警')
        setLastUpdateAt(new Date().toLocaleString())
        setLastUpdateError(
          failedParts.length > 0
            ? `更新失败：${failedParts.join('、')}`
            : null
        )
      }
      if (!success && !silent) {
        if (healthOk || metricsOk || alertsOk) {
          message.warning('部分健康数据获取失败')
        } else {
          message.error('获取健康数据失败')
        }
      }
    } catch (error) {
      success = false
      logger.error('加载健康数据失败:', error)
      if (mountedRef.current) {
        setLastUpdateAt(new Date().toLocaleString())
        setLastUpdateError('更新失败')
      }
      if (!silent) {
        message.error('获取健康数据失败')
      }
    } finally {
      loadingRef.current = false
      if (mountedRef.current) {
        if (silent) {
          setRefreshing(false)
        } else {
          setLoading(false)
        }
      }
    }
    return success
  }, [])

  // 手动刷新
  const handleRefresh = async () => {
    const refreshed = await loadHealthData()
    if (refreshed) message.success('数据已刷新')
  }

  // 初始加载和自动刷新
  useEffect(() => {
    return () => {
      mountedRef.current = false
    }
  }, [])

  useEffect(() => {
    loadHealthData()
  }, [loadHealthData])

  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(() => {
        if (!document.hidden) {
          loadHealthData(true)
        }
      }, 30000) // 30秒刷新一次
      return () => clearInterval(interval)
    }
  }, [autoRefresh, loadHealthData])

  const getStatusColor = (status: string) => {
    const colors = {
      healthy: 'success',
      degraded: 'warning',
      unhealthy: 'error',
    }
    return colors[status as keyof typeof colors] || 'default'
  }

  const getStatusIcon = (status: string) => {
    const icons = {
      healthy: <CheckCircleOutlined />,
      degraded: <WarningOutlined />,
      unhealthy: <CloseCircleOutlined />,
    }
    return icons[status as keyof typeof icons]
  }

  const getMetricStatusColor = (status: string) => {
    const colors = {
      normal: '#52c41a',
      warning: '#faad14',
      critical: '#ff4d4f',
    }
    return colors[status as keyof typeof colors]
  }

  const getAlertColor = (level: string) => {
    const colors = {
      info: 'blue',
      warning: 'orange',
      error: 'red',
      critical: 'red',
    }
    return colors[level as keyof typeof colors]
  }

  const getAlertStatus = (
    level: string
  ): 'success' | 'processing' | 'default' | 'error' | 'warning' => {
    const statusMap = {
      info: 'processing' as const,
      warning: 'warning' as const,
      error: 'error' as const,
      critical: 'error' as const,
    }
    return statusMap[level as keyof typeof statusMap] || 'default'
  }

  const healthSummary = useMemo(() => {
    let healthy = 0
    let degraded = 0
    let unhealthy = 0
    for (const check of healthChecks) {
      if (check.status === 'healthy') healthy += 1
      else if (check.status === 'degraded') degraded += 1
      else unhealthy += 1
    }
    const total = healthChecks.length
    const score = total > 0 ? (healthy / total) * 100 : 0
    return { healthy, degraded, unhealthy, total, score }
  }, [healthChecks])

  const alertSummary = useMemo(() => {
    let unresolved = 0
    let critical = 0
    let error = 0
    for (const alert of alerts) {
      if (alert.resolved) continue
      unresolved += 1
      if (alert.level === 'critical') critical += 1
      if (alert.level === 'error') error += 1
    }
    return { unresolved, critical, error }
  }, [alerts])

  const timelineItems = useMemo(() => {
    if (alerts.length === 0) return []
    return [...alerts]
      .sort((a, b) => b.timestampMs - a.timestampMs)
      .slice(0, 10)
      .map(alert => ({
        color: getAlertColor(alert.level),
        children: (
          <div>
            <Space>
              <Tag color={getAlertColor(alert.level)}>
                {alert.level.toUpperCase()}
              </Tag>
              <Text strong>{alert.message || '告警'}</Text>
            </Space>
            <div>
              <Text type="secondary">
                服务: {alert.service} · {alert.timestamp}
              </Text>
            </div>
          </div>
        ),
      }))
  }, [alerts])

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
      ),
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
      ),
    },
    {
      title: '响应时间',
      dataIndex: 'responseTime',
      key: 'responseTime',
      render: (time: number | null) =>
        typeof time === 'number' ? (
          <Text
            style={{
              color:
                time > 500 ? '#ff4d4f' : time > 200 ? '#faad14' : '#52c41a',
            }}
          >
            {time}ms
          </Text>
        ) : (
          <Text type="secondary">-</Text>
        ),
    },
    {
      title: '正常运行时间',
      dataIndex: 'uptime',
      key: 'uptime',
      render: (uptime: number | null) => (
        <Text type="secondary">{formatDuration(uptime)}</Text>
      ),
    },
    {
      title: '最后检查',
      dataIndex: 'lastCheck',
      key: 'lastCheck',
    },
    {
      title: '详情',
      dataIndex: 'details',
      key: 'details',
      render: (details?: string) => details || '-',
    },
  ]

  return (
    <div className="p-6">
      <div className="mb-6">
        <div className="flex justify-between items-center mb-4">
          <Title level={2}>{title}</Title>
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
            {refreshing && (
              <Text type="secondary" style={{ fontSize: 12 }}>
                后台刷新中...
              </Text>
            )}
            {lastUpdateAt && (
              <Text type="secondary" style={{ fontSize: 12 }}>
                最近更新: {lastUpdateAt}
              </Text>
            )}
            {lastUpdateError && (
              <Text type="danger" style={{ fontSize: 12 }}>
                {lastUpdateError}
              </Text>
            )}
          </Space>
        </div>

        {healthSummary.score < 80 && (
          <Alert
            message="系统健康状态异常"
            description={`当前有 ${healthSummary.total - healthSummary.healthy} 个服务存在问题，建议立即检查系统状态`}
            type="error"
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
                value={healthSummary.score}
                precision={1}
                suffix="%"
                valueStyle={{
                  color:
                    healthSummary.score > 90
                      ? '#3f8600'
                      : healthSummary.score > 70
                        ? '#faad14'
                        : '#cf1322',
                }}
                prefix={<HeartOutlined />}
              />
              <Progress
                percent={healthSummary.score}
                strokeColor={
                  healthSummary.score > 90
                    ? '#52c41a'
                    : healthSummary.score > 70
                      ? '#faad14'
                      : '#ff4d4f'
                }
                className="mt-2"
              />
            </Card>
          </Col>
          <Col span={8}>
            <Card>
              <Statistic
                title="健康服务"
                value={healthSummary.healthy}
                suffix={`/ ${healthSummary.total}`}
                valueStyle={{ color: '#3f8600' }}
                prefix={<CheckCircleOutlined />}
              />
              <div className="mt-2 text-xs text-gray-500">
                {healthSummary.degraded} 个降级, {healthSummary.unhealthy}{' '}
                个不健康
              </div>
            </Card>
          </Col>
          <Col span={8}>
            <Card>
              <Statistic
                title="未解决告警"
                value={alertSummary.unresolved}
                valueStyle={{ color: '#cf1322' }}
                prefix={<ExclamationCircleOutlined />}
              />
              <div className="mt-2 text-xs text-gray-500">
                {alertSummary.critical} 个严重, {alertSummary.error} 个错误
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
            <>
              {systemMetrics.length === 0 ? (
                <Empty description="暂无系统指标数据" />
              ) : (
                <Row gutter={16}>
                  {systemMetrics.map((metric, index) => (
                    <Col span={8} key={index} className="mb-4">
                      <Card>
                        <div className="flex justify-between items-center mb-2">
                          <Text strong>{metric.name}</Text>
                          <Badge
                            status={
                              metric.status === 'normal'
                                ? 'success'
                                : metric.status === 'warning'
                                  ? 'warning'
                                  : 'error'
                            }
                          />
                        </div>
                        <div
                          className="text-2xl font-bold mb-2"
                          style={{ color: getMetricStatusColor(metric.status) }}
                        >
                          {metric.value}
                          {metric.unit}
                        </div>
                        <Progress
                          percent={(metric.value / metric.threshold) * 100}
                          strokeColor={getMetricStatusColor(metric.status)}
                          size="small"
                        />
                        <div className="text-xs text-gray-500 mt-1">
                          阈值: {metric.threshold}
                          {metric.unit}
                        </div>
                      </Card>
                    </Col>
                  ))}
                </Row>
              )}
            </>
          )}
        </TabPane>

        <TabPane tab={`告警 (${alertSummary.unresolved})`} key="alerts">
          <Card title="系统告警">
            {alerts.length === 0 ? (
              <Empty description="暂无告警" />
            ) : (
              <List
                dataSource={alerts}
                renderItem={alert => (
                  <List.Item
                    style={{
                      backgroundColor: alert.resolved ? '#f6f6f6' : 'white',
                      opacity: alert.resolved ? 0.7 : 1,
                    }}
                  >
                    <List.Item.Meta
                      avatar={
                        <Badge
                          status={
                            alert.resolved
                              ? 'default'
                              : getAlertStatus(alert.level)
                          }
                        />
                      }
                      title={
                        <div className="flex justify-between items-center">
                          <Space>
                            <Tag color={getAlertColor(alert.level)}>
                              {alert.level.toUpperCase()}
                            </Tag>
                            <Text strong={!alert.resolved}>
                              {alert.message}
                            </Text>
                          </Space>
                          <Text type="secondary" className="text-xs">
                            {alert.timestamp}
                          </Text>
                        </div>
                      }
                      description={
                        <div className="flex justify-between items-center">
                          <Text type="secondary">服务: {alert.service}</Text>
                          {alert.resolved && <Tag color="green">已解决</Tag>}
                        </div>
                      }
                    />
                  </List.Item>
                )}
              />
            )}
          </Card>
        </TabPane>

        <TabPane tab="实时日志" key="logs">
          <Card title="健康检查日志">
            {timelineItems.length === 0 ? (
              <Empty description="暂无健康事件" />
            ) : (
              <Timeline items={timelineItems} />
            )}
          </Card>
        </TabPane>
      </Tabs>
    </div>
  )
}

export default HealthMonitorPage
