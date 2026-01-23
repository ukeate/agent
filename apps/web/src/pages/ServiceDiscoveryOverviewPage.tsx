import React, { useState, useEffect, useRef } from 'react'
import {
  Card,
  Row,
  Col,
  Statistic,
  Progress,
  Table,
  Badge,
  Space,
  Button,
  Typography,
  Tag,
  Timeline,
  Alert,
} from 'antd'
import { logger } from '../utils/logger'
import {
  CloudServerOutlined,
  TeamOutlined,
  ThunderboltOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  ExclamationCircleOutlined,
  SyncOutlined,
  GlobalOutlined,
  MonitorOutlined,
  BarChartOutlined,
  ReloadOutlined,
  SettingOutlined,
} from '@ant-design/icons'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  AreaChart,
  Area,
} from 'recharts'
import { serviceDiscoveryService } from '../services/serviceDiscoveryService'
import { clusterManagementService } from '../services/clusterManagementService'

const { Title, Paragraph, Text } = Typography

interface ServiceDiscoveryOverviewPageProps {}

interface AgentStats {
  total: number
  healthy: number
  unhealthy: number
  registering: number
  activeConnections: number
  avgResponseTime: number
  throughputPerSecond: number
  uptime: number
}

interface ServiceMetric {
  timestamp: string
  registrations: number
  discoveries: number
  healthChecks: number
  responseTime: number
  errors: number
}

interface LoadBalancerStats {
  algorithm: string
  requestsPerSecond: number
  avgLatency: number
  successRate: number
  activeNodes: number
}

interface ClusterNode {
  id: string
  name: string
  status: 'healthy' | 'unhealthy' | 'joining'
  ip: string
  port: number
  role: 'leader' | 'follower' | 'learner'
  uptime: number
  load: number
  connections: number
}

const ServiceDiscoveryOverviewPage: React.FC<
  ServiceDiscoveryOverviewPageProps
> = () => {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [systemStatus, setSystemStatus] = useState<string>('initializing')
  const [agentStats, setAgentStats] = useState<AgentStats>({
    total: 0,
    healthy: 0,
    unhealthy: 0,
    registering: 0,
    activeConnections: 0,
    avgResponseTime: 0,
    throughputPerSecond: 0,
    uptime: 0,
  })

  const [metrics, setMetrics] = useState<ServiceMetric[]>([])

  const [loadBalancerStats, setLoadBalancerStats] = useState<LoadBalancerStats>(
    {
      algorithm: 'capability_based',
      requestsPerSecond: 0,
      avgLatency: 0,
      successRate: 0,
      activeNodes: 0,
    }
  )

  const [clusterNodes, setClusterNodes] = useState<ClusterNode[]>([])

  const [recentEvents, setRecentEvents] = useState<
    Array<{ time: string; type: string; message: string; status: string }>
  >([])
  const lastSnapshotRef = useRef<{
    total: number
    unhealthy: number
    status: string
  } | null>(null)

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return '#52c41a'
      case 'unhealthy':
        return '#ff4d4f'
      case 'joining':
        return '#faad14'
      default:
        return '#d9d9d9'
    }
  }

  const getRoleColor = (role: string) => {
    switch (role) {
      case 'leader':
        return 'red'
      case 'follower':
        return 'blue'
      case 'learner':
        return 'orange'
      default:
        return 'default'
    }
  }

  const getEventColor = (status: string) => {
    switch (status) {
      case 'success':
        return 'green'
      case 'warning':
        return 'orange'
      case 'error':
        return 'red'
      case 'processing':
        return 'blue'
      default:
        return 'default'
    }
  }

  const loadData = async () => {
    try {
      setLoading(true)
      setError(null)

      const [stats, agents] = await Promise.all([
        serviceDiscoveryService.getSystemStats(),
        clusterManagementService.getAgents().catch(() => []),
      ])
      const statusFromApi = stats?.system_status || 'unknown'
      const agentList = Array.isArray(agents) ? agents : []

      const registry = stats?.registry || {}
      const statusCounts = registry.agents_by_status || {}
      const total =
        typeof registry.registered_agents === 'number'
          ? registry.registered_agents
          : typeof registry.total_agents === 'number'
            ? registry.total_agents
            : 0
      const healthyFromStats = statusCounts.active || 0
      const unhealthyFromStats = statusCounts.unhealthy || 0
      const registeringFromStats = statusCounts.maintenance || 0
      const derivedTotal = total || agentList.length
      const derivedHealthy =
        healthyFromStats ||
        agentList.filter((a: any) => a.is_healthy || a.status === 'online')
          .length
      const derivedUnhealthy =
        unhealthyFromStats ||
        agentList.filter(
          (a: any) =>
            a.status === 'offline' ||
            a.status === 'failed' ||
            a.is_healthy === false
        ).length
      const derivedRegistering =
        registeringFromStats ||
        agentList.filter(
          (a: any) => a.status === 'starting' || a.status === 'registering'
        ).length

      const lb = stats?.load_balancer || {}
      const connectionStats = lb.connection_stats || {}
      let activeConnections = 0
      Object.values(connectionStats || {}).forEach((item: any) => {
        if (item && typeof (item as any).active_connections === 'number') {
          activeConnections += (item as any).active_connections
        }
      })
      if (!activeConnections && agentList.length) {
        activeConnections = agentList.reduce((sum: number, a: any) => {
          const tasks =
            a.resource_usage &&
            typeof a.resource_usage.active_tasks === 'number'
              ? a.resource_usage.active_tasks
              : 0
          return sum + tasks
        }, 0)
      }

      const avgResponseSeconds = registry.avg_response_time || 0
      const healthChecks = registry.health_checks || 0
      const failedHealth = registry.failed_health_checks || 0
      let successRate = 100
      if (healthChecks > 0) {
        successRate = ((healthChecks - failedHealth) / healthChecks) * 100
      } else if (derivedTotal > 0) {
        successRate = (derivedHealthy / derivedTotal) * 100
      }
      const resolvedStatus =
        derivedTotal > 0
          ? derivedUnhealthy > 0
            ? 'degraded'
            : 'healthy'
          : statusFromApi === 'unknown'
            ? 'no_agents'
            : statusFromApi
      setSystemStatus(resolvedStatus)

      setAgentStats({
        total: derivedTotal,
        healthy: derivedHealthy,
        unhealthy: derivedUnhealthy,
        registering: derivedRegistering,
        activeConnections,
        avgResponseTime: Math.round((avgResponseSeconds || 0) * 1000),
        throughputPerSecond: registry.discovery_requests || 0,
        uptime: Math.round(successRate * 10) / 10,
      })

      const newMetric: ServiceMetric = {
        timestamp: new Date().toLocaleTimeString('zh-CN', {
          hour12: false,
          hour: '2-digit',
          minute: '2-digit',
        }),
        registrations: derivedTotal,
        discoveries: registry.discovery_requests || 0,
        healthChecks: healthChecks || 0,
        responseTime: Math.round((avgResponseSeconds || 0) * 1000),
        errors: failedHealth || 0,
      }

      setMetrics(prev => {
        const next = [...prev, newMetric]
        return next.length > 20 ? next.slice(next.length - 20) : next
      })

      setLoadBalancerStats({
        algorithm:
          (lb.available_strategies && lb.available_strategies[0]) ||
          'capability_based',
        requestsPerSecond: registry.discovery_requests || 0,
        avgLatency: Math.round((avgResponseSeconds || 0) * 1000),
        successRate: Math.round(successRate * 10) / 10,
        activeNodes: derivedHealthy,
      })

      const nowLabel = new Date().toLocaleTimeString('zh-CN', {
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      })
      const nextEvents: Array<{
        time: string
        type: string
        message: string
        status: string
      }> = []
      const lastSnapshot = lastSnapshotRef.current
      if (!lastSnapshot) {
        nextEvents.push({
          time: nowLabel,
          type: 'system',
          message: `系统状态：${resolvedStatus}`,
          status:
            resolvedStatus === 'healthy'
              ? 'success'
              : resolvedStatus === 'degraded'
                ? 'warning'
                : 'processing',
        })
      } else {
        if (lastSnapshot.status !== resolvedStatus) {
          nextEvents.push({
            time: nowLabel,
            type: 'system',
            message: `系统状态变更：${lastSnapshot.status} → ${resolvedStatus}`,
            status: resolvedStatus === 'healthy' ? 'success' : 'warning',
          })
        }
        if (lastSnapshot.total !== derivedTotal) {
          const diff = derivedTotal - lastSnapshot.total
          nextEvents.push({
            time: nowLabel,
            type: 'agent',
            message:
              diff > 0
                ? `新增智能体 ${diff} 个`
                : `移除智能体 ${Math.abs(diff)} 个`,
            status: diff > 0 ? 'success' : 'warning',
          })
        }
        if (lastSnapshot.unhealthy !== derivedUnhealthy) {
          const diff = derivedUnhealthy - lastSnapshot.unhealthy
          nextEvents.push({
            time: nowLabel,
            type: 'health',
            message:
              diff > 0
                ? `不健康智能体增加 ${diff} 个`
                : `不健康智能体减少 ${Math.abs(diff)} 个`,
            status: diff > 0 ? 'error' : 'success',
          })
        }
      }
      lastSnapshotRef.current = {
        total: derivedTotal,
        unhealthy: derivedUnhealthy,
        status: resolvedStatus,
      }
      setRecentEvents(prev => {
        const merged = [...nextEvents, ...prev]
        if (!merged.length) {
          return [
            {
              time: nowLabel,
              type: 'system',
              message: '暂无事件更新',
              status: 'processing',
            },
          ]
        }
        return merged.slice(0, 6)
      })

      setClusterNodes(
        agentList.map((agent: any, index: number) => {
          const endpoint = agent.endpoint || ''
          let ip = agent.node_id || 'unknown'
          let port = 0
          if (endpoint) {
            try {
              const url = new URL(endpoint)
              ip = url.hostname
              port = url.port ? parseInt(url.port, 10) : 80
            } catch {
              const parts = endpoint.split('://').pop() || ''
              const hostPort = parts.split('/')[0] || ''
              const split = hostPort.split(':')
              if (split[0]) ip = split[0]
              if (split[1]) port = parseInt(split[1], 10)
            }
          }
          const status: 'healthy' | 'unhealthy' | 'joining' =
            agent.is_healthy || agent.status === 'online'
              ? 'healthy'
              : agent.status === 'starting'
                ? 'joining'
                : 'unhealthy'
          const role: 'leader' | 'follower' | 'learner' =
            index === 0 ? 'leader' : 'follower'
          const uptimePercent = agent.uptime
            ? Math.min(100, Math.max(0, Math.round(agent.uptime / 36)))
            : 0
          const load =
            typeof agent.current_load === 'number' ? agent.current_load : 0
          const connections =
            agent.resource_usage &&
            typeof agent.resource_usage.active_tasks === 'number'
              ? agent.resource_usage.active_tasks
              : 0

          return {
            id: agent.agent_id,
            name: agent.name || agent.agent_id,
            status,
            ip,
            port,
            role,
            uptime: uptimePercent,
            load,
            connections,
          }
        })
      )
    } catch (err) {
      logger.error('加载服务发现数据失败:', err)
      setError((err as Error).message || '加载服务发现数据失败')
    } finally {
      setLoading(false)
    }
  }

  const pieData = [
    { name: '健康服务', value: agentStats.healthy, color: '#52c41a' },
    { name: '异常服务', value: agentStats.unhealthy, color: '#ff4d4f' },
    { name: '注册中', value: agentStats.registering, color: '#faad14' },
  ]

  const clusterColumns = [
    {
      title: '节点名称',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: ClusterNode) => (
        <Space>
          <Badge color={getStatusColor(record.status)} />
          <Text strong>{text}</Text>
          <Tag color={getRoleColor(record.role)}>
            {record.role.toUpperCase()}
          </Tag>
        </Space>
      ),
    },
    {
      title: '地址',
      key: 'address',
      render: (_, record: ClusterNode) => `${record.ip}:${record.port}`,
    },
    {
      title: '运行时间',
      dataIndex: 'uptime',
      key: 'uptime',
      render: (uptime: number) => (
        <Space>
          <Progress
            percent={uptime}
            size="small"
            status={
              uptime > 95 ? 'success' : uptime > 85 ? 'active' : 'exception'
            }
          />
          <Text>{uptime}%</Text>
        </Space>
      ),
    },
    {
      title: '系统负载',
      dataIndex: 'load',
      key: 'load',
      render: (load: number) => (
        <Progress
          percent={load}
          size="small"
          status={load < 70 ? 'success' : load < 85 ? 'active' : 'exception'}
        />
      ),
    },
    {
      title: '连接数',
      dataIndex: 'connections',
      key: 'connections',
      render: (connections: number) => (
        <Statistic value={connections} valueStyle={{ fontSize: '14px' }} />
      ),
    },
  ]

  useEffect(() => {
    loadData()
    const interval = setInterval(loadData, 30000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div style={{ padding: '24px', background: '#f5f5f5', minHeight: '100vh' }}>
      <div style={{ maxWidth: '1400px', margin: '0 auto' }}>
        {/* 页面标题 */}
        <div style={{ marginBottom: '24px' }}>
          <Title level={2}>
            <GlobalOutlined /> 智能代理服务发现总览
          </Title>
          <Paragraph>
            基于etcd的分布式智能代理服务发现系统，提供服务注册、发现、负载均衡和健康监控功能。
          </Paragraph>
          <Space>
            <Button
              type="primary"
              icon={<ReloadOutlined />}
              onClick={loadData}
              loading={loading}
            >
              刷新数据
            </Button>
            <Button icon={<SettingOutlined />}>系统配置</Button>
            <Button icon={<MonitorOutlined />}>监控面板</Button>
          </Space>
        </div>

        {error && (
          <Alert
            message="加载服务发现数据失败"
            description={error}
            type="error"
            showIcon
            style={{ marginBottom: '16px' }}
          />
        )}

        {/* 系统状态告警 */}
        <Alert
          message={
            systemStatus === 'healthy'
              ? '系统运行正常'
              : systemStatus === 'no_agents'
                ? '暂无已注册智能体'
                : `系统状态: ${systemStatus}`
          }
          description={`已注册智能体: ${agentStats.total}，活跃: ${agentStats.healthy}，异常: ${agentStats.unhealthy}`}
          type={systemStatus === 'healthy' ? 'success' : 'warning'}
          icon={<CheckCircleOutlined />}
          showIcon
          style={{ marginBottom: '24px' }}
        />

        {/* 核心指标卡片 */}
        <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
          <Col xs={24} sm={12} lg={6}>
            <Card>
              <Statistic
                title="总注册服务"
                value={agentStats.total}
                prefix={<TeamOutlined />}
                valueStyle={{ color: '#1890ff' }}
              />
              <div style={{ marginTop: '8px' }}>
                <Text type="secondary">
                  健康: {agentStats.healthy} | 异常: {agentStats.unhealthy}
                </Text>
              </div>
            </Card>
          </Col>
          <Col xs={24} sm={12} lg={6}>
            <Card>
              <Statistic
                title="活跃连接"
                value={agentStats.activeConnections}
                prefix={<CloudServerOutlined />}
                valueStyle={{ color: '#52c41a' }}
              />
              <div style={{ marginTop: '8px' }}>
                <Text type="secondary">
                  QPS: {agentStats.throughputPerSecond}
                </Text>
              </div>
            </Card>
          </Col>
          <Col xs={24} sm={12} lg={6}>
            <Card>
              <Statistic
                title="平均响应时间"
                value={agentStats.avgResponseTime}
                suffix="ms"
                prefix={<ThunderboltOutlined />}
                valueStyle={{ color: '#faad14' }}
              />
              <div style={{ marginTop: '8px' }}>
                <Text type="secondary">目标: 未配置</Text>
              </div>
            </Card>
          </Col>
          <Col xs={24} sm={12} lg={6}>
            <Card>
              <Statistic
                title="系统可用性"
                value={agentStats.uptime}
                suffix="%"
                prefix={<CheckCircleOutlined />}
                valueStyle={{
                  color: agentStats.uptime > 99 ? '#52c41a' : '#faad14',
                }}
              />
              <div style={{ marginTop: '8px' }}>
                <Text type="secondary">SLA: 未配置</Text>
              </div>
            </Card>
          </Col>
        </Row>

        <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
          {/* 服务健康状态分布 */}
          <Col xs={24} lg={8}>
            <Card title="服务健康状态分布" extra={<BarChartOutlined />}>
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={40}
                    outerRadius={80}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
              <div style={{ textAlign: 'center', marginTop: '16px' }}>
                {pieData.map((item, index) => (
                  <Tag key={index} color={item.color} style={{ margin: '4px' }}>
                    {item.name}: {item.value}
                  </Tag>
                ))}
              </div>
            </Card>
          </Col>

          {/* 负载均衡状态 */}
          <Col xs={24} lg={8}>
            <Card title="负载均衡状态" extra={<ThunderboltOutlined />}>
              <div style={{ padding: '16px 0' }}>
                <Row gutter={[16, 16]}>
                  <Col span={12}>
                    <Statistic
                      title="当前策略"
                      value={loadBalancerStats.algorithm}
                      valueStyle={{ fontSize: '16px' }}
                    />
                  </Col>
                  <Col span={12}>
                    <Statistic
                      title="活跃节点"
                      value={loadBalancerStats.activeNodes}
                      valueStyle={{ color: '#52c41a' }}
                    />
                  </Col>
                  <Col span={12}>
                    <Statistic
                      title="请求/秒"
                      value={loadBalancerStats.requestsPerSecond}
                      valueStyle={{ fontSize: '14px' }}
                    />
                  </Col>
                  <Col span={12}>
                    <Statistic
                      title="成功率"
                      value={loadBalancerStats.successRate}
                      suffix="%"
                      valueStyle={{ color: '#52c41a', fontSize: '14px' }}
                    />
                  </Col>
                </Row>
              </div>
            </Card>
          </Col>

          {/* 最近事件 */}
          <Col xs={24} lg={8}>
            <Card title="最近事件" extra={<ClockCircleOutlined />}>
              <Timeline size="small">
                {recentEvents.slice(0, 6).map((event, index) => (
                  <Timeline.Item
                    key={index}
                    dot={
                      event.status === 'success' ? (
                        <CheckCircleOutlined style={{ color: '#52c41a' }} />
                      ) : event.status === 'warning' ? (
                        <ExclamationCircleOutlined
                          style={{ color: '#faad14' }}
                        />
                      ) : event.status === 'processing' ? (
                        <SyncOutlined spin style={{ color: '#1890ff' }} />
                      ) : (
                        <ClockCircleOutlined style={{ color: '#d9d9d9' }} />
                      )
                    }
                  >
                    <div>
                      <Text type="secondary" style={{ fontSize: '12px' }}>
                        {event.time}
                      </Text>
                      <br />
                      <Text style={{ fontSize: '13px' }}>{event.message}</Text>
                    </div>
                  </Timeline.Item>
                ))}
              </Timeline>
            </Card>
          </Col>
        </Row>

        {/* 性能指标趋势 */}
        <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
          <Col xs={24} lg={12}>
            <Card title="服务发现趋势" extra={<BarChartOutlined />}>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={metrics}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="timestamp" />
                  <YAxis />
                  <Tooltip />
                  <Area
                    type="monotone"
                    dataKey="registrations"
                    stackId="1"
                    stroke="#1890ff"
                    fill="#1890ff"
                    fillOpacity={0.3}
                  />
                  <Area
                    type="monotone"
                    dataKey="discoveries"
                    stackId="2"
                    stroke="#52c41a"
                    fill="#52c41a"
                    fillOpacity={0.3}
                  />
                  <Area
                    type="monotone"
                    dataKey="healthChecks"
                    stackId="3"
                    stroke="#faad14"
                    fill="#faad14"
                    fillOpacity={0.3}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </Card>
          </Col>

          <Col xs={24} lg={12}>
            <Card title="响应时间与错误率" extra={<MonitorOutlined />}>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={metrics}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="timestamp" />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" />
                  <Tooltip />
                  <Line
                    yAxisId="left"
                    type="monotone"
                    dataKey="responseTime"
                    stroke="#1890ff"
                    strokeWidth={2}
                    name="响应时间 (ms)"
                  />
                  <Line
                    yAxisId="right"
                    type="monotone"
                    dataKey="errors"
                    stroke="#ff4d4f"
                    strokeWidth={2}
                    name="错误数"
                  />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          </Col>
        </Row>

        {/* etcd集群状态 */}
        <Card
          title="etcd集群状态"
          extra={<CloudServerOutlined />}
          style={{ marginBottom: '24px' }}
        >
          <Table
            columns={clusterColumns}
            dataSource={clusterNodes}
            rowKey="id"
            pagination={false}
            size="small"
          />
        </Card>
      </div>
    </div>
  )
}

export default ServiceDiscoveryOverviewPage
