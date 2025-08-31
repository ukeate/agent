import React, { useState, useEffect } from 'react'
import { Card, Row, Col, Statistic, Progress, Table, Badge, Space, Button, Typography, Tag, Timeline, Alert } from 'antd'
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
  SettingOutlined
} from '@ant-design/icons'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, AreaChart, Area } from 'recharts'

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

const ServiceDiscoveryOverviewPage: React.FC<ServiceDiscoveryOverviewPageProps> = () => {
  const [agentStats, setAgentStats] = useState<AgentStats>({
    total: 125,
    healthy: 118,
    unhealthy: 4,
    registering: 3,
    activeConnections: 1247,
    avgResponseTime: 45,
    throughputPerSecond: 892,
    uptime: 99.7
  })

  const [metrics, setMetrics] = useState<ServiceMetric[]>([
    { timestamp: '10:00', registrations: 65, discoveries: 145, healthChecks: 289, responseTime: 42, errors: 2 },
    { timestamp: '10:05', registrations: 72, discoveries: 156, healthChecks: 298, responseTime: 38, errors: 1 },
    { timestamp: '10:10', registrations: 68, discoveries: 142, healthChecks: 305, responseTime: 46, errors: 3 },
    { timestamp: '10:15', registrations: 75, discoveries: 168, healthChecks: 312, responseTime: 41, errors: 1 },
    { timestamp: '10:20', registrations: 71, discoveries: 159, healthChecks: 295, responseTime: 44, errors: 2 },
    { timestamp: '10:25', registrations: 78, divisions: 175, healthChecks: 321, responseTime: 39, errors: 0 }
  ])

  const [loadBalancerStats, setLoadBalancerStats] = useState<LoadBalancerStats>({
    algorithm: 'Capability-Based',
    requestsPerSecond: 1247,
    avgLatency: 32,
    successRate: 99.8,
    activeNodes: 45
  })

  const [clusterNodes, setClusterNodes] = useState<ClusterNode[]>([
    { id: 'node-1', name: 'etcd-master-1', status: 'healthy', ip: '192.168.1.101', port: 2379, role: 'leader', uptime: 99.9, load: 45, connections: 234 },
    { id: 'node-2', name: 'etcd-master-2', status: 'healthy', ip: '192.168.1.102', port: 2379, role: 'follower', uptime: 99.8, load: 38, connections: 198 },
    { id: 'node-3', name: 'etcd-master-3', status: 'healthy', ip: '192.168.1.103', port: 2379, role: 'follower', uptime: 99.7, load: 42, connections: 205 },
    { id: 'node-4', name: 'etcd-backup-1', status: 'unhealthy', ip: '192.168.1.104', port: 2379, role: 'learner', uptime: 85.2, load: 0, connections: 0 },
    { id: 'node-5', name: 'etcd-backup-2', status: 'joining', ip: '192.168.1.105', port: 2379, role: 'learner', uptime: 0, load: 0, connections: 0 }
  ])

  const [recentEvents] = useState([
    { time: '10:25', type: 'registration', message: 'Agent ml-processor-7 注册成功', status: 'success' },
    { time: '10:23', type: 'health', message: 'Agent data-analyzer-3 健康检查失败', status: 'warning' },
    { time: '10:21', type: 'discovery', message: '发现15个新的推荐引擎服务', status: 'info' },
    { time: '10:19', type: 'load-balancing', message: '负载均衡策略切换为地理位置优先', status: 'info' },
    { time: '10:17', type: 'cluster', message: 'etcd集群节点 node-5 正在加入', status: 'processing' },
    { time: '10:15', type: 'alert', message: '检测到异常流量模式，自动扩容', status: 'warning' }
  ])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return '#52c41a'
      case 'unhealthy': return '#ff4d4f'
      case 'joining': return '#faad14'
      default: return '#d9d9d9'
    }
  }

  const getRoleColor = (role: string) => {
    switch (role) {
      case 'leader': return 'red'
      case 'follower': return 'blue'
      case 'learner': return 'orange'
      default: return 'default'
    }
  }

  const getEventColor = (status: string) => {
    switch (status) {
      case 'success': return 'green'
      case 'warning': return 'orange'
      case 'error': return 'red'
      case 'processing': return 'blue'
      default: return 'default'
    }
  }

  const pieData = [
    { name: '健康服务', value: agentStats.healthy, color: '#52c41a' },
    { name: '异常服务', value: agentStats.unhealthy, color: '#ff4d4f' },
    { name: '注册中', value: agentStats.registering, color: '#faad14' }
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
          <Tag color={getRoleColor(record.role)}>{record.role.toUpperCase()}</Tag>
        </Space>
      )
    },
    {
      title: '地址',
      key: 'address',
      render: (_, record: ClusterNode) => `${record.ip}:${record.port}`
    },
    {
      title: '运行时间',
      dataIndex: 'uptime',
      key: 'uptime',
      render: (uptime: number) => (
        <Space>
          <Progress percent={uptime} size="small" status={uptime > 95 ? 'success' : uptime > 85 ? 'active' : 'exception'} />
          <Text>{uptime}%</Text>
        </Space>
      )
    },
    {
      title: '系统负载',
      dataIndex: 'load',
      key: 'load',
      render: (load: number) => (
        <Progress percent={load} size="small" status={load < 70 ? 'success' : load < 85 ? 'active' : 'exception'} />
      )
    },
    {
      title: '连接数',
      dataIndex: 'connections',
      key: 'connections',
      render: (connections: number) => (
        <Statistic value={connections} valueStyle={{ fontSize: '14px' }} />
      )
    }
  ]

  useEffect(() => {
    const interval = setInterval(() => {
      setAgentStats(prev => ({
        ...prev,
        activeConnections: prev.activeConnections + Math.floor(Math.random() * 10) - 5,
        avgResponseTime: prev.avgResponseTime + Math.floor(Math.random() * 6) - 3,
        throughputPerSecond: prev.throughputPerSecond + Math.floor(Math.random() * 20) - 10
      }))
    }, 5000)

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
            <Button type="primary" icon={<ReloadOutlined />}>刷新数据</Button>
            <Button icon={<SettingOutlined />}>系统配置</Button>
            <Button icon={<MonitorOutlined />}>监控面板</Button>
          </Space>
        </div>

        {/* 系统状态告警 */}
        <Alert
          message="系统运行正常"
          description="所有关键服务运行正常，etcd集群状态健康，服务发现延迟在正常范围内。"
          type="success"
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
                <Text type="secondary">健康: {agentStats.healthy} | 异常: {agentStats.unhealthy}</Text>
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
                <Text type="secondary">QPS: {agentStats.throughputPerSecond}</Text>
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
                <Text type="secondary">目标: &lt; 50ms</Text>
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
                valueStyle={{ color: agentStats.uptime > 99 ? '#52c41a' : '#faad14' }}
              />
              <div style={{ marginTop: '8px' }}>
                <Text type="secondary">SLA: &gt; 99.5%</Text>
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
                      event.status === 'success' ? <CheckCircleOutlined style={{ color: '#52c41a' }} /> :
                      event.status === 'warning' ? <ExclamationCircleOutlined style={{ color: '#faad14' }} /> :
                      event.status === 'processing' ? <SyncOutlined spin style={{ color: '#1890ff' }} /> :
                      <ClockCircleOutlined style={{ color: '#d9d9d9' }} />
                    }
                  >
                    <div>
                      <Text type="secondary" style={{ fontSize: '12px' }}>{event.time}</Text>
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
                  <Area type="monotone" dataKey="registrations" stackId="1" stroke="#1890ff" fill="#1890ff" fillOpacity={0.3} />
                  <Area type="monotone" dataKey="discoveries" stackId="2" stroke="#52c41a" fill="#52c41a" fillOpacity={0.3} />
                  <Area type="monotone" dataKey="healthChecks" stackId="3" stroke="#faad14" fill="#faad14" fillOpacity={0.3} />
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
                  <Line yAxisId="left" type="monotone" dataKey="responseTime" stroke="#1890ff" strokeWidth={2} name="响应时间 (ms)" />
                  <Line yAxisId="right" type="monotone" dataKey="errors" stroke="#ff4d4f" strokeWidth={2} name="错误数" />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          </Col>
        </Row>

        {/* etcd集群状态 */}
        <Card title="etcd集群状态" extra={<CloudServerOutlined />} style={{ marginBottom: '24px' }}>
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