import React, { useState, useEffect } from 'react'
import { Card, Row, Col, Table, Badge, Button, Space, Typography, Alert, Progress, Statistic, Tag, Modal, Form, Input, Select, Tooltip, Drawer, Timeline } from 'antd'
import { 
  CloudServerOutlined, 
  CheckCircleOutlined, 
  ExclamationCircleOutlined, 
  CloseCircleOutlined, 
  ReloadOutlined, 
  PlusOutlined,
  SettingOutlined,
  MonitorOutlined,
  WarningOutlined,
  InfoCircleOutlined,
  DeleteOutlined,
  EditOutlined,
  EyeOutlined,
  ThunderboltOutlined,
  DatabaseOutlined,
  NetworkOutlined,
  SyncOutlined,
  CrownOutlined,
  UserOutlined,
  BookOutlined
} from '@ant-design/icons'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as ChartTooltip, ResponsiveContainer, AreaChart, Area } from 'recharts'

const { Title, Paragraph, Text } = Typography
const { Option } = Select

interface ServiceClusterManagementPageProps {}

interface ClusterNode {
  id: string
  name: string
  ip: string
  port: number
  status: 'healthy' | 'unhealthy' | 'joining' | 'leaving' | 'leader' | 'follower' | 'learner'
  role: 'leader' | 'follower' | 'learner'
  region: string
  datacenter: string
  uptime: number
  loadAverage: number
  memoryUsage: number
  diskUsage: number
  networkIO: number
  connections: number
  lastSeen: string
  version: string
  raftState: {
    term: number
    index: number
    applied: number
    commit: number
  }
  metrics: {
    requestsPerSecond: number
    responseTime: number
    errorRate: number
    throughput: number
  }
  config: {
    maxConnections: number
    timeoutMs: number
    heartbeatInterval: number
    electionTimeout: number
  }
}

interface ClusterEvent {
  id: string
  type: 'node_joined' | 'node_left' | 'leader_elected' | 'member_failed' | 'config_changed' | 'split_brain'
  nodeId: string
  nodeName: string
  message: string
  timestamp: string
  severity: 'info' | 'warning' | 'error' | 'success'
  details?: any
}

const ServiceClusterManagementPage: React.FC<ServiceClusterManagementPageProps> = () => {
  const [clusterNodes, setClusterNodes] = useState<ClusterNode[]>([
    {
      id: 'etcd-node-1',
      name: 'etcd-master-1',
      ip: '192.168.1.101',
      port: 2379,
      status: 'leader',
      role: 'leader',
      region: 'us-east-1',
      datacenter: 'dc-1',
      uptime: 99.9,
      loadAverage: 0.45,
      memoryUsage: 78,
      diskUsage: 34,
      networkIO: 125,
      connections: 234,
      lastSeen: '2024-08-26T14:25:00Z',
      version: '3.5.10',
      raftState: {
        term: 15,
        index: 125678,
        applied: 125676,
        commit: 125676
      },
      metrics: {
        requestsPerSecond: 892,
        responseTime: 12,
        errorRate: 0.1,
        throughput: 45.6
      },
      config: {
        maxConnections: 1000,
        timeoutMs: 5000,
        heartbeatInterval: 100,
        electionTimeout: 1000
      }
    },
    {
      id: 'etcd-node-2',
      name: 'etcd-master-2',
      ip: '192.168.1.102',
      port: 2379,
      status: 'follower',
      role: 'follower',
      region: 'us-east-1',
      datacenter: 'dc-1',
      uptime: 99.8,
      loadAverage: 0.38,
      memoryUsage: 65,
      diskUsage: 28,
      networkIO: 98,
      connections: 198,
      lastSeen: '2024-08-26T14:24:30Z',
      version: '3.5.10',
      raftState: {
        term: 15,
        index: 125677,
        applied: 125675,
        commit: 125675
      },
      metrics: {
        requestsPerSecond: 654,
        responseTime: 15,
        errorRate: 0.2,
        throughput: 32.1
      },
      config: {
        maxConnections: 1000,
        timeoutMs: 5000,
        heartbeatInterval: 100,
        electionTimeout: 1000
      }
    },
    {
      id: 'etcd-node-3',
      name: 'etcd-master-3',
      ip: '192.168.1.103',
      port: 2379,
      status: 'follower',
      role: 'follower',
      region: 'us-west-2',
      datacenter: 'dc-2',
      uptime: 99.7,
      loadAverage: 0.42,
      memoryUsage: 72,
      diskUsage: 31,
      networkIO: 112,
      connections: 205,
      lastSeen: '2024-08-26T14:24:45Z',
      version: '3.5.10',
      raftState: {
        term: 15,
        index: 125676,
        applied: 125674,
        commit: 125674
      },
      metrics: {
        requestsPerSecond: 723,
        responseTime: 18,
        errorRate: 0.3,
        throughput: 38.9
      },
      config: {
        maxConnections: 1000,
        timeoutMs: 5000,
        heartbeatInterval: 100,
        electionTimeout: 1000
      }
    },
    {
      id: 'etcd-node-4',
      name: 'etcd-backup-1',
      ip: '192.168.1.104',
      port: 2379,
      status: 'unhealthy',
      role: 'learner',
      region: 'eu-central-1',
      datacenter: 'dc-3',
      uptime: 85.2,
      loadAverage: 0.95,
      memoryUsage: 95,
      diskUsage: 67,
      networkIO: 45,
      connections: 0,
      lastSeen: '2024-08-26T13:45:00Z',
      version: '3.5.9',
      raftState: {
        term: 12,
        index: 98756,
        applied: 98750,
        commit: 98750
      },
      metrics: {
        requestsPerSecond: 0,
        responseTime: 0,
        errorRate: 100,
        throughput: 0
      },
      config: {
        maxConnections: 500,
        timeoutMs: 10000,
        heartbeatInterval: 200,
        electionTimeout: 2000
      }
    },
    {
      id: 'etcd-node-5',
      name: 'etcd-backup-2',
      ip: '192.168.1.105',
      port: 2379,
      status: 'joining',
      role: 'learner',
      region: 'ap-southeast-1',
      datacenter: 'dc-4',
      uptime: 0,
      loadAverage: 0.12,
      memoryUsage: 15,
      diskUsage: 8,
      networkIO: 78,
      connections: 0,
      lastSeen: '2024-08-26T14:20:00Z',
      version: '3.5.10',
      raftState: {
        term: 0,
        index: 0,
        applied: 0,
        commit: 0
      },
      metrics: {
        requestsPerSecond: 0,
        responseTime: 0,
        errorRate: 0,
        throughput: 0
      },
      config: {
        maxConnections: 500,
        timeoutMs: 5000,
        heartbeatInterval: 100,
        electionTimeout: 1000
      }
    }
  ])

  const [clusterEvents] = useState<ClusterEvent[]>([
    {
      id: 'evt-001',
      type: 'node_joined',
      nodeId: 'etcd-node-5',
      nodeName: 'etcd-backup-2',
      message: '新节点正在加入集群，正在同步数据',
      timestamp: '2024-08-26T14:20:00Z',
      severity: 'info'
    },
    {
      id: 'evt-002',
      type: 'member_failed',
      nodeId: 'etcd-node-4',
      nodeName: 'etcd-backup-1',
      message: '节点健康检查失败，连接超时',
      timestamp: '2024-08-26T13:45:00Z',
      severity: 'error'
    },
    {
      id: 'evt-003',
      type: 'leader_elected',
      nodeId: 'etcd-node-1',
      nodeName: 'etcd-master-1',
      message: '领导者选举完成，新领导者已确立',
      timestamp: '2024-08-26T12:30:00Z',
      severity: 'success'
    },
    {
      id: 'evt-004',
      type: 'config_changed',
      nodeId: 'etcd-cluster',
      nodeName: '集群配置',
      message: 'etcd集群配置已更新，心跳间隔调整为100ms',
      timestamp: '2024-08-26T11:15:00Z',
      severity: 'info'
    }
  ])

  const [performanceData] = useState([
    { time: '14:00', requests: 2340, latency: 15, errors: 3 },
    { time: '14:05', requests: 2567, latency: 12, errors: 2 },
    { time: '14:10', requests: 2234, latency: 18, errors: 5 },
    { time: '14:15', requests: 2456, latency: 14, errors: 1 },
    { time: '14:20', requests: 2189, latency: 16, errors: 4 },
    { time: '14:25', requests: 2398, latency: 13, errors: 2 }
  ])

  const [loading, setLoading] = useState(false)
  const [addNodeModalVisible, setAddNodeModalVisible] = useState(false)
  const [nodeDetailDrawerVisible, setNodeDetailDrawerVisible] = useState(false)
  const [selectedNode, setSelectedNode] = useState<ClusterNode | null>(null)

  const [form] = Form.useForm()

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return '#52c41a'
      case 'leader': return '#1890ff'
      case 'follower': return '#52c41a'
      case 'learner': return '#faad14'
      case 'unhealthy': return '#ff4d4f'
      case 'joining': return '#722ed1'
      case 'leaving': return '#fa8c16'
      default: return '#d9d9d9'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy': return <CheckCircleOutlined />
      case 'leader': return <CrownOutlined />
      case 'follower': return <UserOutlined />
      case 'learner': return <BookOutlined />
      case 'unhealthy': return <CloseCircleOutlined />
      case 'joining': return <SyncOutlined spin />
      case 'leaving': return <ExclamationCircleOutlined />
      default: return <InfoCircleOutlined />
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

  const getEventIcon = (type: string, severity: string) => {
    switch (severity) {
      case 'success': return <CheckCircleOutlined style={{ color: '#52c41a' }} />
      case 'warning': return <ExclamationCircleOutlined style={{ color: '#faad14' }} />
      case 'error': return <CloseCircleOutlined style={{ color: '#ff4d4f' }} />
      default: return <InfoCircleOutlined style={{ color: '#1890ff' }} />
    }
  }

  const clusterStats = {
    totalNodes: clusterNodes.length,
    healthyNodes: clusterNodes.filter(n => n.status === 'healthy' || n.status === 'leader' || n.status === 'follower').length,
    unhealthyNodes: clusterNodes.filter(n => n.status === 'unhealthy').length,
    leaderNode: clusterNodes.find(n => n.status === 'leader'),
    avgUptime: clusterNodes.reduce((sum, n) => sum + n.uptime, 0) / clusterNodes.length || 0,
    totalConnections: clusterNodes.reduce((sum, n) => sum + n.connections, 0),
    totalRequests: clusterNodes.reduce((sum, n) => sum + n.metrics.requestsPerSecond, 0)
  }

  const handleRefresh = async () => {
    setLoading(true)
    await new Promise(resolve => setTimeout(resolve, 1000))
    // 模拟数据更新
    setClusterNodes(prev => prev.map(node => ({
      ...node,
      lastSeen: new Date().toISOString(),
      loadAverage: node.loadAverage + (Math.random() - 0.5) * 0.1,
      connections: node.connections + Math.floor((Math.random() - 0.5) * 10)
    })))
    setLoading(false)
  }

  const handleAddNode = async (values: any) => {
    try {
      setLoading(true)
      await new Promise(resolve => setTimeout(resolve, 1500))
      
      const newNode: ClusterNode = {
        id: `etcd-node-${Date.now()}`,
        name: values.name,
        ip: values.ip,
        port: values.port || 2379,
        status: 'joining',
        role: 'learner',
        region: values.region || 'default',
        datacenter: values.datacenter || 'dc-default',
        uptime: 0,
        loadAverage: 0,
        memoryUsage: 0,
        diskUsage: 0,
        networkIO: 0,
        connections: 0,
        lastSeen: new Date().toISOString(),
        version: '3.5.10',
        raftState: { term: 0, index: 0, applied: 0, commit: 0 },
        metrics: { requestsPerSecond: 0, responseTime: 0, errorRate: 0, throughput: 0 },
        config: {
          maxConnections: 500,
          timeoutMs: 5000,
          heartbeatInterval: 100,
          electionTimeout: 1000
        }
      }
      
      setClusterNodes(prev => [...prev, newNode])
      setAddNodeModalVisible(false)
      form.resetFields()
    } catch (error) {
      console.error('添加节点失败:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleRemoveNode = (nodeId: string) => {
    Modal.confirm({
      title: '确认移除节点',
      content: '确定要从集群中移除这个节点吗？此操作可能影响集群稳定性。',
      onOk: () => {
        setClusterNodes(prev => prev.filter(node => node.id !== nodeId))
      }
    })
  }

  const handleViewNodeDetail = (node: ClusterNode) => {
    setSelectedNode(node)
    setNodeDetailDrawerVisible(true)
  }

  const columns = [
    {
      title: '节点信息',
      key: 'node',
      render: (_, node: ClusterNode) => (
        <Space>
          <Badge color={getStatusColor(node.status)} />
          <div>
            <div style={{ display: 'flex', alignItems: 'center' }}>
              {getStatusIcon(node.status)}
              <Text strong style={{ marginLeft: 4 }}>{node.name}</Text>
              <Tag color={getRoleColor(node.role)} style={{ marginLeft: 8 }}>
                {node.role.toUpperCase()}
              </Tag>
            </div>
            <Text type="secondary" style={{ fontSize: '12px' }}>
              {node.ip}:{node.port}
            </Text>
          </div>
        </Space>
      )
    },
    {
      title: '状态',
      key: 'status',
      render: (_, node: ClusterNode) => (
        <div>
          <Tag color={getStatusColor(node.status)}>
            {node.status.toUpperCase()}
          </Tag>
          <div>
            <Progress
              percent={node.uptime}
              size="small"
              status={node.uptime > 99 ? 'success' : node.uptime > 95 ? 'active' : 'exception'}
              showInfo={false}
            />
            <Text type="secondary" style={{ fontSize: '12px' }}>
              可用性: {node.uptime.toFixed(1)}%
            </Text>
          </div>
        </div>
      )
    },
    {
      title: '资源使用',
      key: 'resources',
      render: (_, node: ClusterNode) => (
        <div style={{ minWidth: '120px' }}>
          <div style={{ marginBottom: '4px' }}>
            <Text type="secondary" style={{ fontSize: '12px' }}>负载:</Text>
            <Text style={{ marginLeft: 4, fontSize: '12px' }}>{node.loadAverage.toFixed(2)}</Text>
          </div>
          <div style={{ marginBottom: '4px' }}>
            <Text type="secondary" style={{ fontSize: '12px' }}>内存:</Text>
            <Text style={{ marginLeft: 4, fontSize: '12px' }}>{node.memoryUsage}%</Text>
          </div>
          <div>
            <Text type="secondary" style={{ fontSize: '12px' }}>磁盘:</Text>
            <Text style={{ marginLeft: 4, fontSize: '12px' }}>{node.diskUsage}%</Text>
          </div>
        </div>
      )
    },
    {
      title: 'Raft状态',
      key: 'raft',
      render: (_, node: ClusterNode) => (
        <div style={{ minWidth: '100px' }}>
          <div>
            <Text type="secondary" style={{ fontSize: '12px' }}>任期:</Text>
            <Text style={{ marginLeft: 4, fontSize: '12px' }}>{node.raftState.term}</Text>
          </div>
          <div>
            <Text type="secondary" style={{ fontSize: '12px' }}>索引:</Text>
            <Text style={{ marginLeft: 4, fontSize: '12px' }}>{node.raftState.index}</Text>
          </div>
          <div>
            <Text type="secondary" style={{ fontSize: '12px' }}>提交:</Text>
            <Text style={{ marginLeft: 4, fontSize: '12px' }}>{node.raftState.commit}</Text>
          </div>
        </div>
      )
    },
    {
      title: '连接数',
      dataIndex: 'connections',
      key: 'connections',
      render: (connections: number) => (
        <Statistic value={connections} valueStyle={{ fontSize: '14px' }} />
      )
    },
    {
      title: '区域',
      key: 'location',
      render: (_, node: ClusterNode) => (
        <div>
          <Tag color="blue">{node.region}</Tag>
          <br />
          <Text type="secondary" style={{ fontSize: '12px' }}>{node.datacenter}</Text>
        </div>
      )
    },
    {
      title: '最后活跃',
      key: 'lastSeen',
      render: (_, node: ClusterNode) => (
        <Text type="secondary" style={{ fontSize: '12px' }}>
          {new Date(node.lastSeen).toLocaleString()}
        </Text>
      )
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, node: ClusterNode) => (
        <Space size="small">
          <Tooltip title="查看详情">
            <Button size="small" icon={<EyeOutlined />} onClick={() => handleViewNodeDetail(node)} />
          </Tooltip>
          <Tooltip title="编辑配置">
            <Button size="small" icon={<EditOutlined />} />
          </Tooltip>
          {node.status !== 'leader' && (
            <Tooltip title="移除节点">
              <Button size="small" danger icon={<DeleteOutlined />} onClick={() => handleRemoveNode(node.id)} />
            </Tooltip>
          )}
        </Space>
      )
    }
  ]

  return (
    <div style={{ padding: '24px', background: '#f5f5f5', minHeight: '100vh' }}>
      <div style={{ maxWidth: '1400px', margin: '0 auto' }}>
        {/* 页面标题 */}
        <div style={{ marginBottom: '24px' }}>
          <Title level={2}>
            <CloudServerOutlined /> etcd集群管理
          </Title>
          <Paragraph>
            管理和监控etcd分布式集群，包括节点状态、Raft一致性、资源使用和集群拓扑。
          </Paragraph>
        </div>

        {/* 集群状态统计 */}
        <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
          <Col xs={24} sm={6}>
            <Card>
              <Statistic
                title="集群节点总数"
                value={clusterStats.totalNodes}
                prefix={<CloudServerOutlined />}
                valueStyle={{ color: '#1890ff' }}
              />
            </Card>
          </Col>
          <Col xs={24} sm={6}>
            <Card>
              <Statistic
                title="健康节点"
                value={clusterStats.healthyNodes}
                prefix={<CheckCircleOutlined />}
                valueStyle={{ color: '#52c41a' }}
              />
            </Card>
          </Col>
          <Col xs={24} sm={6}>
            <Card>
              <Statistic
                title="总连接数"
                value={clusterStats.totalConnections}
                prefix={<NetworkOutlined />}
                valueStyle={{ color: '#faad14' }}
              />
            </Card>
          </Col>
          <Col xs={24} sm={6}>
            <Card>
              <Statistic
                title="总QPS"
                value={clusterStats.totalRequests}
                suffix="/s"
                prefix={<ThunderboltOutlined />}
                valueStyle={{ color: '#722ed1' }}
              />
            </Card>
          </Col>
        </Row>

        {/* 集群状态告警 */}
        <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
          <Col xs={24}>
            {clusterStats.unhealthyNodes > 0 && (
              <Alert
                message={`检测到 ${clusterStats.unhealthyNodes} 个节点异常`}
                description="etcd-backup-1 节点连接超时，可能影响集群的高可用性。"
                type="error"
                icon={<ExclamationCircleOutlined />}
                showIcon
                closable
                style={{ marginBottom: '16px' }}
              />
            )}
            {clusterNodes.some(n => n.status === 'joining') && (
              <Alert
                message="有新节点正在加入集群"
                description="etcd-backup-2 正在同步数据，预计需要3-5分钟完成。"
                type="info"
                icon={<InfoCircleOutlined />}
                showIcon
                closable
              />
            )}
          </Col>
        </Row>

        <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
          {/* 集群性能趋势 */}
          <Col xs={24} lg={16}>
            <Card title="集群性能趋势" extra={<MonitorOutlined />}>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={performanceData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis />
                  <ChartTooltip />
                  <Line type="monotone" dataKey="requests" stroke="#1890ff" strokeWidth={2} name="请求数" />
                  <Line type="monotone" dataKey="latency" stroke="#52c41a" strokeWidth={2} name="延迟(ms)" />
                  <Line type="monotone" dataKey="errors" stroke="#ff4d4f" strokeWidth={2} name="错误数" />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          </Col>

          {/* 集群事件 */}
          <Col xs={24} lg={8}>
            <Card title="集群事件" extra={<DatabaseOutlined />}>
              <Timeline size="small">
                {clusterEvents.map((event) => (
                  <Timeline.Item
                    key={event.id}
                    dot={getEventIcon(event.type, event.severity)}
                  >
                    <div>
                      <div style={{ marginBottom: '4px' }}>
                        <Text strong style={{ fontSize: '13px' }}>
                          {event.nodeName}
                        </Text>
                        <Tag size="small" color={
                          event.severity === 'error' ? 'red' : 
                          event.severity === 'warning' ? 'orange' : 
                          event.severity === 'success' ? 'green' : 'blue'
                        }>
                          {event.type.toUpperCase().replace('_', ' ')}
                        </Tag>
                      </div>
                      <div style={{ fontSize: '12px', marginBottom: '2px' }}>
                        {event.message}
                      </div>
                      <Text type="secondary" style={{ fontSize: '11px' }}>
                        {new Date(event.timestamp).toLocaleString()}
                      </Text>
                    </div>
                  </Timeline.Item>
                ))}
              </Timeline>
            </Card>
          </Col>
        </Row>

        {/* 操作栏 */}
        <Card style={{ marginBottom: '16px' }}>
          <Row justify="space-between" align="middle">
            <Col>
              <Space>
                <Button type="primary" icon={<PlusOutlined />} onClick={() => setAddNodeModalVisible(true)}>
                  添加节点
                </Button>
                <Button icon={<ReloadOutlined />} loading={loading} onClick={handleRefresh}>
                  刷新集群
                </Button>
                <Button icon={<SettingOutlined />}>
                  集群配置
                </Button>
              </Space>
            </Col>
            <Col>
              <Space>
                <Badge status="processing" text={`领导者: ${clusterStats.leaderNode?.name || '未知'}`} />
                <Badge status="success" text={`平均可用性: ${clusterStats.avgUptime.toFixed(1)}%`} />
              </Space>
            </Col>
          </Row>
        </Card>

        {/* 节点列表 */}
        <Card title="集群节点">
          <Table
            columns={columns}
            dataSource={clusterNodes}
            rowKey="id"
            loading={loading}
            pagination={false}
            size="small"
          />
        </Card>

        {/* 添加节点Modal */}
        <Modal
          title="添加集群节点"
          visible={addNodeModalVisible}
          onOk={form.submit}
          onCancel={() => {
            setAddNodeModalVisible(false)
            form.resetFields()
          }}
          confirmLoading={loading}
        >
          <Form
            form={form}
            layout="vertical"
            onFinish={handleAddNode}
          >
            <Form.Item
              name="name"
              label="节点名称"
              rules={[{ required: true, message: '请输入节点名称' }]}
            >
              <Input placeholder="例如: etcd-backup-3" />
            </Form.Item>
            
            <Row gutter={16}>
              <Col span={16}>
                <Form.Item
                  name="ip"
                  label="IP地址"
                  rules={[
                    { required: true, message: '请输入IP地址' },
                    { pattern: /^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$/, message: '请输入有效的IP地址' }
                  ]}
                >
                  <Input placeholder="192.168.1.106" />
                </Form.Item>
              </Col>
              <Col span={8}>
                <Form.Item name="port" label="端口">
                  <Input placeholder="2379" type="number" />
                </Form.Item>
              </Col>
            </Row>

            <Row gutter={16}>
              <Col span={12}>
                <Form.Item name="region" label="区域">
                  <Select placeholder="选择区域">
                    <Option value="us-east-1">美国东部-1</Option>
                    <Option value="us-west-2">美国西部-2</Option>
                    <Option value="eu-central-1">欧洲中部-1</Option>
                    <Option value="ap-southeast-1">亚太东南-1</Option>
                  </Select>
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item name="datacenter" label="数据中心">
                  <Input placeholder="例如: dc-1" />
                </Form.Item>
              </Col>
            </Row>
          </Form>
        </Modal>

        {/* 节点详情Drawer */}
        <Drawer
          title="节点详细信息"
          visible={nodeDetailDrawerVisible}
          onClose={() => setNodeDetailDrawerVisible(false)}
          width={600}
        >
          {selectedNode && (
            <div>
              <Alert
                message={`节点状态: ${selectedNode.status.toUpperCase()}`}
                type={selectedNode.status === 'leader' || selectedNode.status === 'follower' ? 'success' : 
                      selectedNode.status === 'joining' || selectedNode.status === 'learner' ? 'info' : 'error'}
                showIcon
                style={{ marginBottom: '16px' }}
              />

              <Title level={4}>基本信息</Title>
              <Row gutter={16} style={{ marginBottom: '16px' }}>
                <Col span={8}>
                  <Text type="secondary">节点名称</Text>
                  <div>{selectedNode.name}</div>
                </Col>
                <Col span={8}>
                  <Text type="secondary">IP地址</Text>
                  <div>{selectedNode.ip}</div>
                </Col>
                <Col span={8}>
                  <Text type="secondary">端口</Text>
                  <div>{selectedNode.port}</div>
                </Col>
                <Col span={8}>
                  <Text type="secondary">角色</Text>
                  <div><Tag color={getRoleColor(selectedNode.role)}>{selectedNode.role.toUpperCase()}</Tag></div>
                </Col>
                <Col span={8}>
                  <Text type="secondary">版本</Text>
                  <div>{selectedNode.version}</div>
                </Col>
                <Col span={8}>
                  <Text type="secondary">可用性</Text>
                  <div>{selectedNode.uptime.toFixed(2)}%</div>
                </Col>
              </Row>

              <Title level={4}>系统资源</Title>
              <Row gutter={16} style={{ marginBottom: '16px' }}>
                <Col span={12}>
                  <Card size="small">
                    <Text type="secondary">负载平均值</Text>
                    <div style={{ fontSize: '18px', fontWeight: 'bold' }}>
                      {selectedNode.loadAverage.toFixed(2)}
                    </div>
                  </Card>
                </Col>
                <Col span={12}>
                  <Card size="small">
                    <Text type="secondary">网络IO</Text>
                    <div style={{ fontSize: '18px', fontWeight: 'bold' }}>
                      {selectedNode.networkIO} MB/s
                    </div>
                  </Card>
                </Col>
                <Col span={12}>
                  <Card size="small">
                    <Text type="secondary">内存使用</Text>
                    <Progress percent={selectedNode.memoryUsage} size="small" />
                    <div>{selectedNode.memoryUsage}%</div>
                  </Card>
                </Col>
                <Col span={12}>
                  <Card size="small">
                    <Text type="secondary">磁盘使用</Text>
                    <Progress percent={selectedNode.diskUsage} size="small" />
                    <div>{selectedNode.diskUsage}%</div>
                  </Card>
                </Col>
              </Row>

              <Title level={4}>Raft一致性状态</Title>
              <Row gutter={16} style={{ marginBottom: '16px' }}>
                <Col span={12}>
                  <Text type="secondary">当前任期: </Text>
                  <Text>{selectedNode.raftState.term}</Text>
                </Col>
                <Col span={12}>
                  <Text type="secondary">日志索引: </Text>
                  <Text>{selectedNode.raftState.index}</Text>
                </Col>
                <Col span={12}>
                  <Text type="secondary">已应用: </Text>
                  <Text>{selectedNode.raftState.applied}</Text>
                </Col>
                <Col span={12}>
                  <Text type="secondary">已提交: </Text>
                  <Text>{selectedNode.raftState.commit}</Text>
                </Col>
              </Row>

              <Title level={4}>性能指标</Title>
              <Row gutter={16}>
                <Col span={12}>
                  <Card size="small">
                    <Statistic title="QPS" value={selectedNode.metrics.requestsPerSecond} />
                  </Card>
                </Col>
                <Col span={12}>
                  <Card size="small">
                    <Statistic title="响应时间" value={selectedNode.metrics.responseTime} suffix="ms" />
                  </Card>
                </Col>
                <Col span={12}>
                  <Card size="small">
                    <Statistic title="错误率" value={selectedNode.metrics.errorRate} suffix="%" precision={1} />
                  </Card>
                </Col>
                <Col span={12}>
                  <Card size="small">
                    <Statistic title="吞吐量" value={selectedNode.metrics.throughput} suffix="MB/s" precision={1} />
                  </Card>
                </Col>
              </Row>
            </div>
          )}
        </Drawer>
      </div>
    </div>
  )
}

export default ServiceClusterManagementPage