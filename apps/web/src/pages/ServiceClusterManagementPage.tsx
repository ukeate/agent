import React, { useState, useEffect } from 'react'
import { Card, Row, Col, Table, Badge, Button, Space, Typography, Alert, Progress, Statistic, Tag, Modal, Form, Input, Select, Tooltip, Drawer, Timeline } from 'antd'
import { 
import { logger } from '../utils/logger'
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
  ShareAltOutlined,
  SyncOutlined,
  CrownOutlined,
  UserOutlined,
  BookOutlined
} from '@ant-design/icons'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as ChartTooltip, ResponsiveContainer, AreaChart, Area } from 'recharts'
import { clusterManagementService } from '../services/clusterManagementService'

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
  const [clusterNodes, setClusterNodes] = useState<ClusterNode[]>([])
  const [clusterEvents, setClusterEvents] = useState<ClusterEvent[]>([])
  const [performanceData, setPerformanceData] = useState<any[]>([])

  const [loading, setLoading] = useState(false)
  const [addNodeModalVisible, setAddNodeModalVisible] = useState(false)
  const [nodeDetailDrawerVisible, setNodeDetailDrawerVisible] = useState(false)
  const [selectedNode, setSelectedNode] = useState<ClusterNode | null>(null)

  const [form] = Form.useForm()

  const mapAgentToNode = (agent: any): ClusterNode => {
    const usage = agent.resource_usage || {}
    return {
      id: agent.agent_id || agent.node_id || agent.id,
      name: agent.name || agent.agent_id || 'unknown',
      ip: agent.host || agent.endpoint || '',
      port: agent.port || 0,
      status: (agent.status || 'healthy') as ClusterNode['status'],
      role: (agent.role || 'follower') as ClusterNode['role'],
      region: agent.region || agent.labels?.region || 'unknown',
      datacenter: agent.datacenter || agent.labels?.dc || 'unknown',
      uptime: Math.min(100, agent.uptime || 0),
      loadAverage: agent.current_load || usage.cpu_usage || 0,
      memoryUsage: usage.memory_usage || 0,
      diskUsage: usage.disk_usage || 0,
      networkIO: usage.network_io || 0,
      connections: usage.active_tasks || 0,
      lastSeen: agent.last_heartbeat || new Date().toISOString(),
      version: agent.version || '',
      raftState: {
        term: agent.raft_term || 0,
        index: agent.raft_index || 0,
        applied: agent.raft_applied || 0,
        commit: agent.raft_commit || 0
      },
      metrics: {
        requestsPerSecond: agent.qps || usage.requests_per_second || 0,
        responseTime: usage.response_time || 0,
        errorRate: usage.error_rate || 0,
        throughput: usage.throughput || 0
      },
      config: {
        maxConnections: agent.max_capacity || 0,
        timeoutMs: 0,
        heartbeatInterval: 0,
        electionTimeout: 0
      }
    }
  }

  const loadData = async () => {
    setLoading(true)
    try {
      const [agents, stats, metrics, health] = await Promise.all([
        clusterManagementService.getAgents(),
        clusterManagementService.getClusterStats().catch(() => null),
        clusterManagementService.getMetrics(undefined, 1800).catch(() => []),
        clusterManagementService.getClusterHealth().catch(() => null)
      ])
      setClusterNodes((agents || []).map(mapAgentToNode))

      const perf = (metrics || []).map((m: any) => ({
        time: new Date(m.timestamp).toLocaleTimeString(),
        requests: m.request_rate || 0,
        latency: m.response_time || 0,
        errors: m.error_rate || 0
      }))
      setPerformanceData(perf)

      if (health?.issues?.length) {
        setClusterEvents(
          health.issues.map((issue: string, idx: number) => ({
            id: `health-${idx}`,
            type: 'member_failed',
            nodeId: '',
            nodeName: 'cluster',
            message: issue,
            timestamp: new Date().toISOString(),
            severity: 'warning'
          }))
        )
      } else {
        setClusterEvents([])
      }

      if (stats) {
        // stats currently not directly bound; already computed via clusterNodes
      }
    } catch (error) {
      logger.error('加载集群数据失败', error)
      setClusterEvents([])
      setPerformanceData([])
      setClusterNodes([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
  }, [])

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
    avgUptime: clusterNodes.length ? clusterNodes.reduce((sum, n) => sum + n.uptime, 0) / clusterNodes.length : 0,
    totalConnections: clusterNodes.reduce((sum, n) => sum + n.connections, 0),
    totalRequests: clusterNodes.reduce((sum, n) => sum + n.metrics.requestsPerSecond, 0)
  }

  const handleRefresh = async () => {
    await loadData()
  }

  const handleAddNode = async (values: any) => {
    try {
      setLoading(true)
      await clusterManagementService.createAgent({
        name: values.name,
        host: values.ip,
        port: values.port || 2379,
        labels: {
          region: values.region || '',
          datacenter: values.datacenter || ''
        }
      })
      await loadData()
      setAddNodeModalVisible(false)
      form.resetFields()
    } catch (error) {
      logger.error('添加节点失败:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleRemoveNode = (nodeId: string) => {
    Modal.confirm({
      title: '确认移除节点',
      content: '确定要从集群中移除这个节点吗？此操作可能影响集群稳定性。',
      onOk: async () => {
        await clusterManagementService.deleteAgent(nodeId)
        await loadData()
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
                prefix={<ShareAltOutlined />}
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
                description="请检查异常节点健康状况，恢复后可自动重新加入。"
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
                description="节点正在同步数据，请稍后查看状态。"
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
              {clusterEvents.length === 0 ? (
                <Text type="secondary">暂无事件</Text>
              ) : (
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
              )}
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
