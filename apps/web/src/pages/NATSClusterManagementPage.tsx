import React, { useState, useEffect } from 'react'
import { 
  Card, 
  Row, 
  Col, 
  Table, 
  Button, 
  Badge, 
  Progress, 
  Statistic, 
  Modal, 
  Form, 
  Input, 
  Select, 
  Switch, 
  Space, 
  Typography, 
  Alert, 
  Tag, 
  Divider,
  Tabs,
  Timeline,
  Tooltip,
  notification
} from 'antd'
import {
  ClusterOutlined,
  ServerOutlined,
  PlusOutlined,
  DeleteOutlined,
  ReloadOutlined,
  SettingOutlined,
  PlayCircleOutlined,
  StopOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  MonitorOutlined,
  ShareAltOutlined as NetworkOutlined,
  ThunderboltOutlined,
  SafetyCertificateOutlined,
  ExclamationCircleOutlined,
  EyeOutlined,
  EditOutlined,
  CloudServerOutlined
} from '@ant-design/icons'

const { Title, Text, Paragraph } = Typography
const { Option } = Select
const { TabPane } = Tabs

interface ClusterNode {
  id: string
  name: string
  host: string
  port: number
  clusterId: string
  version: string
  status: 'online' | 'offline' | 'starting' | 'stopping' | 'error'
  role: 'primary' | 'replica'
  connections: number
  maxConnections: number
  messages: number
  bytes: number
  cpu: number
  memory: number
  uptime: string
  lastSeen: string
  routes: number
  subscriptions: number
  slowConsumers: number
}

interface StreamConfig {
  name: string
  subjects: string[]
  retention: 'limits' | 'interest' | 'workqueue'
  maxMsgs: number
  maxBytes: number
  maxAge: string
  replicas: number
  storageType: 'file' | 'memory'
  compression: 'none' | 's2'
  status: 'active' | 'inactive' | 'error'
  messages: number
  consumers: number
}

interface ConsumerInfo {
  stream: string
  name: string
  config: {
    durable: boolean
    deliverSubject?: string
    ackPolicy: 'none' | 'all' | 'explicit'
    replayPolicy: 'instant' | 'original'
    maxDeliver: number
    maxWaiting: number
  }
  delivered: number
  acknowledged: number
  pending: number
  redelivered: number
  numWaiting: number
  lastActive: string
}

const NATSClusterManagementPage: React.FC = () => {
  const [nodes, setNodes] = useState<ClusterNode[]>([
    {
      id: 'nats-node-1',
      name: 'NATS主节点',
      host: '172.20.0.10',
      port: 4222,
      clusterId: 'nats-cluster-prod',
      version: '2.10.7',
      status: 'online',
      role: 'primary',
      connections: 245,
      maxConnections: 1000,
      messages: 1247583,
      bytes: 524288000,
      cpu: 45.2,
      memory: 62.8,
      uptime: '15天 8小时 42分钟',
      lastSeen: '2025-08-26 12:45:30',
      routes: 2,
      subscriptions: 156,
      slowConsumers: 0
    },
    {
      id: 'nats-node-2',
      name: 'NATS从节点1',
      host: '172.20.0.11',
      port: 4222,
      clusterId: 'nats-cluster-prod',
      version: '2.10.7',
      status: 'online',
      role: 'replica',
      connections: 198,
      maxConnections: 1000,
      messages: 1195847,
      bytes: 498765432,
      cpu: 38.7,
      memory: 55.1,
      uptime: '15天 8小时 42分钟',
      lastSeen: '2025-08-26 12:45:29',
      routes: 2,
      subscriptions: 134,
      slowConsumers: 1
    },
    {
      id: 'nats-node-3',
      name: 'NATS从节点2',
      host: '172.20.0.12',
      port: 4222,
      clusterId: 'nats-cluster-prod',
      version: '2.10.7',
      status: 'online',
      role: 'replica',
      connections: 201,
      maxConnections: 1000,
      messages: 1201935,
      bytes: 501234567,
      cpu: 52.3,
      memory: 68.4,
      uptime: '15天 8小时 42分钟',
      lastSeen: '2025-08-26 12:45:31',
      routes: 2,
      subscriptions: 142,
      slowConsumers: 0
    }
  ])

  const [streams, setStreams] = useState<StreamConfig[]>([
    {
      name: 'AGENTS_TASKS',
      subjects: ['agents.tasks.>'],
      retention: 'workqueue',
      maxMsgs: 1000000,
      maxBytes: 1073741824,
      maxAge: '7d',
      replicas: 3,
      storageType: 'file',
      compression: 's2',
      status: 'active',
      messages: 45632,
      consumers: 5
    },
    {
      name: 'AGENTS_EVENTS',
      subjects: ['agents.events.>', 'system.events.>'],
      retention: 'limits',
      maxMsgs: 500000,
      maxBytes: 536870912,
      maxAge: '30d',
      replicas: 3,
      storageType: 'file',
      compression: 's2',
      status: 'active',
      messages: 128945,
      consumers: 8
    },
    {
      name: 'AGENTS_DIRECT',
      subjects: ['agents.direct.>'],
      retention: 'interest',
      maxMsgs: 100000,
      maxBytes: 104857600,
      maxAge: '1h',
      replicas: 2,
      storageType: 'memory',
      compression: 'none',
      status: 'active',
      messages: 8467,
      consumers: 12
    }
  ])

  const [consumers, setConsumers] = useState<ConsumerInfo[]>([
    {
      stream: 'AGENTS_TASKS',
      name: 'task-processor-01',
      config: {
        durable: true,
        deliverSubject: 'agents.tasks.process.01',
        ackPolicy: 'explicit',
        replayPolicy: 'instant',
        maxDeliver: 3,
        maxWaiting: 100
      },
      delivered: 8467,
      acknowledged: 8445,
      pending: 22,
      redelivered: 12,
      numWaiting: 5,
      lastActive: '2025-08-26 12:44:55'
    },
    {
      stream: 'AGENTS_EVENTS',
      name: 'event-logger',
      config: {
        durable: true,
        ackPolicy: 'none',
        replayPolicy: 'instant',
        maxDeliver: 1,
        maxWaiting: 0
      },
      delivered: 24589,
      acknowledged: 24589,
      pending: 0,
      redelivered: 0,
      numWaiting: 0,
      lastActive: '2025-08-26 12:45:12'
    }
  ])

  const [loading, setLoading] = useState(false)
  const [selectedNode, setSelectedNode] = useState<ClusterNode | null>(null)
  const [nodeModalVisible, setNodeModalVisible] = useState(false)
  const [streamModalVisible, setStreamModalVisible] = useState(false)
  const [consumerModalVisible, setConsumerModalVisible] = useState(false)
  const [activeTab, setActiveTab] = useState('nodes')

  const nodeColumns = [
    {
      title: '节点信息',
      key: 'info',
      render: (record: ClusterNode) => (
        <div>
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '4px' }}>
            <Badge 
              status={
                record.status === 'online' ? 'success' : 
                record.status === 'offline' ? 'default' : 
                record.status === 'error' ? 'error' : 'processing'
              }
            />
            <Text strong style={{ marginLeft: '8px' }}>{record.name}</Text>
            <Tag color={record.role === 'primary' ? 'gold' : 'blue'} style={{ marginLeft: '8px' }}>
              {record.role === 'primary' ? '主节点' : '从节点'}
            </Tag>
          </div>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            {record.host}:{record.port} | v{record.version}
          </Text>
        </div>
      )
    },
    {
      title: '连接状态',
      key: 'connections',
      render: (record: ClusterNode) => (
        <div>
          <div style={{ marginBottom: '4px' }}>
            <Text>{record.connections}/{record.maxConnections}</Text>
          </div>
          <Progress 
            percent={Math.round((record.connections / record.maxConnections) * 100)} 
            size="small"
            status={record.connections > record.maxConnections * 0.8 ? 'exception' : 'success'}
            showInfo={false}
          />
        </div>
      )
    },
    {
      title: '资源使用',
      key: 'resources',
      render: (record: ClusterNode) => (
        <div>
          <div style={{ marginBottom: '4px' }}>
            <Text style={{ fontSize: '12px' }}>CPU: {record.cpu.toFixed(1)}%</Text>
            <Progress 
              percent={record.cpu} 
              size="small" 
              status={record.cpu > 80 ? 'exception' : record.cpu > 60 ? 'active' : 'success'}
              showInfo={false}
              style={{ marginBottom: '2px' }}
            />
          </div>
          <div>
            <Text style={{ fontSize: '12px' }}>内存: {record.memory.toFixed(1)}%</Text>
            <Progress 
              percent={record.memory} 
              size="small"
              status={record.memory > 85 ? 'exception' : record.memory > 70 ? 'active' : 'success'}
              showInfo={false}
            />
          </div>
        </div>
      )
    },
    {
      title: '消息统计',
      key: 'messages',
      render: (record: ClusterNode) => (
        <div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '12px' }}>消息: {record.messages.toLocaleString()}</Text>
          </div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '12px' }}>订阅: {record.subscriptions}</Text>
          </div>
          <div>
            <Text style={{ fontSize: '12px' }}>路由: {record.routes}</Text>
          </div>
        </div>
      )
    },
    {
      title: '运行时间',
      key: 'uptime',
      render: (record: ClusterNode) => (
        <div>
          <Text style={{ fontSize: '12px' }}>{record.uptime}</Text>
          <br />
          <Text type="secondary" style={{ fontSize: '11px' }}>
            最后心跳: {record.lastSeen}
          </Text>
        </div>
      )
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: ClusterNode) => (
        <Space>
          <Tooltip title="查看详情">
            <Button 
              type="text" 
              size="small" 
              icon={<EyeOutlined />}
              onClick={() => {
                setSelectedNode(record)
                setNodeModalVisible(true)
              }}
            />
          </Tooltip>
          <Tooltip title="配置节点">
            <Button 
              type="text" 
              size="small" 
              icon={<SettingOutlined />}
              onClick={() => handleConfigureNode(record)}
            />
          </Tooltip>
          {record.status === 'online' ? (
            <Tooltip title="停止节点">
              <Button 
                type="text" 
                size="small" 
                icon={<StopOutlined />}
                danger
                onClick={() => handleStopNode(record)}
              />
            </Tooltip>
          ) : (
            <Tooltip title="启动节点">
              <Button 
                type="text" 
                size="small" 
                icon={<PlayCircleOutlined />}
                onClick={() => handleStartNode(record)}
              />
            </Tooltip>
          )}
        </Space>
      )
    }
  ]

  const streamColumns = [
    {
      title: 'Stream名称',
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record: StreamConfig) => (
        <div>
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <Badge status={record.status === 'active' ? 'success' : 'error'} />
            <Text strong style={{ marginLeft: '8px' }}>{name}</Text>
          </div>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            主题: {record.subjects.join(', ')}
          </Text>
        </div>
      )
    },
    {
      title: '保留策略',
      dataIndex: 'retention',
      key: 'retention',
      render: (retention: string) => (
        <Tag color={
          retention === 'limits' ? 'blue' : 
          retention === 'interest' ? 'green' : 'orange'
        }>
          {retention === 'limits' ? '限制' : 
           retention === 'interest' ? '兴趣' : '工作队列'}
        </Tag>
      )
    },
    {
      title: '存储配置',
      key: 'storage',
      render: (record: StreamConfig) => (
        <div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '12px' }}>
              类型: {record.storageType === 'file' ? '文件' : '内存'}
            </Text>
          </div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '12px' }}>
              副本: {record.replicas}
            </Text>
          </div>
          <div>
            <Text style={{ fontSize: '12px' }}>
              压缩: {record.compression === 'none' ? '无' : record.compression}
            </Text>
          </div>
        </div>
      )
    },
    {
      title: '限制配置',
      key: 'limits',
      render: (record: StreamConfig) => (
        <div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '12px' }}>
              消息: {record.maxMsgs.toLocaleString()}
            </Text>
          </div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '12px' }}>
              字节: {(record.maxBytes / 1024 / 1024).toFixed(0)}MB
            </Text>
          </div>
          <div>
            <Text style={{ fontSize: '12px' }}>
              时间: {record.maxAge}
            </Text>
          </div>
        </div>
      )
    },
    {
      title: '使用统计',
      key: 'usage',
      render: (record: StreamConfig) => (
        <div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '12px' }}>
              消息: {record.messages.toLocaleString()}
            </Text>
          </div>
          <div>
            <Text style={{ fontSize: '12px' }}>
              消费者: {record.consumers}
            </Text>
          </div>
        </div>
      )
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: StreamConfig) => (
        <Space>
          <Button type="text" size="small" icon={<EyeOutlined />} />
          <Button type="text" size="small" icon={<EditOutlined />} />
          <Button type="text" size="small" icon={<DeleteOutlined />} danger />
        </Space>
      )
    }
  ]

  const consumerColumns = [
    {
      title: '消费者信息',
      key: 'info',
      render: (record: ConsumerInfo) => (
        <div>
          <div style={{ marginBottom: '2px' }}>
            <Text strong>{record.name}</Text>
          </div>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            Stream: {record.stream}
          </Text>
        </div>
      )
    },
    {
      title: '配置',
      key: 'config',
      render: (record: ConsumerInfo) => (
        <div>
          <div style={{ marginBottom: '2px' }}>
            <Tag color={record.config.durable ? 'green' : 'orange'}>
              {record.config.durable ? '持久化' : '临时'}
            </Tag>
          </div>
          <Text style={{ fontSize: '12px' }}>
            ACK策略: {record.config.ackPolicy}
          </Text>
        </div>
      )
    },
    {
      title: '消息统计',
      key: 'stats',
      render: (record: ConsumerInfo) => (
        <div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '12px' }}>
              已投递: {record.delivered.toLocaleString()}
            </Text>
          </div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '12px' }}>
              已确认: {record.acknowledged.toLocaleString()}
            </Text>
          </div>
          <div>
            <Text style={{ fontSize: '12px' }}>
              待处理: {record.pending}
            </Text>
          </div>
        </div>
      )
    },
    {
      title: '状态',
      key: 'status',
      render: (record: ConsumerInfo) => (
        <div>
          <div style={{ marginBottom: '2px' }}>
            {record.pending > 0 && (
              <Badge status="processing" text={`${record.pending} 待处理`} />
            )}
            {record.redelivered > 0 && (
              <Badge status="warning" text={`${record.redelivered} 重投递`} />
            )}
            {record.pending === 0 && record.redelivered === 0 && (
              <Badge status="success" text="正常" />
            )}
          </div>
          <Text type="secondary" style={{ fontSize: '11px' }}>
            最后活跃: {record.lastActive}
          </Text>
        </div>
      )
    },
    {
      title: '操作',
      key: 'actions',
      render: () => (
        <Space>
          <Button type="text" size="small" icon={<EyeOutlined />} />
          <Button type="text" size="small" icon={<EditOutlined />} />
          <Button type="text" size="small" icon={<DeleteOutlined />} danger />
        </Space>
      )
    }
  ]

  const handleRefresh = async () => {
    setLoading(true)
    // 模拟API刷新
    setTimeout(() => {
      notification.success({
        message: '刷新成功',
        description: '集群状态已更新'
      })
      setLoading(false)
    }, 1000)
  }

  const handleStartNode = (node: ClusterNode) => {
    Modal.confirm({
      title: '启动节点',
      content: `确定要启动节点 "${node.name}" 吗？`,
      onOk: () => {
        notification.success({
          message: '节点启动成功',
          description: `${node.name} 已成功启动`
        })
      }
    })
  }

  const handleStopNode = (node: ClusterNode) => {
    Modal.confirm({
      title: '停止节点',
      content: `确定要停止节点 "${node.name}" 吗？这可能会影响集群性能。`,
      onOk: () => {
        notification.warning({
          message: '节点已停止',
          description: `${node.name} 已停止运行`
        })
      }
    })
  }

  const handleConfigureNode = (node: ClusterNode) => {
    notification.info({
      message: '配置节点',
      description: `打开 ${node.name} 的配置页面`
    })
  }

  const getClusterHealth = () => {
    const onlineNodes = nodes.filter(n => n.status === 'online').length
    const totalNodes = nodes.length
    const healthPercent = (onlineNodes / totalNodes) * 100
    
    if (healthPercent === 100) return { status: 'healthy', color: 'green', text: '健康' }
    if (healthPercent >= 80) return { status: 'warning', color: 'orange', text: '警告' }
    return { status: 'error', color: 'red', text: '异常' }
  }

  const clusterHealth = getClusterHealth()
  const totalConnections = nodes.reduce((sum, node) => sum + node.connections, 0)
  const totalMessages = nodes.reduce((sum, node) => sum + node.messages, 0)
  const avgCpuUsage = nodes.reduce((sum, node) => sum + node.cpu, 0) / nodes.length

  return (
    <div style={{ padding: '24px', background: '#f5f5f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <ClusterOutlined style={{ marginRight: '12px', color: '#1890ff' }} />
          NATS JetStream集群管理
        </Title>
        <Paragraph>
          管理和监控NATS JetStream集群，包括节点状态、Stream配置、消费者管理等功能
        </Paragraph>
      </div>

      {/* 集群健康状况告警 */}
      {clusterHealth.status !== 'healthy' && (
        <Alert
          message="集群状态警告"
          description={
            clusterHealth.status === 'warning' 
              ? "部分节点离线，建议检查网络连接和节点状态" 
              : "多个节点离线，集群可能无法正常工作，请立即处理"
          }
          type={clusterHealth.status === 'warning' ? 'warning' : 'error'}
          showIcon
          style={{ marginBottom: '24px' }}
        />
      )}

      {/* 集群概览指标 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="集群健康度"
              value={clusterHealth.text}
              valueStyle={{ color: clusterHealth.color }}
              prefix={clusterHealth.status === 'healthy' ? <CheckCircleOutlined /> : <ExclamationCircleOutlined />}
            />
            <div style={{ marginTop: '8px' }}>
              <Text type="secondary">
                在线节点: {nodes.filter(n => n.status === 'online').length}/{nodes.length}
              </Text>
            </div>
          </Card>
        </Col>
        
        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="总连接数"
              value={totalConnections}
              precision={0}
              valueStyle={{ color: '#1890ff' }}
              prefix={<NetworkOutlined />}
            />
            <div style={{ marginTop: '8px' }}>
              <Text type="secondary">
                最大容量: {nodes.reduce((sum, node) => sum + node.maxConnections, 0)}
              </Text>
            </div>
          </Card>
        </Col>
        
        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="消息总量"
              value={totalMessages}
              precision={0}
              valueStyle={{ color: '#52c41a' }}
              prefix={<ThunderboltOutlined />}
            />
            <div style={{ marginTop: '8px' }}>
              <Text type="secondary">
                活跃Stream: {streams.filter(s => s.status === 'active').length}
              </Text>
            </div>
          </Card>
        </Col>
        
        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="平均CPU使用率"
              value={avgCpuUsage.toFixed(1)}
              precision={1}
              suffix="%"
              valueStyle={{ color: avgCpuUsage > 70 ? '#ff4d4f' : '#52c41a' }}
              prefix={<MonitorOutlined />}
            />
            <div style={{ marginTop: '8px' }}>
              <Progress 
                percent={avgCpuUsage} 
                size="small" 
                showInfo={false}
                status={avgCpuUsage > 80 ? 'exception' : 'success'}
              />
            </div>
          </Card>
        </Col>
      </Row>

      {/* 主要管理界面 */}
      <Card>
        <div style={{ marginBottom: '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Space>
            <Button 
              type="primary" 
              icon={<PlusOutlined />}
              onClick={() => setNodeModalVisible(true)}
            >
              添加节点
            </Button>
            <Button 
              icon={<ReloadOutlined />} 
              loading={loading}
              onClick={handleRefresh}
            >
              刷新状态
            </Button>
          </Space>
          
          <Space>
            <Button icon={<SettingOutlined />}>集群配置</Button>
            <Button icon={<SafetyCertificateOutlined />}>安全设置</Button>
          </Space>
        </div>

        <Tabs activeKey={activeTab} onChange={setActiveTab}>
          <TabPane tab="节点管理" key="nodes" icon={<ServerOutlined />}>
            <Table
              columns={nodeColumns}
              dataSource={nodes}
              rowKey="id"
              size="small"
              pagination={{ pageSize: 10 }}
              scroll={{ x: 1200 }}
            />
          </TabPane>
          
          <TabPane tab="Stream管理" key="streams" icon={<CloudServerOutlined />}>
            <div style={{ marginBottom: '16px' }}>
              <Button 
                type="primary" 
                icon={<PlusOutlined />}
                onClick={() => setStreamModalVisible(true)}
              >
                创建Stream
              </Button>
            </div>
            <Table
              columns={streamColumns}
              dataSource={streams}
              rowKey="name"
              size="small"
              pagination={{ pageSize: 10 }}
            />
          </TabPane>
          
          <TabPane tab="消费者管理" key="consumers" icon={<ThunderboltOutlined />}>
            <div style={{ marginBottom: '16px' }}>
              <Button 
                type="primary" 
                icon={<PlusOutlined />}
                onClick={() => setConsumerModalVisible(true)}
              >
                创建消费者
              </Button>
            </div>
            <Table
              columns={consumerColumns}
              dataSource={consumers}
              rowKey={(record) => `${record.stream}-${record.name}`}
              size="small"
              pagination={{ pageSize: 10 }}
            />
          </TabPane>
        </Tabs>
      </Card>

      {/* 节点详情Modal */}
      <Modal
        title={selectedNode ? `${selectedNode.name} - 详细信息` : '节点详情'}
        visible={nodeModalVisible}
        onCancel={() => setNodeModalVisible(false)}
        footer={[
          <Button key="close" onClick={() => setNodeModalVisible(false)}>
            关闭
          </Button>
        ]}
        width={800}
      >
        {selectedNode && (
          <div>
            <Row gutter={[16, 16]}>
              <Col span={12}>
                <Card title="基本信息" size="small">
                  <div style={{ marginBottom: '8px' }}>
                    <Text strong>节点ID: </Text>
                    <Text code>{selectedNode.id}</Text>
                  </div>
                  <div style={{ marginBottom: '8px' }}>
                    <Text strong>集群ID: </Text>
                    <Text code>{selectedNode.clusterId}</Text>
                  </div>
                  <div style={{ marginBottom: '8px' }}>
                    <Text strong>地址: </Text>
                    <Text>{selectedNode.host}:{selectedNode.port}</Text>
                  </div>
                  <div style={{ marginBottom: '8px' }}>
                    <Text strong>版本: </Text>
                    <Text>{selectedNode.version}</Text>
                  </div>
                  <div>
                    <Text strong>角色: </Text>
                    <Tag color={selectedNode.role === 'primary' ? 'gold' : 'blue'}>
                      {selectedNode.role === 'primary' ? '主节点' : '从节点'}
                    </Tag>
                  </div>
                </Card>
              </Col>
              
              <Col span={12}>
                <Card title="运行状态" size="small">
                  <div style={{ marginBottom: '8px' }}>
                    <Text strong>状态: </Text>
                    <Badge status={selectedNode.status === 'online' ? 'success' : 'error'} />
                    <Text>{selectedNode.status}</Text>
                  </div>
                  <div style={{ marginBottom: '8px' }}>
                    <Text strong>运行时间: </Text>
                    <Text>{selectedNode.uptime}</Text>
                  </div>
                  <div style={{ marginBottom: '8px' }}>
                    <Text strong>最后心跳: </Text>
                    <Text>{selectedNode.lastSeen}</Text>
                  </div>
                  <div>
                    <Text strong>慢消费者: </Text>
                    <Text style={{ color: selectedNode.slowConsumers > 0 ? '#ff4d4f' : '#52c41a' }}>
                      {selectedNode.slowConsumers}
                    </Text>
                  </div>
                </Card>
              </Col>
            </Row>
            
            <Row gutter={[16, 16]} style={{ marginTop: '16px' }}>
              <Col span={24}>
                <Card title="性能指标" size="small">
                  <Row gutter={[16, 16]}>
                    <Col span={8}>
                      <Statistic 
                        title="CPU使用率" 
                        value={selectedNode.cpu} 
                        precision={1} 
                        suffix="%" 
                      />
                      <Progress 
                        percent={selectedNode.cpu} 
                        size="small" 
                        status={selectedNode.cpu > 80 ? 'exception' : 'success'}
                      />
                    </Col>
                    <Col span={8}>
                      <Statistic 
                        title="内存使用率" 
                        value={selectedNode.memory} 
                        precision={1} 
                        suffix="%" 
                      />
                      <Progress 
                        percent={selectedNode.memory} 
                        size="small"
                        status={selectedNode.memory > 85 ? 'exception' : 'success'}
                      />
                    </Col>
                    <Col span={8}>
                      <Statistic 
                        title="连接使用率" 
                        value={(selectedNode.connections / selectedNode.maxConnections * 100).toFixed(1)} 
                        suffix="%" 
                      />
                      <Progress 
                        percent={(selectedNode.connections / selectedNode.maxConnections) * 100} 
                        size="small"
                        status={(selectedNode.connections / selectedNode.maxConnections) > 0.8 ? 'exception' : 'success'}
                      />
                    </Col>
                  </Row>
                </Card>
              </Col>
            </Row>
          </div>
        )}
      </Modal>
    </div>
  )
}

export default NATSClusterManagementPage
