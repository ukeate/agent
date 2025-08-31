import React, { useState, useEffect } from 'react'
import { Card, Row, Col, Statistic, Progress, Timeline, Alert, Button, Space, Typography, Tag, Divider, Table, Badge } from 'antd'
import { 
  NetworkOutlined, 
  MessageOutlined, 
  ClusterOutlined, 
  ThunderboltOutlined,
  SafetyCertificateOutlined,
  MonitorOutlined,
  ApiOutlined,
  ShareAltOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  ReloadOutlined
} from '@ant-design/icons'

const { Title, Paragraph, Text } = Typography

interface SystemMetrics {
  totalMessages: number
  activeConnections: number
  clusterNodes: number
  messagesThroughput: number
  averageLatency: number
  errorRate: number
  systemHealth: 'healthy' | 'warning' | 'error'
}

interface ClusterNode {
  id: string
  name: string
  status: 'online' | 'offline' | 'error'
  cpu: number
  memory: number
  connections: number
  uptime: string
}

interface MessagePattern {
  name: string
  description: string
  messagesProcessed: number
  averageLatency: number
  successRate: number
  status: 'active' | 'idle'
}

const DistributedMessageOverviewPage: React.FC = () => {
  const [metrics, setMetrics] = useState<SystemMetrics>({
    totalMessages: 1250847,
    activeConnections: 245,
    clusterNodes: 3,
    messagesThroughput: 1850,
    averageLatency: 12.5,
    errorRate: 0.02,
    systemHealth: 'healthy'
  })

  const [loading, setLoading] = useState(false)

  const clusterNodes: ClusterNode[] = [
    {
      id: 'nats-1',
      name: 'NATS主节点',
      status: 'online',
      cpu: 45,
      memory: 62,
      connections: 89,
      uptime: '15天 8小时'
    },
    {
      id: 'nats-2', 
      name: 'NATS从节点1',
      status: 'online',
      cpu: 38,
      memory: 55,
      connections: 76,
      uptime: '15天 8小时'
    },
    {
      id: 'nats-3',
      name: 'NATS从节点2', 
      status: 'online',
      cpu: 52,
      memory: 68,
      connections: 80,
      uptime: '15天 8小时'
    }
  ]

  const messagePatterns: MessagePattern[] = [
    {
      name: '点对点通信',
      description: '智能体间直接消息传递',
      messagesProcessed: 458923,
      averageLatency: 8.5,
      successRate: 99.95,
      status: 'active'
    },
    {
      name: '多播通信',
      description: '群组消息广播',
      messagesProcessed: 123456,
      averageLatency: 15.2,
      successRate: 99.87,
      status: 'active'
    },
    {
      name: '流式传输',
      description: '大文件数据流传输',
      messagesProcessed: 23847,
      averageLatency: 125.6,
      successRate: 99.92,
      status: 'active'
    },
    {
      name: '请求响应',
      description: '同步请求响应模式',
      messagesProcessed: 645123,
      averageLatency: 22.8,
      successRate: 99.88,
      status: 'active'
    }
  ]

  const systemTimeline = [
    {
      color: 'green',
      children: (
        <div>
          <Text strong>NATS集群启动完成</Text>
          <br />
          <Text type="secondary">3个节点成功启动，集群状态正常</Text>
          <br />
          <Text type="secondary">2分钟前</Text>
        </div>
      ),
    },
    {
      color: 'blue',
      children: (
        <div>
          <Text strong>高级通信模式激活</Text>
          <br />
          <Text type="secondary">多播、流式传输和智能路由模式已启用</Text>
          <br />
          <Text type="secondary">5分钟前</Text>
        </div>
      ),
    },
    {
      color: 'green',
      children: (
        <div>
          <Text strong>可靠性保证机制部署</Text>
          <br />
          <Text type="secondary">消息重试、死信队列和ACK确认机制已配置</Text>
          <br />
          <Text type="secondary">10分钟前</Text>
        </div>
      ),
    },
    {
      color: 'blue',
      children: (
        <div>
          <Text strong>监控和性能优化系统启动</Text>
          <br />
          <Text type="secondary">实时指标收集和告警系统已激活</Text>
          <br />
          <Text type="secondary">15分钟前</Text>
        </div>
      ),
    },
  ]

  const clusterColumns = [
    {
      title: '节点名称',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: ClusterNode) => (
        <Space>
          <Badge 
            status={record.status === 'online' ? 'success' : record.status === 'offline' ? 'default' : 'error'}
          />
          <Text strong>{text}</Text>
        </Space>
      )
    },
    {
      title: 'CPU使用率',
      dataIndex: 'cpu',
      key: 'cpu',
      render: (value: number) => (
        <Progress 
          percent={value} 
          size="small" 
          status={value > 80 ? 'exception' : value > 60 ? 'active' : 'success'}
          format={(percent) => `${percent}%`}
        />
      )
    },
    {
      title: '内存使用率', 
      dataIndex: 'memory',
      key: 'memory',
      render: (value: number) => (
        <Progress 
          percent={value} 
          size="small"
          status={value > 85 ? 'exception' : value > 70 ? 'active' : 'success'}
          format={(percent) => `${percent}%`}
        />
      )
    },
    {
      title: '活跃连接',
      dataIndex: 'connections',
      key: 'connections',
      render: (value: number) => <Text>{value}</Text>
    },
    {
      title: '运行时间',
      dataIndex: 'uptime',
      key: 'uptime',
      render: (text: string) => <Text type="secondary">{text}</Text>
    }
  ]

  const patternColumns = [
    {
      title: '通信模式',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: MessagePattern) => (
        <div>
          <Text strong>{text}</Text>
          <br />
          <Text type="secondary" style={{ fontSize: '12px' }}>{record.description}</Text>
        </div>
      )
    },
    {
      title: '消息处理量',
      dataIndex: 'messagesProcessed',
      key: 'messagesProcessed',
      render: (value: number) => <Statistic value={value} precision={0} />
    },
    {
      title: '平均延迟',
      dataIndex: 'averageLatency',
      key: 'averageLatency',
      render: (value: number) => (
        <Tag color={value < 20 ? 'green' : value < 50 ? 'orange' : 'red'}>
          {value}ms
        </Tag>
      )
    },
    {
      title: '成功率',
      dataIndex: 'successRate',
      key: 'successRate',
      render: (value: number) => (
        <div>
          <Progress 
            percent={value} 
            size="small"
            status={value > 99 ? 'success' : value > 95 ? 'active' : 'exception'}
            format={(percent) => `${percent}%`}
          />
        </div>
      )
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Badge 
          status={status === 'active' ? 'processing' : 'default'}
          text={status === 'active' ? '活跃' : '空闲'}
        />
      )
    }
  ]

  const refreshMetrics = async () => {
    setLoading(true)
    // 模拟API调用
    setTimeout(() => {
      setMetrics(prev => ({
        ...prev,
        totalMessages: prev.totalMessages + Math.floor(Math.random() * 1000),
        messagesThroughput: 1500 + Math.floor(Math.random() * 700),
        averageLatency: 10 + Math.random() * 10,
        errorRate: Math.random() * 0.05
      }))
      setLoading(false)
    }, 1000)
  }

  useEffect(() => {
    const interval = setInterval(refreshMetrics, 30000)
    return () => clearInterval(interval)
  }, [])

  const getHealthColor = () => {
    switch (metrics.systemHealth) {
      case 'healthy': return 'green'
      case 'warning': return 'orange'
      case 'error': return 'red'
      default: return 'gray'
    }
  }

  const getHealthIcon = () => {
    switch (metrics.systemHealth) {
      case 'healthy': return <CheckCircleOutlined />
      case 'warning': return <ExclamationCircleOutlined />
      case 'error': return <ExclamationCircleOutlined />
      default: return <NetworkOutlined />
    }
  }

  return (
    <div style={{ padding: '24px', background: '#f5f5f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <NetworkOutlined style={{ marginRight: '12px', color: '#1890ff' }} />
          分布式消息通信框架总览
        </Title>
        <Paragraph>
          基于NATS JetStream的智能体间通信系统，提供可靠、高效、可扩展的消息传递服务
        </Paragraph>
      </div>

      {/* 系统状态告警 */}
      {metrics.systemHealth !== 'healthy' && (
        <Alert
          message="系统状态警告"
          description={
            metrics.systemHealth === 'warning' 
              ? "系统运行正常但有轻微异常，建议检查相关指标" 
              : "系统检测到严重问题，请立即处理"
          }
          type={metrics.systemHealth === 'warning' ? 'warning' : 'error'}
          showIcon
          style={{ marginBottom: '24px' }}
        />
      )}

      {/* 核心指标卡片 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="消息总量"
              value={metrics.totalMessages}
              precision={0}
              valueStyle={{ color: '#3f8600' }}
              prefix={<MessageOutlined />}
            />
            <div style={{ marginTop: '8px' }}>
              <Text type="secondary">今日新增: +{Math.floor(metrics.totalMessages * 0.05)}</Text>
            </div>
          </Card>
        </Col>
        
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="活跃连接"
              value={metrics.activeConnections}
              precision={0}
              valueStyle={{ color: '#1890ff' }}
              prefix={<ClusterOutlined />}
            />
            <div style={{ marginTop: '8px' }}>
              <Text type="secondary">峰值: {metrics.activeConnections + 50}</Text>
            </div>
          </Card>
        </Col>
        
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="消息吞吐量"
              value={metrics.messagesThroughput}
              precision={0}
              valueStyle={{ color: '#722ed1' }}
              prefix={<ThunderboltOutlined />}
              suffix="msg/s"
            />
            <div style={{ marginTop: '8px' }}>
              <Progress 
                percent={Math.round((metrics.messagesThroughput / 2500) * 100)} 
                size="small" 
                showInfo={false}
              />
            </div>
          </Card>
        </Col>
        
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="系统健康度"
              value={metrics.systemHealth === 'healthy' ? '正常' : metrics.systemHealth === 'warning' ? '警告' : '异常'}
              valueStyle={{ color: getHealthColor() }}
              prefix={getHealthIcon()}
            />
            <div style={{ marginTop: '8px' }}>
              <Tag color={getHealthColor()}>
                延迟: {metrics.averageLatency.toFixed(1)}ms
              </Tag>
              <Tag color={metrics.errorRate < 0.01 ? 'green' : 'orange'}>
                错误率: {(metrics.errorRate * 100).toFixed(2)}%
              </Tag>
            </div>
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        {/* 集群节点状态 */}
        <Col xs={24} lg={14}>
          <Card 
            title={
              <Space>
                <ClusterOutlined style={{ color: '#1890ff' }} />
                <span>NATS集群节点状态</span>
                <Button 
                  type="link" 
                  size="small" 
                  icon={<ReloadOutlined />}
                  loading={loading}
                  onClick={refreshMetrics}
                >
                  刷新
                </Button>
              </Space>
            }
          >
            <Table 
              columns={clusterColumns}
              dataSource={clusterNodes}
              rowKey="id"
              size="small"
              pagination={false}
            />
          </Card>
        </Col>

        {/* 系统活动时间线 */}
        <Col xs={24} lg={10}>
          <Card title={
            <Space>
              <MonitorOutlined style={{ color: '#52c41a' }} />
              <span>系统活动日志</span>
            </Space>
          }>
            <Timeline items={systemTimeline} />
          </Card>
        </Col>
      </Row>

      {/* 通信模式统计 */}
      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Card 
            title={
              <Space>
                <ShareAltOutlined style={{ color: '#722ed1' }} />
                <span>通信模式统计</span>
              </Space>
            }
          >
            <Table 
              columns={patternColumns}
              dataSource={messagePatterns}
              rowKey="name"
              size="small"
              pagination={false}
            />
          </Card>
        </Col>
      </Row>

      {/* 功能模块快速访问 */}
      <Row gutter={[16, 16]} style={{ marginTop: '24px' }}>
        <Col span={24}>
          <Card title="功能模块快速访问">
            <Row gutter={[16, 16]}>
              <Col xs={12} sm={8} md={6}>
                <Card 
                  size="small" 
                  hoverable
                  style={{ textAlign: 'center', cursor: 'pointer' }}
                  onClick={() => window.location.href = '/nats-cluster-management'}
                >
                  <ClusterOutlined style={{ fontSize: '32px', color: '#1890ff', marginBottom: '8px' }} />
                  <div>集群管理</div>
                </Card>
              </Col>
              <Col xs={12} sm={8} md={6}>
                <Card 
                  size="small" 
                  hoverable
                  style={{ textAlign: 'center', cursor: 'pointer' }}
                  onClick={() => window.location.href = '/basic-message-communication'}
                >
                  <MessageOutlined style={{ fontSize: '32px', color: '#52c41a', marginBottom: '8px' }} />
                  <div>基础通信</div>
                </Card>
              </Col>
              <Col xs={12} sm={8} md={6}>
                <Card 
                  size="small" 
                  hoverable
                  style={{ textAlign: 'center', cursor: 'pointer' }}
                  onClick={() => window.location.href = '/acl-protocol-management'}
                >
                  <ApiOutlined style={{ fontSize: '32px', color: '#722ed1', marginBottom: '8px' }} />
                  <div>ACL协议</div>
                </Card>
              </Col>
              <Col xs={12} sm={8} md={6}>
                <Card 
                  size="small" 
                  hoverable
                  style={{ textAlign: 'center', cursor: 'pointer' }}
                  onClick={() => window.location.href = '/message-reliability-management'}
                >
                  <SafetyCertificateOutlined style={{ fontSize: '32px', color: '#fa8c16', marginBottom: '8px' }} />
                  <div>可靠性保证</div>
                </Card>
              </Col>
            </Row>
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default DistributedMessageOverviewPage