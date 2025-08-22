import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Button,
  Space,
  Table,
  Tag,
  Typography,
  Tabs,
  Statistic,
  Progress,
  Timeline,
  List,
  Modal,
  Form,
  Input,
  Select,
  Switch,
  Alert,
  Tooltip,
  Badge
} from 'antd'
import {
  RobotOutlined,
  ApiOutlined,
  SettingOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  StopOutlined,
  ReloadOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ClockCircleOutlined,
  ThunderboltOutlined,
  TeamOutlined,
  MessageOutlined,
  ControlOutlined,
  MonitorOutlined
} from '@ant-design/icons'

const { Title, Text } = Typography
const { TabPane } = Tabs
const { Option } = Select
const { TextArea } = Input

interface AgentInterface {
  id: string
  name: string
  type: 'chat' | 'task' | 'multi-agent' | 'supervisor'
  status: 'active' | 'idle' | 'busy' | 'offline'
  version: string
  lastActive: string
  totalRequests: number
  avgResponseTime: number
  successRate: number
  currentTasks: number
}

interface InterfaceMetric {
  name: string
  value: number
  unit: string
  trend: 'up' | 'down' | 'stable'
  status: 'good' | 'warning' | 'critical'
}

interface APIRequest {
  id: string
  interface: string
  method: string
  endpoint: string
  status: 'success' | 'error' | 'timeout'
  responseTime: number
  timestamp: string
  requestSize: number
  responseSize: number
}

const AgentInterfacePage: React.FC = () => {
  const [interfaces, setInterfaces] = useState<AgentInterface[]>([
    {
      id: 'chat-interface',
      name: '聊天接口',
      type: 'chat',
      status: 'active',
      version: 'v1.2.3',
      lastActive: '刚刚',
      totalRequests: 1523,
      avgResponseTime: 230,
      successRate: 98.5,
      currentTasks: 3
    },
    {
      id: 'task-interface',
      name: '任务执行接口',
      type: 'task',
      status: 'busy',
      version: 'v1.2.1',
      lastActive: '30秒前',
      totalRequests: 856,
      avgResponseTime: 450,
      successRate: 95.2,
      currentTasks: 8
    },
    {
      id: 'multi-agent-interface',
      name: '多代理协作接口',
      type: 'multi-agent',
      status: 'active',
      version: 'v1.1.5',
      lastActive: '1分钟前',
      totalRequests: 324,
      avgResponseTime: 680,
      successRate: 92.1,
      currentTasks: 2
    },
    {
      id: 'supervisor-interface',
      name: '监督者接口',
      type: 'supervisor',
      status: 'idle',
      version: 'v1.0.8',
      lastActive: '5分钟前',
      totalRequests: 145,
      avgResponseTime: 320,
      successRate: 97.8,
      currentTasks: 0
    }
  ])

  const [metrics] = useState<InterfaceMetric[]>([
    { name: 'QPS (每秒请求)', value: 45, unit: 'req/s', trend: 'up', status: 'good' },
    { name: '平均响应时间', value: 285, unit: 'ms', trend: 'stable', status: 'good' },
    { name: '错误率', value: 2.3, unit: '%', trend: 'down', status: 'good' },
    { name: '并发连接数', value: 128, unit: '个', trend: 'up', status: 'warning' },
    { name: '内存使用率', value: 67, unit: '%', trend: 'stable', status: 'good' },
    { name: 'CPU使用率', value: 45, unit: '%', trend: 'up', status: 'good' }
  ])

  const [recentRequests] = useState<APIRequest[]>([
    {
      id: '1',
      interface: '聊天接口',
      method: 'POST',
      endpoint: '/api/v1/agent-interface/chat',
      status: 'success',
      responseTime: 245,
      timestamp: '14:35:22',
      requestSize: 1.2,
      responseSize: 3.4
    },
    {
      id: '2',
      interface: '任务执行接口',
      method: 'POST',
      endpoint: '/api/v1/agent-interface/task',
      status: 'success',
      responseTime: 456,
      timestamp: '14:35:18',
      requestSize: 2.1,
      responseSize: 5.6
    },
    {
      id: '3',
      interface: '多代理协作接口',
      method: 'GET',
      endpoint: '/api/v1/agent-interface/status',
      status: 'error',
      responseTime: 0,
      timestamp: '14:35:15',
      requestSize: 0.5,
      responseSize: 0.2
    }
  ])

  const [showConfigModal, setShowConfigModal] = useState(false)
  const [selectedInterface, setSelectedInterface] = useState<AgentInterface | null>(null)

  const getStatusColor = (status: string) => {
    const colors = {
      active: 'success',
      idle: 'default',
      busy: 'processing',
      offline: 'error'
    }
    return colors[status as keyof typeof colors] || 'default'
  }

  const getStatusIcon = (status: string) => {
    const icons = {
      active: <CheckCircleOutlined />,
      idle: <ClockCircleOutlined />,
      busy: <PlayCircleOutlined />,
      offline: <CloseCircleOutlined />
    }
    return icons[status as keyof typeof icons]
  }

  const getTypeIcon = (type: string) => {
    const icons = {
      chat: <MessageOutlined />,
      task: <ThunderboltOutlined />,
      'multi-agent': <TeamOutlined />,
      supervisor: <ControlOutlined />
    }
    return icons[type as keyof typeof icons]
  }

  const getTypeName = (type: string) => {
    const names = {
      chat: '聊天接口',
      task: '任务接口',
      'multi-agent': '多代理接口',
      supervisor: '监督者接口'
    }
    return names[type as keyof typeof names] || type
  }

  const getMetricStatusColor = (status: string) => {
    const colors = {
      good: '#52c41a',
      warning: '#faad14',
      critical: '#ff4d4f'
    }
    return colors[status as keyof typeof colors]
  }

  const getTrendIcon = (trend: string) => {
    const icons = {
      up: '↗',
      down: '↘',
      stable: '→'
    }
    return icons[trend as keyof typeof icons]
  }

  const getRequestStatusColor = (status: string) => {
    const colors = {
      success: 'green',
      error: 'red',
      timeout: 'orange'
    }
    return colors[status as keyof typeof colors]
  }

  const columns = [
    {
      title: '接口名称',
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record: AgentInterface) => (
        <div>
          <Space>
            {getTypeIcon(record.type)}
            <Text strong>{name}</Text>
          </Space>
          <br />
          <Text type="secondary" className="text-xs">
            {record.id} - {record.version}
          </Text>
        </div>
      )
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => (
        <Tag icon={getTypeIcon(type)}>
          {getTypeName(type)}
        </Tag>
      )
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={getStatusColor(status)} icon={getStatusIcon(status)}>
          {status === 'active' && '活跃'}
          {status === 'idle' && '空闲'}
          {status === 'busy' && '忙碌'}
          {status === 'offline' && '离线'}
        </Tag>
      )
    },
    {
      title: '请求统计',
      key: 'requests',
      render: (record: AgentInterface) => (
        <div>
          <Text strong>{record.totalRequests}</Text>
          <Text type="secondary"> 总请求</Text>
          <br />
          <Text type="secondary">{record.currentTasks} 当前任务</Text>
        </div>
      )
    },
    {
      title: '性能指标',
      key: 'performance',
      render: (record: AgentInterface) => (
        <div>
          <Text>响应: {record.avgResponseTime}ms</Text>
          <br />
          <Text>成功率: </Text>
          <Text style={{ color: record.successRate > 95 ? '#52c41a' : '#faad14' }}>
            {record.successRate}%
          </Text>
        </div>
      )
    },
    {
      title: '最后活跃',
      dataIndex: 'lastActive',
      key: 'lastActive'
    },
    {
      title: '操作',
      key: 'action',
      render: (_: any, record: AgentInterface) => (
        <Space>
          {record.status === 'active' && (
            <Tooltip title="暂停">
              <Button size="small" icon={<PauseCircleOutlined />} />
            </Tooltip>
          )}
          {record.status === 'idle' && (
            <Tooltip title="启动">
              <Button size="small" icon={<PlayCircleOutlined />} />
            </Tooltip>
          )}
          <Tooltip title="重启">
            <Button size="small" icon={<ReloadOutlined />} />
          </Tooltip>
          <Tooltip title="配置">
            <Button 
              size="small" 
              icon={<SettingOutlined />}
              onClick={() => {
                setSelectedInterface(record)
                setShowConfigModal(true)
              }}
            />
          </Tooltip>
        </Space>
      )
    }
  ]

  const requestColumns = [
    {
      title: '时间',
      dataIndex: 'timestamp',
      key: 'timestamp'
    },
    {
      title: '接口',
      dataIndex: 'interface',
      key: 'interface'
    },
    {
      title: '方法',
      dataIndex: 'method',
      key: 'method',
      render: (method: string) => (
        <Tag color={method === 'POST' ? 'blue' : method === 'GET' ? 'green' : 'orange'}>
          {method}
        </Tag>
      )
    },
    {
      title: '端点',
      dataIndex: 'endpoint',
      key: 'endpoint',
      render: (endpoint: string) => (
        <Text code className="text-xs">{endpoint}</Text>
      )
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={getRequestStatusColor(status)}>
          {status.toUpperCase()}
        </Tag>
      )
    },
    {
      title: '响应时间',
      dataIndex: 'responseTime',
      key: 'responseTime',
      render: (time: number) => (
        <Text style={{ color: time > 500 ? '#ff4d4f' : time > 200 ? '#faad14' : '#52c41a' }}>
          {time}ms
        </Text>
      )
    },
    {
      title: '数据大小',
      key: 'size',
      render: (record: APIRequest) => (
        <div>
          <Text className="text-xs">请求: {record.requestSize}KB</Text>
          <br />
          <Text className="text-xs">响应: {record.responseSize}KB</Text>
        </div>
      )
    }
  ]

  const totalRequests = interfaces.reduce((sum, iface) => sum + iface.totalRequests, 0)
  const avgSuccessRate = interfaces.reduce((sum, iface) => sum + iface.successRate, 0) / interfaces.length
  const activeInterfaces = interfaces.filter(iface => iface.status === 'active').length

  return (
    <div className="p-6">
      <div className="mb-6">
        <div className="flex justify-between items-center mb-4">
          <Title level={2}>Agent接口管理</Title>
          <Space>
            <Button 
              icon={<ReloadOutlined />}
              onClick={() => console.log('刷新所有接口状态')}
            >
              刷新状态
            </Button>
            <Button 
              icon={<MonitorOutlined />}
              onClick={() => console.log('查看详细监控')}
            >
              详细监控
            </Button>
            <Button 
              type="primary"
              icon={<SettingOutlined />}
              onClick={() => setShowConfigModal(true)}
            >
              全局配置
            </Button>
          </Space>
        </div>

        <Row gutter={16} className="mb-6">
          <Col span={6}>
            <Card>
              <Statistic
                title="活跃接口"
                value={activeInterfaces}
                suffix={`/ ${interfaces.length}`}
                valueStyle={{ color: '#3f8600' }}
                prefix={<ApiOutlined />}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="总请求数"
                value={totalRequests}
                prefix={<ThunderboltOutlined />}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="平均成功率"
                value={avgSuccessRate}
                precision={1}
                suffix="%"
                valueStyle={{ color: avgSuccessRate > 95 ? '#3f8600' : '#faad14' }}
                prefix={<CheckCircleOutlined />}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="活跃任务"
                value={interfaces.reduce((sum, iface) => sum + iface.currentTasks, 0)}
                prefix={<RobotOutlined />}
              />
            </Card>
          </Col>
        </Row>
      </div>

      <Tabs defaultActiveKey="interfaces">
        <TabPane tab="接口管理" key="interfaces">
          <Card>
            <Table
              columns={columns}
              dataSource={interfaces}
              rowKey="id"
              pagination={false}
              size="small"
            />
          </Card>
        </TabPane>

        <TabPane tab="性能监控" key="metrics">
          <Row gutter={16}>
            {metrics.map((metric, index) => (
              <Col span={8} key={index} className="mb-4">
                <Card>
                  <div className="flex justify-between items-start mb-2">
                    <Text strong>{metric.name}</Text>
                    <div className="flex items-center">
                      <span className="text-xs mr-1">{getTrendIcon(metric.trend)}</span>
                      <Badge 
                        status={metric.status === 'good' ? 'success' : metric.status === 'warning' ? 'warning' : 'error'}
                      />
                    </div>
                  </div>
                  <div 
                    className="text-2xl font-bold mb-2"
                    style={{ color: getMetricStatusColor(metric.status) }}
                  >
                    {metric.value}{metric.unit}
                  </div>
                  <Progress 
                    percent={metric.value > 100 ? 100 : metric.value}
                    strokeColor={getMetricStatusColor(metric.status)}
                    size="small"
                  />
                </Card>
              </Col>
            ))}
          </Row>
        </TabPane>

        <TabPane tab="请求日志" key="requests">
          <Card title="最近API请求">
            <Table
              columns={requestColumns}
              dataSource={recentRequests}
              rowKey="id"
              pagination={{ pageSize: 50 }}
              size="small"
            />
          </Card>
        </TabPane>

        <TabPane tab="接口文档" key="docs">
          <Row gutter={16}>
            <Col span={12}>
              <Card title="聊天接口 API">
                <Space direction="vertical" style={{ width: '100%' }}>
                  <div>
                    <Text strong>POST</Text> <Text code>/api/v1/agent-interface/chat</Text>
                  </div>
                  <div>
                    <Text type="secondary">发送聊天消息并获取AI响应</Text>
                  </div>
                  <div>
                    <Text strong>参数:</Text>
                    <br />
                    <Text code>message: string</Text> - 用户消息
                    <br />
                    <Text code>conversation_id: string</Text> - 会话ID
                  </div>
                </Space>
              </Card>
            </Col>
            <Col span={12}>
              <Card title="任务执行接口 API">
                <Space direction="vertical" style={{ width: '100%' }}>
                  <div>
                    <Text strong>POST</Text> <Text code>/api/v1/agent-interface/task</Text>
                  </div>
                  <div>
                    <Text type="secondary">执行复杂任务并返回结果</Text>
                  </div>
                  <div>
                    <Text strong>参数:</Text>
                    <br />
                    <Text code>task_type: string</Text> - 任务类型
                    <br />
                    <Text code>parameters: object</Text> - 任务参数
                  </div>
                </Space>
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab="活动日志" key="logs">
          <Card title="接口活动日志">
            <Timeline
              items={[
                {
                  color: 'green',
                  children: (
                    <div>
                      <Text strong>聊天接口响应正常</Text>
                      <br />
                      <Text type="secondary">处理用户请求，响应时间245ms - 刚刚</Text>
                    </div>
                  )
                },
                {
                  color: 'blue',
                  children: (
                    <div>
                      <Text strong>任务执行接口启动新任务</Text>
                      <br />
                      <Text type="secondary">开始处理数据分析任务 - 30秒前</Text>
                    </div>
                  )
                },
                {
                  color: 'red',
                  children: (
                    <div>
                      <Text strong>多代理接口连接失败</Text>
                      <br />
                      <Text type="secondary">网络连接超时，正在重试 - 1分钟前</Text>
                    </div>
                  )
                },
                {
                  color: 'orange',
                  children: (
                    <div>
                      <Text strong>监督者接口进入空闲状态</Text>
                      <br />
                      <Text type="secondary">所有监督任务已完成 - 5分钟前</Text>
                    </div>
                  )
                }
              ]}
            />
          </Card>
        </TabPane>
      </Tabs>

      {/* 配置Modal */}
      <Modal
        title={selectedInterface ? `配置 - ${selectedInterface.name}` : "全局接口配置"}
        open={showConfigModal}
        onCancel={() => {
          setShowConfigModal(false)
          setSelectedInterface(null)
        }}
        footer={null}
        width={600}
      >
        <Form layout="vertical">
          <Form.Item label="接口名称">
            <Input 
              defaultValue={selectedInterface?.name || ''}
              placeholder="输入接口名称"
            />
          </Form.Item>
          <Form.Item label="版本">
            <Input 
              defaultValue={selectedInterface?.version || 'v1.0.0'}
              placeholder="接口版本"
            />
          </Form.Item>
          <Form.Item label="超时时间(秒)">
            <Select defaultValue={30}>
              <Option value={10}>10秒</Option>
              <Option value={30}>30秒</Option>
              <Option value={60}>60秒</Option>
              <Option value={300}>5分钟</Option>
            </Select>
          </Form.Item>
          <Form.Item label="最大并发数">
            <Select defaultValue={100}>
              <Option value={50}>50</Option>
              <Option value={100}>100</Option>
              <Option value={200}>200</Option>
              <Option value={500}>500</Option>
            </Select>
          </Form.Item>
          <Form.Item label="启用限流">
            <Switch defaultChecked />
          </Form.Item>
          <Form.Item label="启用缓存">
            <Switch defaultChecked />
          </Form.Item>
          <Form.Item label="配置参数">
            <TextArea 
              rows={4} 
              placeholder="JSON格式的配置参数"
              defaultValue={JSON.stringify({
                cache_ttl: 3600,
                retry_count: 3,
                rate_limit: "100/minute"
              }, null, 2)}
            />
          </Form.Item>
          <Form.Item>
            <Space>
              <Button type="primary">保存配置</Button>
              <Button onClick={() => setShowConfigModal(false)}>取消</Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}

export default AgentInterfacePage