import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useState, useEffect } from 'react'
import {
import { logger } from '../utils/logger'
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
  const [interfaces, setInterfaces] = useState<AgentInterface[]>([])
  const [metrics, setMetrics] = useState<InterfaceMetric[]>([])
  const [recentRequests, setRecentRequests] = useState<APIRequest[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
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
  const avgSuccessRate = interfaces.length > 0 ? interfaces.reduce((sum, iface) => sum + iface.successRate, 0) / interfaces.length : 0
  const activeInterfaces = interfaces.filter(iface => iface.status === 'active').length

  const loadData = async () => {
    setLoading(true)
    setError('')
    try {
      const statusRes = await apiFetch(buildApiUrl('/api/v1/agent/status'))
      const statusBody = await statusRes.json()
      const statusData = statusBody.data || statusBody
      if (statusData) {
        const iface: AgentInterface = {
          id: statusData.agent_info?.agent_id || 'agent-interface',
          name: '主接口',
          type: 'chat',
          status: statusData.health === 'degraded' ? 'busy' : statusData.health === 'unhealthy' ? 'offline' : 'active',
          version: statusData.agent_info?.version || 'v1.0.0',
          lastActive: statusData.last_activity ? new Date(statusData.last_activity).toLocaleString('zh-CN') : '',
          totalRequests: Math.round(statusData.performance_metrics?.requests_per_minute || 0),
          avgResponseTime: Math.round(statusData.performance_metrics?.average_response_time || 0),
          successRate: Number(statusData.performance_metrics?.success_rate || 0),
          currentTasks: statusData.agent_info?.active_conversations || 0
        }
        setInterfaces([iface])
        const metricList: InterfaceMetric[] = [
          {
            name: '平均响应时间',
            value: Math.round(statusData.performance_metrics?.average_response_time || 0),
            unit: 'ms',
            trend: 'stable',
            status: 'good'
          },
          {
            name: '请求速率',
            value: Number(statusData.performance_metrics?.requests_per_minute || 0),
            unit: 'rpm',
            trend: 'up',
            status: 'good'
          },
          {
            name: '成功率',
            value: Number(statusData.performance_metrics?.success_rate || 0),
            unit: '%',
            trend: 'stable',
            status: 'good'
          },
          {
            name: 'CPU使用率',
            value: Number(statusData.system_resources?.cpu_usage || 0),
            unit: '%',
            trend: 'stable',
            status: 'warning'
          },
          {
            name: '内存使用率',
            value: Number(statusData.system_resources?.memory_usage || 0),
            unit: '%',
            trend: 'stable',
            status: 'warning'
          },
          {
            name: '活跃会话',
            value: Number(statusData.agent_info?.active_conversations || 0),
            unit: '个',
            trend: 'stable',
            status: 'good'
          }
        ]
        setMetrics(metricList)
      }
      try {
        const metricRes = await apiFetch(buildApiUrl('/api/v1/agent/metrics'))
        const metricBody = await metricRes.json()
        const metricData = metricBody.data || metricBody
        if (metricData && metricData.requests) {
          const mapped = (metricData.requests || []).slice(-50).map((req: any, idx: number) => ({
            id: req.id || String(idx),
            interface: req.interface || 'agent',
            method: (req.method || 'GET').toUpperCase(),
            endpoint: req.path || req.endpoint || '',
            status: (req.status || 'success') as APIRequest['status'],
            responseTime: Math.round(req.latency_ms || req.response_time || 0),
            timestamp: req.timestamp || new Date().toLocaleTimeString('zh-CN'),
            requestSize: Number(req.request_size_kb || 0),
            responseSize: Number(req.response_size_kb || 0)
          }))
          setRecentRequests(mapped)
        } else {
          setRecentRequests([])
        }
      } catch {
        setRecentRequests([])
      }
    } catch (e) {
      setError('加载接口状态失败')
      setInterfaces([])
      setMetrics([])
      setRecentRequests([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
  }, [])

  return (
    <div className="p-6">
      <div className="mb-6">
        <div className="flex justify-between items-center mb-4">
          <Title level={2}>Agent接口管理</Title>
          <Space>
            <Button 
              icon={<ReloadOutlined />}
              loading={loading}
              onClick={loadData}
            >
              刷新状态
            </Button>
            <Button 
              icon={<MonitorOutlined />}
              onClick={() => logger.log('查看详细监控')}
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
              loading={loading}
            />
          </Card>
        </TabPane>

        <TabPane tab="性能监控" key="metrics">
          <Row gutter={16}>
            {metrics.length === 0 ? (
              <Col span={24}>
                <Alert message="暂无实时性能数据" type="info" showIcon />
              </Col>
            ) : (
              metrics.map((metric, index) => (
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
              ))
            )}
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
              locale={{ emptyText: '暂无请求数据' }}
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
            <Alert message="暂无活动日志数据" type="info" showIcon />
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
