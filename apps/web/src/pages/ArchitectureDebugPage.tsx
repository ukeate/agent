import React, { useEffect, useMemo, useState } from 'react'
import {
  Card,
  Row,
  Col,
  Button,
  Space,
  Tree,
  Badge,
  Tag,
  Descriptions,
  Table,
  Tabs,
  Alert,
  Timeline
} from 'antd'
import {
  NodeIndexOutlined,
  ApiOutlined,
  DatabaseOutlined,
  CloudOutlined,
  BugOutlined,
  ReloadOutlined,
  SettingOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  WarningOutlined,
  InfoCircleOutlined
} from '@ant-design/icons'
import apiClient from '../services/apiClient'
import { healthService } from '../services/healthService'

const { TabPane } = Tabs

interface Component {
  id: string
  name: string
  type: 'service' | 'database' | 'api' | 'ui'
  status: 'healthy' | 'warning' | 'error' | 'unknown'
  version: string
  dependencies: string[]
  endpoints?: string[]
  metrics: {
    uptime: number | null
    responseTime: number | null
    errorRate: number | null
  }
}

interface DebugSession {
  id: string
  timestamp: string
  component: string
  issue: string
  status: 'investigating' | 'resolved' | 'pending'
  priority: 'low' | 'medium' | 'high' | 'critical'
}

const ArchitectureDebugPage: React.FC = () => {
  const [components, setComponents] = useState<Component[]>([])
  const [debugSessions, setDebugSessions] = useState<DebugSession[]>([])
  const [selectedComponent, setSelectedComponent] = useState<Component | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const statusColors = {
    healthy: 'success',
    warning: 'warning',
    error: 'error',
    unknown: 'default'
  }

  const statusIcons = {
    healthy: <CheckCircleOutlined style={{ color: '#52c41a' }} />,
    warning: <WarningOutlined style={{ color: '#faad14' }} />,
    error: <CloseCircleOutlined style={{ color: '#ff4d4f' }} />,
    unknown: <InfoCircleOutlined style={{ color: '#d9d9d9' }} />
  }

  const typeIcons = {
    ui: <NodeIndexOutlined />,
    api: <ApiOutlined />,
    service: <CloudOutlined />,
    database: <DatabaseOutlined />
  }

  const priorityColors = {
    low: 'default',
    medium: 'warning',
    high: 'error',
    critical: 'error'
  }

  const statusSessionColors = {
    investigating: 'processing',
    resolved: 'success',
    pending: 'warning'
  }

  const mapHealthStatus = (status: string): Component['status'] => {
    if (status === 'healthy') return 'healthy'
    if (status === 'degraded') return 'warning'
    if (status === 'unhealthy') return 'error'
    return 'unknown'
  }

  const formatDuration = (seconds: number | null) => {
    if (seconds === null || Number.isNaN(seconds)) return '-'
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    return `${hours}h ${minutes}m`
  }

  const loadData = async () => {
    setLoading(true)
    setError(null)
    try {
      const [health, alertsRes] = await Promise.all([
        healthService.getHealth(true),
        apiClient.get('/health/alerts')
      ])
      const componentEntries = Object.entries(health.components || {})
      const nextComponents: Component[] = componentEntries.map(([name, data]) => {
        const responseTime = typeof (data as any).response_time_ms === 'number'
          ? (data as any).response_time_ms
          : (typeof (data as any).response_time === 'number' ? (data as any).response_time : null)
        const uptimeSeconds = typeof (data as any).uptime_seconds === 'number'
          ? (data as any).uptime_seconds
          : null
        const componentType: Component['type'] =
          name.includes('database') || name.includes('redis') ? 'database' :
          name === 'api' ? 'api' : 'service'
        return {
          id: name,
          name,
          type: componentType,
          status: mapHealthStatus((data as any).status),
          version: '-',
          dependencies: [],
          metrics: {
            uptime: uptimeSeconds,
            responseTime,
            errorRate: null
          }
        }
      })
      setComponents(nextComponents)
      setSelectedComponent(prev => prev || nextComponents[0] || null)

      const alerts = alertsRes.data?.alerts || []
      const sessions: DebugSession[] = alerts.map((alert: any, idx: number) => ({
        id: alert.name || `alert_${idx}`,
        timestamp: alert.timestamp || new Date().toISOString(),
        component: alert.component || alert.name || 'system',
        issue: alert.message || '检测到系统告警',
        status: 'investigating',
        priority: alert.severity === 'critical' ? 'critical' : alert.severity === 'warning' ? 'medium' : 'low'
      }))
      setDebugSessions(sessions)
    } catch (err) {
      setError((err as Error).message || '加载架构数据失败')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
  }, [])

  const treeData = useMemo(() => {
    const groups: Array<{ key: string; title: string; items: Component[] }> = [
      { key: 'api', title: 'API层', items: components.filter(c => c.type === 'api') },
      { key: 'services', title: '服务层', items: components.filter(c => c.type === 'service') },
      { key: 'database', title: '数据层', items: components.filter(c => c.type === 'database') }
    ].filter(group => group.items.length > 0)

    return [
      {
        title: (
          <Space>
            <NodeIndexOutlined />
            <span>系统架构</span>
          </Space>
        ),
        key: 'root',
        children: groups.map(group => ({
          title: (
            <Space>
              <Badge status={statusColors.healthy} />
              <span>{group.title}</span>
            </Space>
          ),
          key: group.key,
          children: group.items.map(item => ({
            title: (
              <Space>
                {statusIcons[item.status]}
                <span>{item.name}</span>
              </Space>
            ),
            key: item.id
          }))
        }))
      }
    ]
  }, [components, statusColors, statusIcons])

  const debugColumns = [
    {
      title: '时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      width: 150,
      render: (time: string) => (
        <div className="text-xs text-gray-600">{time}</div>
      )
    },
    {
      title: '组件',
      dataIndex: 'component',
      key: 'component'
    },
    {
      title: '问题',
      dataIndex: 'issue',
      key: 'issue'
    },
    {
      title: '优先级',
      dataIndex: 'priority',
      key: 'priority',
      render: (priority: keyof typeof priorityColors) => (
        <Tag color={priorityColors[priority]}>
          {priority.toUpperCase()}
        </Tag>
      )
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: keyof typeof statusSessionColors) => (
        <Badge status={statusSessionColors[status]} text={
          status === 'investigating' ? '调查中' :
          status === 'resolved' ? '已解决' : '待处理'
        } />
      )
    }
  ]

  const onTreeSelect = (selectedKeys: React.Key[]) => {
    if (selectedKeys.length > 0) {
      const componentId = selectedKeys[0] as string
      const component = components.find(c => c.id === componentId)
      if (component) {
        setSelectedComponent(component)
      }
    }
  }

  const healthyComponents = components.filter(c => c.status === 'healthy').length
  const warningComponents = components.filter(c => c.status === 'warning').length
  const errorComponents = components.filter(c => c.status === 'error').length
  const responseTimes = components
    .map(c => c.metrics.responseTime)
    .filter((value): value is number => typeof value === 'number')
  const avgResponseTime = responseTimes.length
    ? Math.round((responseTimes.reduce((sum, value) => sum + value, 0) / responseTimes.length) * 10) / 10
    : null

  return (
    <div className="p-6">
        <div className="mb-6">
          <div className="flex justify-between items-center mb-4">
            <h1 className="text-2xl font-bold">架构调试中心</h1>
            <Space>
              <Button icon={<BugOutlined />}>
                开始调试会话
              </Button>
              <Button icon={<ReloadOutlined />} onClick={loadData} loading={loading}>
                刷新状态
              </Button>
              <Button icon={<SettingOutlined />}>
                系统设置
              </Button>
            </Space>
          </div>

          {error && (
            <Alert
              message="架构数据加载失败"
              description={error}
              type="error"
              showIcon
              className="mb-4"
            />
          )}

          <Row gutter={16} className="mb-6">
            <Col span={6}>
              <Card>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-500">{healthyComponents}</div>
                  <div className="text-gray-600">健康组件</div>
                </div>
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <div className="text-center">
                  <div className="text-2xl font-bold text-orange-500">{warningComponents}</div>
                  <div className="text-gray-600">警告组件</div>
                </div>
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <div className="text-center">
                  <div className="text-2xl font-bold text-red-500">{errorComponents}</div>
                  <div className="text-gray-600">错误组件</div>
                </div>
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-500">
                    {avgResponseTime !== null ? `${avgResponseTime}ms` : '-'}
                  </div>
                  <div className="text-gray-600">平均响应时间</div>
                </div>
              </Card>
            </Col>
          </Row>

          {errorComponents > 0 && (
            <Alert
              message="系统告警"
              description={`检测到 ${errorComponents} 个组件存在错误，请及时处理`}
              variant="destructive"
              showIcon
              closable
              className="mb-4"
            />
          )}
        </div>

        <Row gutter={16}>
          <Col span={8}>
            <Card title="系统拓扑" className="mb-4">
              <Tree
                treeData={treeData}
                defaultExpandAll
                onSelect={onTreeSelect}
                className="architecture-tree"
              />
            </Card>
          </Col>

          <Col span={16}>
            <Tabs defaultActiveKey="component">
              <TabPane tab="组件详情" key="component">
                {selectedComponent ? (
                  <Card title={`${selectedComponent.name} 详情`}>
                    <Descriptions column={2} bordered size="small">
                      <Descriptions.Item label="状态">
                        <Space>
                          {statusIcons[selectedComponent.status]}
                          <Tag color={statusColors[selectedComponent.status]}>
                            {selectedComponent.status.toUpperCase()}
                          </Tag>
                        </Space>
                      </Descriptions.Item>
                      <Descriptions.Item label="版本">
                        {selectedComponent.version}
                      </Descriptions.Item>
                      <Descriptions.Item label="类型">
                        <Space>
                          {typeIcons[selectedComponent.type]}
                          {selectedComponent.type}
                        </Space>
                      </Descriptions.Item>
                      <Descriptions.Item label="运行时长">
                        {formatDuration(selectedComponent.metrics.uptime)}
                      </Descriptions.Item>
                      <Descriptions.Item label="响应时间">
                        {selectedComponent.metrics.responseTime !== null ? `${selectedComponent.metrics.responseTime}ms` : '-'}
                      </Descriptions.Item>
                      <Descriptions.Item label="错误率">
                        {selectedComponent.metrics.errorRate !== null ? `${selectedComponent.metrics.errorRate}%` : '-'}
                      </Descriptions.Item>
                      <Descriptions.Item label="依赖" span={2}>
                        <Space wrap>
                          {selectedComponent.dependencies.map(dep => (
                            <Tag key={dep}>{dep}</Tag>
                          ))}
                        </Space>
                      </Descriptions.Item>
                      {selectedComponent.endpoints && (
                        <Descriptions.Item label="端点" span={2}>
                          <Space direction="vertical">
                            {selectedComponent.endpoints.map(endpoint => (
                              <Tag key={endpoint} color="blue">{endpoint}</Tag>
                            ))}
                          </Space>
                        </Descriptions.Item>
                      )}
                    </Descriptions>
                  </Card>
                ) : (
                  <Card>
                    <div className="text-center text-gray-500 py-8">
                      请从左侧选择组件查看详情
                    </div>
                  </Card>
                )}
              </TabPane>

              <TabPane tab="调试会话" key="debug">
                <Card title="调试会话列表">
                  <Table
                    columns={debugColumns}
                    dataSource={debugSessions}
                    rowKey="id"
                    loading={loading}
                    pagination={false}
                    size="small"
                  />
                </Card>
              </TabPane>

              <TabPane tab="系统日志" key="logs">
                <Card title="系统诊断日志">
                  <Timeline
                    items={[
                      {
                        color: 'red',
                        children: (
                          <div>
                            <div className="font-medium">Qdrant连接失败</div>
                            <div className="text-xs text-gray-500">2024-01-15 14:30:00</div>
                            <div className="text-sm">连接超时，正在尝试重连...</div>
                          </div>
                        )
                      },
                      {
                        color: 'blue',
                        children: (
                          <div>
                            <div className="font-medium">智能体服务启动</div>
                            <div className="text-xs text-gray-500">2024-01-15 14:25:00</div>
                            <div className="text-sm">服务已成功启动，等待请求</div>
                          </div>
                        )
                      },
                      {
                        color: 'green',
                        children: (
                          <div>
                            <div className="font-medium">系统健康检查</div>
                            <div className="text-xs text-gray-500">2024-01-15 14:20:00</div>
                            <div className="text-sm">所有核心服务运行正常</div>
                          </div>
                        )
                      }
                    ]}
                  />
                </Card>
              </TabPane>
            </Tabs>
          </Col>
        </Row>
    </div>
  )
}

export default ArchitectureDebugPage
