import React, { useState } from 'react'
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
  Timeline,
  Select,
  Input
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
    uptime: number
    responseTime: number
    errorRate: number
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
  const [components] = useState<Component[]>([
    {
      id: '1',
      name: 'Web前端',
      type: 'ui',
      status: 'healthy',
      version: '1.0.0',
      dependencies: ['api-gateway'],
      metrics: { uptime: 99.9, responseTime: 120, errorRate: 0.1 }
    },
    {
      id: '2', 
      name: 'API网关',
      type: 'api',
      status: 'healthy',
      version: '2.1.0',
      dependencies: ['agent-service', 'rag-service'],
      endpoints: ['/api/agents', '/api/rag', '/api/workflows'],
      metrics: { uptime: 99.8, responseTime: 85, errorRate: 0.2 }
    },
    {
      id: '3',
      name: '智能体服务',
      type: 'service',
      status: 'warning',
      version: '1.5.2',
      dependencies: ['postgresql', 'redis'],
      metrics: { uptime: 98.5, responseTime: 250, errorRate: 1.5 }
    },
    {
      id: '4',
      name: 'RAG服务',
      type: 'service', 
      status: 'healthy',
      version: '1.3.1',
      dependencies: ['qdrant', 'postgresql'],
      metrics: { uptime: 99.5, responseTime: 180, errorRate: 0.3 }
    },
    {
      id: '5',
      name: 'PostgreSQL',
      type: 'database',
      status: 'healthy',
      version: '15.2',
      dependencies: [],
      metrics: { uptime: 99.9, responseTime: 45, errorRate: 0.0 }
    },
    {
      id: '6',
      name: 'Qdrant向量库',
      type: 'database',
      status: 'error',
      version: '1.7.0',
      dependencies: [],
      metrics: { uptime: 95.2, responseTime: 350, errorRate: 3.2 }
    },
    {
      id: '7',
      name: 'Redis缓存',
      type: 'database',
      status: 'healthy',
      version: '7.0.8',
      dependencies: [],
      metrics: { uptime: 99.7, responseTime: 15, errorRate: 0.1 }
    }
  ])

  const [debugSessions] = useState<DebugSession[]>([
    {
      id: '1',
      timestamp: '2024-01-15 14:30:00',
      component: 'Qdrant向量库',
      issue: '连接超时频繁发生',
      status: 'investigating',
      priority: 'high'
    },
    {
      id: '2',
      timestamp: '2024-01-15 13:45:00', 
      component: '智能体服务',
      issue: '响应时间过长',
      status: 'pending',
      priority: 'medium'
    },
    {
      id: '3',
      timestamp: '2024-01-15 12:20:00',
      component: 'API网关',
      issue: '负载均衡配置问题',
      status: 'resolved',
      priority: 'low'
    }
  ])

  const [selectedComponent, setSelectedComponent] = useState<Component | null>(components[0])

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

  const treeData = [
    {
      title: (
        <Space>
          <NodeIndexOutlined />
          <span>系统架构</span>
        </Space>
      ),
      key: 'root',
      children: [
        {
          title: (
            <Space>
              <Badge status={statusColors.healthy} />
              <span>前端层</span>
            </Space>
          ),
          key: 'frontend',
          children: [
            {
              title: (
                <Space>
                  {statusIcons.healthy}
                  <span>Web前端</span>
                </Space>
              ),
              key: '1'
            }
          ]
        },
        {
          title: (
            <Space>
              <Badge status={statusColors.healthy} />
              <span>API层</span>
            </Space>
          ),
          key: 'api',
          children: [
            {
              title: (
                <Space>
                  {statusIcons.healthy}
                  <span>API网关</span>
                </Space>
              ),
              key: '2'
            }
          ]
        },
        {
          title: (
            <Space>
              <Badge status={statusColors.warning} />
              <span>服务层</span>
            </Space>
          ),
          key: 'services',
          children: [
            {
              title: (
                <Space>
                  {statusIcons.warning}
                  <span>智能体服务</span>
                </Space>
              ),
              key: '3'
            },
            {
              title: (
                <Space>
                  {statusIcons.healthy}
                  <span>RAG服务</span>
                </Space>
              ),
              key: '4'
            }
          ]
        },
        {
          title: (
            <Space>
              <Badge status={statusColors.error} />
              <span>数据层</span>
            </Space>
          ),
          key: 'data',
          children: [
            {
              title: (
                <Space>
                  {statusIcons.healthy}
                  <span>PostgreSQL</span>
                </Space>
              ),
              key: '5'
            },
            {
              title: (
                <Space>
                  {statusIcons.error}
                  <span>Qdrant向量库</span>
                </Space>
              ),
              key: '6'
            },
            {
              title: (
                <Space>
                  {statusIcons.healthy}
                  <span>Redis缓存</span>
                </Space>
              ),
              key: '7'
            }
          ]
        }
      ]
    }
  ]

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
  const avgUptime = Math.round(components.reduce((sum, c) => sum + c.metrics.uptime, 0) / components.length * 10) / 10

  return (
    <div className="p-6">
        <div className="mb-6">
          <div className="flex justify-between items-center mb-4">
            <h1 className="text-2xl font-bold">架构调试中心</h1>
            <Space>
              <Button icon={<BugOutlined />}>
                开始调试会话
              </Button>
              <Button icon={<ReloadOutlined />}>
                刷新状态
              </Button>
              <Button icon={<SettingOutlined />}>
                系统设置
              </Button>
            </Space>
          </div>

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
                  <div className="text-2xl font-bold text-blue-500">{avgUptime}%</div>
                  <div className="text-gray-600">平均可用性</div>
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
                      <Descriptions.Item label="可用性">
                        {selectedComponent.metrics.uptime}%
                      </Descriptions.Item>
                      <Descriptions.Item label="响应时间">
                        {selectedComponent.metrics.responseTime}ms
                      </Descriptions.Item>
                      <Descriptions.Item label="错误率">
                        {selectedComponent.metrics.errorRate}%
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