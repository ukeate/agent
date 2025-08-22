import React, { useState, useEffect } from 'react'
import { 
  Card, 
  Row, 
  Col, 
  Button, 
  Space, 
  Table, 
  Progress,
  Tag,
  Statistic,
  Alert,
  Typography,
  Divider,
  Tabs,
  List,
  Timeline,
  Badge,
  Tooltip,
  Switch
} from 'antd'
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Legend, 
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell
} from 'recharts'
import {
  HeartOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  CloseCircleOutlined,
  ReloadOutlined,
  MonitorOutlined,
  ServerOutlined,
  DatabaseOutlined,
  ApiOutlined,
  CloudOutlined,
  SecurityScanOutlined,
  DashboardOutlined,
  WarningOutlined
} from '@ant-design/icons'

const { Title, Text, Paragraph } = Typography
const { TabPane } = Tabs

// 健康检查状态
enum HealthStatus {
  HEALTHY = 'healthy',
  WARNING = 'warning',
  CRITICAL = 'critical',
  DOWN = 'down'
}

// 生成健康监控数据
const generateHealthData = () => {
  const services = [
    { name: 'API Gateway', endpoint: '/api/health', critical: true },
    { name: 'Authentication Service', endpoint: '/api/auth/health', critical: true },
    { name: 'Database (PostgreSQL)', endpoint: '/health/db', critical: true },
    { name: 'Redis Cache', endpoint: '/health/redis', critical: true },
    { name: 'Vector Database (Qdrant)', endpoint: '/health/vector', critical: true },
    { name: 'Q-Learning Service', endpoint: '/api/qlearning/health', critical: false },
    { name: 'RAG Service', endpoint: '/api/rag/health', critical: false },
    { name: 'Multi-Agent System', endpoint: '/api/multi-agent/health', critical: false },
    { name: 'File Service', endpoint: '/api/files/health', critical: false },
    { name: 'Workflow Engine', endpoint: '/api/workflows/health', critical: false }
  ]

  return services.map((service, index) => {
    const rand = Math.random()
    let status: HealthStatus
    let responseTime = Math.floor(Math.random() * 200) + 10
    
    if (rand < 0.7) {
      status = HealthStatus.HEALTHY
    } else if (rand < 0.85) {
      status = HealthStatus.WARNING
      responseTime = Math.floor(Math.random() * 500) + 200
    } else if (rand < 0.95) {
      status = HealthStatus.CRITICAL  
      responseTime = Math.floor(Math.random() * 1000) + 500
    } else {
      status = HealthStatus.DOWN
      responseTime = 0
    }

    return {
      id: index + 1,
      name: service.name,
      endpoint: service.endpoint,
      status,
      responseTime,
      critical: service.critical,
      uptime: Math.random() * 100,
      lastCheck: new Date(Date.now() - Math.random() * 300000),
      errorCount: status === HealthStatus.HEALTHY ? 0 : Math.floor(Math.random() * 10),
      version: `v${Math.floor(Math.random() * 3) + 1}.${Math.floor(Math.random() * 10)}.${Math.floor(Math.random() * 10)}`
    }
  })
}

// 生成系统指标数据
const generateSystemMetrics = () => {
  const data = []
  
  for (let i = 0; i < 24; i++) {
    data.push({
      time: new Date(Date.now() - (24 - i) * 3600000),
      cpuUsage: Math.random() * 80 + 10,
      memoryUsage: Math.random() * 90 + 5,
      diskUsage: Math.random() * 70 + 20,
      networkIn: Math.random() * 1000,
      networkOut: Math.random() * 800,
      activeConnections: Math.floor(Math.random() * 1000) + 100,
      requestRate: Math.floor(Math.random() * 5000) + 1000
    })
  }
  
  return data
}

// 生成告警数据
const generateAlerts = () => [
  {
    id: 1,
    level: 'critical',
    service: 'Database Connection Pool',
    message: '数据库连接池使用率过高 (95%)',
    timestamp: new Date(Date.now() - 300000),
    resolved: false
  },
  {
    id: 2,
    level: 'warning',
    service: 'Q-Learning Service',
    message: '响应时间超过阈值 (>500ms)',
    timestamp: new Date(Date.now() - 600000),
    resolved: false
  },
  {
    id: 3,
    level: 'info',
    service: 'Redis Cache',
    message: '缓存命中率下降至85%',
    timestamp: new Date(Date.now() - 900000),
    resolved: true
  },
  {
    id: 4,
    level: 'warning',
    service: 'File Service',
    message: '磁盘空间使用率达到80%',
    timestamp: new Date(Date.now() - 1200000),
    resolved: false
  }
]

const HealthComprehensivePage: React.FC = () => {
  const [healthData, setHealthData] = useState(() => generateHealthData())
  const [systemMetrics] = useState(() => generateSystemMetrics())
  const [alerts] = useState(() => generateAlerts())
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [refreshing, setRefreshing] = useState(false)

  // 自动刷新
  useEffect(() => {
    let interval: NodeJS.Timeout | null = null
    
    if (autoRefresh) {
      interval = setInterval(() => {
        setHealthData(generateHealthData())
      }, 30000) // 30秒刷新一次
    }
    
    return () => {
      if (interval) clearInterval(interval)
    }
  }, [autoRefresh])

  // 手动刷新
  const handleRefresh = async () => {
    setRefreshing(true)
    await new Promise(resolve => setTimeout(resolve, 1000))
    setHealthData(generateHealthData())
    setRefreshing(false)
  }

  // 计算整体状态
  const overallStatus = () => {
    const criticalServices = healthData.filter(s => s.critical)
    const downCritical = criticalServices.filter(s => s.status === HealthStatus.DOWN).length
    const criticalIssues = criticalServices.filter(s => s.status === HealthStatus.CRITICAL).length
    const warningIssues = criticalServices.filter(s => s.status === HealthStatus.WARNING).length
    
    if (downCritical > 0) return HealthStatus.DOWN
    if (criticalIssues > 0) return HealthStatus.CRITICAL
    if (warningIssues > 0) return HealthStatus.WARNING
    return HealthStatus.HEALTHY
  }

  const getStatusColor = (status: HealthStatus): string => {
    const colors = {
      [HealthStatus.HEALTHY]: '#52c41a',
      [HealthStatus.WARNING]: '#faad14', 
      [HealthStatus.CRITICAL]: '#ff7875',
      [HealthStatus.DOWN]: '#ff4d4f'
    }
    return colors[status]
  }

  const getStatusText = (status: HealthStatus): string => {
    const texts = {
      [HealthStatus.HEALTHY]: '健康',
      [HealthStatus.WARNING]: '警告',
      [HealthStatus.CRITICAL]: '严重',
      [HealthStatus.DOWN]: '离线'
    }
    return texts[status]
  }

  // 整体状态卡片
  const OverallStatusCard = () => {
    const status = overallStatus()
    const healthyCount = healthData.filter(s => s.status === HealthStatus.HEALTHY).length
    const totalCount = healthData.length
    
    return (
      <Card>
        <Row align="middle">
          <Col span={6}>
            <div style={{ textAlign: 'center' }}>
              <HeartOutlined 
                style={{ 
                  fontSize: 48, 
                  color: getStatusColor(status) 
                }} 
              />
              <div style={{ marginTop: 8 }}>
                <Text strong style={{ color: getStatusColor(status) }}>
                  系统{getStatusText(status)}
                </Text>
              </div>
            </div>
          </Col>
          <Col span={18}>
            <Row gutter={16}>
              <Col span={6}>
                <Statistic
                  title="健康服务"
                  value={healthyCount}
                  suffix={`/ ${totalCount}`}
                  valueStyle={{ color: '#52c41a' }}
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="平均响应时间"
                  value={Math.round(healthData.reduce((sum, s) => sum + s.responseTime, 0) / healthData.length)}
                  suffix="ms"
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="系统正常运行时间"
                  value="99.9"
                  suffix="%"
                  precision={1}
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="活跃告警"
                  value={alerts.filter(a => !a.resolved).length}
                  valueStyle={{ color: '#ff4d4f' }}
                />
              </Col>
            </Row>
          </Col>
        </Row>
      </Card>
    )
  }

  // 服务状态表格
  const ServiceStatusTable = () => {
    const columns = [
      {
        title: '服务名称',
        dataIndex: 'name',
        key: 'name',
        render: (name: string, record: any) => (
          <Space>
            <Badge 
              status={record.status === HealthStatus.HEALTHY ? 'success' : 
                     record.status === HealthStatus.WARNING ? 'warning' :
                     record.status === HealthStatus.CRITICAL ? 'error' : 'default'} 
            />
            <Text strong={record.critical}>{name}</Text>
            {record.critical && <Tag color="red" size="small">核心</Tag>}
          </Space>
        ),
      },
      {
        title: '状态',
        dataIndex: 'status',
        key: 'status',
        render: (status: HealthStatus) => (
          <Tag color={getStatusColor(status)}>{getStatusText(status)}</Tag>
        ),
      },
      {
        title: '响应时间',
        dataIndex: 'responseTime',
        key: 'responseTime',
        render: (time: number) => (
          <Text style={{ color: time > 500 ? '#ff4d4f' : time > 200 ? '#faad14' : '#52c41a' }}>
            {time > 0 ? `${time}ms` : 'N/A'}
          </Text>
        ),
      },
      {
        title: '正常运行时间',
        dataIndex: 'uptime',
        key: 'uptime',
        render: (uptime: number) => (
          <Progress 
            percent={uptime} 
            size="small" 
            format={percent => `${percent?.toFixed(1)}%`}
            strokeColor={uptime > 99 ? '#52c41a' : uptime > 95 ? '#faad14' : '#ff4d4f'}
          />
        ),
      },
      {
        title: '错误计数',
        dataIndex: 'errorCount',
        key: 'errorCount',
        render: (count: number) => (
          <Text style={{ color: count > 0 ? '#ff4d4f' : '#52c41a' }}>
            {count}
          </Text>
        ),
      },
      {
        title: '版本',
        dataIndex: 'version',
        key: 'version',
      },
      {
        title: '最后检查',
        dataIndex: 'lastCheck',
        key: 'lastCheck',
        render: (time: Date) => (
          <Tooltip title={time.toLocaleString()}>
            <Text type="secondary">{time.toLocaleTimeString()}</Text>
          </Tooltip>
        ),
      }
    ]

    return (
      <Card 
        title="服务状态详情" 
        size="small"
        extra={
          <Space>
            <Switch 
              checked={autoRefresh}
              onChange={setAutoRefresh}
              checkedChildren="自动刷新"
              unCheckedChildren="手动刷新"
            />
            <Button 
              icon={<ReloadOutlined />}
              loading={refreshing}
              onClick={handleRefresh}
            >
              刷新
            </Button>
          </Space>
        }
      >
        <Table
          columns={columns}
          dataSource={healthData}
          rowKey="id"
          size="small"
          pagination={false}
          scroll={{ x: 1000 }}
        />
      </Card>
    )
  }

  // 系统指标图表
  const SystemMetricsChart = () => (
    <Card title="系统性能指标 (24小时)" size="small">
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={systemMetrics}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="time" 
            tickFormatter={(time) => new Date(time).getHours() + ':00'}
          />
          <YAxis />
          <Tooltip 
            labelFormatter={(time) => new Date(time).toLocaleString()}
            formatter={(value: number, name: string) => [
              name === 'cpuUsage' || name === 'memoryUsage' || name === 'diskUsage' ? 
                `${value.toFixed(1)}%` : `${value.toFixed(0)}`,
              name === 'cpuUsage' ? 'CPU使用率' :
              name === 'memoryUsage' ? '内存使用率' :
              name === 'diskUsage' ? '磁盘使用率' :
              name === 'networkIn' ? '网络流入' :
              name === 'networkOut' ? '网络流出' :
              name === 'activeConnections' ? '活跃连接' :
              name === 'requestRate' ? '请求速率' : name
            ]}
          />
          <Legend />
          <Line type="monotone" dataKey="cpuUsage" stroke="#1890ff" name="CPU使用率" />
          <Line type="monotone" dataKey="memoryUsage" stroke="#52c41a" name="内存使用率" />
          <Line type="monotone" dataKey="diskUsage" stroke="#fa8c16" name="磁盘使用率" />
        </LineChart>
      </ResponsiveContainer>
    </Card>
  )

  // 网络流量图表
  const NetworkTrafficChart = () => (
    <Card title="网络流量" size="small">
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={systemMetrics}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="time" 
            tickFormatter={(time) => new Date(time).getHours() + ':00'}
          />
          <YAxis />
          <Tooltip 
            labelFormatter={(time) => new Date(time).toLocaleString()}
            formatter={(value: number) => [`${value.toFixed(0)} MB/s`]}
          />
          <Legend />
          <Area 
            type="monotone" 
            dataKey="networkIn" 
            stackId="1" 
            stroke="#1890ff" 
            fill="#1890ff" 
            name="流入"
            fillOpacity={0.6}
          />
          <Area 
            type="monotone" 
            dataKey="networkOut" 
            stackId="1" 
            stroke="#52c41a" 
            fill="#52c41a" 
            name="流出"
            fillOpacity={0.6}
          />
        </AreaChart>
      </ResponsiveContainer>
    </Card>
  )

  // 告警时间线
  const AlertsTimeline = () => (
    <Card title="系统告警" size="small">
      <Timeline>
        {alerts.map(alert => (
          <Timeline.Item
            key={alert.id}
            color={
              alert.level === 'critical' ? 'red' :
              alert.level === 'warning' ? 'orange' : 'blue'
            }
            dot={
              alert.level === 'critical' ? <CloseCircleOutlined /> :
              alert.level === 'warning' ? <ExclamationCircleOutlined /> :
              <CheckCircleOutlined />
            }
          >
            <div>
              <Space>
                <Text strong>{alert.service}</Text>
                {!alert.resolved && (
                  <Tag color={alert.level === 'critical' ? 'red' : 'orange'}>
                    未解决
                  </Tag>
                )}
              </Space>
              <div>
                <Text>{alert.message}</Text>
              </div>
              <div>
                <Text type="secondary">{alert.timestamp.toLocaleString()}</Text>
              </div>
            </div>
          </Timeline.Item>
        ))}
      </Timeline>
    </Card>
  )

  // 服务依赖拓扑
  const ServiceTopology = () => (
    <Card title="服务依赖拓扑" size="small">
      <div style={{ 
        height: 300, 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center',
        border: '1px dashed #d9d9d9',
        borderRadius: '6px'
      }}>
        <Space direction="vertical" align="center">
          <ApiOutlined style={{ fontSize: 48, color: '#1890ff' }} />
          <Text type="secondary">服务依赖关系图</Text>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            显示各服务间的调用关系和依赖状态
          </Text>
        </Space>
      </div>
    </Card>
  )

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>
        <HeartOutlined /> 全面健康检查
      </Title>
      <Paragraph type="secondary">
        实时监控系统各个组件的健康状态，包括服务可用性、性能指标、系统资源使用情况和告警信息
      </Paragraph>
      
      <Divider />

      <Tabs defaultActiveKey="1">
        <TabPane tab="总览仪表板" key="1">
          <Space direction="vertical" size="large" style={{ width: '100%' }}>
            <OverallStatusCard />

            <Row gutter={16}>
              <Col span={16}>
                <SystemMetricsChart />
              </Col>
              <Col span={8}>
                <AlertsTimeline />
              </Col>
            </Row>

            <ServiceStatusTable />
          </Space>
        </TabPane>

        <TabPane tab="系统指标" key="2">
          <Row gutter={16}>
            <Col span={12}>
              <SystemMetricsChart />
            </Col>
            <Col span={12}>
              <NetworkTrafficChart />
            </Col>
          </Row>
          
          <div style={{ marginTop: 16 }}>
            <Row gutter={16}>
              <Col span={8}>
                <Card title="CPU & 内存" size="small">
                  <Row gutter={16}>
                    <Col span={12}>
                      <Statistic 
                        title="CPU使用率" 
                        value={75.3} 
                        suffix="%" 
                        precision={1}
                        valueStyle={{ color: '#fa8c16' }}
                      />
                    </Col>
                    <Col span={12}>
                      <Statistic 
                        title="内存使用率" 
                        value={68.7} 
                        suffix="%" 
                        precision={1}
                        valueStyle={{ color: '#52c41a' }}
                      />
                    </Col>
                  </Row>
                </Card>
              </Col>
              <Col span={8}>
                <Card title="存储 & 网络" size="small">
                  <Row gutter={16}>
                    <Col span={12}>
                      <Statistic 
                        title="磁盘使用率" 
                        value={45.2} 
                        suffix="%" 
                        precision={1}
                      />
                    </Col>
                    <Col span={12}>
                      <Statistic 
                        title="网络延迟" 
                        value={12.5} 
                        suffix="ms" 
                        precision={1}
                      />
                    </Col>
                  </Row>
                </Card>
              </Col>
              <Col span={8}>
                <Card title="连接 & 请求" size="small">
                  <Row gutter={16}>
                    <Col span={12}>
                      <Statistic 
                        title="活跃连接" 
                        value={847} 
                      />
                    </Col>
                    <Col span={12}>
                      <Statistic 
                        title="请求速率" 
                        value={3240} 
                        suffix="/min"
                      />
                    </Col>
                  </Row>
                </Card>
              </Col>
            </Row>
          </div>
        </TabPane>

        <TabPane tab="服务拓扑" key="3">
          <Row gutter={16}>
            <Col span={16}>
              <ServiceTopology />
            </Col>
            <Col span={8}>
              <Card title="关键服务状态" size="small">
                <List
                  dataSource={healthData.filter(s => s.critical)}
                  renderItem={service => (
                    <List.Item>
                      <List.Item.Meta
                        avatar={
                          <Badge 
                            status={
                              service.status === HealthStatus.HEALTHY ? 'success' : 
                              service.status === HealthStatus.WARNING ? 'warning' :
                              service.status === HealthStatus.CRITICAL ? 'error' : 'default'
                            }
                          />
                        }
                        title={service.name}
                        description={`${service.responseTime}ms • ${getStatusText(service.status)}`}
                      />
                    </List.Item>
                  )}
                />
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab="历史监控" key="4">
          <Card title="24小时健康状态历史" size="small">
            <Alert
              message="历史监控数据"
              description="显示过去24小时内各服务的健康状态变化趋势和关键事件"
              variant="default"
              style={{ marginBottom: 16 }}
            />
            <div style={{ 
              height: 400, 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center',
              border: '1px dashed #d9d9d9',
              borderRadius: '6px'
            }}>
              <Space direction="vertical" align="center">
                <DashboardOutlined style={{ fontSize: 48, color: '#1890ff' }} />
                <Text type="secondary">历史监控图表区域</Text>
                <Text type="secondary" style={{ fontSize: '12px' }}>
                  展示服务可用性时间线、性能趋势和关键指标变化
                </Text>
              </Space>
            </div>
          </Card>
        </TabPane>
      </Tabs>
    </div>
  )
}

export default HealthComprehensivePage