import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Statistic,
  Progress,
  Table,
  Badge,
  Alert,
  Space,
  Button,
  Tabs,
  List,
  Tag,
  Timeline,
  Descriptions,
  Select,
  Switch,
  message,
  Tooltip,
} from 'antd'
import {
  DashboardOutlined,
  MonitorOutlined,
  ClusterOutlined,
  DatabaseOutlined,
  ApiOutlined,
  CloudServerOutlined,
  AlertOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  SyncOutlined,
  ReloadOutlined,
  WarningOutlined,
  ThunderboltOutlined,
  RocketOutlined,
  FieldTimeOutlined,
} from '@ant-design/icons'
import ReactECharts from 'echarts-for-react'
import apiClient from '../services/apiClient'

const { TabPane } = Tabs

interface ServiceHealth {
  name: string
  status: 'healthy' | 'degraded' | 'unhealthy'
  latency: number
  uptime: number
  lastCheck: string
}

interface MetricData {
  timestamp: string
  value?: number
  cpu?: number
  memory?: number
  disk?: number
}

interface SystemMetrics {
  cpu_usage: number
  memory_usage: number
  disk_usage: number
  network_io: number
  active_connections: number
  request_rate: number
}

interface AlertItem {
  id: string
  level: 'info' | 'warning' | 'error' | 'critical'
  message: string
  timestamp: string
  service: string
  resolved: boolean
}

const MonitoringDashboardPage: React.FC = () => {
  const [services, setServices] = useState<ServiceHealth[]>([])
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null)
  const [alerts, setAlerts] = useState<AlertItem[]>([])
  const [performanceData, setPerformanceData] = useState<MetricData[]>([])
  const [requestDistribution, setRequestDistribution] = useState<
    { name: string; value: number }[]
  >([])
  const [loading, setLoading] = useState(false)
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [refreshInterval, setRefreshInterval] = useState(5000)

  useEffect(() => {
    fetchDashboardData()

    if (autoRefresh) {
      const interval = setInterval(fetchDashboardData, refreshInterval)
      return () => clearInterval(interval)
    }
  }, [autoRefresh, refreshInterval])

  // 获取仪表板数据
  const fetchDashboardData = async () => {
    setLoading(true)
    try {
      // 获取服务健康状态
      const healthResponse = await apiClient.get('/health')

      // 获取系统指标
      const metricsResponse = await apiClient.get('/metrics')

      // 获取告警信息
      const alertsResponse = await apiClient.get('/alerts')

      // 获取性能数据
      const perfResponse = await apiClient
        .get('/metrics/performance')
        .catch(() => ({ data: [] }))
      const distributionResponse = await apiClient
        .get('/metrics/request-distribution')
        .catch(() => ({ data: { distribution: [] } }))

      // 处理响应数据
      processHealthData(healthResponse.data)
      processMetricsData(metricsResponse.data)
      processAlertsData(alertsResponse.data)
      setPerformanceData(
        Array.isArray(perfResponse.data) ? perfResponse.data : []
      )
      setRequestDistribution(
        Array.isArray(distributionResponse.data?.distribution)
          ? distributionResponse.data.distribution
          : []
      )
    } catch (error) {
      message.error('获取监控数据失败')
      setServices([])
      setMetrics(null)
      setAlerts([])
      setPerformanceData([])
      setRequestDistribution([])
    } finally {
      setLoading(false)
    }
  }

  // 处理健康数据
  const processHealthData = (data: any) => {
    const healthServices: ServiceHealth[] = Object.entries(
      data.components || {}
    ).map(([name, info]: any) => ({
      name,
      status: info.status || 'healthy',
      latency: info.response_time_ms || 0,
      uptime: info.uptime_percent || 0,
      lastCheck: info.last_check || new Date().toISOString(),
    }))
    setServices(healthServices)
  }

  // 处理指标数据
  const processMetricsData = (data: any) => {
    setMetrics({
      cpu_usage: data.cpu?.usage || 0,
      memory_usage: data.memory?.usage || 0,
      disk_usage: data.disk?.usage || 0,
      network_io: data.network?.throughput || 0,
      active_connections: data.connections || 0,
      request_rate: data.requests?.rate || 0,
    })
  }

  // 处理告警数据
  const processAlertsData = (data: any) => {
    setAlerts(Array.isArray(data.alerts) ? data.alerts : [])
  }

  // 获取性能图表选项
  const getPerformanceChartOption = () => {
    return {
      title: { text: '系统性能趋势', left: 'center' },
      tooltip: { trigger: 'axis' },
      xAxis: {
        type: 'category',
        data: performanceData.map(d =>
          new Date(d.timestamp).toLocaleTimeString()
        ),
      },
      yAxis: { type: 'value', name: '使用率 (%)' },
      series: [
        {
          name: 'CPU',
          type: 'line',
          data: performanceData.map(d => d.cpu ?? d.value ?? 0),
          smooth: true,
        },
        {
          name: '内存',
          type: 'line',
          data: performanceData.map(d => d.memory ?? d.value ?? 0),
          smooth: true,
        },
        {
          name: '磁盘',
          type: 'line',
          data: performanceData.map(d => d.disk ?? d.value ?? 0),
          smooth: true,
        },
      ],
      legend: { bottom: 0 },
    }
  }

  // 获取请求分布图表选项
  const getRequestDistributionOption = () => {
    return {
      title: { text: 'API请求分布', left: 'center' },
      tooltip: { trigger: 'item' },
      series: [
        {
          type: 'pie',
          radius: ['40%', '70%'],
          data: requestDistribution,
        },
      ],
    }
  }

  // 服务健康表格列
  const serviceColumns = [
    {
      title: '服务名称',
      dataIndex: 'name',
      key: 'name',
      render: (text: string) => (
        <Space>
          <DatabaseOutlined />
          {text}
        </Space>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const statusConfig: any = {
          healthy: {
            color: 'success',
            icon: <CheckCircleOutlined />,
            text: '健康',
          },
          degraded: {
            color: 'warning',
            icon: <WarningOutlined />,
            text: '降级',
          },
          unhealthy: {
            color: 'error',
            icon: <CloseCircleOutlined />,
            text: '异常',
          },
        }
        const config = statusConfig[status]
        return (
          <Badge
            status={config.color}
            text={
              <Space>
                {config.icon}
                {config.text}
              </Space>
            }
          />
        )
      },
    },
    {
      title: '延迟',
      dataIndex: 'latency',
      key: 'latency',
      render: (latency: number) => (
        <Tag color={latency < 50 ? 'green' : latency < 100 ? 'orange' : 'red'}>
          {latency}ms
        </Tag>
      ),
    },
    {
      title: '可用性',
      dataIndex: 'uptime',
      key: 'uptime',
      render: (uptime: number) => (
        <Progress
          percent={uptime}
          size="small"
          status={
            uptime > 99 ? 'success' : uptime > 95 ? 'normal' : 'exception'
          }
        />
      ),
    },
    {
      title: '最后检查',
      dataIndex: 'lastCheck',
      key: 'lastCheck',
      render: (time: string) => new Date(time).toLocaleTimeString(),
    },
  ]

  // 告警列表渲染
  const renderAlert = (alert: AlertItem) => {
    const levelConfig: any = {
      info: { color: 'blue', icon: <AlertOutlined /> },
      warning: { color: 'orange', icon: <WarningOutlined /> },
      error: { color: 'red', icon: <CloseCircleOutlined /> },
      critical: { color: 'red', icon: <ThunderboltOutlined /> },
    }
    const config = levelConfig[alert.level]

    return (
      <List.Item>
        <List.Item.Meta
          avatar={config.icon}
          title={
            <Space>
              <Tag color={config.color}>{alert.level.toUpperCase()}</Tag>
              <span>{alert.service}</span>
              {alert.resolved && <Tag color="green">已解决</Tag>}
            </Space>
          }
          description={
            <div>
              <div>{alert.message}</div>
              <div style={{ fontSize: '12px', color: '#999' }}>
                {new Date(alert.timestamp).toLocaleString()}
              </div>
            </div>
          }
        />
      </List.Item>
    )
  }

  return (
    <div style={{ padding: '24px' }}>
      <Card
        title={
          <Space>
            <DashboardOutlined />
            <span>监控仪表板</span>
          </Space>
        }
        extra={
          <Space>
            <span>自动刷新</span>
            <Switch checked={autoRefresh} onChange={setAutoRefresh} />
            <Select
              value={refreshInterval}
              onChange={setRefreshInterval}
              style={{ width: 100 }}
            >
              <Select.Option value={5000}>5秒</Select.Option>
              <Select.Option value={10000}>10秒</Select.Option>
              <Select.Option value={30000}>30秒</Select.Option>
            </Select>
            <Button icon={<ReloadOutlined />} onClick={fetchDashboardData}>
              刷新
            </Button>
          </Space>
        }
      >
        {/* 关键指标卡片 */}
        <Row gutter={16} style={{ marginBottom: 16 }}>
          <Col span={4}>
            <Card>
              <Statistic
                title="CPU使用率"
                value={metrics?.cpu_usage || 0}
                precision={1}
                suffix="%"
                valueStyle={{
                  color: metrics?.cpu_usage! > 80 ? '#cf1322' : '#3f8600',
                }}
                prefix={<MonitorOutlined />}
              />
            </Card>
          </Col>
          <Col span={4}>
            <Card>
              <Statistic
                title="内存使用率"
                value={metrics?.memory_usage || 0}
                precision={1}
                suffix="%"
                valueStyle={{
                  color: metrics?.memory_usage! > 80 ? '#cf1322' : '#3f8600',
                }}
                prefix={<CloudServerOutlined />}
              />
            </Card>
          </Col>
          <Col span={4}>
            <Card>
              <Statistic
                title="磁盘使用率"
                value={metrics?.disk_usage || 0}
                precision={1}
                suffix="%"
                prefix={<DatabaseOutlined />}
              />
            </Card>
          </Col>
          <Col span={4}>
            <Card>
              <Statistic
                title="活动连接"
                value={metrics?.active_connections || 0}
                prefix={<ClusterOutlined />}
              />
            </Card>
          </Col>
          <Col span={4}>
            <Card>
              <Statistic
                title="请求速率"
                value={metrics?.request_rate || 0}
                suffix="/s"
                prefix={<ApiOutlined />}
              />
            </Card>
          </Col>
          <Col span={4}>
            <Card>
              <Statistic
                title="网络I/O"
                value={metrics?.network_io || 0}
                precision={1}
                suffix="MB/s"
                prefix={<RocketOutlined />}
              />
            </Card>
          </Col>
        </Row>

        <Tabs defaultActiveKey="overview">
          <TabPane tab="总览" key="overview">
            <Row gutter={16}>
              <Col span={12}>
                <Card title="性能趋势">
                  <ReactECharts
                    option={getPerformanceChartOption()}
                    style={{ height: 300 }}
                  />
                </Card>
              </Col>
              <Col span={12}>
                <Card title="请求分布">
                  <ReactECharts
                    option={getRequestDistributionOption()}
                    style={{ height: 300 }}
                  />
                </Card>
              </Col>
            </Row>
          </TabPane>

          <TabPane tab="服务健康" key="services">
            <Card>
              <Table
                dataSource={services}
                columns={serviceColumns}
                rowKey="name"
                pagination={false}
                loading={loading}
              />
            </Card>
          </TabPane>

          <TabPane tab="告警中心" key="alerts">
            <Card
              title={`活动告警 (${alerts.filter(a => !a.resolved).length})`}
              extra={
                <Space>
                  <Tag color="blue">
                    信息: {alerts.filter(a => a.level === 'info').length}
                  </Tag>
                  <Tag color="orange">
                    警告: {alerts.filter(a => a.level === 'warning').length}
                  </Tag>
                  <Tag color="red">
                    错误: {alerts.filter(a => a.level === 'error').length}
                  </Tag>
                </Space>
              }
            >
              <List
                dataSource={alerts}
                renderItem={renderAlert}
                loading={loading}
              />
            </Card>
          </TabPane>

          <TabPane tab="系统信息" key="system">
            <Row gutter={16}>
              <Col span={12}>
                <Card title="环境信息">
                  <Descriptions column={1}>
                    <Descriptions.Item label="环境">
                      Production
                    </Descriptions.Item>
                    <Descriptions.Item label="版本">v1.0.0</Descriptions.Item>
                    <Descriptions.Item label="部署时间">
                      {new Date().toLocaleDateString()}
                    </Descriptions.Item>
                    <Descriptions.Item label="运行时长">
                      15天 6小时
                    </Descriptions.Item>
                  </Descriptions>
                </Card>
              </Col>
              <Col span={12}>
                <Card title="资源配置">
                  <Descriptions column={1}>
                    <Descriptions.Item label="CPU核心">16</Descriptions.Item>
                    <Descriptions.Item label="内存">64GB</Descriptions.Item>
                    <Descriptions.Item label="磁盘">1TB SSD</Descriptions.Item>
                    <Descriptions.Item label="网络">10Gbps</Descriptions.Item>
                  </Descriptions>
                </Card>
              </Col>
            </Row>

            <Card title="服务依赖" style={{ marginTop: 16 }}>
              <Timeline>
                <Timeline.Item color="green">
                  PostgreSQL - 主数据库 (版本: 15.2)
                </Timeline.Item>
                <Timeline.Item color="green">
                  Redis - 缓存服务 (版本: 7.0)
                </Timeline.Item>
                <Timeline.Item color="green">
                  Qdrant - 向量数据库 (版本: 1.7)
                </Timeline.Item>
                <Timeline.Item color="yellow">
                  NATS - 消息队列 (版本: 2.10)
                </Timeline.Item>
                <Timeline.Item color="green">
                  Neo4j - 图数据库 (版本: 5.0)
                </Timeline.Item>
              </Timeline>
            </Card>
          </TabPane>
        </Tabs>
      </Card>
    </div>
  )
}

export default MonitoringDashboardPage
