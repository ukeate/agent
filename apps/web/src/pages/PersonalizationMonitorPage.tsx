import React, { useState, useEffect } from 'react'
import { Card, Row, Col, Progress, Alert, Timeline, Button, Space, Tabs, Tag, Statistic, Typography, List, Badge } from 'antd'
import { Line, Area, Column, Gauge } from '@ant-design/plots'
import { 
  DashboardOutlined,
  AlertOutlined,
  CheckCircleOutlined,
  WarningOutlined,
  CloseCircleOutlined,
  ThunderboltOutlined,
  HistoryOutlined,
  SettingOutlined,
  SyncOutlined,
  RiseOutlined,
  FallOutlined
} from '@ant-design/icons'

const { Title, Text, Paragraph } = Typography
const { TabPane } = Tabs

interface MetricData {
  time: string
  value: number
  type: string
}

interface AlertItem {
  id: string
  type: 'error' | 'warning' | 'info'
  message: string
  timestamp: string
  resolved: boolean
}

interface OptimizationSuggestion {
  id: string
  category: string
  suggestion: string
  impact: 'high' | 'medium' | 'low'
  status: 'pending' | 'applied' | 'rejected'
}

const PersonalizationMonitorPage: React.FC = () => {
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [latencyData, setLatencyData] = useState<MetricData[]>([])
  const [throughputData, setThroughputData] = useState<MetricData[]>([])
  const [errorRateData, setErrorRateData] = useState<MetricData[]>([])
  const [resourceData, setResourceData] = useState<MetricData[]>([])
  
  const [systemHealth, setSystemHealth] = useState({
    overall: 'healthy',
    latency: 'good',
    throughput: 'good',
    errors: 'good',
    resources: 'warning'
  })

  const [alerts, setAlerts] = useState<AlertItem[]>([
    {
      id: '1',
      type: 'warning',
      message: 'P99延迟接近阈值 (95ms)',
      timestamp: '10:23:45',
      resolved: false
    },
    {
      id: '2',
      type: 'info',
      message: '缓存命中率提升到85%',
      timestamp: '10:15:30',
      resolved: true
    },
    {
      id: '3',
      type: 'error',
      message: '模型服务响应超时',
      timestamp: '09:45:12',
      resolved: true
    }
  ])

  const [optimizations, setOptimizations] = useState<OptimizationSuggestion[]>([
    {
      id: '1',
      category: '缓存优化',
      suggestion: '增加L1缓存大小至2GB',
      impact: 'high',
      status: 'pending'
    },
    {
      id: '2',
      category: '模型优化',
      suggestion: '启用INT8量化减少推理时间',
      impact: 'medium',
      status: 'applied'
    },
    {
      id: '3',
      category: '批处理',
      suggestion: '调整批大小至64',
      impact: 'low',
      status: 'pending'
    }
  ])

  // 生成模拟数据
  useEffect(() => {
    const generateData = () => {
      const now = new Date()
      const newLatency: MetricData[] = []
      const newThroughput: MetricData[] = []
      const newErrorRate: MetricData[] = []
      const newResource: MetricData[] = []

      for (let i = 59; i >= 0; i--) {
        const time = new Date(now.getTime() - i * 1000).toLocaleTimeString()
        
        // P50, P95, P99延迟
        newLatency.push(
          { time, value: 20 + Math.random() * 10, type: 'P50' },
          { time, value: 50 + Math.random() * 20, type: 'P95' },
          { time, value: 80 + Math.random() * 30, type: 'P99' }
        )
        
        // 吞吐量
        newThroughput.push({
          time,
          value: 1000 + Math.random() * 500,
          type: 'RPS'
        })
        
        // 错误率
        newErrorRate.push({
          time,
          value: Math.random() * 0.5,
          type: 'Error Rate'
        })
        
        // 资源使用
        newResource.push(
          { time, value: 60 + Math.random() * 20, type: 'CPU' },
          { time, value: 70 + Math.random() * 15, type: 'Memory' }
        )
      }

      setLatencyData(newLatency)
      setThroughputData(newThroughput)
      setErrorRateData(newErrorRate)
      setResourceData(newResource)
    }

    generateData()
    
    if (autoRefresh) {
      const interval = setInterval(generateData, 5000)
      return () => clearInterval(interval)
    }
  }, [autoRefresh])

  const latencyConfig = {
    data: latencyData,
    xField: 'time',
    yField: 'value',
    seriesField: 'type',
    smooth: true,
    animation: {
      appear: {
        animation: 'path-in',
        duration: 1000,
      },
    },
    yAxis: {
      title: {
        text: '延迟 (ms)',
      },
    },
    legend: {
      position: 'top-right' as const,
    },
  }

  const throughputConfig = {
    data: throughputData,
    xField: 'time',
    yField: 'value',
    smooth: true,
    areaStyle: {
      fill: 'l(270) 0:#ffffff 0.5:#7ec2f3 1:#1890ff',
    },
    yAxis: {
      title: {
        text: 'RPS',
      },
    },
  }

  const errorRateConfig = {
    data: errorRateData,
    xField: 'time',
    yField: 'value',
    columnStyle: {
      fill: (datum: any) => {
        return datum.value > 0.3 ? '#ff4d4f' : '#52c41a'
      },
    },
    yAxis: {
      title: {
        text: '错误率 (%)',
      },
    },
  }

  const cpuGaugeConfig = {
    percent: resourceData.find(d => d.type === 'CPU')?.value / 100 || 0.7,
    range: {
      color: 'l(0) 0:#52c41a 0.5:#faad14 1:#ff4d4f',
    },
    indicator: {
      pointer: {
        style: {
          stroke: '#D0D0D0',
        },
      },
      pin: {
        style: {
          stroke: '#D0D0D0',
        },
      },
    },
    statistic: {
      content: {
        style: {
          fontSize: '24px',
        },
        formatter: ({ percent }: any) => `CPU ${(percent * 100).toFixed(0)}%`,
      },
    },
  }

  const getHealthIcon = (status: string) => {
    switch (status) {
      case 'good':
      case 'healthy':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />
      case 'warning':
        return <WarningOutlined style={{ color: '#faad14' }} />
      case 'error':
      case 'critical':
        return <CloseCircleOutlined style={{ color: '#ff4d4f' }} />
      default:
        return null
    }
  }

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'high': return 'red'
      case 'medium': return 'orange'
      case 'low': return 'blue'
      default: return 'default'
    }
  }

  const getStatusTag = (status: string) => {
    switch (status) {
      case 'applied': return <Tag color="success">已应用</Tag>
      case 'pending': return <Tag color="processing">待处理</Tag>
      case 'rejected': return <Tag color="default">已拒绝</Tag>
      default: return null
    }
  }

  return (
    <div style={{ padding: '24px' }}>
      <Row justify="space-between" align="middle" style={{ marginBottom: 24 }}>
        <Col>
          <Title level={2}>
            <DashboardOutlined /> 性能监控仪表板
          </Title>
          <Paragraph type="secondary">
            实时监控个性化引擎性能指标和系统健康状况
          </Paragraph>
        </Col>
        <Col>
          <Space>
            <Button 
              type={autoRefresh ? 'primary' : 'default'}
              icon={<SyncOutlined spin={autoRefresh} />}
              onClick={() => setAutoRefresh(!autoRefresh)}
            >
              {autoRefresh ? '自动刷新' : '手动刷新'}
            </Button>
            <Button icon={<SettingOutlined />}>配置</Button>
          </Space>
        </Col>
      </Row>

      {/* 系统健康状态 */}
      <Card style={{ marginBottom: 24 }}>
        <Row gutter={16}>
          <Col span={4}>
            <Statistic
              title="系统状态"
              value={systemHealth.overall === 'healthy' ? '健康' : '异常'}
              prefix={getHealthIcon(systemHealth.overall)}
              valueStyle={{ 
                color: systemHealth.overall === 'healthy' ? '#52c41a' : '#ff4d4f' 
              }}
            />
          </Col>
          <Col span={5}>
            <Space>
              {getHealthIcon(systemHealth.latency)}
              <Text>延迟状态: 正常</Text>
            </Space>
          </Col>
          <Col span={5}>
            <Space>
              {getHealthIcon(systemHealth.throughput)}
              <Text>吞吐量: 正常</Text>
            </Space>
          </Col>
          <Col span={5}>
            <Space>
              {getHealthIcon(systemHealth.errors)}
              <Text>错误率: 正常</Text>
            </Space>
          </Col>
          <Col span={5}>
            <Space>
              {getHealthIcon(systemHealth.resources)}
              <Text>资源使用: 警告</Text>
            </Space>
          </Col>
        </Row>
      </Card>

      <Tabs defaultActiveKey="1">
        <TabPane tab={<span><ThunderboltOutlined /> 实时指标</span>} key="1">
          <Row gutter={[16, 16]}>
            <Col xs={24} lg={12}>
              <Card title="延迟趋势" size="small">
                <Line {...latencyConfig} height={200} />
              </Card>
            </Col>
            <Col xs={24} lg={12}>
              <Card title="吞吐量" size="small">
                <Area {...throughputConfig} height={200} />
              </Card>
            </Col>
            <Col xs={24} lg={12}>
              <Card title="错误率" size="small">
                <Column {...errorRateConfig} height={200} />
              </Card>
            </Col>
            <Col xs={24} lg={12}>
              <Card title="资源使用" size="small">
                <Row>
                  <Col span={12}>
                    <Gauge {...cpuGaugeConfig} height={200} />
                  </Col>
                  <Col span={12}>
                    <Gauge 
                      {...cpuGaugeConfig} 
                      percent={resourceData.find(d => d.type === 'Memory')?.value / 100 || 0.75}
                      statistic={{
                        content: {
                          style: { fontSize: '24px' },
                          formatter: ({ percent }: any) => `内存 ${(percent * 100).toFixed(0)}%`,
                        },
                      }}
                      height={200} 
                    />
                  </Col>
                </Row>
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab={<span><AlertOutlined /> 告警中心</span>} key="2">
          <Card>
            <List
              dataSource={alerts}
              renderItem={alert => (
                <List.Item
                  actions={[
                    alert.resolved ? 
                      <Tag color="success">已解决</Tag> : 
                      <Button size="small" type="link">处理</Button>
                  ]}
                >
                  <List.Item.Meta
                    avatar={
                      alert.type === 'error' ? <CloseCircleOutlined style={{ color: '#ff4d4f' }} /> :
                      alert.type === 'warning' ? <WarningOutlined style={{ color: '#faad14' }} /> :
                      <AlertOutlined style={{ color: '#1890ff' }} />
                    }
                    title={alert.message}
                    description={`发生时间: ${alert.timestamp}`}
                  />
                </List.Item>
              )}
            />
          </Card>
        </TabPane>

        <TabPane tab={<span><RiseOutlined /> 优化建议</span>} key="3">
          <Card>
            <List
              dataSource={optimizations}
              renderItem={item => (
                <List.Item
                  actions={[
                    getStatusTag(item.status),
                    item.status === 'pending' && (
                      <Space>
                        <Button size="small" type="primary">应用</Button>
                        <Button size="small">忽略</Button>
                      </Space>
                    )
                  ].filter(Boolean)}
                >
                  <List.Item.Meta
                    title={
                      <Space>
                        <Tag color={getImpactColor(item.impact)}>
                          {item.impact === 'high' ? '高影响' : 
                           item.impact === 'medium' ? '中影响' : '低影响'}
                        </Tag>
                        <Text strong>{item.category}</Text>
                      </Space>
                    }
                    description={item.suggestion}
                  />
                </List.Item>
              )}
            />
          </Card>
        </TabPane>

        <TabPane tab={<span><HistoryOutlined /> 历史趋势</span>} key="4">
          <Card>
            <Timeline>
              <Timeline.Item color="green">
                <Text strong>10:30</Text> - 系统性能优化完成，P99延迟降低20%
              </Timeline.Item>
              <Timeline.Item color="blue">
                <Text strong>10:15</Text> - 缓存命中率提升至85%
              </Timeline.Item>
              <Timeline.Item color="orange">
                <Text strong>09:45</Text> - 检测到模型服务响应延迟
              </Timeline.Item>
              <Timeline.Item color="red">
                <Text strong>09:30</Text> - 错误率超过阈值，触发告警
              </Timeline.Item>
              <Timeline.Item>
                <Text strong>09:00</Text> - 系统启动，开始监控
              </Timeline.Item>
            </Timeline>
          </Card>
        </TabPane>
      </Tabs>
    </div>
  )
}

export default PersonalizationMonitorPage