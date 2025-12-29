import React, { useEffect, useState } from 'react'
import { 
  Card, 
  Typography, 
  Row, 
  Col, 
  Space, 
  Statistic,
  Progress,
  Table,
  Tag,
  Alert,
  Tabs,
  Button,
  List
} from 'antd'
import { 
  MonitorOutlined, 
  ThunderboltOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined
} from '@ant-design/icons'
import { healthService } from '../services/healthService'
import { sparqlService } from '../services/sparqlService'

const { Title, Paragraph } = Typography
const { TabPane } = Tabs

const SparqlPerformance: React.FC = () => {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [metricSnapshot, setMetricSnapshot] = useState<{
    throughput: number | null
    avgTime: number | null
    maxTime: number | null
    errorRate: number | null
  } | null>(null)
  const [prevSnapshot, setPrevSnapshot] = useState<typeof metricSnapshot>(null)
  const [slowQueries, setSlowQueries] = useState<Array<{
    query: string
    execution_time: number
    max_time: number
    frequency: number
  }>>([])
  const [recommendations, setRecommendations] = useState<string[]>([])
  const [resourceStats, setResourceStats] = useState<{
    cpu: number | null
    memory: number | null
    disk: number | null
    networkConnections: number | null
  }>({
    cpu: null,
    memory: null,
    disk: null,
    networkConnections: null
  })

  const isNumber = (value: number | null): value is number =>
    typeof value === 'number' && !Number.isNaN(value)

  const loadData = async () => {
    setLoading(true)
    setError(null)
    try {
      const [performanceReport, healthMetrics] = await Promise.all([
        sparqlService.getPerformanceReport(),
        healthService.getSystemMetrics()
      ])
      const summary = performanceReport.performance_report?.performance_summary || {}
      const execStats = summary.execution_time || {}
      const queryCountStats = summary.query_count || {}
      const windowMinutes = performanceReport.performance_report?.window_minutes || 60
      const throughput = queryCountStats.count
        ? queryCountStats.count / (windowMinutes * 60)
        : null
      const avgTime = isNumber(execStats.mean) ? execStats.mean : null
      const maxTime = isNumber(execStats.max) ? execStats.max : null
      const totalQueries = performanceReport.sparql_engine_stats?.total_queries || 0
      const failedQueries = performanceReport.sparql_engine_stats?.failed_queries || 0
      const errorRate = totalQueries > 0 ? (failedQueries / totalQueries) * 100 : null

      setPrevSnapshot(metricSnapshot)
      setMetricSnapshot({
        throughput: isNumber(throughput) ? throughput : null,
        avgTime,
        maxTime,
        errorRate
      })

      const slow = performanceReport.performance_report?.top_slow_queries || []
      setSlowQueries(slow.map(item => ({
        query: item.query,
        execution_time: item.execution_time,
        max_time: item.max_time,
        frequency: item.frequency
      })))
      setRecommendations(performanceReport.recommendations || [])

      const system = (healthMetrics as any).system || {}
      setResourceStats({
        cpu: isNumber(system.cpu_percent) ? system.cpu_percent : null,
        memory: isNumber(system.memory_percent) ? system.memory_percent : null,
        disk: isNumber(system.disk_percent) ? system.disk_percent : null,
        networkConnections: isNumber(system.network_connections) ? system.network_connections : null
      })
    } catch (err) {
      setError((err as Error).message || '加载性能数据失败')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
  }, [])

  const buildChange = (current: number | null, previous: number | null) => {
    if (!isNumber(current) || !isNumber(previous) || previous === 0) return null
    const diff = current - previous
    const percent = (diff / previous) * 100
    return {
      trend: diff >= 0 ? 'up' : 'down',
      change: `${diff >= 0 ? '+' : ''}${percent.toFixed(1)}%`
    }
  }

  const performanceMetrics = [
    {
      name: '查询吞吐量',
      value: metricSnapshot?.throughput ?? null,
      unit: 'QPS',
      ...(buildChange(metricSnapshot?.throughput ?? null, prevSnapshot?.throughput ?? null) || {})
    },
    {
      name: '平均响应时间',
      value: metricSnapshot?.avgTime ?? null,
      unit: 'ms',
      ...(buildChange(metricSnapshot?.avgTime ?? null, prevSnapshot?.avgTime ?? null) || {})
    },
    {
      name: '峰值响应时间',
      value: metricSnapshot?.maxTime ?? null,
      unit: 'ms',
      ...(buildChange(metricSnapshot?.maxTime ?? null, prevSnapshot?.maxTime ?? null) || {})
    },
    {
      name: '错误率',
      value: metricSnapshot?.errorRate ?? null,
      unit: '%',
      ...(buildChange(metricSnapshot?.errorRate ?? null, prevSnapshot?.errorRate ?? null) || {})
    }
  ]

  const columns = [
    {
      title: '查询模式',
      dataIndex: 'query',
      key: 'query',
      ellipsis: true,
      render: (query: string) => (
        <code style={{ fontSize: '12px' }}>{query.substring(0, 50)}...</code>
      )
    },
    {
      title: '平均时间',
      dataIndex: 'execution_time',
      key: 'execution_time',
      render: (time: number) => `${time}ms`
    },
    {
      title: '最大时间',
      dataIndex: 'max_time',
      key: 'max_time',
      render: (time: number) => `${time}ms`
    },
    {
      title: '执行次数',
      dataIndex: 'frequency',
      key: 'frequency'
    }
  ]

  return (
    <div style={{ padding: '24px', background: '#f0f2f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <MonitorOutlined style={{ marginRight: '8px', color: '#1890ff' }} />
          SPARQL性能监控
        </Title>
        <Paragraph>
          实时监控SPARQL查询性能、响应时间和系统资源使用情况
        </Paragraph>
        {error && (
          <Alert type="error" message={error} showIcon style={{ marginTop: 12 }} />
        )}
      </div>

      <Row gutter={[24, 24]}>
        <Col span={24}>
          <Card title="性能概览" size="small">
            <Row gutter={16}>
              {performanceMetrics.map((metric, index) => (
                <Col span={6} key={index}>
                  <Statistic
                    title={metric.name}
                    value={metric.value ?? '-'}
                    suffix={metric.unit}
                    prefix={index === 0 ? <ThunderboltOutlined /> : 
                           index === 1 ? <ClockCircleOutlined /> : 
                           index === 2 ? <ExclamationCircleOutlined /> : 
                           <CheckCircleOutlined />}
                    valueStyle={{ 
                      color: metric.trend === 'down' ? '#cf1322' : '#3f8600' 
                    }}
                  />
                  <div style={{ marginTop: '8px' }}>
                    {metric.change ? (
                      <Tag color={metric.trend === 'down' ? 'red' : 'green'}>
                        {metric.change}
                      </Tag>
                    ) : (
                      <Tag color="default">-</Tag>
                    )}
                  </div>
                </Col>
              ))}
            </Row>
          </Card>
        </Col>

        <Col span={16}>
          <Tabs defaultActiveKey="queries" size="small">
            <TabPane tab="查询性能" key="queries">
              <Card size="small">
                <Table
                  dataSource={slowQueries.map((item, index) => ({ key: index, ...item }))}
                  columns={columns}
                  pagination={false}
                  size="small"
                  loading={loading}
                  locale={{ emptyText: '暂无慢查询数据' }}
                />
              </Card>
            </TabPane>

            <TabPane tab="资源使用" key="resources">
              <Card size="small">
                <Row gutter={[16, 16]}>
                  <Col span={12}>
                    <Card size="small" title="CPU使用率">
                      {isNumber(resourceStats.cpu) ? (
                        <>
                          <Progress percent={resourceStats.cpu} status="active" />
                          <p style={{ marginTop: '8px', fontSize: '12px', color: '#666' }}>
                            当前: {resourceStats.cpu}%
                          </p>
                        </>
                      ) : (
                        <Text type="secondary">暂无数据</Text>
                      )}
                    </Card>
                  </Col>
                  <Col span={12}>
                    <Card size="small" title="内存使用率">
                      {isNumber(resourceStats.memory) ? (
                        <>
                          <Progress percent={resourceStats.memory} status="active" strokeColor="#52c41a" />
                          <p style={{ marginTop: '8px', fontSize: '12px', color: '#666' }}>
                            当前: {resourceStats.memory}%
                          </p>
                        </>
                      ) : (
                        <Text type="secondary">暂无数据</Text>
                      )}
                    </Card>
                  </Col>
                  <Col span={12}>
                    <Card size="small" title="磁盘使用率">
                      {isNumber(resourceStats.disk) ? (
                        <>
                          <Progress percent={resourceStats.disk} status="active" strokeColor="#1890ff" />
                          <p style={{ marginTop: '8px', fontSize: '12px', color: '#666' }}>
                            当前: {resourceStats.disk}%
                          </p>
                        </>
                      ) : (
                        <Text type="secondary">暂无数据</Text>
                      )}
                    </Card>
                  </Col>
                  <Col span={12}>
                    <Card size="small" title="网络连接数">
                      {isNumber(resourceStats.networkConnections) ? (
                        <Statistic value={resourceStats.networkConnections} />
                      ) : (
                        <Text type="secondary">暂无数据</Text>
                      )}
                    </Card>
                  </Col>
                </Row>
              </Card>
            </TabPane>
          </Tabs>
        </Col>

        <Col span={8}>
          <Card title="性能建议" size="small">
            <Space direction="vertical" style={{ width: '100%' }}>
              {recommendations.length === 0 && (
                <Alert
                  message="暂无性能建议"
                  type="info"
                  showIcon
                />
              )}
              {recommendations.map((rec, index) => (
                <Alert
                  key={index}
                  message={`建议 ${index + 1}`}
                  description={rec}
                  type="info"
                  showIcon
                />
              ))}
            </Space>
          </Card>

          <Card title="快速操作" size="small" style={{ marginTop: '16px' }}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Button type="primary" block>
                生成性能报告
              </Button>
              <Button block>
                导出监控数据
              </Button>
              <Button block>
                配置告警规则
              </Button>
            </Space>
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default SparqlPerformance
