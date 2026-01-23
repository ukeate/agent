import React, { useEffect, useState } from 'react'
import {
  Alert,
  Card,
  Row,
  Col,
  Statistic,
  Typography,
  Space,
  Button,
  Tabs,
  Table,
} from 'antd'
import {
  DashboardOutlined,
  CloudServerOutlined,
  BankOutlined,
  HddOutlined,
  ThunderboltOutlined,
  DatabaseOutlined,
  ClockCircleOutlined,
  WifiOutlined,
  ReloadOutlined,
} from '@ant-design/icons'
import { healthService } from '../services/healthService'
import { sparqlService } from '../services/sparqlService'

const { Title, Text } = Typography

const KnowledgeGraphPerformanceMonitor: React.FC = () => {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [systemMetrics, setSystemMetrics] = useState<{
    cpu_usage: number | null
    memory_usage: number | null
    disk_usage: number | null
    cache_hit_rate: number | null
    active_requests: number | null
    query_response_time: number | null
    throughput: number | null
    network_connections: number | null
  }>({
    cpu_usage: null,
    memory_usage: null,
    disk_usage: null,
    cache_hit_rate: null,
    active_requests: null,
    query_response_time: null,
    throughput: null,
    network_connections: null,
  })
  const [performanceReport, setPerformanceReport] = useState<null | {
    performance_report: {
      top_slow_queries: Array<{
        query: string
        execution_time: number
        max_time: number
        frequency: number
        timestamp: number
      }>
    }
    sparql_engine_stats: Record<string, any>
  }>(null)
  const [cacheStats, setCacheStats] = useState<null | {
    cache_stats: Record<string, any>
    total_queries: number
    cached_queries: number
    cache_hit_rate: number
  }>(null)

  const isNumber = (value: number | null): value is number =>
    typeof value === 'number' && !Number.isNaN(value)

  const loadData = async () => {
    setLoading(true)
    setError(null)
    try {
      const [healthMetrics, performanceReport, cacheStats] = await Promise.all([
        healthService.getSystemMetrics(),
        sparqlService.getPerformanceReport(),
        sparqlService.getCacheStats(),
      ])
      const system = (healthMetrics as any).system || {}
      const performance = (healthMetrics as any).performance || {}
      const summary =
        performanceReport.performance_report?.performance_summary || {}
      const execStats = summary.execution_time || {}
      const queryCountStats = summary.query_count || {}
      const windowMinutes =
        performanceReport.performance_report?.window_minutes || 60
      const throughput = queryCountStats.count
        ? queryCountStats.count / (windowMinutes * 60)
        : null
      const avgResponse = isNumber(execStats.mean)
        ? execStats.mean
        : performanceReport.sparql_engine_stats?.average_execution_time || null

      setSystemMetrics({
        cpu_usage: isNumber(system.cpu_percent) ? system.cpu_percent : null,
        memory_usage: isNumber(system.memory_percent)
          ? system.memory_percent
          : null,
        disk_usage: isNumber(system.disk_percent) ? system.disk_percent : null,
        cache_hit_rate: isNumber(cacheStats.cache_hit_rate)
          ? cacheStats.cache_hit_rate * 100
          : null,
        active_requests: isNumber(performance.active_requests)
          ? performance.active_requests
          : null,
        query_response_time: isNumber(avgResponse) ? avgResponse : null,
        throughput: isNumber(throughput) ? throughput : null,
        network_connections: isNumber(system.network_connections)
          ? system.network_connections
          : null,
      })
      setPerformanceReport(performanceReport)
      setCacheStats(cacheStats)
    } catch (err) {
      setError((err as Error).message || '加载性能数据失败')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
  }, [])

  const slowQueries =
    performanceReport?.performance_report?.top_slow_queries || []
  const slowQueryColumns = [
    {
      title: '查询',
      dataIndex: 'query',
      key: 'query',
      render: (text: string) => (
        <Text type="secondary" style={{ fontSize: 12 }}>
          {text.length > 80 ? `${text.slice(0, 80)}...` : text}
        </Text>
      ),
    },
    {
      title: '平均耗时(ms)',
      dataIndex: 'execution_time',
      key: 'execution_time',
      render: (value: number) => value.toFixed(2),
    },
    {
      title: '最大耗时(ms)',
      dataIndex: 'max_time',
      key: 'max_time',
      render: (value: number) => value.toFixed(2),
    },
    {
      title: '频次',
      dataIndex: 'frequency',
      key: 'frequency',
    },
  ]

  const tabItems = [
    {
      key: 'overview',
      label: '系统概览',
      children: (
        <div>
          <Row gutter={16} style={{ marginBottom: '24px' }}>
            <Col span={6}>
              <Card>
                <Statistic
                  title="CPU使用率"
                  value={systemMetrics.cpu_usage ?? '-'}
                  precision={1}
                  suffix="%"
                  prefix={<CloudServerOutlined />}
                  valueStyle={{
                    color:
                      isNumber(systemMetrics.cpu_usage) &&
                      systemMetrics.cpu_usage > 80
                        ? '#ff4d4f'
                        : '#52c41a',
                  }}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="内存使用率"
                  value={systemMetrics.memory_usage ?? '-'}
                  precision={1}
                  suffix="%"
                  prefix={<BankOutlined />}
                  valueStyle={{
                    color:
                      isNumber(systemMetrics.memory_usage) &&
                      systemMetrics.memory_usage > 80
                        ? '#ff4d4f'
                        : '#52c41a',
                  }}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="磁盘使用率"
                  value={systemMetrics.disk_usage ?? '-'}
                  precision={1}
                  suffix="%"
                  prefix={<HddOutlined />}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="缓存命中率"
                  value={systemMetrics.cache_hit_rate ?? '-'}
                  precision={1}
                  suffix="%"
                  prefix={<ThunderboltOutlined />}
                  valueStyle={{ color: '#52c41a' }}
                />
              </Card>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col span={6}>
              <Card>
                <Statistic
                  title="活动请求数"
                  value={systemMetrics.active_requests ?? '-'}
                  prefix={<DatabaseOutlined />}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="查询响应时间"
                  value={systemMetrics.query_response_time ?? '-'}
                  suffix="ms"
                  prefix={<ClockCircleOutlined />}
                  valueStyle={{
                    color:
                      isNumber(systemMetrics.query_response_time) &&
                      systemMetrics.query_response_time > 500
                        ? '#ff4d4f'
                        : '#52c41a',
                  }}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="系统吞吐量"
                  value={systemMetrics.throughput ?? '-'}
                  suffix="QPS"
                  prefix={<DatabaseOutlined />}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="网络连接数"
                  value={systemMetrics.network_connections ?? '-'}
                  precision={1}
                  suffix="个"
                  prefix={<WifiOutlined />}
                />
              </Card>
            </Col>
          </Row>
        </div>
      ),
    },
    {
      key: 'queries',
      label: '查询性能',
      children: (
        <Card title="查询性能分析">
          <Table
            dataSource={slowQueries.map((item, index) => ({
              key: index,
              ...item,
            }))}
            columns={slowQueryColumns}
            pagination={false}
            size="small"
            locale={{ emptyText: '暂无慢查询数据' }}
          />
        </Card>
      ),
    },
    {
      key: 'cache',
      label: '缓存监控',
      children: (
        <Card title="缓存监控">
          <Row gutter={16}>
            <Col span={8}>
              <Statistic
                title="缓存大小"
                value={cacheStats?.cache_stats?.size ?? '-'}
              />
            </Col>
            <Col span={8}>
              <Statistic
                title="命中率"
                value={
                  isNumber(cacheStats?.cache_hit_rate ?? null)
                    ? (cacheStats?.cache_hit_rate as number) * 100
                    : '-'
                }
                suffix="%"
              />
            </Col>
            <Col span={8}>
              <Statistic
                title="缓存查询数"
                value={cacheStats?.cached_queries ?? '-'}
              />
            </Col>
          </Row>
        </Card>
      ),
    },
  ]

  return (
    <div style={{ padding: '24px' }}>
      <Card style={{ marginBottom: '24px' }}>
        <Row justify="space-between" align="middle">
          <Col>
            <Space>
              <DashboardOutlined style={{ fontSize: '24px' }} />
              <Title level={2} style={{ margin: 0 }}>
                性能监控
              </Title>
              <Text type="secondary">
                监控知识图谱系统性能指标和系统资源使用情况
              </Text>
            </Space>
          </Col>
          <Col>
            <Button
              icon={<ReloadOutlined />}
              onClick={loadData}
              loading={loading}
            >
              刷新
            </Button>
          </Col>
        </Row>
      </Card>

      {error && (
        <Alert
          type="error"
          message={error}
          showIcon
          style={{ marginBottom: 16 }}
        />
      )}

      <Tabs items={tabItems} />
    </div>
  )
}

export default KnowledgeGraphPerformanceMonitor
