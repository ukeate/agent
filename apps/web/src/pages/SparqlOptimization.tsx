import React, { useEffect, useState, useMemo } from 'react'
import {
  Card,
  Typography,
  Row,
  Col,
  Space,
  Button,
  Table,
  Tabs,
  Select,
  Statistic,
  Tag,
  Progress,
  message,
  Alert,
  Timeline,
  List,
} from 'antd'
import {
  ThunderboltOutlined,
  LineChartOutlined,
  SettingOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  BulbOutlined,
  DatabaseOutlined,
  ClockCircleOutlined,
} from '@ant-design/icons'
import { sparqlService } from '../services/sparqlService'

const { Title, Text, Paragraph } = Typography
const { TabPane } = Tabs
const { Option } = Select

interface OptimizationRule {
  id: string
  description: string
  source: string
}

interface QueryPlan {
  original: string
  optimized: string
  steps: string[]
  estimatedCost: number | null
  actualCost: number | null
  improvement: number | null
}

const SparqlOptimization: React.FC = () => {
  const [loading, setLoading] = useState(false)
  const [selectedLevel, setSelectedLevel] = useState('standard')
  const [error, setError] = useState<string | null>(null)
  const [queryPlan, setQueryPlan] = useState<QueryPlan>({
    original: '',
    optimized: '',
    steps: [],
    estimatedCost: null,
    actualCost: null,
    improvement: null,
  })
  const [optimizationRules, setOptimizationRules] = useState<
    OptimizationRule[]
  >([])
  const [optimizationHistory, setOptimizationHistory] = useState<
    Array<{
      timestamp: string
      query: string
      execution_time: number
      cached?: boolean
      success?: boolean
      result_count?: number
    }>
  >([])
  const [performanceSnapshot, setPerformanceSnapshot] = useState<{
    avgTime: number | null
    throughput: number | null
    errorRate: number | null
    cacheHitRate: number | null
  } | null>(null)
  const [prevSnapshot, setPrevSnapshot] =
    useState<typeof performanceSnapshot>(null)

  const optimizationColumns = [
    {
      title: '优化建议',
      dataIndex: 'description',
      key: 'description',
      ellipsis: true,
    },
    {
      title: '来源',
      dataIndex: 'source',
      key: 'source',
      render: (source: string) => <Tag color="blue">{source}</Tag>,
    },
  ]

  const performanceData = useMemo(() => {
    const metrics = [
      { key: 'throughput', metric: '查询吞吐量', unit: 'QPS' },
      { key: 'avgTime', metric: '平均响应时间', unit: 'ms' },
      { key: 'errorRate', metric: '错误率', unit: '%' },
      { key: 'cacheHitRate', metric: '缓存命中率', unit: '%' },
    ]
    return metrics.map(item => ({
      metric: item.metric,
      unit: item.unit,
      before: prevSnapshot ? (prevSnapshot as any)[item.key] : null,
      after: performanceSnapshot
        ? (performanceSnapshot as any)[item.key]
        : null,
    }))
  }, [performanceSnapshot, prevSnapshot])

  const isNumber = (value: number | null): value is number =>
    typeof value === 'number' && !Number.isNaN(value)

  const levelMap: Record<string, 'NONE' | 'BASIC' | 'STANDARD' | 'AGGRESSIVE'> =
    {
      basic: 'BASIC',
      standard: 'STANDARD',
      aggressive: 'AGGRESSIVE',
      custom: 'NONE',
    }

  const buildPlanString = (plan: any) => {
    if (!plan || !Array.isArray(plan.steps)) return ''
    return plan.steps
      .map((step: any) => {
        const cost =
          typeof step.estimated_cost === 'number'
            ? ` cost=${step.estimated_cost}`
            : ''
        return `${step.operation}: ${step.description}${cost}`
      })
      .join('\n')
  }

  const loadData = async () => {
    setLoading(true)
    setError(null)
    try {
      const [performanceReport, cacheStats, history] = await Promise.all([
        sparqlService.getPerformanceReport(),
        sparqlService.getCacheStats(),
        sparqlService.getQueryHistory(20, 0),
      ])

      const summary =
        performanceReport.performance_report?.performance_summary || {}
      const execStats = summary.execution_time || {}
      const queryCountStats = summary.query_count || {}
      const windowMinutes =
        performanceReport.performance_report?.window_minutes || 60
      const throughput = queryCountStats.count
        ? queryCountStats.count / (windowMinutes * 60)
        : null
      const totalQueries =
        performanceReport.sparql_engine_stats?.total_queries || 0
      const failedQueries =
        performanceReport.sparql_engine_stats?.failed_queries || 0
      const errorRate =
        totalQueries > 0 ? (failedQueries / totalQueries) * 100 : null
      const avgTime = isNumber(execStats.mean) ? execStats.mean : null

      setPrevSnapshot(performanceSnapshot)
      setPerformanceSnapshot({
        avgTime,
        throughput: isNumber(throughput) ? throughput : null,
        errorRate: isNumber(errorRate) ? errorRate : null,
        cacheHitRate: isNumber(cacheStats.cache_hit_rate)
          ? cacheStats.cache_hit_rate * 100
          : null,
      })

      const recs =
        performanceReport.recommendations ||
        performanceReport.performance_report?.recommendations ||
        []
      setOptimizationRules(
        recs.map((rec: string, index: number) => ({
          id: `rec_${index}`,
          description: rec,
          source: '性能分析',
        }))
      )

      const historyItems = Array.isArray(history) ? history : []
      setOptimizationHistory(
        historyItems.map((item: any) => ({
          timestamp: item.timestamp,
          query: item.query,
          execution_time: item.execution_time,
          cached: item.cached,
          success: item.status === 'success',
          result_count: item.result_count,
        }))
      )

      const latest =
        historyItems.find((item: any) => item.status === 'success') ||
        historyItems[0]
      if (latest && latest.query) {
        const optimizeLevel = levelMap[selectedLevel] || 'STANDARD'
        const optimized = await sparqlService.optimizeQuery(
          latest.query,
          optimizeLevel as any
        )
        const originalExplain = await sparqlService.explainQuery({
          query: latest.query,
          include_optimization: true,
          include_statistics: true,
        })
        const optimizedExplain = optimized.optimized_query
          ? await sparqlService.explainQuery({
              query: optimized.optimized_query,
              include_optimization: true,
              include_statistics: true,
            })
          : null

        const estimatedCost = originalExplain.execution_plan?.total_cost ?? null
        const actualCost = optimizedExplain?.execution_plan?.total_cost ?? null
        const improvement =
          isNumber(estimatedCost) && isNumber(actualCost) && estimatedCost > 0
            ? ((estimatedCost - actualCost) / estimatedCost) * 100
            : typeof optimized.estimated_speedup === 'number'
              ? optimized.estimated_speedup * 100
              : null

        setQueryPlan({
          original: buildPlanString(originalExplain.execution_plan),
          optimized: optimizedExplain
            ? buildPlanString(optimizedExplain.execution_plan)
            : optimized.optimized_query || '',
          steps: Array.isArray(originalExplain.execution_plan?.steps)
            ? originalExplain.execution_plan.steps.map(
                (step: any) => step.description || step.operation
              )
            : [],
          estimatedCost,
          actualCost,
          improvement,
        })
      } else {
        setQueryPlan({
          original: '',
          optimized: '',
          steps: [],
          estimatedCost: null,
          actualCost: null,
          improvement: null,
        })
      }
    } catch (err) {
      setError((err as Error).message || '加载优化数据失败')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
  }, [selectedLevel])

  const runOptimizationAnalysis = async () => {
    try {
      await loadData()
      message.success('优化分析完成')
    } catch (error) {
      message.error('优化分析失败')
    }
  }

  return (
    <div style={{ padding: '24px', background: '#f0f2f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <ThunderboltOutlined
            style={{ marginRight: '8px', color: '#1890ff' }}
          />
          SPARQL查询优化器
        </Title>
        <Paragraph>智能查询优化系统，支持多种优化策略和性能监控</Paragraph>
        {error && (
          <Alert
            type="error"
            message={error}
            showIcon
            style={{ marginTop: 12 }}
          />
        )}
      </div>

      <Row gutter={[24, 24]}>
        <Col span={24}>
          <Card title="优化控制面板" size="small">
            <Row gutter={16}>
              <Col span={6}>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Text>优化级别:</Text>
                  <Select
                    value={selectedLevel}
                    onChange={setSelectedLevel}
                    style={{ width: '100%' }}
                  >
                    <Option value="basic">基础优化</Option>
                    <Option value="standard">标准优化</Option>
                    <Option value="aggressive">激进优化</Option>
                    <Option value="custom">自定义</Option>
                  </Select>
                </Space>
              </Col>
              <Col span={6}>
                <Statistic
                  title="总体改进率"
                  value={
                    isNumber(queryPlan.improvement)
                      ? queryPlan.improvement
                      : '-'
                  }
                  suffix="%"
                  valueStyle={{ color: '#3f8600' }}
                  prefix={<BulbOutlined />}
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="活跃规则数"
                  value={optimizationRules.length}
                  suffix=""
                  prefix={<SettingOutlined />}
                />
              </Col>
              <Col span={6}>
                <Button
                  type="primary"
                  icon={<LineChartOutlined />}
                  loading={loading}
                  onClick={runOptimizationAnalysis}
                >
                  运行分析
                </Button>
              </Col>
            </Row>
          </Card>
        </Col>

        <Col span={16}>
          <Tabs defaultActiveKey="rules" size="small">
            <TabPane tab="优化规则" key="rules">
              <Card size="small">
                <Table
                  dataSource={optimizationRules}
                  columns={optimizationColumns}
                  rowKey="id"
                  pagination={false}
                  size="small"
                />
              </Card>
            </TabPane>

            <TabPane tab="执行计划" key="plan">
              <Card size="small">
                <Row gutter={16}>
                  <Col span={12}>
                    <Card title="原始执行计划" size="small">
                      <pre
                        style={{
                          background: '#f5f5f5',
                          padding: '12px',
                          fontSize: '12px',
                          fontFamily: 'monospace',
                        }}
                      >
                        {queryPlan.original || '暂无执行计划'}
                      </pre>
                      <Statistic
                        title="预估成本"
                        value={
                          isNumber(queryPlan.estimatedCost)
                            ? queryPlan.estimatedCost
                            : '-'
                        }
                        suffix="单位"
                      />
                    </Card>
                  </Col>
                  <Col span={12}>
                    <Card title="优化执行计划" size="small">
                      <pre
                        style={{
                          background: '#f6ffed',
                          padding: '12px',
                          fontSize: '12px',
                          fontFamily: 'monospace',
                          border: '1px solid #b7eb8f',
                        }}
                      >
                        {queryPlan.optimized || '暂无优化计划'}
                      </pre>
                      <Row gutter={16}>
                        <Col span={12}>
                          <Statistic
                            title="实际成本"
                            value={
                              isNumber(queryPlan.actualCost)
                                ? queryPlan.actualCost
                                : '-'
                            }
                            suffix="单位"
                            valueStyle={{ color: '#3f8600' }}
                          />
                        </Col>
                        <Col span={12}>
                          <Statistic
                            title="性能提升"
                            value={
                              isNumber(queryPlan.improvement)
                                ? queryPlan.improvement
                                : '-'
                            }
                            suffix="%"
                            valueStyle={{ color: '#3f8600' }}
                          />
                        </Col>
                      </Row>
                    </Card>
                  </Col>
                </Row>

                <Card
                  title="优化步骤"
                  style={{ marginTop: '16px' }}
                  size="small"
                >
                  <Timeline>
                    {queryPlan.steps.length === 0 && (
                      <Timeline.Item color="gray">暂无优化步骤</Timeline.Item>
                    )}
                    {queryPlan.steps.map((step, index) => (
                      <Timeline.Item
                        key={index}
                        color="green"
                        dot={<CheckCircleOutlined />}
                      >
                        {step}
                      </Timeline.Item>
                    ))}
                  </Timeline>
                </Card>
              </Card>
            </TabPane>

            <TabPane tab="性能对比" key="performance">
              <Card size="small">
                <Row gutter={[16, 16]}>
                  {performanceData.map((data, index) => (
                    <Col span={12} key={index}>
                      <Card size="small" title={data.metric}>
                        <Row gutter={16}>
                          <Col span={12}>
                            <Statistic
                              title="优化前"
                              value={isNumber(data.before) ? data.before : '-'}
                              suffix={data.unit}
                            />
                          </Col>
                          <Col span={12}>
                            <Statistic
                              title="优化后"
                              value={isNumber(data.after) ? data.after : '-'}
                              suffix={data.unit}
                              valueStyle={{ color: '#3f8600' }}
                            />
                          </Col>
                        </Row>
                        {isNumber(data.before) &&
                        isNumber(data.after) &&
                        data.before > 0 ? (
                          <Progress
                            percent={
                              ((data.before - data.after) / data.before) * 100
                            }
                            status="success"
                            style={{ marginTop: '8px' }}
                          />
                        ) : (
                          <Text
                            type="secondary"
                            style={{ marginTop: '8px', display: 'block' }}
                          >
                            暂无对比数据
                          </Text>
                        )}
                      </Card>
                    </Col>
                  ))}
                </Row>

                <Alert
                  message="优化效果评估"
                  description={
                    isNumber(queryPlan.improvement)
                      ? `当前优化预计提升 ${queryPlan.improvement.toFixed(1)}%`
                      : '暂无优化评估数据'
                  }
                  type={
                    isNumber(queryPlan.improvement) && queryPlan.improvement > 0
                      ? 'success'
                      : 'info'
                  }
                  showIcon
                  style={{ marginTop: '16px' }}
                />
              </Card>
            </TabPane>
          </Tabs>
        </Col>

        <Col span={8}>
          <Card title="优化历史" size="small" style={{ height: '600px' }}>
            <List
              size="small"
              dataSource={optimizationHistory}
              renderItem={(item, index) => (
                <List.Item>
                  <Card size="small" style={{ width: '100%' }}>
                    <Space
                      direction="vertical"
                      size="small"
                      style={{ width: '100%' }}
                    >
                      <div
                        style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                        }}
                      >
                        <Tag color={item.success ? 'green' : 'red'}>
                          {isNumber(item.execution_time)
                            ? `${item.execution_time.toFixed(1)}ms`
                            : '-'}
                        </Tag>
                        <Text type="secondary" style={{ fontSize: '12px' }}>
                          {item.timestamp}
                        </Text>
                      </div>
                      <Text code style={{ fontSize: '12px' }}>
                        {item.query.substring(0, 50)}...
                      </Text>
                      <div>
                        <Text type="secondary" style={{ fontSize: '12px' }}>
                          结果数: {item.result_count ?? '-'} | 缓存:{' '}
                          {item.cached ? '是' : '否'}
                        </Text>
                      </div>
                    </Space>
                  </Card>
                </List.Item>
              )}
            />
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default SparqlOptimization
