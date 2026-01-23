import React, { useState, useEffect } from 'react'
import {
  Card,
  Button,
  Input,
  Select,
  Switch,
  Spin,
  Alert,
  Tabs,
  Tag,
  message,
  Statistic,
  Form,
  DatePicker,
  Modal,
  Space,
} from 'antd'
import { logger } from '../utils/logger'
import {
  SearchOutlined,
  ThunderboltOutlined,
  NodeIndexOutlined,
  BranchesOutlined,
  DatabaseOutlined,
  HeartOutlined,
  BarChartOutlined,
  BugOutlined,
  SettingOutlined,
  BulbOutlined,
  ShareAltOutlined,
  CheckCircleOutlined,
} from '@ant-design/icons'
import {
  graphRAGService,
  type GraphRAGResponse,
} from '../services/graphRAGService'
import apiClient from '../services/apiClient'

const { TextArea } = Input
const { Option } = Select
const { TabPane } = Tabs
const { RangePicker } = DatePicker

const GraphRAGPage: React.FC = () => {
  const [query, setQuery] = useState('')
  const [retrievalMode, setRetrievalMode] = useState<
    'vector' | 'graph' | 'hybrid' | 'adaptive'
  >('hybrid')
  const [maxDocs, setMaxDocs] = useState(10)
  const [includeReasoning, setIncludeReasoning] = useState(true)
  const [expansionDepth, setExpansionDepth] = useState(2)
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.6)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<GraphRAGResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  // RAG系统状态
  const [ragStats, setRagStats] = useState<any>(null)
  const [ragHealth, setRagHealth] = useState<any>(null)
  const [statsLoading, setStatsLoading] = useState(false)

  // 新增功能状态
  const [queryAnalysis, setQueryAnalysis] = useState<any>(null)
  const [reasoningResult, setReasoningResult] = useState<any>(null)
  const [fusionResult, setFusionResult] = useState<any>(null)
  const [performanceStats, setPerformanceStats] = useState<any>(null)
  const [debugResult, setDebugResult] = useState<any>(null)
  const [graphConfig, setGraphConfig] = useState<any>(null)
  const [consistencyCheck, setConsistencyCheck] = useState<any>(null)
  const [performanceComparison, setPerformanceComparison] = useState<any>(null)
  const [benchmarkResult, setBenchmarkResult] = useState<any>(null)

  // 新增功能Loading状态
  const [analysisLoading, setAnalysisLoading] = useState(false)
  const [reasoningLoading, setReasoningLoading] = useState(false)
  const [fusionLoading, setFusionLoading] = useState(false)
  const [statsQueryLoading, setStatsQueryLoading] = useState(false)
  const [debugLoading, setDebugLoading] = useState(false)
  const [configLoading, setConfigLoading] = useState(false)
  const [consistencyLoading, setConsistencyLoading] = useState(false)
  const [comparisonLoading, setComparisonLoading] = useState(false)
  const [benchmarkLoading, setBenchmarkLoading] = useState(false)

  // 表单和输入状态
  const [form] = Form.useForm()
  const [reasoningModalVisible, setReasoningModalVisible] = useState(false)
  const [fusionModalVisible, setFusionModalVisible] = useState(false)
  const [debugModalVisible, setDebugModalVisible] = useState(false)
  const [benchmarkModalVisible, setBenchmarkModalVisible] = useState(false)
  const [performanceModalVisible, setPerformanceModalVisible] = useState(false)

  // 页面加载时获取RAG系统状态
  useEffect(() => {
    const loadRagSystemStatus = async () => {
      setStatsLoading(true)
      try {
        const [statsResp, healthResp] = await Promise.all([
          apiClient.get('/graphrag/performance/stats'),
          apiClient.get('/health', { params: { detailed: true } }),
        ])
        const rawStats: any = statsResp.data
        setRagStats({
          ...(rawStats?.data || {}),
          timestamp: rawStats?.timestamp,
        })
        setRagHealth(healthResp.data)
      } catch (error) {
        logger.error('加载RAG系统状态失败:', error)
        message.error('加载RAG系统状态失败')
      } finally {
        setStatsLoading(false)
      }
    }

    loadRagSystemStatus()
  }, [])

  // 查询分析
  const handleQueryAnalysis = async (queryText: string) => {
    try {
      setAnalysisLoading(true)
      const analysis = await graphRAGService.analyzeQuery(queryText)
      setQueryAnalysis(analysis)
      message.success('查询分析完成')
    } catch (error) {
      logger.error('查询分析失败:', error)
      message.error('查询分析失败')
    } finally {
      setAnalysisLoading(false)
    }
  }

  // 推理查询
  const handleReasoningQuery = async (values: any) => {
    try {
      setReasoningLoading(true)
      const reasoning = await graphRAGService.queryReasoning(
        values.entity1,
        values.entity2,
        values.maxHops,
        values.maxPaths
      )
      setReasoningResult(reasoning)
      setReasoningModalVisible(false)
      message.success('推理查询完成')
    } catch (error) {
      logger.error('推理查询失败:', error)
      message.error('推理查询失败')
    } finally {
      setReasoningLoading(false)
    }
  }

  // 多源融合查询
  const handleMultiSourceFusion = async (values: any) => {
    try {
      setFusionLoading(true)
      const vector_results = JSON.parse(values.vector_results || '[]')
      const graph_results = JSON.parse(values.graph_results || '{}')
      const reasoning_results = JSON.parse(values.reasoning_results || '[]')
      const confidence_threshold =
        typeof values.confidence_threshold === 'number'
          ? values.confidence_threshold
          : Number(values.confidence_threshold)
      const fusion = await graphRAGService.multiSourceFusion({
        vector_results,
        graph_results,
        reasoning_results,
        confidence_threshold: Number.isFinite(confidence_threshold)
          ? confidence_threshold
          : undefined,
      })
      setFusionResult(fusion)
      setFusionModalVisible(false)
      message.success('多源融合查询完成')
    } catch (error) {
      logger.error('多源融合查询失败:', error)
      message.error('多源融合查询失败')
    } finally {
      setFusionLoading(false)
    }
  }

  // 获取性能统计
  const handleGetPerformanceStats = async (
    startTime?: string,
    endTime?: string
  ) => {
    try {
      setStatsQueryLoading(true)
      const stats = await graphRAGService.getPerformanceStats(
        startTime,
        endTime
      )
      setPerformanceStats(stats)
      message.success('性能统计获取完成')
    } catch (error) {
      logger.error('获取性能统计失败:', error)
      message.error('获取性能统计失败')
    } finally {
      setStatsQueryLoading(false)
    }
  }

  // 调试解释
  const handleDebugExplain = async (values: any) => {
    try {
      setDebugLoading(true)
      const reasoning_paths = (result?.reasoning_results || []) as any[]
      if (reasoning_paths.length === 0) {
        message.error('请先执行GraphRAG查询并开启推理后再解释')
        return
      }
      const debug = await graphRAGService.explainResult({
        query: values.query,
        reasoning_paths,
      })
      setDebugResult(debug)
      setDebugModalVisible(false)
      message.success('调试解释完成')
    } catch (error) {
      logger.error('调试解释失败:', error)
      message.error('调试解释失败')
    } finally {
      setDebugLoading(false)
    }
  }

  // 获取配置
  const handleGetConfig = async () => {
    try {
      setConfigLoading(true)
      const config = await graphRAGService.getConfig()
      setGraphConfig(config)
      message.success('配置获取完成')
    } catch (error) {
      logger.error('获取配置失败:', error)
      message.error('获取配置失败')
    } finally {
      setConfigLoading(false)
    }
  }

  // 一致性检查
  const handleConsistencyCheck = async () => {
    try {
      setConsistencyLoading(true)
      const consistency = await graphRAGService.getConsistencyCheck()
      setConsistencyCheck(consistency)
      message.success('一致性检查完成')
    } catch (error) {
      logger.error('一致性检查失败:', error)
      message.error('一致性检查失败')
    } finally {
      setConsistencyLoading(false)
    }
  }

  // 性能对比
  const handlePerformanceComparison = async (values: any) => {
    try {
      setComparisonLoading(true)
      const baselineDate = values.baselineDate?.toISOString?.()
      const comparisonDate = values.comparisonDate?.toISOString?.()
      if (!baselineDate || !comparisonDate) {
        message.error('请选择基线日期和对比日期')
        return
      }
      const comparison = await graphRAGService.getPerformanceComparison(
        baselineDate,
        comparisonDate
      )
      setPerformanceComparison(comparison)
      setPerformanceModalVisible(false)
      message.success('性能对比完成')
    } catch (error) {
      logger.error('性能对比失败:', error)
      message.error('性能对比失败')
    } finally {
      setComparisonLoading(false)
    }
  }

  // 基准测试
  const handleBenchmark = async (values: any) => {
    try {
      setBenchmarkLoading(true)
      const test_queries = JSON.parse(values.test_queries || '[]')
      const confidence_threshold =
        typeof values.confidence_threshold === 'number'
          ? values.confidence_threshold
          : Number(values.confidence_threshold)
      const benchmark = await graphRAGService.runBenchmark({
        test_queries,
        config: {
          max_docs: values.max_docs,
          include_reasoning: values.include_reasoning,
          confidence_threshold: Number.isFinite(confidence_threshold)
            ? confidence_threshold
            : undefined,
        },
      })
      setBenchmarkResult(benchmark)
      setBenchmarkModalVisible(false)
      message.success('基准测试完成')
    } catch (error) {
      logger.error('基准测试失败:', error)
      message.error('基准测试失败')
    } finally {
      setBenchmarkLoading(false)
    }
  }

  const handleSubmit = async () => {
    if (!query.trim()) {
      message.warning('请输入查询内容')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const data = await graphRAGService.query({
        query,
        retrieval_mode: retrievalMode,
        max_docs: maxDocs,
        include_reasoning: includeReasoning,
        expansion_depth: expansionDepth,
        confidence_threshold: confidenceThreshold,
      })

      setResult(data)
      message.success('查询成功')
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || err.message || '查询失败'
      setError(errorMsg)
      message.error(errorMsg)
    } finally {
      setLoading(false)
    }
  }

  const getRetrievalModeIcon = (mode: string) => {
    switch (mode) {
      case 'vector':
        return <SearchOutlined />
      case 'graph':
        return <NodeIndexOutlined />
      case 'hybrid':
        return <BranchesOutlined />
      case 'adaptive':
        return <ThunderboltOutlined />
      default:
        return <SearchOutlined />
    }
  }

  const getRetrievalModeColor = (mode: string) => {
    switch (mode) {
      case 'vector':
        return 'blue'
      case 'graph':
        return 'green'
      case 'hybrid':
        return 'purple'
      case 'adaptive':
        return 'orange'
      default:
        return 'default'
    }
  }

  return (
    <div className="container mx-auto p-6">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          GraphRAG 图谱增强检索系统
        </h1>
        <p className="text-gray-600 mb-8">
          结合知识图谱和向量检索的混合RAG系统，提供更智能、更准确的知识问答
        </p>

        {/* RAG系统状态概览 */}
        <Card
          title={
            <span>
              <DatabaseOutlined className="mr-2" />
              RAG系统状态
            </span>
          }
          className="mb-6"
        >
          {statsLoading ? (
            <div className="text-center p-4">
              <Spin tip="加载系统状态中..." />
            </div>
          ) : (
            <div className="space-y-6">
              {/* 索引统计 */}
              {ragStats && (
                <div>
                  <h4 className="text-lg font-semibold mb-3 flex items-center">
                    <DatabaseOutlined className="mr-2 text-blue-500" />
                    查询统计
                  </h4>
                  <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                    <div className="text-center p-3 bg-blue-50 rounded-lg">
                      <Statistic
                        title="总查询数"
                        value={ragStats.query_statistics?.total_queries || 0}
                        valueStyle={{ color: '#1890ff', fontSize: '20px' }}
                      />
                    </div>
                    <div className="text-center p-3 bg-green-50 rounded-lg">
                      <Statistic
                        title="成功"
                        value={
                          ragStats.query_statistics?.successful_queries || 0
                        }
                        valueStyle={{ color: '#52c41a', fontSize: '20px' }}
                      />
                    </div>
                    <div className="text-center p-3 bg-purple-50 rounded-lg">
                      <Statistic
                        title="失败"
                        value={ragStats.query_statistics?.failed_queries || 0}
                        valueStyle={{ color: '#722ed1', fontSize: '20px' }}
                      />
                    </div>
                    <div className="text-center p-3 bg-orange-50 rounded-lg">
                      <Statistic
                        title="平均响应"
                        value={
                          ragStats.query_statistics?.average_response_time || 0
                        }
                        suffix="ms"
                        valueStyle={{ color: '#fa8c16', fontSize: '20px' }}
                      />
                    </div>
                    <div className="text-center p-3 bg-red-50 rounded-lg">
                      <Statistic
                        title="缓存命中率"
                        value={
                          (ragStats.performance_metrics?.cache_hit_rate || 0) *
                          100
                        }
                        suffix="%"
                        precision={1}
                        valueStyle={{ color: '#f5222d', fontSize: '20px' }}
                      />
                    </div>
                    <div className="text-center p-3 bg-gray-50 rounded-lg">
                      <div className="text-xs text-gray-600">最后更新</div>
                      <div className="text-sm font-semibold text-gray-900 mt-1">
                        {ragStats.timestamp
                          ? new Date(ragStats.timestamp).toLocaleTimeString()
                          : '-'}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* 健康状态 */}
              {ragHealth && (
                <div>
                  <h4 className="text-lg font-semibold mb-3 flex items-center">
                    <HeartOutlined className="mr-2 text-red-500" />
                    系统健康状态
                  </h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <div className="grid grid-cols-2 gap-4">
                        <div className="text-center p-3 bg-green-50 rounded-lg">
                          <div className="text-sm text-gray-600">系统状态</div>
                          <Tag
                            color={
                              ragHealth.status === 'healthy'
                                ? 'green'
                                : ragHealth.status === 'degraded'
                                  ? 'orange'
                                  : 'red'
                            }
                            className="mt-2"
                          >
                            {String(ragHealth.status).toUpperCase()}
                          </Tag>
                        </div>
                        <div className="text-center p-3 bg-blue-50 rounded-lg">
                          <Statistic
                            title="运行时间"
                            value={Math.floor(
                              (ragHealth.components?.api?.uptime_seconds || 0) /
                                3600
                            )}
                            suffix="小时"
                            valueStyle={{ fontSize: '18px' }}
                          />
                        </div>
                        <div className="text-center p-3 bg-purple-50 rounded-lg">
                          <Statistic
                            title="内存使用"
                            value={
                              ragHealth.metrics?.process?.memory_rss_mb || 0
                            }
                            suffix="MB"
                            valueStyle={{ fontSize: '18px' }}
                          />
                        </div>
                        <div className="text-center p-3 bg-orange-50 rounded-lg">
                          <Statistic
                            title="CPU使用率"
                            value={ragHealth.metrics?.process?.cpu_percent || 0}
                            suffix="%"
                            valueStyle={{ fontSize: '18px' }}
                          />
                        </div>
                      </div>
                    </div>
                    <div>
                      <div className="bg-gray-50 p-4 rounded-lg">
                        <h5 className="font-semibold mb-3">组件状态</h5>
                        <div className="space-y-2">
                          {Object.entries(ragHealth.components || {}).map(
                            ([component, detail]: any) => (
                              <div
                                key={component}
                                className="flex justify-between items-center"
                              >
                                <span className="text-gray-700 capitalize">
                                  {component.replace('_', ' ')}
                                </span>
                                <Tag
                                  color={
                                    detail?.status === 'healthy'
                                      ? 'green'
                                      : detail?.status === 'degraded'
                                        ? 'orange'
                                        : 'red'
                                  }
                                >
                                  {String(
                                    detail?.status || 'unknown'
                                  ).toUpperCase()}
                                </Tag>
                              </div>
                            )
                          )}
                        </div>
                        <div className="mt-3 text-xs text-gray-500">
                          <div>
                            活跃连接:{' '}
                            {ragHealth.components?.api?.active_connections || 0}
                          </div>
                          <div>
                            最后检查:{' '}
                            {ragHealth.timestamp
                              ? new Date(ragHealth.timestamp).toLocaleString()
                              : '-'}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </Card>

        {/* 查询配置 */}
        <Card title="查询配置" className="mb-6">
          <div className="space-y-4">
            {/* 查询输入 */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                查询内容
              </label>
              <TextArea
                placeholder="输入您的问题，例如：什么是机器学习？它与人工智能的关系是什么？"
                value={query}
                onChange={e => setQuery(e.target.value)}
                name="graphrag-query"
                rows={3}
                disabled={loading}
              />
            </div>

            {/* 配置选项 */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  检索模式
                </label>
                <Select
                  value={retrievalMode}
                  onChange={setRetrievalMode}
                  name="graphrag-retrieval-mode"
                  className="w-full"
                  disabled={loading}
                >
                  <Option value="vector">
                    <SearchOutlined /> 纯向量检索
                  </Option>
                  <Option value="graph">
                    <NodeIndexOutlined /> 纯图谱检索
                  </Option>
                  <Option value="hybrid">
                    <BranchesOutlined /> 混合检索
                  </Option>
                  <Option value="adaptive">
                    <ThunderboltOutlined /> 自适应检索
                  </Option>
                </Select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  最大文档数
                </label>
                <Select
                  value={maxDocs}
                  onChange={setMaxDocs}
                  name="graphrag-max-docs"
                  className="w-full"
                  disabled={loading}
                >
                  <Option value={5}>5</Option>
                  <Option value={10}>10</Option>
                  <Option value={20}>20</Option>
                  <Option value={50}>50</Option>
                </Select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  扩展深度
                </label>
                <Select
                  value={expansionDepth}
                  onChange={setExpansionDepth}
                  name="graphrag-expansion-depth"
                  className="w-full"
                  disabled={loading}
                >
                  <Option value={1}>1层</Option>
                  <Option value={2}>2层</Option>
                  <Option value={3}>3层</Option>
                  <Option value={4}>4层</Option>
                </Select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  置信度阈值
                </label>
                <Select
                  value={confidenceThreshold}
                  onChange={setConfidenceThreshold}
                  name="graphrag-confidence-threshold"
                  className="w-full"
                  disabled={loading}
                >
                  <Option value={0.4}>0.4 (低)</Option>
                  <Option value={0.6}>0.6 (中)</Option>
                  <Option value={0.8}>0.8 (高)</Option>
                  <Option value={0.9}>0.9 (很高)</Option>
                </Select>
              </div>

              <div className="flex items-center">
                <Switch
                  checked={includeReasoning}
                  onChange={setIncludeReasoning}
                  disabled={loading}
                />
                <span className="ml-2 text-sm text-gray-700">包含推理路径</span>
              </div>
            </div>

            {/* 查询按钮 */}
            <Button
              type="primary"
              size="large"
              icon={getRetrievalModeIcon(retrievalMode)}
              onClick={handleSubmit}
              loading={loading}
              className="w-full md:w-auto"
            >
              {loading ? '处理中...' : 'GraphRAG 查询'}
            </Button>
          </div>
        </Card>

        {/* 错误提示 */}
        {error && (
          <Alert
            message="查询失败"
            description={error}
            type="error"
            showIcon
            closable
            className="mb-6"
          />
        )}

        {/* 高级功能标签页 */}
        <Card title="高级功能" className="mb-6">
          <Tabs defaultActiveKey="analysis" type="card">
            <TabPane
              tab={
                <span>
                  <BulbOutlined />
                  查询分析
                </span>
              }
              key="analysis"
            >
              <div className="space-y-4">
                <div className="flex gap-4">
                  <Input.Search
                    placeholder="输入查询内容进行分析"
                    enterButton="分析查询"
                    size="large"
                    loading={analysisLoading}
                    name="graphrag-analysis-query"
                    onSearch={handleQueryAnalysis}
                  />
                </div>
                {queryAnalysis && (
                  <div className="space-y-4">
                    <div className="bg-blue-50 p-4 rounded-lg">
                      <h4 className="font-semibold mb-2">分解结果</h4>
                      <p>
                        <strong>策略:</strong>{' '}
                        {queryAnalysis.decomposition.decomposition_strategy}
                      </p>
                      <p>
                        <strong>复杂度:</strong>{' '}
                        {queryAnalysis.decomposition.complexity_score}
                      </p>
                      <p>
                        <strong>子查询数:</strong>{' '}
                        {queryAnalysis.analysis.sub_queries_count}
                      </p>
                    </div>
                    <div className="bg-green-50 p-4 rounded-lg">
                      <h4 className="font-semibold mb-2">统计</h4>
                      <p>
                        <strong>检测类型:</strong>{' '}
                        {queryAnalysis.analysis.detected_query_type}
                      </p>
                      <p>
                        <strong>复杂度分数:</strong>{' '}
                        {queryAnalysis.analysis.complexity_score}
                      </p>
                      <p>
                        <strong>实体查询数:</strong>{' '}
                        {queryAnalysis.analysis.entities_count}
                      </p>
                      <p>
                        <strong>关系查询数:</strong>{' '}
                        {queryAnalysis.analysis.relations_count}
                      </p>
                    </div>
                    {queryAnalysis.decomposition.sub_queries?.length > 0 && (
                      <div className="bg-purple-50 p-4 rounded-lg">
                        <h4 className="font-semibold mb-2">子查询</h4>
                        <ul className="list-disc list-inside">
                          {queryAnalysis.decomposition.sub_queries.map(
                            (q: string, index: number) => (
                              <li key={index}>{q}</li>
                            )
                          )}
                        </ul>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </TabPane>

            <TabPane
              tab={
                <span>
                  <ShareAltOutlined />
                  推理查询
                </span>
              }
              key="reasoning"
            >
              <div className="space-y-4">
                <Button
                  type="primary"
                  onClick={() => setReasoningModalVisible(true)}
                  icon={<ShareAltOutlined />}
                >
                  开始推理查询
                </Button>
                {reasoningResult && (
                  <div className="bg-green-50 p-4 rounded-lg">
                    <h4 className="font-semibold mb-3">推理路径结果</h4>
                    <div className="space-y-2">
                      {reasoningResult.paths.map((path: any, index: number) => (
                        <div
                          key={index}
                          className="bg-white p-3 rounded border"
                        >
                          <div className="flex justify-between mb-2">
                            <span className="font-medium">
                              路径 {index + 1}
                            </span>
                            <span>
                              评分: {(path.path_score * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="text-sm text-gray-600">
                            实体: {path.entities.join(' → ')}
                          </div>
                          <div className="text-sm mt-1">{path.explanation}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </TabPane>

            <TabPane
              tab={
                <span>
                  <BranchesOutlined />
                  多源融合
                </span>
              }
              key="fusion"
            >
              <div className="space-y-4">
                <Button
                  type="primary"
                  onClick={() => setFusionModalVisible(true)}
                  icon={<BranchesOutlined />}
                >
                  配置多源融合
                </Button>
                {fusionResult && (
                  <div className="space-y-4">
                    <div className="bg-blue-50 p-4 rounded-lg">
                      <h4 className="font-semibold mb-2">融合结果</h4>
                      <p>
                        <strong>输出文档:</strong>{' '}
                        {fusionResult.statistics?.output_documents ?? 0}
                      </p>
                      <p>
                        <strong>一致性:</strong>{' '}
                        {(
                          ((fusionResult.fusion_results
                            ?.consistency_score as number) || 0) * 100
                        ).toFixed(1)}
                        %
                      </p>
                      <p>
                        <strong>融合策略:</strong>{' '}
                        {String(
                          fusionResult.fusion_results?.resolution_strategy ||
                            '-'
                        )}
                      </p>
                      <p>
                        <strong>冲突数:</strong>{' '}
                        {
                          (
                            fusionResult.fusion_results?.conflicts_detected ||
                            []
                          ).length
                        }
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </TabPane>

            <TabPane
              tab={
                <span>
                  <BarChartOutlined />
                  性能监控
                </span>
              }
              key="performance"
            >
              <div className="space-y-4">
                <div className="flex gap-4">
                  <Button
                    type="primary"
                    onClick={() => handleGetPerformanceStats()}
                    loading={statsQueryLoading}
                    icon={<BarChartOutlined />}
                  >
                    获取性能统计
                  </Button>
                  <Button onClick={() => setPerformanceModalVisible(true)}>
                    性能对比
                  </Button>
                  <Button onClick={() => setBenchmarkModalVisible(true)}>
                    基准测试
                  </Button>
                </div>
                {performanceStats && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="bg-blue-50 p-4 rounded-lg">
                      <h4 className="font-semibold mb-3">查询统计</h4>
                      <div className="space-y-2">
                        <p>
                          总查询数:{' '}
                          {performanceStats.query_statistics.total_queries}
                        </p>
                        <p>
                          成功查询:{' '}
                          {performanceStats.query_statistics.successful_queries}
                        </p>
                        <p>
                          失败查询:{' '}
                          {performanceStats.query_statistics.failed_queries}
                        </p>
                        <p>
                          平均响应时间:{' '}
                          {performanceStats.query_statistics.average_response_time.toFixed(
                            2
                          )}
                          ms
                        </p>
                      </div>
                    </div>
                    <div className="bg-green-50 p-4 rounded-lg">
                      <h4 className="font-semibold mb-3">性能指标</h4>
                      <div className="space-y-2">
                        <p>
                          平均检索时间:{' '}
                          {performanceStats.performance_metrics.average_retrieval_time.toFixed(
                            2
                          )}
                          ms
                        </p>
                        <p>
                          平均推理时间:{' '}
                          {performanceStats.performance_metrics.average_reasoning_time.toFixed(
                            2
                          )}
                          ms
                        </p>
                        <p>
                          平均融合时间:{' '}
                          {performanceStats.performance_metrics.average_fusion_time.toFixed(
                            2
                          )}
                          ms
                        </p>
                        <p>
                          缓存命中率:{' '}
                          {(
                            performanceStats.performance_metrics
                              .cache_hit_rate * 100
                          ).toFixed(1)}
                          %
                        </p>
                      </div>
                    </div>
                  </div>
                )}
                {performanceComparison && (
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <h4 className="font-semibold mb-2">性能对比结果</h4>
                    <pre className="text-xs overflow-auto">
                      {JSON.stringify(performanceComparison, null, 2)}
                    </pre>
                  </div>
                )}
                {benchmarkResult && (
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <h4 className="font-semibold mb-2">基准测试结果</h4>
                    <pre className="text-xs overflow-auto">
                      {JSON.stringify(benchmarkResult, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            </TabPane>

            <TabPane
              tab={
                <span>
                  <BugOutlined />
                  调试工具
                </span>
              }
              key="debug"
            >
              <div className="space-y-4">
                <Button
                  type="primary"
                  onClick={() => setDebugModalVisible(true)}
                  icon={<BugOutlined />}
                >
                  调试查询
                </Button>
                {debugResult && (
                  <div className="space-y-4">
                    <div className="bg-blue-50 p-4 rounded-lg">
                      <h4 className="font-semibold mb-2">推理解释</h4>
                      <p>
                        <strong>查询:</strong> {debugResult.query}
                      </p>
                      <p>
                        <strong>路径数:</strong>{' '}
                        {debugResult.reasoning_paths_count}
                      </p>
                      <div className="text-gray-800 whitespace-pre-line mt-3">
                        {debugResult.explanation}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </TabPane>

            <TabPane
              tab={
                <span>
                  <SettingOutlined />
                  系统配置
                </span>
              }
              key="config"
            >
              <div className="space-y-4">
                <div className="flex gap-4">
                  <Button
                    type="primary"
                    onClick={handleGetConfig}
                    loading={configLoading}
                    icon={<SettingOutlined />}
                  >
                    获取配置
                  </Button>
                  <Button
                    onClick={handleConsistencyCheck}
                    loading={consistencyLoading}
                    icon={<CheckCircleOutlined />}
                  >
                    一致性检查
                  </Button>
                </div>
                {graphConfig && (
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <div className="text-sm text-gray-600 mb-2">
                      状态: {graphConfig.status}
                    </div>
                    <pre className="text-xs overflow-auto">
                      {JSON.stringify(graphConfig.config, null, 2)}
                    </pre>
                  </div>
                )}
                {consistencyCheck && (
                  <div className="bg-green-50 p-4 rounded-lg mt-4">
                    <h4 className="font-semibold mb-2">一致性检查结果</h4>
                    <p>
                      <strong>一致性得分:</strong>{' '}
                      {(consistencyCheck.consistency_score * 100).toFixed(1)}%
                    </p>
                    <p>
                      <strong>文档数:</strong>{' '}
                      {consistencyCheck.documents_checked}
                    </p>
                    <p>
                      <strong>查询ID:</strong> {consistencyCheck.query_id}
                    </p>
                  </div>
                )}
              </div>
            </TabPane>
          </Tabs>
        </Card>

        {/* 查询结果 */}
        {result && (
          <Card title="查询结果" className="mb-6">
            <Tabs defaultActiveKey="answer">
              <TabPane tab="检索结果" key="answer">
                <div className="bg-blue-50 p-4 rounded-lg mb-4">
                  <h3 className="text-lg font-semibold text-blue-900 mb-2">
                    {result.query}
                  </h3>
                  {result.documents.length === 0 ? (
                    <div className="text-gray-600">暂无命中文档</div>
                  ) : (
                    <div className="space-y-3">
                      {result.documents.slice(0, 5).map((doc, index) => (
                        <div
                          key={index}
                          className="bg-white p-3 rounded border"
                        >
                          <div className="flex justify-between items-center mb-2">
                            <span className="font-medium">Top {index + 1}</span>
                            <span className="text-sm text-gray-600">
                              分数:{' '}
                              {(
                                ((doc.final_score ?? doc.score) || 0) * 100
                              ).toFixed(1)}
                              %
                            </span>
                          </div>
                          <div className="text-gray-800 whitespace-pre-line">
                            {doc.content || '-'}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                <div className="flex flex-wrap gap-2 mb-4">
                  <Tag color={getRetrievalModeColor(retrievalMode)}>
                    {getRetrievalModeIcon(retrievalMode)}{' '}
                    {retrievalMode.toUpperCase()}
                  </Tag>
                  <Tag color="blue">
                    一致性:{' '}
                    {(
                      ((result.fusion_results?.consistency_score as number) ||
                        0) * 100
                    ).toFixed(1)}
                    %
                  </Tag>
                  <Tag color="green">
                    用时:{' '}
                    {(
                      ((result.performance_metrics.total_time as number) || 0) *
                      1000
                    ).toFixed(0)}
                    ms
                  </Tag>
                  <Tag color="purple">
                    查询ID: {result.query_id.substring(0, 8)}...
                  </Tag>
                </div>
              </TabPane>

              <TabPane
                tab={`知识源 (${result.knowledge_sources.length})`}
                key="sources"
              >
                <div className="space-y-4">
                  {result.knowledge_sources.map((source, index) => (
                    <div key={index} className="border rounded-lg p-4">
                      <div className="flex justify-between items-center mb-2">
                        <Tag
                          color={
                            source.source_type === 'vector' ? 'blue' : 'green'
                          }
                        >
                          {source.source_type.toUpperCase()}
                        </Tag>
                        <span className="text-sm text-gray-600">
                          置信度: {(source.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="text-gray-800">{source.content}</div>
                      {source.metadata &&
                        Object.keys(source.metadata).length > 0 && (
                          <div className="mt-2 text-xs text-gray-500">
                            元数据: {JSON.stringify(source.metadata)}
                          </div>
                        )}
                    </div>
                  ))}
                </div>
              </TabPane>

              <TabPane
                tab={`图谱上下文 (${result.graph_context.entities.length}实体)`}
                key="context"
              >
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="text-lg font-semibold mb-3">实体</h4>
                    <div className="space-y-2 max-h-64 overflow-y-auto">
                      {result.graph_context.entities.map((entity, index) => (
                        <div key={index} className="bg-gray-50 p-2 rounded">
                          <span className="font-medium">
                            {entity.name || entity.id}
                          </span>
                          {entity.type && (
                            <Tag size="small" className="ml-2">
                              {entity.type}
                            </Tag>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                  <div>
                    <h4 className="text-lg font-semibold mb-3">关系</h4>
                    <div className="space-y-2 max-h-64 overflow-y-auto">
                      {result.graph_context.relations.map((relation, index) => (
                        <div key={index} className="bg-gray-50 p-2 rounded">
                          <span className="text-sm">
                            {relation.source}
                            <Tag size="small" color="blue" className="mx-2">
                              {relation.type}
                            </Tag>
                            {relation.target}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
                <div className="mt-4 flex gap-4 text-sm text-gray-600">
                  <span>扩展深度: {result.graph_context.expansion_depth}</span>
                  <span>
                    上下文置信度:{' '}
                    {(result.graph_context.confidence_score * 100).toFixed(1)}%
                  </span>
                </div>
              </TabPane>

              {includeReasoning && result.reasoning_results.length > 0 && (
                <TabPane
                  tab={`推理路径 (${result.reasoning_results.length})`}
                  key="reasoning"
                >
                  <div className="space-y-4">
                    {result.reasoning_results.map(path => (
                      <div key={path.path_id} className="border rounded-lg p-4">
                        <div className="flex justify-between items-center mb-3">
                          <Tag color="purple">
                            路径 {path.path_id.substring(0, 8)}...
                          </Tag>
                          <div className="flex gap-2 text-sm text-gray-600">
                            <span>
                              评分: {(path.path_score * 100).toFixed(1)}%
                            </span>
                            <span>跳数: {path.hops_count}</span>
                          </div>
                        </div>

                        <div className="mb-3">
                          <div className="flex flex-wrap items-center gap-2">
                            {path.entities.map((entity, index) => (
                              <React.Fragment key={index}>
                                <Tag color="blue">{entity}</Tag>
                                {index < path.entities.length - 1 && (
                                  <span className="text-gray-400">
                                    {path.relations[index] || '→'}
                                  </span>
                                )}
                              </React.Fragment>
                            ))}
                          </div>
                        </div>

                        <div className="text-gray-800 bg-gray-50 p-3 rounded">
                          {path.explanation}
                        </div>
                      </div>
                    ))}
                  </div>
                </TabPane>
              )}

              <TabPane tab="性能指标" key="performance">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center p-4 bg-blue-50 rounded">
                    <div className="text-2xl font-bold text-blue-600">
                      {(
                        ((result.performance_metrics.total_time as number) ||
                          0) * 1000
                      ).toFixed(0)}
                      ms
                    </div>
                    <div className="text-sm text-gray-600">总用时</div>
                  </div>
                  <div className="text-center p-4 bg-green-50 rounded">
                    <div className="text-2xl font-bold text-green-600">
                      {(
                        ((result.performance_metrics
                          .retrieval_time as number) || 0) * 1000
                      ).toFixed(0)}
                      ms
                    </div>
                    <div className="text-sm text-gray-600">检索用时</div>
                  </div>
                  <div className="text-center p-4 bg-purple-50 rounded">
                    <div className="text-2xl font-bold text-purple-600">
                      {(
                        ((result.performance_metrics
                          .reasoning_time as number) || 0) * 1000
                      ).toFixed(0)}
                      ms
                    </div>
                    <div className="text-sm text-gray-600">推理用时</div>
                  </div>
                  <div className="text-center p-4 bg-orange-50 rounded">
                    <div className="text-2xl font-bold text-orange-600">
                      {(
                        ((result.performance_metrics.fusion_time as number) ||
                          0) * 1000
                      ).toFixed(0)}
                      ms
                    </div>
                    <div className="text-sm text-gray-600">融合用时</div>
                  </div>
                </div>

                <div className="mt-6">
                  <h4 className="text-lg font-semibold mb-3">融合策略</h4>
                  <div className="bg-gray-50 p-4 rounded">
                    <div className="flex justify-between items-center mb-2">
                      <span>
                        策略:{' '}
                        {String(
                          result.fusion_results?.resolution_strategy || '-'
                        )}
                      </span>
                      <span>
                        一致性:{' '}
                        {(
                          ((result.fusion_results
                            ?.consistency_score as number) || 0) * 100
                        ).toFixed(1)}
                        %
                      </span>
                    </div>
                    <div className="text-sm text-gray-600">
                      冲突数:{' '}
                      {(result.fusion_results?.conflicts_detected || []).length}
                    </div>
                  </div>
                </div>
              </TabPane>
            </Tabs>
          </Card>
        )}

        {/* 功能说明 */}
        <Card title="GraphRAG 功能特性" className="mt-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="text-center p-4">
              <SearchOutlined className="text-3xl text-blue-500 mb-2" />
              <h4 className="font-semibold">向量检索</h4>
              <p className="text-sm text-gray-600">高效的语义相似度匹配</p>
            </div>
            <div className="text-center p-4">
              <NodeIndexOutlined className="text-3xl text-green-500 mb-2" />
              <h4 className="font-semibold">图谱推理</h4>
              <p className="text-sm text-gray-600">基于知识图谱的逻辑推理</p>
            </div>
            <div className="text-center p-4">
              <BranchesOutlined className="text-3xl text-purple-500 mb-2" />
              <h4 className="font-semibold">混合检索</h4>
              <p className="text-sm text-gray-600">融合多种检索策略</p>
            </div>
            <div className="text-center p-4">
              <ThunderboltOutlined className="text-3xl text-orange-500 mb-2" />
              <h4 className="font-semibold">智能优化</h4>
              <p className="text-sm text-gray-600">自适应选择最优策略</p>
            </div>
          </div>
        </Card>

        {/* 推理查询模态框 */}
        <Modal
          title="推理查询配置"
          visible={reasoningModalVisible}
          onCancel={() => setReasoningModalVisible(false)}
          footer={null}
          width={600}
        >
          <Form
            layout="vertical"
            onFinish={handleReasoningQuery}
            initialValues={{
              maxHops: 3,
              maxPaths: 10,
            }}
          >
            <Form.Item
              name="entity1"
              label="源实体"
              rules={[{ required: true, message: '请输入源实体' }]}
            >
              <Input placeholder="输入源实体名称" />
            </Form.Item>
            <Form.Item
              name="entity2"
              label="目标实体"
              rules={[{ required: true, message: '请输入目标实体' }]}
            >
              <Input placeholder="输入目标实体名称" />
            </Form.Item>
            <Form.Item name="maxHops" label="最大跳数">
              <Select>
                <Option value={1}>1</Option>
                <Option value={2}>2</Option>
                <Option value={3}>3</Option>
                <Option value={4}>4</Option>
                <Option value={5}>5</Option>
              </Select>
            </Form.Item>
            <Form.Item name="maxPaths" label="最大路径数">
              <Select>
                <Option value={5}>5</Option>
                <Option value={10}>10</Option>
                <Option value={20}>20</Option>
                <Option value={50}>50</Option>
              </Select>
            </Form.Item>
            <Form.Item>
              <Space>
                <Button
                  type="primary"
                  htmlType="submit"
                  loading={reasoningLoading}
                >
                  开始推理
                </Button>
                <Button onClick={() => setReasoningModalVisible(false)}>
                  取消
                </Button>
              </Space>
            </Form.Item>
          </Form>
        </Modal>

        {/* 多源融合模态框 */}
        <Modal
          title="多源融合查询配置"
          visible={fusionModalVisible}
          onCancel={() => setFusionModalVisible(false)}
          footer={null}
          width={700}
        >
          <Form layout="vertical" onFinish={handleMultiSourceFusion}>
            <Form.Item
              name="vector_results"
              label="向量检索结果 (JSON数组)"
              rules={[{ required: true, message: '请输入向量检索结果' }]}
              extra='示例: [{"content":"...","score":0.82,"metadata":{}}]'
            >
              <TextArea
                rows={4}
                placeholder={JSON.stringify(
                  [
                    {
                      content: '示例内容',
                      score: 0.82,
                      metadata: { source: 'vector_store' },
                    },
                  ],
                  null,
                  2
                )}
              />
            </Form.Item>
            <Form.Item
              name="graph_results"
              label="图谱检索结果 (JSON对象)"
              rules={[{ required: true, message: '请输入图谱检索结果' }]}
              extra='示例: {"entities":[...],"relations":[...]}'
            >
              <TextArea
                rows={4}
                placeholder={JSON.stringify(
                  { entities: [], relations: [] },
                  null,
                  2
                )}
              />
            </Form.Item>
            <Form.Item
              name="reasoning_results"
              label="推理结果 (JSON数组)"
              rules={[{ required: true, message: '请输入推理结果' }]}
              extra='示例: [{"path_id":"...","entities":["A","B"],"relations":["REL"],"path_score":0.7,"explanation":"...","hops_count":1}]'
            >
              <TextArea rows={4} placeholder={JSON.stringify([], null, 2)} />
            </Form.Item>
            <Form.Item
              name="confidence_threshold"
              label="置信度阈值"
              initialValue={0.6}
            >
              <Input type="number" min={0} max={1} step={0.05} />
            </Form.Item>
            <Form.Item>
              <Space>
                <Button
                  type="primary"
                  htmlType="submit"
                  loading={fusionLoading}
                >
                  开始融合查询
                </Button>
                <Button onClick={() => setFusionModalVisible(false)}>
                  取消
                </Button>
              </Space>
            </Form.Item>
          </Form>
        </Modal>

        {/* 调试模态框 */}
        <Modal
          title="调试查询配置"
          visible={debugModalVisible}
          onCancel={() => setDebugModalVisible(false)}
          footer={null}
          width={600}
        >
          <Form layout="vertical" onFinish={handleDebugExplain}>
            <Form.Item
              name="query"
              label="调试查询"
              rules={[{ required: true, message: '请输入调试查询' }]}
            >
              <TextArea rows={3} placeholder="输入要调试的查询内容" />
            </Form.Item>
            <Form.Item>
              <Space>
                <Button type="primary" htmlType="submit" loading={debugLoading}>
                  开始调试
                </Button>
                <Button onClick={() => setDebugModalVisible(false)}>
                  取消
                </Button>
              </Space>
            </Form.Item>
          </Form>
        </Modal>

        {/* 基准测试模态框 */}
        <Modal
          title="基准测试配置"
          visible={benchmarkModalVisible}
          onCancel={() => setBenchmarkModalVisible(false)}
          footer={null}
          width={700}
        >
          <Form
            layout="vertical"
            onFinish={handleBenchmark}
            initialValues={{
              max_docs: 10,
              include_reasoning: true,
              confidence_threshold: 0.6,
            }}
          >
            <Form.Item
              name="test_queries"
              label="测试查询 (JSON数组)"
              rules={[{ required: true, message: '请输入测试查询' }]}
              extra='示例: ["什么是AI?", "机器学习的应用有哪些?"]'
            >
              <TextArea
                rows={4}
                placeholder={JSON.stringify(
                  [
                    '什么是机器学习?',
                    '深度学习与机器学习的区别',
                    '人工智能的应用领域',
                  ],
                  null,
                  2
                )}
              />
            </Form.Item>
            <Form.Item name="max_docs" label="每次查询最大文档数">
              <Select>
                <Option value={5}>5</Option>
                <Option value={10}>10</Option>
                <Option value={20}>20</Option>
                <Option value={50}>50</Option>
              </Select>
            </Form.Item>
            <Form.Item
              name="include_reasoning"
              label="包含推理"
              valuePropName="checked"
            >
              <Switch />
            </Form.Item>
            <Form.Item name="confidence_threshold" label="置信度阈值">
              <Input type="number" min={0} max={1} step={0.05} />
            </Form.Item>
            <Form.Item>
              <Space>
                <Button
                  type="primary"
                  htmlType="submit"
                  loading={benchmarkLoading}
                >
                  开始测试
                </Button>
                <Button onClick={() => setBenchmarkModalVisible(false)}>
                  取消
                </Button>
              </Space>
            </Form.Item>
          </Form>
        </Modal>

        {/* 性能对比模态框 */}
        <Modal
          title="性能对比配置"
          visible={performanceModalVisible}
          onCancel={() => setPerformanceModalVisible(false)}
          footer={null}
          width={600}
        >
          <Form layout="vertical" onFinish={handlePerformanceComparison}>
            <Form.Item
              name="baselineDate"
              label="基线日期"
              rules={[{ required: true, message: '请选择基线日期' }]}
            >
              <DatePicker style={{ width: '100%' }} />
            </Form.Item>
            <Form.Item
              name="comparisonDate"
              label="对比日期"
              rules={[{ required: true, message: '请选择对比日期' }]}
            >
              <DatePicker style={{ width: '100%' }} />
            </Form.Item>
            <Form.Item>
              <Space>
                <Button
                  type="primary"
                  htmlType="submit"
                  loading={comparisonLoading}
                >
                  开始对比
                </Button>
                <Button onClick={() => setPerformanceModalVisible(false)}>
                  取消
                </Button>
              </Space>
            </Form.Item>
          </Form>
        </Modal>
      </div>
    </div>
  )
}

export default GraphRAGPage
