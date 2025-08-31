import React, { useState, useEffect } from 'react'
import { Card, Button, Input, Select, Switch, Spin, Alert, Tabs, Tag } from 'antd'
import { SearchOutlined, ThunderboltOutlined, NodeIndexOutlined, BranchesOutlined } from '@ant-design/icons'

const { TextArea } = Input
const { Option } = Select
const { TabPane } = Tabs

interface GraphRAGResponse {
  success: boolean
  query_id: string
  original_query: string
  final_answer: string
  knowledge_sources: Array<{
    source_type: string
    content: string
    confidence: number
    metadata: any
  }>
  graph_context: {
    entities: Array<any>
    relations: Array<any>
    expansion_depth: number
    confidence_score: number
  }
  reasoning_paths: Array<{
    path_id: string
    entities: string[]
    relations: string[]
    path_score: number
    explanation: string
    hops_count: number
  }>
  fusion_result: {
    confidence_score: number
    fusion_strategy: string
    source_weights: Record<string, number>
  }
  performance_metrics: {
    total_time: number
    query_analysis_time: number
    retrieval_time: number
    reasoning_time: number
  }
}

const GraphRAGPage: React.FC = () => {
  const [query, setQuery] = useState('')
  const [retrievalMode, setRetrievalMode] = useState('hybrid')
  const [maxDocs, setMaxDocs] = useState(10)
  const [includeReasoning, setIncludeReasoning] = useState(true)
  const [expansionDepth, setExpansionDepth] = useState(2)
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.6)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<GraphRAGResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async () => {
    if (!query.trim()) {
      setError('请输入查询内容')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await fetch('/api/v1/graphrag/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query,
          retrieval_mode: retrievalMode,
          max_docs: maxDocs,
          include_reasoning: includeReasoning,
          expansion_depth: expansionDepth,
          confidence_threshold: confidenceThreshold,
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : '查询失败')
    } finally {
      setLoading(false)
    }
  }

  const getRetrievalModeIcon = (mode: string) => {
    switch (mode) {
      case 'vector': return <SearchOutlined />
      case 'graph': return <NodeIndexOutlined />
      case 'hybrid': return <BranchesOutlined />
      case 'adaptive': return <ThunderboltOutlined />
      default: return <SearchOutlined />
    }
  }

  const getRetrievalModeColor = (mode: string) => {
    switch (mode) {
      case 'vector': return 'blue'
      case 'graph': return 'green'
      case 'hybrid': return 'purple'
      case 'adaptive': return 'orange'
      default: return 'default'
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
                onChange={(e) => setQuery(e.target.value)}
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

        {/* 查询结果 */}
        {result && (
          <Card title="查询结果" className="mb-6">
            <Tabs defaultActiveKey="answer">
              <TabPane tab="最终答案" key="answer">
                <div className="bg-blue-50 p-4 rounded-lg mb-4">
                  <h3 className="text-lg font-semibold text-blue-900 mb-2">
                    {result.original_query}
                  </h3>
                  <div className="text-gray-800 whitespace-pre-line">
                    {result.final_answer}
                  </div>
                </div>

                <div className="flex flex-wrap gap-2 mb-4">
                  <Tag color={getRetrievalModeColor(retrievalMode)}>
                    {getRetrievalModeIcon(retrievalMode)} {retrievalMode.toUpperCase()}
                  </Tag>
                  <Tag color="blue">
                    置信度: {(result.fusion_result.confidence_score * 100).toFixed(1)}%
                  </Tag>
                  <Tag color="green">
                    用时: {result.performance_metrics.total_time.toFixed(0)}ms
                  </Tag>
                  <Tag color="purple">
                    查询ID: {result.query_id.substring(0, 8)}...
                  </Tag>
                </div>
              </TabPane>

              <TabPane tab={`知识源 (${result.knowledge_sources.length})`} key="sources">
                <div className="space-y-4">
                  {result.knowledge_sources.map((source, index) => (
                    <div key={index} className="border rounded-lg p-4">
                      <div className="flex justify-between items-center mb-2">
                        <Tag color={source.source_type === 'vector' ? 'blue' : 'green'}>
                          {source.source_type.toUpperCase()}
                        </Tag>
                        <span className="text-sm text-gray-600">
                          置信度: {(source.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="text-gray-800">
                        {source.content}
                      </div>
                      {source.metadata && Object.keys(source.metadata).length > 0 && (
                        <div className="mt-2 text-xs text-gray-500">
                          元数据: {JSON.stringify(source.metadata)}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </TabPane>

              <TabPane tab={`图谱上下文 (${result.graph_context.entities.length}实体)`} key="context">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="text-lg font-semibold mb-3">实体</h4>
                    <div className="space-y-2 max-h-64 overflow-y-auto">
                      {result.graph_context.entities.map((entity, index) => (
                        <div key={index} className="bg-gray-50 p-2 rounded">
                          <span className="font-medium">{entity.name || entity.id}</span>
                          {entity.type && (
                            <Tag size="small" className="ml-2">{entity.type}</Tag>
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
                  <span>上下文置信度: {(result.graph_context.confidence_score * 100).toFixed(1)}%</span>
                </div>
              </TabPane>

              {includeReasoning && result.reasoning_paths.length > 0 && (
                <TabPane tab={`推理路径 (${result.reasoning_paths.length})`} key="reasoning">
                  <div className="space-y-4">
                    {result.reasoning_paths.map((path) => (
                      <div key={path.path_id} className="border rounded-lg p-4">
                        <div className="flex justify-between items-center mb-3">
                          <Tag color="purple">
                            路径 {path.path_id.substring(0, 8)}...
                          </Tag>
                          <div className="flex gap-2 text-sm text-gray-600">
                            <span>评分: {(path.path_score * 100).toFixed(1)}%</span>
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
                      {result.performance_metrics.total_time.toFixed(0)}ms
                    </div>
                    <div className="text-sm text-gray-600">总用时</div>
                  </div>
                  <div className="text-center p-4 bg-green-50 rounded">
                    <div className="text-2xl font-bold text-green-600">
                      {result.performance_metrics.query_analysis_time.toFixed(0)}ms
                    </div>
                    <div className="text-sm text-gray-600">查询分析</div>
                  </div>
                  <div className="text-center p-4 bg-purple-50 rounded">
                    <div className="text-2xl font-bold text-purple-600">
                      {result.performance_metrics.retrieval_time.toFixed(0)}ms
                    </div>
                    <div className="text-sm text-gray-600">检索用时</div>
                  </div>
                  <div className="text-center p-4 bg-orange-50 rounded">
                    <div className="text-2xl font-bold text-orange-600">
                      {result.performance_metrics.reasoning_time.toFixed(0)}ms
                    </div>
                    <div className="text-sm text-gray-600">推理用时</div>
                  </div>
                </div>

                <div className="mt-6">
                  <h4 className="text-lg font-semibold mb-3">融合策略</h4>
                  <div className="bg-gray-50 p-4 rounded">
                    <div className="flex justify-between items-center mb-2">
                      <span>策略: {result.fusion_result.fusion_strategy}</span>
                      <span>总置信度: {(result.fusion_result.confidence_score * 100).toFixed(1)}%</span>
                    </div>
                    <div className="text-sm text-gray-600">
                      权重分配: {Object.entries(result.fusion_result.source_weights)
                        .map(([key, value]) => `${key}: ${(value * 100).toFixed(1)}%`)
                        .join(', ')}
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
      </div>
    </div>
  )
}

export default GraphRAGPage