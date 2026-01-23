import React, { useState, useEffect } from 'react'
import { logger } from '../utils/logger'
import {
  Card,
  Input,
  Button,
  Space,
  Table,
  Alert,
  Tooltip,
  Row,
  Col,
  Statistic,
  Tabs,
  Typography,
  Divider,
  Tag,
  Progress,
  Select,
  message,
  Spin,
  Empty,
} from 'antd'
import {
  PlayCircleOutlined,
  SaveOutlined,
  HistoryOutlined,
  DatabaseOutlined,
  NodeIndexOutlined,
  ShareAltOutlined,
  ClockCircleOutlined,
  ThunderboltOutlined,
  FileTextOutlined,
  CopyOutlined,
  DeleteOutlined,
  ExportOutlined,
} from '@ant-design/icons'
import {
  knowledgeGraphService,
  type QueryResult,
  type QueryTemplate,
} from '../services/knowledgeGraphService'

const { Title, Text, Paragraph } = Typography
const { TextArea } = Input
const { TabPane } = Tabs

interface QueryHistory {
  id: string
  query: string
  timestamp: string
  execution_time: number
  status: 'success' | 'error'
  error?: string
  result_count: number
}

const KnowledgeGraphQueryEngine: React.FC = () => {
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [queryResult, setQueryResult] = useState<QueryResult | null>(null)
  const [queryHistory, setQueryHistory] = useState<QueryHistory[]>([])
  const [queryTemplates, setQueryTemplates] = useState<QueryTemplate[]>([])
  const [selectedTemplate, setSelectedTemplate] = useState<string>('')
  const [activeTab, setActiveTab] = useState('query')
  const [qualityIssues, setQualityIssues] = useState<any[]>([])
  const [qualityReport, setQualityReport] = useState<any>(null)
  const [slowQueries, setSlowQueries] = useState<any[]>([])
  const [graphStats, setGraphStats] = useState<any>(null)

  // 新增状态：推理功能
  const [reasoningQuery, setReasoningQuery] = useState('')
  const [reasoningResult, setReasoningResult] = useState<any>(null)
  const [reasoningStrategy, setReasoningStrategy] = useState('default')
  const [strategyPerformance, setStrategyPerformance] = useState<any[]>([])

  // GraphRAG功能状态
  const [graphragQuestion, setGraphragQuestion] = useState('')
  const [graphragResult, setGraphragResult] = useState<any>(null)
  const [graphragType, setGraphragType] = useState('causal')

  const [langGraphDemo, setLangGraphDemo] = useState<any>(null)
  const [demoType, setDemoType] = useState<
    'context-api' | 'durability' | 'caching' | 'hooks'
  >('context-api')

  // RAG集成状态
  const [ragQuestion, setRagQuestion] = useState('')
  const [ragResult, setRagResult] = useState<any>(null)

  // 实体管理状态
  const [newEntity, setNewEntity] = useState({
    canonical_form: '',
    entity_type: '',
    confidence: 0.9,
  })
  const [entitySearchTerm, setEntitySearchTerm] = useState('')
  const [searchedEntities, setSearchedEntities] = useState<any[]>([])
  const [batchEntitiesJson, setBatchEntitiesJson] = useState('')

  // 关系管理状态
  const [newRelation, setNewRelation] = useState({
    source_entity_id: '',
    target_entity_id: '',
    relation_type: '',
    context: '',
    source_sentence: '',
  })

  // 加载查询模板
  useEffect(() => {
    loadQueryTemplates()
    loadQualityData()
    loadPerformanceData()
    loadGraphStats()
    loadStrategyPerformance()
  }, [])

  const loadQueryTemplates = async () => {
    try {
      const templates = await knowledgeGraphService.getQueryTemplates()
      setQueryTemplates(templates)
    } catch (error) {
      logger.error('加载查询模板失败:', error)
      setQueryTemplates([])
    }
  }

  const loadQualityData = async () => {
    try {
      const issues = await knowledgeGraphService.getQualityIssues()
      const report = await knowledgeGraphService.getQualityReport()
      setQualityIssues(issues)
      setQualityReport(report)
    } catch (error) {
      logger.error('加载质量数据失败:', error)
    }
  }

  const loadPerformanceData = async () => {
    try {
      const queries = await knowledgeGraphService.getSlowQueries()
      setSlowQueries(queries)
    } catch (error) {
      logger.error('加载性能数据失败:', error)
    }
  }

  const loadGraphStats = async () => {
    try {
      const stats = await knowledgeGraphService.getStatistics()
      setGraphStats(stats)
    } catch (error) {
      logger.error('加载统计数据失败:', error)
    }
  }

  // 加载推理策略性能
  const loadStrategyPerformance = async () => {
    try {
      const strategies = await knowledgeGraphService.getStrategyPerformance()
      setStrategyPerformance(strategies)
    } catch (error) {
      logger.error('加载策略性能失败:', error)
      setStrategyPerformance([])
    }
  }

  // 执行推理查询
  const executeReasoningQuery = async () => {
    if (!reasoningQuery.trim()) {
      message.error('请输入推理查询')
      return
    }

    setLoading(true)
    try {
      const result = await knowledgeGraphService.queryReasoning({
        query: reasoningQuery,
        reasoning_strategy: reasoningStrategy,
        max_depth: 3,
        confidence_threshold: 0.7,
      })
      setReasoningResult(result)
      message.success('推理查询执行成功')
    } catch (error) {
      logger.error('推理查询失败:', error)
      message.error('推理查询失败')
      setReasoningResult(null)
    } finally {
      setLoading(false)
    }
  }

  // 执行GraphRAG查询
  const executeGraphRAGQuery = async () => {
    if (!graphragQuestion.trim()) {
      message.error('请输入GraphRAG问题')
      return
    }

    setLoading(true)
    try {
      const result = await knowledgeGraphService.graphragReasoningQuery({
        question: graphragQuestion,
        reasoning_type: graphragType as any,
        evidence_threshold: 0.8,
      })
      setGraphragResult(result)
      message.success('GraphRAG查询执行成功')
    } catch (error) {
      logger.error('GraphRAG查询失败:', error)
      message.error('GraphRAG查询失败')
      setGraphragResult(null)
    } finally {
      setLoading(false)
    }
  }

  // 执行LangGraph演示（调用真实API，失败即提示）
  const executeLangGraphDemo = async () => {
    setLoading(true)
    try {
      const result = await knowledgeGraphService.demoContextApi({
        demo_type: demoType,
        parameters: { example_input: 'demo_data' },
      })
      setLangGraphDemo(result)
      message.success('LangGraph演示执行成功')
    } catch (error) {
      logger.error('LangGraph演示失败:', error)
      message.error('LangGraph演示失败')
      setLangGraphDemo(null)
    } finally {
      setLoading(false)
    }
  }

  // 执行RAG集成查询
  const executeRAGQuery = async () => {
    if (!ragQuestion.trim()) {
      message.error('请输入RAG问题')
      return
    }

    setLoading(true)
    try {
      const result = await knowledgeGraphService.ragGraphragQuery({
        question: ragQuestion,
        document_collection: 'default',
        hybrid_search: true,
        rerank_results: true,
      })
      setRagResult(result)
      message.success('RAG查询执行成功')
    } catch (error) {
      logger.error('RAG查询失败:', error)
      message.error('RAG查询失败')
      setRagResult(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <DatabaseOutlined style={{ marginRight: '8px' }} />
          图查询引擎 (扩展版)
        </Title>
        <Paragraph type="secondary">
          集成了推理查询、GraphRAG、LangGraph演示和RAG集成的知识图谱查询引擎
        </Paragraph>
      </div>

      {/* 统计信息 */}
      <Row gutter={16} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="总查询次数"
              value={queryHistory.length}
              prefix={<DatabaseOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="成功查询"
              value={queryHistory.filter(h => h.status === 'success').length}
              valueStyle={{ color: '#3f8600' }}
              prefix={<ThunderboltOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="平均执行时间"
              value={
                queryHistory.length > 0
                  ? Math.round(
                      queryHistory.reduce(
                        (sum, h) => sum + h.execution_time,
                        0
                      ) / queryHistory.length
                    )
                  : 0
              }
              suffix="ms"
              valueStyle={{ color: '#1890ff' }}
              prefix={<ClockCircleOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="可用模板"
              value={queryTemplates.length}
              prefix={<FileTextOutlined />}
            />
          </Card>
        </Col>
      </Row>

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="推理查询" key="reasoning">
          <Card title="知识图谱推理" style={{ marginBottom: '16px' }}>
            <Row gutter={16}>
              <Col span={16}>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <div>
                    <Text strong>推理查询:</Text>
                    <Input.TextArea
                      placeholder="输入推理查询，例如：张三在哪家公司工作？"
                      value={reasoningQuery}
                      onChange={e => setReasoningQuery(e.target.value)}
                      name="kg-reasoning-query"
                      rows={3}
                    />
                  </div>
                  <div>
                    <Text strong>推理策略:</Text>
                    <Select
                      value={reasoningStrategy}
                      onChange={setReasoningStrategy}
                      name="kg-reasoning-strategy"
                      style={{ width: '100%' }}
                    >
                      <Select.Option value="default">默认推理</Select.Option>
                      <Select.Option value="deep_reasoning">
                        深度推理
                      </Select.Option>
                      <Select.Option value="causal">因果推理</Select.Option>
                      <Select.Option value="temporal">时序推理</Select.Option>
                    </Select>
                  </div>
                  <Button
                    type="primary"
                    onClick={executeReasoningQuery}
                    loading={loading}
                    disabled={!reasoningQuery.trim()}
                  >
                    执行推理查询
                  </Button>
                </Space>

                {reasoningResult && (
                  <Card title="推理结果" style={{ marginTop: '16px' }}>
                    <div style={{ marginBottom: '16px' }}>
                      <Text strong>结果: </Text>
                      <Text>{reasoningResult.result}</Text>
                    </div>
                    <div style={{ marginBottom: '16px' }}>
                      <Text strong>总体置信度: </Text>
                      <Progress
                        percent={Math.round(
                          reasoningResult.total_confidence * 100
                        )}
                        status={
                          reasoningResult.total_confidence > 0.8
                            ? 'success'
                            : 'normal'
                        }
                      />
                    </div>
                    <div>
                      <Text strong>推理过程:</Text>
                      <div style={{ marginTop: '8px' }}>
                        {reasoningResult.reasoning_trace?.map(
                          (step: any, index: number) => (
                            <Card
                              key={index}
                              size="small"
                              style={{ marginBottom: '8px' }}
                            >
                              <Row justify="space-between">
                                <Col span={18}>
                                  <Text strong>
                                    步骤{step.step}: {step.operation}
                                  </Text>
                                  <br />
                                  <Text type="secondary">
                                    {step.explanation}
                                  </Text>
                                </Col>
                                <Col span={6}>
                                  <Progress
                                    type="circle"
                                    percent={Math.round(step.confidence * 100)}
                                    width={60}
                                  />
                                </Col>
                              </Row>
                            </Card>
                          )
                        )}
                      </div>
                    </div>
                  </Card>
                )}
              </Col>

              <Col span={8}>
                <Card title="推理策略性能">
                  {strategyPerformance.length > 0 ? (
                    <div>
                      {strategyPerformance.map((strategy, index) => (
                        <Card
                          key={index}
                          size="small"
                          style={{ marginBottom: '12px' }}
                        >
                          <div style={{ marginBottom: '8px' }}>
                            <Text strong>{strategy.strategy_name}</Text>
                          </div>
                          <Row gutter={8}>
                            <Col span={12}>
                              <Statistic
                                title="平均时间"
                                value={strategy.avg_execution_time}
                                suffix="ms"
                                valueStyle={{ fontSize: '14px' }}
                              />
                            </Col>
                            <Col span={12}>
                              <Statistic
                                title="准确率"
                                value={Math.round(
                                  strategy.accuracy_score * 100
                                )}
                                suffix="%"
                                valueStyle={{ fontSize: '14px' }}
                              />
                            </Col>
                          </Row>
                        </Card>
                      ))}
                    </div>
                  ) : (
                    <Empty description="暂无策略数据" />
                  )}
                </Card>
              </Col>
            </Row>
          </Card>
        </TabPane>

        <TabPane tab="GraphRAG" key="graphrag">
          <Row gutter={16}>
            <Col span={14}>
              <Card title="GraphRAG查询" style={{ marginBottom: '16px' }}>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <div>
                    <Text strong>问题:</Text>
                    <Input.TextArea
                      placeholder="输入问题，GraphRAG将结合知识图谱和文档进行回答"
                      value={graphragQuestion}
                      onChange={e => setGraphragQuestion(e.target.value)}
                      name="kg-graphrag-question"
                      rows={3}
                    />
                  </div>
                  <div>
                    <Text strong>推理类型:</Text>
                    <Select
                      value={graphragType}
                      onChange={setGraphragType}
                      name="kg-graphrag-type"
                      style={{ width: '100%' }}
                    >
                      <Select.Option value="causal">因果推理</Select.Option>
                      <Select.Option value="comparative">
                        对比分析
                      </Select.Option>
                      <Select.Option value="temporal">时序分析</Select.Option>
                      <Select.Option value="compositional">
                        组合推理
                      </Select.Option>
                    </Select>
                  </div>
                  <Button
                    type="primary"
                    onClick={executeGraphRAGQuery}
                    loading={loading}
                    disabled={!graphragQuestion.trim()}
                  >
                    执行GraphRAG查询
                  </Button>
                </Space>

                {graphragResult && (
                  <Card title="GraphRAG结果" style={{ marginTop: '16px' }}>
                    <div style={{ marginBottom: '16px' }}>
                      <Text strong>答案: </Text>
                      <Paragraph>{graphragResult.answer}</Paragraph>
                    </div>
                    <div style={{ marginBottom: '16px' }}>
                      <Text strong>推理类型: </Text>
                      <Tag color="blue">{graphragResult.reasoning_type}</Tag>
                    </div>
                  </Card>
                )}
              </Card>
            </Col>

            <Col span={10}>
              <Card title="RAG集成查询" style={{ marginBottom: '16px' }}>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <div>
                    <Text strong>RAG问题:</Text>
                    <Input.TextArea
                      placeholder="结合文档检索和知识图谱的混合查询"
                      value={ragQuestion}
                      onChange={e => setRagQuestion(e.target.value)}
                      name="kg-rag-question"
                      rows={2}
                    />
                  </div>
                  <Button
                    type="primary"
                    onClick={executeRAGQuery}
                    loading={loading}
                    disabled={!ragQuestion.trim()}
                  >
                    执行RAG查询
                  </Button>
                </Space>

                {ragResult && (
                  <div style={{ marginTop: '16px' }}>
                    <div style={{ marginBottom: '12px' }}>
                      <Text strong>答案:</Text>
                      <Paragraph>{ragResult.answer}</Paragraph>
                    </div>
                  </div>
                )}
              </Card>

              <Card title="LangGraph演示">
                <Space direction="vertical" style={{ width: '100%' }}>
                  <div>
                    <Text strong>演示类型:</Text>
                    <Select
                      value={demoType}
                      onChange={setDemoType}
                      name="kg-langgraph-demo"
                      style={{ width: '100%' }}
                    >
                      <Select.Option value="context-api">
                        上下文API
                      </Select.Option>
                      <Select.Option value="durability">持久化</Select.Option>
                      <Select.Option value="caching">缓存机制</Select.Option>
                      <Select.Option value="hooks">钩子函数</Select.Option>
                    </Select>
                  </div>
                  <Button
                    type="primary"
                    onClick={executeLangGraphDemo}
                    loading={loading}
                    block
                  >
                    执行演示
                  </Button>
                </Space>

                {langGraphDemo && (
                  <div style={{ marginTop: '16px' }}>
                    <div style={{ marginBottom: '8px' }}>
                      <Text strong>演示结果:</Text>
                      <div
                        style={{
                          background: '#f5f5f5',
                          padding: '8px',
                          borderRadius: '4px',
                          marginTop: '4px',
                        }}
                      >
                        <Text code>
                          {JSON.stringify(langGraphDemo.result, null, 2)}
                        </Text>
                      </div>
                    </div>
                    <div style={{ marginBottom: '8px' }}>
                      <Text strong>执行时间:</Text>{' '}
                      {langGraphDemo.execution_time}ms
                    </div>
                  </div>
                )}
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab="基础查询" key="query">
          <Alert
            message="基础查询功能"
            description="此标签页包含传统的Cypher查询功能。"
            type="info"
            showIcon
            style={{ marginBottom: '16px' }}
          />

          <Card title="Cypher查询编辑器">
            <Space direction="vertical" style={{ width: '100%' }}>
              <TextArea
                value={query}
                onChange={e => setQuery(e.target.value)}
                placeholder="输入Cypher查询语句..."
                name="kg-cypher-query"
                rows={8}
                style={{ fontFamily: 'Monaco, monospace' }}
              />
              <Row justify="space-between">
                <Col>
                  <Space>
                    <Button
                      type="primary"
                      icon={<PlayCircleOutlined />}
                      loading={loading}
                    >
                      执行查询
                    </Button>
                    <Button icon={<SaveOutlined />}>保存查询</Button>
                    <Button onClick={() => setQuery('')}>清空</Button>
                  </Space>
                </Col>
              </Row>
            </Space>
          </Card>
        </TabPane>
      </Tabs>
    </div>
  )
}

export default KnowledgeGraphQueryEngine
