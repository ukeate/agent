import React, { useState, useEffect } from 'react'
import {
import { logger } from '../utils/logger'
  Card,
  Button,
  Input,
  Select,
  Tag,
  Space,
  Row,
  Col,
  Progress,
  Timeline,
  Statistic,
  Alert,
  Tabs,
  Collapse,
  List,
  Badge,
  Divider,
  message,
  Spin,
  Rate
} from 'antd'
import {
  SearchOutlined,
  ExpandOutlined,
  RobotOutlined,
  CheckCircleOutlined,
  SyncOutlined,
  BulbOutlined,
  BarChartOutlined,
  QuestionCircleOutlined,
  ThunderboltOutlined
} from '@ant-design/icons'
import agenticRagService, {
  QueryIntentType,
  ExpansionStrategyType,
  RetrievalStrategyType,
  StreamEventType,
  type AgenticQueryResponse,
  type StreamEvent,
  type AgenticRagStats
} from '../services/agenticRagService'

const { TextArea } = Input
const { Option } = Select
const { Panel } = Collapse
const { TabPane } = Tabs

const AgenticRagPage: React.FC = () => {
  // 查询状态
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [streaming, setStreaming] = useState(false)
  const [sessionId] = useState(`session-${Date.now()}`)
  
  // 策略选择
  const [expansionStrategies, setExpansionStrategies] = useState<ExpansionStrategyType[]>([
    ExpansionStrategyType.SEMANTIC,
    ExpansionStrategyType.CONTEXTUAL
  ])
  const [retrievalStrategies, setRetrievalStrategies] = useState<RetrievalStrategyType[]>([
    RetrievalStrategyType.HYBRID,
    RetrievalStrategyType.KEYWORD
  ])
  
  // 查询结果
  const [queryResult, setQueryResult] = useState<AgenticQueryResponse | null>(null)
  const [streamProgress, setStreamProgress] = useState(0)
  const [streamEvents, setStreamEvents] = useState<StreamEvent[]>([])
  
  // 统计信息
  const [stats, setStats] = useState<AgenticRagStats | null>(null)
  const [healthStatus, setHealthStatus] = useState<'healthy' | 'unhealthy' | 'unknown'>('unknown')
  
  // 历史记录
  const [queryHistory, setQueryHistory] = useState<Array<{
    query: string
    timestamp: Date
    queryId: string
    quality: number
  }>>([])

  // 加载统计信息
  useEffect(() => {
    loadStats()
    checkHealth()
    const interval = setInterval(() => {
      loadStats()
      checkHealth()
    }, 30000)
    return () => clearInterval(interval)
  }, [])

  const loadStats = async () => {
    try {
      const data = await agenticRagService.getStats()
      setStats(data)
    } catch (error) {
      logger.error('加载统计信息失败:', error)
    }
  }

  const checkHealth = async () => {
    try {
      const health = await agenticRagService.healthCheck()
      setHealthStatus(health.status)
    } catch (error) {
      setHealthStatus('unhealthy')
    }
  }

  // 执行智能查询
  const handleQuery = async (useStream: boolean = false) => {
    if (!query.trim()) {
      message.warning('请输入查询内容')
      return
    }

    setLoading(true)
    setStreaming(useStream)
    setQueryResult(null)
    setStreamEvents([])
    setStreamProgress(0)

    try {
      if (useStream) {
        // 流式查询
        await agenticRagService.queryStream(
          {
            query,
            expansion_strategies: expansionStrategies,
            retrieval_strategies: retrievalStrategies,
            max_results: 10,
            include_explanation: true,
            session_id: sessionId
          },
          (event: StreamEvent) => {
            setStreamEvents(prev => [...prev, event])
            setStreamProgress(event.progress * 100)
            
            if (event.event_type === StreamEventType.COMPLETE) {
              setQueryResult(event.data.results)
              addToHistory(event.data.results)
            } else if (event.event_type === StreamEventType.ERROR) {
              message.error(event.message)
            }
          }
        )
      } else {
        // 普通查询
        const result = await agenticRagService.query({
          query,
          expansion_strategies: expansionStrategies,
          retrieval_strategies: retrievalStrategies,
          max_results: 10,
          include_explanation: true,
          session_id: sessionId
        })
        
        setQueryResult(result)
        addToHistory(result)
        
        if (result.success) {
          message.success('智能检索完成')
        } else {
          message.error(result.error || '检索失败')
        }
      }
    } catch (error) {
      message.error('查询失败，请重试')
    } finally {
      setLoading(false)
      setStreaming(false)
    }
  }

  const addToHistory = (result: AgenticQueryResponse) => {
    if (result.success) {
      setQueryHistory(prev => [{
        query,
        timestamp: new Date(),
        queryId: result.query_id,
        quality: result.validation_result?.overall_quality || 0
      }, ...prev.slice(0, 9)])
    }
  }

  // 提交反馈
  const handleFeedback = async (queryId: string, rating: number) => {
    try {
      await agenticRagService.submitFeedback({
        query_id: queryId,
        ratings: { overall: rating },
        comments: '用户评分'
      })
      message.success('感谢您的反馈')
    } catch (error) {
      message.error('反馈提交失败')
    }
  }

  // 渲染查询分析结果
  const renderQueryAnalysis = () => {
    if (!queryResult?.query_analysis) return null
    
    const analysis = queryResult.query_analysis
    return (
      <Card title="查询分析" size="small">
        <Space direction="vertical" style={{ width: '100%' }}>
          <div>
            <span>意图类型：</span>
            <Tag color="blue">{analysis.intent_type}</Tag>
            <span className="ml-4">置信度：</span>
            <Progress 
              percent={analysis.confidence * 100} 
              size="small" 
              style={{ width: 100 }}
            />
          </div>
          <div>
            <span>复杂度评分：</span>
            <Progress 
              percent={analysis.complexity_score * 100} 
              size="small" 
              strokeColor="#52c41a"
              style={{ width: 100 }}
            />
          </div>
          <div>
            <span>关键词：</span>
            {analysis.keywords.map(kw => (
              <Tag key={kw}>{kw}</Tag>
            ))}
          </div>
          <div>
            <span>实体：</span>
            {analysis.entities.map(entity => (
              <Tag key={entity} color="orange">{entity}</Tag>
            ))}
          </div>
        </Space>
      </Card>
    )
  }

  // 渲染扩展查询
  const renderExpandedQueries = () => {
    if (!queryResult?.expanded_queries) return null
    
    return (
      <Card title="查询扩展" size="small">
        <Collapse ghost>
          {queryResult.expanded_queries.map((eq, idx) => (
            <Panel 
              key={idx}
              header={
                <div>
                  <Tag color="purple">{eq.strategy}</Tag>
                  <span className="ml-2">置信度: {(eq.confidence * 100).toFixed(1)}%</span>
                </div>
              }
            >
              <List
                size="small"
                dataSource={eq.expanded_queries}
                renderItem={q => (
                  <List.Item>
                    <QuestionCircleOutlined className="mr-2" />
                    {q}
                  </List.Item>
                )}
              />
              {eq.sub_questions && (
                <div className="mt-2">
                  <strong>子问题：</strong>
                  <List
                    size="small"
                    dataSource={eq.sub_questions}
                    renderItem={q => <List.Item>{q}</List.Item>}
                  />
                </div>
              )}
            </Panel>
          ))}
        </Collapse>
      </Card>
    )
  }

  // 渲染检索结果
  const renderRetrievalResults = () => {
    if (!queryResult?.retrieval_results) return null
    
    return (
      <Card title="多智能体检索结果" size="small">
        <Tabs>
          {queryResult.retrieval_results.map((result, idx) => (
            <TabPane 
              key={idx}
              tab={
                <span>
                  <RobotOutlined />
                  {result.agent_type}
                  <Badge 
                    count={result.results.length} 
                    style={{ marginLeft: 8 }}
                  />
                </span>
              }
            >
              <div className="mb-2">
                <Space>
                  <span>评分: {result.score.toFixed(2)}</span>
                  <span>置信度: {(result.confidence * 100).toFixed(1)}%</span>
                  <span>处理时间: {result.processing_time.toFixed(2)}s</span>
                </Space>
              </div>
              <List
                size="small"
                dataSource={result.results}
                renderItem={item => (
                  <List.Item>
                    <List.Item.Meta
                      title={
                        <div>
                          <Tag color="blue">{item.content_type || 'text'}</Tag>
                          <span className="ml-2">相关度: {(item.score * 100).toFixed(1)}%</span>
                        </div>
                      }
                      description={
                        <div>
                          <div className="text-sm">{item.content.substring(0, 200)}...</div>
                          {item.file_path && (
                            <div className="text-xs text-gray-500 mt-1">
                              来源: {item.file_path}
                            </div>
                          )}
                        </div>
                      }
                    />
                  </List.Item>
                )}
              />
            </TabPane>
          ))}
        </Tabs>
      </Card>
    )
  }

  // 渲染验证结果
  const renderValidationResult = () => {
    if (!queryResult?.validation_result) return null
    
    const validation = queryResult.validation_result
    return (
      <Card title="结果验证" size="small">
        <Space direction="vertical" style={{ width: '100%' }}>
          <div>
            <span>整体质量：</span>
            <Progress 
              percent={validation.overall_quality * 100} 
              strokeColor={validation.overall_quality > 0.7 ? '#52c41a' : '#faad14'}
            />
          </div>
          <div>
            <span>整体置信度：</span>
            <Progress 
              percent={validation.overall_confidence * 100} 
              size="small"
            />
          </div>
          {validation.conflicts.length > 0 && (
            <Alert
              variant="warning"
              message="检测到冲突"
              description={
                <ul>
                  {validation.conflicts.map((conflict, idx) => (
                    <li key={idx}>{conflict}</li>
                  ))}
                </ul>
              }
            />
          )}
          {validation.recommendations.length > 0 && (
            <div>
              <strong>优化建议：</strong>
              <ul>
                {validation.recommendations.map((rec, idx) => (
                  <li key={idx}>{rec}</li>
                ))}
              </ul>
            </div>
          )}
        </Space>
      </Card>
    )
  }

  // 渲染流式事件时间线
  const renderStreamTimeline = () => {
    if (streamEvents.length === 0) return null
    
    return (
      <Card title="处理流程" size="small">
        <Timeline>
          {streamEvents.map((event, idx) => (
            <Timeline.Item 
              key={idx}
              color={event.event_type === StreamEventType.ERROR ? 'red' : 'blue'}
              dot={
                event.event_type === StreamEventType.COMPLETE ? 
                <CheckCircleOutlined /> : 
                <SyncOutlined spin={streaming} />
              }
            >
              <div>
                <strong>{event.message}</strong>
                <div className="text-xs text-gray-500">
                  {new Date(event.timestamp).toLocaleTimeString()}
                </div>
              </div>
            </Timeline.Item>
          ))}
        </Timeline>
      </Card>
    )
  }

  return (
    <div className="p-6">
        <div className="mb-6">
          <div className="flex justify-between items-center mb-4">
            <h1 className="text-2xl font-bold">Agentic RAG 智能检索系统</h1>
            <Space>
              <Badge 
                status={healthStatus === 'healthy' ? 'success' : 'error'} 
                text={healthStatus === 'healthy' ? '系统正常' : '系统异常'}
              />
              <Button 
                icon={<SyncOutlined />} 
                onClick={() => {
                  loadStats()
                  checkHealth()
                }}
              >
                刷新状态
              </Button>
            </Space>
          </div>

          {/* 统计信息 */}
          {stats && (
            <Row gutter={16} className="mb-4">
              <Col span={6}>
                <Card>
                  <Statistic
                    title="总查询数"
                    value={stats.total_queries}
                    prefix={<SearchOutlined />}
                  />
                </Card>
              </Col>
              <Col span={6}>
                <Card>
                  <Statistic
                    title="成功率"
                    value={stats.performance_metrics.success_rate * 100}
                    suffix="%"
                    precision={1}
                    valueStyle={{ color: '#3f8600' }}
                  />
                </Card>
              </Col>
              <Col span={6}>
                <Card>
                  <Statistic
                    title="平均响应时间"
                    value={stats.average_response_time}
                    suffix="秒"
                    precision={2}
                    prefix={<ThunderboltOutlined />}
                  />
                </Card>
              </Col>
              <Col span={6}>
                <Card>
                  <Statistic
                    title="平均质量评分"
                    value={stats.average_quality_score * 100}
                    suffix="%"
                    precision={1}
                    prefix={<BarChartOutlined />}
                  />
                </Card>
              </Col>
            </Row>
          )}

          {/* 查询输入区 */}
          <Card className="mb-4">
            <Space direction="vertical" style={{ width: '100%' }}>
              <TextArea
                rows={3}
                placeholder="输入您的查询内容..."
                value={query}
                onChange={e => setQuery(e.target.value)}
                name="agentic-query"
                disabled={loading}
              />
              
              <Row gutter={16}>
                <Col span={12}>
                  <div className="mb-2">扩展策略：</div>
                  <Select
                    mode="multiple"
                    style={{ width: '100%' }}
                    placeholder="选择查询扩展策略"
                    value={expansionStrategies}
                    onChange={setExpansionStrategies}
                    name="agentic-expansion-strategies"
                    disabled={loading}
                  >
                    <Option value={ExpansionStrategyType.SEMANTIC}>语义扩展</Option>
                    <Option value={ExpansionStrategyType.SYNONYM}>同义词扩展</Option>
                    <Option value={ExpansionStrategyType.CONTEXTUAL}>上下文扩展</Option>
                    <Option value={ExpansionStrategyType.MULTILINGUAL}>多语言扩展</Option>
                    <Option value={ExpansionStrategyType.DECOMPOSITION}>问题分解</Option>
                  </Select>
                </Col>
                <Col span={12}>
                  <div className="mb-2">检索策略：</div>
                  <Select
                    mode="multiple"
                    style={{ width: '100%' }}
                    placeholder="选择检索策略"
                    value={retrievalStrategies}
                    onChange={setRetrievalStrategies}
                    name="agentic-retrieval-strategies"
                    disabled={loading}
                  >
                    <Option value={RetrievalStrategyType.HYBRID}>混合检索</Option>
                    <Option value={RetrievalStrategyType.SEMANTIC}>语义检索</Option>
                    <Option value={RetrievalStrategyType.KEYWORD}>关键词检索</Option>
                    <Option value={RetrievalStrategyType.STRUCTURED}>结构化检索</Option>
                  </Select>
                </Col>
              </Row>

              <Space>
                <Button
                  type="primary"
                  icon={<SearchOutlined />}
                  onClick={() => handleQuery(false)}
                  loading={loading && !streaming}
                  disabled={streaming}
                >
                  智能检索
                </Button>
                <Button
                  icon={<SyncOutlined />}
                  onClick={() => handleQuery(true)}
                  loading={streaming}
                  disabled={loading && !streaming}
                >
                  流式检索
                </Button>
              </Space>

              {streaming && (
                <Progress percent={streamProgress} status="active" />
              )}
            </Space>
          </Card>

          {/* 查询结果 */}
          {(queryResult || streamEvents.length > 0) && (
            <Row gutter={16}>
              <Col span={16}>
                <Space direction="vertical" style={{ width: '100%' }}>
                  {renderQueryAnalysis()}
                  {renderExpandedQueries()}
                  {renderRetrievalResults()}
                  {renderValidationResult()}
                  
                  {/* 用户反馈 */}
                  {queryResult?.success && (
                    <Card title="结果评价" size="small">
                      <div className="text-center">
                        <div className="mb-2">您对本次检索结果的满意度：</div>
                        <Rate 
                          onChange={(value) => handleFeedback(queryResult.query_id, value)}
                        />
                      </div>
                    </Card>
                  )}
                </Space>
              </Col>
              
              <Col span={8}>
                <Space direction="vertical" style={{ width: '100%' }}>
                  {renderStreamTimeline()}
                  
                  {/* 查询历史 */}
                  <Card title="查询历史" size="small">
                    <List
                      size="small"
                      dataSource={queryHistory}
                      renderItem={item => (
                        <List.Item>
                          <List.Item.Meta
                            title={item.query.substring(0, 50)}
                            description={
                              <div>
                                <div className="text-xs">
                                  {new Date(item.timestamp).toLocaleString()}
                                </div>
                                <Progress 
                                  percent={item.quality * 100} 
                                  size="small" 
                                  showInfo={false}
                                />
                              </div>
                            }
                          />
                        </List.Item>
                      )}
                    />
                  </Card>
                </Space>
              </Col>
            </Row>
          )}
        </div>
    </div>
  )
}

export default AgenticRagPage
