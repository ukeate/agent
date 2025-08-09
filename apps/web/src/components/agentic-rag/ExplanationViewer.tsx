/**
 * 解释查看器组件
 * 
 * 功能包括：
 * - 显示智能检索的推理过程和决策逻辑
 * - 可视化查询理解、策略选择、结果评估过程
 * - 提供检索路径追踪和透明度展示
 * - 支持交互式解释探索和深入分析
 */

import React, { useState, useMemo, useCallback } from 'react';
import {
  Card,
  Steps,
  Collapse,
  Space,
  Typography,
  Row,
  Col,
  Tag,
  Button,
  Tooltip,
  Tree,
  Timeline,
  Progress,
  Alert,
  Divider,
  List,
  Badge,
  Tabs,
  Statistic,
  Empty,
  message,
} from 'antd';
import {
  QuestionCircleOutlined,
  BulbOutlined,
  SearchOutlined,
  BranchesOutlined,
  ShareAltOutlined,
  DownloadOutlined,
  ClockCircleOutlined,
  InfoCircleOutlined,
  TrophyOutlined,
  BarChartOutlined,
} from '@ant-design/icons';
import { useRagStore } from '../../stores/ragStore';
import { AgenticQueryResponse } from '../../services/ragService';

const { Text, Title, Paragraph } = Typography;
const { Panel } = Collapse;
const { Step } = Steps;
const { TabPane } = Tabs;

// ==================== 组件props类型 ====================

interface ExplanationViewerProps {
  agenticResults?: AgenticQueryResponse | null;
  explanationData?: any;
  query?: string;
  className?: string;
  onShare?: (explanation: any) => void;
  onExport?: (explanation: any) => void;
  compact?: boolean;
}

// ==================== 辅助类型 ====================

interface ReasoningStep {
  id: string;
  title: string;
  description: string;
  details: any;
  confidence: number;
  reasoning: string;
  alternatives: string[];
  impact_score: number;
}

// interface DecisionPath {
//   step: string;
//   decision: string;
//   rationale: string;
//   confidence: number;
//   alternatives: Array<{
//     option: string;
//     score: number;
//     reason: string;
//   }>;
// }

interface ExplanationStructure {
  query_understanding: {
    intent: string;
    complexity: number;
    key_concepts: string[];
    assumptions: string[];
  };
  strategy_selection: {
    chosen_strategies: string[];
    reasoning: string;
    alternatives_considered: string[];
    trade_offs: string[];
  };
  retrieval_process: {
    agent_coordination: any[];
    search_paths: any[];
    result_fusion: any;
  };
  quality_assessment: {
    evaluation_criteria: string[];
    scores: Record<string, number>;
    validation_steps: string[];
  };
  final_reasoning: {
    summary: string;
    confidence: number;
    limitations: string[];
    improvements: string[];
  };
}

// ==================== 主组件 ====================

const ExplanationViewer: React.FC<ExplanationViewerProps> = ({
  agenticResults: propAgenticResults,
  explanationData: propExplanationData,
  query: propQuery,
  className = '',
  onShare,
  onExport,
  compact = false,
}) => {
  // ==================== 状态管理 ====================
  
  const {
    agenticResults,
    explanationData,
    currentQuery,
    showExplanation,
    setShowExplanation,
  } = useRagStore();

  // 使用props数据或store数据
  const results = propAgenticResults || agenticResults;
  const explanation = propExplanationData || explanationData;
  const query = propQuery || currentQuery;

  // ==================== 本地状态 ====================
  
  const [activeTab, setActiveTab] = useState<string>('overview');
  const [expandedPanels, setExpandedPanels] = useState<string[]>(['understanding']);
  const [selectedStep, setSelectedStep] = useState<string | null>(null);
  // const [showDetails, setShowDetails] = useState(false);
  const [viewMode, setViewMode] = useState<'structured' | 'timeline' | 'tree'>('structured');

  // ==================== 数据处理 ====================
  
  // 生成模拟解释数据
  const mockExplanation = useMemo((): ExplanationStructure => {
    if (!results) {
      return {
        query_understanding: {
          intent: 'unknown',
          complexity: 0,
          key_concepts: [],
          assumptions: [],
        },
        strategy_selection: {
          chosen_strategies: [],
          reasoning: '',
          alternatives_considered: [],
          trade_offs: [],
        },
        retrieval_process: {
          agent_coordination: [],
          search_paths: [],
          result_fusion: {},
        },
        quality_assessment: {
          evaluation_criteria: [],
          scores: {},
          validation_steps: [],
        },
        final_reasoning: {
          summary: '',
          confidence: 0,
          limitations: [],
          improvements: [],
        },
      };
    }

    return {
      query_understanding: {
        intent: results.analysis_info?.intent_type || 'factual',
        complexity: results.analysis_info?.complexity_score || 0.5,
        key_concepts: results.analysis_info?.entities || ['AI', '代理', '系统'],
        assumptions: [
          '用户寻求技术相关信息',
          '需要详细的实现指导',
          '关注实用性和可操作性'
        ],
      },
      strategy_selection: {
        chosen_strategies: results.expansion_strategies || ['semantic', 'contextual'],
        reasoning: '基于查询复杂度和意图类型，选择语义扩展和上下文扩展策略，以获得最佳的召回率和精确度平衡。',
        alternatives_considered: ['synonym', 'decomposition', 'multilingual'],
        trade_offs: [
          '语义扩展: 高召回率但可能引入噪音',
          '上下文扩展: 高精确度但可能遗漏相关内容',
          '同义词扩展: 基础但稳定的结果'
        ],
      },
      retrieval_process: {
        agent_coordination: [
          {
            agent: 'semantic_retriever',
            action: '语义向量检索',
            results: results.results?.filter(r => r.score > 0.8).length || 0,
            time: 150,
          },
          {
            agent: 'keyword_retriever',
            action: 'BM25关键词检索',
            results: results.results?.filter(r => r.score > 0.6).length || 0,
            time: 80,
          },
          {
            agent: 'structured_retriever',
            action: '结构化检索',
            results: results.results?.filter(r => r.content_type === 'code').length || 0,
            time: 120,
          },
        ],
        search_paths: [
          {
            path: 'query → semantic_embedding → vector_search → ranking',
            results_count: Math.floor((results.results?.length || 0) * 0.6),
            avg_score: 0.82,
          },
          {
            path: 'query → keyword_extraction → bm25_search → scoring',
            results_count: Math.floor((results.results?.length || 0) * 0.3),
            avg_score: 0.74,
          },
          {
            path: 'query → structure_analysis → code_search → validation',
            results_count: Math.floor((results.results?.length || 0) * 0.1),
            avg_score: 0.68,
          },
        ],
        result_fusion: {
          method: 'weighted_reciprocal_rank',
          weights: { semantic: 0.5, keyword: 0.3, structured: 0.2 },
          final_count: results.results?.length || 0,
        },
      },
      quality_assessment: {
        evaluation_criteria: [
          '相关性 (Relevance)',
          '准确性 (Accuracy)', 
          '完整性 (Completeness)',
          '时效性 (Timeliness)',
          '可信度 (Credibility)',
          '清晰度 (Clarity)'
        ],
        scores: {
          relevance: 0.85,
          accuracy: 0.78,
          completeness: 0.72,
          timeliness: 0.66,
          credibility: 0.81,
          clarity: 0.74,
        },
        validation_steps: [
          '去重和内容重叠检测',
          '来源可信度验证',
          '时间戳新鲜度检查',
          '内容质量评分计算',
          '相关性阈值过滤'
        ],
      },
      final_reasoning: {
        summary: `基于对查询"${query}"的分析，系统采用了多代理协作检索策略，通过语义、关键词和结构化检索的结合，获得了${results.results?.length || 0}个高质量结果。整体置信度为${Math.round((results.confidence || 0.8) * 100)}%。`,
        confidence: results.confidence || 0.8,
        limitations: [
          '某些专业术语的语义理解可能存在偏差',
          '时效性评估依赖于文档元数据的完整性',
          '跨语言内容的检索覆盖有限'
        ],
        improvements: [
          '增加领域特定的词汇表和概念图',
          '集成实时数据源以提升时效性',
          '优化多语言检索策略'
        ],
      },
    };
  }, [results, query]);

  // 推理步骤数据
  const reasoningSteps = useMemo((): ReasoningStep[] => [
    {
      id: 'understanding',
      title: '查询理解',
      description: '分析用户查询意图和复杂度',
      details: mockExplanation.query_understanding,
      confidence: 0.9,
      reasoning: '通过NLP技术分析查询的语义结构、实体识别和意图分类',
      alternatives: ['直接关键词匹配', '模糊语义理解'],
      impact_score: 0.9,
    },
    {
      id: 'strategy',
      title: '策略选择',
      description: '选择最优的检索和扩展策略',
      details: mockExplanation.strategy_selection,
      confidence: 0.8,
      reasoning: '根据查询特征和历史性能数据选择最佳策略组合',
      alternatives: ['单一策略', '随机策略组合'],
      impact_score: 0.85,
    },
    {
      id: 'retrieval',
      title: '多代理检索',
      description: '协调多个检索代理并行工作',
      details: mockExplanation.retrieval_process,
      confidence: 0.88,
      reasoning: '通过负载均衡和结果融合实现最优检索效果',
      alternatives: ['单代理顺序检索', '简单并行检索'],
      impact_score: 0.92,
    },
    {
      id: 'validation',
      title: '结果验证',
      description: '评估检索结果的质量和相关性',
      details: mockExplanation.quality_assessment,
      confidence: 0.75,
      reasoning: '多维度质量评估确保结果的可靠性和实用性',
      alternatives: ['简单相关度排序', '基于规则的过滤'],
      impact_score: 0.78,
    },
  ], [mockExplanation]);

  // ==================== 事件处理 ====================
  
  const handleStepSelect = useCallback((stepId: string) => {
    setSelectedStep(stepId);
    // setShowDetails(true);
  }, []);

  const handleShare = useCallback(() => {
    const shareData = {
      query,
      explanation: mockExplanation,
      timestamp: new Date().toISOString(),
    };
    
    onShare?.(shareData);
    message.success('解释内容已准备分享');
  }, [query, mockExplanation, onShare]);

  const handleExport = useCallback(() => {
    const exportData = {
      query,
      explanation: mockExplanation,
      reasoning_steps: reasoningSteps,
      timestamp: new Date().toISOString(),
    };
    
    const dataStr = JSON.stringify(exportData, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr);
    
    const exportFileDefaultName = `rag_explanation_${Date.now()}.json`;
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
    
    onExport?.(exportData);
    message.success('解释内容已导出');
  }, [query, mockExplanation, reasoningSteps, onExport]);

  // ==================== 渲染辅助函数 ====================
  
  const renderConfidenceBadge = (confidence: number) => (
    <Badge
      count={`${Math.round(confidence * 100)}%`}
      style={{
        backgroundColor: confidence >= 0.8 ? '#52c41a' : 
                        confidence >= 0.6 ? '#faad14' : '#ff4d4f'
      }}
    />
  );

  const renderQualityScores = () => (
    <Row gutter={16}>
      {Object.entries(mockExplanation.quality_assessment.scores).map(([criterion, score]) => (
        <Col key={criterion} span={8}>
          <Statistic
            title={criterion}
            value={Math.round(score * 100)}
            suffix="%"
            valueStyle={{ 
              fontSize: 16,
              color: score >= 0.8 ? '#52c41a' : score >= 0.6 ? '#faad14' : '#ff4d4f'
            }}
          />
        </Col>
      ))}
    </Row>
  );

  const renderDecisionTree = () => {
    const treeData = [
      {
        title: '查询分析',
        key: 'analysis',
        children: [
          { title: `意图: ${mockExplanation.query_understanding.intent}`, key: 'intent' },
          { title: `复杂度: ${Math.round(mockExplanation.query_understanding.complexity * 100)}%`, key: 'complexity' },
          { title: `关键概念: ${mockExplanation.query_understanding.key_concepts.join(', ')}`, key: 'concepts' },
        ],
      },
      {
        title: '策略选择',
        key: 'strategy',
        children: mockExplanation.strategy_selection.chosen_strategies.map((strategy, index) => ({
          title: strategy,
          key: `strategy_${index}`,
        })),
      },
      {
        title: '检索执行',
        key: 'retrieval',
        children: mockExplanation.retrieval_process.agent_coordination.map((agent, index) => ({
          title: `${agent.agent}: ${agent.results}个结果`,
          key: `agent_${index}`,
        })),
      },
    ];

    return (
      <Tree
        treeData={treeData}
        defaultExpandAll
        showLine
        showIcon={false}
      />
    );
  };

  const renderTimeline = () => (
    <Timeline>
      {reasoningSteps.map((step, index) => (
        <Timeline.Item
          key={step.id}
          dot={
            <Badge 
              count={index + 1}
              style={{ backgroundColor: '#1890ff' }}
            />
          }
          color="blue"
        >
          <Card size="small" style={{ marginBottom: 16 }}>
            <Row align="middle" justify="space-between">
              <Col>
                <Space>
                  <Text strong>{step.title}</Text>
                  {renderConfidenceBadge(step.confidence)}
                </Space>
              </Col>
              <Col>
                <Button
                  size="small"
                  type="link"
                  onClick={() => handleStepSelect(step.id)}
                >
                  详情
                </Button>
              </Col>
            </Row>
            <Paragraph style={{ marginTop: 8, marginBottom: 0 }}>
              {step.description}
            </Paragraph>
          </Card>
        </Timeline.Item>
      ))}
    </Timeline>
  );

  // ==================== 渲染主组件 ====================

  return (
    <div className={`explanation-viewer ${className}`}>
      
      {!results && !explanation && (
        <Card>
          <Empty 
            description="暂无解释数据"
            image={Empty.PRESENTED_IMAGE_SIMPLE}
          />
        </Card>
      )}

      {(results || explanation) && (
        <Card
          title={
            <Space>
              <QuestionCircleOutlined />
              <Title level={4} style={{ margin: 0 }}>检索过程解释</Title>
              <Tag color="blue">智能分析</Tag>
            </Space>
          }
          extra={
            !compact && (
              <Space>
                <Tooltip title="分享解释">
                  <Button
                    icon={<ShareAltOutlined />}
                    onClick={handleShare}
                    size="small"
                  />
                </Tooltip>
                <Tooltip title="导出解释">
                  <Button
                    icon={<DownloadOutlined />}
                    onClick={handleExport}
                    size="small"
                  />
                </Tooltip>
                <Button
                  size="small"
                  onClick={() => setShowExplanation(!showExplanation)}
                >
                  {showExplanation ? '隐藏' : '显示'}
                </Button>
              </Space>
            )
          }
        >
          {showExplanation && (
            <Tabs activeKey={activeTab} onChange={setActiveTab}>
              
              {/* 总览标签页 */}
              <TabPane
                tab={
                  <Space>
                    <InfoCircleOutlined />
                    总览
                  </Space>
                }
                key="overview"
              >
                <Space direction="vertical" style={{ width: '100%' }} size="middle">
                  
                  {/* 总体摘要 */}
                  <Alert
                    message="智能检索摘要"
                    description={mockExplanation.final_reasoning.summary}
                    type="info"
                    showIcon
                    icon={<BulbOutlined />}
                  />

                  {/* 置信度和质量指标 */}
                  <Row gutter={16}>
                    <Col span={8}>
                      <Card size="small" title="整体置信度">
                        <Progress
                          type="dashboard"
                          percent={Math.round(mockExplanation.final_reasoning.confidence * 100)}
                          format={(percent) => `${percent}%`}
                          strokeColor={{
                            '0%': '#ff4d4f',
                            '50%': '#faad14',
                            '100%': '#52c41a',
                          }}
                        />
                      </Card>
                    </Col>
                    <Col span={16}>
                      <Card size="small" title="质量评估">
                        {renderQualityScores()}
                      </Card>
                    </Col>
                  </Row>

                  {/* 关键统计 */}
                  <Row gutter={16}>
                    <Col span={6}>
                      <Statistic
                        title="检索结果数"
                        value={results?.results?.length || 0}
                        prefix={<SearchOutlined />}
                      />
                    </Col>
                    <Col span={6}>
                      <Statistic
                        title="扩展查询数"
                        value={results?.expanded_queries?.length || 0}
                        prefix={<BranchesOutlined />}
                      />
                    </Col>
                    <Col span={6}>
                      <Statistic
                        title="处理时间"
                        value={results?.processing_time || 0}
                        suffix="ms"
                        prefix={<ClockCircleOutlined />}
                      />
                    </Col>
                    <Col span={6}>
                      <Statistic
                        title="参与代理数"
                        value={mockExplanation.retrieval_process.agent_coordination.length}
                        prefix={<TrophyOutlined />}
                      />
                    </Col>
                  </Row>
                </Space>
              </TabPane>

              {/* 推理过程标签页 */}
              <TabPane
                tab={
                  <Space>
                    <BranchesOutlined />
                    推理过程
                  </Space>
                }
                key="reasoning"
              >
                <Row gutter={16}>
                  <Col span={compact ? 24 : 16}>
                    {viewMode === 'structured' && (
                      <Steps
                        current={reasoningSteps.findIndex(s => s.id === selectedStep)}
                        direction="vertical"
                        onChange={(current) => handleStepSelect(reasoningSteps[current].id)}
                      >
                        {reasoningSteps.map((step) => (
                          <Step
                            key={step.id}
                            title={
                              <Space>
                                {step.title}
                                {renderConfidenceBadge(step.confidence)}
                              </Space>
                            }
                            description={
                              <Space direction="vertical" size="small">
                                <Text>{step.description}</Text>
                                <Progress
                                  percent={Math.round(step.impact_score * 100)}
                                  size="small"
                                  format={(percent) => `影响度: ${percent}%`}
                                />
                              </Space>
                            }
                            status={selectedStep === step.id ? 'process' : 'wait'}
                          />
                        ))}
                      </Steps>
                    )}

                    {viewMode === 'timeline' && renderTimeline()}
                    {viewMode === 'tree' && renderDecisionTree()}
                  </Col>

                  {!compact && (
                    <Col span={8}>
                      <Space direction="vertical" style={{ width: '100%' }}>
                        <Card size="small" title="视图切换">
                          <Button.Group size="small">
                            <Button
                              type={viewMode === 'structured' ? 'primary' : 'default'}
                              onClick={() => setViewMode('structured')}
                            >
                              结构化
                            </Button>
                            <Button
                              type={viewMode === 'timeline' ? 'primary' : 'default'}
                              onClick={() => setViewMode('timeline')}
                            >
                              时间线
                            </Button>
                            <Button
                              type={viewMode === 'tree' ? 'primary' : 'default'}
                              onClick={() => setViewMode('tree')}
                            >
                              树形图
                            </Button>
                          </Button.Group>
                        </Card>

                        {selectedStep && (
                          <Card size="small" title="步骤详情">
                            {(() => {
                              const step = reasoningSteps.find(s => s.id === selectedStep);
                              return step ? (
                                <Space direction="vertical" size="small">
                                  <Text strong>{step.title}</Text>
                                  <Text type="secondary">{step.reasoning}</Text>
                                  <Divider style={{ margin: '8px 0' }} />
                                  <Text strong>备选方案:</Text>
                                  <List
                                    size="small"
                                    dataSource={step.alternatives}
                                    renderItem={(item) => (
                                      <List.Item style={{ padding: '4px 0' }}>
                                        <Text type="secondary">• {item}</Text>
                                      </List.Item>
                                    )}
                                  />
                                </Space>
                              ) : null;
                            })()}
                          </Card>
                        )}
                      </Space>
                    </Col>
                  )}
                </Row>
              </TabPane>

              {/* 详细分析标签页 */}
              <TabPane
                tab={
                  <Space>
                    <BarChartOutlined />
                    详细分析
                  </Space>
                }
                key="analysis"
              >
                <Collapse
                  activeKey={expandedPanels}
                  onChange={setExpandedPanels}
                  ghost
                >
                  <Panel header="查询理解分析" key="understanding">
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Row gutter={16}>
                        <Col span={8}>
                          <Tag color="blue">
                            意图: {mockExplanation.query_understanding.intent}
                          </Tag>
                        </Col>
                        <Col span={8}>
                          <Tag color="green">
                            复杂度: {Math.round(mockExplanation.query_understanding.complexity * 100)}%
                          </Tag>
                        </Col>
                        <Col span={8}>
                          <Tag color="orange">
                            概念数: {mockExplanation.query_understanding.key_concepts.length}
                          </Tag>
                        </Col>
                      </Row>
                      <div>
                        <Text strong>关键概念: </Text>
                        {mockExplanation.query_understanding.key_concepts.map(concept => (
                          <Tag key={concept} color="cyan">{concept}</Tag>
                        ))}
                      </div>
                      <div>
                        <Text strong>假设条件: </Text>
                        <List
                          size="small"
                          dataSource={mockExplanation.query_understanding.assumptions}
                          renderItem={(item) => (
                            <List.Item style={{ padding: '2px 0' }}>
                              <Text type="secondary">• {item}</Text>
                            </List.Item>
                          )}
                        />
                      </div>
                    </Space>
                  </Panel>

                  <Panel header="策略选择分析" key="strategy">
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <div>
                        <Text strong>选择的策略: </Text>
                        {mockExplanation.strategy_selection.chosen_strategies.map(strategy => (
                          <Tag key={strategy} color="blue">{strategy}</Tag>
                        ))}
                      </div>
                      <Alert
                        message="选择理由"
                        description={mockExplanation.strategy_selection.reasoning}
                        type="info"
                        showIcon={false}
                      />
                      <div>
                        <Text strong>权衡考量: </Text>
                        <List
                          size="small"
                          dataSource={mockExplanation.strategy_selection.trade_offs}
                          renderItem={(item) => (
                            <List.Item style={{ padding: '2px 0' }}>
                              <Text type="secondary">• {item}</Text>
                            </List.Item>
                          )}
                        />
                      </div>
                    </Space>
                  </Panel>

                  <Panel header="检索过程分析" key="process">
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Text strong>代理协作:</Text>
                      <List
                        dataSource={mockExplanation.retrieval_process.agent_coordination}
                        renderItem={(agent) => (
                          <List.Item>
                            <List.Item.Meta
                              title={agent.agent}
                              description={
                                <Space>
                                  <Text>{agent.action}</Text>
                                  <Tag color="green">{agent.results}个结果</Tag>
                                  <Tag color="blue">{agent.time}ms</Tag>
                                </Space>
                              }
                            />
                          </List.Item>
                        )}
                      />
                    </Space>
                  </Panel>
                </Collapse>
              </TabPane>

            </Tabs>
          )}
        </Card>
      )}

    </div>
  );
};

export default ExplanationViewer;