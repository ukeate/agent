/**
 * Agentic RAG智能查询面板
 * 
 * 功能包括：
 * - 智能查询意图识别和建议
 * - 查询扩展策略选择和预览
 * - 多代理检索策略配置
 * - 实时查询分析和优化建议
 */

import React, { useState, useCallback } from 'react';
import {
  Card,
  Input,
  Button,
  Space,
  Typography,
  Row,
  Col,
  Select,
  Tag,
  Collapse,
  Alert,
  Tooltip,
  Switch,
  Slider,
  Badge,
  Progress,
  message,
} from 'antd';
import {
  SearchOutlined,
  BulbOutlined,
  SettingOutlined,
  ThunderboltOutlined,
  RobotOutlined,
  QuestionCircleOutlined,
  ExperimentOutlined,
  BranchesOutlined,
  FilterOutlined,
} from '@ant-design/icons';
import { useRagStore } from '../../stores/ragStore';
import { ragService, AgenticQueryRequest, QueryIntentType, ExpansionStrategyType, RetrievalStrategyType } from '../../services/ragService';

const { TextArea } = Input;
const { Text, Title } = Typography;
const { Option } = Select;
const { Panel } = Collapse;

// ==================== 组件props类型 ====================

interface AgenticQueryPanelProps {
  onSearch?: (request: AgenticQueryRequest) => void;
  onResults?: (response: any) => void;
  className?: string;
  disabled?: boolean;
  autoAnalyze?: boolean;
}

// ==================== 辅助类型 ====================

interface QueryAnalysisPreview {
  intent: QueryIntentType;
  confidence: number;
  suggested_strategies: ExpansionStrategyType[];
  complexity_score: number;
  keywords: string[];
  entities: string[];
  suggestions: string[];
}

interface QueryOptimization {
  optimized_query: string;
  improvements: string[];
  score_improvement: number;
}

// ==================== 主组件 ====================

const AgenticQueryPanel: React.FC<AgenticQueryPanelProps> = ({
  onSearch,
  onResults,
  className = '',
  disabled = false,
  autoAnalyze = true,
}) => {
  // ==================== 状态管理 ====================
  
  const {
    currentQuery,
    setCurrentQuery,
    isAgenticQuerying,
    setIsAgenticQuerying,
    // queryAnalysis,
    setQueryAnalysis,
    // expandedQueries,
    setExpandedQueries,
    currentSession,
    createSession,
    // switchSession,
    addToSessionHistory,
    // sessions,
    // error,
    setError,
    clearErrors,
  } = useRagStore();

  // ==================== 本地状态 ====================
  
  const [queryText, setQueryText] = useState(currentQuery);
  const [analysisPreview, setAnalysisPreview] = useState<QueryAnalysisPreview | null>(null);
  const [queryOptimization, setQueryOptimization] = useState<QueryOptimization | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  
  // 高级配置状态
  const [selectedStrategies, setSelectedStrategies] = useState<ExpansionStrategyType[]>([]);
  const [retrievalStrategies, setRetrievalStrategies] = useState<RetrievalStrategyType[]>(['semantic', 'keyword']);
  const [maxResults, setMaxResults] = useState(10);
  const [scoreThreshold, setScoreThreshold] = useState(0.7);
  const [enableExplanation, setEnableExplanation] = useState(true);
  const [enableFallback, setEnableFallback] = useState(true);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // ==================== 查询分析逻辑 ====================
  
  const analyzeQuery = useCallback(async (query: string) => {
    if (!query.trim()) {
      setAnalysisPreview(null);
      return;
    }

    setIsAnalyzing(true);
    try {
      // 模拟查询分析（实际应该调用后端API）
      await new Promise(resolve => setTimeout(resolve, 800));
      
      const mockAnalysis: QueryAnalysisPreview = {
        intent: query.includes('代码') || query.includes('function') ? 'code' : 
               query.includes('如何') || query.includes('怎么') ? 'procedural' : 'factual',
        confidence: 0.85 + Math.random() * 0.1,
        suggested_strategies: ['synonym', 'semantic', 'contextual'],
        complexity_score: Math.min(0.9, query.length / 100 + Math.random() * 0.3),
        keywords: query.split(/\s+/).slice(0, 3),
        entities: ['AI', '代理', '系统'].filter(() => Math.random() > 0.5),
        suggestions: [
          '尝试添加更具体的关键词',
          '使用同义词扩展可能会提供更多结果',
          '考虑分解为多个子问题'
        ].slice(0, 2),
      };
      
      setAnalysisPreview(mockAnalysis);
      setSelectedStrategies(mockAnalysis.suggested_strategies);
      
    } catch (error: any) {
      message.error('查询分析失败: ' + error.message);
    } finally {
      setIsAnalyzing(false);
    }
  }, []);

  const optimizeQuery = useCallback(async (query: string) => {
    if (!query.trim()) return;
    
    try {
      // 模拟查询优化（实际应该调用后端API）
      await new Promise(resolve => setTimeout(resolve, 500));
      
      const mockOptimization: QueryOptimization = {
        optimized_query: query + ' 智能检索',
        improvements: [
          '添加了领域相关术语',
          '优化了查询结构',
          '增强了语义匹配度'
        ],
        score_improvement: 0.15 + Math.random() * 0.1,
      };
      
      setQueryOptimization(mockOptimization);
      
    } catch (error: any) {
      message.error('查询优化失败: ' + error.message);
    }
  }, []);

  // ==================== 事件处理 ====================
  
  const handleQueryChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const value = e.target.value;
    setQueryText(value);
    setCurrentQuery(value);
    
    if (autoAnalyze && value.trim()) {
      // 防抖分析
      const timeoutId = setTimeout(() => {
        analyzeQuery(value);
      }, 1000);
      
      return () => clearTimeout(timeoutId);
    }
  }, [autoAnalyze, analyzeQuery, setCurrentQuery]);

  const handleSearch = useCallback(async () => {
    if (!queryText.trim()) {
      message.warning('请输入查询内容');
      return;
    }

    if (!currentSession) {
      createSession('智能检索会话');
    }

    clearErrors();
    setIsAgenticQuerying(true);

    try {
      const request: AgenticQueryRequest = {
        query: queryText,
        context_history: currentSession?.context_history || [],
        expansion_strategies: selectedStrategies.length > 0 ? selectedStrategies : undefined,
        retrieval_strategies: retrievalStrategies,
        max_results: maxResults,
        // score_threshold: scoreThreshold,
        include_explanation: enableExplanation,
        // enable_fallback: enableFallback,
      };

      onSearch?.(request);
      
      // 添加到会话历史
      if (currentSession) {
        addToSessionHistory(currentSession.id, queryText);
      }

      // 实际调用服务
      const response = await ragService.agenticQuery(request);
      
      if (response.success) {
        setQueryAnalysis(response.analysis_info || null);
        setExpandedQueries(response.expanded_queries || []);
        onResults?.(response);
        message.success('智能检索完成');
      } else {
        throw new Error(response.error || '智能检索失败');
      }

    } catch (error: any) {
      setError(error.message || '智能检索失败');
      message.error(error.message || '智能检索失败');
    } finally {
      setIsAgenticQuerying(false);
    }
  }, [
    queryText, currentSession, createSession, clearErrors, setIsAgenticQuerying,
    selectedStrategies, retrievalStrategies, maxResults, scoreThreshold, 
    enableExplanation, enableFallback, onSearch, addToSessionHistory,
    setQueryAnalysis, setExpandedQueries, onResults, setError
  ]);

  const handleUseOptimization = useCallback(() => {
    if (queryOptimization) {
      setQueryText(queryOptimization.optimized_query);
      setCurrentQuery(queryOptimization.optimized_query);
      message.success('已应用优化建议');
    }
  }, [queryOptimization, setCurrentQuery]);

  const handleQuickQuery = useCallback((query: string) => {
    setQueryText(query);
    setCurrentQuery(query);
    analyzeQuery(query);
  }, [setCurrentQuery, analyzeQuery]);

  // ==================== 渲染辅助函数 ====================
  
  const renderIntentBadge = (intent: QueryIntentType, confidence: number) => {
    const intentConfig = {
      factual: { color: 'blue', text: '事实查询' },
      procedural: { color: 'green', text: '过程查询' },
      code: { color: 'purple', text: '代码查询' },
      creative: { color: 'orange', text: '创意查询' },
      exploratory: { color: 'cyan', text: '探索查询' },
    };

    const config = intentConfig[intent];
    const confidencePercent = Math.round(confidence * 100);

    return (
      <Badge 
        count={`${confidencePercent}%`} 
        size="small"
        style={{ backgroundColor: config.color }}
      >
        <Tag color={config.color}>
          {config.text}
        </Tag>
      </Badge>
    );
  };

  const renderStrategySelector = () => (
    <div>
      <Text strong style={{ marginBottom: 8, display: 'block' }}>扩展策略:</Text>
      <Select
        mode="multiple"
        placeholder="选择查询扩展策略"
        value={selectedStrategies}
        onChange={setSelectedStrategies}
        style={{ width: '100%' }}
        size="small"
      >
        <Option value="synonym">同义词扩展</Option>
        <Option value="semantic">语义扩展</Option>
        <Option value="contextual">上下文扩展</Option>
        <Option value="decomposition">查询分解</Option>
        <Option value="multilingual">多语言扩展</Option>
      </Select>
    </div>
  );

  const renderQuickQueries = () => {
    const quickQueries = [
      '如何实现多智能体协作?',
      '什么是RAG检索增强生成?', 
      'LangGraph框架使用方法',
      'FastAPI异步编程最佳实践',
      'Vector数据库性能优化',
    ];

    return (
      <div>
        <Text strong style={{ marginBottom: 8, display: 'block' }}>快速查询:</Text>
        <Space size={[8, 8]} wrap>
          {quickQueries.map((query, index) => (
            <Button
              key={index}
              size="small"
              type="dashed"
              onClick={() => handleQuickQuery(query)}
            >
              {query}
            </Button>
          ))}
        </Space>
      </div>
    );
  };

  // ==================== 渲染主组件 ====================

  return (
    <div className={`agentic-query-panel ${className}`}>
      
      {/* 主查询卡片 */}
      <Card
        title={
          <Space>
            <RobotOutlined />
            <Title level={4} style={{ margin: 0 }}>智能查询</Title>
            {isAnalyzing && <Progress type="circle" size={16} />}
          </Space>
        }
        extra={
          <Space>
            <Tooltip title="高级设置">
              <Button
                icon={<SettingOutlined />}
                onClick={() => setShowAdvanced(!showAdvanced)}
                size="small"
              />
            </Tooltip>
          </Space>
        }
      >
        
        {/* 查询输入区域 */}
        <Space direction="vertical" style={{ width: '100%' }} size="middle">
          
          {/* 查询输入框 */}
          <div>
            <TextArea
              value={queryText}
              onChange={handleQueryChange}
              placeholder="请输入您的查询问题，AI将智能分析意图并优化检索策略..."
              rows={3}
              showCount
              maxLength={500}
              disabled={disabled || isAgenticQuerying}
            />
          </div>

          {/* 查询分析预览 */}
          {analysisPreview && (
            <Alert
              message={
                <Space>
                  <Text>查询意图:</Text>
                  {renderIntentBadge(analysisPreview.intent, analysisPreview.confidence)}
                  <Text type="secondary">
                    复杂度: {Math.round(analysisPreview.complexity_score * 100)}%
                  </Text>
                </Space>
              }
              description={
                <Space direction="vertical" size="small">
                  {analysisPreview.keywords.length > 0 && (
                    <div>
                      <Text strong>关键词: </Text>
                      {analysisPreview.keywords.map(keyword => (
                        <Tag key={keyword}>{keyword}</Tag>
                      ))}
                    </div>
                  )}
                  {analysisPreview.suggestions.length > 0 && (
                    <div>
                      <Text strong>建议: </Text>
                      <ul style={{ margin: 0, paddingLeft: 16 }}>
                        {analysisPreview.suggestions.map((suggestion, index) => (
                          <li key={index} style={{ fontSize: 12 }}>
                            {suggestion}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </Space>
              }
              type="info"
              showIcon
              icon={<BulbOutlined />}
            />
          )}

          {/* 查询优化建议 */}
          {queryOptimization && (
            <Alert
              message="查询优化建议"
              description={
                <Space direction="vertical" style={{ width: '100%' }}>
                  <div>
                    <Text strong>优化后查询: </Text>
                    <Text code>{queryOptimization.optimized_query}</Text>
                  </div>
                  <div>
                    <Text strong>改进效果: </Text>
                    <Progress 
                      percent={Math.round(queryOptimization.score_improvement * 100)}
                      size="small"
                      format={() => `+${Math.round(queryOptimization.score_improvement * 100)}%`}
                    />
                  </div>
                  <Button 
                    size="small" 
                    type="primary"
                    onClick={handleUseOptimization}
                  >
                    应用优化
                  </Button>
                </Space>
              }
              type="success"
              showIcon
              icon={<ThunderboltOutlined />}
            />
          )}

          {/* 操作按钮区域 */}
          <Row gutter={16}>
            <Col span={16}>
              <Button
                type="primary"
                icon={<SearchOutlined />}
                loading={isAgenticQuerying}
                onClick={handleSearch}
                disabled={disabled || !queryText.trim()}
                block
              >
                {isAgenticQuerying ? '智能检索中...' : '智能检索'}
              </Button>
            </Col>
            <Col span={8}>
              <Button
                icon={<ExperimentOutlined />}
                onClick={() => optimizeQuery(queryText)}
                disabled={disabled || !queryText.trim() || isAnalyzing}
                block
              >
                优化查询
              </Button>
            </Col>
          </Row>

        </Space>
      </Card>

      {/* 高级配置面板 */}
      {showAdvanced && (
        <Card
          size="small"
          title={
            <Space>
              <SettingOutlined />
              <Text strong>高级配置</Text>
            </Space>
          }
          style={{ marginTop: 16 }}
        >
          <Collapse size="small" ghost>
            
            {/* 扩展策略配置 */}
            <Panel 
              header={
                <Space>
                  <BranchesOutlined />
                  <Text>查询扩展策略</Text>
                </Space>
              } 
              key="expansion"
            >
              <Space direction="vertical" style={{ width: '100%' }}>
                {renderStrategySelector()}
                <Text type="secondary" style={{ fontSize: 12 }}>
                  智能查询扩展可以提高召回率，但可能影响精确度
                </Text>
              </Space>
            </Panel>

            {/* 检索策略配置 */}
            <Panel 
              header={
                <Space>
                  <FilterOutlined />
                  <Text>检索策略</Text>
                </Space>
              } 
              key="retrieval"
            >
              <Space direction="vertical" style={{ width: '100%' }}>
                <div>
                  <Text strong>检索方法:</Text>
                  <Select
                    mode="multiple"
                    value={retrievalStrategies}
                    onChange={setRetrievalStrategies}
                    style={{ width: '100%', marginTop: 4 }}
                    size="small"
                  >
                    <Option value="semantic">语义检索</Option>
                    <Option value="keyword">关键词检索</Option>
                    <Option value="structured">结构化检索</Option>
                  </Select>
                </div>
                
                <Row gutter={16}>
                  <Col span={12}>
                    <Text strong>结果数量:</Text>
                    <Slider
                      min={5}
                      max={50}
                      value={maxResults}
                      onChange={setMaxResults}
                      marks={{ 5: '5', 25: '25', 50: '50' }}
                    />
                  </Col>
                  <Col span={12}>
                    <Text strong>分数阈值:</Text>
                    <Slider
                      min={0.3}
                      max={1.0}
                      step={0.1}
                      value={scoreThreshold}
                      onChange={setScoreThreshold}
                      marks={{ 0.3: '0.3', 0.7: '0.7', 1.0: '1.0' }}
                    />
                  </Col>
                </Row>
              </Space>
            </Panel>

            {/* 系统功能配置 */}
            <Panel 
              header={
                <Space>
                  <QuestionCircleOutlined />
                  <Text>系统功能</Text>
                </Space>
              } 
              key="features"
            >
              <Space direction="vertical" style={{ width: '100%' }}>
                <Row>
                  <Col span={12}>
                    <Space>
                      <Switch 
                        checked={enableExplanation} 
                        onChange={setEnableExplanation}
                        size="small"
                      />
                      <Text>启用结果解释</Text>
                    </Space>
                  </Col>
                  <Col span={12}>
                    <Space>
                      <Switch 
                        checked={enableFallback} 
                        onChange={setEnableFallback}
                        size="small"
                      />
                      <Text>启用后备策略</Text>
                    </Space>
                  </Col>
                </Row>
              </Space>
            </Panel>

          </Collapse>
        </Card>
      )}

      {/* 快速查询面板 */}
      <Card size="small" title="快速查询" style={{ marginTop: 16 }}>
        {renderQuickQueries()}
      </Card>

    </div>
  );
};

export default AgenticQueryPanel;