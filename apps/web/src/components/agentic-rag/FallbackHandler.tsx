/**
 * 后备处理器组件
 * 
 * 功能包括：
 * - 处理RAG检索失败或结果不满意的场景
 * - 提供替代搜索策略和数据源
 * - 实现智能降级和错误恢复机制
 * - 收集失败情况反馈和改进建议
 */

import React, { useState, useCallback, useEffect } from 'react';
import {
  Card,
  Space,
  Typography,
  Row,
  Col,
  Button,
  Alert,
  Steps,
  List,
  Tag,
  Modal,
  Input,
  Select,
  Badge,
  Timeline,
  message,
  Spin,
} from 'antd';
import {
  ExclamationCircleOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  RocketOutlined,
  ToolOutlined,
  QuestionCircleOutlined,
  SendOutlined,
  HistoryOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons';
import { useRagStore } from '../../stores/ragStore';
import { ragService, QueryRequest } from '../../services/ragService';

const { Text, Title, Paragraph } = Typography;
const { TextArea } = Input;
const { Option } = Select;
const { Step } = Steps;

// ==================== 组件props类型 ====================

interface FallbackHandlerProps {
  className?: string;
  onFallbackSuccess?: (results: any) => void;
  onFallbackFailed?: (error: string) => void;
  autoTrigger?: boolean;
  showHistory?: boolean;
}

// ==================== 辅助类型 ====================

interface FallbackStrategy {
  id: string;
  name: string;
  description: string;
  type: 'search_modification' | 'data_source' | 'algorithm_change' | 'manual_help';
  confidence: number;
  estimated_time: number;
  success_rate: number;
}

interface FallbackAttempt {
  id: string;
  timestamp: Date;
  original_query: string;
  failure_reason: string;
  strategy_used: FallbackStrategy;
  status: 'pending' | 'running' | 'success' | 'failed';
  results?: any;
  error?: string;
  execution_time?: number;
}

interface FailureAnalysis {
  query: string;
  failure_type: 'no_results' | 'low_quality' | 'timeout' | 'service_error' | 'relevance_low';
  possible_reasons: string[];
  suggested_strategies: FallbackStrategy[];
  user_context?: string;
}

// ==================== 预定义策略 ====================

const FALLBACK_STRATEGIES: FallbackStrategy[] = [
  {
    id: 'query_simplification',
    name: '查询简化',
    description: '移除复杂术语和修饰词，使用更基础的关键词',
    type: 'search_modification',
    confidence: 0.8,
    estimated_time: 2000,
    success_rate: 0.75,
  },
  {
    id: 'query_expansion',
    name: '查询扩展',
    description: '添加同义词和相关术语，扩大搜索范围',
    type: 'search_modification',
    confidence: 0.7,
    estimated_time: 3000,
    success_rate: 0.65,
  },
  {
    id: 'semantic_similarity',
    name: '语义相似搜索',
    description: '使用语义向量进行相似度搜索',
    type: 'algorithm_change',
    confidence: 0.85,
    estimated_time: 4000,
    success_rate: 0.70,
  },
  {
    id: 'fuzzy_matching',
    name: '模糊匹配',
    description: '降低匹配精度，允许部分匹配',
    type: 'algorithm_change',
    confidence: 0.6,
    estimated_time: 2500,
    success_rate: 0.60,
  },
  {
    id: 'external_search',
    name: '外部搜索',
    description: '使用外部搜索引擎作为补充数据源',
    type: 'data_source',
    confidence: 0.9,
    estimated_time: 5000,
    success_rate: 0.80,
  },
  {
    id: 'manual_assistance',
    name: '人工协助',
    description: '提交给人工专家进行处理',
    type: 'manual_help',
    confidence: 0.95,
    estimated_time: 300000,
    success_rate: 0.90,
  },
];

const FAILURE_TYPES = {
  no_results: '无检索结果',
  low_quality: '结果质量低',
  timeout: '请求超时',
  service_error: '服务错误',
  relevance_low: '相关性低',
};

// ==================== 主组件 ====================

const FallbackHandler: React.FC<FallbackHandlerProps> = ({
  className = '',
  onFallbackSuccess,
  onFallbackFailed,
  autoTrigger = false,
  showHistory = true,
}) => {
  // ==================== 状态管理 ====================
  
  const {
    currentQuery,
    queryResults,
    isQuerying,
    error,
    clearErrors,
  } = useRagStore();

  // ==================== 本地状态 ====================
  
  const [isActive, setIsActive] = useState(false);
  const [currentAnalysis, setCurrentAnalysis] = useState<FailureAnalysis | null>(null);
  const [attemptHistory, setAttemptHistory] = useState<FallbackAttempt[]>([]);
  const [currentAttempt, setCurrentAttempt] = useState<FallbackAttempt | null>(null);
  const [selectedStrategies, setSelectedStrategies] = useState<FallbackStrategy[]>([]);
  const [isExecuting, setIsExecuting] = useState(false);
  const [showManualHelp, setShowManualHelp] = useState(false);
  const [manualRequest, setManualRequest] = useState('');
  const [userFeedback, setUserFeedback] = useState('');

  // ==================== 自动触发逻辑 ====================
  
  useEffect(() => {
    if (!autoTrigger) return;

    // 检查是否需要触发后备处理
    const shouldTrigger = 
      (error && !isQuerying) ||
      (queryResults && queryResults.length === 0 && currentQuery && !isQuerying) ||
      (queryResults && queryResults.length > 0 && 
       queryResults.every(r => r.score < 0.3) && !isQuerying);

    if (shouldTrigger && !isActive) {
      handleAnalyzeFailure();
    }
  }, [error, queryResults, currentQuery, isQuerying, autoTrigger, isActive]);

  // ==================== 故障分析 ====================
  
  const analyzeFailure = useCallback((query: string, results: any[], error?: string): FailureAnalysis => {
    let failureType: FailureAnalysis['failure_type'] = 'no_results';
    let possibleReasons: string[] = [];
    
    if (error) {
      if (error.includes('timeout') || error.includes('超时')) {
        failureType = 'timeout';
        possibleReasons = ['网络连接问题', '服务器响应慢', '查询过于复杂'];
      } else {
        failureType = 'service_error';
        possibleReasons = ['服务暂时不可用', '配置错误', '系统故障'];
      }
    } else if (!results || results.length === 0) {
      failureType = 'no_results';
      possibleReasons = [
        '搜索关键词过于具体',
        '数据库中无相关内容',
        '查询语法不正确',
        '索引未完全建立'
      ];
    } else if (results.every(r => r.score < 0.3)) {
      failureType = 'relevance_low';
      possibleReasons = [
        '查询意图理解偏差',
        '关键词匹配度低',
        '语义理解不准确',
        '数据源质量问题'
      ];
    } else {
      failureType = 'low_quality';
      possibleReasons = [
        '结果排序算法需优化',
        '数据源内容质量参差不齐',
        '过滤条件过于宽松'
      ];
    }

    // 根据故障类型推荐策略
    const suggestedStrategies = FALLBACK_STRATEGIES.filter(strategy => {
      switch (failureType) {
        case 'no_results':
          return ['query_expansion', 'query_simplification', 'fuzzy_matching', 'external_search'].includes(strategy.id);
        case 'low_quality':
        case 'relevance_low':
          return ['semantic_similarity', 'query_modification', 'external_search'].includes(strategy.id);
        case 'timeout':
          return ['query_simplification', 'fuzzy_matching'].includes(strategy.id);
        case 'service_error':
          return ['external_search', 'manual_assistance'].includes(strategy.id);
        default:
          return true;
      }
    }).sort((a, b) => b.success_rate - a.success_rate);

    return {
      query,
      failure_type: failureType,
      possible_reasons: possibleReasons,
      suggested_strategies: suggestedStrategies,
    };
  }, []);

  // ==================== 事件处理 ====================
  
  const handleAnalyzeFailure = useCallback(() => {
    if (!currentQuery) return;

    const analysis = analyzeFailure(currentQuery, queryResults || [], error || '');
    setCurrentAnalysis(analysis);
    setSelectedStrategies(analysis.suggested_strategies.slice(0, 3));
    setIsActive(true);
    
    message.info('检测到检索问题，正在分析并提供解决方案...');
  }, [currentQuery, queryResults, error, analyzeFailure]);

  const handleExecuteStrategies = useCallback(async () => {
    if (!currentAnalysis || selectedStrategies.length === 0) return;

    setIsExecuting(true);
    clearErrors();

    const attempt: FallbackAttempt = {
      id: `attempt_${Date.now()}`,
      timestamp: new Date(),
      original_query: currentAnalysis.query,
      failure_reason: FAILURE_TYPES[currentAnalysis.failure_type],
      strategy_used: selectedStrategies[0], // 执行第一个策略
      status: 'running',
    };

    setCurrentAttempt(attempt);

    try {
      const strategy = selectedStrategies[0];
      let modifiedQuery = currentAnalysis.query;
      let searchParams: Partial<QueryRequest> = {};

      // 根据策略修改查询
      switch (strategy.id) {
        case 'query_simplification':
          // 简化查询：移除修饰词，保留核心关键词
          modifiedQuery = currentAnalysis.query
            .replace(/[的地得]/g, '')
            .replace(/如何|怎么|什么|哪些/g, '')
            .replace(/\s+/g, ' ')
            .trim();
          break;
          
        case 'query_expansion':
          // 查询扩展：添加相关术语
          modifiedQuery = currentAnalysis.query + ' 相关 实现 方法 技术';
          break;
          
        case 'semantic_similarity':
          searchParams.search_type = 'semantic';
          searchParams.score_threshold = 0.5;
          break;
          
        case 'fuzzy_matching':
          searchParams.score_threshold = 0.3;
          break;
          
        case 'external_search':
          // 模拟外部搜索
          await new Promise(resolve => setTimeout(resolve, strategy.estimated_time));
          throw new Error('外部搜索功能需要后端支持');
      }

      // 执行修改后的查询
      const request: QueryRequest = {
        query: modifiedQuery,
        limit: 10,
        search_type: 'hybrid',
        score_threshold: 0.6,
        ...searchParams,
      };

      const startTime = Date.now();
      const response = await ragService.query(request);
      const executionTime = Date.now() - startTime;

      if (response.success && response.results.length > 0) {
        // 后备处理成功
        const successAttempt: FallbackAttempt = {
          ...attempt,
          status: 'success',
          results: response.results,
          execution_time: executionTime,
        };

        setCurrentAttempt(successAttempt);
        setAttemptHistory(prev => [successAttempt, ...prev].slice(0, 20));
        
        onFallbackSuccess?.(response.results);
        message.success(`后备策略 "${strategy.name}" 执行成功，找到 ${response.results.length} 个结果`);
        
        // 延迟关闭
        setTimeout(() => {
          setIsActive(false);
          setCurrentAttempt(null);
        }, 3000);
      } else {
        throw new Error(response.error || '后备处理未找到满意结果');
      }

    } catch (error: any) {
      const failedAttempt: FallbackAttempt = {
        ...attempt,
        status: 'failed',
        error: error.message,
        execution_time: Date.now() - new Date(attempt.timestamp).getTime(),
      };

      setCurrentAttempt(failedAttempt);
      setAttemptHistory(prev => [failedAttempt, ...prev].slice(0, 20));
      
      onFallbackFailed?.(error.message);
      message.error(`后备策略执行失败: ${error.message}`);

      // 尝试下一个策略
      if (selectedStrategies.length > 1) {
        message.info('正在尝试下一个后备策略...');
        setSelectedStrategies(prev => prev.slice(1));
        setTimeout(() => handleExecuteStrategies(), 2000);
      }
    } finally {
      setIsExecuting(false);
    }
  }, [currentAnalysis, selectedStrategies, clearErrors, onFallbackSuccess, onFallbackFailed]);

  const handleManualHelp = useCallback(() => {
    if (!manualRequest.trim()) {
      message.error('请描述您的具体需求');
      return;
    }

    const helpRequest = {
      query: currentAnalysis?.query || '',
      failure_type: currentAnalysis?.failure_type || 'no_results',
      user_request: manualRequest,
      user_feedback: userFeedback,
      timestamp: new Date().toISOString(),
    };

    // 模拟提交人工协助请求
    message.success('人工协助请求已提交，我们会尽快为您处理');
    console.log('Manual help request:', helpRequest);
    
    setShowManualHelp(false);
    setManualRequest('');
    setUserFeedback('');
  }, [manualRequest, userFeedback, currentAnalysis]);

  const handleClose = useCallback(() => {
    setIsActive(false);
    setCurrentAnalysis(null);
    setCurrentAttempt(null);
    setSelectedStrategies([]);
  }, []);

  // ==================== 渲染辅助函数 ====================
  
  const renderFailureAnalysis = () => {
    if (!currentAnalysis) return null;

    return (
      <Card size="small" title="故障分析">
        <Space direction="vertical" style={{ width: '100%' }}>
          
          <Alert
            message={`检测到问题: ${FAILURE_TYPES[currentAnalysis.failure_type]}`}
            description={
              <div>
                <Text>查询: "{currentAnalysis.query}"</Text>
                <br />
                <Text type="secondary">
                  可能原因: {currentAnalysis.possible_reasons.join('、')}
                </Text>
              </div>
            }
            type="warning"
            showIcon
          />

          <div>
            <Text strong>推荐策略:</Text>
            <div style={{ marginTop: 8 }}>
              {currentAnalysis.suggested_strategies.slice(0, 4).map(strategy => (
                <Tag
                  key={strategy.id}
                  color={selectedStrategies.includes(strategy) ? 'blue' : 'default'}
                  style={{ margin: '2px', cursor: 'pointer' }}
                  onClick={() => {
                    if (selectedStrategies.includes(strategy)) {
                      setSelectedStrategies(prev => prev.filter(s => s.id !== strategy.id));
                    } else {
                      setSelectedStrategies(prev => [...prev, strategy]);
                    }
                  }}
                >
                  {strategy.name} ({Math.round(strategy.success_rate * 100)}%)
                </Tag>
              ))}
            </div>
          </div>

          <Row gutter={16}>
            <Col span={12}>
              <Button
                type="primary"
                icon={<RocketOutlined />}
                onClick={handleExecuteStrategies}
                loading={isExecuting}
                disabled={selectedStrategies.length === 0}
                block
              >
                执行后备策略
              </Button>
            </Col>
            <Col span={12}>
              <Button
                icon={<QuestionCircleOutlined />}
                onClick={() => setShowManualHelp(true)}
                block
              >
                请求人工协助
              </Button>
            </Col>
          </Row>

        </Space>
      </Card>
    );
  };

  const renderExecutionProgress = () => {
    if (!currentAttempt) return null;

    const getStatusIcon = () => {
      switch (currentAttempt.status) {
        case 'running':
          return <Spin size="small" />;
        case 'success':
          return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
        case 'failed':
          return <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />;
        default:
          return <ClockCircleOutlined />;
      }
    };

    const getStatusText = () => {
      switch (currentAttempt.status) {
        case 'running':
          return '执行中...';
        case 'success':
          return '执行成功';
        case 'failed':
          return '执行失败';
        default:
          return '等待执行';
      }
    };

    return (
      <Card size="small" title="执行进度">
        <Timeline>
          <Timeline.Item dot={getStatusIcon()}>
            <Space direction="vertical" size="small">
              <Space>
                <Text strong>{currentAttempt.strategy_used.name}</Text>
                <Badge status={
                  currentAttempt.status === 'running' ? 'processing' :
                  currentAttempt.status === 'success' ? 'success' : 'error'
                } text={getStatusText()} />
              </Space>
              
              <Text type="secondary">
                {currentAttempt.strategy_used.description}
              </Text>
              
              {currentAttempt.execution_time && (
                <Text type="secondary" style={{ fontSize: 12 }}>
                  执行时间: {currentAttempt.execution_time}ms
                </Text>
              )}
              
              {currentAttempt.results && (
                <Text type="secondary" style={{ fontSize: 12 }}>
                  找到结果: {currentAttempt.results.length} 个
                </Text>
              )}
              
              {currentAttempt.error && (
                <Alert
                  message={currentAttempt.error}
                  type="error"
                  size="small"
                  showIcon
                />
              )}
            </Space>
          </Timeline.Item>
        </Timeline>
      </Card>
    );
  };

  const renderAttemptHistory = () => {
    if (!showHistory || attemptHistory.length === 0) return null;

    return (
      <Card size="small" title="历史记录">
        <List
          size="small"
          dataSource={attemptHistory.slice(0, 5)}
          renderItem={(attempt) => (
            <List.Item>
              <Row style={{ width: '100%' }} align="middle">
                <Col span={16}>
                  <Space direction="vertical" size="small">
                    <Text strong ellipsis style={{ maxWidth: 200 }}>
                      {attempt.original_query}
                    </Text>
                    <Space size="small">
                      <Tag size="small">{attempt.strategy_used.name}</Tag>
                      <Text type="secondary" style={{ fontSize: 12 }}>
                        {new Date(attempt.timestamp).toLocaleString()}
                      </Text>
                    </Space>
                  </Space>
                </Col>
                <Col span={8} style={{ textAlign: 'right' }}>
                  <Badge
                    status={
                      attempt.status === 'success' ? 'success' :
                      attempt.status === 'failed' ? 'error' : 'processing'
                    }
                    text={
                      attempt.status === 'success' ? '成功' :
                      attempt.status === 'failed' ? '失败' : '进行中'
                    }
                  />
                </Col>
              </Row>
            </List.Item>
          )}
        />
      </Card>
    );
  };

  // ==================== 渲染主组件 ====================

  return (
    <div className={`fallback-handler ${className}`}>
      
      {/* 触发按钮 */}
      {!isActive && (
        <Button
          type="dashed"
          icon={<ToolOutlined />}
          onClick={handleAnalyzeFailure}
          disabled={!currentQuery || isQuerying}
          block
        >
          智能后备处理
        </Button>
      )}

      {/* 后备处理面板 */}
      {isActive && (
        <Card
          title={
            <Space>
              <WarningOutlined style={{ color: '#faad14' }} />
              <Title level={4} style={{ margin: 0 }}>智能后备处理</Title>
              <Tag color="orange">问题解决</Tag>
            </Space>
          }
          extra={
            <Button size="small" onClick={handleClose}>
              关闭
            </Button>
          }
        >
          <Space direction="vertical" style={{ width: '100%' }} size="middle">
            
            {renderFailureAnalysis()}
            {renderExecutionProgress()}
            {renderAttemptHistory()}

          </Space>
        </Card>
      )}

      {/* 人工协助模态框 */}
      <Modal
        title="请求人工协助"
        open={showManualHelp}
        onOk={handleManualHelp}
        onCancel={() => setShowManualHelp(false)}
        okText="提交请求"
        cancelText="取消"
      >
        <Space direction="vertical" style={{ width: '100%' }} size="middle">
          
          <Alert
            message="我们的专家会在24小时内回复您的请求"
            type="info"
            showIcon
          />

          <div>
            <Text strong>请详细描述您的需求:</Text>
            <TextArea
              value={manualRequest}
              onChange={(e) => setManualRequest(e.target.value)}
              placeholder="请描述您希望找到什么信息，或者遇到的具体问题..."
              rows={4}
              maxLength={500}
              showCount
              style={{ marginTop: 8 }}
            />
          </div>

          <div>
            <Text strong>补充反馈 (可选):</Text>
            <TextArea
              value={userFeedback}
              onChange={(e) => setUserFeedback(e.target.value)}
              placeholder="对当前搜索结果有什么看法，或者其他建议..."
              rows={3}
              maxLength={300}
              showCount
              style={{ marginTop: 8 }}
            />
          </div>

        </Space>
      </Modal>

    </div>
  );
};

export default FallbackHandler;