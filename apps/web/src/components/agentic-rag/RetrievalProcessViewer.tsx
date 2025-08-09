/**
 * 检索过程可视化组件
 * 
 * 功能包括：
 * - 实时显示多代理检索的执行流程和状态
 * - 可视化查询分析、扩展、检索、验证各个阶段
 * - 展示每个代理的工作进度和中间结果
 * - 提供交互式流程图和时间线视图
 */

import React, { useState, useCallback, useEffect } from 'react';
import {
  Card,
  Steps,
  Timeline,
  Progress,
  Space,
  Typography,
  Row,
  Col,
  Tag,
  Button,
  Tooltip,
  Collapse,
  Alert,
  Statistic,
  Divider,
  List,
  Badge,
  Tabs,
  Empty,
  Spin,
} from 'antd';
import {
  RobotOutlined,
  SearchOutlined,
  BulbOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  ExclamationCircleOutlined,
  ThunderboltOutlined,
  BranchesOutlined,
  FilterOutlined,
  VerifiedOutlined,
  CompressOutlined,
  QuestionCircleOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
} from '@ant-design/icons';
import { useRagStore } from '../../stores/ragStore';

const { Text, Title, Paragraph } = Typography;
const { Step } = Steps;
const { Panel } = Collapse;
const { TabPane } = Tabs;

// ==================== 组件props类型 ====================

interface RetrievalProcessViewerProps {
  className?: string;
  showTimeline?: boolean;
  autoUpdate?: boolean;
  compact?: boolean;
}

// ==================== 辅助类型 ====================

interface ProcessStep {
  id: string;
  name: string;
  status: 'waiting' | 'process' | 'finish' | 'error';
  description: string;
  duration?: number;
  details?: any;
  agent?: string;
  progress?: number;
  message?: string;
}

interface AgentStatus {
  id: string;
  name: string;
  type: 'analyzer' | 'expander' | 'retriever' | 'validator' | 'composer' | 'explainer';
  status: 'idle' | 'working' | 'completed' | 'error';
  progress: number;
  current_task?: string;
  results_count?: number;
  processing_time?: number;
}

// ==================== 主组件 ====================

const RetrievalProcessViewer: React.FC<RetrievalProcessViewerProps> = ({
  className = '',
  showTimeline = true,
  autoUpdate = true,
  compact = false,
}) => {
  // ==================== 状态管理 ====================
  
  const {
    retrievalProgress,
    isAgenticQuerying,
    queryAnalysis,
    agenticResults,
  } = useRagStore();

  // ==================== 本地状态 ====================
  
  const [viewMode, setViewMode] = useState<'steps' | 'timeline' | 'agents'>('steps');
  const [processSteps, setProcessSteps] = useState<ProcessStep[]>([]);
  const [agentStatuses, setAgentStatuses] = useState<AgentStatus[]>([]);
  const [totalDuration, setTotalDuration] = useState(0);
  const [currentStepIndex, setCurrentStepIndex] = useState(-1);
  const [isPaused, setIsPaused] = useState(false);

  // ==================== 初始化数据 ====================
  
  const initializeProcessSteps = useCallback(() => {
    const steps: ProcessStep[] = [
      {
        id: 'analysis',
        name: '查询分析',
        status: 'waiting',
        description: '分析查询意图、提取关键信息、确定复杂度',
        agent: 'analyzer',
      },
      {
        id: 'expansion',
        name: '查询扩展',
        status: 'waiting',
        description: '生成同义词、语义扩展、上下文增强',
        agent: 'expander',
      },
      {
        id: 'retrieval',
        name: '多代理检索',
        status: 'waiting',
        description: '并行执行语义、关键词、结构化检索',
        agent: 'retriever',
      },
      {
        id: 'validation',
        name: '结果验证',
        status: 'waiting',
        description: '评估结果质量、过滤低质量内容',
        agent: 'validator',
      },
      {
        id: 'composition',
        name: '上下文组合',
        status: 'waiting',
        description: '整合多源结果、构建连贯上下文',
        agent: 'composer',
      },
      {
        id: 'explanation',
        name: '结果解释',
        status: 'waiting',
        description: '生成检索推理、提供可解释性',
        agent: 'explainer',
      },
    ];
    
    setProcessSteps(steps);
  }, []);

  const initializeAgentStatuses = useCallback(() => {
    const agents: AgentStatus[] = [
      {
        id: 'analyzer',
        name: '查询分析代理',
        type: 'analyzer',
        status: 'idle',
        progress: 0,
      },
      {
        id: 'expander', 
        name: '查询扩展代理',
        type: 'expander',
        status: 'idle',
        progress: 0,
      },
      {
        id: 'semantic_retriever',
        name: '语义检索代理',
        type: 'retriever',
        status: 'idle',
        progress: 0,
      },
      {
        id: 'keyword_retriever',
        name: '关键词检索代理',
        type: 'retriever',
        status: 'idle',
        progress: 0,
      },
      {
        id: 'validator',
        name: '结果验证代理',
        type: 'validator',
        status: 'idle',
        progress: 0,
      },
      {
        id: 'composer',
        name: '上下文组合代理',
        type: 'composer',
        status: 'idle',
        progress: 0,
      },
      {
        id: 'explainer',
        name: '解释生成代理',
        type: 'explainer',
        status: 'idle',
        progress: 0,
      },
    ];
    
    setAgentStatuses(agents);
  }, []);

  // ==================== 状态更新逻辑 ====================
  
  useEffect(() => {
    if (retrievalProgress) {
      // 更新当前步骤状态
      const newSteps = [...processSteps];
      const currentStep = newSteps.find(step => 
        step.id === retrievalProgress.stage || 
        (retrievalProgress.stage === 'complete' && step.id === 'explanation')
      );
      
      if (currentStep) {
        currentStep.status = retrievalProgress.stage === 'complete' ? 'finish' : 'process';
        currentStep.progress = retrievalProgress.progress;
        currentStep.message = retrievalProgress.message;
        
        // 更新之前的步骤为完成状态
        const currentIndex = newSteps.findIndex(step => step.id === currentStep.id);
        setCurrentStepIndex(currentIndex);
        
        for (let i = 0; i < currentIndex; i++) {
          if (newSteps[i].status !== 'error') {
            newSteps[i].status = 'finish';
            newSteps[i].progress = 100;
          }
        }
        
        setProcessSteps(newSteps);
      }

      // 更新代理状态
      const newAgentStatuses = [...agentStatuses];
      const agent = newAgentStatuses.find(a => a.type === retrievalProgress.stage);
      if (agent) {
        agent.status = retrievalProgress.stage === 'complete' ? 'completed' : 'working';
        agent.progress = retrievalProgress.progress;
        agent.current_task = retrievalProgress.message;
      }
      
      setAgentStatuses(newAgentStatuses);
    }
  }, [retrievalProgress, processSteps, agentStatuses]);

  useEffect(() => {
    initializeProcessSteps();
    initializeAgentStatuses();
  }, [initializeProcessSteps, initializeAgentStatuses]);

  // ==================== 渲染辅助函数 ====================
  
  const getStepIcon = (step: ProcessStep) => {
    switch (step.status) {
      case 'finish':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'process':
        return <Spin size="small" />;
      case 'error':
        return <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />;
      default:
        return <ClockCircleOutlined style={{ color: '#d9d9d9' }} />;
    }
  };

  const getAgentIcon = (type: AgentStatus['type']) => {
    const iconMap = {
      analyzer: <BulbOutlined />,
      expander: <BranchesOutlined />,
      retriever: <SearchOutlined />,
      validator: <VerifiedOutlined />,
      composer: <CompressOutlined />,
      explainer: <QuestionCircleOutlined />,
    };
    return iconMap[type];
  };

  const getAgentColor = (status: AgentStatus['status']) => {
    const colorMap = {
      idle: '#d9d9d9',
      working: '#1890ff',
      completed: '#52c41a',
      error: '#ff4d4f',
    };
    return colorMap[status];
  };

  // ==================== 渲染步骤视图 ====================
  
  const renderStepsView = () => (
    <Card
      title={
        <Space>
          <ThunderboltOutlined />
          <Title level={4} style={{ margin: 0 }}>检索流程</Title>
          {isAgenticQuerying && <Badge status="processing" text="进行中" />}
        </Space>
      }
      extra={
        !compact && (
          <Space>
            <Text type="secondary">
              总耗时: {totalDuration > 0 ? `${totalDuration.toFixed(1)}s` : '--'}
            </Text>
            <Button
              size="small"
              icon={isPaused ? <PlayCircleOutlined /> : <PauseCircleOutlined />}
              onClick={() => setIsPaused(!isPaused)}
              disabled={!isAgenticQuerying}
            >
              {isPaused ? '继续' : '暂停'}
            </Button>
          </Space>
        )
      }
    >
      <Steps
        current={currentStepIndex}
        direction={compact ? 'horizontal' : 'vertical'}
        size={compact ? 'small' : 'default'}
      >
        {processSteps.map((step, index) => (
          <Step
            key={step.id}
            title={step.name}
            description={
              compact ? null : (
                <Space direction="vertical" size="small">
                  <Text type="secondary">{step.description}</Text>
                  {step.progress !== undefined && step.status === 'process' && (
                    <Progress 
                      percent={step.progress} 
                      size="small"
                      format={(percent) => `${percent}%`}
                    />
                  )}
                  {step.message && (
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      {step.message}
                    </Text>
                  )}
                  {step.duration && (
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      耗时: {step.duration.toFixed(1)}s
                    </Text>
                  )}
                </Space>
              )
            }
            status={step.status}
            icon={getStepIcon(step)}
          />
        ))}
      </Steps>
    </Card>
  );

  // ==================== 渲染时间线视图 ====================
  
  const renderTimelineView = () => (
    <Card
      title={
        <Space>
          <ClockCircleOutlined />
          <Title level={4} style={{ margin: 0 }}>执行时间线</Title>
        </Space>
      }
    >
      <Timeline>
        {processSteps.map(step => (
          <Timeline.Item
            key={step.id}
            dot={getStepIcon(step)}
            color={step.status === 'finish' ? 'green' : step.status === 'error' ? 'red' : 'blue'}
          >
            <Space direction="vertical" size="small">
              <Text strong>{step.name}</Text>
              <Text type="secondary">{step.description}</Text>
              {step.message && (
                <Text type="secondary" style={{ fontSize: 12 }}>
                  {step.message}
                </Text>
              )}
              {step.duration && (
                <Tag size="small">
                  {step.duration.toFixed(1)}s
                </Tag>
              )}
            </Space>
          </Timeline.Item>
        ))}
      </Timeline>
    </Card>
  );

  // ==================== 渲染代理状态视图 ====================
  
  const renderAgentsView = () => (
    <Card
      title={
        <Space>
          <RobotOutlined />
          <Title level={4} style={{ margin: 0 }}>代理状态</Title>
        </Space>
      }
    >
      <Row gutter={[16, 16]}>
        {agentStatuses.map(agent => (
          <Col 
            key={agent.id} 
            xs={24} 
            sm={12} 
            md={compact ? 24 : 8}
          >
            <Card 
              size="small"
              className="agent-status-card"
              style={{ 
                borderLeft: `4px solid ${getAgentColor(agent.status)}`,
              }}
            >
              <Space direction="vertical" size="small" style={{ width: '100%' }}>
                <Row align="middle" justify="space-between">
                  <Col>
                    <Space size="small">
                      {getAgentIcon(agent.type)}
                      <Text strong>{agent.name}</Text>
                    </Space>
                  </Col>
                  <Col>
                    <Badge 
                      status={
                        agent.status === 'idle' ? 'default' :
                        agent.status === 'working' ? 'processing' :
                        agent.status === 'completed' ? 'success' : 'error'
                      }
                      text={
                        agent.status === 'idle' ? '空闲' :
                        agent.status === 'working' ? '工作中' :
                        agent.status === 'completed' ? '已完成' : '错误'
                      }
                    />
                  </Col>
                </Row>

                {agent.progress > 0 && (
                  <Progress 
                    percent={agent.progress}
                    size="small"
                    status={agent.status === 'error' ? 'exception' : 'active'}
                  />
                )}

                {agent.current_task && (
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    {agent.current_task}
                  </Text>
                )}

                <Row>
                  {agent.results_count !== undefined && (
                    <Col span={12}>
                      <Statistic
                        title="结果数量"
                        value={agent.results_count}
                        prefix={<SearchOutlined />}
                        valueStyle={{ fontSize: 14 }}
                      />
                    </Col>
                  )}
                  {agent.processing_time !== undefined && (
                    <Col span={12}>
                      <Statistic
                        title="处理时间"
                        value={agent.processing_time}
                        suffix="ms"
                        prefix={<ClockCircleOutlined />}
                        valueStyle={{ fontSize: 14 }}
                      />
                    </Col>
                  )}
                </Row>
              </Space>
            </Card>
          </Col>
        ))}
      </Row>
    </Card>
  );

  // ==================== 渲染统计信息 ====================
  
  const renderStatistics = () => {
    if (!agenticResults || compact) return null;

    return (
      <Card size="small" title="执行统计">
        <Row gutter={16}>
          <Col span={6}>
            <Statistic
              title="总结果数量"
              value={agenticResults.results?.length || 0}
              prefix={<SearchOutlined />}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="扩展查询数"
              value={agenticResults.expanded_queries?.length || 0}
              prefix={<BranchesOutlined />}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="平均相关度"
              value={agenticResults.results ? 
                (agenticResults.results.reduce((sum, r) => sum + r.score, 0) / agenticResults.results.length * 100).toFixed(1) : 
                0
              }
              suffix="%"
              prefix={<VerifiedOutlined />}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="处理时间"
              value={agenticResults.processing_time || 0}
              suffix="ms"
              prefix={<ThunderboltOutlined />}
            />
          </Col>
        </Row>
      </Card>
    );
  };

  // ==================== 渲染主组件 ====================

  return (
    <div className={`retrieval-process-viewer ${className}`}>
      
      {/* 无数据状态 */}
      {!isAgenticQuerying && !agenticResults && (
        <Card>
          <Empty
            description="暂无检索过程数据"
            image={Empty.PRESENTED_IMAGE_SIMPLE}
          />
        </Card>
      )}

      {/* 有数据时的视图 */}
      {(isAgenticQuerying || agenticResults) && (
        <Space direction="vertical" size="middle" style={{ width: '100%' }}>
          
          {/* 统计信息 */}
          {renderStatistics()}

          {/* 主视图切换 */}
          {!compact && (
            <Tabs activeKey={viewMode} onChange={(key) => setViewMode(key as any)}>
              <TabPane 
                tab={
                  <Space size="small">
                    <ThunderboltOutlined />
                    <span>流程步骤</span>
                  </Space>
                }
                key="steps"
              >
                {renderStepsView()}
              </TabPane>
              
              {showTimeline && (
                <TabPane 
                  tab={
                    <Space size="small">
                      <ClockCircleOutlined />
                      <span>时间线</span>
                    </Space>
                  }
                  key="timeline"
                >
                  {renderTimelineView()}
                </TabPane>
              )}
              
              <TabPane 
                tab={
                  <Space size="small">
                    <RobotOutlined />
                    <span>代理状态</span>
                  </Space>
                }
                key="agents"
              >
                {renderAgentsView()}
              </TabPane>
            </Tabs>
          )}

          {/* 紧凑模式直接显示步骤 */}
          {compact && renderStepsView()}

        </Space>
      )}

    </div>
  );
};

export default RetrievalProcessViewer;