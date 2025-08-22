import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Button,
  Input,
  Badge,
  Tabs,
  Progress,
  Typography,
  Row,
  Col,
  Space,
  Slider,
  Select,
  Switch,
  Alert,
  Statistic,
  Tag,
  Divider,
  Form,
  Spin
} from 'antd';
import { 
  PlayCircleOutlined, 
  PauseCircleOutlined, 
  StopOutlined, 
  ReloadOutlined, 
  BranchesOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ExclamationCircleOutlined,
  SettingOutlined,
  EyeOutlined,
  DownloadOutlined,
  UploadOutlined
} from '@ant-design/icons';
import { 
  multiStepReasoningApi,
  type TaskDAG,
  type WorkflowDefinition,
  type ExecutionResponse,
  type SystemMetrics,
  type DecompositionResponse
} from '../services/multiStepReasoningApi';

const { Title, Text, Paragraph } = Typography;
const { TextArea } = Input;
const { TabPane } = Tabs;

// 多步推理工作流页面
const MultiStepReasoningPage: React.FC = () => {
  // 状态管理
  const [problemStatement, setProblemStatement] = useState('');
  const [currentExecution, setCurrentExecution] = useState<ExecutionResponse | null>(null);
  const [taskDAG, setTaskDAG] = useState<TaskDAG | null>(null);
  const [workflowDefinition, setWorkflowDefinition] = useState<WorkflowDefinition | null>(null);
  const [executionStatus, setExecutionStatus] = useState<'idle' | 'decomposing' | 'scheduling' | 'executing' | 'completed' | 'failed'>('idle');
  const [selectedStep, setSelectedStep] = useState<string | null>(null);
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [decompositionConfig, setDecompositionConfig] = useState({
    strategy: 'analysis',
    max_depth: 5,
    target_complexity: 5,
    enable_branching: false
  });
  const [executionConfig, setExecutionConfig] = useState({
    execution_mode: 'parallel',
    max_parallel_steps: 3,
    scheduling_strategy: 'critical_path'
  });

  // 获取系统指标
  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const metrics = await multiStepReasoningApi.getSystemMetrics();
        setSystemMetrics(metrics);
      } catch (error) {
        console.error('获取系统指标失败:', error);
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 5000);
    return () => clearInterval(interval);
  }, []);

  // 清除错误
  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => setError(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [error]);

  // 开始问题分解
  const handleStartDecomposition = async () => {
    if (!problemStatement.trim()) return;
    
    setExecutionStatus('decomposing');
    setError(null);
    setIsLoading(true);
    
    try {
      // 调用问题分解 API
      const decompositionResult = await multiStepReasoningApi.decomposeProblem({
        problem_statement: problemStatement,
        strategy: decompositionConfig.strategy,
        max_depth: decompositionConfig.max_depth,
        target_complexity: decompositionConfig.target_complexity,
        enable_branching: decompositionConfig.enable_branching
      });
      
      // 设置分解结果
      setTaskDAG(decompositionResult.task_dag);
      setWorkflowDefinition(decompositionResult.workflow_definition);
      
      // 启动执行
      setExecutionStatus('scheduling');
      const executionResult = await multiStepReasoningApi.startExecution({
        workflow_definition_id: decompositionResult.workflow_definition.id,
        execution_mode: executionConfig.execution_mode,
        max_parallel_steps: executionConfig.max_parallel_steps,
        scheduling_strategy: executionConfig.scheduling_strategy
      });
      
      setCurrentExecution(executionResult);
      setExecutionStatus('executing');
      
      // 开始轮询执行状态
      multiStepReasoningApi.pollExecutionStatus(executionResult.execution_id, (status) => {
        setCurrentExecution(status);
        if (status.status === 'completed') {
          setExecutionStatus('completed');
        } else if (status.status === 'failed') {
          setExecutionStatus('failed');
        }
      });
      
    } catch (error) {
      console.error('问题分解失败:', error);
      setError('问题分解失败，请检查输入并重试');
      setExecutionStatus('failed');
    } finally {
      setIsLoading(false);
    }
  };

  // 执行控制
  const handleExecutionControl = async (action: string) => {
    if (!currentExecution) return;
    
    try {
      await multiStepReasoningApi.controlExecution({
        execution_id: currentExecution.execution_id,
        action
      });
    } catch (error) {
      console.error('执行控制失败:', error);
      setError('执行控制失败');
    }
  };

  // 获取步骤状态颜色
  const getStepStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'success';
      case 'running': return 'processing';
      case 'failed': return 'error';
      case 'pending': return 'default';
      default: return 'default';
    }
  };

  // 获取步骤状态图标
  const getStepStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircleOutlined />;
      case 'running': return <ClockCircleOutlined />;
      case 'failed': return <CloseCircleOutlined />;
      case 'pending': return <ExclamationCircleOutlined />;
      default: return <ExclamationCircleOutlined />;
    }
  };

  return (
    <div style={{ padding: '24px', background: '#f0f2f5', minHeight: '100vh' }}>
      <Card style={{ marginBottom: '24px' }}>
        <Title level={2} style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '12px' }}>
          <BranchesOutlined style={{ color: '#1890ff' }} />
          多步推理工作流
        </Title>
        <Paragraph style={{ margin: '8px 0 0 0', color: '#666' }}>
          Complex Problem → CoT Decomposition → Task DAG → Distributed Execution
        </Paragraph>
      </Card>

      <Row gutter={[24, 24]}>
        {/* 左侧：问题输入和配置 */}
        <Col xs={24} lg={12}>
          <Card title="问题输入" style={{ marginBottom: '24px' }}>
            <Space direction="vertical" style={{ width: '100%' }} size="large">
              <div>
                <Text strong>问题描述:</Text>
                <TextArea
                  rows={4}
                  value={problemStatement}
                  onChange={(e) => setProblemStatement(e.target.value)}
                  placeholder="输入需要分解的复杂问题..."
                  style={{ marginTop: '8px' }}
                />
              </div>

              <Tabs defaultActiveKey="decomposition">
                <TabPane tab="分解配置" key="decomposition">
                  <Form layout="vertical">
                    <Form.Item label="分解策略">
                      <Select
                        value={decompositionConfig.strategy}
                        onChange={(value) => setDecompositionConfig(prev => ({ ...prev, strategy: value }))}
                      >
                        <Select.Option value="analysis">分析型分解</Select.Option>
                        <Select.Option value="research">研究型分解</Select.Option>
                        <Select.Option value="optimization">优化型分解</Select.Option>
                        <Select.Option value="development">开发型分解</Select.Option>
                      </Select>
                    </Form.Item>

                    <Form.Item label={`最大深度: ${decompositionConfig.max_depth}`}>
                      <Slider
                        min={3}
                        max={10}
                        value={decompositionConfig.max_depth}
                        onChange={(value) => setDecompositionConfig(prev => ({ ...prev, max_depth: value }))}
                      />
                    </Form.Item>

                    <Form.Item label={`目标复杂度: ${decompositionConfig.target_complexity}`}>
                      <Slider
                        min={1}
                        max={10}
                        value={decompositionConfig.target_complexity}
                        onChange={(value) => setDecompositionConfig(prev => ({ ...prev, target_complexity: value }))}
                      />
                    </Form.Item>

                    <Form.Item label="启用分支">
                      <Switch
                        checked={decompositionConfig.enable_branching}
                        onChange={(checked) => setDecompositionConfig(prev => ({ ...prev, enable_branching: checked }))}
                      />
                    </Form.Item>
                  </Form>
                </TabPane>

                <TabPane tab="执行配置" key="execution">
                  <Form layout="vertical">
                    <Form.Item label="执行模式">
                      <Select
                        value={executionConfig.execution_mode}
                        onChange={(value) => setExecutionConfig(prev => ({ ...prev, execution_mode: value }))}
                      >
                        <Select.Option value="sequential">顺序执行</Select.Option>
                        <Select.Option value="parallel">并行执行</Select.Option>
                        <Select.Option value="hybrid">混合执行</Select.Option>
                      </Select>
                    </Form.Item>

                    <Form.Item label={`最大并行数: ${executionConfig.max_parallel_steps}`}>
                      <Slider
                        min={1}
                        max={8}
                        value={executionConfig.max_parallel_steps}
                        onChange={(value) => setExecutionConfig(prev => ({ ...prev, max_parallel_steps: value }))}
                      />
                    </Form.Item>

                    <Form.Item label="调度策略">
                      <Select
                        value={executionConfig.scheduling_strategy}
                        onChange={(value) => setExecutionConfig(prev => ({ ...prev, scheduling_strategy: value }))}
                      >
                        <Select.Option value="critical_path">关键路径优先</Select.Option>
                        <Select.Option value="priority">优先级调度</Select.Option>
                        <Select.Option value="resource_aware">资源感知调度</Select.Option>
                        <Select.Option value="fifo">先进先出</Select.Option>
                      </Select>
                    </Form.Item>
                  </Form>
                </TabPane>
              </Tabs>

              <Button
                type="primary"
                size="large"
                icon={<PlayCircleOutlined />}
                loading={isLoading}
                disabled={!problemStatement.trim() || isLoading}
                onClick={handleStartDecomposition}
                style={{ width: '100%' }}
              >
                {isLoading ? '分解问题中...' : '开始分解执行'}
              </Button>
            </Space>
          </Card>

          {/* 执行控制 */}
          {currentExecution && (
            <Card title="执行控制">
              <Space direction="vertical" style={{ width: '100%' }}>
                <div>
                  <Text strong>执行ID: </Text>
                  <Text code>{currentExecution.execution_id}</Text>
                </div>
                <div>
                  <Text strong>状态: </Text>
                  <Badge status={getStepStatusColor(currentExecution.status)} text={currentExecution.status} />
                </div>
                <div>
                  <Text strong>总体进度: </Text>
                  <Progress percent={Math.round(currentExecution.progress)} />
                </div>
                <Space>
                  <Button icon={<PauseCircleOutlined />} onClick={() => handleExecutionControl('pause')}>
                    暂停
                  </Button>
                  <Button icon={<PlayCircleOutlined />} onClick={() => handleExecutionControl('resume')}>
                    继续
                  </Button>
                  <Button icon={<StopOutlined />} danger onClick={() => handleExecutionControl('cancel')}>
                    取消
                  </Button>
                  <Button icon={<ReloadOutlined />}>
                    重置
                  </Button>
                </Space>
              </Space>
            </Card>
          )}
        </Col>

        {/* 右侧：DAG可视化和监控 */}
        <Col xs={24} lg={12}>
          {/* 任务依赖图 */}
          {taskDAG && (
            <Card title="任务依赖图 (DAG)" style={{ marginBottom: '24px' }}>
              <Space direction="vertical" style={{ width: '100%' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '16px' }}>
                  <Space>
                    <Button size="small" icon={<EyeOutlined />}>全览</Button>
                    <Button size="small" icon={<DownloadOutlined />}>导出</Button>
                  </Space>
                  <Tag color="blue">关键路径: {taskDAG.critical_path?.length || 0} 步</Tag>
                </div>

                {/* 简化的任务节点展示 */}
                <div style={{ border: '1px solid #d9d9d9', borderRadius: '6px', padding: '16px' }}>
                  <Row gutter={[8, 8]}>
                    {taskDAG.nodes?.map((node: any) => (
                      <Col span={12} key={node.id}>
                        <Card 
                          size="small" 
                          hoverable
                          onClick={() => setSelectedStep(node.id)}
                          style={{ 
                            cursor: 'pointer',
                            borderColor: selectedStep === node.id ? '#1890ff' : undefined
                          }}
                        >
                          <Space direction="vertical" size="small">
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                              <Text strong style={{ fontSize: '12px' }}>{node.name}</Text>
                              {getStepStatusIcon(node.status)}
                            </div>
                            <Tag size="small" color={getStepStatusColor(node.status)}>
                              {node.task_type}
                            </Tag>
                            <Text style={{ fontSize: '11px', color: '#666' }}>
                              复杂度: {node.complexity_score} | 耗时: {node.estimated_duration_minutes}分钟
                            </Text>
                          </Space>
                        </Card>
                      </Col>
                    ))}
                  </Row>
                </div>

                {/* 选中步骤的详情 */}
                {selectedStep && (
                  <Card size="small" title="步骤详情">
                    {(() => {
                      const step = taskDAG.nodes?.find((n: any) => n.id === selectedStep);
                      return step ? (
                        <div>
                          <p><Text strong>步骤ID:</Text> {step.id}</p>
                          <p><Text strong>类型:</Text> {step.task_type}</p>
                          <p><Text strong>状态:</Text> <Badge status={getStepStatusColor(step.status)} text={step.status} /></p>
                          <p><Text strong>复杂度:</Text> {step.complexity_score}</p>
                          <p><Text strong>描述:</Text> {step.description}</p>
                        </div>
                      ) : null;
                    })()}
                  </Card>
                )}
              </Space>
            </Card>
          )}

          {/* 系统监控 */}
          <Card title="系统监控">
            {systemMetrics ? (
              <Row gutter={16}>
                <Col span={12}>
                  <Statistic
                    title="活跃工作器"
                    value={systemMetrics.active_workers}
                    prefix={<SettingOutlined />}
                  />
                </Col>
                <Col span={12}>
                  <Statistic
                    title="队列任务"
                    value={systemMetrics.queue_depth}
                    prefix={<ClockCircleOutlined />}
                  />
                </Col>
                <Col span={12}>
                  <Statistic
                    title="平均等待"
                    value={systemMetrics.average_wait_time}
                    suffix="s"
                    precision={1}
                  />
                </Col>
                <Col span={12}>
                  <Statistic
                    title="成功率"
                    value={systemMetrics.success_rate * 100}
                    suffix="%"
                    precision={0}
                  />
                </Col>
              </Row>
            ) : (
              <Spin size="large" style={{ display: 'block', textAlign: 'center' }} />
            )}
          </Card>
        </Col>
      </Row>

      {/* 错误显示 */}
      {error && (
        <Alert
          message="错误"
          description={error}
          variant="destructive"
          showIcon
          closable
          onClose={() => setError(null)}
          style={{ marginTop: '24px' }}
        />
      )}

      {/* 执行结果 */}
      {executionStatus === 'completed' && (
        <Card title="执行结果" style={{ marginTop: '24px' }}>
          <Tabs defaultActiveKey="summary">
            <TabPane tab="结果摘要" key="summary">
              <Alert
                message="工作流执行完成"
                description="所有任务已成功完成，查看详细结果。"
                type="success"
                showIcon
              />
            </TabPane>
            <TabPane tab="验证报告" key="validation">
              <Text>验证得分: 85.5/100</Text>
            </TabPane>
            <TabPane tab="原始数据" key="raw">
              <Text code>
                {JSON.stringify(currentExecution, null, 2)}
              </Text>
            </TabPane>
            <TabPane tab="格式化输出" key="formats">
              <Space>
                <Button icon={<DownloadOutlined />}>下载 JSON</Button>
                <Button icon={<DownloadOutlined />}>下载 XML</Button>
                <Button icon={<DownloadOutlined />}>下载 Markdown</Button>
              </Space>
            </TabPane>
          </Tabs>
        </Card>
      )}
    </div>
  );
};

export default MultiStepReasoningPage;