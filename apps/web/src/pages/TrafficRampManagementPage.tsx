import React, { useState, useEffect } from 'react';
import { Card, Button, Input, Select, Alert, Tabs, Space, Typography, Row, Col, Form, message, Badge, Modal } from 'antd';
import { ReloadOutlined, PlayCircleOutlined, PauseCircleOutlined, RollbackOutlined } from '@ant-design/icons';
import { 
import { logger } from '../utils/logger'
  trafficRampService,
  RampStrategy,
  RampStatus,
  RolloutPhase,
  type CreateRampPlanRequest,
  type RampExecution,
  type RampPlan,
  type StrategyInfo,
  type PhaseInfo
} from '../services/trafficRampService';

const { Option } = Select;
const { Title, Text } = Typography;

const TrafficRampManagementPage: React.FC = () => {
  const [executions, setExecutions] = useState<RampExecution[]>([]);
  const [plans, setPlans] = useState<any[]>([]);
  const [strategies, setStrategies] = useState<StrategyInfo[]>([]);
  const [phases, setPhases] = useState<PhaseInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // 创建计划表单
  const [planForm, setPlanForm] = useState<CreateRampPlanRequest>({
    experiment_id: '',
    variant: 'treatment',
    strategy: RampStrategy.LINEAR,
    start_percentage: 0,
    target_percentage: 100,
    duration_hours: 24,
    num_steps: 10
  });

  // 快速创建表单
  const [quickForm, setQuickForm] = useState({
    experiment_id: '',
    phase: RolloutPhase.CANARY,
    duration_hours: undefined as number | undefined
  });

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    await Promise.all([
      loadExecutions(),
      loadPlans(),
      loadStrategies(),
      loadPhases()
    ]);
  };

  const loadExecutions = async () => {
    try {
      const response = await trafficRampService.listExecutions();
      if (response.success) {
        setExecutions(response.executions);
      }
    } catch (err) {
      logger.warn('加载执行记录失败:', err);
    }
  };

  const loadPlans = async () => {
    try {
      const response = await trafficRampService.listPlans();
      if (response.success) {
        setPlans(response.plans);
      }
    } catch (err) {
      logger.warn('加载计划失败:', err);
    }
  };

  const loadStrategies = async () => {
    try {
      const response = await trafficRampService.listStrategies();
      if (response.success) {
        setStrategies(response.strategies);
      }
    } catch (err) {
      logger.warn('加载策略失败:', err);
    }
  };

  const loadPhases = async () => {
    try {
      const response = await trafficRampService.listPhases();
      if (response.success) {
        setPhases(response.phases);
      }
    } catch (err) {
      logger.warn('加载阶段失败:', err);
    }
  };

  const handleCreatePlan = async () => {
    if (!planForm.experiment_id) {
      setError('请填写实验ID');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      
      const response = await trafficRampService.createRampPlan(planForm);
      if (response.success) {
        setSuccess('爬坡计划创建成功');
        await loadPlans();
      }
    } catch (err) {
      setError('创建计划失败: ' + (err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const handleQuickRamp = async () => {
    if (!quickForm.experiment_id) {
      setError('请填写实验ID');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      
      const response = await trafficRampService.quickRamp(quickForm);
      if (response.success) {
        setSuccess(`${quickForm.phase}阶段流量爬坡已启动`);
        await Promise.all([loadPlans(), loadExecutions()]);
      }
    } catch (err) {
      setError('快速爬坡失败: ' + (err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const handleStartRamp = async (planId: string) => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await trafficRampService.startRamp(planId);
      if (response.success) {
        setSuccess('爬坡已启动');
        await loadExecutions();
      }
    } catch (err) {
      setError('启动失败: ' + (err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const handlePauseRamp = async (execId: string) => {
    try {
      setLoading(true);
      const response = await trafficRampService.pauseRamp(execId);
      if (response.success) {
        setSuccess('爬坡已暂停');
        await loadExecutions();
      }
    } catch (err) {
      setError('暂停失败: ' + (err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const handleResumeRamp = async (execId: string) => {
    try {
      setLoading(true);
      const response = await trafficRampService.resumeRamp(execId);
      if (response.success) {
        setSuccess('爬坡已恢复');
        await loadExecutions();
      }
    } catch (err) {
      setError('恢复失败: ' + (err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const handleRollback = async (execId: string) => {
    if (!confirm('确定要回滚吗？此操作不可逆。')) return;

    try {
      setLoading(true);
      const response = await trafficRampService.rollbackRamp(execId, '用户手动回滚');
      if (response.success) {
        setSuccess('流量已回滚');
        await loadExecutions();
      }
    } catch (err) {
      setError('回滚失败: ' + (err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status: RampStatus) => {
    switch (status) {
      case RampStatus.RUNNING: return 'bg-blue-100 text-blue-800';
      case RampStatus.COMPLETED: return 'bg-green-100 text-green-800';
      case RampStatus.PAUSED: return 'bg-yellow-100 text-yellow-800';
      case RampStatus.FAILED: return 'bg-red-100 text-red-800';
      case RampStatus.ROLLBACK: return 'bg-orange-100 text-orange-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getProgressPercentage = (execution: RampExecution) => {
    return Math.min(100, Math.max(0, execution.current_percentage));
  };

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <Title level={2}>流量爬坡管理</Title>
        <Button 
          icon={<ReloadOutlined />} 
          onClick={loadData} 
          loading={loading}
        >
          刷新数据
        </Button>
      </div>

      {error && (
        <Alert
          message="错误"
          description={error}
          type="error"
          closable
          style={{ marginBottom: 16 }}
          onClose={() => setError(null)}
        />
      )}

      {success && (
        <Alert
          message="成功"
          description={success}
          type="success"
          closable
          style={{ marginBottom: 16 }}
          onClose={() => setSuccess(null)}
        />
      )}

      {/* 活跃爬坡概览 */}
      <Card title="活跃爬坡" style={{ marginBottom: '24px' }}>
          {executions.filter(e => e.status === RampStatus.RUNNING || e.status === RampStatus.PAUSED).length > 0 ? (
            <Space direction="vertical" style={{ width: '100%' }}>
              {executions
                .filter(e => e.status === RampStatus.RUNNING || e.status === RampStatus.PAUSED)
                .map((execution) => (
                  <Card key={execution.exec_id} style={{ borderLeft: '4px solid #1890ff' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                        <div style={{ flex: 1 }}>
                          <Space direction="vertical" style={{ width: '100%' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                              <Text strong>实验: {execution.experiment_id}</Text>
                              <Badge 
                                status={execution.status === RampStatus.RUNNING ? 'processing' : 
                                       execution.status === RampStatus.COMPLETED ? 'success' :
                                       execution.status === RampStatus.PAUSED ? 'warning' : 'error'} 
                                text={execution.status}
                              />
                            </div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '16px', fontSize: '14px' }}>
                              <span>进度: {execution.current_percentage.toFixed(1)}%</span>
                              <span>步骤: {execution.current_step}</span>
                            </div>
                            <div style={{ width: '100%', backgroundColor: '#f0f0f0', borderRadius: '4px', height: '8px' }}>
                              <div 
                                style={{ 
                                  backgroundColor: '#1890ff', 
                                  height: '8px', 
                                  borderRadius: '4px',
                                  width: `${getProgressPercentage(execution)}%`
                                }}
                              ></div>
                            </div>
                          </Space>
                        </div>
                        <div style={{ display: 'flex', gap: '8px' }}>
                          {execution.status === RampStatus.RUNNING && (
                            <Button 
                              size="small" 
                              icon={<PauseCircleOutlined />}
                              onClick={() => handlePauseRamp(execution.exec_id)}
                            >
                              暂停
                            </Button>
                          )}
                          {execution.status === RampStatus.PAUSED && (
                            <Button 
                              size="small"
                              type="primary"
                              icon={<PlayCircleOutlined />}
                              onClick={() => handleResumeRamp(execution.exec_id)}
                            >
                              恢复
                            </Button>
                          )}
                          <Button 
                            size="small" 
                            danger
                            icon={<RollbackOutlined />}
                            onClick={() => handleRollback(execution.exec_id)}
                          >
                            回滚
                          </Button>
                        </div>
                      </div>
                  </Card>
                ))}
            </Space>
          ) : (
            <div style={{ textAlign: 'center', padding: '40px', color: '#999' }}>
              当前没有活跃的流量爬坡
            </div>
          )}
      </Card>

      <Tabs defaultActiveKey="quick" type="card" items={[
        {
          key: 'quick',
          label: '快速创建',
          children: (
            <Card title="快速流量爬坡">
              <Space direction="vertical" style={{ width: '100%' }}>
                <Row gutter={[16, 16]}>
                  <Col span={8}>
                    <Input
                      placeholder="实验ID *"
                      value={quickForm.experiment_id}
                      onChange={(e) => setQuickForm({...quickForm, experiment_id: e.target.value})}
                    />
                  </Col>
                  <Col span={8}>
                    <Select 
                      style={{ width: '100%' }}
                      placeholder="选择阶段"
                      value={quickForm.phase} 
                      onChange={(value) => setQuickForm({...quickForm, phase: value as RolloutPhase})}
                    >
                      {phases.map((phase) => (
                        <Option key={phase.value} value={phase.value}>
                          {phase.name} ({phase.range})
                        </Option>
                      ))}
                    </Select>
                  </Col>
                  <Col span={8}>
                    <Input
                      type="number"
                      placeholder="持续时间(小时)"
                      value={quickForm.duration_hours || ''}
                      onChange={(e) => setQuickForm({
                        ...quickForm, 
                        duration_hours: e.target.value ? parseInt(e.target.value) : undefined
                      })}
                    />
                  </Col>
                </Row>
                
                <Button 
                  type="primary" 
                  onClick={handleQuickRamp} 
                  loading={loading}
                >
                  启动快速爬坡
                </Button>

                <Row gutter={[16, 16]}>
                  {phases.map((phase) => (
                    <Col key={phase.value} span={24/5}>
                      <Card 
                        hoverable
                        style={{ textAlign: 'center', cursor: 'pointer' }}
                      >
                        <Title level={5}>{phase.name}</Title>
                        <Text type="secondary">{phase.range}</Text>
                        <div style={{ marginTop: '8px' }}>
                          <Text style={{ fontSize: '12px' }}>{phase.description}</Text>
                        </div>
                      </Card>
                    </Col>
                  ))}
                </Row>
              </Space>
            </Card>
          )
        },

        {
          key: 'advanced',
          label: '高级创建',
          children: (
            <Card title="创建爬坡计划">
              <Space direction="vertical" style={{ width: '100%' }}>
                <Row gutter={[16, 16]}>
                  <Col span={12}>
                    <Input
                      placeholder="实验ID *"
                      value={planForm.experiment_id}
                      onChange={(e) => setPlanForm({...planForm, experiment_id: e.target.value})}
                    />
                  </Col>
                  <Col span={12}>
                    <Input
                      placeholder="变体名称"
                      value={planForm.variant}
                      onChange={(e) => setPlanForm({...planForm, variant: e.target.value})}
                    />
                  </Col>
                  <Col span={12}>
                    <Select 
                      style={{ width: '100%' }}
                      placeholder="选择策略"
                      value={planForm.strategy} 
                      onChange={(value) => setPlanForm({...planForm, strategy: value as RampStrategy})}
                    >
                      {strategies.map((strategy) => (
                        <Option key={strategy.value} value={strategy.value}>
                          {strategy.name}
                        </Option>
                      ))}
                    </Select>
                  </Col>
                  <Col span={12}>
                    <Input
                      type="number"
                      placeholder="起始百分比"
                      value={planForm.start_percentage}
                      onChange={(e) => setPlanForm({...planForm, start_percentage: parseFloat(e.target.value)})}
                    />
                  </Col>
                  <Col span={12}>
                    <Input
                      type="number"
                      placeholder="目标百分比"
                      value={planForm.target_percentage}
                      onChange={(e) => setPlanForm({...planForm, target_percentage: parseFloat(e.target.value)})}
                    />
                  </Col>
                  <Col span={12}>
                    <Input
                      type="number"
                      placeholder="持续时间(小时)"
                      value={planForm.duration_hours}
                      onChange={(e) => setPlanForm({...planForm, duration_hours: parseFloat(e.target.value)})}
                    />
                  </Col>
                  <Col span={12}>
                    <Input
                      type="number"
                      placeholder="步骤数量"
                      value={planForm.num_steps}
                      onChange={(e) => setPlanForm({...planForm, num_steps: parseInt(e.target.value)})}
                    />
                  </Col>
                </Row>
              
                <Button 
                  type="primary" 
                  onClick={handleCreatePlan} 
                  loading={loading}
                >
                  创建计划
                </Button>

                <div>
                  <Title level={5}>可用策略</Title>
                  <Space direction="vertical" style={{ width: '100%' }}>
                    {strategies.map((strategy) => (
                      <Card key={strategy.value} size="small">
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                          <div>
                            <Text strong>{strategy.name}</Text>
                            <div>
                              <Text type="secondary">{strategy.description}</Text>
                            </div>
                          </div>
                          <Badge color="blue">{strategy.use_case}</Badge>
                        </div>
                      </Card>
                    ))}
                  </Space>
                </div>
              </Space>
            </Card>
          )
        },

        {
          key: 'plans',
          label: '计划管理',
          children: (
            <Card title={`爬坡计划 (${plans.length})`}>
              {plans.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="w-full border-collapse border border-gray-300">
                    <thead>
                      <tr className="bg-gray-50">
                        <th className="border border-gray-300 px-4 py-2 text-left">实验ID</th>
                        <th className="border border-gray-300 px-4 py-2 text-left">策略</th>
                        <th className="border border-gray-300 px-4 py-2 text-left">范围</th>
                        <th className="border border-gray-300 px-4 py-2 text-left">持续时间</th>
                        <th className="border border-gray-300 px-4 py-2 text-left">步骤</th>
                        <th className="border border-gray-300 px-4 py-2 text-left">创建时间</th>
                        <th className="border border-gray-300 px-4 py-2 text-left">操作</th>
                      </tr>
                    </thead>
                    <tbody>
                      {plans.map((plan) => (
                        <tr key={plan.plan_id}>
                          <td className="border border-gray-300 px-4 py-2">{plan.experiment_id}</td>
                          <td className="border border-gray-300 px-4 py-2">
                            <Badge variant="secondary">{plan.strategy}</Badge>
                          </td>
                          <td className="border border-gray-300 px-4 py-2">
                            {plan.start_percentage}% → {plan.target_percentage}%
                          </td>
                          <td className="border border-gray-300 px-4 py-2">{plan.duration_hours}h</td>
                          <td className="border border-gray-300 px-4 py-2">{plan.num_steps}</td>
                          <td className="border border-gray-300 px-4 py-2 text-sm">
                            {new Date(plan.created_at).toLocaleString('zh-CN')}
                          </td>
                          <td className="border border-gray-300 px-4 py-2">
                            <Button 
                              size="sm" 
                              onClick={() => handleStartRamp(plan.plan_id)}
                              disabled={loading}
                            >
                              启动
                            </Button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  没有创建的计划
                </div>
              )}
            </Card>
          )
        },

        {
          key: 'history',
          label: '执行历史',
          children: (
            <Card title={`执行历史 (${executions.length})`}>
              {executions.length > 0 ? (
                <Space direction="vertical" style={{ width: '100%' }}>
                  {executions.map((execution) => (
                    <Card key={execution.exec_id} style={{ borderLeft: '4px solid #d9d9d9' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                          <div style={{ flex: 1 }}>
                            <Space direction="vertical" style={{ width: '100%' }}>
                              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                <Text strong>实验: {execution.experiment_id}</Text>
                                <Badge 
                                  status={execution.status === RampStatus.RUNNING ? 'processing' : 
                                         execution.status === RampStatus.COMPLETED ? 'success' :
                                         execution.status === RampStatus.PAUSED ? 'warning' : 'error'} 
                                  text={execution.status}
                                />
                              </div>
                              <div style={{ fontSize: '14px', color: '#666' }}>
                                <div>执行ID: {execution.exec_id}</div>
                                <div>计划ID: {execution.plan_id}</div>
                                <div>当前进度: {execution.current_percentage.toFixed(1)}% (步骤 {execution.current_step})</div>
                                {execution.started_at && (
                                  <div>开始时间: {new Date(execution.started_at).toLocaleString('zh-CN')}</div>
                                )}
                                {execution.completed_at && (
                                  <div>完成时间: {new Date(execution.completed_at).toLocaleString('zh-CN')}</div>
                                )}
                                {execution.rollback_reason && (
                                  <div style={{ color: '#ff4d4f' }}>回滚原因: {execution.rollback_reason}</div>
                                )}
                              </div>
                              <div style={{ width: '100%', backgroundColor: '#f0f0f0', borderRadius: '4px', height: '8px' }}>
                                <div 
                                  style={{ 
                                    height: '8px', 
                                    borderRadius: '4px',
                                    width: `${getProgressPercentage(execution)}%`,
                                    backgroundColor: execution.status === RampStatus.COMPLETED ? '#52c41a' :
                                                   execution.status === RampStatus.FAILED || execution.status === RampStatus.ROLLBACK ? '#ff4d4f' :
                                                   '#1890ff'
                                  }}
                                ></div>
                              </div>
                            </Space>
                          </div>
                          {(execution.status === RampStatus.RUNNING || execution.status === RampStatus.PAUSED) && (
                            <div style={{ display: 'flex', gap: '8px' }}>
                              {execution.status === RampStatus.RUNNING && (
                                <Button 
                                  size="small" 
                                  icon={<PauseCircleOutlined />}
                                  onClick={() => handlePauseRamp(execution.exec_id)}
                                >
                                  暂停
                                </Button>
                              )}
                              {execution.status === RampStatus.PAUSED && (
                                <Button 
                                  size="small"
                                  type="primary"
                                  icon={<PlayCircleOutlined />}
                                  onClick={() => handleResumeRamp(execution.exec_id)}
                                >
                                  恢复
                                </Button>
                              )}
                              <Button 
                                size="small" 
                                danger
                                icon={<RollbackOutlined />}
                                onClick={() => handleRollback(execution.exec_id)}
                              >
                                回滚
                              </Button>
                            </div>
                          )}
                        </div>
                    </Card>
                  ))}
                </Space>
              ) : (
                <div style={{ textAlign: 'center', padding: '40px', color: '#999' }}>
                  没有执行记录
                </div>
              )}
            </Card>
          )
        },
      ]} />
    </div>
  );
};

export default TrafficRampManagementPage;