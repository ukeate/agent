import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Typography,
  Space,
  Button,
  Table,
  Tag,
  Alert,
  Input,
  Form,
  Modal,
  Select,
  Slider,
  Switch,
  Tooltip,
  Statistic,
  Timeline,
  Progress,
  message,
  Tabs,
  Steps,
  Radio,
  Divider,
} from 'antd';
import {
  RocketOutlined,
  SettingOutlined,
  PlusOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  SyncOutlined,
  PauseCircleOutlined,
  PlayCircleOutlined,
  RollbackOutlined,
  WarningOutlined,
  InfoCircleOutlined,
  ClockCircleOutlined,
  ThunderboltOutlined,
  BranchesOutlined,
  AimOutlined,
  DeploymentUnitOutlined,
  MonitorOutlined,
  SafetyCertificateOutlined,
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';

const { Title, Text, Paragraph } = Typography;
const { Option } = Select;
const { TabPane } = Tabs;
const { Step } = Steps;

interface ReleaseStrategy {
  id: string;
  name: string;
  strategy_type: 'canary' | 'blue_green' | 'rolling' | 'feature_flag' | 'ab_test';
  experiment_id: string;
  experiment_name: string;
  status: 'draft' | 'active' | 'paused' | 'completed' | 'rolled_back';
  target_percentage: number;
  current_percentage: number;
  rollout_steps: {
    step: number;
    percentage: number;
    duration_hours: number;
    criteria: string[];
    status: 'pending' | 'active' | 'completed' | 'failed';
    started_at?: string;
    completed_at?: string;
  }[];
  success_criteria: {
    metric: string;
    threshold: number;
    comparison: 'greater_than' | 'less_than' | 'equal';
    current_value: number;
    status: 'pass' | 'fail' | 'pending';
  }[];
  rollback_triggers: {
    trigger: string;
    threshold: number;
    enabled: boolean;
  }[];
  created_at: string;
  updated_at: string;
  creator: string;
}

interface RiskAssessment {
  overall_risk: 'low' | 'medium' | 'high';
  risk_factors: {
    factor: string;
    level: 'low' | 'medium' | 'high';
    description: string;
    impact: number;
  }[];
  mitigation_actions: string[];
  confidence_score: number;
}

const ReleaseStrategyPage: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [selectedTab, setSelectedTab] = useState<string>('strategies');
  const [modalVisible, setModalVisible] = useState(false);
  const [rollbackModalVisible, setRollbackModalVisible] = useState(false);
  const [selectedStrategy, setSelectedStrategy] = useState<string>('');
  const [form] = Form.useForm();

  // 模拟发布策略数据
  const releaseStrategies: ReleaseStrategy[] = [
    {
      id: 'strategy_001',
      name: '首页改版金丝雀发布',
      strategy_type: 'canary',
      experiment_id: 'exp_001',
      experiment_name: '首页改版A/B测试',
      status: 'active',
      target_percentage: 100,
      current_percentage: 25,
      rollout_steps: [
        {
          step: 1,
          percentage: 5,
          duration_hours: 24,
          criteria: ['error_rate < 1%', 'response_time < 200ms'],
          status: 'completed',
          started_at: '2024-01-20 10:00',
          completed_at: '2024-01-21 10:00',
        },
        {
          step: 2,
          percentage: 25,
          duration_hours: 48,
          criteria: ['error_rate < 0.5%', 'user_satisfaction > 4.0'],
          status: 'active',
          started_at: '2024-01-21 10:00',
        },
        {
          step: 3,
          percentage: 50,
          duration_hours: 72,
          criteria: ['conversion_rate >= baseline', 'bounce_rate < 30%'],
          status: 'pending',
        },
        {
          step: 4,
          percentage: 100,
          duration_hours: 96,
          criteria: ['all_metrics_stable'],
          status: 'pending',
        },
      ],
      success_criteria: [
        {
          metric: '错误率',
          threshold: 1.0,
          comparison: 'less_than',
          current_value: 0.3,
          status: 'pass',
        },
        {
          metric: '响应时间',
          threshold: 200,
          comparison: 'less_than',
          current_value: 145,
          status: 'pass',
        },
        {
          metric: '转化率',
          threshold: 14.5,
          comparison: 'greater_than',
          current_value: 16.2,
          status: 'pass',
        },
      ],
      rollback_triggers: [
        { trigger: '错误率超过2%', threshold: 2.0, enabled: true },
        { trigger: '响应时间超过500ms', threshold: 500, enabled: true },
        { trigger: '用户投诉增加50%', threshold: 50, enabled: true },
      ],
      created_at: '2024-01-18',
      updated_at: '2024-01-22',
      creator: 'Product Team',
    },
    {
      id: 'strategy_002',
      name: '结算页蓝绿部署',
      strategy_type: 'blue_green',
      experiment_id: 'exp_002',
      experiment_name: '结算页面优化',
      status: 'completed',
      target_percentage: 100,
      current_percentage: 100,
      rollout_steps: [
        {
          step: 1,
          percentage: 0,
          duration_hours: 2,
          criteria: ['deployment_health_check'],
          status: 'completed',
          started_at: '2024-01-15 14:00',
          completed_at: '2024-01-15 16:00',
        },
        {
          step: 2,
          percentage: 100,
          duration_hours: 1,
          criteria: ['traffic_switch_successful'],
          status: 'completed',
          started_at: '2024-01-15 16:00',
          completed_at: '2024-01-15 17:00',
        },
      ],
      success_criteria: [
        {
          metric: '部署成功率',
          threshold: 100,
          comparison: 'equal',
          current_value: 100,
          status: 'pass',
        },
        {
          metric: '流量切换时间',
          threshold: 60,
          comparison: 'less_than',
          current_value: 15,
          status: 'pass',
        },
      ],
      rollback_triggers: [
        { trigger: '部署失败', threshold: 0, enabled: true },
        { trigger: '健康检查失败', threshold: 0, enabled: true },
      ],
      created_at: '2024-01-12',
      updated_at: '2024-01-15',
      creator: 'DevOps Team',
    },
    {
      id: 'strategy_003',
      name: '推荐算法特性开关',
      strategy_type: 'feature_flag',
      experiment_id: 'exp_003',
      experiment_name: '推荐算法测试',
      status: 'paused',
      target_percentage: 20,
      current_percentage: 20,
      rollout_steps: [
        {
          step: 1,
          percentage: 20,
          duration_hours: 168, // 7 days
          criteria: ['algorithm_performance_stable'],
          status: 'active',
          started_at: '2024-01-19 09:00',
        },
      ],
      success_criteria: [
        {
          metric: 'CTR提升',
          threshold: 5.0,
          comparison: 'greater_than',
          current_value: 3.2,
          status: 'pending',
        },
        {
          metric: '算法响应时间',
          threshold: 100,
          comparison: 'less_than',
          current_value: 85,
          status: 'pass',
        },
      ],
      rollback_triggers: [
        { trigger: 'CTR下降超过2%', threshold: -2.0, enabled: true },
        { trigger: '算法错误率超过1%', threshold: 1.0, enabled: true },
      ],
      created_at: '2024-01-18',
      updated_at: '2024-01-22',
      creator: 'ML Team',
    },
  ];

  const riskAssessment: RiskAssessment = {
    overall_risk: 'medium',
    risk_factors: [
      {
        factor: '用户体验影响',
        level: 'medium',
        description: '新版本可能影响用户使用习惯',
        impact: 70,
      },
      {
        factor: '技术复杂度',
        level: 'low',
        description: '变更涉及前端展示层，风险可控',
        impact: 30,
      },
      {
        factor: '业务影响范围',
        level: 'high',
        description: '影响核心业务流程和转化',
        impact: 85,
      },
      {
        factor: '回滚复杂度',
        level: 'low',
        description: '可快速回滚到原版本',
        impact: 20,
      },
    ],
    mitigation_actions: [
      '设置严格的成功标准和自动回滚触发器',
      '实施分阶段发布，每阶段充分验证',
      '准备详细的回滚预案和操作手册',
      '建立实时监控和告警机制',
    ],
    confidence_score: 78.5,
  };

  const getStrategyTypeText = (type: string) => {
    switch (type) {
      case 'canary': return '金丝雀发布';
      case 'blue_green': return '蓝绿部署';
      case 'rolling': return '滚动更新';
      case 'feature_flag': return '特性开关';
      case 'ab_test': return 'A/B测试';
      default: return type;
    }
  };

  const getStrategyTypeColor = (type: string) => {
    switch (type) {
      case 'canary': return 'orange';
      case 'blue_green': return 'blue';
      case 'rolling': return 'green';
      case 'feature_flag': return 'purple';
      case 'ab_test': return 'gold';
      default: return 'default';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'draft': return 'default';
      case 'active': return 'processing';
      case 'paused': return 'warning';
      case 'completed': return 'success';
      case 'rolled_back': return 'error';
      default: return 'default';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'draft': return '草稿';
      case 'active': return '执行中';
      case 'paused': return '已暂停';
      case 'completed': return '已完成';
      case 'rolled_back': return '已回滚';
      default: return status;
    }
  };

  const getRiskColor = (level: string) => {
    switch (level) {
      case 'low': return '#52c41a';
      case 'medium': return '#faad14';
      case 'high': return '#ff4d4f';
      default: return '#d9d9d9';
    }
  };

  const getRiskText = (level: string) => {
    switch (level) {
      case 'low': return '低';
      case 'medium': return '中';
      case 'high': return '高';
      default: return level;
    }
  };

  const strategiesColumns: ColumnsType<ReleaseStrategy> = [
    {
      title: '策略名称',
      key: 'name',
      width: 200,
      render: (_, record: ReleaseStrategy) => (
        <div>
          <Text strong>{record.name}</Text>
          <br />
          <Tag color={getStrategyTypeColor(record.strategy_type)}>
            {getStrategyTypeText(record.strategy_type)}
          </Tag>
        </div>
      ),
    },
    {
      title: '关联实验',
      dataIndex: 'experiment_name',
      key: 'experiment_name',
      width: 180,
      render: (name: string, record: ReleaseStrategy) => (
        <div>
          <Text>{name}</Text>
          <br />
          <Text type="secondary" style={{ fontSize: '12px' }}>
            {record.experiment_id}
          </Text>
        </div>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (status: string) => (
        <Tag color={getStatusColor(status)}>
          {getStatusText(status)}
        </Tag>
      ),
    },
    {
      title: '发布进度',
      key: 'progress',
      width: 150,
      render: (_, record: ReleaseStrategy) => (
        <div>
          <div style={{ marginBottom: '4px' }}>
            <Text strong>{record.current_percentage}% / {record.target_percentage}%</Text>
          </div>
          <Progress
            percent={(record.current_percentage / record.target_percentage) * 100}
            size="small"
            strokeColor="#1890ff"
          />
        </div>
      ),
    },
    {
      title: '当前阶段',
      key: 'current_stage',
      width: 120,
      render: (_, record: ReleaseStrategy) => {
        const currentStep = record.rollout_steps.find(step => step.status === 'active') || 
                           record.rollout_steps[record.rollout_steps.length - 1];
        return (
          <div>
            <Text>第{currentStep.step}阶段</Text>
            <br />
            <Text type="secondary" style={{ fontSize: '12px' }}>
              {currentStep.percentage}% 流量
            </Text>
          </div>
        );
      },
    },
    {
      title: '成功标准',
      key: 'success_criteria_status',
      width: 120,
      render: (_, record: ReleaseStrategy) => {
        const passed = record.success_criteria.filter(c => c.status === 'pass').length;
        const total = record.success_criteria.length;
        const passRate = total > 0 ? (passed / total) * 100 : 0;
        
        return (
          <div style={{ textAlign: 'center' }}>
            <Progress
              type="circle"
              size={50}
              percent={passRate}
              format={() => `${passed}/${total}`}
              strokeColor={passRate === 100 ? '#52c41a' : passRate >= 50 ? '#faad14' : '#ff4d4f'}
            />
          </div>
        );
      },
    },
    {
      title: '操作',
      key: 'actions',
      width: 120,
      render: (_, record: ReleaseStrategy) => (
        <Space size="small">
          <Tooltip title="查看详情">
            <Button 
              type="text" 
              size="small" 
              icon={<MonitorOutlined />}
              onClick={() => {
                setSelectedStrategy(record.id);
                // 展开详情逻辑
              }}
            />
          </Tooltip>
          {record.status === 'active' && (
            <Tooltip title="暂停">
              <Button 
                type="text" 
                size="small" 
                icon={<PauseCircleOutlined />}
                onClick={() => message.success('策略已暂停')}
              />
            </Tooltip>
          )}
          {(record.status === 'paused' || record.status === 'draft') && (
            <Tooltip title="启动">
              <Button 
                type="text" 
                size="small" 
                icon={<PlayCircleOutlined />}
                onClick={() => message.success('策略已启动')}
              />
            </Tooltip>
          )}
          <Tooltip title="回滚">
            <Button 
              type="text" 
              size="small" 
              icon={<RollbackOutlined />}
              danger
              onClick={() => {
                setSelectedStrategy(record.id);
                setRollbackModalVisible(true);
              }}
            />
          </Tooltip>
        </Space>
      ),
    },
  ];

  const handleCreateStrategy = () => {
    setModalVisible(true);
  };

  useEffect(() => {
    setLoading(true);
    setTimeout(() => {
      setLoading(false);
    }, 500);
  }, [selectedTab]);

  return (
    <div style={{ padding: '24px' }}>
      {/* 页面标题 */}
      <div style={{ marginBottom: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <Title level={2} style={{ margin: 0 }}>
              <RocketOutlined /> 发布策略
            </Title>
            <Text type="secondary">管理实验结果的发布和部署策略</Text>
          </div>
          <Space>
            <Button icon={<BranchesOutlined />}>
              策略模板
            </Button>
            <Button type="primary" icon={<PlusOutlined />} onClick={handleCreateStrategy}>
              创建策略
            </Button>
          </Space>
        </div>
      </div>

      {/* 策略概览统计 */}
      <Row gutter={16} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="总策略数"
              value={releaseStrategies.length}
              prefix={<DeploymentUnitOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="执行中"
              value={releaseStrategies.filter(s => s.status === 'active').length}
              valueStyle={{ color: '#1890ff' }}
              prefix={<SyncOutlined spin />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="已完成"
              value={releaseStrategies.filter(s => s.status === 'completed').length}
              valueStyle={{ color: '#52c41a' }}
              prefix={<CheckCircleOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="成功率"
              value={85.7}
              precision={1}
              suffix="%"
              valueStyle={{ color: '#52c41a' }}
              prefix={<AimOutlined />}
            />
          </Card>
        </Col>
      </Row>

      {/* 主要内容 */}
      <Tabs activeKey={selectedTab} onChange={setSelectedTab}>
        <TabPane tab="发布策略" key="strategies">
          <Card>
            <Table
              columns={strategiesColumns}
              dataSource={releaseStrategies}
              rowKey="id"
              loading={loading}
              pagination={false}
              expandable={{
                expandedRowRender: (record) => (
                  <div style={{ margin: 0 }}>
                    <Row gutter={16}>
                      <Col span={12}>
                        <Title level={5}>发布步骤</Title>
                        <Steps
                          direction="vertical"
                          size="small"
                          current={record.rollout_steps.findIndex(step => step.status === 'active')}
                        >
                          {record.rollout_steps.map((step, index) => (
                            <Step
                              key={index}
                              title={`第${step.step}阶段 - ${step.percentage}%`}
                              description={
                                <div>
                                  <div>持续时间: {step.duration_hours}小时</div>
                                  <div>条件: {step.criteria.join(', ')}</div>
                                  {step.started_at && (
                                    <div>开始时间: {step.started_at}</div>
                                  )}
                                </div>
                              }
                              status={
                                step.status === 'completed' ? 'finish' :
                                step.status === 'active' ? 'process' :
                                step.status === 'failed' ? 'error' : 'wait'
                              }
                            />
                          ))}
                        </Steps>
                      </Col>
                      <Col span={12}>
                        <Title level={5}>成功标准</Title>
                        <Space direction="vertical" style={{ width: '100%' }}>
                          {record.success_criteria.map((criterion, index) => (
                            <div key={index} style={{ 
                              padding: '8px', 
                              border: '1px solid #f0f0f0', 
                              borderRadius: '4px' 
                            }}>
                              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <Text strong>{criterion.metric}</Text>
                                <Tag color={
                                  criterion.status === 'pass' ? 'green' :
                                  criterion.status === 'fail' ? 'red' : 'orange'
                                }>
                                  {criterion.status === 'pass' ? '通过' :
                                   criterion.status === 'fail' ? '失败' : '待测'}
                                </Tag>
                              </div>
                              <div style={{ marginTop: '4px', fontSize: '12px' }}>
                                当前值: {criterion.current_value} {criterion.comparison === 'greater_than' ? '>' : 
                                        criterion.comparison === 'less_than' ? '<' : '='} {criterion.threshold}
                              </div>
                            </div>
                          ))}
                        </Space>
                      </Col>
                    </Row>
                  </div>
                ),
              }}
            />
          </Card>
        </TabPane>

        <TabPane tab="风险评估" key="risk">
          <Row gutter={16}>
            <Col span={16}>
              <Card title="风险因素分析">
                <div style={{ marginBottom: '16px' }}>
                  <Row gutter={16}>
                    <Col span={8}>
                      <div style={{ textAlign: 'center' }}>
                        <div style={{ 
                          fontSize: '48px', 
                          fontWeight: 'bold',
                          color: getRiskColor(riskAssessment.overall_risk)
                        }}>
                          {getRiskText(riskAssessment.overall_risk).toUpperCase()}
                        </div>
                        <div style={{ marginTop: '8px' }}>
                          <Text>整体风险等级</Text>
                        </div>
                      </div>
                    </Col>
                    <Col span={16}>
                      <div>
                        <Text strong>风险因素详情：</Text>
                        <div style={{ marginTop: '12px' }}>
                          {riskAssessment.risk_factors.map((factor, index) => (
                            <div key={index} style={{ marginBottom: '16px' }}>
                              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                                <Text strong>{factor.factor}</Text>
                                <Tag color={getRiskColor(factor.level)}>
                                  {getRiskText(factor.level)}风险
                                </Tag>
                              </div>
                              <div style={{ marginBottom: '4px' }}>
                                <Text style={{ fontSize: '12px' }}>{factor.description}</Text>
                              </div>
                              <Progress
                                percent={factor.impact}
                                size="small"
                                strokeColor={getRiskColor(factor.level)}
                                format={(percent) => `影响度: ${percent}%`}
                              />
                            </div>
                          ))}
                        </div>
                      </div>
                    </Col>
                  </Row>
                </div>
                
                <Divider />
                
                <div>
                  <Text strong>缓解措施：</Text>
                  <div style={{ marginTop: '8px' }}>
                    {riskAssessment.mitigation_actions.map((action, index) => (
                      <div key={index} style={{ marginBottom: '4px' }}>
                        <CheckCircleOutlined style={{ color: '#52c41a', marginRight: '8px' }} />
                        <Text>{action}</Text>
                      </div>
                    ))}
                  </div>
                </div>
              </Card>
            </Col>

            <Col span={8}>
              <Card title="置信度评分" style={{ marginBottom: '16px' }}>
                <div style={{ textAlign: 'center' }}>
                  <Progress
                    type="circle"
                    percent={riskAssessment.confidence_score}
                    format={(percent) => `${percent}%`}
                    strokeColor={
                      riskAssessment.confidence_score >= 80 ? '#52c41a' :
                      riskAssessment.confidence_score >= 60 ? '#faad14' : '#ff4d4f'
                    }
                    size={120}
                  />
                  <div style={{ marginTop: '16px' }}>
                    <Text>基于历史数据和专家评估</Text>
                  </div>
                </div>
              </Card>

              <Card title="回滚触发器" size="small">
                <Space direction="vertical" style={{ width: '100%' }}>
                  {releaseStrategies[0].rollback_triggers.map((trigger, index) => (
                    <div key={index} style={{ 
                      display: 'flex', 
                      justifyContent: 'space-between', 
                      alignItems: 'center',
                      padding: '8px',
                      backgroundColor: trigger.enabled ? '#f6ffed' : '#f5f5f5',
                      borderRadius: '4px'
                    }}>
                      <div>
                        <Text strong style={{ fontSize: '12px' }}>{trigger.trigger}</Text>
                        <br />
                        <Text type="secondary" style={{ fontSize: '11px' }}>
                          阈值: {trigger.threshold}
                        </Text>
                      </div>
                      <Switch 
                        size="small" 
                        checked={trigger.enabled}
                        checkedChildren={<CheckCircleOutlined />}
                        unCheckedChildren={<CloseCircleOutlined />}
                      />
                    </div>
                  ))}
                </Space>
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab="发布历史" key="history">
          <Card>
            <Timeline>
              <Timeline.Item color="green" dot={<CheckCircleOutlined />}>
                <div>
                  <Text strong>结算页蓝绿部署完成</Text>
                  <br />
                  <Text type="secondary">2024-01-15 17:00 - 成功切换到新版本，零停机时间</Text>
                </div>
              </Timeline.Item>
              <Timeline.Item color="blue" dot={<SyncOutlined />}>
                <div>
                  <Text strong>首页改版金丝雀发布进行中</Text>
                  <br />
                  <Text type="secondary">2024-01-21 10:00 - 当前阶段：25%流量，所有指标正常</Text>
                </div>
              </Timeline.Item>
              <Timeline.Item color="red" dot={<RollbackOutlined />}>
                <div>
                  <Text strong>推荐算法特性开关暂停</Text>
                  <br />
                  <Text type="secondary">2024-01-22 15:30 - 由于性能指标未达预期，暂停发布</Text>
                </div>
              </Timeline.Item>
              <Timeline.Item color="gray" dot={<ClockCircleOutlined />}>
                <div>
                  <Text strong>新实验策略创建</Text>
                  <br />
                  <Text type="secondary">2024-01-18 14:20 - 创建了3个新的发布策略草稿</Text>
                </div>
              </Timeline.Item>
            </Timeline>
          </Card>
        </TabPane>
      </Tabs>

      {/* 创建策略模态框 */}
      <Modal
        title="创建发布策略"
        open={modalVisible}
        onCancel={() => setModalVisible(false)}
        footer={[
          <Button key="cancel" onClick={() => setModalVisible(false)}>
            取消
          </Button>,
          <Button key="submit" type="primary" onClick={() => {
            message.success('发布策略已创建');
            setModalVisible(false);
          }}>
            创建策略
          </Button>,
        ]}
        width={700}
      >
        <Form form={form} layout="vertical">
          <Form.Item label="策略名称" name="name" required>
            <Input placeholder="输入发布策略名称" />
          </Form.Item>
          
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item label="策略类型" name="strategy_type" required>
                <Select placeholder="选择发布策略类型">
                  <Option value="canary">金丝雀发布</Option>
                  <Option value="blue_green">蓝绿部署</Option>
                  <Option value="rolling">滚动更新</Option>
                  <Option value="feature_flag">特性开关</Option>
                  <Option value="ab_test">A/B测试</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="关联实验" name="experiment_id" required>
                <Select placeholder="选择要发布的实验">
                  <Option value="exp_001">首页改版A/B测试</Option>
                  <Option value="exp_002">结算页面优化</Option>
                  <Option value="exp_003">推荐算法测试</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>

          <Form.Item label="目标流量百分比" name="target_percentage">
            <Slider
              min={0}
              max={100}
              defaultValue={100}
              marks={{
                0: '0%',
                25: '25%',
                50: '50%',
                75: '75%',
                100: '100%',
              }}
            />
          </Form.Item>

          <Form.Item label="发布阶段" name="rollout_steps">
            <div style={{ border: '1px solid #d9d9d9', padding: '16px', borderRadius: '6px' }}>
              <Text strong>阶段1：</Text>
              <Row gutter={8} style={{ marginTop: '8px', marginBottom: '16px' }}>
                <Col span={8}>
                  <Input placeholder="流量比例" suffix="%" />
                </Col>
                <Col span={8}>
                  <Input placeholder="持续时间" suffix="小时" />
                </Col>
                <Col span={8}>
                  <Input placeholder="成功条件" />
                </Col>
              </Row>
              <Button type="dashed" style={{ width: '100%' }}>
                <PlusOutlined /> 添加阶段
              </Button>
            </div>
          </Form.Item>

          <Form.Item label="回滚触发器" name="rollback_triggers">
            <div style={{ border: '1px solid #d9d9d9', padding: '16px', borderRadius: '6px' }}>
              <Row gutter={8} style={{ marginBottom: '8px' }}>
                <Col span={16}>
                  <Input placeholder="触发条件描述" />
                </Col>
                <Col span={6}>
                  <Input placeholder="阈值" />
                </Col>
                <Col span={2}>
                  <Switch size="small" defaultChecked />
                </Col>
              </Row>
              <Button type="dashed" style={{ width: '100%' }}>
                <PlusOutlined /> 添加触发器
              </Button>
            </div>
          </Form.Item>

          <Form.Item label="立即启动" name="auto_start">
            <Switch />
          </Form.Item>
        </Form>
      </Modal>

      {/* 回滚确认模态框 */}
      <Modal
        title="确认回滚"
        open={rollbackModalVisible}
        onCancel={() => setRollbackModalVisible(false)}
        footer={[
          <Button key="cancel" onClick={() => setRollbackModalVisible(false)}>
            取消
          </Button>,
          <Button key="rollback" type="primary" danger onClick={() => {
            message.success('回滚操作已执行');
            setRollbackModalVisible(false);
          }}>
            确认回滚
          </Button>,
        ]}
      >
        <Alert
          message="回滚警告"
          description="此操作将回滚到上一个稳定版本。请确认您了解回滚的影响范围和可能的业务中断。"
          variant="warning"
          showIcon
          style={{ marginBottom: '16px' }}
        />
        
        <Form layout="vertical">
          <Form.Item label="回滚原因" required>
            <Select placeholder="选择回滚原因">
              <Option value="performance_issue">性能问题</Option>
              <Option value="error_rate_high">错误率过高</Option>
              <Option value="user_feedback">用户反馈负面</Option>
              <Option value="business_impact">业务影响严重</Option>
              <Option value="other">其他原因</Option>
            </Select>
          </Form.Item>
          
          <Form.Item label="回滚说明">
            <Input.TextArea 
              rows={3} 
              placeholder="详细描述回滚的具体原因和发现的问题" 
            />
          </Form.Item>

          <Form.Item label="回滚策略">
            <Radio.Group defaultValue="immediate">
              <Radio value="immediate">立即回滚</Radio>
              <Radio value="gradual">渐进式回滚</Radio>
            </Radio.Group>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default ReleaseStrategyPage;