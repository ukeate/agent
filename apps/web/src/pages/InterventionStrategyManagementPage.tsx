import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Table,
  Tag,
  Button,
  Space,
  Modal,
  Form,
  Input,
  Select,
  Typography,
  Divider,
  Progress,
  Timeline,
  Badge,
  Tooltip,
  Avatar,
  List,
  Rate,
  Tabs
} from 'antd';
import {
  HeartHandshakeIcon,
  UserIcon,
  TargetIcon,
  TrendingUpIcon,
  ClockIcon,
  CheckCircleIcon,
  AlertCircleIcon,
  SettingsIcon,
  BookOpenIcon,
  ActivityIcon
} from 'lucide-react';

const { Title, Text } = Typography;
const { TextArea } = Input;
const { Option } = Select;
const { TabPane } = Tabs;

interface InterventionStrategy {
  strategy_id: string;
  name: string;
  type: 'supportive' | 'corrective' | 'crisis';
  description: string;
  target_risk_factors: string[];
  effectiveness_score: number;
  usage_count: number;
  success_rate: number;
  duration_range: string;
  resource_requirements: string[];
  created_at: string;
  last_updated: string;
}

interface InterventionPlan {
  plan_id: string;
  user_id: string;
  intervention_type: string;
  urgency_level: 'low' | 'medium' | 'high' | 'critical';
  target_risk_factors: string[];
  strategies: InterventionStrategy[];
  primary_strategy: InterventionStrategy;
  timeline: any;
  success_metrics: string[];
  monitoring_frequency: string;
  status: 'draft' | 'active' | 'completed' | 'cancelled';
  created_at: string;
  progress: number;
}

interface InterventionOutcome {
  plan_id: string;
  outcome_type: 'success' | 'partial_success' | 'no_improvement' | 'deterioration';
  improvement_score: number;
  feedback_rating: number;
  duration_days: number;
  notes: string;
  recorded_at: string;
}

const InterventionStrategyManagementPage: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [strategies, setStrategies] = useState<InterventionStrategy[]>([]);
  const [activePlans, setActivePlans] = useState<InterventionPlan[]>([]);
  const [outcomes, setOutcomes] = useState<InterventionOutcome[]>([]);
  const [strategyModalVisible, setStrategyModalVisible] = useState(false);
  const [planModalVisible, setPlanModalVisible] = useState(false);
  const [selectedStrategy, setSelectedStrategy] = useState<InterventionStrategy | null>(null);
  const [selectedPlan, setSelectedPlan] = useState<InterventionPlan | null>(null);
  
  const [stats, setStats] = useState({
    totalStrategies: 0,
    activePlans: 0,
    successRate: 0,
    avgImprovementScore: 0
  });

  const [strategyForm] = Form.useForm();
  const [planForm] = Form.useForm();

  // 模拟数据
  useEffect(() => {
    const mockStrategies: InterventionStrategy[] = [
      {
        strategy_id: '1',
        name: '认知行为疗法支持',
        type: 'supportive',
        description: '通过认知重构和行为调整帮助用户改善情感状态',
        target_risk_factors: ['depression_indicators', 'anxiety_indicators'],
        effectiveness_score: 4.5,
        usage_count: 156,
        success_rate: 78.5,
        duration_range: '4-8周',
        resource_requirements: ['专业心理咨询师', '结构化干预材料', '定期跟进'],
        created_at: '2024-01-15',
        last_updated: '2024-03-20'
      },
      {
        strategy_id: '2',
        name: '危机干预协议',
        type: 'crisis',
        description: '针对自杀风险和严重危机的立即响应策略',
        target_risk_factors: ['suicidal_ideation', 'severe_depression'],
        effectiveness_score: 4.8,
        usage_count: 45,
        success_rate: 92.3,
        duration_range: '立即-72小时',
        resource_requirements: ['24小时热线', '紧急心理医生', '安全监护'],
        created_at: '2024-02-01',
        last_updated: '2024-03-25'
      },
      {
        strategy_id: '3',
        name: '正念减压训练',
        type: 'corrective',
        description: '通过正念练习和压力管理技巧缓解焦虑症状',
        target_risk_factors: ['anxiety_indicators', 'stress_overload'],
        effectiveness_score: 4.2,
        usage_count: 203,
        success_rate: 71.4,
        duration_range: '2-6周',
        resource_requirements: ['正念训练师', '练习材料', '定期指导'],
        created_at: '2024-01-10',
        last_updated: '2024-03-18'
      }
    ];

    const mockPlans: InterventionPlan[] = [
      {
        plan_id: '1',
        user_id: 'user_001',
        intervention_type: 'corrective',
        urgency_level: 'high',
        target_risk_factors: ['depression_indicators'],
        strategies: [mockStrategies[0]],
        primary_strategy: mockStrategies[0],
        timeline: {
          start_date: '2024-03-01',
          end_date: '2024-04-30',
          milestones: ['初始评估', '中期检查', '最终评估']
        },
        success_metrics: ['情绪改善度', '日常功能恢复', '自我报告满意度'],
        monitoring_frequency: '每周',
        status: 'active',
        created_at: '2024-03-01',
        progress: 65
      },
      {
        plan_id: '2',
        user_id: 'user_002',
        intervention_type: 'crisis',
        urgency_level: 'critical',
        target_risk_factors: ['suicidal_ideation'],
        strategies: [mockStrategies[1]],
        primary_strategy: mockStrategies[1],
        timeline: {
          start_date: '2024-03-25',
          end_date: '2024-03-28',
          milestones: ['立即安全评估', '24小时监护', '稳定化处理']
        },
        success_metrics: ['安全状态', '危机解除', '稳定性评估'],
        monitoring_frequency: '每小时',
        status: 'active',
        created_at: '2024-03-25',
        progress: 40
      }
    ];

    const mockOutcomes: InterventionOutcome[] = [
      {
        plan_id: 'plan_123',
        outcome_type: 'success',
        improvement_score: 8.5,
        feedback_rating: 4.7,
        duration_days: 42,
        notes: '用户情绪显著改善，日常功能完全恢复',
        recorded_at: '2024-03-20'
      },
      {
        plan_id: 'plan_124',
        outcome_type: 'partial_success',
        improvement_score: 6.2,
        feedback_rating: 4.1,
        duration_days: 35,
        notes: '部分改善，需要继续跟进',
        recorded_at: '2024-03-18'
      }
    ];

    setStrategies(mockStrategies);
    setActivePlans(mockPlans);
    setOutcomes(mockOutcomes);
    setStats({
      totalStrategies: mockStrategies.length,
      activePlans: mockPlans.length,
      successRate: mockOutcomes.filter(o => o.outcome_type === 'success').length / mockOutcomes.length * 100,
      avgImprovementScore: mockOutcomes.reduce((acc, o) => acc + o.improvement_score, 0) / mockOutcomes.length
    });
  }, []);

  const strategyTypeColor = (type: string) => {
    const colors: Record<string, string> = {
      'supportive': '#52c41a',
      'corrective': '#faad14',
      'crisis': '#f5222d'
    };
    return colors[type] || '#d9d9d9';
  };

  const urgencyColor = (level: string) => {
    const colors: Record<string, string> = {
      'low': 'green',
      'medium': 'orange',
      'high': 'red',
      'critical': 'purple'
    };
    return colors[level] || 'default';
  };

  const statusColor = (status: string) => {
    const colors: Record<string, string> = {
      'draft': 'default',
      'active': 'processing',
      'completed': 'success',
      'cancelled': 'error'
    };
    return colors[status] || 'default';
  };

  const handleCreateStrategy = async (values: any) => {
    setLoading(true);
    try {
      // 模拟API调用
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const newStrategy: InterventionStrategy = {
        strategy_id: Date.now().toString(),
        name: values.name,
        type: values.type,
        description: values.description,
        target_risk_factors: values.target_risk_factors || [],
        effectiveness_score: 0,
        usage_count: 0,
        success_rate: 0,
        duration_range: values.duration_range,
        resource_requirements: values.resource_requirements || [],
        created_at: new Date().toISOString().split('T')[0],
        last_updated: new Date().toISOString().split('T')[0]
      };

      setStrategies(prev => [newStrategy, ...prev]);
      setStrategyModalVisible(false);
      strategyForm.resetFields();
    } catch (error) {
      console.error('策略创建失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleCreatePlan = async (values: any) => {
    setLoading(true);
    try {
      // 模拟API调用
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const selectedStrategy = strategies.find(s => s.strategy_id === values.primary_strategy_id);
      const newPlan: InterventionPlan = {
        plan_id: Date.now().toString(),
        user_id: values.user_id,
        intervention_type: values.intervention_type,
        urgency_level: values.urgency_level,
        target_risk_factors: values.target_risk_factors || [],
        strategies: selectedStrategy ? [selectedStrategy] : [],
        primary_strategy: selectedStrategy!,
        timeline: {
          start_date: new Date().toISOString().split('T')[0],
          end_date: values.end_date,
          milestones: values.milestones || []
        },
        success_metrics: values.success_metrics || [],
        monitoring_frequency: values.monitoring_frequency,
        status: 'draft',
        created_at: new Date().toISOString(),
        progress: 0
      };

      setActivePlans(prev => [newPlan, ...prev]);
      setPlanModalVisible(false);
      planForm.resetFields();
    } catch (error) {
      console.error('干预计划创建失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const strategyColumns = [
    {
      title: '策略名称',
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record: InterventionStrategy) => (
        <Space direction="vertical" size="small">
          <Text strong>{name}</Text>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            {record.description}
          </Text>
        </Space>
      )
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => (
        <Tag color={strategyTypeColor(type)}>
          {type.toUpperCase()}
        </Tag>
      )
    },
    {
      title: '效果评分',
      dataIndex: 'effectiveness_score',
      key: 'effectiveness_score',
      render: (score: number) => (
        <Rate disabled value={score} allowHalf style={{ fontSize: '14px' }} />
      )
    },
    {
      title: '使用次数',
      dataIndex: 'usage_count',
      key: 'usage_count',
      render: (count: number) => (
        <Badge count={count} overflowCount={999} />
      )
    },
    {
      title: '成功率',
      dataIndex: 'success_rate',
      key: 'success_rate',
      render: (rate: number) => (
        <Progress
          percent={rate}
          size="small"
          format={percent => `${percent}%`}
          strokeColor={rate > 80 ? '#52c41a' : rate > 60 ? '#faad14' : '#f5222d'}
        />
      )
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: InterventionStrategy) => (
        <Space>
          <Button 
            size="small"
            onClick={() => {
              setSelectedStrategy(record);
              setStrategyModalVisible(true);
            }}
          >
            详情
          </Button>
          <Button size="small" type="primary" ghost>
            编辑
          </Button>
          <Button size="small" type="primary">
            使用
          </Button>
        </Space>
      )
    }
  ];

  const planColumns = [
    {
      title: '用户',
      dataIndex: 'user_id',
      key: 'user_id',
      render: (userId: string) => (
        <Space>
          <Avatar icon={<UserIcon size={14} />} size="small" />
          {userId}
        </Space>
      )
    },
    {
      title: '干预类型',
      dataIndex: 'intervention_type',
      key: 'intervention_type',
      render: (type: string) => (
        <Tag color={strategyTypeColor(type)}>
          {type.toUpperCase()}
        </Tag>
      )
    },
    {
      title: '紧急程度',
      dataIndex: 'urgency_level',
      key: 'urgency_level',
      render: (level: string) => (
        <Tag color={urgencyColor(level)}>
          {level.toUpperCase()}
        </Tag>
      )
    },
    {
      title: '主策略',
      dataIndex: 'primary_strategy',
      key: 'primary_strategy',
      render: (strategy: InterventionStrategy) => (
        <Tooltip title={strategy.description}>
          <Text>{strategy.name}</Text>
        </Tooltip>
      )
    },
    {
      title: '进度',
      dataIndex: 'progress',
      key: 'progress',
      render: (progress: number) => (
        <Progress percent={progress} size="small" />
      )
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Badge status={statusColor(status)} text={status.toUpperCase()} />
      )
    },
    {
      title: '监控频率',
      dataIndex: 'monitoring_frequency',
      key: 'monitoring_frequency',
      render: (frequency: string) => (
        <Space>
          <ClockIcon size={14} />
          {frequency}
        </Space>
      )
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: InterventionPlan) => (
        <Space>
          <Button 
            size="small"
            onClick={() => {
              setSelectedPlan(record);
              setPlanModalVisible(true);
            }}
          >
            详情
          </Button>
          <Button size="small" type="primary" ghost>
            编辑
          </Button>
          {record.status === 'active' && (
            <Button size="small" type="primary">
              更新
            </Button>
          )}
        </Space>
      )
    }
  ];

  return (
    <div style={{ padding: '24px' }}>
      {/* 页面标题 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={24}>
          <Card>
            <Space align="center">
              <HeartHandshakeIcon size={32} />
              <div>
                <Title level={2} style={{ margin: 0 }}>干预策略管理中心</Title>
                <Text type="secondary">管理干预策略库、制定个性化干预计划并跟踪执行效果</Text>
              </div>
            </Space>
          </Card>
        </Col>
      </Row>

      {/* 统计卡片 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="干预策略总数"
              value={stats.totalStrategies}
              prefix={<BookOpenIcon size={16} />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="活跃干预计划"
              value={stats.activePlans}
              prefix={<ActivityIcon size={16} />}
              valueStyle={{ color: '#faad14' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="平均成功率"
              value={stats.successRate.toFixed(1)}
              suffix="%"
              prefix={<CheckCircleIcon size={16} />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="平均改善分数"
              value={stats.avgImprovementScore.toFixed(1)}
              suffix="/10"
              prefix={<TrendingUpIcon size={16} />}
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>
      </Row>

      {/* 操作按钮 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={24}>
          <Space>
            <Button 
              type="primary" 
              icon={<BookOpenIcon size={16} />}
              onClick={() => {
                setSelectedStrategy(null);
                setStrategyModalVisible(true);
              }}
            >
              新建策略
            </Button>
            <Button 
              type="primary"
              ghost
              icon={<TargetIcon size={16} />}
              onClick={() => {
                setSelectedPlan(null);
                setPlanModalVisible(true);
              }}
            >
              创建干预计划
            </Button>
            <Button icon={<SettingsIcon size={16} />}>
              批量操作
            </Button>
            <Button icon={<TrendingUpIcon size={16} />}>
              效果分析
            </Button>
          </Space>
        </Col>
      </Row>

      {/* 主要内容 */}
      <Tabs defaultActiveKey="strategies" type="card">
        <TabPane tab="策略库管理" key="strategies">
          <Card 
            title={
              <Space>
                <BookOpenIcon size={20} />
                干预策略库
              </Space>
            }
            extra={<Button type="link">导入策略</Button>}
          >
            <Table
              dataSource={strategies}
              columns={strategyColumns}
              rowKey="strategy_id"
              pagination={{ pageSize: 10 }}
            />
          </Card>
        </TabPane>

        <TabPane tab="活跃计划" key="plans">
          <Card 
            title={
              <Space>
                <ActivityIcon size={20} />
                活跃干预计划
              </Space>
            }
            extra={<Button type="link">导出报告</Button>}
          >
            <Table
              dataSource={activePlans}
              columns={planColumns}
              rowKey="plan_id"
              pagination={{ pageSize: 10 }}
            />
          </Card>
        </TabPane>

        <TabPane tab="效果评估" key="outcomes">
          <Card 
            title={
              <Space>
                <TrendingUpIcon size={20} />
                干预效果评估
              </Space>
            }
          >
            <List
              itemLayout="horizontal"
              dataSource={outcomes}
              renderItem={(outcome) => (
                <List.Item
                  actions={[
                    <Button size="small">详情</Button>
                  ]}
                >
                  <List.Item.Meta
                    avatar={
                      <Badge 
                        status={outcome.outcome_type === 'success' ? 'success' : 
                               outcome.outcome_type === 'partial_success' ? 'processing' : 'error'}
                      />
                    }
                    title={
                      <Space>
                        <Text>计划 {outcome.plan_id}</Text>
                        <Tag color={outcome.outcome_type === 'success' ? 'green' : 
                                   outcome.outcome_type === 'partial_success' ? 'orange' : 'red'}>
                          {outcome.outcome_type}
                        </Tag>
                      </Space>
                    }
                    description={
                      <Space direction="vertical" size="small">
                        <Space>
                          <Text>改善分数: {outcome.improvement_score}/10</Text>
                          <Text>满意度: {outcome.feedback_rating}/5</Text>
                          <Text>持续时间: {outcome.duration_days}天</Text>
                        </Space>
                        <Text style={{ fontSize: '12px' }}>{outcome.notes}</Text>
                      </Space>
                    }
                  />
                </List.Item>
              )}
            />
          </Card>
        </TabPane>
      </Tabs>

      {/* 策略创建/详情模态框 */}
      <Modal
        title={selectedStrategy ? "策略详情" : "新建干预策略"}
        open={strategyModalVisible}
        onCancel={() => {
          setStrategyModalVisible(false);
          setSelectedStrategy(null);
          strategyForm.resetFields();
        }}
        footer={null}
        width={700}
      >
        {selectedStrategy ? (
          // 显示策略详情
          <div>
            <Row gutter={[16, 16]}>
              <Col span={24}>
                <Card size="small" title="基本信息">
                  <Row gutter={16}>
                    <Col span={12}>
                      <Space direction="vertical" style={{ width: '100%' }}>
                        <div>
                          <Text strong>策略名称: </Text>
                          <Text>{selectedStrategy.name}</Text>
                        </div>
                        <div>
                          <Text strong>类型: </Text>
                          <Tag color={strategyTypeColor(selectedStrategy.type)}>
                            {selectedStrategy.type.toUpperCase()}
                          </Tag>
                        </div>
                        <div>
                          <Text strong>目标风险因子: </Text>
                          <div style={{ marginTop: 4 }}>
                            {selectedStrategy.target_risk_factors.map(factor => (
                              <Tag key={factor} size="small">{factor}</Tag>
                            ))}
                          </div>
                        </div>
                      </Space>
                    </Col>
                    <Col span={12}>
                      <Space direction="vertical" style={{ width: '100%' }}>
                        <div>
                          <Text strong>效果评分: </Text>
                          <Rate disabled value={selectedStrategy.effectiveness_score} allowHalf />
                        </div>
                        <div>
                          <Text strong>使用次数: </Text>
                          <Badge count={selectedStrategy.usage_count} />
                        </div>
                        <div>
                          <Text strong>成功率: </Text>
                          <Progress 
                            percent={selectedStrategy.success_rate} 
                            size="small"
                            style={{ width: 100 }}
                          />
                        </div>
                      </Space>
                    </Col>
                  </Row>
                </Card>
              </Col>

              <Col span={24}>
                <Card size="small" title="策略描述">
                  <Text>{selectedStrategy.description}</Text>
                </Card>
              </Col>

              <Col span={24}>
                <Card size="small" title="实施要求">
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <div>
                      <Text strong>持续时间: </Text>
                      <Text>{selectedStrategy.duration_range}</Text>
                    </div>
                    <div>
                      <Text strong>资源要求: </Text>
                      <div style={{ marginTop: 4 }}>
                        {selectedStrategy.resource_requirements.map(resource => (
                          <Tag key={resource}>{resource}</Tag>
                        ))}
                      </div>
                    </div>
                  </Space>
                </Card>
              </Col>
            </Row>
          </div>
        ) : (
          // 新建策略表单
          <Form
            form={strategyForm}
            layout="vertical"
            onFinish={handleCreateStrategy}
          >
            <Row gutter={16}>
              <Col span={12}>
                <Form.Item
                  name="name"
                  label="策略名称"
                  rules={[{ required: true, message: '请输入策略名称' }]}
                >
                  <Input placeholder="输入策略名称" />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item
                  name="type"
                  label="策略类型"
                  rules={[{ required: true, message: '请选择策略类型' }]}
                >
                  <Select placeholder="选择策略类型">
                    <Option value="supportive">支持性</Option>
                    <Option value="corrective">纠正性</Option>
                    <Option value="crisis">危机干预</Option>
                  </Select>
                </Form.Item>
              </Col>
            </Row>

            <Form.Item
              name="description"
              label="策略描述"
              rules={[{ required: true, message: '请输入策略描述' }]}
            >
              <TextArea placeholder="详细描述策略的目的和方法..." rows={3} />
            </Form.Item>

            <Row gutter={16}>
              <Col span={12}>
                <Form.Item
                  name="target_risk_factors"
                  label="目标风险因子"
                >
                  <Select mode="multiple" placeholder="选择目标风险因子">
                    <Option value="depression_indicators">抑郁指标</Option>
                    <Option value="anxiety_indicators">焦虑指标</Option>
                    <Option value="suicidal_ideation">自杀倾向</Option>
                    <Option value="social_isolation">社交孤立</Option>
                  </Select>
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item
                  name="duration_range"
                  label="预计持续时间"
                  rules={[{ required: true, message: '请输入持续时间' }]}
                >
                  <Input placeholder="如：2-4周" />
                </Form.Item>
              </Col>
            </Row>

            <Form.Item
              name="resource_requirements"
              label="资源要求"
            >
              <Select mode="tags" placeholder="输入所需资源...">
                <Option value="专业心理咨询师">专业心理咨询师</Option>
                <Option value="24小时热线">24小时热线</Option>
                <Option value="训练材料">训练材料</Option>
                <Option value="定期跟进">定期跟进</Option>
              </Select>
            </Form.Item>

            <Form.Item>
              <Space>
                <Button type="primary" htmlType="submit" loading={loading}>
                  创建策略
                </Button>
                <Button onClick={() => setStrategyModalVisible(false)}>
                  取消
                </Button>
              </Space>
            </Form.Item>
          </Form>
        )}
      </Modal>

      {/* 干预计划创建/详情模态框 */}
      <Modal
        title={selectedPlan ? "干预计划详情" : "创建干预计划"}
        open={planModalVisible}
        onCancel={() => {
          setPlanModalVisible(false);
          setSelectedPlan(null);
          planForm.resetFields();
        }}
        footer={null}
        width={800}
      >
        {selectedPlan ? (
          // 显示计划详情
          <div>
            <Row gutter={[16, 16]}>
              <Col span={12}>
                <Card size="small" title="基本信息">
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <div>
                      <Text strong>用户ID: </Text>
                      <Text>{selectedPlan.user_id}</Text>
                    </div>
                    <div>
                      <Text strong>干预类型: </Text>
                      <Tag color={strategyTypeColor(selectedPlan.intervention_type)}>
                        {selectedPlan.intervention_type.toUpperCase()}
                      </Tag>
                    </div>
                    <div>
                      <Text strong>紧急程度: </Text>
                      <Tag color={urgencyColor(selectedPlan.urgency_level)}>
                        {selectedPlan.urgency_level.toUpperCase()}
                      </Tag>
                    </div>
                    <div>
                      <Text strong>进度: </Text>
                      <Progress percent={selectedPlan.progress} size="small" />
                    </div>
                  </Space>
                </Card>
              </Col>
              <Col span={12}>
                <Card size="small" title="执行状态">
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <div>
                      <Text strong>状态: </Text>
                      <Badge status={statusColor(selectedPlan.status)} text={selectedPlan.status.toUpperCase()} />
                    </div>
                    <div>
                      <Text strong>监控频率: </Text>
                      <Text>{selectedPlan.monitoring_frequency}</Text>
                    </div>
                    <div>
                      <Text strong>创建时间: </Text>
                      <Text>{new Date(selectedPlan.created_at).toLocaleString()}</Text>
                    </div>
                  </Space>
                </Card>
              </Col>
            </Row>

            <Divider />

            <Card size="small" title="主要策略" style={{ marginBottom: 16 }}>
              <Space direction="vertical" style={{ width: '100%' }}>
                <Text strong>{selectedPlan.primary_strategy.name}</Text>
                <Text type="secondary">{selectedPlan.primary_strategy.description}</Text>
                <Space>
                  <Text>效果评分: </Text>
                  <Rate disabled value={selectedPlan.primary_strategy.effectiveness_score} allowHalf size="small" />
                  <Text>成功率: {selectedPlan.primary_strategy.success_rate}%</Text>
                </Space>
              </Space>
            </Card>

            <Row gutter={16}>
              <Col span={12}>
                <Card size="small" title="目标风险因子">
                  {selectedPlan.target_risk_factors.map(factor => (
                    <Tag key={factor} style={{ marginBottom: 4 }}>{factor}</Tag>
                  ))}
                </Card>
              </Col>
              <Col span={12}>
                <Card size="small" title="成功指标">
                  <Timeline size="small">
                    {selectedPlan.success_metrics.map((metric, index) => (
                      <Timeline.Item key={index}>
                        {metric}
                      </Timeline.Item>
                    ))}
                  </Timeline>
                </Card>
              </Col>
            </Row>
          </div>
        ) : (
          // 新建计划表单
          <Form
            form={planForm}
            layout="vertical"
            onFinish={handleCreatePlan}
          >
            <Row gutter={16}>
              <Col span={12}>
                <Form.Item
                  name="user_id"
                  label="目标用户"
                  rules={[{ required: true, message: '请输入用户ID' }]}
                >
                  <Input placeholder="输入用户ID" />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item
                  name="intervention_type"
                  label="干预类型"
                  rules={[{ required: true, message: '请选择干预类型' }]}
                >
                  <Select placeholder="选择干预类型">
                    <Option value="supportive">支持性</Option>
                    <Option value="corrective">纠正性</Option>
                    <Option value="crisis">危机干预</Option>
                  </Select>
                </Form.Item>
              </Col>
            </Row>

            <Row gutter={16}>
              <Col span={12}>
                <Form.Item
                  name="urgency_level"
                  label="紧急程度"
                  rules={[{ required: true, message: '请选择紧急程度' }]}
                >
                  <Select placeholder="选择紧急程度">
                    <Option value="low">低</Option>
                    <Option value="medium">中</Option>
                    <Option value="high">高</Option>
                    <Option value="critical">严重</Option>
                  </Select>
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item
                  name="primary_strategy_id"
                  label="主要策略"
                  rules={[{ required: true, message: '请选择主要策略' }]}
                >
                  <Select placeholder="选择主要策略">
                    {strategies.map(strategy => (
                      <Option key={strategy.strategy_id} value={strategy.strategy_id}>
                        {strategy.name}
                      </Option>
                    ))}
                  </Select>
                </Form.Item>
              </Col>
            </Row>

            <Form.Item
              name="target_risk_factors"
              label="目标风险因子"
            >
              <Select mode="multiple" placeholder="选择目标风险因子">
                <Option value="depression_indicators">抑郁指标</Option>
                <Option value="anxiety_indicators">焦虑指标</Option>
                <Option value="suicidal_ideation">自杀倾向</Option>
                <Option value="social_isolation">社交孤立</Option>
              </Select>
            </Form.Item>

            <Form.Item
              name="monitoring_frequency"
              label="监控频率"
              rules={[{ required: true, message: '请输入监控频率' }]}
            >
              <Select placeholder="选择监控频率">
                <Option value="每小时">每小时</Option>
                <Option value="每天">每天</Option>
                <Option value="每周">每周</Option>
                <Option value="每月">每月</Option>
              </Select>
            </Form.Item>

            <Form.Item>
              <Space>
                <Button type="primary" htmlType="submit" loading={loading}>
                  创建计划
                </Button>
                <Button onClick={() => setPlanModalVisible(false)}>
                  取消
                </Button>
              </Space>
            </Form.Item>
          </Form>
        )}
      </Modal>
    </div>
  );
};

export default InterventionStrategyManagementPage;