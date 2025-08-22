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
  Switch,
  Tooltip,
  Statistic,
  Progress,
  message,
  Tabs,
  Slider,
  InputNumber,
  Divider,
  Radio,
  Descriptions,
  Badge,
  Timeline,
} from 'antd';
import {
  ThunderboltOutlined,
  SettingOutlined,
  PlusOutlined,
  RobotOutlined,
  ExperimentOutlined,
  TrophyOutlined,
  LineChartOutlined,
  BarChartOutlined,
  PieChartOutlined,
  FundOutlined,
  BulbOutlined,
  AimOutlined,
  FireOutlined,
  StarOutlined,
  CrownOutlined,
  EditOutlined,
  DeleteOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  EyeOutlined,
  InfoCircleOutlined,
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';

const { Title, Text, Paragraph } = Typography;
const { Option } = Select;
const { TabPane } = Tabs;
const { TextArea } = Input;

interface AdvancedAlgorithm {
  id: string;
  name: string;
  algorithm_type: 'multi_armed_bandit' | 'bayesian_optimization' | 'reinforcement_learning' | 'causal_inference' | 'contextual_bandit';
  description: string;
  status: 'draft' | 'active' | 'paused' | 'completed';
  experiment_ids: string[];
  parameters: {
    learning_rate: number;
    exploration_rate: number;
    confidence_threshold: number;
    update_frequency: number;
  };
  performance_metrics: {
    reward_rate: number;
    regret_bound: number;
    convergence_speed: number;
    exploration_efficiency: number;
  };
  results: {
    total_iterations: number;
    best_variant: string;
    confidence_score: number;
    cumulative_reward: number;
  };
  created_at: string;
  updated_at: string;
  created_by: string;
}

interface BanditArm {
  id: string;
  name: string;
  pull_count: number;
  reward_sum: number;
  reward_rate: number;
  confidence_interval: {
    lower: number;
    upper: number;
  };
  last_updated: string;
}

interface AlgorithmComparison {
  algorithm: string;
  performance_score: number;
  convergence_time: number;
  exploration_efficiency: number;
  recommendation: string;
  pros: string[];
  cons: string[];
}

const AdvancedAlgorithmsPage: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [selectedTab, setSelectedTab] = useState<string>('algorithms');
  const [modalVisible, setModalVisible] = useState(false);
  const [banditModalVisible, setBanditModalVisible] = useState(false);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<string>('');
  const [form] = Form.useForm();

  // 模拟高级算法数据
  const algorithms: AdvancedAlgorithm[] = [
    {
      id: 'algo_001',
      name: 'UCB1多臂老虎机优化',
      algorithm_type: 'multi_armed_bandit',
      description: '基于上置信界的多臂老虎机算法，自动优化实验变体选择',
      status: 'active',
      experiment_ids: ['exp_001', 'exp_002'],
      parameters: {
        learning_rate: 0.1,
        exploration_rate: 0.2,
        confidence_threshold: 0.95,
        update_frequency: 60, // minutes
      },
      performance_metrics: {
        reward_rate: 0.847,
        regret_bound: 0.032,
        convergence_speed: 0.765,
        exploration_efficiency: 0.892,
      },
      results: {
        total_iterations: 15420,
        best_variant: 'variant_b',
        confidence_score: 0.94,
        cumulative_reward: 2156.7,
      },
      created_at: '2024-01-15',
      updated_at: '2024-01-22',
      created_by: 'ML Team',
    },
    {
      id: 'algo_002',
      name: 'Thompson采样算法',
      algorithm_type: 'multi_armed_bandit',
      description: 'Thompson采样算法，基于贝叶斯推断的概率性选择策略',
      status: 'active',
      experiment_ids: ['exp_003'],
      parameters: {
        learning_rate: 0.05,
        exploration_rate: 0.15,
        confidence_threshold: 0.90,
        update_frequency: 30,
      },
      performance_metrics: {
        reward_rate: 0.823,
        regret_bound: 0.028,
        convergence_speed: 0.712,
        exploration_efficiency: 0.856,
      },
      results: {
        total_iterations: 8950,
        best_variant: 'variant_a',
        confidence_score: 0.91,
        cumulative_reward: 1678.3,
      },
      created_at: '2024-01-18',
      updated_at: '2024-01-22',
      created_by: 'Data Science Team',
    },
    {
      id: 'algo_003',
      name: '贝叶斯优化引擎',
      algorithm_type: 'bayesian_optimization',
      description: '使用高斯过程的贝叶斯优化，智能探索参数空间',
      status: 'completed',
      experiment_ids: ['exp_004'],
      parameters: {
        learning_rate: 0.08,
        exploration_rate: 0.25,
        confidence_threshold: 0.99,
        update_frequency: 120,
      },
      performance_metrics: {
        reward_rate: 0.901,
        regret_bound: 0.019,
        convergence_speed: 0.834,
        exploration_efficiency: 0.923,
      },
      results: {
        total_iterations: 3420,
        best_variant: 'variant_c',
        confidence_score: 0.98,
        cumulative_reward: 3245.6,
      },
      created_at: '2024-01-10',
      updated_at: '2024-01-20',
      created_by: 'AI Research Team',
    },
    {
      id: 'algo_004',
      name: '上下文感知bandit',
      algorithm_type: 'contextual_bandit',
      description: '考虑用户上下文信息的多臂老虎机算法，个性化推荐',
      status: 'draft',
      experiment_ids: [],
      parameters: {
        learning_rate: 0.12,
        exploration_rate: 0.18,
        confidence_threshold: 0.92,
        update_frequency: 45,
      },
      performance_metrics: {
        reward_rate: 0.0,
        regret_bound: 0.0,
        convergence_speed: 0.0,
        exploration_efficiency: 0.0,
      },
      results: {
        total_iterations: 0,
        best_variant: '',
        confidence_score: 0.0,
        cumulative_reward: 0.0,
      },
      created_at: '2024-01-22',
      updated_at: '2024-01-22',
      created_by: 'ML Team',
    },
  ];

  const banditArms: BanditArm[] = [
    {
      id: 'arm_001',
      name: '原版首页 (对照组)',
      pull_count: 7820,
      reward_sum: 1134,
      reward_rate: 0.145,
      confidence_interval: { lower: 0.137, upper: 0.153 },
      last_updated: '2024-01-22 14:30:25',
    },
    {
      id: 'arm_002',
      name: '新版首页 (实验组A)',
      pull_count: 8600,
      reward_sum: 1462,
      reward_rate: 0.170,
      confidence_interval: { lower: 0.161, upper: 0.179 },
      last_updated: '2024-01-22 14:30:25',
    },
    {
      id: 'arm_003',
      name: '极简版首页 (实验组B)',
      pull_count: 2450,
      reward_sum: 392,
      reward_rate: 0.160,
      confidence_interval: { lower: 0.145, upper: 0.175 },
      last_updated: '2024-01-22 14:30:25',
    },
  ];

  const algorithmComparisons: AlgorithmComparison[] = [
    {
      algorithm: 'UCB1多臂老虎机',
      performance_score: 0.847,
      convergence_time: 2.3,
      exploration_efficiency: 0.892,
      recommendation: '推荐用于快速决策场景',
      pros: ['收敛速度快', '理论保证强', '实现简单'],
      cons: ['对噪声敏感', '需要调参'],
    },
    {
      algorithm: 'Thompson采样',
      performance_score: 0.823,
      convergence_time: 3.1,
      exploration_efficiency: 0.856,
      recommendation: '适合不确定性较大的场景',
      pros: ['概率性策略', '适应性强', '理论最优'],
      cons: ['计算复杂', '参数敏感'],
    },
    {
      algorithm: '贝叶斯优化',
      performance_score: 0.901,
      convergence_time: 1.8,
      exploration_efficiency: 0.923,
      recommendation: '最适合复杂参数优化',
      pros: ['全局最优', '样本高效', '自适应'],
      cons: ['计算开销大', '需要先验知识'],
    },
    {
      algorithm: '上下文bandit',
      performance_score: 0.0,
      convergence_time: 0.0,
      exploration_efficiency: 0.0,
      recommendation: '开发中，待验证',
      pros: ['个性化强', '上下文感知', '精确定向'],
      cons: ['实现复杂', '数据需求大'],
    },
  ];

  const getAlgorithmTypeText = (type: string) => {
    switch (type) {
      case 'multi_armed_bandit': return '多臂老虎机';
      case 'bayesian_optimization': return '贝叶斯优化';
      case 'reinforcement_learning': return '强化学习';
      case 'causal_inference': return '因果推断';
      case 'contextual_bandit': return '上下文bandit';
      default: return type;
    }
  };

  const getAlgorithmTypeColor = (type: string) => {
    switch (type) {
      case 'multi_armed_bandit': return 'orange';
      case 'bayesian_optimization': return 'blue';
      case 'reinforcement_learning': return 'green';
      case 'causal_inference': return 'purple';
      case 'contextual_bandit': return 'gold';
      default: return 'default';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'draft': return 'default';
      case 'active': return 'processing';
      case 'paused': return 'warning';
      case 'completed': return 'success';
      default: return 'default';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'draft': return '草稿';
      case 'active': return '运行中';
      case 'paused': return '已暂停';
      case 'completed': return '已完成';
      default: return status;
    }
  };

  const getPerformanceColor = (score: number): string => {
    if (score >= 0.8) return '#52c41a';
    if (score >= 0.6) return '#faad14';
    if (score >= 0.4) return '#ff7a45';
    return '#ff4d4f';
  };

  const algorithmsColumns: ColumnsType<AdvancedAlgorithm> = [
    {
      title: '算法名称',
      key: 'name',
      width: 200,
      render: (_, record: AdvancedAlgorithm) => (
        <div>
          <Text strong>{record.name}</Text>
          <br />
          <Tag color={getAlgorithmTypeColor(record.algorithm_type)}>
            {getAlgorithmTypeText(record.algorithm_type)}
          </Tag>
        </div>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 80,
      render: (status: string) => (
        <Badge status={getStatusColor(status)} text={getStatusText(status)} />
      ),
    },
    {
      title: '关联实验',
      dataIndex: 'experiment_ids',
      key: 'experiment_ids',
      width: 120,
      render: (ids: string[]) => (
        <div>
          {ids.length > 0 ? ids.map(id => (
            <Tag key={id} size="small" style={{ marginBottom: '2px' }}>
              {id}
            </Tag>
          )) : <Text type="secondary">暂无</Text>}
        </div>
      ),
    },
    {
      title: '奖励率',
      key: 'reward_rate',
      width: 120,
      render: (_, record: AdvancedAlgorithm) => (
        <div style={{ textAlign: 'center' }}>
          <Progress
            type="circle"
            size={60}
            percent={record.performance_metrics.reward_rate * 100}
            format={() => `${(record.performance_metrics.reward_rate * 100).toFixed(1)}%`}
            strokeColor={getPerformanceColor(record.performance_metrics.reward_rate)}
          />
        </div>
      ),
    },
    {
      title: '收敛效率',
      key: 'convergence_speed',
      width: 100,
      render: (_, record: AdvancedAlgorithm) => (
        <div>
          <div style={{ marginBottom: '4px' }}>
            <Text strong>{(record.performance_metrics.convergence_speed * 100).toFixed(1)}%</Text>
          </div>
          <Progress
            percent={record.performance_metrics.convergence_speed * 100}
            size="small"
            strokeColor={getPerformanceColor(record.performance_metrics.convergence_speed)}
            showInfo={false}
          />
        </div>
      ),
    },
    {
      title: '探索效率',
      key: 'exploration_efficiency',
      width: 100,
      render: (_, record: AdvancedAlgorithm) => (
        <div>
          <div style={{ marginBottom: '4px' }}>
            <Text strong>{(record.performance_metrics.exploration_efficiency * 100).toFixed(1)}%</Text>
          </div>
          <Progress
            percent={record.performance_metrics.exploration_efficiency * 100}
            size="small"
            strokeColor={getPerformanceColor(record.performance_metrics.exploration_efficiency)}
            showInfo={false}
          />
        </div>
      ),
    },
    {
      title: '累积奖励',
      key: 'cumulative_reward',
      width: 100,
      align: 'right',
      render: (_, record: AdvancedAlgorithm) => (
        <Text strong style={{ fontSize: '14px' }}>
          {record.results.cumulative_reward.toFixed(1)}
        </Text>
      ),
    },
    {
      title: '操作',
      key: 'actions',
      width: 120,
      render: (_, record: AdvancedAlgorithm) => (
        <Space size="small">
          <Tooltip title="查看详情">
            <Button 
              type="text" 
              size="small" 
              icon={<EyeOutlined />}
              onClick={() => {
                setSelectedAlgorithm(record.id);
                setBanditModalVisible(true);
              }}
            />
          </Tooltip>
          <Tooltip title="编辑">
            <Button type="text" size="small" icon={<EditOutlined />} />
          </Tooltip>
          <Tooltip title={record.status === 'active' ? '暂停' : '启动'}>
            <Button 
              type="text" 
              size="small" 
              icon={record.status === 'active' ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
              onClick={() => message.success(record.status === 'active' ? '算法已暂停' : '算法已启动')}
            />
          </Tooltip>
        </Space>
      ),
    },
  ];

  const banditArmsColumns: ColumnsType<BanditArm> = [
    {
      title: '变体名称',
      dataIndex: 'name',
      key: 'name',
      width: 200,
    },
    {
      title: '拉取次数',
      dataIndex: 'pull_count',
      key: 'pull_count',
      width: 100,
      render: (count: number) => <Text strong>{count.toLocaleString()}</Text>,
    },
    {
      title: '奖励总和',
      dataIndex: 'reward_sum',
      key: 'reward_sum',
      width: 100,
      render: (sum: number) => <Text>{sum.toLocaleString()}</Text>,
    },
    {
      title: '奖励率',
      dataIndex: 'reward_rate',
      key: 'reward_rate',
      width: 100,
      render: (rate: number) => (
        <Text strong style={{ fontSize: '16px', color: '#1890ff' }}>
          {(rate * 100).toFixed(1)}%
        </Text>
      ),
    },
    {
      title: '置信区间',
      key: 'confidence_interval',
      width: 120,
      render: (_, record: BanditArm) => (
        <Text style={{ fontSize: '12px' }}>
          [{(record.confidence_interval.lower * 100).toFixed(1)}%, {(record.confidence_interval.upper * 100).toFixed(1)}%]
        </Text>
      ),
    },
    {
      title: '最后更新',
      dataIndex: 'last_updated',
      key: 'last_updated',
      width: 130,
      render: (time: string) => (
        <Text style={{ fontSize: '12px' }}>{time}</Text>
      ),
    },
  ];

  const comparisonColumns: ColumnsType<AlgorithmComparison> = [
    {
      title: '算法',
      dataIndex: 'algorithm',
      key: 'algorithm',
      width: 150,
      render: (name: string) => <Text strong>{name}</Text>,
    },
    {
      title: '性能得分',
      dataIndex: 'performance_score',
      key: 'performance_score',
      width: 100,
      render: (score: number) => (
        <div style={{ textAlign: 'center' }}>
          <Progress
            type="circle"
            size={50}
            percent={score * 100}
            format={() => `${(score * 100).toFixed(0)}`}
            strokeColor={getPerformanceColor(score)}
          />
        </div>
      ),
    },
    {
      title: '收敛时间',
      dataIndex: 'convergence_time',
      key: 'convergence_time',
      width: 100,
      render: (time: number) => time > 0 ? <Text>{time}小时</Text> : <Text type="secondary">-</Text>,
    },
    {
      title: '探索效率',
      dataIndex: 'exploration_efficiency',
      key: 'exploration_efficiency',
      width: 100,
      render: (efficiency: number) => (
        efficiency > 0 ? <Text strong>{(efficiency * 100).toFixed(1)}%</Text> : <Text type="secondary">-</Text>
      ),
    },
    {
      title: '推荐度',
      dataIndex: 'recommendation',
      key: 'recommendation',
      width: 150,
      render: (rec: string) => <Text>{rec}</Text>,
    },
  ];

  const handleCreateAlgorithm = () => {
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
              <ThunderboltOutlined /> 高级算法
            </Title>
            <Text type="secondary">智能优化算法和自适应实验策略</Text>
          </div>
          <Space>
            <Button icon={<BulbOutlined />}>
              算法建议
            </Button>
            <Button type="primary" icon={<PlusOutlined />} onClick={handleCreateAlgorithm}>
              创建算法
            </Button>
          </Space>
        </div>
      </div>

      {/* 算法性能概览 */}
      <Row gutter={16} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="活跃算法"
              value={algorithms.filter(a => a.status === 'active').length}
              prefix={<RobotOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="平均奖励率"
              value={84.7}
              precision={1}
              suffix="%"
              prefix={<TrophyOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="累积奖励"
              value={7080.6}
              precision={1}
              prefix={<CrownOutlined />}
              valueStyle={{ color: '#faad14' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="平均收敛时间"
              value={2.4}
              precision={1}
              suffix="小时"
              prefix={<FireOutlined />}
            />
          </Card>
        </Col>
      </Row>

      {/* 主要内容 */}
      <Tabs activeKey={selectedTab} onChange={setSelectedTab}>
        <TabPane tab="算法管理" key="algorithms">
          <Card>
            <Table
              columns={algorithmsColumns}
              dataSource={algorithms}
              rowKey="id"
              loading={loading}
              pagination={false}
              expandable={{
                expandedRowRender: (record) => (
                  <div style={{ margin: 0 }}>
                    <Row gutter={16}>
                      <Col span={12}>
                        <Title level={5}>算法参数</Title>
                        <Descriptions size="small" column={2}>
                          <Descriptions.Item label="学习率">
                            {record.parameters.learning_rate}
                          </Descriptions.Item>
                          <Descriptions.Item label="探索率">
                            {record.parameters.exploration_rate}
                          </Descriptions.Item>
                          <Descriptions.Item label="置信阈值">
                            {record.parameters.confidence_threshold}
                          </Descriptions.Item>
                          <Descriptions.Item label="更新频率">
                            {record.parameters.update_frequency}分钟
                          </Descriptions.Item>
                        </Descriptions>
                      </Col>
                      <Col span={12}>
                        <Title level={5}>性能指标</Title>
                        <div>
                          <div style={{ marginBottom: '8px' }}>
                            <Text>后悔界限: </Text>
                            <Text strong style={{ color: '#52c41a' }}>
                              {record.performance_metrics.regret_bound.toFixed(3)}
                            </Text>
                          </div>
                          <div style={{ marginBottom: '8px' }}>
                            <Text>迭代总数: </Text>
                            <Text strong>{record.results.total_iterations.toLocaleString()}</Text>
                          </div>
                          <div style={{ marginBottom: '8px' }}>
                            <Text>最佳变体: </Text>
                            <Tag color="green">{record.results.best_variant || '尚未确定'}</Tag>
                          </div>
                          <div>
                            <Text>置信分数: </Text>
                            <Text strong style={{ color: '#1890ff' }}>
                              {(record.results.confidence_score * 100).toFixed(1)}%
                            </Text>
                          </div>
                        </div>
                      </Col>
                    </Row>
                    <Divider style={{ margin: '16px 0' }} />
                    <div>
                      <Text>{record.description}</Text>
                    </div>
                  </div>
                ),
              }}
            />
          </Card>
        </TabPane>

        <TabPane tab="多臂老虎机" key="bandit">
          <Row gutter={16}>
            <Col span={16}>
              <Card title="变体臂状态">
                <Table
                  columns={banditArmsColumns}
                  dataSource={banditArms}
                  rowKey="id"
                  loading={loading}
                  pagination={false}
                  size="small"
                />
              </Card>
            </Col>
            <Col span={8}>
              <Card title="算法状态" size="small" style={{ marginBottom: '16px' }}>
                <div style={{ textAlign: 'center', marginBottom: '16px' }}>
                  <div style={{ fontSize: '32px', fontWeight: 'bold', color: '#1890ff' }}>
                    UCB1
                  </div>
                  <Text type="secondary">当前算法</Text>
                </div>
                <div style={{ marginBottom: '12px' }}>
                  <Text>总拉取次数: </Text>
                  <Text strong>{banditArms.reduce((sum, arm) => sum + arm.pull_count, 0).toLocaleString()}</Text>
                </div>
                <div style={{ marginBottom: '12px' }}>
                  <Text>最佳臂: </Text>
                  <Tag color="gold">新版首页</Tag>
                </div>
                <div>
                  <Text>置信度: </Text>
                  <Text strong style={{ color: '#52c41a' }}>94%</Text>
                </div>
              </Card>

              <Card title="奖励率趋势" size="small">
                <div style={{ textAlign: 'center' }}>
                  <LineChartOutlined style={{ fontSize: '48px', color: '#1890ff', marginBottom: '16px' }} />
                  <div>
                    <Text strong style={{ fontSize: '18px', color: '#52c41a' }}>+12.3%</Text>
                  </div>
                  <div>
                    <Text type="secondary">相比随机分配提升</Text>
                  </div>
                </div>
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab="算法对比" key="comparison">
          <Card>
            <Table
              columns={comparisonColumns}
              dataSource={algorithmComparisons}
              rowKey="algorithm"
              loading={loading}
              pagination={false}
              expandable={{
                expandedRowRender: (record) => (
                  <div style={{ margin: 0 }}>
                    <Row gutter={16}>
                      <Col span={12}>
                        <div>
                          <Text strong style={{ color: '#52c41a' }}>优点：</Text>
                          <ul style={{ marginTop: '8px', marginBottom: '0' }}>
                            {record.pros.map((pro, index) => (
                              <li key={index}>{pro}</li>
                            ))}
                          </ul>
                        </div>
                      </Col>
                      <Col span={12}>
                        <div>
                          <Text strong style={{ color: '#ff4d4f' }}>缺点：</Text>
                          <ul style={{ marginTop: '8px', marginBottom: '0' }}>
                            {record.cons.map((con, index) => (
                              <li key={index}>{con}</li>
                            ))}
                          </ul>
                        </div>
                      </Col>
                    </Row>
                  </div>
                ),
              }}
            />
          </Card>
        </TabPane>

        <TabPane tab="算法历史" key="history">
          <Card>
            <Timeline>
              <Timeline.Item color="green" dot={<StarOutlined />}>
                <div>
                  <Text strong>贝叶斯优化算法完成</Text>
                  <Tag color="green" size="small" style={{ marginLeft: '8px' }}>已完成</Tag>
                  <br />
                  <Text type="secondary">2024-01-20 18:30 - 获得最高奖励率90.1%，显著优于其他算法</Text>
                </div>
              </Timeline.Item>
              <Timeline.Item color="blue" dot={<RobotOutlined />}>
                <div>
                  <Text strong>UCB1多臂老虎机启动</Text>
                  <Tag color="blue" size="small" style={{ marginLeft: '8px' }}>运行中</Tag>
                  <br />
                  <Text type="secondary">2024-01-15 10:00 - 开始自动优化实验变体选择，当前表现良好</Text>
                </div>
              </Timeline.Item>
              <Timeline.Item color="orange" dot={<ThunderboltOutlined />}>
                <div>
                  <Text strong>Thompson采样算法部署</Text>
                  <Tag color="orange" size="small" style={{ marginLeft: '8px' }}>运行中</Tag>
                  <br />
                  <Text type="secondary">2024-01-18 14:20 - 基于贝叶斯推断的新算法投入使用</Text>
                </div>
              </Timeline.Item>
              <Timeline.Item dot={<BulbOutlined />}>
                <div>
                  <Text strong>上下文bandit算法设计</Text>
                  <Tag color="default" size="small" style={{ marginLeft: '8px' }}>草稿</Tag>
                  <br />
                  <Text type="secondary">2024-01-22 09:15 - 新的个性化推荐算法进入设计阶段</Text>
                </div>
              </Timeline.Item>
            </Timeline>
          </Card>
        </TabPane>
      </Tabs>

      {/* 创建算法模态框 */}
      <Modal
        title="创建高级算法"
        open={modalVisible}
        onCancel={() => setModalVisible(false)}
        footer={[
          <Button key="cancel" onClick={() => setModalVisible(false)}>
            取消
          </Button>,
          <Button key="submit" type="primary" onClick={() => {
            message.success('算法已创建');
            setModalVisible(false);
          }}>
            创建算法
          </Button>,
        ]}
        width={700}
      >
        <Form form={form} layout="vertical">
          <Form.Item label="算法名称" name="name" required>
            <Input placeholder="输入算法名称" />
          </Form.Item>
          
          <Form.Item label="算法类型" name="algorithm_type" required>
            <Select placeholder="选择算法类型">
              <Option value="multi_armed_bandit">多臂老虎机</Option>
              <Option value="bayesian_optimization">贝叶斯优化</Option>
              <Option value="reinforcement_learning">强化学习</Option>
              <Option value="causal_inference">因果推断</Option>
              <Option value="contextual_bandit">上下文bandit</Option>
            </Select>
          </Form.Item>

          <Form.Item label="描述" name="description">
            <TextArea rows={3} placeholder="描述算法的工作原理和应用场景" />
          </Form.Item>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item label="学习率" name="learning_rate">
                <Slider
                  min={0.01}
                  max={0.5}
                  step={0.01}
                  defaultValue={0.1}
                  marks={{
                    0.01: '0.01',
                    0.1: '0.1',
                    0.2: '0.2',
                    0.5: '0.5',
                  }}
                />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="探索率" name="exploration_rate">
                <Slider
                  min={0.1}
                  max={0.5}
                  step={0.01}
                  defaultValue={0.2}
                  marks={{
                    0.1: '0.1',
                    0.2: '0.2',
                    0.3: '0.3',
                    0.5: '0.5',
                  }}
                />
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item label="置信阈值" name="confidence_threshold">
                <InputNumber
                  min={0.8}
                  max={0.99}
                  step={0.01}
                  defaultValue={0.95}
                  formatter={value => `${(Number(value) * 100).toFixed(0)}%`}
                  parser={value => Number(value!.replace('%', '')) / 100}
                  style={{ width: '100%' }}
                />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="更新频率" name="update_frequency">
                <InputNumber
                  min={10}
                  max={240}
                  step={10}
                  defaultValue={60}
                  formatter={value => `${value}分钟`}
                  parser={value => Number(value!.replace('分钟', ''))}
                  style={{ width: '100%' }}
                />
              </Form.Item>
            </Col>
          </Row>

          <Form.Item label="关联实验" name="experiment_ids">
            <Select
              mode="multiple"
              placeholder="选择要应用算法的实验"
              style={{ width: '100%' }}
            >
              <Option value="exp_001">首页改版A/B测试</Option>
              <Option value="exp_002">结算页面优化</Option>
              <Option value="exp_003">推荐算法测试</Option>
              <Option value="exp_004">定价策略实验</Option>
            </Select>
          </Form.Item>

          <Form.Item label="立即启动" name="auto_start">
            <Switch />
          </Form.Item>
        </Form>
      </Modal>

      {/* 多臂老虎机详情模态框 */}
      <Modal
        title="多臂老虎机详细状态"
        open={banditModalVisible}
        onCancel={() => setBanditModalVisible(false)}
        footer={[
          <Button key="close" onClick={() => setBanditModalVisible(false)}>
            关闭
          </Button>,
        ]}
        width={800}
      >
        <Row gutter={16} style={{ marginBottom: '16px' }}>
          <Col span={8}>
            <Card size="small" style={{ textAlign: 'center' }}>
              <Statistic
                title="当前最佳臂"
                value="新版首页"
                prefix={<TrophyOutlined />}
                valueStyle={{ color: '#52c41a' }}
              />
            </Card>
          </Col>
          <Col span={8}>
            <Card size="small" style={{ textAlign: 'center' }}>
              <Statistic
                title="置信度"
                value={94}
                suffix="%"
                prefix={<AimOutlined />}
                valueStyle={{ color: '#1890ff' }}
              />
            </Card>
          </Col>
          <Col span={8}>
            <Card size="small" style={{ textAlign: 'center' }}>
              <Statistic
                title="后悔值"
                value={0.032}
                precision={3}
                prefix={<InfoCircleOutlined />}
                valueStyle={{ color: '#faad14' }}
              />
            </Card>
          </Col>
        </Row>
        
        <Table
          columns={banditArmsColumns}
          dataSource={banditArms}
          rowKey="id"
          size="small"
          pagination={false}
        />
        
        <div style={{ marginTop: '16px' }}>
          <Alert
            message="算法建议"
            description="基于当前数据分析，建议继续运行UCB1算法。新版首页表现最佳，建议逐步增加其流量分配至60%以获得最大化收益。"
            variant="default"
            showIcon
          />
        </div>
      </Modal>
    </div>
  );
};

export default AdvancedAlgorithmsPage;