import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Row, 
  Col, 
  Table, 
  Button, 
  Modal, 
  Form, 
  Input, 
  Select, 
  Progress, 
  Statistic, 
  Tabs, 
  Timeline, 
  Badge,
  Space,
  Alert,
  Descriptions,
  Tag,
  Tooltip,
  Popconfirm,
  Switch,
  Slider,
  Upload,
  message,
  List,
  Avatar,
  Divider
} from 'antd';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip as RechartsTooltip, 
  Legend, 
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import { 
  ExperimentOutlined,
  RocketOutlined,
  StopOutlined,
  PlayCircleOutlined,
  BarChartOutlined,
  UploadOutlined,
  DownloadOutlined,
  ReloadOutlined,
  SettingOutlined,
  TrophyOutlined,
  BulbOutlined,
  FireOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined
} from '@ant-design/icons';

const { TabPane } = Tabs;
const { Option } = Select;
const { TextArea } = Input;

interface ABTest {
  id: string;
  name: string;
  description: string;
  model_a_id: string;
  model_a_name: string;
  model_b_id: string;
  model_b_name: string;
  traffic_split: number;
  status: 'pending' | 'running' | 'completed' | 'paused';
  start_time: string;
  end_time?: string;
  total_requests: number;
  model_a_requests: number;
  model_b_requests: number;
  model_a_performance: {
    accuracy: number;
    latency: number;
    error_rate: number;
  };
  model_b_performance: {
    accuracy: number;
    latency: number;
    error_rate: number;
  };
  statistical_significance: number;
  winner?: 'A' | 'B' | 'tie';
}

interface LearningJob {
  id: string;
  name: string;
  model_id: string;
  model_name: string;
  learning_type: 'incremental' | 'continual' | 'federated';
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  data_samples: number;
  start_time: string;
  estimated_completion: string;
  metrics: {
    current_accuracy: number;
    baseline_accuracy: number;
    improvement: number;
  };
}

interface FeedbackData {
  id: string;
  model_id: string;
  prediction: any;
  ground_truth: any;
  feedback_score: number;
  user_feedback: string;
  timestamp: string;
  processed: boolean;
}

const OnlineLearningPage: React.FC = () => {
  const [abTests, setAbTests] = useState<ABTest[]>([]);
  const [learningJobs, setLearningJobs] = useState<LearningJob[]>([]);
  const [feedbackData, setFeedbackData] = useState<FeedbackData[]>([]);
  const [loading, setLoading] = useState(false);
  const [testModalVisible, setTestModalVisible] = useState(false);
  const [jobModalVisible, setJobModalVisible] = useState(false);
  const [selectedTest, setSelectedTest] = useState<ABTest | null>(null);
  const [testForm] = Form.useForm();
  const [jobForm] = Form.useForm();

  const systemStats = {
    active_tests: abTests.filter(t => t.status === 'running').length,
    active_jobs: learningJobs.filter(j => j.status === 'running').length,
    total_feedback: feedbackData.length,
    avg_improvement: 12.5
  };

  useEffect(() => {
    fetchABTests();
    fetchLearningJobs();
    fetchFeedbackData();
  }, []);

  const fetchABTests = async () => {
    setLoading(true);
    try {
      const mockTests: ABTest[] = [
        {
          id: '1',
          name: '文本分类模型优化测试',
          description: '测试BERT vs RoBERTa在文本分类任务上的性能',
          model_a_id: 'model_001',
          model_a_name: 'BERT-base',
          model_b_id: 'model_002',
          model_b_name: 'RoBERTa-base',
          traffic_split: 50,
          status: 'running',
          start_time: '2024-01-20T10:00:00Z',
          total_requests: 12450,
          model_a_requests: 6225,
          model_b_requests: 6225,
          model_a_performance: {
            accuracy: 0.892,
            latency: 45.6,
            error_rate: 0.12
          },
          model_b_performance: {
            accuracy: 0.915,
            latency: 52.3,
            error_rate: 0.08
          },
          statistical_significance: 0.95,
          winner: 'B'
        },
        {
          id: '2',
          name: '情感分析延迟优化',
          description: '测试量化模型vs原始模型的延迟和准确性权衡',
          model_a_id: 'model_003',
          model_a_name: '原始模型',
          model_b_id: 'model_004',
          model_b_name: '量化模型',
          traffic_split: 30,
          status: 'completed',
          start_time: '2024-01-15T14:00:00Z',
          end_time: '2024-01-22T14:00:00Z',
          total_requests: 8960,
          model_a_requests: 6272,
          model_b_requests: 2688,
          model_a_performance: {
            accuracy: 0.876,
            latency: 68.2,
            error_rate: 0.15
          },
          model_b_performance: {
            accuracy: 0.851,
            latency: 23.1,
            error_rate: 0.18
          },
          statistical_significance: 0.92,
          winner: 'A'
        }
      ];
      setAbTests(mockTests);
    } catch (error) {
      message.error('加载A/B测试失败');
    } finally {
      setLoading(false);
    }
  };

  const fetchLearningJobs = async () => {
    try {
      const mockJobs: LearningJob[] = [
        {
          id: '1',
          name: '增量学习-客户反馈',
          model_id: 'model_001',
          model_name: 'BERT文本分类器',
          learning_type: 'incremental',
          status: 'running',
          progress: 65,
          data_samples: 1250,
          start_time: '2024-01-22T09:30:00Z',
          estimated_completion: '2024-01-22T15:45:00Z',
          metrics: {
            current_accuracy: 0.908,
            baseline_accuracy: 0.892,
            improvement: 1.6
          }
        },
        {
          id: '2',
          name: '持续学习-新数据适应',
          model_id: 'model_002',
          model_name: '情感分析模型',
          learning_type: 'continual',
          status: 'completed',
          progress: 100,
          data_samples: 3400,
          start_time: '2024-01-20T16:20:00Z',
          estimated_completion: '2024-01-21T12:15:00Z',
          metrics: {
            current_accuracy: 0.934,
            baseline_accuracy: 0.915,
            improvement: 1.9
          }
        },
        {
          id: '3',
          name: '联邦学习-多节点协同',
          model_id: 'model_005',
          model_name: '推荐系统模型',
          learning_type: 'federated',
          status: 'pending',
          progress: 0,
          data_samples: 0,
          start_time: '2024-01-23T08:00:00Z',
          estimated_completion: '2024-01-24T18:00:00Z',
          metrics: {
            current_accuracy: 0.0,
            baseline_accuracy: 0.823,
            improvement: 0.0
          }
        }
      ];
      setLearningJobs(mockJobs);
    } catch (error) {
      message.error('加载学习任务失败');
    }
  };

  const fetchFeedbackData = async () => {
    try {
      const mockFeedback: FeedbackData[] = [
        {
          id: '1',
          model_id: 'model_001',
          prediction: '正面情感',
          ground_truth: '负面情感',
          feedback_score: 2,
          user_feedback: '预测错误，实际是负面评价',
          timestamp: '2024-01-22T14:30:00Z',
          processed: false
        },
        {
          id: '2',
          model_id: 'model_001',
          prediction: '中性情感',
          ground_truth: '正面情感',
          feedback_score: 3,
          user_feedback: '基本正确，但情感强度判断不够准确',
          timestamp: '2024-01-22T13:15:00Z',
          processed: true
        }
      ];
      setFeedbackData(mockFeedback);
    } catch (error) {
      message.error('加载反馈数据失败');
    }
  };

  const handleCreateTest = async (values: any) => {
    try {
      const newTest: ABTest = {
        id: Date.now().toString(),
        name: values.name,
        description: values.description,
        model_a_id: values.model_a_id,
        model_a_name: values.model_a_name || 'Model A',
        model_b_id: values.model_b_id,
        model_b_name: values.model_b_name || 'Model B',
        traffic_split: values.traffic_split,
        status: 'pending',
        start_time: new Date().toISOString(),
        total_requests: 0,
        model_a_requests: 0,
        model_b_requests: 0,
        model_a_performance: {
          accuracy: 0,
          latency: 0,
          error_rate: 0
        },
        model_b_performance: {
          accuracy: 0,
          latency: 0,
          error_rate: 0
        },
        statistical_significance: 0
      };
      
      setAbTests([...abTests, newTest]);
      setTestModalVisible(false);
      testForm.resetFields();
      message.success('A/B测试已创建');
    } catch (error) {
      message.error('创建A/B测试失败');
    }
  };

  const handleCreateJob = async (values: any) => {
    try {
      const newJob: LearningJob = {
        id: Date.now().toString(),
        name: values.name,
        model_id: values.model_id,
        model_name: values.model_name || 'Unknown Model',
        learning_type: values.learning_type,
        status: 'pending',
        progress: 0,
        data_samples: 0,
        start_time: new Date().toISOString(),
        estimated_completion: new Date(Date.now() + 6 * 60 * 60 * 1000).toISOString(),
        metrics: {
          current_accuracy: 0,
          baseline_accuracy: 0,
          improvement: 0
        }
      };
      
      setLearningJobs([...learningJobs, newJob]);
      setJobModalVisible(false);
      jobForm.resetFields();
      message.success('学习任务已创建');
    } catch (error) {
      message.error('创建学习任务失败');
    }
  };

  const handleTestAction = async (testId: string, action: 'start' | 'pause' | 'stop') => {
    try {
      const newStatus = action === 'start' ? 'running' : action === 'pause' ? 'paused' : 'completed';
      setAbTests(prev => prev.map(test => 
        test.id === testId ? { ...test, status: newStatus as ABTest['status'] } : test
      ));
      message.success(`A/B测试已${action === 'start' ? '启动' : action === 'pause' ? '暂停' : '停止'}`);
    } catch (error) {
      message.error('操作失败');
    }
  };

  const getStatusColor = (status: string) => {
    const colors = {
      pending: 'orange',
      running: 'green',
      completed: 'blue',
      paused: 'default',
      failed: 'red'
    };
    return colors[status as keyof typeof colors] || 'default';
  };

  const getStatusText = (status: string) => {
    const texts = {
      pending: '待启动',
      running: '运行中',
      completed: '已完成',
      paused: '已暂停',
      failed: '失败'
    };
    return texts[status as keyof typeof texts] || status;
  };

  const getLearningTypeText = (type: string) => {
    const texts = {
      incremental: '增量学习',
      continual: '持续学习',
      federated: '联邦学习'
    };
    return texts[type as keyof typeof texts] || type;
  };

  const testColumns = [
    {
      title: '测试名称',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: ABTest) => (
        <Space>
          <Avatar style={{ backgroundColor: '#1890ff' }}>
            {text.charAt(0)}
          </Avatar>
          <div>
            <div>{text}</div>
            <div style={{ fontSize: '12px', color: '#999' }}>{record.description}</div>
          </div>
        </Space>
      )
    },
    {
      title: '模型对比',
      key: 'models',
      render: (_, record: ABTest) => (
        <div>
          <div>A: {record.model_a_name}</div>
          <div>B: {record.model_b_name}</div>
        </div>
      )
    },
    {
      title: '流量分配',
      dataIndex: 'traffic_split',
      key: 'traffic_split',
      render: (split: number) => (
        <div>
          <div>A: {100 - split}%</div>
          <div>B: {split}%</div>
        </div>
      )
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Badge status={getStatusColor(status) as any} text={getStatusText(status)} />
      )
    },
    {
      title: '总请求数',
      dataIndex: 'total_requests',
      key: 'total_requests',
      render: (requests: number) => requests.toLocaleString()
    },
    {
      title: '胜出模型',
      key: 'winner',
      render: (_, record: ABTest) => {
        if (!record.winner) return <Tag>待确定</Tag>;
        const winner = record.winner === 'A' ? record.model_a_name : record.model_b_name;
        return <Tag color="gold" icon={<TrophyOutlined />}>{winner}</Tag>;
      }
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record: ABTest) => (
        <Space>
          {record.status === 'running' ? (
            <Button 
              icon={<StopOutlined />} 
              size="small"
              onClick={() => handleTestAction(record.id, 'pause')}
            >
              暂停
            </Button>
          ) : (
            <Button 
              icon={<PlayCircleOutlined />} 
              size="small" 
              type="primary"
              onClick={() => handleTestAction(record.id, 'start')}
            >
              启动
            </Button>
          )}
          <Button 
            icon={<BarChartOutlined />} 
            size="small"
            onClick={() => setSelectedTest(record)}
          >
            详情
          </Button>
        </Space>
      )
    }
  ];

  const jobColumns = [
    {
      title: '任务名称',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: LearningJob) => (
        <Space>
          <Avatar style={{ backgroundColor: '#52c41a' }}>
            <BulbOutlined />
          </Avatar>
          <div>
            <div>{text}</div>
            <div style={{ fontSize: '12px', color: '#999' }}>{record.model_name}</div>
          </div>
        </Space>
      )
    },
    {
      title: '学习类型',
      dataIndex: 'learning_type',
      key: 'learning_type',
      render: (type: string) => (
        <Tag color={type === 'incremental' ? 'blue' : type === 'continual' ? 'green' : 'purple'}>
          {getLearningTypeText(type)}
        </Tag>
      )
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Badge status={getStatusColor(status) as any} text={getStatusText(status)} />
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
      title: '数据样本',
      dataIndex: 'data_samples',
      key: 'data_samples',
      render: (samples: number) => samples.toLocaleString()
    },
    {
      title: '性能提升',
      key: 'improvement',
      render: (_, record: LearningJob) => (
        <div style={{ color: record.metrics.improvement > 0 ? '#52c41a' : '#999' }}>
          {record.metrics.improvement > 0 ? '+' : ''}{record.metrics.improvement.toFixed(1)}%
        </div>
      )
    },
    {
      title: '预计完成',
      dataIndex: 'estimated_completion',
      key: 'estimated_completion',
      render: (time: string) => {
        const now = new Date();
        const timeDate = new Date(time);
        const diffMs = now.getTime() - timeDate.getTime();
        const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
        const diffDays = Math.floor(diffHours / 24);
        
        if (diffDays > 0) {
          return `${diffDays}天前`;
        } else if (diffHours > 0) {
          return `${diffHours}小时前`;
        } else {
          const diffMinutes = Math.floor(diffMs / (1000 * 60));
          return `${Math.max(1, diffMinutes)}分钟前`;
        }
      }
    }
  ];

  // 生成性能对比数据
  const generatePerformanceData = (test: ABTest) => {
    return [
      {
        metric: '准确率',
        modelA: test.model_a_performance.accuracy * 100,
        modelB: test.model_b_performance.accuracy * 100
      },
      {
        metric: '延迟(ms)',
        modelA: test.model_a_performance.latency,
        modelB: test.model_b_performance.latency
      },
      {
        metric: '错误率(%)',
        modelA: test.model_a_performance.error_rate,
        modelB: test.model_b_performance.error_rate
      }
    ];
  };

  return (
    <div style={{ padding: '24px' }}>
      {/* 统计概览 */}
      <Row gutter={16} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="活跃A/B测试"
              value={systemStats.active_tests}
              prefix={<ExperimentOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="运行中任务"
              value={systemStats.active_jobs}
              prefix={<RocketOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="反馈数据"
              value={systemStats.total_feedback}
              prefix={<FireOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="平均性能提升"
              value={systemStats.avg_improvement}
              suffix="%"
              precision={1}
              prefix={<TrophyOutlined />}
              valueStyle={{ color: '#cf1322' }}
            />
          </Card>
        </Col>
      </Row>

      <Tabs defaultActiveKey="ab-tests">
        <TabPane tab="A/B测试" key="ab-tests">
          <Card
            title="A/B测试管理"
            extra={
              <Button 
                type="primary" 
                icon={<ExperimentOutlined />}
                onClick={() => setTestModalVisible(true)}
              >
                创建A/B测试
              </Button>
            }
          >
            <Table
              columns={testColumns}
              dataSource={abTests}
              rowKey="id"
              loading={loading}
            />
          </Card>
        </TabPane>

        <TabPane tab="在线学习" key="learning">
          <Card
            title="在线学习任务"
            extra={
              <Button 
                type="primary" 
                icon={<BulbOutlined />}
                onClick={() => setJobModalVisible(true)}
              >
                创建学习任务
              </Button>
            }
          >
            <Table
              columns={jobColumns}
              dataSource={learningJobs}
              rowKey="id"
              loading={loading}
            />
          </Card>
        </TabPane>

        <TabPane tab="反馈管理" key="feedback">
          <Row gutter={16}>
            <Col span={16}>
              <Card title="用户反馈数据">
                <List
                  itemLayout="horizontal"
                  dataSource={feedbackData}
                  renderItem={feedback => (
                    <List.Item
                      actions={[
                        <Button 
                          type="link" 
                          size="small"
                          onClick={() => {
                            setFeedbackData(prev => prev.map(f => 
                              f.id === feedback.id ? { ...f, processed: true } : f
                            ));
                            message.success('反馈已处理');
                          }}
                          disabled={feedback.processed}
                        >
                          {feedback.processed ? '已处理' : '处理'}
                        </Button>
                      ]}
                    >
                      <List.Item.Meta
                        avatar={
                          <Avatar 
                            style={{ 
                              backgroundColor: feedback.processed ? '#52c41a' : '#faad14' 
                            }}
                          >
                            {feedback.processed ? <CheckCircleOutlined /> : <ClockCircleOutlined />}
                          </Avatar>
                        }
                        title={
                          <Space>
                            <span>预测: {String(feedback.prediction)}</span>
                            <span>→</span>
                            <span>实际: {String(feedback.ground_truth)}</span>
                            <Tag color={feedback.feedback_score >= 4 ? 'green' : feedback.feedback_score >= 3 ? 'orange' : 'red'}>
                              评分: {feedback.feedback_score}/5
                            </Tag>
                          </Space>
                        }
                        description={
                          <div>
                            <div>{feedback.user_feedback}</div>
                            <div style={{ fontSize: '12px', color: '#999', marginTop: '4px' }}>
                              {new Date(feedback.timestamp).toLocaleString('zh-CN', {
                                year: 'numeric',
                                month: '2-digit', 
                                day: '2-digit',
                                hour: '2-digit',
                                minute: '2-digit',
                                second: '2-digit'
                              })}
                            </div>
                          </div>
                        }
                      />
                    </List.Item>
                  )}
                />
              </Card>
            </Col>
            <Col span={8}>
              <Card title="反馈统计">
                <Statistic
                  title="待处理反馈"
                  value={feedbackData.filter(f => !f.processed).length}
                  suffix="条"
                  style={{ marginBottom: '16px' }}
                />
                <Statistic
                  title="平均评分"
                  value={feedbackData.reduce((sum, f) => sum + f.feedback_score, 0) / feedbackData.length || 0}
                  precision={1}
                  suffix="/5"
                />
              </Card>
              
              <Card title="反馈导入" style={{ marginTop: '16px' }}>
                <Upload
                  accept=".csv,.json"
                  showUploadList={false}
                  beforeUpload={(file) => {
                    message.success(`${file.name} 反馈数据导入成功`);
                    return false;
                  }}
                >
                  <Button icon={<UploadOutlined />} block>
                    批量导入反馈数据
                  </Button>
                </Upload>
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab="性能分析" key="analysis">
          <Row gutter={16}>
            <Col span={12}>
              <Card title="模型性能趋势">
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={[
                    { time: '00:00', accuracy: 89.2, latency: 45 },
                    { time: '06:00', accuracy: 89.8, latency: 43 },
                    { time: '12:00', accuracy: 90.5, latency: 41 },
                    { time: '18:00', accuracy: 91.2, latency: 39 },
                    { time: '24:00', accuracy: 91.8, latency: 37 }
                  ]}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis yAxisId="left" />
                    <YAxis yAxisId="right" orientation="right" />
                    <RechartsTooltip />
                    <Legend />
                    <Line 
                      yAxisId="left"
                      type="monotone" 
                      dataKey="accuracy" 
                      stroke="#52c41a" 
                      name="准确率 (%)"
                    />
                    <Line 
                      yAxisId="right"
                      type="monotone" 
                      dataKey="latency" 
                      stroke="#1890ff" 
                      name="延迟 (ms)"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </Card>
            </Col>
            <Col span={12}>
              <Card title="学习效果统计">
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={learningJobs.map(job => ({
                    name: job.name,
                    improvement: job.metrics.improvement,
                    baseline: job.metrics.baseline_accuracy * 100,
                    current: job.metrics.current_accuracy * 100
                  }))}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <RechartsTooltip />
                    <Legend />
                    <Bar dataKey="baseline" fill="#faad14" name="基线准确率" />
                    <Bar dataKey="current" fill="#52c41a" name="当前准确率" />
                  </BarChart>
                </ResponsiveContainer>
              </Card>
            </Col>
          </Row>
        </TabPane>
      </Tabs>

      {/* A/B测试创建模态框 */}
      <Modal
        title="创建A/B测试"
        visible={testModalVisible}
        onCancel={() => setTestModalVisible(false)}
        footer={null}
        width={600}
      >
        <Form
          form={testForm}
          layout="vertical"
          onFinish={handleCreateTest}
        >
          <Form.Item
            name="name"
            label="测试名称"
            rules={[{ required: true, message: '请输入测试名称' }]}
          >
            <Input placeholder="请输入测试名称" />
          </Form.Item>

          <Form.Item
            name="description"
            label="测试描述"
          >
            <TextArea rows={3} placeholder="请输入测试描述" />
          </Form.Item>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="model_a_id"
                label="模型A"
                rules={[{ required: true, message: '请选择模型A' }]}
              >
                <Select placeholder="选择模型A">
                  <Option value="model_001">BERT-base</Option>
                  <Option value="model_002">RoBERTa-base</Option>
                  <Option value="model_003">DistilBERT</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="model_b_id"
                label="模型B"
                rules={[{ required: true, message: '请选择模型B' }]}
              >
                <Select placeholder="选择模型B">
                  <Option value="model_001">BERT-base</Option>
                  <Option value="model_002">RoBERTa-base</Option>
                  <Option value="model_003">DistilBERT</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>

          <Form.Item
            name="traffic_split"
            label="流量分配 (模型B的流量百分比)"
            initialValue={50}
          >
            <Slider 
              min={10} 
              max={90} 
              marks={{ 10: '10%', 50: '50%', 90: '90%' }}
            />
          </Form.Item>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit">
                创建测试
              </Button>
              <Button onClick={() => setTestModalVisible(false)}>
                取消
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* 学习任务创建模态框 */}
      <Modal
        title="创建学习任务"
        visible={jobModalVisible}
        onCancel={() => setJobModalVisible(false)}
        footer={null}
        width={600}
      >
        <Form
          form={jobForm}
          layout="vertical"
          onFinish={handleCreateJob}
        >
          <Form.Item
            name="name"
            label="任务名称"
            rules={[{ required: true, message: '请输入任务名称' }]}
          >
            <Input placeholder="请输入任务名称" />
          </Form.Item>

          <Form.Item
            name="model_id"
            label="目标模型"
            rules={[{ required: true, message: '请选择模型' }]}
          >
            <Select placeholder="选择要优化的模型">
              <Option value="model_001">BERT文本分类器</Option>
              <Option value="model_002">情感分析模型</Option>
              <Option value="model_003">推荐系统模型</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="learning_type"
            label="学习类型"
            rules={[{ required: true, message: '请选择学习类型' }]}
          >
            <Select placeholder="选择学习类型">
              <Option value="incremental">增量学习</Option>
              <Option value="continual">持续学习</Option>
              <Option value="federated">联邦学习</Option>
            </Select>
          </Form.Item>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit">
                创建任务
              </Button>
              <Button onClick={() => setJobModalVisible(false)}>
                取消
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* A/B测试详情模态框 */}
      <Modal
        title="A/B测试详情"
        visible={!!selectedTest}
        onCancel={() => setSelectedTest(null)}
        footer={null}
        width={800}
      >
        {selectedTest && (
          <Tabs defaultActiveKey="overview">
            <TabPane tab="测试概览" key="overview">
              <Descriptions bordered column={2}>
                <Descriptions.Item label="测试名称">{selectedTest.name}</Descriptions.Item>
                <Descriptions.Item label="状态">{getStatusText(selectedTest.status)}</Descriptions.Item>
                <Descriptions.Item label="开始时间">{new Date(selectedTest.start_time).toLocaleString('zh-CN', {
                  year: 'numeric',
                  month: '2-digit',
                  day: '2-digit', 
                  hour: '2-digit',
                  minute: '2-digit',
                  second: '2-digit'
                })}</Descriptions.Item>
                <Descriptions.Item label="总请求数">{selectedTest.total_requests.toLocaleString()}</Descriptions.Item>
                <Descriptions.Item label="统计显著性">{(selectedTest.statistical_significance * 100).toFixed(1)}%</Descriptions.Item>
                <Descriptions.Item label="胜出模型">{selectedTest.winner ? (selectedTest.winner === 'A' ? selectedTest.model_a_name : selectedTest.model_b_name) : '待确定'}</Descriptions.Item>
              </Descriptions>
            </TabPane>
            <TabPane tab="性能对比" key="comparison">
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={generatePerformanceData(selectedTest)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="metric" />
                  <YAxis />
                  <RechartsTooltip />
                  <Legend />
                  <Bar dataKey="modelA" fill="#1890ff" name={selectedTest.model_a_name} />
                  <Bar dataKey="modelB" fill="#52c41a" name={selectedTest.model_b_name} />
                </BarChart>
              </ResponsiveContainer>
            </TabPane>
          </Tabs>
        )}
      </Modal>
    </div>
  );
};

export default OnlineLearningPage;