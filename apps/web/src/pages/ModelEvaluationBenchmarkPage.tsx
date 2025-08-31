import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Row, 
  Col, 
  Button, 
  Table, 
  Form, 
  Input, 
  Select, 
  Modal, 
  Typography, 
  Space, 
  Tag, 
  Progress, 
  Alert, 
  Tabs, 
  Statistic, 
  Divider,
  Switch,
  InputNumber,
  Upload,
  message,
  Tooltip,
  Drawer,
  List,
  Timeline
} from 'antd';
import {
  PlayCircleOutlined,
  StopOutlined,
  DownloadOutlined,
  UploadOutlined,
  SettingOutlined,
  ExperimentOutlined,
  LineChartOutlined,
  TrophyOutlined,
  DeleteOutlined,
  EditOutlined,
  EyeOutlined,
  PlusOutlined,
  ReloadOutlined,
  FileTextOutlined,
  DatabaseOutlined,
  RocketOutlined,
  WarningOutlined
} from '@ant-design/icons';

const { Title, Text, Paragraph } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;
const { TextArea } = Input;

interface BenchmarkDefinition {
  id: string;
  name: string;
  display_name: string;
  description: string;
  category: string;
  difficulty: string;
  tasks: string[];
  languages: string[];
  metrics: string[];
  num_samples: number;
  estimated_runtime_minutes: number;
  memory_requirements_gb: number;
  requires_gpu: boolean;
}

interface ModelInfo {
  id: string;
  name: string;
  version?: string;
  description?: string;
  model_path: string;
  model_type: string;
  architecture?: string;
  parameters_count?: number;
}

interface EvaluationJob {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  current_task?: string;
  models: any[];
  benchmarks: any[];
  started_at?: string;
  completed_at?: string;
  results: any[];
  error_message?: string;
  created_at: string;
}

const ModelEvaluationBenchmarkPage: React.FC = () => {
  const [benchmarks, setBenchmarks] = useState<BenchmarkDefinition[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [evaluationJobs, setEvaluationJobs] = useState<EvaluationJob[]>([]);
  const [loading, setLoading] = useState(false);
  
  // Modal states
  const [isEvaluationModalVisible, setIsEvaluationModalVisible] = useState(false);
  const [isBenchmarkModalVisible, setIsBenchmarkModalVisible] = useState(false);
  const [isModelModalVisible, setIsModelModalVisible] = useState(false);
  const [isJobDetailDrawerVisible, setIsJobDetailDrawerVisible] = useState(false);
  const [selectedJob, setSelectedJob] = useState<EvaluationJob | null>(null);
  
  // Form instances
  const [evaluationForm] = Form.useForm();
  const [benchmarkForm] = Form.useForm();
  const [modelForm] = Form.useForm();

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 30000); // 30秒更新一次
    return () => clearInterval(interval);
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      
      // 模拟API调用 - 获取基准测试列表
      const benchmarksData: BenchmarkDefinition[] = [
        {
          id: 'glue-cola',
          name: 'cola',
          display_name: 'CoLA',
          description: 'Corpus of Linguistic Acceptability - 语言可接受性语料库',
          category: 'nlp',
          difficulty: 'medium',
          tasks: ['cola'],
          languages: ['en'],
          metrics: ['accuracy', 'f1'],
          num_samples: 1043,
          estimated_runtime_minutes: 5,
          memory_requirements_gb: 2.0,
          requires_gpu: false
        },
        {
          id: 'mmlu',
          name: 'mmlu',
          display_name: 'MMLU',
          description: 'Massive Multitask Language Understanding - 大规模多任务语言理解',
          category: 'knowledge',
          difficulty: 'hard',
          tasks: ['mmlu'],
          languages: ['en'],
          metrics: ['accuracy'],
          num_samples: 14042,
          estimated_runtime_minutes: 60,
          memory_requirements_gb: 8.0,
          requires_gpu: true
        },
        {
          id: 'humaneval',
          name: 'humaneval',
          display_name: 'HumanEval',
          description: 'Evaluating Large Language Models Trained on Code - 评估在代码上训练的大型语言模型',
          category: 'code',
          difficulty: 'hard',
          tasks: ['humaneval'],
          languages: ['python'],
          metrics: ['pass@1', 'pass@10', 'pass@100'],
          num_samples: 164,
          estimated_runtime_minutes: 30,
          memory_requirements_gb: 6.0,
          requires_gpu: true
        }
      ];
      
      // 模拟API调用 - 获取模型列表
      const modelsData: ModelInfo[] = [
        {
          id: 'bert-base-uncased',
          name: 'BERT Base Uncased',
          version: '1.0.0',
          description: 'BERT预训练模型，适用于各种NLP任务',
          model_path: '/models/bert-base-uncased',
          model_type: 'text_classification',
          architecture: 'transformer',
          parameters_count: 110000000
        },
        {
          id: 'gpt-3.5-turbo',
          name: 'GPT-3.5 Turbo',
          version: '2023-12',
          description: 'OpenAI的GPT-3.5 Turbo模型',
          model_path: '/models/gpt-3.5-turbo',
          model_type: 'text_generation',
          architecture: 'transformer',
          parameters_count: 175000000000
        },
        {
          id: 'claude-3-sonnet',
          name: 'Claude-3 Sonnet',
          version: '20240229',
          description: 'Anthropic的Claude-3 Sonnet模型',
          model_path: '/models/claude-3-sonnet',
          model_type: 'text_generation',
          architecture: 'transformer'
        }
      ];
      
      // 模拟API调用 - 获取评估任务列表
      const jobsData: EvaluationJob[] = [
        {
          id: 'job_001',
          name: 'BERT在GLUE基准上的评估',
          status: 'running',
          progress: 0.65,
          current_task: '正在评估CoLA任务',
          models: [{ name: 'BERT Base Uncased' }],
          benchmarks: [{ name: 'CoLA' }, { name: 'SST-2' }],
          started_at: '2024-01-15T14:30:00Z',
          results: [],
          created_at: '2024-01-15T14:25:00Z'
        },
        {
          id: 'job_002', 
          name: 'GPT模型代码生成评估',
          status: 'completed',
          progress: 1.0,
          models: [{ name: 'GPT-3.5 Turbo' }],
          benchmarks: [{ name: 'HumanEval' }],
          started_at: '2024-01-15T12:00:00Z',
          completed_at: '2024-01-15T12:45:00Z',
          results: [
            { benchmark: 'HumanEval', accuracy: 0.734, pass_at_1: 0.456 }
          ],
          created_at: '2024-01-15T11:55:00Z'
        }
      ];
      
      setBenchmarks(benchmarksData);
      setModels(modelsData);
      setEvaluationJobs(jobsData);
      
    } catch (error) {
      console.error('加载数据失败:', error);
      message.error('加载数据失败');
    } finally {
      setLoading(false);
    }
  };

  const handleStartEvaluation = async (values: any) => {
    try {
      console.log('启动评估任务:', values);
      
      // 模拟API调用
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      message.success('评估任务已创建，正在后台执行');
      setIsEvaluationModalVisible(false);
      evaluationForm.resetFields();
      loadData();
      
    } catch (error) {
      message.error('创建评估任务失败');
    }
  };

  const handleStopEvaluation = async (jobId: string) => {
    try {
      console.log('停止评估任务:', jobId);
      
      // 模拟API调用
      await new Promise(resolve => setTimeout(resolve, 500));
      
      message.success('评估任务已停止');
      loadData();
      
    } catch (error) {
      message.error('停止评估任务失败');
    }
  };

  const handleDeleteEvaluation = async (jobId: string) => {
    try {
      console.log('删除评估任务:', jobId);
      
      // 模拟API调用
      await new Promise(resolve => setTimeout(resolve, 500));
      
      message.success('评估任务已删除');
      loadData();
      
    } catch (error) {
      message.error('删除评估任务失败');
    }
  };

  const handleAddBenchmark = async (values: any) => {
    try {
      console.log('添加基准测试:', values);
      
      // 模拟API调用
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      message.success('基准测试已添加');
      setIsBenchmarkModalVisible(false);
      benchmarkForm.resetFields();
      loadData();
      
    } catch (error) {
      message.error('添加基准测试失败');
    }
  };

  const handleAddModel = async (values: any) => {
    try {
      console.log('添加模型:', values);
      
      // 模拟API调用
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      message.success('模型已添加');
      setIsModelModalVisible(false);
      modelForm.resetFields();
      loadData();
      
    } catch (error) {
      message.error('添加模型失败');
    }
  };

  const getDifficultyColor = (difficulty: string) => {
    const colorMap = {
      easy: 'green',
      medium: 'orange', 
      hard: 'red',
      expert: 'purple'
    };
    return colorMap[difficulty as keyof typeof colorMap] || 'default';
  };

  const getStatusColor = (status: string) => {
    const colorMap = {
      pending: 'default',
      running: 'processing',
      completed: 'success',
      failed: 'error',
      cancelled: 'warning'
    };
    return colorMap[status as keyof typeof colorMap] || 'default';
  };

  const getStatusText = (status: string) => {
    const textMap = {
      pending: '待执行',
      running: '运行中',
      completed: '已完成',
      failed: '失败',
      cancelled: '已取消'
    };
    return textMap[status as keyof typeof textMap] || status;
  };

  const benchmarkColumns = [
    {
      title: '名称',
      dataIndex: 'display_name',
      key: 'display_name',
      width: 120,
    },
    {
      title: '描述',
      dataIndex: 'description',
      key: 'description',
      ellipsis: true,
    },
    {
      title: '类别',
      dataIndex: 'category',
      key: 'category',
      width: 100,
      render: (category: string) => (
        <Tag color="blue">{category}</Tag>
      ),
    },
    {
      title: '难度',
      dataIndex: 'difficulty',
      key: 'difficulty',
      width: 100,
      render: (difficulty: string) => (
        <Tag color={getDifficultyColor(difficulty)}>{difficulty}</Tag>
      ),
    },
    {
      title: '样本数',
      dataIndex: 'num_samples',
      key: 'num_samples',
      width: 100,
      render: (num: number) => num.toLocaleString(),
    },
    {
      title: '预估时间',
      dataIndex: 'estimated_runtime_minutes',
      key: 'estimated_runtime_minutes',
      width: 100,
      render: (minutes: number) => `${minutes}分钟`,
    },
    {
      title: 'GPU需求',
      dataIndex: 'requires_gpu',
      key: 'requires_gpu',
      width: 100,
      render: (requires: boolean) => (
        <Tag color={requires ? 'red' : 'green'}>
          {requires ? '需要' : '不需要'}
        </Tag>
      ),
    },
    {
      title: '操作',
      key: 'action',
      width: 120,
      render: (_, record: BenchmarkDefinition) => (
        <Space size="small">
          <Button 
            type="link" 
            size="small" 
            icon={<EyeOutlined />}
            onClick={() => console.log('查看详情', record.id)}
          >
            详情
          </Button>
          <Button 
            type="link" 
            size="small" 
            icon={<EditOutlined />}
            onClick={() => console.log('编辑', record.id)}
          >
            编辑
          </Button>
        </Space>
      ),
    },
  ];

  const modelColumns = [
    {
      title: '模型名称',
      dataIndex: 'name',
      key: 'name',
      width: 200,
    },
    {
      title: '版本',
      dataIndex: 'version',
      key: 'version',
      width: 100,
    },
    {
      title: '类型',
      dataIndex: 'model_type',
      key: 'model_type',
      width: 120,
      render: (type: string) => (
        <Tag color="purple">{type}</Tag>
      ),
    },
    {
      title: '架构',
      dataIndex: 'architecture',
      key: 'architecture',
      width: 120,
    },
    {
      title: '参数量',
      dataIndex: 'parameters_count',
      key: 'parameters_count',
      width: 120,
      render: (count?: number) => count ? 
        `${(count / 1000000).toFixed(0)}M` : '-',
    },
    {
      title: '描述',
      dataIndex: 'description',
      key: 'description',
      ellipsis: true,
    },
    {
      title: '操作',
      key: 'action',
      width: 150,
      render: (_, record: ModelInfo) => (
        <Space size="small">
          <Button 
            type="link" 
            size="small" 
            icon={<PlayCircleOutlined />}
            onClick={() => {
              evaluationForm.setFieldValue('models', [record.id]);
              setIsEvaluationModalVisible(true);
            }}
          >
            评估
          </Button>
          <Button 
            type="link" 
            size="small" 
            icon={<EditOutlined />}
            onClick={() => console.log('编辑模型', record.id)}
          >
            编辑
          </Button>
        </Space>
      ),
    },
  ];

  const jobColumns = [
    {
      title: '任务名称',
      dataIndex: 'name',
      key: 'name',
      width: 200,
      ellipsis: true,
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
      title: '进度',
      dataIndex: 'progress',
      key: 'progress',
      width: 120,
      render: (progress: number, record: EvaluationJob) => (
        <div>
          <Progress 
            percent={Math.round(progress * 100)} 
            size="small"
            status={record.status === 'failed' ? 'exception' : 'active'}
          />
          {record.current_task && (
            <Text type="secondary" style={{ fontSize: '12px' }}>
              {record.current_task}
            </Text>
          )}
        </div>
      ),
    },
    {
      title: '模型数',
      key: 'model_count',
      width: 80,
      render: (_, record: EvaluationJob) => record.models.length,
    },
    {
      title: '基准测试数',
      key: 'benchmark_count', 
      width: 100,
      render: (_, record: EvaluationJob) => record.benchmarks.length,
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 150,
      render: (time: string) => new Date(time).toLocaleString(),
    },
    {
      title: '操作',
      key: 'action',
      width: 180,
      render: (_, record: EvaluationJob) => (
        <Space size="small">
          <Button 
            type="link" 
            size="small" 
            icon={<EyeOutlined />}
            onClick={() => {
              setSelectedJob(record);
              setIsJobDetailDrawerVisible(true);
            }}
          >
            详情
          </Button>
          {record.status === 'running' && (
            <Button 
              type="link" 
              size="small" 
              icon={<StopOutlined />}
              onClick={() => handleStopEvaluation(record.id)}
              danger
            >
              停止
            </Button>
          )}
          {record.status === 'completed' && (
            <Button 
              type="link" 
              size="small" 
              icon={<DownloadOutlined />}
              onClick={() => console.log('下载报告', record.id)}
            >
              报告
            </Button>
          )}
          <Button 
            type="link" 
            size="small" 
            icon={<DeleteOutlined />}
            onClick={() => handleDeleteEvaluation(record.id)}
            danger
          >
            删除
          </Button>
        </Space>
      ),
    },
  ];

  return (
    <div style={{ padding: '24px', background: '#f5f5f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>模型评估与基准测试</Title>
        <Text type="secondary">
          管理AI模型评估任务，配置基准测试，监控评估进度
        </Text>
      </div>

      {/* 快速统计 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="可用基准测试"
              value={benchmarks.length}
              prefix={<DatabaseOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="注册模型"
              value={models.length}
              prefix={<TrophyOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="运行中任务"
              value={evaluationJobs.filter(job => job.status === 'running').length}
              prefix={<RocketOutlined />}
              valueStyle={{ color: '#fa8c16' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="完成任务"
              value={evaluationJobs.filter(job => job.status === 'completed').length}
              prefix={<ExperimentOutlined />}
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>
      </Row>

      {/* 主要内容区域 */}
      <Card>
        <Tabs defaultActiveKey="1">
          <TabPane tab="评估任务" key="1">
            <div style={{ marginBottom: '16px' }}>
              <Space>
                <Button 
                  type="primary" 
                  icon={<PlayCircleOutlined />}
                  onClick={() => setIsEvaluationModalVisible(true)}
                >
                  创建评估任务
                </Button>
                <Button 
                  icon={<ReloadOutlined />}
                  onClick={loadData}
                  loading={loading}
                >
                  刷新
                </Button>
              </Space>
            </div>
            <Table
              dataSource={evaluationJobs}
              columns={jobColumns}
              rowKey="id"
              pagination={{
                pageSize: 10,
                showSizeChanger: true,
                showQuickJumper: true,
                showTotal: (total, range) => 
                  `第 ${range[0]}-${range[1]} 项 / 共 ${total} 项`
              }}
              loading={loading}
            />
          </TabPane>
          
          <TabPane tab="基准测试" key="2">
            <div style={{ marginBottom: '16px' }}>
              <Space>
                <Button 
                  type="primary" 
                  icon={<PlusOutlined />}
                  onClick={() => setIsBenchmarkModalVisible(true)}
                >
                  添加基准测试
                </Button>
                <Button 
                  icon={<UploadOutlined />}
                  onClick={() => console.log('批量导入基准测试')}
                >
                  批量导入
                </Button>
              </Space>
            </div>
            <Table
              dataSource={benchmarks}
              columns={benchmarkColumns}
              rowKey="id"
              pagination={{
                pageSize: 10,
                showSizeChanger: true,
                showQuickJumper: true,
                showTotal: (total, range) => 
                  `第 ${range[0]}-${range[1]} 项 / 共 ${total} 项`
              }}
              loading={loading}
            />
          </TabPane>
          
          <TabPane tab="模型管理" key="3">
            <div style={{ marginBottom: '16px' }}>
              <Space>
                <Button 
                  type="primary" 
                  icon={<PlusOutlined />}
                  onClick={() => setIsModelModalVisible(true)}
                >
                  添加模型
                </Button>
                <Button 
                  icon={<UploadOutlined />}
                  onClick={() => console.log('批量注册模型')}
                >
                  批量注册
                </Button>
              </Space>
            </div>
            <Table
              dataSource={models}
              columns={modelColumns}
              rowKey="id"
              pagination={{
                pageSize: 10,
                showSizeChanger: true,
                showQuickJumper: true,
                showTotal: (total, range) => 
                  `第 ${range[0]}-${range[1]} 项 / 共 ${total} 项`
              }}
              loading={loading}
            />
          </TabPane>
        </Tabs>
      </Card>

      {/* 创建评估任务模态框 */}
      <Modal
        title="创建评估任务"
        open={isEvaluationModalVisible}
        onCancel={() => {
          setIsEvaluationModalVisible(false);
          evaluationForm.resetFields();
        }}
        footer={null}
        width={800}
      >
        <Form
          form={evaluationForm}
          layout="vertical"
          onFinish={handleStartEvaluation}
        >
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="name"
                label="任务名称"
                rules={[{ required: true, message: '请输入任务名称' }]}
              >
                <Input placeholder="输入评估任务名称" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="priority"
                label="优先级"
              >
                <Select placeholder="选择优先级">
                  <Option value="low">低</Option>
                  <Option value="medium">中</Option>
                  <Option value="high">高</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>
          
          <Form.Item
            name="description"
            label="描述"
          >
            <TextArea rows={3} placeholder="输入任务描述（可选）" />
          </Form.Item>

          <Form.Item
            name="models"
            label="选择模型"
            rules={[{ required: true, message: '请选择至少一个模型' }]}
          >
            <Select
              mode="multiple"
              placeholder="选择要评估的模型"
              style={{ width: '100%' }}
            >
              {models.map(model => (
                <Option key={model.id} value={model.id}>
                  {model.name} {model.version && `(${model.version})`}
                </Option>
              ))}
            </Select>
          </Form.Item>

          <Form.Item
            name="benchmarks"
            label="选择基准测试"
            rules={[{ required: true, message: '请选择至少一个基准测试' }]}
          >
            <Select
              mode="multiple"
              placeholder="选择基准测试"
              style={{ width: '100%' }}
            >
              {benchmarks.map(benchmark => (
                <Option key={benchmark.id} value={benchmark.id}>
                  <div>
                    <Text>{benchmark.display_name}</Text>
                    <br />
                    <Text type="secondary" style={{ fontSize: '12px' }}>
                      {benchmark.description}
                    </Text>
                  </div>
                </Option>
              ))}
            </Select>
          </Form.Item>

          <Divider>评估配置</Divider>

          <Row gutter={16}>
            <Col span={8}>
              <Form.Item
                name="batch_size"
                label="批次大小"
                initialValue={8}
              >
                <InputNumber 
                  min={1} 
                  max={64} 
                  style={{ width: '100%' }}
                />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item
                name="max_length"
                label="最大长度"
                initialValue={512}
              >
                <InputNumber 
                  min={128} 
                  max={4096} 
                  style={{ width: '100%' }}
                />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item
                name="device"
                label="设备"
                initialValue="auto"
              >
                <Select>
                  <Option value="auto">自动选择</Option>
                  <Option value="cpu">CPU</Option>
                  <Option value="cuda">GPU</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>

          <Form.Item
            name="enable_optimizations"
            valuePropName="checked"
            initialValue={true}
          >
            <Switch /> 启用性能优化
          </Form.Item>

          <Form.Item style={{ marginTop: '24px', marginBottom: 0 }}>
            <Space>
              <Button type="primary" htmlType="submit">
                创建并启动
              </Button>
              <Button 
                onClick={() => {
                  setIsEvaluationModalVisible(false);
                  evaluationForm.resetFields();
                }}
              >
                取消
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* 添加基准测试模态框 */}
      <Modal
        title="添加基准测试"
        open={isBenchmarkModalVisible}
        onCancel={() => {
          setIsBenchmarkModalVisible(false);
          benchmarkForm.resetFields();
        }}
        footer={null}
        width={600}
      >
        <Form
          form={benchmarkForm}
          layout="vertical"
          onFinish={handleAddBenchmark}
        >
          <Form.Item
            name="name"
            label="基准测试名称"
            rules={[{ required: true, message: '请输入基准测试名称' }]}
          >
            <Input placeholder="例如: glue-cola" />
          </Form.Item>

          <Form.Item
            name="display_name"
            label="显示名称"
            rules={[{ required: true, message: '请输入显示名称' }]}
          >
            <Input placeholder="例如: CoLA" />
          </Form.Item>

          <Form.Item
            name="description"
            label="描述"
            rules={[{ required: true, message: '请输入描述' }]}
          >
            <TextArea rows={3} placeholder="输入基准测试的详细描述" />
          </Form.Item>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="category"
                label="类别"
                rules={[{ required: true, message: '请选择类别' }]}
              >
                <Select placeholder="选择类别">
                  <Option value="nlp">NLP</Option>
                  <Option value="knowledge">知识</Option>
                  <Option value="reasoning">推理</Option>
                  <Option value="code">代码</Option>
                  <Option value="multimodal">多模态</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="difficulty"
                label="难度"
                rules={[{ required: true, message: '请选择难度' }]}
              >
                <Select placeholder="选择难度">
                  <Option value="easy">简单</Option>
                  <Option value="medium">中等</Option>
                  <Option value="hard">困难</Option>
                  <Option value="expert">专家</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="num_samples"
                label="样本数量"
                rules={[{ required: true, message: '请输入样本数量' }]}
              >
                <InputNumber 
                  min={1} 
                  style={{ width: '100%' }}
                  placeholder="输入样本数量"
                />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="estimated_runtime_minutes"
                label="预估运行时间（分钟）"
                rules={[{ required: true, message: '请输入预估运行时间' }]}
              >
                <InputNumber 
                  min={1} 
                  style={{ width: '100%' }}
                  placeholder="输入预估时间"
                />
              </Form.Item>
            </Col>
          </Row>

          <Form.Item
            name="requires_gpu"
            valuePropName="checked"
            initialValue={false}
          >
            <Switch /> 需要GPU支持
          </Form.Item>

          <Form.Item style={{ marginTop: '24px', marginBottom: 0 }}>
            <Space>
              <Button type="primary" htmlType="submit">
                添加基准测试
              </Button>
              <Button 
                onClick={() => {
                  setIsBenchmarkModalVisible(false);
                  benchmarkForm.resetFields();
                }}
              >
                取消
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* 添加模型模态框 */}
      <Modal
        title="添加模型"
        open={isModelModalVisible}
        onCancel={() => {
          setIsModelModalVisible(false);
          modelForm.resetFields();
        }}
        footer={null}
        width={600}
      >
        <Form
          form={modelForm}
          layout="vertical"
          onFinish={handleAddModel}
        >
          <Form.Item
            name="name"
            label="模型名称"
            rules={[{ required: true, message: '请输入模型名称' }]}
          >
            <Input placeholder="输入模型名称" />
          </Form.Item>

          <Form.Item
            name="model_path"
            label="模型路径"
            rules={[{ required: true, message: '请输入模型路径' }]}
          >
            <Input placeholder="输入模型文件路径或HuggingFace模型ID" />
          </Form.Item>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="model_type"
                label="模型类型"
                rules={[{ required: true, message: '请选择模型类型' }]}
              >
                <Select placeholder="选择模型类型">
                  <Option value="text_generation">文本生成</Option>
                  <Option value="text_classification">文本分类</Option>
                  <Option value="question_answering">问答</Option>
                  <Option value="summarization">摘要</Option>
                  <Option value="translation">翻译</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="version"
                label="版本"
              >
                <Input placeholder="输入模型版本（可选）" />
              </Form.Item>
            </Col>
          </Row>

          <Form.Item
            name="description"
            label="描述"
          >
            <TextArea rows={3} placeholder="输入模型描述（可选）" />
          </Form.Item>

          <Form.Item style={{ marginTop: '24px', marginBottom: 0 }}>
            <Space>
              <Button type="primary" htmlType="submit">
                添加模型
              </Button>
              <Button 
                onClick={() => {
                  setIsModelModalVisible(false);
                  modelForm.resetFields();
                }}
              >
                取消
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* 任务详情抽屉 */}
      <Drawer
        title="评估任务详情"
        placement="right"
        width={600}
        open={isJobDetailDrawerVisible}
        onClose={() => {
          setIsJobDetailDrawerVisible(false);
          setSelectedJob(null);
        }}
      >
        {selectedJob && (
          <div>
            <Title level={4}>{selectedJob.name}</Title>
            
            <Row gutter={16} style={{ marginBottom: '16px' }}>
              <Col span={12}>
                <Statistic 
                  title="状态" 
                  value={getStatusText(selectedJob.status)}
                  valueStyle={{ 
                    color: selectedJob.status === 'completed' ? '#52c41a' : 
                           selectedJob.status === 'failed' ? '#f5222d' : '#1890ff' 
                  }}
                />
              </Col>
              <Col span={12}>
                <Statistic 
                  title="进度" 
                  value={Math.round(selectedJob.progress * 100)}
                  suffix="%"
                />
              </Col>
            </Row>

            {selectedJob.current_task && (
              <Alert 
                message="当前任务" 
                description={selectedJob.current_task}
                type="info" 
                style={{ marginBottom: '16px' }}
              />
            )}

            <Divider>基本信息</Divider>
            <List size="small">
              <List.Item>
                <Text strong>任务ID:</Text> {selectedJob.id}
              </List.Item>
              <List.Item>
                <Text strong>创建时间:</Text> {new Date(selectedJob.created_at).toLocaleString()}
              </List.Item>
              {selectedJob.started_at && (
                <List.Item>
                  <Text strong>开始时间:</Text> {new Date(selectedJob.started_at).toLocaleString()}
                </List.Item>
              )}
              {selectedJob.completed_at && (
                <List.Item>
                  <Text strong>完成时间:</Text> {new Date(selectedJob.completed_at).toLocaleString()}
                </List.Item>
              )}
              <List.Item>
                <Text strong>模型数量:</Text> {selectedJob.models.length}
              </List.Item>
              <List.Item>
                <Text strong>基准测试数量:</Text> {selectedJob.benchmarks.length}
              </List.Item>
            </List>

            <Divider>模型列表</Divider>
            <List
              size="small"
              dataSource={selectedJob.models}
              renderItem={(model: any) => (
                <List.Item>
                  <Text>{model.name}</Text>
                </List.Item>
              )}
            />

            <Divider>基准测试列表</Divider>
            <List
              size="small" 
              dataSource={selectedJob.benchmarks}
              renderItem={(benchmark: any) => (
                <List.Item>
                  <Text>{benchmark.name}</Text>
                </List.Item>
              )}
            />

            {selectedJob.results.length > 0 && (
              <>
                <Divider>评估结果</Divider>
                <List
                  size="small"
                  dataSource={selectedJob.results}
                  renderItem={(result: any) => (
                    <List.Item>
                      <div style={{ width: '100%' }}>
                        <Text strong>{result.benchmark}</Text>
                        <br />
                        <Text>准确率: {(result.accuracy * 100).toFixed(1)}%</Text>
                        {result.pass_at_1 && (
                          <>
                            <br />
                            <Text>Pass@1: {(result.pass_at_1 * 100).toFixed(1)}%</Text>
                          </>
                        )}
                      </div>
                    </List.Item>
                  )}
                />
              </>
            )}

            {selectedJob.error_message && (
              <>
                <Divider>错误信息</Divider>
                <Alert 
                  message="执行错误" 
                  description={selectedJob.error_message}
                  type="error" 
                  showIcon
                />
              </>
            )}
          </div>
        )}
      </Drawer>
    </div>
  );
};

export default ModelEvaluationBenchmarkPage;