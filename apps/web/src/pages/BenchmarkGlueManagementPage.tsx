import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Table, Button, Modal, Form, Input, Select, Tag, Space, Progress, Statistic, Tabs, Switch, message, Tooltip, Popconfirm, Upload, Alert } from 'antd';
import { FileTextOutlined, PlusOutlined, EditOutlined, DeleteOutlined, UploadOutlined, DownloadOutlined, PlayCircleOutlined, PauseCircleOutlined, SettingOutlined, InfoCircleOutlined, CheckCircleOutlined } from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import dayjs from 'dayjs';

const { Option } = Select;
const { TextArea } = Input;
const { TabPane } = Tabs;

// GLUE基准测试任务定义
interface GlueTask {
  id: string;
  name: string;
  fullName: string;
  description: string;
  category: 'single_sentence' | 'similarity_matching' | 'inference';
  metrics: string[];
  samples: {
    train: number;
    validation: number;
    test: number;
  };
  status: 'active' | 'disabled' | 'deprecated';
  difficulty: 'easy' | 'medium' | 'hard';
  dataFormat: string;
  lastUpdated: string;
}

interface GlueBenchmarkConfig {
  id: string;
  name: string;
  description: string;
  selectedTasks: string[];
  status: 'draft' | 'active' | 'running' | 'completed';
  createdAt: string;
  results?: GlueTaskResult[];
}

interface GlueTaskResult {
  taskId: string;
  taskName: string;
  accuracy: number;
  f1Score?: number;
  matthewsCorr?: number;
  pearsonCorr?: number;
  spearmanCorr?: number;
}

const BenchmarkGlueManagementPage: React.FC = () => {
  // 状态管理
  const [glueTasks, setGlueTasks] = useState<GlueTask[]>([]);
  const [benchmarkConfigs, setBenchmarkConfigs] = useState<GlueBenchmarkConfig[]>([]);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('tasks');

  // 模态框状态
  const [isTaskModalVisible, setIsTaskModalVisible] = useState(false);
  const [isConfigModalVisible, setIsConfigModalVisible] = useState(false);
  const [editingTask, setEditingTask] = useState<GlueTask | null>(null);
  const [editingConfig, setEditingConfig] = useState<GlueBenchmarkConfig | null>(null);

  // 表单
  const [taskForm] = Form.useForm();
  const [configForm] = Form.useForm();

  // 模拟数据加载
  useEffect(() => {
    loadMockData();
  }, []);

  const loadMockData = () => {
    const mockGlueTasks: GlueTask[] = [
      {
        id: '1',
        name: 'CoLA',
        fullName: 'Corpus of Linguistic Acceptability',
        description: '判断英语句子的语法可接受性',
        category: 'single_sentence',
        metrics: ['accuracy', 'matthews_correlation'],
        samples: {
          train: 8551,
          validation: 1043,
          test: 1063,
        },
        status: 'active',
        difficulty: 'hard',
        dataFormat: 'TSV',
        lastUpdated: '2024-01-15T10:00:00',
      },
      {
        id: '2',
        name: 'SST-2',
        fullName: 'Stanford Sentiment Treebank',
        description: '电影评论情感分析（正面/负面）',
        category: 'single_sentence',
        metrics: ['accuracy'],
        samples: {
          train: 67349,
          validation: 872,
          test: 1821,
        },
        status: 'active',
        difficulty: 'easy',
        dataFormat: 'TSV',
        lastUpdated: '2024-01-15T10:00:00',
      },
      {
        id: '3',
        name: 'MRPC',
        fullName: 'Microsoft Research Paraphrase Corpus',
        description: '判断两个句子是否语义相同',
        category: 'similarity_matching',
        metrics: ['accuracy', 'f1_score'],
        samples: {
          train: 3668,
          validation: 408,
          test: 1725,
        },
        status: 'active',
        difficulty: 'medium',
        dataFormat: 'TSV',
        lastUpdated: '2024-01-15T10:00:00',
      },
      {
        id: '4',
        name: 'STS-B',
        fullName: 'Semantic Textual Similarity Benchmark',
        description: '计算两个句子的语义相似度得分',
        category: 'similarity_matching',
        metrics: ['pearson_correlation', 'spearman_correlation'],
        samples: {
          train: 5749,
          validation: 1500,
          test: 1379,
        },
        status: 'active',
        difficulty: 'medium',
        dataFormat: 'TSV',
        lastUpdated: '2024-01-15T10:00:00',
      },
      {
        id: '5',
        name: 'QQP',
        fullName: 'Quora Question Pairs',
        description: '判断两个问题是否语义重复',
        category: 'similarity_matching',
        metrics: ['accuracy', 'f1_score'],
        samples: {
          train: 363846,
          validation: 40430,
          test: 390965,
        },
        status: 'active',
        difficulty: 'medium',
        dataFormat: 'TSV',
        lastUpdated: '2024-01-15T10:00:00',
      },
      {
        id: '6',
        name: 'MNLI',
        fullName: 'Multi-Genre Natural Language Inference',
        description: '自然语言推理（蕴含、中性、矛盾）',
        category: 'inference',
        metrics: ['accuracy'],
        samples: {
          train: 392702,
          validation: 9815,
          test: 9796,
        },
        status: 'active',
        difficulty: 'hard',
        dataFormat: 'JSONL',
        lastUpdated: '2024-01-15T10:00:00',
      },
      {
        id: '7',
        name: 'QNLI',
        fullName: 'Question-answering Natural Language Inference',
        description: '基于问答的自然语言推理',
        category: 'inference',
        metrics: ['accuracy'],
        samples: {
          train: 104743,
          validation: 5463,
          test: 5463,
        },
        status: 'active',
        difficulty: 'medium',
        dataFormat: 'TSV',
        lastUpdated: '2024-01-15T10:00:00',
      },
      {
        id: '8',
        name: 'RTE',
        fullName: 'Recognizing Textual Entailment',
        description: '文本蕴含识别',
        category: 'inference',
        metrics: ['accuracy'],
        samples: {
          train: 2490,
          validation: 277,
          test: 3000,
        },
        status: 'active',
        difficulty: 'hard',
        dataFormat: 'JSONL',
        lastUpdated: '2024-01-15T10:00:00',
      },
      {
        id: '9',
        name: 'WNLI',
        fullName: 'Winograd Natural Language Inference',
        description: 'Winograd模式的自然语言推理',
        category: 'inference',
        metrics: ['accuracy'],
        samples: {
          train: 635,
          validation: 71,
          test: 146,
        },
        status: 'deprecated',
        difficulty: 'hard',
        dataFormat: 'JSONL',
        lastUpdated: '2024-01-15T10:00:00',
      },
    ];

    const mockBenchmarkConfigs: GlueBenchmarkConfig[] = [
      {
        id: '1',
        name: 'GLUE完整基准测试',
        description: '包含所有GLUE任务的完整评估配置',
        selectedTasks: ['CoLA', 'SST-2', 'MRPC', 'STS-B', 'QQP', 'MNLI', 'QNLI', 'RTE'],
        status: 'active',
        createdAt: '2024-01-10T10:00:00',
        results: [
          { taskId: '1', taskName: 'CoLA', accuracy: 0.685, matthewsCorr: 0.425 },
          { taskId: '2', taskName: 'SST-2', accuracy: 0.945 },
          { taskId: '3', taskName: 'MRPC', accuracy: 0.883, f1Score: 0.915 },
          { taskId: '4', taskName: 'STS-B', pearsonCorr: 0.887, spearmanCorr: 0.886 },
          { taskId: '5', taskName: 'QQP', accuracy: 0.915, f1Score: 0.878 },
          { taskId: '6', taskName: 'MNLI', accuracy: 0.865 },
          { taskId: '7', taskName: 'QNLI', accuracy: 0.920 },
          { taskId: '8', taskName: 'RTE', accuracy: 0.753 },
        ],
      },
      {
        id: '2',
        name: '情感分析专项测试',
        description: '专注于情感分析相关任务',
        selectedTasks: ['SST-2'],
        status: 'running',
        createdAt: '2024-01-18T14:30:00',
      },
    ];

    setGlueTasks(mockGlueTasks);
    setBenchmarkConfigs(mockBenchmarkConfigs);
  };

  // 任务操作
  const handleCreateTask = () => {
    setEditingTask(null);
    taskForm.resetFields();
    setIsTaskModalVisible(true);
  };

  const handleEditTask = (task: GlueTask) => {
    setEditingTask(task);
    taskForm.setFieldsValue(task);
    setIsTaskModalVisible(true);
  };

  const handleDeleteTask = (id: string) => {
    setGlueTasks(glueTasks.filter(task => task.id !== id));
    message.success('任务已删除');
  };

  const handleToggleTask = (id: string, status: GlueTask['status']) => {
    setGlueTasks(glueTasks.map(task => 
      task.id === id 
        ? { ...task, status: status === 'active' ? 'disabled' : 'active' }
        : task
    ));
    message.success(`任务已${status === 'active' ? '禁用' : '启用'}`);
  };

  const handleSaveTask = async () => {
    try {
      const values = await taskForm.validateFields();
      
      if (editingTask) {
        setGlueTasks(glueTasks.map(task => 
          task.id === editingTask.id 
            ? { ...task, ...values, lastUpdated: new Date().toISOString() }
            : task
        ));
        message.success('任务已更新');
      } else {
        const newTask: GlueTask = {
          ...values,
          id: Date.now().toString(),
          lastUpdated: new Date().toISOString(),
        };
        setGlueTasks([...glueTasks, newTask]);
        message.success('任务已创建');
      }
      
      setIsTaskModalVisible(false);
      taskForm.resetFields();
    } catch (error) {
      message.error('保存失败，请检查输入');
    }
  };

  // 基准配置操作
  const handleCreateConfig = () => {
    setEditingConfig(null);
    configForm.resetFields();
    setIsConfigModalVisible(true);
  };

  const handleRunBenchmark = (id: string) => {
    setBenchmarkConfigs(benchmarkConfigs.map(config => 
      config.id === id 
        ? { ...config, status: 'running' }
        : config
    ));
    message.success('基准测试已启动');
  };

  const handleSaveConfig = async () => {
    try {
      const values = await configForm.validateFields();
      
      if (editingConfig) {
        setBenchmarkConfigs(benchmarkConfigs.map(config => 
          config.id === editingConfig.id 
            ? { ...config, ...values }
            : config
        ));
        message.success('配置已更新');
      } else {
        const newConfig: GlueBenchmarkConfig = {
          ...values,
          id: Date.now().toString(),
          status: 'draft',
          createdAt: new Date().toISOString(),
        };
        setBenchmarkConfigs([...benchmarkConfigs, newConfig]);
        message.success('配置已创建');
      }
      
      setIsConfigModalVisible(false);
      configForm.resetFields();
    } catch (error) {
      message.error('保存失败，请检查输入');
    }
  };

  // 表格列定义
  const taskColumns: ColumnsType<GlueTask> = [
    {
      title: '任务名称',
      key: 'name',
      render: (record: GlueTask) => (
        <div>
          <div style={{ fontWeight: 500, display: 'flex', alignItems: 'center' }}>
            {record.name}
            <Tooltip title={record.description}>
              <InfoCircleOutlined style={{ marginLeft: '8px', color: '#999' }} />
            </Tooltip>
          </div>
          <div style={{ fontSize: '12px', color: '#666' }}>
            {record.fullName}
          </div>
        </div>
      ),
    },
    {
      title: '分类',
      dataIndex: 'category',
      key: 'category',
      render: (category: string) => {
        const colors = {
          single_sentence: 'blue',
          similarity_matching: 'green',
          inference: 'orange',
        };
        const labels = {
          single_sentence: '单句任务',
          similarity_matching: '相似度匹配',
          inference: '推理任务',
        };
        return (
          <Tag color={colors[category as keyof typeof colors]}>
            {labels[category as keyof typeof labels]}
          </Tag>
        );
      },
    },
    {
      title: '难度',
      dataIndex: 'difficulty',
      key: 'difficulty',
      render: (difficulty: string) => {
        const colors = {
          easy: 'green',
          medium: 'orange',
          hard: 'red',
        };
        const labels = {
          easy: '简单',
          medium: '中等',
          hard: '困难',
        };
        return (
          <Tag color={colors[difficulty as keyof typeof colors]}>
            {labels[difficulty as keyof typeof labels]}
          </Tag>
        );
      },
    },
    {
      title: '数据量',
      key: 'samples',
      render: (record: GlueTask) => (
        <div>
          <div>训练: {record.samples.train.toLocaleString()}</div>
          <div>验证: {record.samples.validation.toLocaleString()}</div>
          <div>测试: {record.samples.test.toLocaleString()}</div>
        </div>
      ),
    },
    {
      title: '评估指标',
      dataIndex: 'metrics',
      key: 'metrics',
      render: (metrics: string[]) => (
        <div>
          {metrics.map(metric => (
            <Tag key={metric} size="small" style={{ marginBottom: '2px' }}>
              {metric}
            </Tag>
          ))}
        </div>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const colors = {
          active: 'green',
          disabled: 'orange',
          deprecated: 'red',
        };
        const labels = {
          active: '启用',
          disabled: '禁用',
          deprecated: '已弃用',
        };
        return (
          <Tag color={colors[status as keyof typeof colors]}>
            {labels[status as keyof typeof labels]}
          </Tag>
        );
      },
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: GlueTask) => (
        <Space>
          <Tooltip title="编辑">
            <Button 
              type="text" 
              icon={<EditOutlined />} 
              size="small"
              onClick={() => handleEditTask(record)}
            />
          </Tooltip>
          <Tooltip title={record.status === 'active' ? '禁用' : '启用'}>
            <Button 
              type="text" 
              icon={record.status === 'active' ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
              size="small"
              onClick={() => handleToggleTask(record.id, record.status)}
            />
          </Tooltip>
          <Popconfirm
            title="确定要删除这个任务吗？"
            onConfirm={() => handleDeleteTask(record.id)}
          >
            <Tooltip title="删除">
              <Button type="text" icon={<DeleteOutlined />} size="small" danger />
            </Tooltip>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  const configColumns: ColumnsType<GlueBenchmarkConfig> = [
    {
      title: '配置名称',
      key: 'name',
      render: (record: GlueBenchmarkConfig) => (
        <div>
          <div style={{ fontWeight: 500 }}>{record.name}</div>
          <div style={{ fontSize: '12px', color: '#666' }}>
            {record.description}
          </div>
        </div>
      ),
    },
    {
      title: '包含任务',
      dataIndex: 'selectedTasks',
      key: 'selectedTasks',
      render: (tasks: string[]) => (
        <div>
          {tasks.slice(0, 3).map(task => (
            <Tag key={task} size="small" style={{ marginBottom: '2px' }}>
              {task}
            </Tag>
          ))}
          {tasks.length > 3 && (
            <Tag size="small">+{tasks.length - 3}个更多</Tag>
          )}
        </div>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string, record: GlueBenchmarkConfig) => {
        const colors = {
          draft: 'default',
          active: 'blue',
          running: 'orange',
          completed: 'green',
        };
        const labels = {
          draft: '草稿',
          active: '就绪',
          running: '运行中',
          completed: '已完成',
        };
        
        return (
          <div>
            <Tag color={colors[status as keyof typeof colors]}>
              {labels[status as keyof typeof labels]}
            </Tag>
            {status === 'running' && (
              <Progress percent={65} size="small" style={{ width: '60px', marginTop: '4px' }} />
            )}
          </div>
        );
      },
    },
    {
      title: '创建时间',
      dataIndex: 'createdAt',
      key: 'createdAt',
      render: (date: string) => dayjs(date).format('YYYY-MM-DD HH:mm'),
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: GlueBenchmarkConfig) => (
        <Space>
          {record.status === 'active' && (
            <Tooltip title="运行基准测试">
              <Button 
                type="text" 
                icon={<PlayCircleOutlined />} 
                size="small"
                onClick={() => handleRunBenchmark(record.id)}
              />
            </Tooltip>
          )}
          {record.status === 'completed' && (
            <Tooltip title="查看结果">
              <Button type="text" icon={<CheckCircleOutlined />} size="small" />
            </Tooltip>
          )}
          <Tooltip title="编辑">
            <Button type="text" icon={<EditOutlined />} size="small" />
          </Tooltip>
          <Tooltip title="下载报告">
            <Button type="text" icon={<DownloadOutlined />} size="small" />
          </Tooltip>
        </Space>
      ),
    },
  ];

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px' }}>
        <h1>GLUE基准测试管理</h1>
        <p>管理GLUE（General Language Understanding Evaluation）基准测试任务和配置</p>
      </div>

      <Alert
        message="GLUE基准测试说明"
        description="GLUE是一个多任务基准测试，包含9个英语理解任务，涵盖单句任务、相似度匹配和推理任务。除WNLI外的8个任务被广泛用于模型评估。"
        type="info"
        showIcon
        style={{ marginBottom: '16px' }}
      />

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="GLUE任务管理" key="tasks">
          <div style={{ marginBottom: '16px' }}>
            <Space>
              <Button 
                type="primary" 
                icon={<PlusOutlined />}
                onClick={handleCreateTask}
              >
                新增GLUE任务
              </Button>
              <Button icon={<UploadOutlined />}>
                导入数据集
              </Button>
              <Button icon={<DownloadOutlined />}>
                导出配置
              </Button>
            </Space>
          </div>
          
          <Table
            columns={taskColumns}
            dataSource={glueTasks}
            rowKey="id"
            loading={loading}
            pagination={{ pageSize: 10 }}
          />
        </TabPane>

        <TabPane tab="基准配置管理" key="configs">
          <div style={{ marginBottom: '16px' }}>
            <Button 
              type="primary" 
              icon={<PlusOutlined />}
              onClick={handleCreateConfig}
            >
              新建基准配置
            </Button>
          </div>
          
          <Table
            columns={configColumns}
            dataSource={benchmarkConfigs}
            rowKey="id"
            loading={loading}
            pagination={{ pageSize: 10 }}
            expandable={{
              expandedRowRender: (record) => (
                <div style={{ padding: '16px' }}>
                  {record.results ? (
                    <Row gutter={16}>
                      <Col span={24}>
                        <h4>评估结果</h4>
                        <Row gutter={16}>
                          {record.results.map(result => (
                            <Col span={8} key={result.taskId} style={{ marginBottom: '8px' }}>
                              <Card size="small">
                                <Statistic
                                  title={result.taskName}
                                  value={result.accuracy}
                                  precision={3}
                                  suffix="acc"
                                />
                                {result.f1Score && (
                                  <div style={{ fontSize: '12px', marginTop: '4px' }}>
                                    F1: {result.f1Score.toFixed(3)}
                                  </div>
                                )}
                                {result.matthewsCorr && (
                                  <div style={{ fontSize: '12px' }}>
                                    MCC: {result.matthewsCorr.toFixed(3)}
                                  </div>
                                )}
                              </Card>
                            </Col>
                          ))}
                        </Row>
                      </Col>
                    </Row>
                  ) : (
                    <div>该配置尚未运行或正在运行中</div>
                  )}
                </div>
              ),
              rowExpandable: (record) => record.status === 'completed',
            }}
          />
        </TabPane>

        <TabPane tab="数据集统计" key="statistics">
          <Row gutter={[16, 16]}>
            <Col span={6}>
              <Card>
                <Statistic
                  title="总任务数"
                  value={glueTasks.length}
                  suffix="个"
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="启用任务"
                  value={glueTasks.filter(t => t.status === 'active').length}
                  suffix="个"
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="总样本数"
                  value={glueTasks.reduce((sum, task) => 
                    sum + task.samples.train + task.samples.validation + task.samples.test, 0
                  )}
                  suffix="条"
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="配置数"
                  value={benchmarkConfigs.length}
                  suffix="个"
                />
              </Card>
            </Col>
          </Row>

          <Row gutter={16} style={{ marginTop: '16px' }}>
            <Col span={12}>
              <Card title="任务分类分布">
                <div>
                  <div>单句任务: {glueTasks.filter(t => t.category === 'single_sentence').length}个</div>
                  <div>相似度匹配: {glueTasks.filter(t => t.category === 'similarity_matching').length}个</div>
                  <div>推理任务: {glueTasks.filter(t => t.category === 'inference').length}个</div>
                </div>
              </Card>
            </Col>
            <Col span={12}>
              <Card title="难度分布">
                <div>
                  <div>简单: {glueTasks.filter(t => t.difficulty === 'easy').length}个</div>
                  <div>中等: {glueTasks.filter(t => t.difficulty === 'medium').length}个</div>
                  <div>困难: {glueTasks.filter(t => t.difficulty === 'hard').length}个</div>
                </div>
              </Card>
            </Col>
          </Row>
        </TabPane>
      </Tabs>

      {/* 任务编辑模态框 */}
      <Modal
        title={editingTask ? "编辑GLUE任务" : "新增GLUE任务"}
        open={isTaskModalVisible}
        onOk={handleSaveTask}
        onCancel={() => {
          setIsTaskModalVisible(false);
          taskForm.resetFields();
        }}
        width={800}
      >
        <Form
          form={taskForm}
          layout="vertical"
        >
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="name"
                label="任务简称"
                rules={[{ required: true, message: '请输入任务简称' }]}
              >
                <Input placeholder="如: CoLA, SST-2" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="fullName"
                label="完整名称"
                rules={[{ required: true, message: '请输入完整名称' }]}
              >
                <Input placeholder="任务的完整英文名称" />
              </Form.Item>
            </Col>
          </Row>

          <Form.Item
            name="description"
            label="任务描述"
            rules={[{ required: true, message: '请输入任务描述' }]}
          >
            <TextArea rows={3} placeholder="详细描述任务的目标和内容" />
          </Form.Item>

          <Row gutter={16}>
            <Col span={8}>
              <Form.Item
                name="category"
                label="任务分类"
                rules={[{ required: true, message: '请选择任务分类' }]}
              >
                <Select placeholder="选择任务分类">
                  <Option value="single_sentence">单句任务</Option>
                  <Option value="similarity_matching">相似度匹配</Option>
                  <Option value="inference">推理任务</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item
                name="difficulty"
                label="难度等级"
                rules={[{ required: true, message: '请选择难度等级' }]}
              >
                <Select placeholder="选择难度等级">
                  <Option value="easy">简单</Option>
                  <Option value="medium">中等</Option>
                  <Option value="hard">困难</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item
                name="dataFormat"
                label="数据格式"
                rules={[{ required: true, message: '请输入数据格式' }]}
              >
                <Select placeholder="选择数据格式">
                  <Option value="TSV">TSV</Option>
                  <Option value="JSONL">JSONL</Option>
                  <Option value="CSV">CSV</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>

          <Form.Item
            name="metrics"
            label="评估指标"
            rules={[{ required: true, message: '请选择评估指标' }]}
          >
            <Select
              mode="multiple"
              placeholder="选择评估指标"
            >
              <Option value="accuracy">准确率</Option>
              <Option value="f1_score">F1分数</Option>
              <Option value="matthews_correlation">马修斯相关系数</Option>
              <Option value="pearson_correlation">皮尔逊相关系数</Option>
              <Option value="spearman_correlation">斯皮尔曼相关系数</Option>
            </Select>
          </Form.Item>
        </Form>
      </Modal>

      {/* 基准配置模态框 */}
      <Modal
        title="创建基准配置"
        open={isConfigModalVisible}
        onOk={handleSaveConfig}
        onCancel={() => {
          setIsConfigModalVisible(false);
          configForm.resetFields();
        }}
        width={600}
      >
        <Form
          form={configForm}
          layout="vertical"
        >
          <Form.Item
            name="name"
            label="配置名称"
            rules={[{ required: true, message: '请输入配置名称' }]}
          >
            <Input placeholder="输入基准配置名称" />
          </Form.Item>

          <Form.Item
            name="description"
            label="配置描述"
            rules={[{ required: true, message: '请输入配置描述' }]}
          >
            <TextArea rows={3} placeholder="描述这个基准配置的用途" />
          </Form.Item>

          <Form.Item
            name="selectedTasks"
            label="选择任务"
            rules={[{ required: true, message: '请选择至少一个任务' }]}
          >
            <Select
              mode="multiple"
              placeholder="选择要包含的GLUE任务"
            >
              {glueTasks.filter(task => task.status === 'active').map(task => (
                <Option key={task.id} value={task.name}>
                  {task.name} - {task.fullName}
                </Option>
              ))}
            </Select>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default BenchmarkGlueManagementPage;