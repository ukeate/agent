import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Table, Button, Modal, Form, Input, Select, Tag, Space, Progress, Statistic, Tabs, message, Tooltip, Popconfirm, Alert, Badge } from 'antd';
import { RocketOutlined, PlusOutlined, EditOutlined, DeleteOutlined, PlayCircleOutlined, PauseCircleOutlined, CheckCircleOutlined, InfoCircleOutlined, DownloadOutlined, UploadOutlined } from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import dayjs from 'dayjs';

const { Option } = Select;
const { TextArea } = Input;
const { TabPane } = Tabs;

// SuperGLUE基准测试任务定义
interface SuperGlueTask {
  id: string;
  name: string;
  fullName: string;
  description: string;
  category: 'reading_comprehension' | 'natural_language_inference' | 'word_sense_disambiguation' | 'coreference_resolution' | 'reasoning';
  metrics: string[];
  humanPerformance: number;
  samples: {
    train: number;
    validation: number;
    test: number;
  };
  status: 'active' | 'disabled' | 'experimental';
  difficulty: 'hard' | 'very_hard' | 'extreme';
  dataFormat: string;
  lastUpdated: string;
}

interface SuperGlueBenchmarkConfig {
  id: string;
  name: string;
  description: string;
  selectedTasks: string[];
  status: 'draft' | 'active' | 'running' | 'completed' | 'failed';
  createdAt: string;
  completedAt?: string;
  results?: SuperGlueTaskResult[];
  overallScore?: number;
}

interface SuperGlueTaskResult {
  taskId: string;
  taskName: string;
  score: number;
  humanPerformance: number;
  humanParity: number; // 与人类表现的比较百分比
  confidence: number;
}

const BenchmarkSupergluePage: React.FC = () => {
  // 状态管理
  const [superGlueTasks, setSuperGlueTasks] = useState<SuperGlueTask[]>([]);
  const [benchmarkConfigs, setBenchmarkConfigs] = useState<SuperGlueBenchmarkConfig[]>([]);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('tasks');

  // 模态框状态
  const [isTaskModalVisible, setIsTaskModalVisible] = useState(false);
  const [isConfigModalVisible, setIsConfigModalVisible] = useState(false);
  const [editingTask, setEditingTask] = useState<SuperGlueTask | null>(null);
  const [editingConfig, setEditingConfig] = useState<SuperGlueBenchmarkConfig | null>(null);

  // 表单
  const [taskForm] = Form.useForm();
  const [configForm] = Form.useForm();

  // 模拟数据加载
  useEffect(() => {
    loadMockData();
  }, []);

  const loadMockData = () => {
    const mockSuperGlueTasks: SuperGlueTask[] = [
      {
        id: '1',
        name: 'BoolQ',
        fullName: 'Boolean Questions',
        description: '阅读理解：根据段落回答是/否问题',
        category: 'reading_comprehension',
        metrics: ['accuracy'],
        humanPerformance: 0.89,
        samples: {
          train: 9427,
          validation: 3270,
          test: 3245,
        },
        status: 'active',
        difficulty: 'hard',
        dataFormat: 'JSONL',
        lastUpdated: '2024-01-15T10:00:00',
      },
      {
        id: '2',
        name: 'CB',
        fullName: 'CommitmentBank',
        description: '自然语言推理：判断句子间的逻辑关系',
        category: 'natural_language_inference',
        metrics: ['accuracy', 'f1_score'],
        humanPerformance: 0.95,
        samples: {
          train: 250,
          validation: 56,
          test: 250,
        },
        status: 'active',
        difficulty: 'very_hard',
        dataFormat: 'JSONL',
        lastUpdated: '2024-01-15T10:00:00',
      },
      {
        id: '3',
        name: 'COPA',
        fullName: 'Choice of Plausible Alternatives',
        description: '因果推理：选择最合理的原因或结果',
        category: 'reasoning',
        metrics: ['accuracy'],
        humanPerformance: 1.0,
        samples: {
          train: 400,
          validation: 100,
          test: 500,
        },
        status: 'active',
        difficulty: 'hard',
        dataFormat: 'JSONL',
        lastUpdated: '2024-01-15T10:00:00',
      },
      {
        id: '4',
        name: 'MultiRC',
        fullName: 'Multi-Sentence Reading Comprehension',
        description: '多句阅读理解：基于段落回答多个问题',
        category: 'reading_comprehension',
        metrics: ['f1a', 'em'],
        humanPerformance: 0.91,
        samples: {
          train: 5100,
          validation: 953,
          test: 1800,
        },
        status: 'active',
        difficulty: 'very_hard',
        dataFormat: 'JSONL',
        lastUpdated: '2024-01-15T10:00:00',
      },
      {
        id: '5',
        name: 'ReCoRD',
        fullName: 'Reading Comprehension with Commonsense Reasoning Dataset',
        description: '常识阅读理解：填入缺失的命名实体',
        category: 'reading_comprehension',
        metrics: ['f1', 'em'],
        humanPerformance: 0.91,
        samples: {
          train: 100730,
          validation: 10000,
          test: 10000,
        },
        status: 'active',
        difficulty: 'extreme',
        dataFormat: 'JSONL',
        lastUpdated: '2024-01-15T10:00:00',
      },
      {
        id: '6',
        name: 'RTE',
        fullName: 'Recognizing Textual Entailment',
        description: '文本蕴含识别：判断前提是否蕴含假设',
        category: 'natural_language_inference',
        metrics: ['accuracy'],
        humanPerformance: 0.93,
        samples: {
          train: 2490,
          validation: 277,
          test: 3000,
        },
        status: 'active',
        difficulty: 'very_hard',
        dataFormat: 'JSONL',
        lastUpdated: '2024-01-15T10:00:00',
      },
      {
        id: '7',
        name: 'WiC',
        fullName: 'Word-in-Context',
        description: '词义消歧：判断同一单词在不同句子中的含义是否相同',
        category: 'word_sense_disambiguation',
        metrics: ['accuracy'],
        humanPerformance: 0.8,
        samples: {
          train: 5428,
          validation: 638,
          test: 1400,
        },
        status: 'active',
        difficulty: 'hard',
        dataFormat: 'JSONL',
        lastUpdated: '2024-01-15T10:00:00',
      },
      {
        id: '8',
        name: 'WSC',
        fullName: 'Winograd Schema Challenge',
        description: '共指消解：识别代词的指代对象',
        category: 'coreference_resolution',
        metrics: ['accuracy'],
        humanPerformance: 1.0,
        samples: {
          train: 554,
          validation: 104,
          test: 146,
        },
        status: 'active',
        difficulty: 'extreme',
        dataFormat: 'JSONL',
        lastUpdated: '2024-01-15T10:00:00',
      },
    ];

    const mockBenchmarkConfigs: SuperGlueBenchmarkConfig[] = [
      {
        id: '1',
        name: 'SuperGLUE完整基准测试',
        description: '包含所有SuperGLUE任务的完整评估配置',
        selectedTasks: ['BoolQ', 'CB', 'COPA', 'MultiRC', 'ReCoRD', 'RTE', 'WiC', 'WSC'],
        status: 'completed',
        createdAt: '2024-01-10T10:00:00',
        completedAt: '2024-01-10T16:30:00',
        overallScore: 0.724,
        results: [
          { taskId: '1', taskName: 'BoolQ', score: 0.798, humanPerformance: 0.89, humanParity: 89.7, confidence: 0.95 },
          { taskId: '2', taskName: 'CB', score: 0.875, humanPerformance: 0.95, humanParity: 92.1, confidence: 0.88 },
          { taskId: '3', taskName: 'COPA', score: 0.91, humanPerformance: 1.0, humanParity: 91.0, confidence: 0.92 },
          { taskId: '4', taskName: 'MultiRC', score: 0.685, humanPerformance: 0.91, humanParity: 75.3, confidence: 0.89 },
          { taskId: '5', taskName: 'ReCoRD', score: 0.742, humanPerformance: 0.91, humanParity: 81.5, confidence: 0.93 },
          { taskId: '6', taskName: 'RTE', score: 0.753, humanPerformance: 0.93, humanParity: 81.0, confidence: 0.87 },
          { taskId: '7', taskName: 'WiC', score: 0.668, humanPerformance: 0.8, humanParity: 83.5, confidence: 0.91 },
          { taskId: '8', taskName: 'WSC', score: 0.651, humanPerformance: 1.0, humanParity: 65.1, confidence: 0.84 },
        ],
      },
      {
        id: '2',
        name: '推理任务专项测试',
        description: '专注于推理能力的任务组合',
        selectedTasks: ['COPA', 'MultiRC', 'WSC'],
        status: 'running',
        createdAt: '2024-01-18T14:30:00',
      },
    ];

    setSuperGlueTasks(mockSuperGlueTasks);
    setBenchmarkConfigs(mockBenchmarkConfigs);
  };

  // 任务操作
  const handleCreateTask = () => {
    setEditingTask(null);
    taskForm.resetFields();
    setIsTaskModalVisible(true);
  };

  const handleEditTask = (task: SuperGlueTask) => {
    setEditingTask(task);
    taskForm.setFieldsValue(task);
    setIsTaskModalVisible(true);
  };

  const handleDeleteTask = (id: string) => {
    setSuperGlueTasks(superGlueTasks.filter(task => task.id !== id));
    message.success('任务已删除');
  };

  const handleToggleTask = (id: string, status: SuperGlueTask['status']) => {
    setSuperGlueTasks(superGlueTasks.map(task => 
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
        setSuperGlueTasks(superGlueTasks.map(task => 
          task.id === editingTask.id 
            ? { ...task, ...values, lastUpdated: new Date().toISOString() }
            : task
        ));
        message.success('任务已更新');
      } else {
        const newTask: SuperGlueTask = {
          ...values,
          id: Date.now().toString(),
          lastUpdated: new Date().toISOString(),
        };
        setSuperGlueTasks([...superGlueTasks, newTask]);
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
    message.success('SuperGLUE基准测试已启动');
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
        const newConfig: SuperGlueBenchmarkConfig = {
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
  const taskColumns: ColumnsType<SuperGlueTask> = [
    {
      title: '任务名称',
      key: 'name',
      render: (record: SuperGlueTask) => (
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
          reading_comprehension: 'blue',
          natural_language_inference: 'green',
          word_sense_disambiguation: 'purple',
          coreference_resolution: 'orange',
          reasoning: 'red',
        };
        const labels = {
          reading_comprehension: '阅读理解',
          natural_language_inference: '自然语言推理',
          word_sense_disambiguation: '词义消歧',
          coreference_resolution: '共指消解',
          reasoning: '推理任务',
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
          hard: 'orange',
          very_hard: 'red',
          extreme: 'black',
        };
        const labels = {
          hard: '困难',
          very_hard: '非常困难',
          extreme: '极度困难',
        };
        return (
          <Tag color={colors[difficulty as keyof typeof colors]}>
            {labels[difficulty as keyof typeof labels]}
          </Tag>
        );
      },
    },
    {
      title: '人类表现',
      dataIndex: 'humanPerformance',
      key: 'humanPerformance',
      render: (performance: number) => (
        <div>
          <Progress 
            percent={performance * 100}
            size="small"
            format={(percent) => `${percent?.toFixed(1)}%`}
          />
        </div>
      ),
    },
    {
      title: '数据量',
      key: 'samples',
      render: (record: SuperGlueTask) => (
        <div style={{ fontSize: '12px' }}>
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
          experimental: 'blue',
        };
        const labels = {
          active: '启用',
          disabled: '禁用',
          experimental: '实验性',
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
      render: (record: SuperGlueTask) => (
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

  const configColumns: ColumnsType<SuperGlueBenchmarkConfig> = [
    {
      title: '配置名称',
      key: 'name',
      render: (record: SuperGlueBenchmarkConfig) => (
        <div>
          <div style={{ fontWeight: 500 }}>
            {record.name}
            {record.overallScore && (
              <Badge 
                count={`${(record.overallScore * 100).toFixed(1)}%`} 
                style={{ backgroundColor: '#52c41a', marginLeft: '8px' }}
              />
            )}
          </div>
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
          {tasks.slice(0, 4).map(task => (
            <Tag key={task} size="small" style={{ marginBottom: '2px' }}>
              {task}
            </Tag>
          ))}
          {tasks.length > 4 && (
            <Tag size="small">+{tasks.length - 4}个更多</Tag>
          )}
        </div>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string, record: SuperGlueBenchmarkConfig) => {
        const colors = {
          draft: 'default',
          active: 'blue',
          running: 'orange',
          completed: 'green',
          failed: 'red',
        };
        const labels = {
          draft: '草稿',
          active: '就绪',
          running: '运行中',
          completed: '已完成',
          failed: '失败',
        };
        
        return (
          <div>
            <Tag color={colors[status as keyof typeof colors]}>
              {labels[status as keyof typeof labels]}
            </Tag>
            {status === 'running' && (
              <Progress percent={45} size="small" style={{ width: '60px', marginTop: '4px' }} />
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
      render: (record: SuperGlueBenchmarkConfig) => (
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
        <h1><RocketOutlined /> SuperGLUE基准测试管理</h1>
        <p>管理SuperGLUE（Super General Language Understanding Evaluation）基准测试任务和配置</p>
      </div>

      <Alert
        message="SuperGLUE基准测试说明"
        description="SuperGLUE是GLUE的后继者，包含8个更具挑战性的自然语言理解任务。这些任务设计得比GLUE更困难，旨在推动语言理解技术的发展。"
        type="info"
        showIcon
        style={{ marginBottom: '16px' }}
      />

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="SuperGLUE任务管理" key="tasks">
          <div style={{ marginBottom: '16px' }}>
            <Space>
              <Button 
                type="primary" 
                icon={<PlusOutlined />}
                onClick={handleCreateTask}
              >
                新增SuperGLUE任务
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
            dataSource={superGlueTasks}
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
                        <div style={{ marginBottom: '16px' }}>
                          <h4>评估结果</h4>
                          <div style={{ fontSize: '14px', marginBottom: '8px' }}>
                            整体得分: <span style={{ fontSize: '18px', fontWeight: 'bold', color: '#52c41a' }}>
                              {(record.overallScore! * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                        <Row gutter={16}>
                          {record.results.map(result => (
                            <Col span={6} key={result.taskId} style={{ marginBottom: '16px' }}>
                              <Card size="small">
                                <div style={{ textAlign: 'center' }}>
                                  <div style={{ fontWeight: 500, marginBottom: '8px' }}>
                                    {result.taskName}
                                  </div>
                                  <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#1890ff', marginBottom: '8px' }}>
                                    {(result.score * 100).toFixed(1)}%
                                  </div>
                                  <div style={{ fontSize: '12px', color: '#666', marginBottom: '8px' }}>
                                    人类: {(result.humanPerformance * 100).toFixed(1)}%
                                  </div>
                                  <Progress 
                                    percent={result.humanParity}
                                    size="small"
                                    format={(percent) => `${percent?.toFixed(0)}%人类水平`}
                                  />
                                </div>
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
                  value={superGlueTasks.length}
                  suffix="个"
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="启用任务"
                  value={superGlueTasks.filter(t => t.status === 'active').length}
                  suffix="个"
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="平均人类表现"
                  value={
                    superGlueTasks.reduce((sum, task) => sum + task.humanPerformance, 0) / 
                    superGlueTasks.length * 100
                  }
                  precision={1}
                  suffix="%"
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
            <Col span={8}>
              <Card title="任务分类分布">
                <div>
                  <div>阅读理解: {superGlueTasks.filter(t => t.category === 'reading_comprehension').length}个</div>
                  <div>自然语言推理: {superGlueTasks.filter(t => t.category === 'natural_language_inference').length}个</div>
                  <div>词义消歧: {superGlueTasks.filter(t => t.category === 'word_sense_disambiguation').length}个</div>
                  <div>共指消解: {superGlueTasks.filter(t => t.category === 'coreference_resolution').length}个</div>
                  <div>推理任务: {superGlueTasks.filter(t => t.category === 'reasoning').length}个</div>
                </div>
              </Card>
            </Col>
            <Col span={8}>
              <Card title="难度分布">
                <div>
                  <div>困难: {superGlueTasks.filter(t => t.difficulty === 'hard').length}个</div>
                  <div>非常困难: {superGlueTasks.filter(t => t.difficulty === 'very_hard').length}个</div>
                  <div>极度困难: {superGlueTasks.filter(t => t.difficulty === 'extreme').length}个</div>
                </div>
              </Card>
            </Col>
            <Col span={8}>
              <Card title="最具挑战性任务">
                <div>
                  {superGlueTasks
                    .sort((a, b) => a.humanPerformance - b.humanPerformance)
                    .slice(0, 3)
                    .map(task => (
                      <div key={task.id} style={{ marginBottom: '4px' }}>
                        {task.name}: {(task.humanPerformance * 100).toFixed(1)}%
                      </div>
                    ))
                  }
                </div>
              </Card>
            </Col>
          </Row>
        </TabPane>
      </Tabs>

      {/* 任务编辑模态框 */}
      <Modal
        title={editingTask ? "编辑SuperGLUE任务" : "新增SuperGLUE任务"}
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
                <Input placeholder="如: BoolQ, CB" />
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
                  <Option value="reading_comprehension">阅读理解</Option>
                  <Option value="natural_language_inference">自然语言推理</Option>
                  <Option value="word_sense_disambiguation">词义消歧</Option>
                  <Option value="coreference_resolution">共指消解</Option>
                  <Option value="reasoning">推理任务</Option>
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
                  <Option value="hard">困难</Option>
                  <Option value="very_hard">非常困难</Option>
                  <Option value="extreme">极度困难</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item
                name="humanPerformance"
                label="人类表现"
                rules={[{ required: true, message: '请输入人类表现水平' }]}
              >
                <Input placeholder="0.0-1.0" type="number" min={0} max={1} step={0.01} />
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
              <Option value="f1a">F1a分数</Option>
              <Option value="em">完全匹配</Option>
              <Option value="f1">F1分数</Option>
            </Select>
          </Form.Item>
        </Form>
      </Modal>

      {/* 基准配置模态框 */}
      <Modal
        title="创建SuperGLUE基准配置"
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
              placeholder="选择要包含的SuperGLUE任务"
            >
              {superGlueTasks.filter(task => task.status === 'active').map(task => (
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

export default BenchmarkSupergluePage;