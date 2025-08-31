import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Table, Tag, Space, Typography, Progress, Button, Drawer, Descriptions, Timeline, Alert, Select, DatePicker, Input, Statistic, Tabs } from 'antd';
import { 
  PlayCircleOutlined, 
  PauseCircleOutlined, 
  StopOutlined,
  EyeOutlined,
  DeleteOutlined,
  ReloadOutlined,
  FilterOutlined,
  SearchOutlined,
  DownloadOutlined,
  ExclamationCircleOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  SyncOutlined,
  ThunderboltOutlined,
  DatabaseOutlined,
  MonitorOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;
const { RangePicker } = DatePicker;
const { Option } = Select;
const { TabPane } = Tabs;

interface EvaluationTask {
  id: string;
  name: string;
  modelName: string;
  modelPath: string;
  benchmark: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  priority: 'low' | 'medium' | 'high' | 'urgent';
  progress: number;
  currentStep: string;
  startTime: string;
  endTime?: string;
  duration?: number;
  estimatedCompletion: string;
  engineId: string;
  engineName: string;
  resourceUsage: {
    cpu: number;
    memory: number;
    gpu: number;
  };
  config: {
    batchSize: number;
    maxSamples: number;
    metrics: string[];
    timeout: number;
  };
  results?: {
    accuracy?: number;
    f1?: number;
    bleu?: number;
  };
  logs: LogEntry[];
  errors?: string[];
}

interface LogEntry {
  timestamp: string;
  level: 'info' | 'warning' | 'error' | 'debug';
  message: string;
  details?: string;
}

const EvaluationTasksMonitorPage: React.FC = () => {
  const [tasks, setTasks] = useState<EvaluationTask[]>([]);
  const [loading, setLoading] = useState(true);
  const [detailDrawerVisible, setDetailDrawerVisible] = useState(false);
  const [selectedTask, setSelectedTask] = useState<EvaluationTask | null>(null);
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [filterEngine, setFilterEngine] = useState<string>('all');
  const [searchText, setSearchText] = useState('');

  useEffect(() => {
    loadTasksData();
    const interval = setInterval(loadTasksData, 5000); // 5秒更新一次
    return () => clearInterval(interval);
  }, []);

  const loadTasksData = async () => {
    try {
      setLoading(true);
      
      // 模拟API调用
      const tasksData: EvaluationTask[] = [
        {
          id: 'task_001',
          name: 'BERT模型GLUE评估',
          modelName: 'BERT-Large-Uncased',
          modelPath: '/models/bert-large-uncased',
          benchmark: 'GLUE',
          status: 'running',
          priority: 'high',
          progress: 67,
          currentStep: '正在评估SST-2任务',
          startTime: '2024-01-15 14:25:30',
          estimatedCompletion: '2024-01-15 16:45:00',
          engineId: 'engine_001',
          engineName: 'LM-Eval-Main',
          resourceUsage: {
            cpu: 45,
            memory: 68,
            gpu: 82
          },
          config: {
            batchSize: 16,
            maxSamples: 1000,
            metrics: ['accuracy', 'f1', 'matthews_correlation'],
            timeout: 7200
          },
          results: {
            accuracy: 0.89,
            f1: 0.87
          },
          logs: [
            {
              timestamp: '2024-01-15 14:25:30',
              level: 'info',
              message: '开始评估任务',
              details: '初始化模型和数据集'
            },
            {
              timestamp: '2024-01-15 14:26:45',
              level: 'info',
              message: '开始CoLA任务评估',
              details: '处理8551个样本'
            },
            {
              timestamp: '2024-01-15 14:42:15',
              level: 'info',
              message: 'CoLA任务完成',
              details: 'Accuracy: 0.85, Matthews Correlation: 0.52'
            },
            {
              timestamp: '2024-01-15 14:43:00',
              level: 'info',
              message: '开始SST-2任务评估',
              details: '处理67349个样本'
            }
          ]
        },
        {
          id: 'task_002',
          name: 'GPT-3.5代码生成评估',
          modelName: 'GPT-3.5-Turbo',
          modelPath: '/models/gpt-3.5-turbo',
          benchmark: 'HumanEval',
          status: 'completed',
          priority: 'medium',
          progress: 100,
          currentStep: '评估完成',
          startTime: '2024-01-15 13:10:00',
          endTime: '2024-01-15 14:25:00',
          duration: 4500,
          estimatedCompletion: '2024-01-15 14:30:00',
          engineId: 'engine_002',
          engineName: 'HuggingFace-Evaluator',
          resourceUsage: {
            cpu: 0,
            memory: 15,
            gpu: 0
          },
          config: {
            batchSize: 1,
            maxSamples: 164,
            metrics: ['pass@1', 'pass@10', 'pass@100'],
            timeout: 1800
          },
          results: {
            accuracy: 0.734
          },
          logs: [
            {
              timestamp: '2024-01-15 13:10:00',
              level: 'info',
              message: '开始HumanEval评估',
              details: '加载164个编程问题'
            },
            {
              timestamp: '2024-01-15 14:25:00',
              level: 'info',
              message: '评估完成',
              details: 'Pass@1: 0.734, Pass@10: 0.812, Pass@100: 0.923'
            }
          ]
        },
        {
          id: 'task_003',
          name: 'Claude-3 MMLU评估',
          modelName: 'Claude-3-Sonnet',
          modelPath: '/models/claude-3-sonnet',
          benchmark: 'MMLU',
          status: 'failed',
          priority: 'high',
          progress: 23,
          currentStep: '评估中断',
          startTime: '2024-01-15 15:00:00',
          endTime: '2024-01-15 15:15:00',
          duration: 900,
          estimatedCompletion: '2024-01-15 18:00:00',
          engineId: 'engine_001',
          engineName: 'LM-Eval-Main',
          resourceUsage: {
            cpu: 0,
            memory: 25,
            gpu: 0
          },
          config: {
            batchSize: 4,
            maxSamples: 1000,
            metrics: ['accuracy'],
            timeout: 14400
          },
          logs: [
            {
              timestamp: '2024-01-15 15:00:00',
              level: 'info',
              message: '开始MMLU评估',
              details: '开始评估57个学科领域'
            },
            {
              timestamp: '2024-01-15 15:15:00',
              level: 'error',
              message: '评估失败',
              details: 'API调用超时，连接被重置'
            }
          ],
          errors: [
            'API调用超时: 连接在处理abstract_algebra任务时被重置',
            '模型响应格式错误: 期望JSON格式，收到纯文本',
            '资源不足: GPU内存不足以处理当前批次'
          ]
        },
        {
          id: 'task_004',
          name: '自定义金融模型评估',
          modelName: 'FinanceBERT',
          modelPath: '/models/finance-bert',
          benchmark: 'Custom-Finance',
          status: 'pending',
          priority: 'medium',
          progress: 0,
          currentStep: '等待执行',
          startTime: '2024-01-15 16:00:00',
          estimatedCompletion: '2024-01-15 18:30:00',
          engineId: 'engine_003',
          engineName: 'Custom-Benchmark-Engine',
          resourceUsage: {
            cpu: 0,
            memory: 0,
            gpu: 0
          },
          config: {
            batchSize: 8,
            maxSamples: 500,
            metrics: ['accuracy', 'precision', 'recall', 'f1'],
            timeout: 5400
          },
          logs: [
            {
              timestamp: '2024-01-15 16:00:00',
              level: 'info',
              message: '任务已加入队列',
              details: '等待引擎可用'
            }
          ]
        },
        {
          id: 'task_005',
          name: 'T5模型SuperGLUE评估',
          modelName: 'T5-3B',
          modelPath: '/models/t5-3b',
          benchmark: 'SuperGLUE',
          status: 'cancelled',
          priority: 'low',
          progress: 15,
          currentStep: '已取消',
          startTime: '2024-01-15 12:30:00',
          endTime: '2024-01-15 12:45:00',
          duration: 900,
          estimatedCompletion: '2024-01-15 15:30:00',
          engineId: 'engine_002',
          engineName: 'HuggingFace-Evaluator',
          resourceUsage: {
            cpu: 0,
            memory: 20,
            gpu: 0
          },
          config: {
            batchSize: 8,
            maxSamples: -1,
            metrics: ['accuracy', 'f1'],
            timeout: 10800
          },
          logs: [
            {
              timestamp: '2024-01-15 12:30:00',
              level: 'info',
              message: '开始SuperGLUE评估',
              details: '开始处理8个任务'
            },
            {
              timestamp: '2024-01-15 12:45:00',
              level: 'warning',
              message: '任务被用户取消',
              details: '用户主动取消了评估任务'
            }
          ]
        }
      ];

      setTasks(tasksData);
    } catch (error) {
      console.error('加载任务数据失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleTaskAction = async (taskId: string, action: 'pause' | 'resume' | 'cancel' | 'restart') => {
    try {
      console.log(`${action} task:`, taskId);
      // 在实际项目中调用API
      await loadTasksData();
    } catch (error) {
      console.error(`任务${action}失败:`, error);
    }
  };

  const handleViewDetails = (task: EvaluationTask) => {
    setSelectedTask(task);
    setDetailDrawerVisible(true);
  };

  const getStatusTag = (status: string) => {
    const statusConfig = {
      pending: { color: 'default', icon: <ClockCircleOutlined />, text: '等待中' },
      running: { color: 'processing', icon: <SyncOutlined spin />, text: '运行中' },
      completed: { color: 'success', icon: <CheckCircleOutlined />, text: '已完成' },
      failed: { color: 'error', icon: <CloseCircleOutlined />, text: '失败' },
      cancelled: { color: 'warning', icon: <ExclamationCircleOutlined />, text: '已取消' }
    };
    const config = statusConfig[status as keyof typeof statusConfig];
    return <Tag color={config.color} icon={config.icon}>{config.text}</Tag>;
  };

  const getPriorityTag = (priority: string) => {
    const priorityConfig = {
      low: { color: 'blue', text: '低' },
      medium: { color: 'orange', text: '中' },
      high: { color: 'red', text: '高' },
      urgent: { color: 'magenta', text: '紧急' }
    };
    const config = priorityConfig[priority as keyof typeof priorityConfig];
    return <Tag color={config.color}>{config.text}</Tag>;
  };

  const getLogLevelTag = (level: string) => {
    const levelConfig = {
      info: { color: 'blue', text: 'INFO' },
      warning: { color: 'orange', text: 'WARN' },
      error: { color: 'red', text: 'ERROR' },
      debug: { color: 'default', text: 'DEBUG' }
    };
    const config = levelConfig[level as keyof typeof levelConfig];
    return <Tag color={config.color}>{config.text}</Tag>;
  };

  const filteredTasks = tasks.filter(task => {
    const statusMatch = filterStatus === 'all' || task.status === filterStatus;
    const engineMatch = filterEngine === 'all' || task.engineId === filterEngine;
    const searchMatch = searchText === '' || 
      task.name.toLowerCase().includes(searchText.toLowerCase()) ||
      task.modelName.toLowerCase().includes(searchText.toLowerCase()) ||
      task.benchmark.toLowerCase().includes(searchText.toLowerCase());
    
    return statusMatch && engineMatch && searchMatch;
  });

  const columns = [
    {
      title: '任务名称',
      dataIndex: 'name',
      key: 'name',
      width: 200,
      render: (text: string, record: EvaluationTask) => (
        <div>
          <div style={{ fontWeight: 'bold' }}>{text}</div>
          <Text type="secondary" style={{ fontSize: '12px' }}>{record.id}</Text>
        </div>
      )
    },
    {
      title: '模型',
      dataIndex: 'modelName',
      key: 'modelName',
      width: 150,
    },
    {
      title: '基准测试',
      dataIndex: 'benchmark',
      key: 'benchmark',
      width: 120,
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => getStatusTag(status),
      width: 100,
    },
    {
      title: '优先级',
      dataIndex: 'priority',
      key: 'priority',
      render: (priority: string) => getPriorityTag(priority),
      width: 80,
    },
    {
      title: '进度',
      dataIndex: 'progress',
      key: 'progress',
      render: (progress: number, record: EvaluationTask) => (
        <div>
          <Progress 
            percent={progress} 
            size="small" 
            status={record.status === 'failed' ? 'exception' : 'active'}
          />
          <Text type="secondary" style={{ fontSize: '12px' }}>{record.currentStep}</Text>
        </div>
      ),
      width: 150,
    },
    {
      title: '引擎',
      dataIndex: 'engineName',
      key: 'engineName',
      width: 140,
    },
    {
      title: '资源使用',
      key: 'resources',
      render: (_, record: EvaluationTask) => (
        <Space direction="vertical" size="small">
          <Text style={{ fontSize: '12px' }}>CPU: {record.resourceUsage.cpu}%</Text>
          <Text style={{ fontSize: '12px' }}>GPU: {record.resourceUsage.gpu}%</Text>
          <Text style={{ fontSize: '12px' }}>内存: {record.resourceUsage.memory}%</Text>
        </Space>
      ),
      width: 100,
    },
    {
      title: '开始时间',
      dataIndex: 'startTime',
      key: 'startTime',
      width: 130,
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record: EvaluationTask) => (
        <Space size="small">
          <Button
            type="link"
            size="small"
            icon={<EyeOutlined />}
            onClick={() => handleViewDetails(record)}
          >
            详情
          </Button>
          {record.status === 'running' && (
            <Button
              type="link"
              size="small"
              icon={<PauseCircleOutlined />}
              onClick={() => handleTaskAction(record.id, 'pause')}
            >
              暂停
            </Button>
          )}
          {record.status === 'pending' && (
            <Button
              type="link"
              size="small"
              icon={<PlayCircleOutlined />}
              onClick={() => handleTaskAction(record.id, 'resume')}
            >
              开始
            </Button>
          )}
          {['running', 'pending'].includes(record.status) && (
            <Button
              type="link"
              size="small"
              icon={<StopOutlined />}
              onClick={() => handleTaskAction(record.id, 'cancel')}
              danger
            >
              取消
            </Button>
          )}
          {['failed', 'cancelled'].includes(record.status) && (
            <Button
              type="link"
              size="small"
              icon={<ReloadOutlined />}
              onClick={() => handleTaskAction(record.id, 'restart')}
            >
              重试
            </Button>
          )}
        </Space>
      ),
      width: 160,
    },
  ];

  const runningTasks = tasks.filter(t => t.status === 'running').length;
  const pendingTasks = tasks.filter(t => t.status === 'pending').length;
  const completedTasks = tasks.filter(t => t.status === 'completed').length;
  const failedTasks = tasks.filter(t => t.status === 'failed').length;

  return (
    <div style={{ padding: '24px', background: '#f5f5f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>评估任务监控</Title>
        <Text type="secondary">
          实时监控所有模型评估任务的执行状态、进度和资源使用情况
        </Text>
      </div>

      {/* 任务统计 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="运行中"
              value={runningTasks}
              prefix={<ThunderboltOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="等待中"
              value={pendingTasks}
              prefix={<ClockCircleOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="已完成"
              value={completedTasks}
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="失败"
              value={failedTasks}
              prefix={<CloseCircleOutlined />}
              valueStyle={{ color: '#ff4d4f' }}
            />
          </Card>
        </Col>
      </Row>

      <Card
        title="任务列表"
        extra={
          <Space>
            <Input
              placeholder="搜索任务..."
              prefix={<SearchOutlined />}
              value={searchText}
              onChange={e => setSearchText(e.target.value)}
              style={{ width: 200 }}
            />
            <Select
              placeholder="状态筛选"
              value={filterStatus}
              onChange={setFilterStatus}
              style={{ width: 120 }}
            >
              <Option value="all">全部状态</Option>
              <Option value="pending">等待中</Option>
              <Option value="running">运行中</Option>
              <Option value="completed">已完成</Option>
              <Option value="failed">失败</Option>
              <Option value="cancelled">已取消</Option>
            </Select>
            <Select
              placeholder="引擎筛选"
              value={filterEngine}
              onChange={setFilterEngine}
              style={{ width: 150 }}
            >
              <Option value="all">全部引擎</Option>
              <Option value="engine_001">LM-Eval-Main</Option>
              <Option value="engine_002">HuggingFace-Evaluator</Option>
              <Option value="engine_003">Custom-Benchmark-Engine</Option>
            </Select>
            <Button
              icon={<ReloadOutlined />}
              onClick={loadTasksData}
              loading={loading}
            >
              刷新
            </Button>
          </Space>
        }
      >
        <Table
          dataSource={filteredTasks}
          columns={columns}
          rowKey="id"
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) => `第 ${range[0]}-${range[1]} 项 / 共 ${total} 项`
          }}
          loading={loading}
          scroll={{ x: 1200 }}
        />
      </Card>

      {/* 任务详情抽屉 */}
      <Drawer
        title="任务详情"
        placement="right"
        size="large"
        open={detailDrawerVisible}
        onClose={() => setDetailDrawerVisible(false)}
      >
        {selectedTask && (
          <div>
            <Tabs defaultActiveKey="overview">
              <TabPane tab="概览" key="overview">
                <Descriptions bordered column={2} style={{ marginBottom: '24px' }}>
                  <Descriptions.Item label="任务名称" span={2}>{selectedTask.name}</Descriptions.Item>
                  <Descriptions.Item label="任务ID">{selectedTask.id}</Descriptions.Item>
                  <Descriptions.Item label="状态">{getStatusTag(selectedTask.status)}</Descriptions.Item>
                  <Descriptions.Item label="模型名称">{selectedTask.modelName}</Descriptions.Item>
                  <Descriptions.Item label="基准测试">{selectedTask.benchmark}</Descriptions.Item>
                  <Descriptions.Item label="优先级">{getPriorityTag(selectedTask.priority)}</Descriptions.Item>
                  <Descriptions.Item label="执行引擎">{selectedTask.engineName}</Descriptions.Item>
                  <Descriptions.Item label="进度">{selectedTask.progress}%</Descriptions.Item>
                  <Descriptions.Item label="当前步骤" span={2}>{selectedTask.currentStep}</Descriptions.Item>
                  <Descriptions.Item label="开始时间">{selectedTask.startTime}</Descriptions.Item>
                  <Descriptions.Item label="结束时间">{selectedTask.endTime || '进行中'}</Descriptions.Item>
                  <Descriptions.Item label="持续时间">
                    {selectedTask.duration ? `${Math.floor(selectedTask.duration / 60)}分钟` : '计算中'}
                  </Descriptions.Item>
                  <Descriptions.Item label="预计完成">{selectedTask.estimatedCompletion}</Descriptions.Item>
                </Descriptions>

                <Card title="配置信息" size="small" style={{ marginBottom: '16px' }}>
                  <Descriptions size="small" column={2}>
                    <Descriptions.Item label="批处理大小">{selectedTask.config.batchSize}</Descriptions.Item>
                    <Descriptions.Item label="最大样本数">{selectedTask.config.maxSamples === -1 ? '无限制' : selectedTask.config.maxSamples}</Descriptions.Item>
                    <Descriptions.Item label="超时时间">{selectedTask.config.timeout}秒</Descriptions.Item>
                    <Descriptions.Item label="评估指标">
                      <Space size={[0, 4]} wrap>
                        {selectedTask.config.metrics.map(metric => <Tag key={metric}>{metric}</Tag>)}
                      </Space>
                    </Descriptions.Item>
                  </Descriptions>
                </Card>

                {selectedTask.results && (
                  <Card title="评估结果" size="small">
                    <Row gutter={16}>
                      {Object.entries(selectedTask.results).map(([key, value]) => (
                        <Col span={8} key={key}>
                          <Statistic 
                            title={key.toUpperCase()} 
                            value={value} 
                            precision={3}
                            suffix={key === 'accuracy' || key === 'f1' ? '' : ''}
                          />
                        </Col>
                      ))}
                    </Row>
                  </Card>
                )}
              </TabPane>

              <TabPane tab="实时日志" key="logs">
                <Timeline mode="left">
                  {selectedTask.logs.map((log, index) => (
                    <Timeline.Item
                      key={index}
                      color={log.level === 'error' ? 'red' : log.level === 'warning' ? 'orange' : 'blue'}
                      label={log.timestamp}
                    >
                      <div>
                        {getLogLevelTag(log.level)} {log.message}
                        {log.details && (
                          <div style={{ marginTop: '4px', color: '#666', fontSize: '12px' }}>
                            {log.details}
                          </div>
                        )}
                      </div>
                    </Timeline.Item>
                  ))}
                </Timeline>
              </TabPane>

              <TabPane tab="资源监控" key="resources">
                <Row gutter={[16, 16]}>
                  <Col span={8}>
                    <Card>
                      <Statistic
                        title="CPU使用率"
                        value={selectedTask.resourceUsage.cpu}
                        suffix="%"
                        valueStyle={{ color: selectedTask.resourceUsage.cpu > 80 ? '#ff4d4f' : '#52c41a' }}
                      />
                    </Card>
                  </Col>
                  <Col span={8}>
                    <Card>
                      <Statistic
                        title="内存使用率"
                        value={selectedTask.resourceUsage.memory}
                        suffix="%"
                        valueStyle={{ color: selectedTask.resourceUsage.memory > 85 ? '#ff4d4f' : '#1890ff' }}
                      />
                    </Card>
                  </Col>
                  <Col span={8}>
                    <Card>
                      <Statistic
                        title="GPU使用率"
                        value={selectedTask.resourceUsage.gpu}
                        suffix="%"
                        valueStyle={{ color: selectedTask.resourceUsage.gpu > 90 ? '#ff4d4f' : '#722ed1' }}
                      />
                    </Card>
                  </Col>
                </Row>
              </TabPane>

              {selectedTask.errors && selectedTask.errors.length > 0 && (
                <TabPane tab="错误信息" key="errors">
                  <Space direction="vertical" style={{ width: '100%' }}>
                    {selectedTask.errors.map((error, index) => (
                      <Alert
                        key={index}
                        type="error"
                        showIcon
                        message={`错误 ${index + 1}`}
                        description={error}
                      />
                    ))}
                  </Space>
                </TabPane>
              )}
            </Tabs>
          </div>
        )}
      </Drawer>
    </div>
  );
};

export default EvaluationTasksMonitorPage;