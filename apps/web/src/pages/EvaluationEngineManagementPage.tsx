import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Button, Table, Tag, Space, Typography, Modal, Form, Input, Select, Progress, Alert, Tabs, Statistic, Switch } from 'antd';
import { 
  PlayCircleOutlined, 
  PauseCircleOutlined, 
  SettingOutlined, 
  DeleteOutlined,
  PlusOutlined,
  ReloadOutlined,
  ExclamationCircleOutlined,
  ThunderboltOutlined,
  DatabaseOutlined,
  CloudServerOutlined,
  MonitorOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;
const { Option } = Select;
const { TabPane } = Tabs;

interface EvaluationEngine {
  id: string;
  name: string;
  status: 'running' | 'stopped' | 'error' | 'configuring';
  type: 'lm-eval' | 'huggingface' | 'custom';
  currentTasks: number;
  maxConcurrency: number;
  cpuUsage: number;
  memoryUsage: number;
  gpu: string;
  uptime: string;
  lastUpdated: string;
  config: {
    batchSize: number;
    device: string;
    precision: string;
    timeout: number;
  };
}

interface RunningTask {
  id: string;
  engineId: string;
  modelName: string;
  benchmark: string;
  progress: number;
  startTime: string;
  estimatedCompletion: string;
  status: 'initializing' | 'running' | 'finishing';
}

const EvaluationEngineManagementPage: React.FC = () => {
  const [engines, setEngines] = useState<EvaluationEngine[]>([]);
  const [runningTasks, setRunningTasks] = useState<RunningTask[]>([]);
  const [loading, setLoading] = useState(true);
  const [configModalVisible, setConfigModalVisible] = useState(false);
  const [selectedEngine, setSelectedEngine] = useState<EvaluationEngine | null>(null);
  const [form] = Form.useForm();

  useEffect(() => {
    loadEngineData();
    const interval = setInterval(loadEngineData, 5000); // 5秒更新一次
    return () => clearInterval(interval);
  }, []);

  const loadEngineData = async () => {
    try {
      setLoading(true);
      
      // 模拟API调用
      const enginesData: EvaluationEngine[] = [
        {
          id: 'engine_001',
          name: 'LM-Eval-Main',
          status: 'running',
          type: 'lm-eval',
          currentTasks: 2,
          maxConcurrency: 4,
          cpuUsage: 45,
          memoryUsage: 68,
          gpu: 'NVIDIA A100',
          uptime: '2天14小时',
          lastUpdated: '2024-01-15 15:30:23',
          config: {
            batchSize: 16,
            device: 'cuda:0',
            precision: 'float16',
            timeout: 3600
          }
        },
        {
          id: 'engine_002',
          name: 'HuggingFace-Evaluator',
          status: 'running',
          type: 'huggingface',
          currentTasks: 1,
          maxConcurrency: 2,
          cpuUsage: 32,
          memoryUsage: 42,
          gpu: 'NVIDIA V100',
          uptime: '1天8小时',
          lastUpdated: '2024-01-15 15:30:18',
          config: {
            batchSize: 8,
            device: 'cuda:1',
            precision: 'float32',
            timeout: 7200
          }
        },
        {
          id: 'engine_003',
          name: 'Custom-Benchmark-Engine',
          status: 'stopped',
          type: 'custom',
          currentTasks: 0,
          maxConcurrency: 1,
          cpuUsage: 0,
          memoryUsage: 15,
          gpu: 'CPU Only',
          uptime: '已停止',
          lastUpdated: '2024-01-15 12:45:10',
          config: {
            batchSize: 4,
            device: 'cpu',
            precision: 'float32',
            timeout: 1800
          }
        },
        {
          id: 'engine_004',
          name: 'GPU-Cluster-Engine',
          status: 'error',
          type: 'lm-eval',
          currentTasks: 0,
          maxConcurrency: 8,
          cpuUsage: 12,
          memoryUsage: 25,
          gpu: 'Multi-GPU',
          uptime: '错误状态',
          lastUpdated: '2024-01-15 14:22:45',
          config: {
            batchSize: 32,
            device: 'cuda',
            precision: 'float16',
            timeout: 5400
          }
        }
      ];

      const tasksData: RunningTask[] = [
        {
          id: 'task_001',
          engineId: 'engine_001',
          modelName: 'BERT-Large-Uncased',
          benchmark: 'GLUE',
          progress: 65,
          startTime: '14:25:30',
          estimatedCompletion: '15:45:00',
          status: 'running'
        },
        {
          id: 'task_002',
          engineId: 'engine_001',
          modelName: 'GPT-3.5-Turbo',
          benchmark: 'MMLU',
          progress: 23,
          startTime: '15:10:15',
          estimatedCompletion: '16:30:00',
          status: 'running'
        },
        {
          id: 'task_003',
          engineId: 'engine_002',
          modelName: 'T5-Base',
          benchmark: 'SuperGLUE',
          progress: 89,
          startTime: '13:30:00',
          estimatedCompletion: '15:35:00',
          status: 'finishing'
        }
      ];

      setEngines(enginesData);
      setRunningTasks(tasksData);
    } catch (error) {
      console.error('加载引擎数据失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleStartEngine = async (engineId: string) => {
    try {
      console.log('启动引擎:', engineId);
      // 在实际项目中调用API启动引擎
      await loadEngineData();
    } catch (error) {
      console.error('启动引擎失败:', error);
    }
  };

  const handleStopEngine = async (engineId: string) => {
    Modal.confirm({
      title: '确认停止引擎',
      icon: <ExclamationCircleOutlined />,
      content: '停止引擎将会中断所有正在运行的评估任务，确定要继续吗？',
      okText: '确定',
      cancelText: '取消',
      onOk: async () => {
        try {
          console.log('停止引擎:', engineId);
          // 在实际项目中调用API停止引擎
          await loadEngineData();
        } catch (error) {
          console.error('停止引擎失败:', error);
        }
      }
    });
  };

  const handleConfigEngine = (engine: EvaluationEngine) => {
    setSelectedEngine(engine);
    form.setFieldsValue({
      name: engine.name,
      maxConcurrency: engine.maxConcurrency,
      batchSize: engine.config.batchSize,
      device: engine.config.device,
      precision: engine.config.precision,
      timeout: engine.config.timeout
    });
    setConfigModalVisible(true);
  };

  const handleSaveConfig = async () => {
    try {
      const values = await form.validateFields();
      console.log('保存配置:', values);
      // 在实际项目中调用API保存配置
      setConfigModalVisible(false);
      await loadEngineData();
    } catch (error) {
      console.error('保存配置失败:', error);
    }
  };

  const getStatusTag = (status: string) => {
    const statusConfig = {
      running: { color: 'success', text: '运行中' },
      stopped: { color: 'default', text: '已停止' },
      error: { color: 'error', text: '错误' },
      configuring: { color: 'processing', text: '配置中' }
    };
    const config = statusConfig[status as keyof typeof statusConfig];
    return <Tag color={config.color}>{config.text}</Tag>;
  };

  const getTypeTag = (type: string) => {
    const typeConfig = {
      'lm-eval': { color: 'blue', text: 'LM-Eval' },
      'huggingface': { color: 'orange', text: 'HuggingFace' },
      'custom': { color: 'purple', text: '自定义' }
    };
    const config = typeConfig[type as keyof typeof typeConfig];
    return <Tag color={config.color}>{config.text}</Tag>;
  };

  const getTaskStatusTag = (status: string) => {
    const statusConfig = {
      initializing: { color: 'processing', text: '初始化' },
      running: { color: 'success', text: '运行中' },
      finishing: { color: 'warning', text: '收尾中' }
    };
    const config = statusConfig[status as keyof typeof statusConfig];
    return <Tag color={config.color}>{config.text}</Tag>;
  };

  const engineColumns = [
    {
      title: '引擎名称',
      dataIndex: 'name',
      key: 'name',
      width: 180,
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => getTypeTag(type),
      width: 100,
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => getStatusTag(status),
      width: 100,
    },
    {
      title: '任务进度',
      key: 'tasks',
      render: (_, record: EvaluationEngine) => (
        <Text>{record.currentTasks} / {record.maxConcurrency}</Text>
      ),
      width: 100,
    },
    {
      title: 'GPU设备',
      dataIndex: 'gpu',
      key: 'gpu',
      width: 140,
    },
    {
      title: '资源使用',
      key: 'resources',
      render: (_, record: EvaluationEngine) => (
        <Space direction="vertical" size="small" style={{ width: '100%' }}>
          <div>
            <Text type="secondary">CPU: </Text>
            <Progress 
              percent={record.cpuUsage} 
              size="small" 
              showInfo={false}
              strokeColor={record.cpuUsage > 80 ? '#ff4d4f' : '#52c41a'}
            />
            <Text style={{ fontSize: '12px', marginLeft: '8px' }}>{record.cpuUsage}%</Text>
          </div>
          <div>
            <Text type="secondary">内存: </Text>
            <Progress 
              percent={record.memoryUsage} 
              size="small" 
              showInfo={false}
              strokeColor={record.memoryUsage > 85 ? '#ff4d4f' : '#1890ff'}
            />
            <Text style={{ fontSize: '12px', marginLeft: '8px' }}>{record.memoryUsage}%</Text>
          </div>
        </Space>
      ),
      width: 150,
    },
    {
      title: '运行时间',
      dataIndex: 'uptime',
      key: 'uptime',
      width: 120,
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record: EvaluationEngine) => (
        <Space size="small">
          {record.status === 'stopped' || record.status === 'error' ? (
            <Button 
              type="link" 
              size="small" 
              icon={<PlayCircleOutlined />}
              onClick={() => handleStartEngine(record.id)}
            >
              启动
            </Button>
          ) : (
            <Button 
              type="link" 
              size="small" 
              icon={<PauseCircleOutlined />}
              onClick={() => handleStopEngine(record.id)}
              danger
            >
              停止
            </Button>
          )}
          <Button 
            type="link" 
            size="small" 
            icon={<SettingOutlined />}
            onClick={() => handleConfigEngine(record)}
          >
            配置
          </Button>
        </Space>
      ),
      width: 120,
    },
  ];

  const taskColumns = [
    {
      title: '任务ID',
      dataIndex: 'id',
      key: 'id',
      width: 120,
    },
    {
      title: '引擎',
      dataIndex: 'engineId',
      key: 'engineId',
      render: (engineId: string) => {
        const engine = engines.find(e => e.id === engineId);
        return engine ? engine.name : engineId;
      },
      width: 150,
    },
    {
      title: '模型',
      dataIndex: 'modelName',
      key: 'modelName',
      width: 180,
    },
    {
      title: '基准测试',
      dataIndex: 'benchmark',
      key: 'benchmark',
      width: 120,
    },
    {
      title: '进度',
      dataIndex: 'progress',
      key: 'progress',
      render: (progress: number) => (
        <Progress percent={progress} size="small" />
      ),
      width: 150,
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => getTaskStatusTag(status),
      width: 100,
    },
    {
      title: '开始时间',
      dataIndex: 'startTime',
      key: 'startTime',
      width: 120,
    },
    {
      title: '预计完成',
      dataIndex: 'estimatedCompletion',
      key: 'estimatedCompletion',
      width: 120,
    },
  ];

  const totalEngines = engines.length;
  const runningEngines = engines.filter(e => e.status === 'running').length;
  const totalTasks = runningTasks.length;
  const avgProgress = runningTasks.length > 0 
    ? runningTasks.reduce((sum, task) => sum + task.progress, 0) / runningTasks.length 
    : 0;

  return (
    <div style={{ padding: '24px', background: '#f5f5f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>评估引擎管理</Title>
        <Text type="secondary">
          管理和监控所有评估引擎的运行状态、资源使用和任务调度
        </Text>
      </div>

      {/* 引擎状态统计 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="总引擎数"
              value={totalEngines}
              prefix={<DatabaseOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="运行中"
              value={runningEngines}
              prefix={<ThunderboltOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="活跃任务"
              value={totalTasks}
              prefix={<MonitorOutlined />}
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="平均进度"
              value={avgProgress}
              precision={1}
              suffix="%"
              prefix={<CloudServerOutlined />}
              valueStyle={{ color: '#fa8c16' }}
            />
          </Card>
        </Col>
      </Row>

      <Tabs defaultActiveKey="engines">
        <TabPane tab="引擎列表" key="engines">
          <Card 
            title="评估引擎列表" 
            extra={
              <Space>
                <Button 
                  type="primary" 
                  icon={<PlusOutlined />}
                  onClick={() => console.log('添加新引擎')}
                >
                  添加引擎
                </Button>
                <Button 
                  icon={<ReloadOutlined />} 
                  onClick={loadEngineData}
                  loading={loading}
                >
                  刷新
                </Button>
              </Space>
            }
          >
            <Table
              dataSource={engines}
              columns={engineColumns}
              rowKey="id"
              pagination={{ 
                pageSize: 10,
                showSizeChanger: true,
                showQuickJumper: true,
                showTotal: (total, range) => `第 ${range[0]}-${range[1]} 项 / 共 ${total} 项`
              }}
              loading={loading}
            />
          </Card>
        </TabPane>

        <TabPane tab="运行任务" key="tasks">
          <Card 
            title="正在运行的任务" 
            extra={
              <Button 
                icon={<ReloadOutlined />} 
                onClick={loadEngineData}
                loading={loading}
              >
                刷新
              </Button>
            }
          >
            {runningTasks.length === 0 ? (
              <Alert
                message="暂无运行中的任务"
                description="当前没有正在执行的评估任务"
                type="info"
                showIcon
                style={{ margin: '20px 0' }}
              />
            ) : (
              <Table
                dataSource={runningTasks}
                columns={taskColumns}
                rowKey="id"
                pagination={false}
                loading={loading}
              />
            )}
          </Card>
        </TabPane>
      </Tabs>

      {/* 配置模态框 */}
      <Modal
        title={`配置引擎: ${selectedEngine?.name}`}
        open={configModalVisible}
        onOk={handleSaveConfig}
        onCancel={() => setConfigModalVisible(false)}
        width={600}
      >
        <Form form={form} layout="vertical">
          <Form.Item
            name="name"
            label="引擎名称"
            rules={[{ required: true, message: '请输入引擎名称' }]}
          >
            <Input placeholder="请输入引擎名称" />
          </Form.Item>
          
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="maxConcurrency"
                label="最大并发数"
                rules={[{ required: true, message: '请输入最大并发数' }]}
              >
                <Input type="number" min={1} max={16} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="batchSize"
                label="批处理大小"
                rules={[{ required: true, message: '请输入批处理大小' }]}
              >
                <Input type="number" min={1} max={64} />
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="device"
                label="计算设备"
                rules={[{ required: true, message: '请选择计算设备' }]}
              >
                <Select placeholder="请选择计算设备">
                  <Option value="cpu">CPU</Option>
                  <Option value="cuda:0">CUDA:0</Option>
                  <Option value="cuda:1">CUDA:1</Option>
                  <Option value="cuda">Multi-GPU</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="precision"
                label="计算精度"
                rules={[{ required: true, message: '请选择计算精度' }]}
              >
                <Select placeholder="请选择计算精度">
                  <Option value="float32">Float32</Option>
                  <Option value="float16">Float16</Option>
                  <Option value="bfloat16">BFloat16</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>

          <Form.Item
            name="timeout"
            label="超时时间(秒)"
            rules={[{ required: true, message: '请输入超时时间' }]}
          >
            <Input type="number" min={300} max={21600} />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default EvaluationEngineManagementPage;