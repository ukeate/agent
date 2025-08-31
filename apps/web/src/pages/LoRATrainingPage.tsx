import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Row, 
  Col, 
  Typography, 
  Progress, 
  Statistic, 
  Table, 
  Button,
  Space,
  Tag,
  Tabs,
  Alert,
  Select,
  Input,
  Form,
  Slider,
  Switch,
  Tooltip,
  Badge,
  Timeline,
  Descriptions
} from 'antd';
import {
  GoldOutlined,
  ThunderboltOutlined,
  DashboardOutlined,
  LineChartOutlined,
  SettingOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  MonitorOutlined,
  DatabaseOutlined,
  CheckCircleOutlined,
  InfoCircleOutlined,
  FireOutlined,
  DesktopOutlined,
  ClockCircleOutlined
} from '@ant-design/icons';
import { Line, Bar, Gauge } from '@ant-design/charts';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { TextArea } = Input;
const { Option } = Select;

// 模拟LoRA训练数据
const generateMockData = () => ({
  trainingMetrics: [
    { epoch: 0, step: 0, trainLoss: 2.3456, evalLoss: 2.4123, learningRate: 2e-4 },
    { epoch: 0, step: 100, trainLoss: 1.8923, evalLoss: 1.9234, learningRate: 1.98e-4 },
    { epoch: 0, step: 200, trainLoss: 1.5643, evalLoss: 1.6234, learningRate: 1.96e-4 },
    { epoch: 1, step: 300, trainLoss: 1.3456, evalLoss: 1.4123, learningRate: 1.94e-4 },
    { epoch: 1, step: 400, trainLoss: 1.1789, evalLoss: 1.2345, learningRate: 1.92e-4 },
    { epoch: 2, step: 500, trainLoss: 0.9876, evalLoss: 1.0987, learningRate: 1.90e-4 },
  ],
  modelInfo: {
    baseModel: 'meta-llama/Llama-2-7b-chat-hf',
    modelSize: '7B参数',
    loraRank: 16,
    loraAlpha: 32,
    loraDropout: 0.1,
    targetModules: ['q_proj', 'v_proj', 'k_proj', 'o_proj'],
    trainableParams: '4,194,304',
    totalParams: '6,738,415,616',
    trainablePercentage: 0.062
  },
  currentStatus: {
    status: 'running',
    currentEpoch: 2,
    totalEpochs: 5,
    currentStep: 567,
    totalSteps: 1500,
    progress: 37.8,
    currentLoss: 0.8743,
    bestLoss: 0.8234,
    learningRate: 1.89e-4,
    gpuUtilization: 87,
    memoryUsage: 18.5,
    totalMemory: 24.0,
    trainingTime: '02:34:15',
    estimatedRemaining: '04:25:30'
  },
  loraLayers: [
    { name: 'model.layers.0.self_attn.q_proj', rank: 16, alpha: 32, trainable: true },
    { name: 'model.layers.0.self_attn.k_proj', rank: 16, alpha: 32, trainable: true },
    { name: 'model.layers.0.self_attn.v_proj', rank: 16, alpha: 32, trainable: true },
    { name: 'model.layers.0.self_attn.o_proj', rank: 16, alpha: 32, trainable: true },
    { name: 'model.layers.1.self_attn.q_proj', rank: 16, alpha: 32, trainable: true },
    { name: 'model.layers.1.self_attn.k_proj', rank: 16, alpha: 32, trainable: true },
  ]
});

const LoRATrainingPage: React.FC = () => {
  const [mockData] = useState(generateMockData());
  const [activeTab, setActiveTab] = useState('overview');
  const [refreshing, setRefreshing] = useState(false);

  // 训练损失图表配置
  const lossChartConfig = {
    data: mockData.trainingMetrics,
    xField: 'step',
    yField: 'trainLoss',
    seriesField: 'type',
    smooth: true,
    animation: {
      appear: {
        animation: 'path-in',
        duration: 1000,
      },
    },
    meta: {
      trainLoss: {
        alias: '训练损失',
      },
      evalLoss: {
        alias: '验证损失',
      },
    },
    color: ['#1890ff', '#52c41a'],
  };

  // 学习率图表配置
  const learningRateConfig = {
    data: mockData.trainingMetrics,
    xField: 'step',
    yField: 'learningRate',
    smooth: true,
    color: '#722ed1',
    point: {
      size: 3,
      shape: 'circle',
    },
  };

  // GPU使用率仪表盘配置
  const gpuGaugeConfig = {
    percent: mockData.currentStatus.gpuUtilization / 100,
    type: 'meter',
    innerRadius: 0.75,
    range: {
      ticks: [0, 1/3, 2/3, 1],
      color: ['#30BF78', '#FAAD14', '#F4664A'],
    },
    indicator: {
      pointer: {
        style: {
          stroke: '#D0D0D0',
        },
      },
      pin: {
        style: {
          stroke: '#D0D0D0',
        },
      },
    },
    statistic: {
      title: {
        formatter: () => 'GPU使用率',
        style: {
          color: '#363636',
          fontSize: '12px',
        },
      },
      content: {
        style: {
          color: '#4B535E',
          fontSize: '20px',
        },
        formatter: () => `${mockData.currentStatus.gpuUtilization}%`,
      },
    },
  };

  // 内存使用率仪表盘配置
  const memoryGaugeConfig = {
    ...gpuGaugeConfig,
    percent: mockData.currentStatus.memoryUsage / mockData.currentStatus.totalMemory,
    statistic: {
      title: {
        formatter: () => '显存使用率',
        style: {
          color: '#363636',
          fontSize: '12px',
        },
      },
      content: {
        style: {
          color: '#4B535E',
          fontSize: '20px',
        },
        formatter: () => `${Math.round((mockData.currentStatus.memoryUsage / mockData.currentStatus.totalMemory) * 100)}%`,
      },
    },
  };

  // LoRA层表格列定义
  const loraLayerColumns = [
    {
      title: '层名称',
      dataIndex: 'name',
      key: 'name',
      width: 300,
      render: (text: string) => <Text code style={{ fontSize: '12px' }}>{text}</Text>
    },
    {
      title: 'Rank',
      dataIndex: 'rank',
      key: 'rank',
      width: 80,
    },
    {
      title: 'Alpha',
      dataIndex: 'alpha',
      key: 'alpha',
      width: 80,
    },
    {
      title: '状态',
      dataIndex: 'trainable',
      key: 'trainable',
      width: 100,
      render: (trainable: boolean) => (
        <Badge 
          status={trainable ? 'processing' : 'default'} 
          text={trainable ? '可训练' : '冻结'} 
        />
      )
    },
    {
      title: '参数量',
      key: 'params',
      width: 100,
      render: (_, record) => {
        const params = record.rank * 2 * 4096; // 简化计算
        return <Text>{(params / 1000).toFixed(1)}K</Text>;
      }
    }
  ];

  const handleRefresh = () => {
    setRefreshing(true);
    setTimeout(() => {
      setRefreshing(false);
    }, 1000);
  };

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <GoldOutlined style={{ marginRight: 8, color: '#faad14' }} />
          LoRA参数高效微调
        </Title>
        <Text type="secondary">
          Low-Rank Adaptation (LoRA) 训练监控和管理界面，实时跟踪训练进度、参数配置和性能指标
        </Text>
      </div>

      {/* 状态概览卡片 */}
      <Row gutter={16} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="训练状态"
              value={mockData.currentStatus.status === 'running' ? '运行中' : '已停止'}
              prefix={<PlayCircleOutlined style={{ color: '#52c41a' }} />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="训练进度"
              value={mockData.currentStatus.progress}
              suffix="%"
              prefix={<DashboardOutlined />}
            />
            <Progress 
              percent={mockData.currentStatus.progress} 
              size="small" 
              style={{ marginTop: 8 }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="当前损失"
              value={mockData.currentStatus.currentLoss}
              precision={4}
              prefix={<LineChartOutlined />}
              valueStyle={{ color: mockData.currentStatus.currentLoss < 1.0 ? '#52c41a' : '#faad14' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="可训练参数"
              value={`${mockData.modelInfo.trainablePercentage}%`}
              prefix={<SettingOutlined />}
              suffix={`(${(parseInt(mockData.modelInfo.trainableParams) / 1000000).toFixed(1)}M)`}
            />
          </Card>
        </Col>
      </Row>

      {/* 主要内容区 */}
      <Card>
        <Tabs activeKey={activeTab} onChange={setActiveTab}>
          {/* 训练概览 */}
          <TabPane tab={
            <span>
              <DashboardOutlined />
              训练概览
            </span>
          } key="overview">
            <Row gutter={16}>
              <Col span={16}>
                <Card title="训练损失曲线" size="small" style={{ height: 400 }}>
                  <Line {...lossChartConfig} height={320} />
                </Card>
              </Col>
              <Col span={8}>
                <Card title="实时状态" size="small" style={{ marginBottom: 16 }}>
                  <Descriptions column={1} size="small">
                    <Descriptions.Item label="当前轮次">
                      {mockData.currentStatus.currentEpoch}/{mockData.currentStatus.totalEpochs}
                    </Descriptions.Item>
                    <Descriptions.Item label="当前步数">
                      {mockData.currentStatus.currentStep}/{mockData.currentStatus.totalSteps}
                    </Descriptions.Item>
                    <Descriptions.Item label="学习率">
                      {mockData.currentStatus.learningRate.toExponential(2)}
                    </Descriptions.Item>
                    <Descriptions.Item label="训练时间">
                      <Text><ClockCircleOutlined /> {mockData.currentStatus.trainingTime}</Text>
                    </Descriptions.Item>
                    <Descriptions.Item label="预计剩余">
                      <Text type="secondary">{mockData.currentStatus.estimatedRemaining}</Text>
                    </Descriptions.Item>
                  </Descriptions>
                </Card>
                
                <Card title="资源使用" size="small">
                  <Row gutter={16}>
                    <Col span={12}>
                      <div style={{ height: 120 }}>
                        <Gauge {...gpuGaugeConfig} height={120} />
                      </div>
                    </Col>
                    <Col span={12}>
                      <div style={{ height: 120 }}>
                        <Gauge {...memoryGaugeConfig} height={120} />
                      </div>
                    </Col>
                  </Row>
                  <div style={{ marginTop: 16, textAlign: 'center' }}>
                    <Text type="secondary">
                      显存: {mockData.currentStatus.memoryUsage}GB / {mockData.currentStatus.totalMemory}GB
                    </Text>
                  </div>
                </Card>
              </Col>
            </Row>

            <Row gutter={16} style={{ marginTop: 16 }}>
              <Col span={12}>
                <Card title="学习率调度" size="small" style={{ height: 300 }}>
                  <Line {...learningRateConfig} height={220} />
                </Card>
              </Col>
              <Col span={12}>
                <Card title="训练日志" size="small" style={{ height: 300 }}>
                  <div style={{ height: 220, overflow: 'auto', fontSize: '12px', fontFamily: 'monospace' }}>
                    <div>[2025-08-23 14:30:15] 开始训练，加载模型...</div>
                    <div>[2025-08-23 14:30:22] LoRA适配器初始化完成</div>
                    <div>[2025-08-23 14:30:28] 开始第1轮训练</div>
                    <div>[2025-08-23 14:31:45] Step 100, Loss: 1.8923</div>
                    <div>[2025-08-23 14:33:12] Step 200, Loss: 1.5643</div>
                    <div>[2025-08-23 14:34:38] 第1轮完成，验证损失: 1.4123</div>
                    <div>[2025-08-23 14:34:42] 开始第2轮训练</div>
                    <div>[2025-08-23 14:36:15] Step 300, Loss: 1.3456</div>
                    <div style={{ color: '#52c41a' }}>[2025-08-23 14:37:44] Step 400, Loss: 1.1789 (新最佳)</div>
                    <div>[2025-08-23 14:38:56] 开始第3轮训练</div>
                    <div style={{ color: '#1890ff' }}>[2025-08-23 14:40:23] Step 500, Loss: 0.9876 (当前)</div>
                  </div>
                </Card>
              </Col>
            </Row>
          </TabPane>

          {/* LoRA配置 */}
          <TabPane tab={
            <span>
              <SettingOutlined />
              LoRA配置
            </span>
          } key="config">
            <Row gutter={16}>
              <Col span={12}>
                <Card title="模型信息" size="small">
                  <Descriptions column={1} bordered size="small">
                    <Descriptions.Item label="基础模型">
                      {mockData.modelInfo.baseModel}
                    </Descriptions.Item>
                    <Descriptions.Item label="模型大小">
                      {mockData.modelInfo.modelSize}
                    </Descriptions.Item>
                    <Descriptions.Item label="总参数量">
                      {(parseInt(mockData.modelInfo.totalParams) / 1000000000).toFixed(1)}B
                    </Descriptions.Item>
                    <Descriptions.Item label="可训练参数">
                      <Text type="success">
                        {(parseInt(mockData.modelInfo.trainableParams) / 1000000).toFixed(1)}M 
                        ({mockData.modelInfo.trainablePercentage}%)
                      </Text>
                    </Descriptions.Item>
                  </Descriptions>
                </Card>

                <Card title="LoRA超参数" size="small" style={{ marginTop: 16 }}>
                  <Form layout="vertical" size="small">
                    <Row gutter={16}>
                      <Col span={12}>
                        <Form.Item label="Rank (r)">
                          <Slider
                            min={1}
                            max={128}
                            value={mockData.modelInfo.loraRank}
                            disabled
                            marks={{
                              1: '1',
                              16: '16',
                              32: '32',
                              64: '64',
                              128: '128'
                            }}
                          />
                          <Text type="secondary">当前: {mockData.modelInfo.loraRank}</Text>
                        </Form.Item>
                      </Col>
                      <Col span={12}>
                        <Form.Item label="Alpha (α)">
                          <Slider
                            min={1}
                            max={128}
                            value={mockData.modelInfo.loraAlpha}
                            disabled
                            marks={{
                              1: '1',
                              16: '16',
                              32: '32',
                              64: '64',
                              128: '128'
                            }}
                          />
                          <Text type="secondary">当前: {mockData.modelInfo.loraAlpha}</Text>
                        </Form.Item>
                      </Col>
                    </Row>
                    <Form.Item label="Dropout">
                      <Slider
                        min={0}
                        max={0.5}
                        step={0.1}
                        value={mockData.modelInfo.loraDropout}
                        disabled
                        marks={{
                          0: '0',
                          0.1: '0.1',
                          0.2: '0.2',
                          0.3: '0.3',
                          0.5: '0.5'
                        }}
                      />
                      <Text type="secondary">当前: {mockData.modelInfo.loraDropout}</Text>
                    </Form.Item>
                    <Form.Item label="目标模块">
                      <Select
                        mode="tags"
                        value={mockData.modelInfo.targetModules}
                        disabled
                        style={{ width: '100%' }}
                      >
                        {mockData.modelInfo.targetModules.map(module => (
                          <Option key={module} value={module}>{module}</Option>
                        ))}
                      </Select>
                    </Form.Item>
                  </Form>
                </Card>
              </Col>

              <Col span={12}>
                <Card 
                  title="LoRA层详情" 
                  size="small"
                  extra={
                    <Tooltip title="显示部分LoRA适配层">
                      <InfoCircleOutlined />
                    </Tooltip>
                  }
                >
                  <Table
                    columns={loraLayerColumns}
                    dataSource={mockData.loraLayers}
                    rowKey="name"
                    size="small"
                    pagination={false}
                    scroll={{ y: 300 }}
                  />
                </Card>

                <Card title="配置说明" size="small" style={{ marginTop: 16 }}>
                  <Alert
                    message="LoRA参数说明"
                    description={
                      <div style={{ fontSize: '12px' }}>
                        <div><strong>Rank (r):</strong> 低秩分解的维度，控制适配器容量</div>
                        <div><strong>Alpha (α):</strong> 缩放因子，通常设置为rank的2倍</div>
                        <div><strong>Dropout:</strong> 防止过拟合的dropout概率</div>
                        <div><strong>目标模块:</strong> 应用LoRA的注意力层</div>
                      </div>
                    }
                    type="info"
                    showIcon
                    style={{ fontSize: '12px' }}
                  />
                </Card>
              </Col>
            </Row>
          </TabPane>

          {/* 性能监控 */}
          <TabPane tab={
            <span>
              <MonitorOutlined />
              性能监控
            </span>
          } key="performance">
            <Row gutter={16}>
              <Col span={8}>
                <Card title="GPU监控" size="small">
                  <div style={{ textAlign: 'center', marginBottom: 16 }}>
                    <Gauge {...gpuGaugeConfig} height={150} />
                  </div>
                  <Descriptions column={1} size="small">
                    <Descriptions.Item label="GPU型号">
                      NVIDIA RTX 4090
                    </Descriptions.Item>
                    <Descriptions.Item label="显存总量">
                      24 GB GDDR6X
                    </Descriptions.Item>
                    <Descriptions.Item label="计算能力">
                      8.9 (Ada Lovelace)
                    </Descriptions.Item>
                    <Descriptions.Item label="温度">
                      73°C
                    </Descriptions.Item>
                  </Descriptions>
                </Card>
              </Col>

              <Col span={8}>
                <Card title="内存监控" size="small">
                  <div style={{ textAlign: 'center', marginBottom: 16 }}>
                    <Gauge {...memoryGaugeConfig} height={150} />
                  </div>
                  <Descriptions column={1} size="small">
                    <Descriptions.Item label="已用显存">
                      {mockData.currentStatus.memoryUsage} GB
                    </Descriptions.Item>
                    <Descriptions.Item label="总显存">
                      {mockData.currentStatus.totalMemory} GB
                    </Descriptions.Item>
                    <Descriptions.Item label="系统内存">
                      32 GB / 64 GB
                    </Descriptions.Item>
                    <Descriptions.Item label="缓存">
                      4.2 GB
                    </Descriptions.Item>
                  </Descriptions>
                </Card>
              </Col>

              <Col span={8}>
                <Card title="训练效率" size="small">
                  <div style={{ marginBottom: 16 }}>
                    <Statistic
                      title="吞吐量"
                      value={142.5}
                      suffix="tokens/s"
                      prefix={<ThunderboltOutlined />}
                    />
                  </div>
                  <Descriptions column={1} size="small">
                    <Descriptions.Item label="批次处理时间">
                      0.85秒
                    </Descriptions.Item>
                    <Descriptions.Item label="前向传播">
                      0.52秒
                    </Descriptions.Item>
                    <Descriptions.Item label="反向传播">
                      0.33秒
                    </Descriptions.Item>
                    <Descriptions.Item label="参数更新">
                      0.12秒
                    </Descriptions.Item>
                  </Descriptions>
                </Card>
              </Col>
            </Row>

            <Card title="训练时间线" size="small" style={{ marginTop: 16 }}>
              <Timeline mode="left">
                <Timeline.Item 
                  color="green"
                  dot={<CheckCircleOutlined style={{ fontSize: '16px' }} />}
                >
                  <div>
                    <strong>14:30:15</strong> - 模型加载完成
                    <br />
                    <Text type="secondary">加载Llama-2-7b基础模型，初始化LoRA适配器</Text>
                  </div>
                </Timeline.Item>
                <Timeline.Item 
                  color="green"
                  dot={<CheckCircleOutlined style={{ fontSize: '16px' }} />}
                >
                  <div>
                    <strong>14:30:28</strong> - 开始第1轮训练
                    <br />
                    <Text type="secondary">训练数据加载，开始前向传播</Text>
                  </div>
                </Timeline.Item>
                <Timeline.Item 
                  color="green"
                  dot={<CheckCircleOutlined style={{ fontSize: '16px' }} />}
                >
                  <div>
                    <strong>14:34:38</strong> - 第1轮完成
                    <br />
                    <Text type="secondary">验证损失: 1.4123，保存检查点</Text>
                  </div>
                </Timeline.Item>
                <Timeline.Item 
                  color="blue"
                  dot={<PlayCircleOutlined style={{ fontSize: '16px' }} />}
                >
                  <div>
                    <strong>14:40:23</strong> - 第3轮训练中
                    <br />
                    <Text type="secondary">当前步数: 567/1500，损失持续下降</Text>
                  </div>
                </Timeline.Item>
                <Timeline.Item 
                  color="gray"
                  dot={<ClockCircleOutlined style={{ fontSize: '16px' }} />}
                >
                  <div>
                    <strong>预计 19:05</strong> - 训练完成
                    <br />
                    <Text type="secondary">预计还需4小时25分钟</Text>
                  </div>
                </Timeline.Item>
              </Timeline>
            </Card>
          </TabPane>

          {/* 操作控制 */}
          <TabPane tab={
            <span>
              <SettingOutlined />
              操作控制
            </span>
          } key="control">
            <Row gutter={16}>
              <Col span={12}>
                <Card title="训练控制" size="small">
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <Button 
                      type="primary" 
                      icon={<PauseCircleOutlined />} 
                      block
                      size="large"
                    >
                      暂停训练
                    </Button>
                    <Button 
                      danger 
                      icon={<FireOutlined />} 
                      block
                    >
                      停止训练
                    </Button>
                    <Button 
                      icon={<DatabaseOutlined />} 
                      block
                    >
                      保存检查点
                    </Button>
                  </Space>
                </Card>

                <Card title="高级设置" size="small" style={{ marginTop: 16 }}>
                  <Form layout="vertical" size="small">
                    <Form.Item label="自动保存间隔">
                      <Select defaultValue="500">
                        <Option value="100">每100步</Option>
                        <Option value="500">每500步</Option>
                        <Option value="1000">每1000步</Option>
                      </Select>
                    </Form.Item>
                    <Form.Item label="验证间隔">
                      <Select defaultValue="epoch">
                        <Option value="epoch">每轮验证</Option>
                        <Option value="500">每500步</Option>
                        <Option value="1000">每1000步</Option>
                      </Select>
                    </Form.Item>
                    <Form.Item>
                      <Space>
                        <Switch defaultChecked size="small" />
                        <Text>启用梯度裁剪</Text>
                      </Space>
                    </Form.Item>
                    <Form.Item>
                      <Space>
                        <Switch defaultChecked size="small" />
                        <Text>启用混合精度</Text>
                      </Space>
                    </Form.Item>
                  </Form>
                </Card>
              </Col>

              <Col span={12}>
                <Card title="日志查看器" size="small">
                  <div style={{ marginBottom: 16 }}>
                    <Space>
                      <Select defaultValue="all" size="small" style={{ width: 100 }}>
                        <Option value="all">全部</Option>
                        <Option value="info">信息</Option>
                        <Option value="warning">警告</Option>
                        <Option value="error">错误</Option>
                      </Select>
                      <Button size="small" onClick={handleRefresh} loading={refreshing}>
                        刷新
                      </Button>
                      <Button size="small">清空</Button>
                      <Button size="small">导出</Button>
                    </Space>
                  </div>
                  <div style={{ 
                    height: 400, 
                    overflow: 'auto', 
                    backgroundColor: '#001529', 
                    color: '#fff',
                    padding: '12px',
                    fontSize: '12px',
                    fontFamily: 'monospace',
                    border: '1px solid #d9d9d9',
                    borderRadius: '6px'
                  }}>
                    <div style={{ color: '#52c41a' }}>[INFO] 2025-08-23 14:40:23 - Step 567/1500</div>
                    <div style={{ color: '#52c41a' }}>[INFO] 2025-08-23 14:40:23 - Train Loss: 0.8743</div>
                    <div style={{ color: '#1890ff' }}>[DEBUG] 2025-08-23 14:40:23 - GPU Memory: 18.5GB/24GB</div>
                    <div style={{ color: '#52c41a' }}>[INFO] 2025-08-23 14:40:24 - Learning Rate: 1.89e-04</div>
                    <div style={{ color: '#faad14' }}>[WARN] 2025-08-23 14:40:25 - High GPU utilization: 87%</div>
                    <div style={{ color: '#52c41a' }}>[INFO] 2025-08-23 14:40:26 - Gradient norm: 0.234</div>
                    <div style={{ color: '#1890ff' }}>[DEBUG] 2025-08-23 14:40:27 - Batch processing time: 0.85s</div>
                    <div style={{ color: '#52c41a' }}>[INFO] 2025-08-23 14:40:28 - Step 568/1500</div>
                    <div style={{ color: '#52c41a' }}>[INFO] 2025-08-23 14:40:28 - Train Loss: 0.8712</div>
                    <div style={{ color: '#52c41a' }}>[INFO] 2025-08-23 14:40:29 - Progress: 37.9%</div>
                  </div>
                </Card>
              </Col>
            </Row>
          </TabPane>
        </Tabs>
      </Card>
    </div>
  );
};

export default LoRATrainingPage;