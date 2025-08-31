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
  Descriptions,
  Divider
} from 'antd';
import {
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
  ClockCircleOutlined,
  CompressOutlined,
  RocketOutlined,
  GoldOutlined
} from '@ant-design/icons';
import { Line, Bar, Gauge, Pie } from '@ant-design/charts';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { TextArea } = Input;
const { Option } = Select;

// 模拟QLoRA训练数据
const generateQLoRAData = () => ({
  trainingMetrics: [
    { epoch: 0, step: 0, trainLoss: 2.1234, evalLoss: 2.2456, learningRate: 1e-4, memoryUsage: 8.2 },
    { epoch: 0, step: 150, trainLoss: 1.7845, evalLoss: 1.8123, learningRate: 9.8e-5, memoryUsage: 8.5 },
    { epoch: 0, step: 300, trainLoss: 1.4567, evalLoss: 1.5234, learningRate: 9.6e-5, memoryUsage: 8.3 },
    { epoch: 1, step: 450, trainLoss: 1.2345, evalLoss: 1.3456, learningRate: 9.4e-5, memoryUsage: 8.4 },
    { epoch: 1, step: 600, trainLoss: 1.0876, evalLoss: 1.1987, learningRate: 9.2e-5, memoryUsage: 8.2 },
    { epoch: 2, step: 750, trainLoss: 0.9123, evalLoss: 1.0234, learningRate: 9.0e-5, memoryUsage: 8.6 },
  ],
  quantizationInfo: {
    quantizationType: 'NF4',
    bits: 4,
    doubleQuantization: true,
    computeDtype: 'bfloat16',
    quantDtype: 'nf4',
    blockSize: 64,
    compressionRatio: 75.2,
    memoryReduction: 68.4
  },
  modelInfo: {
    baseModel: 'mistralai/Mistral-7B-Instruct-v0.1',
    modelSize: '7B参数',
    originalSize: '13.5 GB',
    quantizedSize: '3.5 GB',
    loraRank: 8,
    loraAlpha: 16,
    loraDropout: 0.05,
    targetModules: ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    trainableParams: '2,359,296',
    totalParams: '7,241,732,096',
    trainablePercentage: 0.033
  },
  currentStatus: {
    status: 'running',
    currentEpoch: 2,
    totalEpochs: 4,
    currentStep: 789,
    totalSteps: 1200,
    progress: 65.8,
    currentLoss: 0.8456,
    bestLoss: 0.7823,
    learningRate: 8.9e-5,
    gpuUtilization: 92,
    memoryUsage: 8.6,
    totalMemory: 24.0,
    quantizedMemoryUsage: 3.5,
    originalMemoryWouldBe: 21.2,
    trainingTime: '01:45:32',
    estimatedRemaining: '00:52:15',
    inferenceSpeedup: 2.3
  },
  quantizationLayers: [
    { name: 'model.layers.0.self_attn.q_proj', bits: 4, dtype: 'nf4', compressed: true, ratio: '4:1' },
    { name: 'model.layers.0.self_attn.k_proj', bits: 4, dtype: 'nf4', compressed: true, ratio: '4:1' },
    { name: 'model.layers.0.self_attn.v_proj', bits: 4, dtype: 'nf4', compressed: true, ratio: '4:1' },
    { name: 'model.layers.0.self_attn.o_proj', bits: 4, dtype: 'nf4', compressed: true, ratio: '4:1' },
    { name: 'model.layers.0.mlp.gate_proj', bits: 4, dtype: 'nf4', compressed: true, ratio: '4:1' },
    { name: 'model.layers.0.mlp.up_proj', bits: 4, dtype: 'nf4', compressed: true, ratio: '4:1' },
  ],
  memoryComparison: [
    { type: '原始FP16', memory: 13.5, color: '#ff4d4f' },
    { type: 'QLoRA 4-bit', memory: 3.5, color: '#52c41a' },
    { type: 'LoRA适配器', memory: 0.1, color: '#1890ff' },
  ]
});

const QLoRATrainingPage: React.FC = () => {
  const [mockData] = useState(generateQLoRAData());
  const [activeTab, setActiveTab] = useState('overview');
  const [refreshing, setRefreshing] = useState(false);

  // 训练损失图表配置
  const lossChartConfig = {
    data: mockData.trainingMetrics,
    xField: 'step',
    yField: 'trainLoss',
    seriesField: 'type',
    smooth: true,
    color: ['#722ed1', '#13c2c2'],
    animation: {
      appear: {
        animation: 'path-in',
        duration: 1000,
      },
    },
  };

  // 内存使用对比图表
  const memoryComparisonConfig = {
    data: mockData.memoryComparison,
    angleField: 'memory',
    colorField: 'type',
    radius: 0.8,
    innerRadius: 0.5,
    label: {
      type: 'inner',
      offset: '-50%',
      style: {
        textAlign: 'center',
        fontSize: 12,
        fill: '#fff',
      },
      formatter: (data: any) => `${data.memory}GB`,
    },
    legend: {
      position: 'bottom',
    },
    color: ['#ff4d4f', '#52c41a', '#1890ff'],
  };

  // 内存节省仪表盘配置
  const memorySavingGaugeConfig = {
    percent: mockData.quantizationInfo.memoryReduction / 100,
    type: 'meter',
    innerRadius: 0.75,
    range: {
      ticks: [0, 0.25, 0.5, 0.75, 1],
      color: ['#F4664A', '#FAAD14', '#30BF78', '#1890FF', '#722ED1'],
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
        formatter: () => '内存节省',
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
        formatter: () => `${mockData.quantizationInfo.memoryReduction}%`,
      },
    },
  };

  // 量化层表格列定义
  const quantizationLayerColumns = [
    {
      title: '层名称',
      dataIndex: 'name',
      key: 'name',
      width: 280,
      render: (text: string) => <Text code style={{ fontSize: '12px' }}>{text}</Text>
    },
    {
      title: '量化位数',
      dataIndex: 'bits',
      key: 'bits',
      width: 80,
      render: (bits: number) => <Tag color="purple">{bits}-bit</Tag>
    },
    {
      title: '数据类型',
      dataIndex: 'dtype',
      key: 'dtype',
      width: 80,
      render: (dtype: string) => <Tag color="cyan">{dtype.toUpperCase()}</Tag>
    },
    {
      title: '压缩状态',
      dataIndex: 'compressed',
      key: 'compressed',
      width: 100,
      render: (compressed: boolean) => (
        <Badge 
          status={compressed ? 'processing' : 'default'} 
          text={compressed ? '已压缩' : '未压缩'} 
        />
      )
    },
    {
      title: '压缩比',
      dataIndex: 'ratio',
      key: 'ratio',
      width: 80,
      render: (ratio: string) => <Text type="success">{ratio}</Text>
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
          <ThunderboltOutlined style={{ marginRight: 8, color: '#722ed1' }} />
          QLoRA量化微调
        </Title>
        <Text type="secondary">
          Quantized LoRA (QLoRA) 4-bit量化微调监控界面，大幅降低显存占用的同时保持训练效果
        </Text>
      </div>

      {/* 状态概览卡片 */}
      <Row gutter={16} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="内存节省"
              value={mockData.quantizationInfo.memoryReduction}
              suffix="%"
              prefix={<CompressOutlined style={{ color: '#52c41a' }} />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="量化后大小"
              value={mockData.modelInfo.quantizedSize}
              prefix={<DatabaseOutlined />}
              suffix={`/ ${mockData.modelInfo.originalSize}`}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="推理加速"
              value={mockData.currentStatus.inferenceSpeedup}
              suffix="x"
              prefix={<RocketOutlined />}
              valueStyle={{ color: '#1890ff' }}
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
                    <Descriptions.Item label="训练状态">
                      <Badge status="processing" text="QLoRA训练中" />
                    </Descriptions.Item>
                    <Descriptions.Item label="当前轮次">
                      {mockData.currentStatus.currentEpoch}/{mockData.currentStatus.totalEpochs}
                    </Descriptions.Item>
                    <Descriptions.Item label="当前步数">
                      {mockData.currentStatus.currentStep}/{mockData.currentStatus.totalSteps}
                    </Descriptions.Item>
                    <Descriptions.Item label="量化类型">
                      <Tag color="purple">{mockData.quantizationInfo.quantizationType}</Tag>
                    </Descriptions.Item>
                    <Descriptions.Item label="学习率">
                      {mockData.currentStatus.learningRate.toExponential(2)}
                    </Descriptions.Item>
                    <Descriptions.Item label="训练时间">
                      <Text><ClockCircleOutlined /> {mockData.currentStatus.trainingTime}</Text>
                    </Descriptions.Item>
                  </Descriptions>
                </Card>
                
                <Card title="内存对比" size="small">
                  <div style={{ height: 200 }}>
                    <Pie {...memoryComparisonConfig} height={200} />
                  </div>
                  <div style={{ marginTop: 12, textAlign: 'center' }}>
                    <Text type="success" strong>
                      节省 {(parseFloat(mockData.modelInfo.originalSize) - parseFloat(mockData.modelInfo.quantizedSize)).toFixed(1)}GB 显存
                    </Text>
                  </div>
                </Card>
              </Col>
            </Row>

            <Row gutter={16} style={{ marginTop: 16 }}>
              <Col span={8}>
                <Card title="量化效果" size="small">
                  <div style={{ textAlign: 'center', marginBottom: 16 }}>
                    <Gauge {...memorySavingGaugeConfig} height={150} />
                  </div>
                  <div style={{ textAlign: 'center' }}>
                    <Space direction="vertical" size="small">
                      <div>
                        <Text type="secondary">原始模型: </Text>
                        <Text strong>{mockData.modelInfo.originalSize}</Text>
                      </div>
                      <div>
                        <Text type="secondary">量化后: </Text>
                        <Text strong style={{ color: '#52c41a' }}>{mockData.modelInfo.quantizedSize}</Text>
                      </div>
                    </Space>
                  </div>
                </Card>
              </Col>
              <Col span={8}>
                <Card title="量化参数" size="small">
                  <Descriptions column={1} size="small">
                    <Descriptions.Item label="量化位数">
                      <Tag color="purple">{mockData.quantizationInfo.bits}-bit</Tag>
                    </Descriptions.Item>
                    <Descriptions.Item label="量化类型">
                      <Tag color="cyan">{mockData.quantizationInfo.quantizationType}</Tag>
                    </Descriptions.Item>
                    <Descriptions.Item label="双量化">
                      <Badge 
                        status={mockData.quantizationInfo.doubleQuantization ? 'processing' : 'default'} 
                        text={mockData.quantizationInfo.doubleQuantization ? '启用' : '禁用'} 
                      />
                    </Descriptions.Item>
                    <Descriptions.Item label="计算精度">
                      {mockData.quantizationInfo.computeDtype}
                    </Descriptions.Item>
                    <Descriptions.Item label="块大小">
                      {mockData.quantizationInfo.blockSize}
                    </Descriptions.Item>
                    <Descriptions.Item label="压缩比">
                      <Text type="success">{mockData.quantizationInfo.compressionRatio}%</Text>
                    </Descriptions.Item>
                  </Descriptions>
                </Card>
              </Col>
              <Col span={8}>
                <Card title="性能提升" size="small">
                  <div style={{ marginBottom: 16 }}>
                    <div>
                      <Text type="secondary">推理速度提升: </Text>
                      <Text strong style={{ color: '#1890ff' }}>
                        {mockData.currentStatus.inferenceSpeedup}x
                      </Text>
                    </div>
                    <Progress 
                      percent={mockData.currentStatus.inferenceSpeedup * 100 / 4} 
                      size="small" 
                      strokeColor="#1890ff"
                      style={{ marginTop: 4 }}
                    />
                  </div>
                  <div style={{ marginBottom: 16 }}>
                    <div>
                      <Text type="secondary">内存效率: </Text>
                      <Text strong style={{ color: '#52c41a' }}>
                        {mockData.quantizationInfo.memoryReduction.toFixed(1)}%
                      </Text>
                    </div>
                    <Progress 
                      percent={mockData.quantizationInfo.memoryReduction} 
                      size="small" 
                      strokeColor="#52c41a"
                      style={{ marginTop: 4 }}
                    />
                  </div>
                  <div>
                    <Text type="secondary">压缩率: </Text>
                    <Text strong style={{ color: '#722ed1' }}>
                      {mockData.quantizationInfo.compressionRatio.toFixed(1)}%
                    </Text>
                    <Progress 
                      percent={mockData.quantizationInfo.compressionRatio} 
                      size="small" 
                      strokeColor="#722ed1"
                      style={{ marginTop: 4 }}
                    />
                  </div>
                </Card>
              </Col>
            </Row>
          </TabPane>

          {/* 量化配置 */}
          <TabPane tab={
            <span>
              <CompressOutlined />
              量化配置
            </span>
          } key="quantization">
            <Row gutter={16}>
              <Col span={12}>
                <Card title="量化方案" size="small">
                  <Alert
                    message="NF4 (NormalFloat4) 量化"
                    description="使用信息理论最优的4-bit量化格式，相比传统INT4量化具有更好的精度保持能力"
                    type="info"
                    showIcon
                    style={{ marginBottom: 16 }}
                  />
                  
                  <Descriptions column={2} bordered size="small">
                    <Descriptions.Item label="量化位数">
                      {mockData.quantizationInfo.bits}-bit
                    </Descriptions.Item>
                    <Descriptions.Item label="量化类型">
                      {mockData.quantizationInfo.quantizationType}
                    </Descriptions.Item>
                    <Descriptions.Item label="计算数据类型">
                      {mockData.quantizationInfo.computeDtype}
                    </Descriptions.Item>
                    <Descriptions.Item label="量化数据类型">
                      {mockData.quantizationInfo.quantDtype}
                    </Descriptions.Item>
                    <Descriptions.Item label="双量化">
                      {mockData.quantizationInfo.doubleQuantization ? '启用' : '禁用'}
                    </Descriptions.Item>
                    <Descriptions.Item label="块大小">
                      {mockData.quantizationInfo.blockSize}
                    </Descriptions.Item>
                  </Descriptions>
                </Card>

                <Card title="量化设置" size="small" style={{ marginTop: 16 }}>
                  <Form layout="vertical" size="small">
                    <Row gutter={16}>
                      <Col span={12}>
                        <Form.Item label="量化位数">
                          <Select value={mockData.quantizationInfo.bits} disabled>
                            <Option value={4}>4-bit</Option>
                            <Option value={8}>8-bit</Option>
                          </Select>
                        </Form.Item>
                      </Col>
                      <Col span={12}>
                        <Form.Item label="量化类型">
                          <Select value={mockData.quantizationInfo.quantizationType} disabled>
                            <Option value="nf4">NF4</Option>
                            <Option value="fp4">FP4</Option>
                            <Option value="int4">INT4</Option>
                          </Select>
                        </Form.Item>
                      </Col>
                    </Row>
                    
                    <Form.Item label="块大小">
                      <Slider
                        min={16}
                        max={256}
                        step={16}
                        value={mockData.quantizationInfo.blockSize}
                        disabled
                        marks={{
                          16: '16',
                          64: '64',
                          128: '128',
                          256: '256'
                        }}
                      />
                      <Text type="secondary">当前: {mockData.quantizationInfo.blockSize}</Text>
                    </Form.Item>

                    <Space direction="vertical">
                      <div>
                        <Switch checked={mockData.quantizationInfo.doubleQuantization} disabled size="small" />
                        <Text style={{ marginLeft: 8 }}>启用双量化</Text>
                        <Tooltip title="进一步压缩量化常数的存储空间">
                          <InfoCircleOutlined style={{ marginLeft: 4, color: '#1890ff' }} />
                        </Tooltip>
                      </div>
                      <div>
                        <Switch checked={true} disabled size="small" />
                        <Text style={{ marginLeft: 8 }}>使用嵌套量化</Text>
                      </div>
                      <div>
                        <Switch checked={true} disabled size="small" />
                        <Text style={{ marginLeft: 8 }}>优化内存布局</Text>
                      </div>
                    </Space>
                  </Form>
                </Card>
              </Col>

              <Col span={12}>
                <Card 
                  title="量化层详情" 
                  size="small"
                  extra={
                    <Space>
                      <Tag color="purple">{mockData.quantizationInfo.bits}-bit</Tag>
                      <Tag color="cyan">{mockData.quantizationInfo.quantizationType}</Tag>
                    </Space>
                  }
                >
                  <Table
                    columns={quantizationLayerColumns}
                    dataSource={mockData.quantizationLayers}
                    rowKey="name"
                    size="small"
                    pagination={false}
                    scroll={{ y: 250 }}
                  />
                </Card>

                <Card title="内存使用分析" size="small" style={{ marginTop: 16 }}>
                  <Row gutter={16}>
                    <Col span={8}>
                      <Statistic
                        title="原始模型"
                        value={mockData.modelInfo.originalSize}
                        valueStyle={{ color: '#ff4d4f' }}
                        prefix={<DatabaseOutlined />}
                      />
                    </Col>
                    <Col span={8}>
                      <Statistic
                        title="量化后"
                        value={mockData.modelInfo.quantizedSize}
                        valueStyle={{ color: '#52c41a' }}
                        prefix={<CompressOutlined />}
                      />
                    </Col>
                    <Col span={8}>
                      <Statistic
                        title="节省"
                        value={`${(parseFloat(mockData.modelInfo.originalSize) - parseFloat(mockData.modelInfo.quantizedSize)).toFixed(1)}GB`}
                        valueStyle={{ color: '#1890ff' }}
                        prefix={<DesktopOutlined />}
                      />
                    </Col>
                  </Row>
                  
                  <Divider />
                  
                  <div style={{ textAlign: 'center' }}>
                    <Text type="secondary">实际显存占用</Text>
                    <br />
                    <Space>
                      <Text>当前: </Text>
                      <Text strong style={{ color: '#52c41a' }}>{mockData.currentStatus.memoryUsage}GB</Text>
                      <Text type="secondary">vs</Text>
                      <Text>原始需求: </Text>
                      <Text strong style={{ color: '#ff4d4f' }}>{mockData.currentStatus.originalMemoryWouldBe}GB</Text>
                    </Space>
                  </div>
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
                <Card title="量化性能" size="small">
                  <div style={{ marginBottom: 20 }}>
                    <div style={{ marginBottom: 8 }}>
                      <Text>内存压缩率</Text>
                      <div style={{ float: 'right' }}>
                        <Text strong style={{ color: '#52c41a' }}>
                          {mockData.quantizationInfo.compressionRatio}%
                        </Text>
                      </div>
                    </div>
                    <Progress 
                      percent={mockData.quantizationInfo.compressionRatio} 
                      strokeColor="#52c41a"
                      size="small"
                    />
                  </div>

                  <div style={{ marginBottom: 20 }}>
                    <div style={{ marginBottom: 8 }}>
                      <Text>推理加速</Text>
                      <div style={{ float: 'right' }}>
                        <Text strong style={{ color: '#1890ff' }}>
                          {mockData.currentStatus.inferenceSpeedup}x
                        </Text>
                      </div>
                    </div>
                    <Progress 
                      percent={mockData.currentStatus.inferenceSpeedup * 25} 
                      strokeColor="#1890ff"
                      size="small"
                    />
                  </div>

                  <div>
                    <div style={{ marginBottom: 8 }}>
                      <Text>精度保持</Text>
                      <div style={{ float: 'right' }}>
                        <Text strong style={{ color: '#722ed1' }}>
                          97.2%
                        </Text>
                      </div>
                    </div>
                    <Progress 
                      percent={97.2} 
                      strokeColor="#722ed1"
                      size="small"
                    />
                  </div>
                </Card>

                <Card title="系统资源" size="small" style={{ marginTop: 16 }}>
                  <Descriptions column={1} size="small">
                    <Descriptions.Item label="GPU使用率">
                      <Text>{mockData.currentStatus.gpuUtilization}%</Text>
                      <Progress 
                        percent={mockData.currentStatus.gpuUtilization} 
                        size="small" 
                        style={{ marginTop: 4 }}
                      />
                    </Descriptions.Item>
                    <Descriptions.Item label="显存使用">
                      <Text>
                        {mockData.currentStatus.memoryUsage}GB / {mockData.currentStatus.totalMemory}GB
                      </Text>
                      <Progress 
                        percent={(mockData.currentStatus.memoryUsage / mockData.currentStatus.totalMemory) * 100} 
                        size="small" 
                        style={{ marginTop: 4 }}
                      />
                    </Descriptions.Item>
                    <Descriptions.Item label="量化开销">
                      <Text>0.2GB (量化常数)</Text>
                    </Descriptions.Item>
                  </Descriptions>
                </Card>
              </Col>

              <Col span={8}>
                <Card title="训练效率对比" size="small">
                  <div style={{ marginBottom: 16 }}>
                    <Alert
                      message="QLoRA优势"
                      description="相比传统LoRA训练，QLoRA在保持相近效果的同时显著降低了内存需求"
                      type="success"
                      showIcon
                      size="small"
                    />
                  </div>

                  <div style={{ marginBottom: 16 }}>
                    <Row>
                      <Col span={8}>
                        <div style={{ textAlign: 'center' }}>
                          <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#ff4d4f' }}>
                            21.2GB
                          </div>
                          <div style={{ fontSize: '12px', color: '#666' }}>传统LoRA</div>
                        </div>
                      </Col>
                      <Col span={8} style={{ textAlign: 'center' }}>
                        <RocketOutlined style={{ fontSize: '24px', color: '#52c41a' }} />
                        <div style={{ fontSize: '12px', color: '#666', marginTop: 4 }}>vs</div>
                      </Col>
                      <Col span={8}>
                        <div style={{ textAlign: 'center' }}>
                          <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#52c41a' }}>
                            8.6GB
                          </div>
                          <div style={{ fontSize: '12px', color: '#666' }}>QLoRA</div>
                        </div>
                      </Col>
                    </Row>
                  </div>

                  <Descriptions column={1} size="small">
                    <Descriptions.Item label="内存节省">
                      <Text style={{ color: '#52c41a' }}>59.4%</Text>
                    </Descriptions.Item>
                    <Descriptions.Item label="训练速度">
                      <Text style={{ color: '#1890ff' }}>提升 15%</Text>
                    </Descriptions.Item>
                    <Descriptions.Item label="精度损失">
                      <Text style={{ color: '#722ed1' }}>{'<'} 3%</Text>
                    </Descriptions.Item>
                    <Descriptions.Item label="收敛性">
                      <Text>相当</Text>
                    </Descriptions.Item>
                  </Descriptions>
                </Card>
              </Col>

              <Col span={8}>
                <Card title="量化质量指标" size="small">
                  <Timeline size="small">
                    <Timeline.Item 
                      color="green"
                      dot={<CheckCircleOutlined style={{ fontSize: '12px' }} />}
                    >
                      <div>
                        <Text strong>量化完成</Text>
                        <br />
                        <Text type="secondary" style={{ fontSize: '12px' }}>
                          32层transformer量化成功
                        </Text>
                      </div>
                    </Timeline.Item>
                    <Timeline.Item 
                      color="blue"
                      dot={<InfoCircleOutlined style={{ fontSize: '12px' }} />}
                    >
                      <div>
                        <Text strong>校准数据集</Text>
                        <br />
                        <Text type="secondary" style={{ fontSize: '12px' }}>
                          使用1024个样本进行量化校准
                        </Text>
                      </div>
                    </Timeline.Item>
                    <Timeline.Item 
                      color="purple"
                      dot={<GoldOutlined style={{ fontSize: '12px' }} />}
                    >
                      <div>
                        <Text strong>精度验证</Text>
                        <br />
                        <Text type="secondary" style={{ fontSize: '12px' }}>
                          BLEU分数保持97.2%
                        </Text>
                      </div>
                    </Timeline.Item>
                    <Timeline.Item 
                      color="cyan"
                      dot={<RocketOutlined style={{ fontSize: '12px' }} />}
                    >
                      <div>
                        <Text strong>推理优化</Text>
                        <br />
                        <Text type="secondary" style={{ fontSize: '12px' }}>
                          INT4 CUDA kernel加速
                        </Text>
                      </div>
                    </Timeline.Item>
                  </Timeline>
                </Card>
              </Col>
            </Row>

            <Card title="量化前后对比" size="small" style={{ marginTop: 16 }}>
              <Row gutter={32}>
                <Col span={12}>
                  <div style={{ textAlign: 'center', padding: '20px', border: '1px dashed #ff4d4f' }}>
                    <Title level={4} style={{ color: '#ff4d4f', margin: 0 }}>
                      原始FP16模型
                    </Title>
                    <div style={{ marginTop: 16 }}>
                      <Row gutter={16}>
                        <Col span={8}>
                          <Statistic title="模型大小" value="13.5GB" />
                        </Col>
                        <Col span={8}>
                          <Statistic title="显存占用" value="21.2GB" />
                        </Col>
                        <Col span={8}>
                          <Statistic title="推理速度" value="1.0x" />
                        </Col>
                      </Row>
                    </div>
                  </div>
                </Col>
                <Col span={12}>
                  <div style={{ textAlign: 'center', padding: '20px', border: '1px dashed #52c41a' }}>
                    <Title level={4} style={{ color: '#52c41a', margin: 0 }}>
                      QLoRA 4-bit模型
                    </Title>
                    <div style={{ marginTop: 16 }}>
                      <Row gutter={16}>
                        <Col span={8}>
                          <Statistic 
                            title="模型大小" 
                            value="3.5GB" 
                            valueStyle={{ color: '#52c41a' }}
                          />
                        </Col>
                        <Col span={8}>
                          <Statistic 
                            title="显存占用" 
                            value="8.6GB" 
                            valueStyle={{ color: '#52c41a' }}
                          />
                        </Col>
                        <Col span={8}>
                          <Statistic 
                            title="推理速度" 
                            value="2.3x" 
                            valueStyle={{ color: '#52c41a' }}
                          />
                        </Col>
                      </Row>
                    </div>
                  </div>
                </Col>
              </Row>
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
                      暂停QLoRA训练
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
                      保存量化检查点
                    </Button>
                    <Button 
                      icon={<RocketOutlined />} 
                      block
                    >
                      优化推理设置
                    </Button>
                  </Space>
                </Card>

                <Card title="量化设置调整" size="small" style={{ marginTop: 16 }}>
                  <Alert
                    message="注意"
                    description="量化设置在训练开始后无法修改，如需调整请重新开始训练"
                    type="warning"
                    showIcon
                    style={{ marginBottom: 16 }}
                  />
                  
                  <Form layout="vertical" size="small">
                    <Form.Item label="推理优化">
                      <Space>
                        <Switch checked={true} size="small" />
                        <Text>启用INT4 CUDA kernel</Text>
                      </Space>
                    </Form.Item>
                    <Form.Item label="内存优化">
                      <Space>
                        <Switch checked={true} size="small" />
                        <Text>启用梯度检查点</Text>
                      </Space>
                    </Form.Item>
                    <Form.Item label="精度监控">
                      <Space>
                        <Switch checked={true} size="small" />
                        <Text>实时精度验证</Text>
                      </Space>
                    </Form.Item>
                  </Form>
                </Card>
              </Col>

              <Col span={12}>
                <Card title="量化状态日志" size="small">
                  <div style={{ marginBottom: 16 }}>
                    <Space>
                      <Select defaultValue="all" size="small" style={{ width: 100 }}>
                        <Option value="all">全部</Option>
                        <Option value="quantization">量化</Option>
                        <Option value="training">训练</Option>
                        <Option value="inference">推理</Option>
                      </Select>
                      <Button size="small" onClick={handleRefresh} loading={refreshing}>
                        刷新
                      </Button>
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
                    <div style={{ color: '#722ed1' }}>[QUANT] 2025-08-23 14:40:23 - 4-bit NF4量化初始化</div>
                    <div style={{ color: '#52c41a' }}>[INFO] 2025-08-23 14:40:24 - 双量化已启用</div>
                    <div style={{ color: '#1890ff' }}>[DEBUG] 2025-08-23 14:40:24 - 块大小: 64, 计算类型: bfloat16</div>
                    <div style={{ color: '#52c41a' }}>[INFO] 2025-08-23 14:40:25 - 量化32层transformer完成</div>
                    <div style={{ color: '#722ed1' }}>[QUANT] 2025-08-23 14:40:26 - 内存占用: 8.6GB (vs 21.2GB)</div>
                    <div style={{ color: '#52c41a' }}>[INFO] 2025-08-23 14:40:27 - QLoRA适配器已就绪</div>
                    <div style={{ color: '#1890ff' }}>[DEBUG] 2025-08-23 14:40:28 - INT4 CUDA kernel已加载</div>
                    <div style={{ color: '#52c41a' }}>[INFO] 2025-08-23 14:40:29 - Step 789/1200</div>
                    <div style={{ color: '#52c41a' }}>[INFO] 2025-08-23 14:40:29 - QLoRA Loss: 0.8456</div>
                    <div style={{ color: '#722ed1' }}>[QUANT] 2025-08-23 14:40:30 - 推理加速: 2.3x</div>
                    <div style={{ color: '#1890ff' }}>[DEBUG] 2025-08-23 14:40:31 - 量化精度验证: 97.2%</div>
                    <div style={{ color: '#52c41a' }}>[INFO] 2025-08-23 14:40:32 - 训练继续...</div>
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

export default QLoRATrainingPage;