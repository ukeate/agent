import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Button,
  Select,
  Slider,
  Switch,
  Progress,
  Statistic,
  Table,
  Tag,
  Space,
  Tabs,
  Form,
  InputNumber,
  Alert,
  Modal,
  List,
  Typography,
  Divider,
  Tooltip
} from 'antd';
import {
  ThunderboltOutlined,
  RocketOutlined,
  DatabaseOutlined,
  ClusterOutlined,
  BarChartOutlined,
  SettingOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  ReloadOutlined,
  ExportOutlined,
  ImportOutlined,
  MonitorOutlined,
  ThunderboltOutlined as GpuOutlined,
  CloudServerOutlined,
  ThunderboltOutlined as OptimizationOutlined,
  ExperimentOutlined
} from '@ant-design/icons';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';

const { Title, Text, Paragraph } = Typography;
const { Option } = Select;

// 性能配置接口
interface PerformanceConfig {
  preset: 'high_performance' | 'memory_efficient' | 'development';
  gpu: {
    enabled: boolean;
    mixed_precision: boolean;
    xla_compilation: boolean;
    batch_multiplier: number;
    memory_limit?: number;
  };
  buffer: {
    strategy: 'uniform' | 'prioritized' | 'recency_weighted' | 'curiosity_driven';
    capacity: number;
    batch_size: number;
    compression: boolean;
    parallel_calls: number;
  };
  distributed: {
    strategy: 'data_parallel' | 'model_parallel' | 'parameter_server' | 'allreduce';
    num_workers: number;
    sync_frequency: number;
    compression_enabled: boolean;
  };
}

// 性能指标接口
interface PerformanceMetrics {
  training_fps: number;
  inference_fps: number;
  memory_usage: number;
  gpu_utilization: number;
  convergence_speed: number;
  sample_efficiency: number;
  network_throughput: number;
  error_rate: number;
}

// 基准测试结果接口
interface BenchmarkResult {
  algorithm: string;
  final_performance: number;
  convergence_time: number;
  memory_usage: number;
  training_time: number;
  stability_score: number;
}

// 优化试验接口
interface OptimizationTrial {
  trial_id: number;
  parameters: Record<string, any>;
  objective_value: number;
  status: 'completed' | 'running' | 'failed';
  duration: number;
}

const QLearningPerformanceOptimizationPage: React.FC = () => {
  const [config, setConfig] = useState<PerformanceConfig>({
    preset: 'high_performance',
    gpu: {
      enabled: true,
      mixed_precision: true,
      xla_compilation: true,
      batch_multiplier: 4,
      memory_limit: 8192
    },
    buffer: {
      strategy: 'prioritized',
      capacity: 100000,
      batch_size: 128,
      compression: true,
      parallel_calls: 8
    },
    distributed: {
      strategy: 'data_parallel',
      num_workers: 4,
      sync_frequency: 100,
      compression_enabled: true
    }
  });

  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    training_fps: 0,
    inference_fps: 0,
    memory_usage: 0,
    gpu_utilization: 0,
    convergence_speed: 0,
    sample_efficiency: 0,
    network_throughput: 0,
    error_rate: 0
  });

  const [benchmarkResults, setBenchmarkResults] = useState<BenchmarkResult[]>([]);
  const [optimizationTrials, setOptimizationTrials] = useState<OptimizationTrial[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [isBenchmarking, setIsBenchmarking] = useState(false);
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [activeTab, setActiveTab] = useState('configuration');
  const [form] = Form.useForm();

  useEffect(() => {}, []);

  // 性能配置预设
  const applyPreset = (preset: string) => {
    const presets = {
      high_performance: {
        gpu: {
          enabled: true,
          mixed_precision: true,
          xla_compilation: true,
          batch_multiplier: 4,
          memory_limit: 8192
        },
        buffer: {
          strategy: 'prioritized' as const,
          capacity: 100000,
          batch_size: 128,
          compression: true,
          parallel_calls: 8
        },
        distributed: {
          strategy: 'data_parallel' as const,
          num_workers: 4,
          sync_frequency: 100,
          compression_enabled: true
        }
      },
      memory_efficient: {
        gpu: {
          enabled: true,
          mixed_precision: true,
          xla_compilation: false,
          batch_multiplier: 2,
          memory_limit: 4096
        },
        buffer: {
          strategy: 'uniform' as const,
          capacity: 50000,
          batch_size: 64,
          compression: true,
          parallel_calls: 4
        },
        distributed: {
          strategy: 'parameter_server' as const,
          num_workers: 2,
          sync_frequency: 200,
          compression_enabled: true
        }
      },
      development: {
        gpu: {
          enabled: false,
          mixed_precision: false,
          xla_compilation: false,
          batch_multiplier: 1,
          memory_limit: 2048
        },
        buffer: {
          strategy: 'uniform' as const,
          capacity: 10000,
          batch_size: 32,
          compression: false,
          parallel_calls: 2
        },
        distributed: {
          strategy: 'data_parallel' as const,
          num_workers: 1,
          sync_frequency: 500,
          compression_enabled: false
        }
      }
    };

    if (preset in presets) {
      setConfig(prev => ({
        ...prev,
        preset: preset as any,
        ...presets[preset as keyof typeof presets]
      }));
    }
  };

  // 基准测试结果表格
  const benchmarkColumns = [
    {
      title: '算法',
      dataIndex: 'algorithm',
      key: 'algorithm',
      render: (text: string) => <Tag color="blue">{text}</Tag>
    },
    {
      title: '最终性能',
      dataIndex: 'final_performance',
      key: 'final_performance',
      render: (value: number) => `${(value * 100).toFixed(1)}%`,
      sorter: (a: any, b: any) => a.final_performance - b.final_performance
    },
    {
      title: '收敛时间',
      dataIndex: 'convergence_time',
      key: 'convergence_time',
      render: (value: number) => `${value.toFixed(0)} episodes`,
      sorter: (a: any, b: any) => a.convergence_time - b.convergence_time
    },
    {
      title: '内存使用',
      dataIndex: 'memory_usage',
      key: 'memory_usage',
      render: (value: number) => `${(value / 1024).toFixed(1)} GB`,
      sorter: (a: any, b: any) => a.memory_usage - b.memory_usage
    },
    {
      title: '训练时间',
      dataIndex: 'training_time',
      key: 'training_time',
      render: (value: number) => `${(value / 60).toFixed(1)} min`,
      sorter: (a: any, b: any) => a.training_time - b.training_time
    },
    {
      title: '稳定性',
      dataIndex: 'stability_score',
      key: 'stability_score',
      render: (value: number) => (
        <Progress 
          percent={value * 100} 
          size="small" 
          format={(percent) => `${percent?.toFixed(1)}%`}
        />
      )
    }
  ];

  // 优化试验表格
  const optimizationColumns = [
    {
      title: '试验ID',
      dataIndex: 'trial_id',
      key: 'trial_id',
      width: 80
    },
    {
      title: '学习率',
      key: 'learning_rate',
      render: (record: OptimizationTrial) => record.parameters.learning_rate?.toFixed(6)
    },
    {
      title: '批次大小',
      key: 'batch_size',
      render: (record: OptimizationTrial) => record.parameters.batch_size
    },
    {
      title: 'Epsilon衰减',
      key: 'epsilon_decay',
      render: (record: OptimizationTrial) => record.parameters.epsilon_decay?.toFixed(4)
    },
    {
      title: '目标值',
      dataIndex: 'objective_value',
      key: 'objective_value',
      render: (value: number) => value.toFixed(4),
      sorter: (a: any, b: any) => a.objective_value - b.objective_value
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const colors = {
          completed: 'green',
          running: 'blue',
          failed: 'red'
        };
        return <Tag color={colors[status as keyof typeof colors]}>{status}</Tag>;
      }
    },
    {
      title: '耗时',
      dataIndex: 'duration',
      key: 'duration',
      render: (value: number) => `${(value / 60).toFixed(1)}min`
    }
  ];

  const performanceTrendData: Array<any> = [];

  return (
    <div style={{ padding: '24px', background: '#f5f5f5', minHeight: '100vh' }}>
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        {/* 页面标题和控制按钮 */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <Title level={2}>
              <OptimizationOutlined /> Q-Learning性能优化中心
            </Title>
            <Paragraph>
              高性能强化学习训练优化，支持GPU加速、分布式训练、自动超参数调优和性能基准测试
            </Paragraph>
          </div>
          <Space>
            <Button 
              type="primary" 
              icon={<PlayCircleOutlined />}
              loading={isTraining}
              onClick={() => setIsTraining(!isTraining)}
              size="large"
            >
              {isTraining ? '训练中...' : '开始训练'}
            </Button>
            <Button 
              icon={<BarChartOutlined />}
              loading={isBenchmarking}
              onClick={() => setIsBenchmarking(!isBenchmarking)}
              size="large"
            >
              运行基准测试
            </Button>
            <Button 
              icon={<SettingOutlined />}
              loading={isOptimizing}
              onClick={() => setIsOptimizing(!isOptimizing)}
              size="large"
            >
              超参数优化
            </Button>
          </Space>
        </div>

        {/* 实时性能指标 */}
        <Row gutter={[16, 16]}>
          <Col span={6}>
            <Card>
              <Statistic
                title="训练FPS"
                value={metrics.training_fps}
                precision={1}
                suffix="samples/s"
                prefix={<ThunderboltOutlined />}
                valueStyle={{ color: '#3f8600' }}
              />
              <Progress percent={Math.min(metrics.training_fps / 5, 100)} showInfo={false} />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="推理FPS"
                value={metrics.inference_fps}
                precision={0}
                suffix="req/s"
                prefix={<RocketOutlined />}
                valueStyle={{ color: '#1677ff' }}
              />
              <Progress percent={Math.min(metrics.inference_fps / 20, 100)} showInfo={false} />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="GPU利用率"
                value={metrics.gpu_utilization}
                precision={1}
                suffix="%"
                prefix={<GpuOutlined />}
                valueStyle={{ color: metrics.gpu_utilization > 80 ? '#3f8600' : '#faad14' }}
              />
              <Progress percent={metrics.gpu_utilization} showInfo={false} />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="内存使用"
                value={metrics.memory_usage}
                precision={1}
                suffix="GB"
                prefix={<DatabaseOutlined />}
                valueStyle={{ color: metrics.memory_usage > 30 ? '#cf1322' : '#3f8600' }}
              />
              <Progress percent={Math.min(metrics.memory_usage * 2, 100)} showInfo={false} />
            </Card>
          </Col>
        </Row>

        {/* 主要功能标签页 */}
        <Card>
          <Tabs activeKey={activeTab} onChange={setActiveTab} items={[
            {
              key: 'configuration',
              label: (
                <span>
                  <SettingOutlined />
                  性能配置
                </span>
              ),
              children: (
                <Row gutter={[24, 24]}>
                  <Col span={24}>
                    <Card title="性能预设配置" extra={
                      <Space>
                        <Select
                          value={config.preset}
                          onChange={applyPreset}
                          style={{ width: 200 }}
                        >
                          <Option value="high_performance">高性能模式</Option>
                          <Option value="memory_efficient">内存优化模式</Option>
                          <Option value="development">开发调试模式</Option>
                        </Select>
                        <Button type="primary" onClick={() => form.setFieldsValue(config)}>
                          应用配置
                        </Button>
                      </Space>
                    }>
                      <Row gutter={[16, 16]}>
                        <Col span={8}>
                          <Card size="small" title={<span><GpuOutlined /> GPU配置</span>}>
                            <Form layout="vertical" size="small">
                              <Form.Item label="启用GPU加速">
                                <Switch 
                                  checked={config.gpu.enabled}
                                  onChange={(checked) => setConfig(prev => ({
                                    ...prev,
                                    gpu: { ...prev.gpu, enabled: checked }
                                  }))}
                                />
                              </Form.Item>
                              <Form.Item label="混合精度训练">
                                <Switch 
                                  checked={config.gpu.mixed_precision}
                                  onChange={(checked) => setConfig(prev => ({
                                    ...prev,
                                    gpu: { ...prev.gpu, mixed_precision: checked }
                                  }))}
                                />
                              </Form.Item>
                              <Form.Item label="XLA编译优化">
                                <Switch 
                                  checked={config.gpu.xla_compilation}
                                  onChange={(checked) => setConfig(prev => ({
                                    ...prev,
                                    gpu: { ...prev.gpu, xla_compilation: checked }
                                  }))}
                                />
                              </Form.Item>
                              <Form.Item label="批处理倍数">
                                <Slider
                                  min={1}
                                  max={8}
                                  value={config.gpu.batch_multiplier}
                                  onChange={(value) => setConfig(prev => ({
                                    ...prev,
                                    gpu: { ...prev.gpu, batch_multiplier: value }
                                  }))}
                                  marks={{ 1: '1x', 4: '4x', 8: '8x' }}
                                />
                              </Form.Item>
                            </Form>
                          </Card>
                        </Col>
                        <Col span={8}>
                          <Card size="small" title={<span><DatabaseOutlined /> 缓冲区配置</span>}>
                            <Form layout="vertical" size="small">
                              <Form.Item label="采样策略">
                                <Select
                                  value={config.buffer.strategy}
                                  onChange={(value) => setConfig(prev => ({
                                    ...prev,
                                    buffer: { ...prev.buffer, strategy: value }
                                  }))}
                                >
                                  <Option value="uniform">均匀采样</Option>
                                  <Option value="prioritized">优先级采样</Option>
                                  <Option value="recency_weighted">近期加权</Option>
                                  <Option value="curiosity_driven">好奇心驱动</Option>
                                </Select>
                              </Form.Item>
                              <Form.Item label="缓冲区容量">
                                <InputNumber
                                  value={config.buffer.capacity}
                                  onChange={(value) => setConfig(prev => ({
                                    ...prev,
                                    buffer: { ...prev.buffer, capacity: value || 10000 }
                                  }))}
                                  min={1000}
                                  max={1000000}
                                  step={10000}
                                  formatter={value => `${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')}
                                />
                              </Form.Item>
                              <Form.Item label="批次大小">
                                <Select
                                  value={config.buffer.batch_size}
                                  onChange={(value) => setConfig(prev => ({
                                    ...prev,
                                    buffer: { ...prev.buffer, batch_size: value }
                                  }))}
                                >
                                  <Option value={32}>32</Option>
                                  <Option value={64}>64</Option>
                                  <Option value={128}>128</Option>
                                  <Option value={256}>256</Option>
                                </Select>
                              </Form.Item>
                              <Form.Item label="数据压缩">
                                <Switch 
                                  checked={config.buffer.compression}
                                  onChange={(checked) => setConfig(prev => ({
                                    ...prev,
                                    buffer: { ...prev.buffer, compression: checked }
                                  }))}
                                />
                              </Form.Item>
                            </Form>
                          </Card>
                        </Col>
                        <Col span={8}>
                          <Card size="small" title={<span><CloudServerOutlined /> 分布式配置</span>}>
                            <Form layout="vertical" size="small">
                              <Form.Item label="分布式策略">
                                <Select
                                  value={config.distributed.strategy}
                                  onChange={(value) => setConfig(prev => ({
                                    ...prev,
                                    distributed: { ...prev.distributed, strategy: value }
                                  }))}
                                >
                                  <Option value="data_parallel">数据并行</Option>
                                  <Option value="model_parallel">模型并行</Option>
                                  <Option value="parameter_server">参数服务器</Option>
                                  <Option value="allreduce">AllReduce</Option>
                                </Select>
                              </Form.Item>
                              <Form.Item label="工作进程数">
                                <Slider
                                  min={1}
                                  max={8}
                                  value={config.distributed.num_workers}
                                  onChange={(value) => setConfig(prev => ({
                                    ...prev,
                                    distributed: { ...prev.distributed, num_workers: value }
                                  }))}
                                  marks={{ 1: '1', 4: '4', 8: '8' }}
                                />
                              </Form.Item>
                              <Form.Item label="同步频率">
                                <InputNumber
                                  value={config.distributed.sync_frequency}
                                  onChange={(value) => setConfig(prev => ({
                                    ...prev,
                                    distributed: { ...prev.distributed, sync_frequency: value || 100 }
                                  }))}
                                  min={10}
                                  max={1000}
                                  step={10}
                                />
                              </Form.Item>
                              <Form.Item label="梯度压缩">
                                <Switch 
                                  checked={config.distributed.compression_enabled}
                                  onChange={(checked) => setConfig(prev => ({
                                    ...prev,
                                    distributed: { ...prev.distributed, compression_enabled: checked }
                                  }))}
                                />
                              </Form.Item>
                            </Form>
                          </Card>
                        </Col>
                      </Row>
                    </Card>
                  </Col>
                </Row>
              )
            },
            {
              key: 'monitoring',
              label: (
                <span>
                  <MonitorOutlined />
                  性能监控
                </span>
              ),
              children: (
                <Row gutter={[16, 16]}>
                  <Col span={12}>
                    <Card title="性能趋势" size="small">
                      <ResponsiveContainer width="100%" height={300}>
                        <LineChart data={performanceTrendData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="time" />
                          <YAxis />
                          <RechartsTooltip />
                          <Legend />
                          <Line 
                            type="monotone" 
                            dataKey="training_fps" 
                            stroke="#8884d8" 
                            name="训练FPS"
                          />
                          <Line 
                            type="monotone" 
                            dataKey="inference_fps" 
                            stroke="#82ca9d" 
                            name="推理FPS"
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </Card>
                  </Col>
                  <Col span={12}>
                    <Card title="资源使用" size="small">
                      <ResponsiveContainer width="100%" height={300}>
                        <LineChart data={performanceTrendData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="time" />
                          <YAxis />
                          <RechartsTooltip />
                          <Legend />
                          <Line 
                            type="monotone" 
                            dataKey="memory_usage" 
                            stroke="#ffc658" 
                            name="内存使用(GB)"
                          />
                          <Line 
                            type="monotone" 
                            dataKey="gpu_utilization" 
                            stroke="#ff7300" 
                            name="GPU利用率(%)"
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </Card>
                  </Col>
                </Row>
              )
            },
            {
              key: 'benchmark',
              label: (
                <span>
                  <BarChartOutlined />
                  基准测试
                </span>
              ),
              children: (
                <Space direction="vertical" size="large" style={{ width: '100%' }}>
                  <Alert
                    message="算法性能对比"
                    description="对比不同Q-Learning算法在相同环境下的性能表现，包括收敛速度、最终性能和资源消耗"
                    variant="default"
                    showIcon
                  />
                  <Table
                    columns={benchmarkColumns}
                    dataSource={benchmarkResults}
                    rowKey="algorithm"
                    size="small"
                    pagination={false}
                  />
                  <Row gutter={16}>
                    <Col span={12}>
                      <Card title="性能对比图" size="small">
                        <ResponsiveContainer width="100%" height={250}>
                          <BarChart data={benchmarkResults}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="algorithm" />
                            <YAxis />
                            <RechartsTooltip />
                            <Bar dataKey="final_performance" fill="#8884d8" name="最终性能" />
                          </BarChart>
                        </ResponsiveContainer>
                      </Card>
                    </Col>
                    <Col span={12}>
                      <Card title="收敛速度对比" size="small">
                        <ResponsiveContainer width="100%" height={250}>
                          <BarChart data={benchmarkResults}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="algorithm" />
                            <YAxis />
                            <RechartsTooltip />
                            <Bar dataKey="convergence_time" fill="#82ca9d" name="收敛时间" />
                          </BarChart>
                        </ResponsiveContainer>
                      </Card>
                    </Col>
                  </Row>
                </Space>
              )
            },
            {
              key: 'optimization',
              label: (
                <span>
                  <ExperimentOutlined />
                  超参数优化
                </span>
              ),
              children: (
                <Space direction="vertical" size="large" style={{ width: '100%' }}>
                  <Alert
                    message="自动超参数优化"
                    description="使用Optuna贝叶斯优化算法自动搜索最优超参数配置，支持多目标优化"
                    type="success"
                    showIcon
                  />
                  <Row gutter={16}>
                    <Col span={16}>
                      <Card title="优化试验历史" size="small">
                        <Table
                          columns={optimizationColumns}
                          dataSource={optimizationTrials}
                          rowKey="trial_id"
                          size="small"
                          pagination={{ pageSize: 10 }}
                          scroll={{ y: 400 }}
                        />
                      </Card>
                    </Col>
                    <Col span={8}>
                      <Card title="优化进度" size="small">
                        <ResponsiveContainer width="100%" height={200}>
                          <LineChart data={optimizationTrials.slice(-10)}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="trial_id" />
                            <YAxis />
                            <RechartsTooltip />
                            <Line 
                              type="monotone" 
                              dataKey="objective_value" 
                              stroke="#8884d8" 
                              name="目标值"
                            />
                          </LineChart>
                        </ResponsiveContainer>
                      </Card>
                      <Card title="最佳参数" size="small" style={{ marginTop: 16 }}>
                        <List size="small">
                          <List.Item>
                            <Text strong>学习率:</Text> 0.001024
                          </List.Item>
                          <List.Item>
                            <Text strong>批次大小:</Text> 128
                          </List.Item>
                          <List.Item>
                            <Text strong>Epsilon衰减:</Text> 0.9956
                          </List.Item>
                          <List.Item>
                            <Text strong>缓冲区大小:</Text> 100,000
                          </List.Item>
                        </List>
                      </Card>
                    </Col>
                  </Row>
                </Space>
              )
            }
          ]} />
        </Card>

        {/* 性能优化技术栈介绍 */}
        <Card title="性能优化技术栈" size="small">
          <Row gutter={[16, 16]}>
            <Col span={6}>
              <Card hoverable size="small" style={{ textAlign: 'center' }}>
                <GpuOutlined style={{ fontSize: '32px', color: '#52c41a', marginBottom: '8px' }} />
                <div><strong>GPU加速训练</strong></div>
                <div style={{ fontSize: '12px', color: '#666' }}>混合精度 + XLA编译</div>
                <div style={{ fontSize: '12px', color: '#666' }}>2-4倍训练加速</div>
              </Card>
            </Col>
            <Col span={6}>
              <Card hoverable size="small" style={{ textAlign: 'center' }}>
                <DatabaseOutlined style={{ fontSize: '32px', color: '#1890ff', marginBottom: '8px' }} />
                <div><strong>优化回放缓冲区</strong></div>
                <div style={{ fontSize: '12px', color: '#666' }}>压缩存储 + 并行采样</div>
                <div style={{ fontSize: '12px', color: '#666' }}>50-70%内存节省</div>
              </Card>
            </Col>
            <Col span={6}>
              <Card hoverable size="small" style={{ textAlign: 'center' }}>
                <CloudServerOutlined style={{ fontSize: '32px', color: '#fa8c16', marginBottom: '8px' }} />
                <div><strong>分布式训练</strong></div>
                <div style={{ fontSize: '12px', color: '#666' }}>多GPU + 参数服务器</div>
                <div style={{ fontSize: '12px', color: '#666' }}>线性扩展性能</div>
              </Card>
            </Col>
            <Col span={6}>
              <Card hoverable size="small" style={{ textAlign: 'center' }}>
                <ExperimentOutlined style={{ fontSize: '32px', color: '#722ed1', marginBottom: '8px' }} />
                <div><strong>自动超参数优化</strong></div>
                <div style={{ fontSize: '12px', color: '#666' }}>Optuna贝叶斯优化</div>
                <div style={{ fontSize: '12px', color: '#666' }}>智能参数搜索</div>
              </Card>
            </Col>
          </Row>
        </Card>
      </Space>
    </div>
  );
};

export default QLearningPerformanceOptimizationPage;
