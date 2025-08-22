import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Row, 
  Col, 
  Typography, 
  Space, 
  Button, 
  Slider,
  Select,
  Alert,
  Progress,
  Statistic,
  Tag,
  Tabs,
  Table,
  Switch,
  Divider
} from 'antd';
import { Line, Column, Area } from '@ant-design/charts';
import {
  LineChartOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  ReloadOutlined,
  CalculatorOutlined,
  FunctionOutlined,
  SettingOutlined,
  LineChartOutlined as TrendingUpOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;

const LearningRateSchedulerPage: React.FC = () => {
  const [isTraining, setIsTraining] = useState(false);
  const [episode, setEpisode] = useState(0);
  const [currentLearningRate, setCurrentLearningRate] = useState(0.1);
  const [schedulerType, setSchedulerType] = useState('exponential');
  
  // 调度器参数
  const [initialLearningRate, setInitialLearningRate] = useState(0.1);
  const [decayRate, setDecayRate] = useState(0.95);
  const [stepSize, setStepSize] = useState(50);
  const [minLearningRate, setMinLearningRate] = useState(0.001);
  
  const [learningRateHistory, setLearningRateHistory] = useState([
    { episode: 0, learningRate: 0.1, performance: 0.2, convergence: 0.1 },
    { episode: 10, learningRate: 0.095, performance: 0.35, convergence: 0.25 },
    { episode: 20, learningRate: 0.09, performance: 0.55, convergence: 0.4 },
    { episode: 30, learningRate: 0.085, performance: 0.68, convergence: 0.55 },
    { episode: 40, learningRate: 0.08, performance: 0.78, convergence: 0.7 },
    { episode: 50, learningRate: 0.076, performance: 0.85, convergence: 0.8 },
    { episode: 60, learningRate: 0.072, performance: 0.88, convergence: 0.85 },
    { episode: 70, learningRate: 0.068, performance: 0.91, convergence: 0.9 },
  ]);

  const [schedulerConfigs, setSchedulerConfigs] = useState([
    { 
      name: '指数衰减', 
      key: 'exponential',
      description: 'lr = initial_lr * decay_rate^(episode/step_size)',
      active: true,
      formula: 'lr = 0.1 × 0.95^(t/50)'
    },
    { 
      name: '步进衰减', 
      key: 'step',
      description: '每隔固定步数降低学习率',
      active: false,
      formula: 'lr = lr × 0.9 每50步'
    },
    { 
      name: '余弦退火', 
      key: 'cosine',
      description: '使用余弦函数平滑调整学习率',
      active: false,
      formula: 'lr = min_lr + 0.5×(max_lr-min_lr)×(1+cos(π×t/T))'
    },
    { 
      name: '自适应调度', 
      key: 'adaptive',
      description: '根据性能变化动态调整学习率',
      active: false,
      formula: '基于性能梯度自动调整'
    },
  ]);

  const [performanceMetrics, setPerformanceMetrics] = useState([
    {
      key: '1',
      metric: '收敛速度',
      currentValue: 0.85,
      optimalRange: '0.7 - 0.9',
      status: 'optimal',
      description: '算法收敛到最优解的速度'
    },
    {
      key: '2',
      metric: '稳定性',
      currentValue: 0.92,
      optimalRange: '0.8 - 1.0',
      status: 'optimal',
      description: '学习过程的稳定程度'
    },
    {
      key: '3',
      metric: '探索效率',
      currentValue: 0.73,
      optimalRange: '0.6 - 0.8',
      status: 'optimal',
      description: '探索新状态的效率'
    },
    {
      key: '4',
      metric: '过拟合风险',
      currentValue: 0.15,
      optimalRange: '0.0 - 0.2',
      status: 'warning',
      description: '模型过拟合的风险程度'
    },
  ]);

  const lineConfig = {
    data: learningRateHistory,
    xField: 'episode',
    yField: 'learningRate',
    smooth: true,
    point: {
      size: 4,
      shape: 'diamond',
    },
    tooltip: {
      showMarkers: false,
    },
    meta: {
      learningRate: {
        alias: '学习率',
        min: 0,
        max: 0.2
      },
    },
    annotations: [
      {
        type: 'line',
        start: ['min', minLearningRate],
        end: ['max', minLearningRate],
        style: {
          stroke: '#ff4d4f',
          lineDash: [4, 4],
        },
      },
    ],
  };

  const performanceConfig = {
    data: learningRateHistory,
    xField: 'episode',
    yField: 'performance',
    smooth: true,
    color: '#52c41a',
    point: {
      size: 4,
      shape: 'circle',
    },
    meta: {
      performance: {
        alias: '性能',
        min: 0,
        max: 1
      },
    },
  };

  const convergenceConfig = {
    data: learningRateHistory,
    xField: 'episode',
    yField: 'convergence',
    smooth: true,
    areaStyle: {
      fill: 'l(270) 0:#ffffff 0.5:#7ec2f3 1:#1890ff',
    },
    meta: {
      convergence: {
        alias: '收敛指标',
      },
    },
  };

  const metricsColumns = [
    {
      title: '性能指标',
      dataIndex: 'metric',
      key: 'metric',
    },
    {
      title: '当前值',
      dataIndex: 'currentValue',
      key: 'currentValue',
      render: (value: number) => <Text strong>{value.toFixed(3)}</Text>,
    },
    {
      title: '最优范围',
      dataIndex: 'optimalRange',
      key: 'optimalRange',
      render: (range: string) => <Tag color="blue">{range}</Tag>,
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={status === 'optimal' ? 'green' : 'orange'}>
          {status === 'optimal' ? '最优' : '警告'}
        </Tag>
      ),
    },
    {
      title: '说明',
      dataIndex: 'description',
      key: 'description',
    },
  ];

  const startTraining = () => {
    setIsTraining(true);
    const interval = setInterval(() => {
      setEpisode(prev => {
        if (prev >= 80) {
          setIsTraining(false);
          clearInterval(interval);
          return prev;
        }
        // 根据选择的调度器更新学习率
        const newEpisode = prev + 1;
        let newLearningRate = currentLearningRate;
        
        switch (schedulerType) {
          case 'exponential':
            newLearningRate = initialLearningRate * Math.pow(decayRate, newEpisode / stepSize);
            break;
          case 'step':
            if (newEpisode % stepSize === 0) {
              newLearningRate = currentLearningRate * decayRate;
            }
            break;
          case 'cosine':
            newLearningRate = minLearningRate + 0.5 * (initialLearningRate - minLearningRate) * 
              (1 + Math.cos(Math.PI * newEpisode / 80));
            break;
          case 'adaptive':
            // 简化的自适应逻辑
            const performanceChange = Math.random() * 0.1 - 0.05;
            if (performanceChange < 0) {
              newLearningRate = Math.max(minLearningRate, currentLearningRate * 0.95);
            }
            break;
        }
        
        setCurrentLearningRate(Math.max(minLearningRate, newLearningRate));
        return newEpisode;
      });
    }, 200);
  };

  const resetTraining = () => {
    setIsTraining(false);
    setEpisode(0);
    setCurrentLearningRate(initialLearningRate);
  };

  const toggleScheduler = (key: string) => {
    setSchedulerConfigs(prev => 
      prev.map(config => 
        config.key === key 
          ? { ...config, active: !config.active }
          : { ...config, active: false }
      )
    );
    setSchedulerType(key);
  };

  return (
    <div style={{ padding: '24px', background: '#f0f2f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <LineChartOutlined style={{ marginRight: '8px', color: '#1890ff' }} />
          学习率调度系统
        </Title>
        <Text type="secondary">
          智能调整学习率以优化训练效果，平衡收敛速度与稳定性
        </Text>
      </div>

      <Row gutter={[24, 24]}>
        {/* 调度器选择 */}
        <Col span={24}>
          <Card title="学习率调度策略">
            <Row gutter={[16, 16]}>
              {schedulerConfigs.map(config => (
                <Col span={6} key={config.key}>
                  <Card
                    size="small"
                    hoverable
                    style={{ 
                      borderColor: config.active ? '#1890ff' : '#d9d9d9',
                      backgroundColor: config.active ? '#f6ffed' : '#fafafa'
                    }}
                    onClick={() => toggleScheduler(config.key)}
                  >
                    <Space direction="vertical" size="small">
                      <Space>
                        <TrendingUpOutlined style={{ color: config.active ? '#52c41a' : '#999' }} />
                        <Text strong={config.active}>{config.name}</Text>
                        {config.active && <Tag color="green">激活</Tag>}
                      </Space>
                      <Text type="secondary" style={{ fontSize: '12px' }}>
                        {config.description}
                      </Text>
                      <Text code style={{ fontSize: '10px' }}>
                        {config.formula}
                      </Text>
                    </Space>
                  </Card>
                </Col>
              ))}
            </Row>
          </Card>
        </Col>

        {/* 训练控制和状态 */}
        <Col span={8}>
          <Card title="训练控制">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Statistic title="当前回合" value={episode} />
              <Statistic
                title="当前学习率"
                value={currentLearningRate}
                precision={6}
                valueStyle={{ color: '#1890ff' }}
              />
              <Progress 
                percent={(episode / 80) * 100} 
                status={isTraining ? 'active' : 'normal'} 
              />
              <Space>
                <Button
                  type="primary"
                  icon={<PlayCircleOutlined />}
                  onClick={startTraining}
                  disabled={isTraining}
                >
                  开始训练
                </Button>
                <Button
                  icon={<ReloadOutlined />}
                  onClick={resetTraining}
                  disabled={isTraining}
                >
                  重置
                </Button>
              </Space>
            </Space>
          </Card>
        </Col>

        {/* 调度器参数 */}
        <Col span={8}>
          <Card title="调度器参数">
            <Space direction="vertical" style={{ width: '100%' }}>
              <div>
                <Text strong>初始学习率: {initialLearningRate}</Text>
                <Slider
                  min={0.001}
                  max={0.5}
                  step={0.001}
                  value={initialLearningRate}
                  onChange={setInitialLearningRate}
                  marks={{ 0.001: '0.001', 0.1: '0.1', 0.3: '0.3', 0.5: '0.5' }}
                />
              </div>
              <div>
                <Text strong>衰减率: {decayRate}</Text>
                <Slider
                  min={0.8}
                  max={0.99}
                  step={0.01}
                  value={decayRate}
                  onChange={setDecayRate}
                  marks={{ 0.8: '0.8', 0.9: '0.9', 0.95: '0.95', 0.99: '0.99' }}
                />
              </div>
              <div>
                <Text strong>衰减步长: {stepSize}</Text>
                <Slider
                  min={10}
                  max={100}
                  value={stepSize}
                  onChange={setStepSize}
                  marks={{ 10: '10', 25: '25', 50: '50', 100: '100' }}
                />
              </div>
              <div>
                <Text strong>最小学习率: {minLearningRate}</Text>
                <Slider
                  min={0.0001}
                  max={0.01}
                  step={0.0001}
                  value={minLearningRate}
                  onChange={setMinLearningRate}
                  marks={{ 0.0001: '0.0001', 0.001: '0.001', 0.005: '0.005', 0.01: '0.01' }}
                />
              </div>
            </Space>
          </Card>
        </Col>

        {/* 实时统计 */}
        <Col span={8}>
          <Card title="实时统计">
            <Row gutter={[16, 16]}>
              <Col span={12}>
                <Statistic
                  title="学习率衰减"
                  value={((initialLearningRate - currentLearningRate) / initialLearningRate * 100)}
                  precision={1}
                  suffix="%"
                  valueStyle={{ color: '#faad14' }}
                />
              </Col>
              <Col span={12}>
                <Statistic
                  title="收敛进度"
                  value={85}
                  suffix="%"
                  valueStyle={{ color: '#52c41a' }}
                />
              </Col>
              <Col span={12}>
                <Statistic
                  title="训练效率"
                  value={0.92}
                  precision={2}
                  valueStyle={{ color: '#1890ff' }}
                />
              </Col>
              <Col span={12}>
                <Statistic
                  title="调度策略"
                  value={schedulerType}
                  valueStyle={{ color: '#722ed1' }}
                />
              </Col>
            </Row>
          </Card>
        </Col>

        {/* 学习率变化曲线 */}
        <Col span={12}>
          <Card title="学习率变化曲线">
            <div style={{ height: 300 }}>
              <Line {...lineConfig} />
            </div>
          </Card>
        </Col>

        {/* 性能变化曲线 */}
        <Col span={12}>
          <Card title="性能变化曲线">
            <div style={{ height: 300 }}>
              <Line {...performanceConfig} />
            </div>
          </Card>
        </Col>

        {/* 收敛指标 */}
        <Col span={24}>
          <Card title="收敛分析">
            <div style={{ height: 200, marginBottom: 16 }}>
              <Area {...convergenceConfig} />
            </div>
            <Alert
              message="收敛状态良好"
              description="当前学习率调度策略有效促进了模型收敛，建议继续使用当前配置"
              type="success"
              showIcon
            />
          </Card>
        </Col>

        {/* 性能指标表格 */}
        <Col span={24}>
          <Card title="性能指标监控">
            <Table
              dataSource={performanceMetrics}
              columns={metricsColumns}
              pagination={false}
              size="middle"
            />
          </Card>
        </Col>

        {/* 高级配置 */}
        <Col span={24}>
          <Card title="高级配置">
            <Tabs defaultActiveKey="1">
              <TabPane tab="自适应调度" key="1">
                <Row gutter={[16, 16]}>
                  <Col span={8}>
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Text strong>性能监控</Text>
                      <Switch defaultChecked />
                      <Text type="secondary">根据性能变化自动调整学习率</Text>
                    </Space>
                  </Col>
                  <Col span={8}>
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Text strong>梯度监控</Text>
                      <Switch defaultChecked />
                      <Text type="secondary">监控梯度变化调整学习率</Text>
                    </Space>
                  </Col>
                  <Col span={8}>
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Text strong>动态范围调整</Text>
                      <Switch />
                      <Text type="secondary">动态调整学习率范围</Text>
                    </Space>
                  </Col>
                </Row>
              </TabPane>
              <TabPane tab="调度历史" key="2">
                <Alert
                  message="调度记录"
                  description="已执行39次学习率调度，平均每20个回合调整一次，有效提升训练效率23%"
                  variant="default"
                  showIcon
                />
              </TabPane>
              <TabPane tab="预设方案" key="3">
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Button type="dashed" block>保存当前配置为预设方案</Button>
                  <Button type="dashed" block>加载经典指数衰减方案</Button>
                  <Button type="dashed" block>加载自适应调度方案</Button>
                </Space>
              </TabPane>
            </Tabs>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default LearningRateSchedulerPage;