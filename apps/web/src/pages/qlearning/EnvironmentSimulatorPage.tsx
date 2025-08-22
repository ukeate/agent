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
  Input,
  Switch,
  Divider
} from 'antd';
import { Line, Column, Radar } from '@ant-design/charts';
import {
  EnvironmentOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  ReloadOutlined,
  CalculatorOutlined,
  FunctionOutlined,
  LineChartOutlined,
  SettingOutlined,
  ThunderboltOutlined,
  ExperimentOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;

const EnvironmentSimulatorPage: React.FC = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [selectedEnvironment, setSelectedEnvironment] = useState('gridworld');
  
  // 环境参数
  const [environmentSize, setEnvironmentSize] = useState(8);
  const [obstacleRatio, setObstacleRatio] = useState(0.2);
  const [rewardSparsity, setRewardSparsity] = useState(0.1);
  const [stochasticity, setStochasticity] = useState(0.1);
  
  // 智能体参数
  const [learningRate, setLearningRate] = useState(0.1);
  const [explorationRate, setExplorationRate] = useState(0.3);
  const [discountFactor, setDiscountFactor] = useState(0.95);
  
  const [simulationData, setSimulationData] = useState([
    { step: 0, reward: 0, qValue: 0, exploration: 0.3, performance: 0 },
    { step: 100, reward: 45, qValue: 0.75, exploration: 0.25, performance: 0.2 },
    { step: 200, reward: 120, qValue: 1.2, exploration: 0.18, performance: 0.45 },
    { step: 300, reward: 180, qValue: 1.6, exploration: 0.12, performance: 0.65 },
    { step: 400, reward: 220, qValue: 1.9, exploration: 0.08, performance: 0.78 },
    { step: 500, reward: 250, qValue: 2.1, exploration: 0.05, performance: 0.85 },
  ]);

  const [environmentTypes, setEnvironmentTypes] = useState([
    { 
      name: 'GridWorld',
      key: 'gridworld', 
      description: '经典网格世界环境，适合基础Q-Learning学习',
      complexity: 'low',
      features: ['离散状态', '简单动作', '确定性转移'],
      active: true 
    },
    { 
      name: 'MountainCar',
      key: 'mountaincar', 
      description: '连续状态空间的小车爬山问题',
      complexity: 'medium',
      features: ['连续状态', '物理仿真', '延迟奖励'],
      active: false 
    },
    { 
      name: 'CartPole',
      key: 'cartpole', 
      description: '倒立摆平衡控制任务',
      complexity: 'medium',
      features: ['连续控制', '平衡任务', '实时反馈'],
      active: false 
    },
    { 
      name: 'Maze',
      key: 'maze', 
      description: '复杂迷宫导航环境',
      complexity: 'high',
      features: ['复杂地形', '多目标', '动态障碍'],
      active: false 
    },
  ]);

  const [performanceMetrics, setPerformanceMetrics] = useState([
    { metric: '成功率', value: 0.85, trend: 'up', description: '任务完成的成功比例' },
    { metric: '平均步数', value: 45, trend: 'down', description: '完成任务的平均步数' },
    { metric: '累积奖励', value: 250, trend: 'up', description: '总的累积奖励值' },
    { metric: '收敛速度', value: 0.92, trend: 'up', description: '算法收敛的速度指标' },
  ]);

  const lineConfig = {
    data: simulationData,
    xField: 'step',
    yField: 'performance',
    smooth: true,
    point: {
      size: 4,
      shape: 'diamond',
    },
    tooltip: {
      showMarkers: false,
    },
    meta: {
      performance: {
        alias: '性能',
        min: 0,
        max: 1
      },
    },
  };

  const rewardConfig = {
    data: simulationData,
    xField: 'step',
    yField: 'reward',
    columnStyle: {
      radius: [2, 2, 0, 0],
    },
    meta: {
      reward: {
        alias: '累积奖励',
      },
    },
  };

  const radarData = [
    { metric: '状态复杂度', value: environmentSize * 0.1 },
    { metric: '动作多样性', value: 0.8 },
    { metric: '奖励稀疏性', value: rewardSparsity },
    { metric: '随机性', value: stochasticity },
    { metric: '可视化程度', value: 0.9 },
    { metric: '计算效率', value: 0.7 },
  ];

  const radarConfig = {
    data: radarData,
    xField: 'metric',
    yField: 'value',
    meta: {
      value: {
        alias: '评分',
        min: 0,
        max: 1,
      },
    },
    xAxis: {
      line: null,
      tickLine: null,
    },
    yAxis: {
      line: null,
      tickLine: null,
      grid: {
        alternateColor: 'rgba(0, 0, 0, 0.04)',
      },
    },
    point: {
      size: 2,
    },
    area: {
      fill: 'rgba(24, 144, 255, 0.2)',
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
      dataIndex: 'value',
      key: 'value',
      render: (value: number, record: any) => (
        <Text strong style={{ color: record.trend === 'up' ? '#52c41a' : '#faad14' }}>
          {typeof value === 'number' && value < 1 ? value.toFixed(3) : value}
        </Text>
      ),
    },
    {
      title: '趋势',
      dataIndex: 'trend',
      key: 'trend',
      render: (trend: string) => (
        <Tag color={trend === 'up' ? 'green' : trend === 'down' ? 'orange' : 'blue'}>
          {trend === 'up' ? '↗ 上升' : trend === 'down' ? '↘ 下降' : '→ 稳定'}
        </Tag>
      ),
    },
    {
      title: '说明',
      dataIndex: 'description',
      key: 'description',
    },
  ];

  const startSimulation = () => {
    setIsRunning(true);
    const interval = setInterval(() => {
      setCurrentStep(prev => {
        if (prev >= 500) {
          setIsRunning(false);
          clearInterval(interval);
          return prev;
        }
        return prev + 1;
      });
    }, 50);
  };

  const resetSimulation = () => {
    setIsRunning(false);
    setCurrentStep(0);
  };

  const toggleEnvironment = (key: string) => {
    setEnvironmentTypes(prev => 
      prev.map(env => 
        env.key === key 
          ? { ...env, active: true }
          : { ...env, active: false }
      )
    );
    setSelectedEnvironment(key);
  };

  const getComplexityColor = (complexity: string) => {
    switch (complexity) {
      case 'low': return 'green';
      case 'medium': return 'orange';
      case 'high': return 'red';
      default: return 'blue';
    }
  };

  return (
    <div style={{ padding: '24px', background: '#f0f2f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <EnvironmentOutlined style={{ marginRight: '8px', color: '#1890ff' }} />
          环境模拟器
        </Title>
        <Text type="secondary">
          配置和测试不同的强化学习环境，评估智能体的学习性能
        </Text>
      </div>

      <Row gutter={[24, 24]}>
        {/* 环境类型选择 */}
        <Col span={24}>
          <Card title="环境类型选择">
            <Row gutter={[16, 16]}>
              {environmentTypes.map(env => (
                <Col span={6} key={env.key}>
                  <Card
                    size="small"
                    hoverable
                    style={{ 
                      borderColor: env.active ? '#1890ff' : '#d9d9d9',
                      backgroundColor: env.active ? '#f6ffed' : '#fafafa'
                    }}
                    onClick={() => toggleEnvironment(env.key)}
                  >
                    <Space direction="vertical" size="small" style={{ width: '100%' }}>
                      <Space>
                        <ExperimentOutlined style={{ color: env.active ? '#52c41a' : '#999' }} />
                        <Text strong={env.active}>{env.name}</Text>
                        {env.active && <Tag color="green">激活</Tag>}
                      </Space>
                      <Tag color={getComplexityColor(env.complexity)}>
                        {env.complexity === 'low' ? '简单' : env.complexity === 'medium' ? '中等' : '复杂'}
                      </Tag>
                      <Text type="secondary" style={{ fontSize: '12px' }}>
                        {env.description}
                      </Text>
                      <div>
                        {env.features.map((feature, index) => (
                          <Tag key={index} size="small">{feature}</Tag>
                        ))}
                      </div>
                    </Space>
                  </Card>
                </Col>
              ))}
            </Row>
          </Card>
        </Col>

        {/* 仿真控制 */}
        <Col span={8}>
          <Card title="仿真控制">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Statistic title="当前步数" value={currentStep} />
              <Progress 
                percent={(currentStep / 500) * 100} 
                status={isRunning ? 'active' : 'normal'} 
              />
              <Space>
                <Button
                  type="primary"
                  icon={<PlayCircleOutlined />}
                  onClick={startSimulation}
                  disabled={isRunning}
                >
                  开始仿真
                </Button>
                <Button
                  icon={<ReloadOutlined />}
                  onClick={resetSimulation}
                  disabled={isRunning}
                >
                  重置
                </Button>
              </Space>
              <Divider />
              <Text strong>当前环境: {selectedEnvironment}</Text>
              <Text type="secondary">
                环境大小: {environmentSize}x{environmentSize}
              </Text>
            </Space>
          </Card>
        </Col>

        {/* 环境参数 */}
        <Col span={8}>
          <Card title="环境参数">
            <Space direction="vertical" style={{ width: '100%' }}>
              <div>
                <Text strong>环境大小: {environmentSize}x{environmentSize}</Text>
                <Slider
                  min={5}
                  max={15}
                  value={environmentSize}
                  onChange={setEnvironmentSize}
                  marks={{ 5: '5', 8: '8', 12: '12', 15: '15' }}
                />
              </div>
              <div>
                <Text strong>障碍物比例: {(obstacleRatio * 100).toFixed(0)}%</Text>
                <Slider
                  min={0}
                  max={0.5}
                  step={0.05}
                  value={obstacleRatio}
                  onChange={setObstacleRatio}
                  marks={{ 0: '0%', 0.2: '20%', 0.4: '40%', 0.5: '50%' }}
                />
              </div>
              <div>
                <Text strong>奖励稀疏性: {(rewardSparsity * 100).toFixed(0)}%</Text>
                <Slider
                  min={0.05}
                  max={0.5}
                  step={0.05}
                  value={rewardSparsity}
                  onChange={setRewardSparsity}
                  marks={{ 0.05: '5%', 0.2: '20%', 0.35: '35%', 0.5: '50%' }}
                />
              </div>
            </Space>
          </Card>
        </Col>

        {/* 智能体参数 */}
        <Col span={8}>
          <Card title="智能体参数">
            <Space direction="vertical" style={{ width: '100%' }}>
              <div>
                <Text strong>学习率: {learningRate}</Text>
                <Slider
                  min={0.01}
                  max={0.5}
                  step={0.01}
                  value={learningRate}
                  onChange={setLearningRate}
                  marks={{ 0.01: '0.01', 0.1: '0.1', 0.3: '0.3', 0.5: '0.5' }}
                />
              </div>
              <div>
                <Text strong>探索率: {explorationRate}</Text>
                <Slider
                  min={0.1}
                  max={0.9}
                  step={0.05}
                  value={explorationRate}
                  onChange={setExplorationRate}
                  marks={{ 0.1: '0.1', 0.3: '0.3', 0.6: '0.6', 0.9: '0.9' }}
                />
              </div>
              <div>
                <Text strong>折扣因子: {discountFactor}</Text>
                <Slider
                  min={0.8}
                  max={0.99}
                  step={0.01}
                  value={discountFactor}
                  onChange={setDiscountFactor}
                  marks={{ 0.8: '0.8', 0.9: '0.9', 0.95: '0.95', 0.99: '0.99' }}
                />
              </div>
            </Space>
          </Card>
        </Col>

        {/* 性能曲线 */}
        <Col span={12}>
          <Card title="学习性能曲线">
            <div style={{ height: 300 }}>
              <Line {...lineConfig} />
            </div>
          </Card>
        </Col>

        {/* 奖励累积 */}
        <Col span={12}>
          <Card title="奖励累积趋势">
            <div style={{ height: 300 }}>
              <Column {...rewardConfig} />
            </div>
          </Card>
        </Col>

        {/* 环境特征雷达图 */}
        <Col span={12}>
          <Card title="环境特征分析">
            <div style={{ height: 300 }}>
              <Radar {...radarConfig} />
            </div>
          </Card>
        </Col>

        {/* 性能指标表格 */}
        <Col span={12}>
          <Card title="性能指标监控">
            <Table
              dataSource={performanceMetrics}
              columns={metricsColumns}
              pagination={false}
              size="small"
            />
          </Card>
        </Col>

        {/* 详细配置 */}
        <Col span={24}>
          <Card title="高级配置">
            <Tabs defaultActiveKey="1">
              <TabPane tab="环境定制" key="1">
                <Row gutter={[16, 16]}>
                  <Col span={8}>
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Text strong>动态障碍物</Text>
                      <Switch />
                      <Text type="secondary">启用移动障碍物增加环境复杂度</Text>
                    </Space>
                  </Col>
                  <Col span={8}>
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Text strong>多目标任务</Text>
                      <Switch />
                      <Text type="secondary">设置多个目标点提高挑战性</Text>
                    </Space>
                  </Col>
                  <Col span={8}>
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Text strong>噪声观察</Text>
                      <Switch />
                      <Text type="secondary">在观察中添加噪声模拟真实环境</Text>
                    </Space>
                  </Col>
                </Row>
              </TabPane>
              <TabPane tab="实验记录" key="2">
                <Alert
                  message="实验历史"
                  description="已完成18次环境仿真实验，平均收敛时间350步"
                  type="success"
                  showIcon
                />
              </TabPane>
              <TabPane tab="数据导出" key="3">
                <Space>
                  <Button type="primary">导出性能数据</Button>
                  <Button>导出环境配置</Button>
                  <Button>生成实验报告</Button>
                </Space>
              </TabPane>
            </Tabs>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default EnvironmentSimulatorPage;