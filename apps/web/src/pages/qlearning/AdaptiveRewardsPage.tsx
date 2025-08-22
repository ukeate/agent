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
import { Line, Column, Area } from '@ant-design/charts';
import {
  ControlOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  ReloadOutlined,
  CalculatorOutlined,
  FunctionOutlined,
  LineChartOutlined,
  SettingOutlined,
  RobotOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;

const AdaptiveRewardsPage: React.FC = () => {
  const [isTraining, setIsTraining] = useState(false);
  const [episode, setEpisode] = useState(0);
  const [adaptiveMode, setAdaptiveMode] = useState('performance_based');
  
  // 自适应策略参数
  const [adaptationRate, setAdaptationRate] = useState(0.1);
  const [performanceThreshold, setPerformanceThreshold] = useState(0.8);
  const [stabilityWindow, setStabilityWindow] = useState(10);
  
  // 当前奖励参数
  const [currentReward, setCurrentReward] = useState(100);
  const [currentPenalty, setCurrentPenalty] = useState(-10);
  const [learningRate, setLearningRate] = useState(0.1);
  
  const [performanceHistory, setPerformanceHistory] = useState([
    { episode: 0, performance: 0.2, reward: 100, penalty: -10, adaptationSignal: 0 },
    { episode: 10, performance: 0.35, reward: 105, penalty: -12, adaptationSignal: 0.15 },
    { episode: 20, performance: 0.55, reward: 110, penalty: -15, adaptationSignal: 0.2 },
    { episode: 30, performance: 0.72, reward: 115, penalty: -18, adaptationSignal: 0.17 },
    { episode: 40, performance: 0.85, reward: 120, penalty: -20, adaptationSignal: 0.13 },
    { episode: 50, performance: 0.88, reward: 118, penalty: -18, adaptationSignal: 0.03 },
    { episode: 60, performance: 0.91, reward: 115, penalty: -15, adaptationSignal: 0.03 },
    { episode: 70, performance: 0.89, reward: 112, penalty: -12, adaptationSignal: -0.02 },
  ]);

  const [adaptationStrategies, setAdaptationStrategies] = useState([
    { 
      name: '基于性能', 
      key: 'performance_based',
      description: '根据智能体性能表现自动调整奖励参数',
      active: true 
    },
    { 
      name: '基于梯度', 
      key: 'gradient_based',
      description: '使用梯度信息优化奖励函数参数',
      active: false 
    },
    { 
      name: '基于方差', 
      key: 'variance_based',
      description: '根据奖励方差调整奖励分布',
      active: false 
    },
    { 
      name: '多目标平衡', 
      key: 'multi_objective',
      description: '平衡多个目标函数的权重',
      active: false 
    },
  ]);

  const [rewardAdjustmentRules, setRewardAdjustmentRules] = useState([
    {
      key: '1',
      condition: 'performance < 0.5',
      action: '增加正奖励 +10%',
      trigger: '性能低于阈值',
      status: '激活',
      frequency: 15
    },
    {
      key: '2', 
      condition: 'performance > 0.9',
      action: '减少奖励 -5%',
      trigger: '性能过高，避免过拟合',
      status: '待机',
      frequency: 3
    },
    {
      key: '3',
      condition: 'variance > threshold',
      action: '增加稳定性惩罚',
      trigger: '奖励方差过大',
      status: '激活',
      frequency: 8
    },
    {
      key: '4',
      condition: 'convergence detected',
      action: '引入探索奖励',
      trigger: '检测到收敛',
      status: '待机', 
      frequency: 1
    },
  ]);

  const lineConfig = {
    data: performanceHistory,
    xField: 'episode',
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
    annotations: [
      {
        type: 'line',
        start: ['min', performanceThreshold],
        end: ['max', performanceThreshold],
        style: {
          stroke: '#ff4d4f',
          lineDash: [4, 4],
        },
      },
    ],
  };

  const areaConfig = {
    data: performanceHistory,
    xField: 'episode',
    yField: 'adaptationSignal',
    smooth: true,
    areaStyle: {
      fill: 'l(270) 0:#ffffff 0.5:#7ec2f3 1:#1890ff',
    },
    meta: {
      adaptationSignal: {
        alias: '自适应信号',
      },
    },
  };

  const rewardConfig = {
    data: performanceHistory,
    isGroup: true,
    xField: 'episode',
    yField: 'value',
    seriesField: 'type',
    color: ['#1890ff', '#f5222d'],
    columnStyle: {
      radius: [2, 2, 0, 0],
    },
  };

  const rulesColumns = [
    {
      title: '触发条件',
      dataIndex: 'condition',
      key: 'condition',
      render: (text: string) => <Tag color="blue">{text}</Tag>,
    },
    {
      title: '调整动作',
      dataIndex: 'action',
      key: 'action',
    },
    {
      title: '触发原因',
      dataIndex: 'trigger',
      key: 'trigger',
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={status === '激活' ? 'green' : 'orange'}>{status}</Tag>
      ),
    },
    {
      title: '触发次数',
      dataIndex: 'frequency',
      key: 'frequency',
      render: (freq: number) => <Text>{freq}</Text>,
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
        return prev + 1;
      });
    }, 150);
  };

  const resetTraining = () => {
    setIsTraining(false);
    setEpisode(0);
    setCurrentReward(100);
    setCurrentPenalty(-10);
  };

  const toggleStrategy = (key: string) => {
    setAdaptationStrategies(prev => 
      prev.map(strategy => 
        strategy.key === key 
          ? { ...strategy, active: !strategy.active }
          : { ...strategy, active: false }
      )
    );
    setAdaptiveMode(key);
  };

  const currentPerformance = performanceHistory[Math.min(Math.floor(episode / 10), performanceHistory.length - 1)]?.performance || 0;

  return (
    <div style={{ padding: '24px', background: '#f0f2f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <ControlOutlined style={{ marginRight: '8px', color: '#1890ff' }} />
          自适应奖励调整系统
        </Title>
        <Text type="secondary">
          根据智能体学习进度和性能表现，动态调整奖励函数参数
        </Text>
      </div>

      <Row gutter={[24, 24]}>
        {/* 自适应策略选择 */}
        <Col span={24}>
          <Card title="自适应策略配置">
            <Row gutter={[16, 16]}>
              {adaptationStrategies.map(strategy => (
                <Col span={6} key={strategy.key}>
                  <Card
                    size="small"
                    hoverable
                    style={{ 
                      borderColor: strategy.active ? '#1890ff' : '#d9d9d9',
                      backgroundColor: strategy.active ? '#f6ffed' : '#fafafa'
                    }}
                    onClick={() => toggleStrategy(strategy.key)}
                  >
                    <Space direction="vertical" size="small">
                      <Space>
                        <RobotOutlined style={{ color: strategy.active ? '#52c41a' : '#999' }} />
                        <Text strong={strategy.active}>{strategy.name}</Text>
                        {strategy.active && <Tag color="green">激活</Tag>}
                      </Space>
                      <Text type="secondary" style={{ fontSize: '12px' }}>
                        {strategy.description}
                      </Text>
                    </Space>
                  </Card>
                </Col>
              ))}
            </Row>
          </Card>
        </Col>

        {/* 训练控制和实时参数 */}
        <Col span={8}>
          <Card title="训练控制">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Statistic title="当前回合" value={episode} />
              <Statistic
                title="当前性能"
                value={currentPerformance}
                precision={3}
                valueStyle={{ 
                  color: currentPerformance > performanceThreshold ? '#52c41a' : '#faad14' 
                }}
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

        {/* 自适应参数设置 */}
        <Col span={8}>
          <Card title="自适应参数">
            <Space direction="vertical" style={{ width: '100%' }}>
              <div>
                <Text strong>适应率: {adaptationRate}</Text>
                <Slider
                  min={0.01}
                  max={0.5}
                  step={0.01}
                  value={adaptationRate}
                  onChange={setAdaptationRate}
                  marks={{ 0.01: '0.01', 0.1: '0.1', 0.3: '0.3', 0.5: '0.5' }}
                />
              </div>
              <div>
                <Text strong>性能阈值: {performanceThreshold}</Text>
                <Slider
                  min={0.5}
                  max={0.95}
                  step={0.05}
                  value={performanceThreshold}
                  onChange={setPerformanceThreshold}
                  marks={{ 0.5: '0.5', 0.7: '0.7', 0.8: '0.8', 0.95: '0.95' }}
                />
              </div>
              <div>
                <Text strong>稳定性窗口: {stabilityWindow}</Text>
                <Slider
                  min={5}
                  max={20}
                  value={stabilityWindow}
                  onChange={setStabilityWindow}
                  marks={{ 5: '5', 10: '10', 15: '15', 20: '20' }}
                />
              </div>
            </Space>
          </Card>
        </Col>

        {/* 当前奖励参数 */}
        <Col span={8}>
          <Card title="当前奖励参数">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Statistic
                title="基础奖励"
                value={currentReward}
                valueStyle={{ color: '#1890ff' }}
                suffix="分"
              />
              <Statistic
                title="基础惩罚"
                value={currentPenalty}
                valueStyle={{ color: '#f5222d' }}
                suffix="分"
              />
              <Statistic
                title="学习率"
                value={learningRate}
                precision={3}
                valueStyle={{ color: '#52c41a' }}
              />
              <div>
                <Text type="secondary">自适应模式: </Text>
                <Tag color="blue">{adaptiveMode}</Tag>
              </div>
            </Space>
          </Card>
        </Col>

        {/* 性能趋势图 */}
        <Col span={12}>
          <Card title="性能表现趋势">
            <div style={{ height: 300 }}>
              <Line {...lineConfig} />
            </div>
          </Card>
        </Col>

        {/* 自适应信号图 */}
        <Col span={12}>
          <Card title="自适应调整信号">
            <div style={{ height: 300 }}>
              <Area {...areaConfig} />
            </div>
          </Card>
        </Col>

        {/* 调整规则表格 */}
        <Col span={24}>
          <Card title="自适应调整规则">
            <Alert
              message="规则说明"
              description="系统会根据以下规则自动调整奖励参数，确保智能体的持续学习和性能优化"
              variant="default"
              showIcon
              style={{ marginBottom: 16 }}
            />
            <Table
              dataSource={rewardAdjustmentRules}
              columns={rulesColumns}
              pagination={false}
              size="middle"
            />
          </Card>
        </Col>

        {/* 详细配置 */}
        <Col span={24}>
          <Card title="高级配置">
            <Tabs defaultActiveKey="1">
              <TabPane tab="奖励曲线调整" key="1">
                <Row gutter={[16, 16]}>
                  <Col span={8}>
                    <Text strong>奖励基数调整</Text>
                    <Slider
                      min={50}
                      max={200}
                      value={currentReward}
                      onChange={setCurrentReward}
                      marks={{ 50: '50', 100: '100', 150: '150', 200: '200' }}
                    />
                  </Col>
                  <Col span={8}>
                    <Text strong>惩罚基数调整</Text>
                    <Slider
                      min={-50}
                      max={0}
                      value={currentPenalty}
                      onChange={setCurrentPenalty}
                      marks={{ '-50': '-50', '-25': '-25', '-10': '-10', '0': '0' }}
                    />
                  </Col>
                  <Col span={8}>
                    <Text strong>学习率调整</Text>
                    <Slider
                      min={0.001}
                      max={0.5}
                      step={0.001}
                      value={learningRate}
                      onChange={setLearningRate}
                      marks={{ 0.001: '0.001', 0.1: '0.1', 0.3: '0.3', 0.5: '0.5' }}
                    />
                  </Col>
                </Row>
              </TabPane>
              <TabPane tab="触发条件设置" key="2">
                <Row gutter={[16, 16]}>
                  <Col span={12}>
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Text strong>性能下降触发</Text>
                      <Switch defaultChecked />
                      <Text type="secondary">当性能连续下降时触发奖励增强</Text>
                    </Space>
                  </Col>
                  <Col span={12}>
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Text strong>收敛检测</Text>
                      <Switch defaultChecked />
                      <Text type="secondary">检测到收敛时引入探索机制</Text>
                    </Space>
                  </Col>
                </Row>
              </TabPane>
              <TabPane tab="历史记录" key="3">
                <Alert
                  message="调整历史"
                  description="系统已进行了23次自适应调整，平均每15个回合调整一次"
                  type="success"
                  showIcon
                />
              </TabPane>
            </Tabs>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default AdaptiveRewardsPage;