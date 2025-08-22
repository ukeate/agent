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
  Divider,
  Timeline
} from 'antd';
import { Line, Column, Heatmap, Scatter } from '@ant-design/charts';
import {
  BulbOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  ReloadOutlined,
  CalculatorOutlined,
  FunctionOutlined,
  LineChartOutlined,
  SettingOutlined,
  ExperimentOutlined,
  FireOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;

const RewardShapingPage: React.FC = () => {
  const [isTraining, setIsTraining] = useState(false);
  const [episode, setEpisode] = useState(0);
  const [shapingMethod, setShapingMethod] = useState('potential_based');
  
  // 奖励塑形参数
  const [shapingWeight, setShapingWeight] = useState(0.3);
  const [decayRate, setDecayRate] = useState(0.95);
  const [potentialThreshold, setPotentialThreshold] = useState(0.5);
  
  // 当前塑形状态
  const [currentPotential, setCurrentPotential] = useState(0.8);
  const [shapingBonus, setShapingBonus] = useState(15);
  const [originalReward, setOriginalReward] = useState(100);

  const [shapingHistory, setShapingHistory] = useState([
    { episode: 0, original: 20, shaped: 25, potential: 0.2, efficiency: 0.3 },
    { episode: 10, original: 35, shaped: 45, potential: 0.4, efficiency: 0.5 },
    { episode: 20, original: 50, shaped: 68, potential: 0.6, efficiency: 0.7 },
    { episode: 30, original: 65, shaped: 82, potential: 0.75, efficiency: 0.82 },
    { episode: 40, original: 80, shaped: 92, potential: 0.85, efficiency: 0.88 },
    { episode: 50, original: 85, shaped: 94, potential: 0.88, efficiency: 0.91 },
    { episode: 60, original: 90, shaped: 95, potential: 0.9, efficiency: 0.93 },
    { episode: 70, original: 88, shaped: 91, potential: 0.85, efficiency: 0.92 },
  ]);

  const [shapingMethods, setShapingMethods] = useState([
    {
      name: '基于势能',
      key: 'potential_based',
      description: '使用势能函数指导学习方向',
      active: true,
      effectiveness: 0.85
    },
    {
      name: '基于距离',
      key: 'distance_based', 
      description: '根据目标距离调整奖励',
      active: false,
      effectiveness: 0.72
    },
    {
      name: '分层塑形',
      key: 'hierarchical',
      description: '多层次目标导向的奖励设计',
      active: false,
      effectiveness: 0.78
    },
    {
      name: '时序塑形',
      key: 'temporal',
      description: '基于时间序列的动态塑形',
      active: false,
      effectiveness: 0.81
    }
  ]);

  const [shapingTechniques, setShapingTechniques] = useState([
    {
      key: '1',
      technique: '势能引导',
      description: '利用势能函数Φ(s)提供中间奖励信号',
      formula: 'R\'(s,a,s\') = R(s,a,s\') + γΦ(s\') - Φ(s)',
      applications: '路径规划、导航任务',
      effectiveness: '95%',
      status: '激活'
    },
    {
      key: '2',
      technique: '距离塑形',
      description: '基于目标距离的奖励调整',
      formula: 'R\'(s,a,s\') = R(s,a,s\') - α|dist(s\',goal) - dist(s,goal)|',
      applications: '机器人控制、游戏AI',
      effectiveness: '87%',
      status: '待机'
    },
    {
      key: '3',
      technique: '分层奖励',
      description: '多层次目标的层次化奖励设计',
      formula: 'R\'(s,a,s\') = Σᵢ wᵢRᵢ(s,a,s\')',
      applications: '复杂任务分解',
      effectiveness: '82%',
      status: '待机'
    },
    {
      key: '4',
      technique: '动态塑形',
      description: '随时间动态调整的塑形函数',
      formula: 'R\'(s,a,s\',t) = R(s,a,s\') + λ(t)F(s,s\',t)',
      applications: '在线学习、自适应系统',
      effectiveness: '89%',
      status: '待机'
    }
  ]);

  const potentialData = [
    { x: 0, y: 0, value: 0.1 }, { x: 0, y: 1, value: 0.2 }, { x: 0, y: 2, value: 0.3 }, { x: 0, y: 3, value: 0.4 }, { x: 0, y: 4, value: 0.5 },
    { x: 1, y: 0, value: 0.2 }, { x: 1, y: 1, value: 0.3 }, { x: 1, y: 2, value: 0.4 }, { x: 1, y: 3, value: 0.5 }, { x: 1, y: 4, value: 0.6 },
    { x: 2, y: 0, value: 0.3 }, { x: 2, y: 1, value: 0.4 }, { x: 2, y: 2, value: 0.5 }, { x: 2, y: 3, value: 0.6 }, { x: 2, y: 4, value: 0.7 },
    { x: 3, y: 0, value: 0.4 }, { x: 3, y: 1, value: 0.5 }, { x: 3, y: 2, value: 0.6 }, { x: 3, y: 3, value: 0.7 }, { x: 3, y: 4, value: 0.8 },
    { x: 4, y: 0, value: 0.5 }, { x: 4, y: 1, value: 0.6 }, { x: 4, y: 2, value: 0.7 }, { x: 4, y: 3, value: 0.8 }, { x: 4, y: 4, value: 1.0 },
  ];

  const lineConfig = {
    data: shapingHistory,
    xField: 'episode',
    yField: 'value',
    seriesField: 'type',
    smooth: true,
    color: ['#1890ff', '#52c41a'],
    point: {
      size: 3,
      shape: 'circle',
    },
    legend: {
      position: 'top' as const,
    },
  };

  const heatmapConfig = {
    data: potentialData,
    xField: 'x',
    yField: 'y',
    colorField: 'value',
    color: ['#174c83', '#7eb8dc', '#bbe6ff', '#b5ff8c', '#75c42d'],
    meta: {
      x: {
        type: 'cat',
      },
      y: {
        type: 'cat',
      },
    },
    label: {
      style: {
        fill: '#fff',
        fontSize: 10,
      },
    },
  };

  const techniqueColumns = [
    {
      title: '塑形技术',
      dataIndex: 'technique',
      key: 'technique',
      render: (text: string) => <Text strong>{text}</Text>,
    },
    {
      title: '描述',
      dataIndex: 'description',
      key: 'description',
    },
    {
      title: '数学公式',
      dataIndex: 'formula',
      key: 'formula',
      render: (formula: string) => (
        <Text code style={{ fontSize: '12px' }}>{formula}</Text>
      ),
    },
    {
      title: '应用场景',
      dataIndex: 'applications',
      key: 'applications',
      render: (apps: string) => <Tag color="blue">{apps}</Tag>,
    },
    {
      title: '有效性',
      dataIndex: 'effectiveness',
      key: 'effectiveness',
      render: (eff: string) => (
        <Progress percent={parseInt(eff)} size="small" />
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={status === '激活' ? 'green' : 'orange'}>{status}</Tag>
      ),
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
  };

  const selectShapingMethod = (method: string) => {
    setShapingMethod(method);
    setShapingMethods(prev => 
      prev.map(m => ({ ...m, active: m.key === method }))
    );
  };

  const shapedReward = originalReward + (shapingBonus * shapingWeight * currentPotential);

  return (
    <div style={{ padding: '24px', background: '#f0f2f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <BulbOutlined style={{ marginRight: '8px', color: '#1890ff' }} />
          奖励塑形技术系统
        </Title>
        <Text type="secondary">
          通过奖励塑形技术加速强化学习收敛，提供中间奖励信号指导智能体学习
        </Text>
      </div>

      <Row gutter={[24, 24]}>
        {/* 塑形方法选择 */}
        <Col span={24}>
          <Card title="奖励塑形方法">
            <Row gutter={[16, 16]}>
              {shapingMethods.map(method => (
                <Col span={6} key={method.key}>
                  <Card
                    size="small"
                    hoverable
                    style={{ 
                      borderColor: method.active ? '#1890ff' : '#d9d9d9',
                      backgroundColor: method.active ? '#f6ffed' : '#fafafa'
                    }}
                    onClick={() => selectShapingMethod(method.key)}
                  >
                    <Space direction="vertical" size="small" style={{ width: '100%' }}>
                      <Space>
                        <FireOutlined style={{ color: method.active ? '#52c41a' : '#999' }} />
                        <Text strong={method.active}>{method.name}</Text>
                        {method.active && <Tag color="green">激活</Tag>}
                      </Space>
                      <Text type="secondary" style={{ fontSize: '12px' }}>
                        {method.description}
                      </Text>
                      <Progress 
                        percent={method.effectiveness * 100} 
                        size="small" 
                        showInfo={false}
                      />
                      <Text type="secondary" style={{ fontSize: '10px' }}>
                        有效性: {(method.effectiveness * 100).toFixed(0)}%
                      </Text>
                    </Space>
                  </Card>
                </Col>
              ))}
            </Row>
          </Card>
        </Col>

        {/* 训练控制 */}
        <Col span={8}>
          <Card title="训练控制">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Statistic title="当前回合" value={episode} />
              <Statistic
                title="当前势能值"
                value={currentPotential}
                precision={3}
                valueStyle={{ color: '#52c41a' }}
              />
              <Progress percent={(episode / 80) * 100} status={isTraining ? 'active' : 'normal'} />
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

        {/* 塑形参数设置 */}
        <Col span={8}>
          <Card title="塑形参数">
            <Space direction="vertical" style={{ width: '100%' }}>
              <div>
                <Text strong>塑形权重: {shapingWeight.toFixed(2)}</Text>
                <Slider
                  min={0}
                  max={1}
                  step={0.05}
                  value={shapingWeight}
                  onChange={setShapingWeight}
                  marks={{ 0: '0', 0.5: '0.5', 1: '1' }}
                />
              </div>
              <div>
                <Text strong>衰减率: {decayRate.toFixed(2)}</Text>
                <Slider
                  min={0.8}
                  max={1}
                  step={0.01}
                  value={decayRate}
                  onChange={setDecayRate}
                  marks={{ 0.8: '0.8', 0.9: '0.9', 0.95: '0.95', 1: '1' }}
                />
              </div>
              <div>
                <Text strong>势能阈值: {potentialThreshold.toFixed(2)}</Text>
                <Slider
                  min={0.1}
                  max={0.9}
                  step={0.1}
                  value={potentialThreshold}
                  onChange={setPotentialThreshold}
                  marks={{ 0.1: '0.1', 0.5: '0.5', 0.9: '0.9' }}
                />
              </div>
            </Space>
          </Card>
        </Col>

        {/* 奖励对比 */}
        <Col span={8}>
          <Card title="奖励对比">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Statistic
                title="原始奖励"
                value={originalReward}
                valueStyle={{ color: '#999' }}
                suffix="分"
              />
              <Statistic
                title="塑形奖励"
                value={shapedReward}
                precision={1}
                valueStyle={{ color: '#1890ff' }}
                suffix="分"
              />
              <Statistic
                title="提升幅度"
                value={((shapedReward - originalReward) / originalReward * 100)}
                precision={1}
                valueStyle={{ color: '#52c41a' }}
                suffix="%"
              />
              <div>
                <Text type="secondary">塑形方法: </Text>
                <Tag color="blue">{shapingMethod}</Tag>
              </div>
            </Space>
          </Card>
        </Col>

        {/* 奖励对比趋势 */}
        <Col span={12}>
          <Card title="奖励对比趋势">
            <div style={{ height: 300 }}>
              <Line 
                {...lineConfig} 
                data={shapingHistory.map(item => [
                  { ...item, type: '原始奖励', value: item.original },
                  { ...item, type: '塑形奖励', value: item.shaped }
                ]).flat()}
              />
            </div>
          </Card>
        </Col>

        {/* 势能函数热力图 */}
        <Col span={12}>
          <Card title="势能函数分布">
            <div style={{ height: 300 }}>
              <Heatmap {...heatmapConfig} />
            </div>
            <Text type="secondary" style={{ fontSize: '12px' }}>
              * 颜色越深表示势能值越高，引导智能体向目标移动
            </Text>
          </Card>
        </Col>

        {/* 塑形技术详情表格 */}
        <Col span={24}>
          <Card title="奖励塑形技术详情">
            <Alert
              message="技术说明"
              description="奖励塑形通过修改原始奖励函数，提供额外的学习信号，加速智能体的学习过程"
              variant="default"
              showIcon
              style={{ marginBottom: 16 }}
            />
            <Table
              dataSource={shapingTechniques}
              columns={techniqueColumns}
              pagination={false}
              size="middle"
            />
          </Card>
        </Col>

        {/* 实验配置和历史 */}
        <Col span={24}>
          <Card title="实验配置与历史">
            <Tabs defaultActiveKey="1">
              <TabPane tab="参数配置" key="1">
                <Row gutter={[16, 16]}>
                  <Col span={8}>
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Text strong>原始奖励基数</Text>
                      <Slider
                        min={50}
                        max={200}
                        value={originalReward}
                        onChange={setOriginalReward}
                        marks={{ 50: '50', 100: '100', 150: '150', 200: '200' }}
                      />
                    </Space>
                  </Col>
                  <Col span={8}>
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Text strong>塑形奖励基数</Text>
                      <Slider
                        min={5}
                        max={50}
                        value={shapingBonus}
                        onChange={setShapingBonus}
                        marks={{ 5: '5', 15: '15', 30: '30', 50: '50' }}
                      />
                    </Space>
                  </Col>
                  <Col span={8}>
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Text strong>当前势能</Text>
                      <Slider
                        min={0}
                        max={1}
                        step={0.1}
                        value={currentPotential}
                        onChange={setCurrentPotential}
                        marks={{ 0: '0', 0.5: '0.5', 1: '1' }}
                      />
                    </Space>
                  </Col>
                </Row>
              </TabPane>
              <TabPane tab="实验历史" key="2">
                <Timeline>
                  <Timeline.Item color="green">
                    <Text strong>实验 #1</Text><br />
                    <Text type="secondary">基于势能的奖励塑形 - 收敛提升35%</Text>
                  </Timeline.Item>
                  <Timeline.Item color="blue">
                    <Text strong>实验 #2</Text><br />
                    <Text type="secondary">距离导向的塑形策略 - 稳定性提升20%</Text>
                  </Timeline.Item>
                  <Timeline.Item color="orange">
                    <Text strong>实验 #3</Text><br />
                    <Text type="secondary">分层奖励塑形 - 复杂任务成功率提升45%</Text>
                  </Timeline.Item>
                  <Timeline.Item>
                    <Text strong>实验 #4</Text><br />
                    <Text type="secondary">动态塑形策略 - 进行中...</Text>
                  </Timeline.Item>
                </Timeline>
              </TabPane>
              <TabPane tab="效果评估" key="3">
                <Row gutter={[16, 16]}>
                  <Col span={6}>
                    <Statistic
                      title="收敛速度提升"
                      value={35}
                      precision={0}
                      valueStyle={{ color: '#52c41a' }}
                      suffix="%"
                    />
                  </Col>
                  <Col span={6}>
                    <Statistic
                      title="平均奖励提升"
                      value={28}
                      precision={0}
                      valueStyle={{ color: '#1890ff' }}
                      suffix="%"
                    />
                  </Col>
                  <Col span={6}>
                    <Statistic
                      title="训练稳定性"
                      value={92}
                      precision={0}
                      valueStyle={{ color: '#faad14' }}
                      suffix="%"
                    />
                  </Col>
                  <Col span={6}>
                    <Statistic
                      title="成功率提升"
                      value={42}
                      precision={0}
                      valueStyle={{ color: '#f5222d' }}
                      suffix="%"
                    />
                  </Col>
                </Row>
              </TabPane>
            </Tabs>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default RewardShapingPage;