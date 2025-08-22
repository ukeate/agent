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
import { Line, Heatmap, Column } from '@ant-design/charts';
import {
  AppstoreOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  ReloadOutlined,
  CalculatorOutlined,
  FunctionOutlined,
  LineChartOutlined,
  SettingOutlined,
  RobotOutlined,
  TargetOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;

const GridWorldPage: React.FC = () => {
  const [isTraining, setIsTraining] = useState(false);
  const [episode, setEpisode] = useState(0);
  const [currentPosition, setCurrentPosition] = useState([0, 0]);
  const [gridSize, setGridSize] = useState(8);
  
  // 网格世界参数
  const [agentPosition, setAgentPosition] = useState([0, 0]);
  const [goalPosition, setGoalPosition] = useState([7, 7]);
  const [obstacles, setObstacles] = useState([[2, 2], [3, 3], [4, 4], [5, 5]]);
  
  // Q-Learning参数
  const [learningRate, setLearningRate] = useState(0.1);
  const [discountFactor, setDiscountFactor] = useState(0.95);
  const [explorationRate, setExplorationRate] = useState(0.3);
  
  // 奖励配置
  const [goalReward, setGoalReward] = useState(100);
  const [stepPenalty, setStepPenalty] = useState(-1);
  const [obstaclePenalty, setObstaclePenalty] = useState(-50);
  
  const [trainingData, setTrainingData] = useState([
    { episode: 0, reward: 0, steps: 0, success: false, qValue: 0, convergence: 0 },
    { episode: 50, reward: -45, steps: 45, success: false, qValue: 0.3, convergence: 0.1 },
    { episode: 100, reward: 55, steps: 38, success: true, qValue: 0.6, convergence: 0.3 },
    { episode: 150, reward: 75, steps: 25, success: true, qValue: 0.8, convergence: 0.5 },
    { episode: 200, reward: 88, steps: 18, success: true, qValue: 0.9, convergence: 0.7 },
    { episode: 250, reward: 95, steps: 12, success: true, qValue: 0.95, convergence: 0.85 },
  ]);

  // Q值热力图数据
  const [qValueHeatmap, setQValueHeatmap] = useState(() => {
    const data = [];
    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        data.push({
          x: i,
          y: j,
          value: Math.random() * 0.8 + 0.1, // 模拟Q值
        });
      }
    }
    return data;
  });

  const [actionHistory, setActionHistory] = useState([
    { step: 1, position: '[0,0]', action: '→ 右', reward: -1, qValue: 0.1, policy: 'ε-greedy' },
    { step: 2, position: '[1,0]', action: '↓ 下', reward: -1, qValue: 0.15, policy: 'ε-greedy' },
    { step: 3, position: '[1,1]', action: '→ 右', reward: -1, qValue: 0.2, policy: 'greedy' },
    { step: 4, position: '[2,1]', action: '↑ 上', reward: -1, qValue: 0.18, policy: 'ε-greedy' },
  ]);

  const [gridStats, setGridStats] = useState({
    totalStates: gridSize * gridSize,
    visitedStates: 24,
    optimalPathLength: 14,
    currentPathLength: 18,
    convergenceRate: 0.85,
    explorationRatio: 0.65,
  });

  const lineConfig = {
    data: trainingData,
    xField: 'episode',
    yField: 'reward',
    smooth: true,
    point: {
      size: 4,
      shape: 'diamond',
    },
    tooltip: {
      showMarkers: false,
    },
    meta: {
      reward: {
        alias: '累积奖励',
      },
    },
  };

  const heatmapConfig = {
    data: qValueHeatmap,
    xField: 'x',
    yField: 'y',
    colorField: 'value',
    reflect: 'y',
    shape: 'square',
    meta: {
      x: {
        type: 'cat',
      },
      y: {
        type: 'cat',
      },
      value: {
        min: 0,
        max: 1,
      },
    },
    xAxis: {
      position: 'top',
      tickLine: null,
      line: null,
      label: {
        offset: 12,
        style: {
          fill: '#aaa',
          fontSize: 11,
        },
      },
    },
    yAxis: {
      tickLine: null,
      line: null,
      label: {
        style: {
          fill: '#aaa',
          fontSize: 11,
        },
      },
    },
  };

  const stepsConfig = {
    data: trainingData,
    xField: 'episode',
    yField: 'steps',
    columnStyle: {
      radius: [2, 2, 0, 0],
    },
    meta: {
      steps: {
        alias: '完成步数',
      },
    },
  };

  const actionColumns = [
    {
      title: '步数',
      dataIndex: 'step',
      key: 'step',
      width: 60,
    },
    {
      title: '位置',
      dataIndex: 'position',
      key: 'position',
      render: (text: string) => <Tag color="blue">{text}</Tag>,
    },
    {
      title: '动作',
      dataIndex: 'action',
      key: 'action',
      render: (text: string) => <Text strong>{text}</Text>,
    },
    {
      title: '奖励',
      dataIndex: 'reward',
      key: 'reward',
      render: (value: number) => (
        <Text style={{ color: value > 0 ? '#52c41a' : '#f5222d' }}>
          {value > 0 ? '+' : ''}{value}
        </Text>
      ),
    },
    {
      title: 'Q值',
      dataIndex: 'qValue',
      key: 'qValue',
      render: (value: number) => <Text>{value.toFixed(3)}</Text>,
    },
    {
      title: '策略',
      dataIndex: 'policy',
      key: 'policy',
      render: (text: string) => (
        <Tag color={text === 'greedy' ? 'green' : 'orange'}>{text}</Tag>
      ),
    },
  ];

  const startTraining = () => {
    setIsTraining(true);
    const interval = setInterval(() => {
      setEpisode(prev => {
        if (prev >= 250) {
          setIsTraining(false);
          clearInterval(interval);
          return prev;
        }
        return prev + 1;
      });
    }, 100);
  };

  const resetTraining = () => {
    setIsTraining(false);
    setEpisode(0);
    setAgentPosition([0, 0]);
  };

  const renderGrid = () => {
    const grid = [];
    for (let row = 0; row < gridSize; row++) {
      const cells = [];
      for (let col = 0; col < gridSize; col++) {
        let cellContent = '';
        let cellStyle: React.CSSProperties = {
          width: '40px',
          height: '40px',
          border: '1px solid #ddd',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: '20px',
          backgroundColor: '#fff',
        };

        // 检查是否是智能体位置
        if (agentPosition[0] === col && agentPosition[1] === row) {
          cellContent = '🤖';
          cellStyle.backgroundColor = '#e6f7ff';
        }
        // 检查是否是目标位置
        else if (goalPosition[0] === col && goalPosition[1] === row) {
          cellContent = '🎯';
          cellStyle.backgroundColor = '#f6ffed';
        }
        // 检查是否是障碍物
        else if (obstacles.some(obs => obs[0] === col && obs[1] === row)) {
          cellContent = '🧱';
          cellStyle.backgroundColor = '#fff2e8';
        }
        // 普通空地，根据Q值设置颜色深度
        else {
          const qValue = qValueHeatmap.find(q => q.x === col && q.y === row)?.value || 0;
          const intensity = Math.floor(qValue * 255);
          cellStyle.backgroundColor = `rgba(24, 144, 255, ${qValue * 0.3})`;
        }

        cells.push(
          <div key={`${row}-${col}`} style={cellStyle}>
            {cellContent}
          </div>
        );
      }
      grid.push(
        <div key={row} style={{ display: 'flex' }}>
          {cells}
        </div>
      );
    }
    return grid;
  };

  return (
    <div style={{ padding: '24px', background: '#f0f2f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <AppstoreOutlined style={{ marginRight: '8px', color: '#1890ff' }} />
          GridWorld环境
        </Title>
        <Text type="secondary">
          经典网格世界环境，智能体需要从起点导航到目标点，避开障碍物
        </Text>
      </div>

      <Row gutter={[24, 24]}>
        {/* 网格世界可视化 */}
        <Col span={12}>
          <Card title="GridWorld环境" extra={
            <Space>
              <Text>网格大小: {gridSize}x{gridSize}</Text>
              <Tag color="blue">回合 {episode}</Tag>
            </Space>
          }>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <div style={{ marginBottom: '16px' }}>
                {renderGrid()}
              </div>
              <Space>
                <Space>
                  <div style={{ display: 'flex', alignItems: 'center' }}>
                    <span style={{ fontSize: '16px', marginRight: '4px' }}>🤖</span>
                    <Text type="secondary">智能体</Text>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center' }}>
                    <span style={{ fontSize: '16px', marginRight: '4px' }}>🎯</span>
                    <Text type="secondary">目标</Text>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center' }}>
                    <span style={{ fontSize: '16px', marginRight: '4px' }}>🧱</span>
                    <Text type="secondary">障碍物</Text>
                  </div>
                </Space>
              </Space>
            </div>
          </Card>
        </Col>

        {/* 训练控制 */}
        <Col span={12}>
          <Card title="训练控制与状态">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Row gutter={16}>
                <Col span={12}>
                  <Statistic title="当前回合" value={episode} />
                </Col>
                <Col span={12}>
                  <Statistic title="当前位置" value={`(${agentPosition[0]}, ${agentPosition[1]})`} />
                </Col>
              </Row>
              <Progress 
                percent={(episode / 250) * 100} 
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
              <Divider />
              <Row gutter={16}>
                <Col span={8}>
                  <Statistic
                    title="已访问状态"
                    value={gridStats.visitedStates}
                    suffix={`/ ${gridStats.totalStates}`}
                    valueStyle={{ color: '#1890ff' }}
                  />
                </Col>
                <Col span={8}>
                  <Statistic
                    title="最优路径长度"
                    value={gridStats.optimalPathLength}
                    valueStyle={{ color: '#52c41a' }}
                  />
                </Col>
                <Col span={8}>
                  <Statistic
                    title="收敛率"
                    value={gridStats.convergenceRate}
                    precision={2}
                    suffix="%"
                    valueStyle={{ color: '#faad14' }}
                  />
                </Col>
              </Row>
            </Space>
          </Card>
        </Col>

        {/* Q-Learning参数配置 */}
        <Col span={8}>
          <Card title="Q-Learning参数">
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
              <div>
                <Text strong>探索率: {explorationRate}</Text>
                <Slider
                  min={0.05}
                  max={0.9}
                  step={0.05}
                  value={explorationRate}
                  onChange={setExplorationRate}
                  marks={{ 0.05: '0.05', 0.3: '0.3', 0.6: '0.6', 0.9: '0.9' }}
                />
              </div>
            </Space>
          </Card>
        </Col>

        {/* 奖励配置 */}
        <Col span={8}>
          <Card title="奖励函数配置">
            <Space direction="vertical" style={{ width: '100%' }}>
              <div>
                <Text strong>目标奖励: {goalReward}</Text>
                <Slider
                  min={50}
                  max={200}
                  value={goalReward}
                  onChange={setGoalReward}
                  marks={{ 50: '50', 100: '100', 150: '150', 200: '200' }}
                />
              </div>
              <div>
                <Text strong>步数惩罚: {stepPenalty}</Text>
                <Slider
                  min={-5}
                  max={0}
                  value={stepPenalty}
                  onChange={setStepPenalty}
                  marks={{ '-5': '-5', '-3': '-3', '-1': '-1', '0': '0' }}
                />
              </div>
              <div>
                <Text strong>碰撞惩罚: {obstaclePenalty}</Text>
                <Slider
                  min={-100}
                  max={-10}
                  value={obstaclePenalty}
                  onChange={setObstaclePenalty}
                  marks={{ '-100': '-100', '-50': '-50', '-25': '-25', '-10': '-10' }}
                />
              </div>
            </Space>
          </Card>
        </Col>

        {/* 环境配置 */}
        <Col span={8}>
          <Card title="环境配置">
            <Space direction="vertical" style={{ width: '100%' }}>
              <div>
                <Text strong>网格大小: {gridSize}x{gridSize}</Text>
                <Slider
                  min={5}
                  max={12}
                  value={gridSize}
                  onChange={setGridSize}
                  marks={{ 5: '5x5', 8: '8x8', 10: '10x10', 12: '12x12' }}
                />
              </div>
              <Divider />
              <Space direction="vertical" size="small" style={{ width: '100%' }}>
                <Text strong>起点位置: ({agentPosition[0]}, {agentPosition[1]})</Text>
                <Text strong>目标位置: ({goalPosition[0]}, {goalPosition[1]})</Text>
                <Text strong>障碍物数量: {obstacles.length}</Text>
                <Button size="small" type="dashed">
                  编辑环境布局
                </Button>
              </Space>
            </Space>
          </Card>
        </Col>

        {/* 训练曲线 */}
        <Col span={12}>
          <Card title="奖励学习曲线">
            <div style={{ height: 300 }}>
              <Line {...lineConfig} />
            </div>
          </Card>
        </Col>

        {/* 步数统计 */}
        <Col span={12}>
          <Card title="完成步数趋势">
            <div style={{ height: 300 }}>
              <Column {...stepsConfig} />
            </div>
          </Card>
        </Col>

        {/* Q值热力图 */}
        <Col span={12}>
          <Card title="Q值热力图">
            <div style={{ height: 300 }}>
              <Heatmap {...heatmapConfig} />
            </div>
            <Alert
              message="颜色深度表示Q值大小"
              description="颜色越深表示该状态的Q值越高，智能体在该位置的期望回报越大"
              variant="default"
              showIcon
              style={{ marginTop: 16 }}
            />
          </Card>
        </Col>

        {/* 动作历史 */}
        <Col span={12}>
          <Card title="最近动作历史">
            <Table
              dataSource={actionHistory}
              columns={actionColumns}
              pagination={{ pageSize: 5 }}
              size="small"
            />
          </Card>
        </Col>

        {/* 详细配置 */}
        <Col span={24}>
          <Card title="高级设置">
            <Tabs defaultActiveKey="1">
              <TabPane tab="策略配置" key="1">
                <Row gutter={[16, 16]}>
                  <Col span={8}>
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Text strong>ε-贪心策略</Text>
                      <Switch defaultChecked />
                      <Text type="secondary">平衡探索与利用</Text>
                    </Space>
                  </Col>
                  <Col span={8}>
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Text strong>动态学习率</Text>
                      <Switch />
                      <Text type="secondary">随训练进度调整学习率</Text>
                    </Space>
                  </Col>
                  <Col span={8}>
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Text strong>经验回放</Text>
                      <Switch />
                      <Text type="secondary">存储和重用历史经验</Text>
                    </Space>
                  </Col>
                </Row>
              </TabPane>
              <TabPane tab="可视化设置" key="2">
                <Row gutter={[16, 16]}>
                  <Col span={8}>
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Text strong>显示Q值</Text>
                      <Switch defaultChecked />
                      <Text type="secondary">在网格中显示Q值大小</Text>
                    </Space>
                  </Col>
                  <Col span={8}>
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Text strong>显示轨迹</Text>
                      <Switch />
                      <Text type="secondary">显示智能体移动轨迹</Text>
                    </Space>
                  </Col>
                  <Col span={8}>
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Text strong>动画速度</Text>
                      <Slider min={1} max={10} defaultValue={5} />
                      <Text type="secondary">调整训练动画播放速度</Text>
                    </Space>
                  </Col>
                </Row>
              </TabPane>
              <TabPane tab="实验记录" key="3">
                <Alert
                  message="训练记录"
                  description="已完成25次GridWorld训练实验，平均收敛时间180回合"
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

export default GridWorldPage;