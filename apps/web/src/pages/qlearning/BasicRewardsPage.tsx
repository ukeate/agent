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
  Switch
} from 'antd';
import { Line, Column } from '@ant-design/charts';
import {
  ThunderboltOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  ReloadOutlined,
  CalculatorOutlined,
  FunctionOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;

const BasicRewardsPage: React.FC = () => {
  const [rewardType, setRewardType] = useState('sparse');
  const [isTraining, setIsTraining] = useState(false);
  const [episode, setEpisode] = useState(0);
  
  // 奖励函数参数
  const [goalReward, setGoalReward] = useState(100);
  const [stepPenalty, setStepPenalty] = useState(-1);
  const [collisionPenalty, setCollisionPenalty] = useState(-50);
  const [explorationBonus, setExplorationBonus] = useState(5);
  
  const [gridWorld] = useState({
    width: 5,
    height: 5,
    start: [0, 0],
    goal: [4, 4],
    obstacles: [[1, 1], [2, 2], [3, 1]]
  });

  const [agentState, setAgentState] = useState({
    position: [0, 0],
    visitedStates: new Set(),
    totalReward: 0,
    stepsInEpisode: 0
  });

  const [rewardHistory, setRewardHistory] = useState<any[]>([]);
  const [rewardBreakdown, setRewardBreakdown] = useState<any[]>([]);

  const rewardTypes = [
    { value: 'sparse', label: '稀疏奖励', description: '仅在达到目标时给予奖励' },
    { value: 'dense', label: '密集奖励', description: '每步都给予距离相关奖励' },
    { value: 'shaped', label: '塑形奖励', description: '结合距离和探索奖励' },
    { value: 'potential', label: '势函数奖励', description: '基于势函数的奖励设计' }
  ];

  // 计算曼哈顿距离
  const manhattanDistance = (pos1: number[], pos2: number[]): number => {
    return Math.abs(pos1[0] - pos2[0]) + Math.abs(pos1[1] - pos2[1]);
  };

  // 计算当前状态的奖励
  const calculateReward = (currentPos: number[], nextPos: number[], action: string): number => {
    let reward = 0;
    let breakdown: any = { step: 0, goal: 0, collision: 0, exploration: 0, distance: 0 };

    // 检查是否撞墙或越界
    if (nextPos[0] < 0 || nextPos[0] >= gridWorld.width || 
        nextPos[1] < 0 || nextPos[1] >= gridWorld.height) {
      reward += collisionPenalty;
      breakdown.collision = collisionPenalty;
      return { reward, breakdown };
    }

    // 检查是否撞到障碍物
    if (gridWorld.obstacles.some(obs => obs[0] === nextPos[0] && obs[1] === nextPos[1])) {
      reward += collisionPenalty;
      breakdown.collision = collisionPenalty;
      return { reward, breakdown };
    }

    // 基础步数惩罚
    reward += stepPenalty;
    breakdown.step = stepPenalty;

    // 到达目标奖励
    if (nextPos[0] === gridWorld.goal[0] && nextPos[1] === gridWorld.goal[1]) {
      reward += goalReward;
      breakdown.goal = goalReward;
    }

    // 不同类型的奖励函数
    switch (rewardType) {
      case 'dense':
        // 距离相关奖励
        const currentDistance = manhattanDistance(currentPos, gridWorld.goal);
        const nextDistance = manhattanDistance(nextPos, gridWorld.goal);
        const distanceReward = (currentDistance - nextDistance) * 2;
        reward += distanceReward;
        breakdown.distance = distanceReward;
        break;
        
      case 'shaped':
        // 探索奖励
        const stateKey = `${nextPos[0]},${nextPos[1]}`;
        if (!agentState.visitedStates.has(stateKey)) {
          reward += explorationBonus;
          breakdown.exploration = explorationBonus;
        }
        
        // 距离奖励
        const currentDist = manhattanDistance(currentPos, gridWorld.goal);
        const nextDist = manhattanDistance(nextPos, gridWorld.goal);
        const distRew = (currentDist - nextDist) * 1.5;
        reward += distRew;
        breakdown.distance = distRew;
        break;
        
      case 'potential':
        // 势函数奖励 (负距离作为势函数)
        const phi_current = -manhattanDistance(currentPos, gridWorld.goal);
        const phi_next = -manhattanDistance(nextPos, gridWorld.goal);
        const potentialReward = phi_next - phi_current;
        reward += potentialReward;
        breakdown.distance = potentialReward;
        break;
        
      case 'sparse':
      default:
        // 稀疏奖励已在上面处理
        break;
    }

    return { reward, breakdown };
  };

  // 模拟智能体行动
  const simulateStep = (): void => {
    const actions = ['up', 'down', 'left', 'right'];
    const action = actions[Math.floor(Math.random() * actions.length)];
    
    let nextPos = [...agentState.position];
    switch (action) {
      case 'up': nextPos[1] = Math.max(0, nextPos[1] - 1); break;
      case 'down': nextPos[1] = Math.min(gridWorld.height - 1, nextPos[1] + 1); break;
      case 'left': nextPos[0] = Math.max(0, nextPos[0] - 1); break;
      case 'right': nextPos[0] = Math.min(gridWorld.width - 1, nextPos[0] + 1); break;
    }

    const { reward, breakdown } = calculateReward(agentState.position, nextPos, action);
    
    setAgentState(prev => {
      const newVisited = new Set(prev.visitedStates);
      newVisited.add(`${nextPos[0]},${nextPos[1]}`);
      
      return {
        position: nextPos,
        visitedStates: newVisited,
        totalReward: prev.totalReward + reward,
        stepsInEpisode: prev.stepsInEpisode + 1
      };
    });

    // 更新历史记录
    setRewardHistory(prev => {
      const newData = [...prev, {
        step: agentState.stepsInEpisode + 1,
        reward,
        cumulativeReward: agentState.totalReward + reward,
        position: `(${nextPos[0]},${nextPos[1]})`,
        action
      }];
      return newData.slice(-50);
    });

    setRewardBreakdown(prev => [
      ...prev,
      {
        step: agentState.stepsInEpisode + 1,
        ...breakdown,
        total: reward
      }
    ].slice(-20));

    // 检查是否到达目标或步数过多
    if ((nextPos[0] === gridWorld.goal[0] && nextPos[1] === gridWorld.goal[1]) || 
        agentState.stepsInEpisode >= 50) {
      // 重置回合
      setTimeout(() => {
        setAgentState({
          position: [0, 0],
          visitedStates: new Set(),
          totalReward: 0,
          stepsInEpisode: 0
        });
        setEpisode(prev => prev + 1);
      }, 1000);
    }
  };

  const simulateTraining = () => {
    const interval = setInterval(() => {
      simulateStep();
    }, 200);

    return () => clearInterval(interval);
  };

  useEffect(() => {
    if (isTraining) {
      const cleanup = simulateTraining();
      return cleanup;
    }
  }, [isTraining]);

  const resetTraining = () => {
    setEpisode(0);
    setAgentState({
      position: [0, 0],
      visitedStates: new Set(),
      totalReward: 0,
      stepsInEpisode: 0
    });
    setRewardHistory([]);
    setRewardBreakdown([]);
  };

  const rewardColumns = [
    { title: '步数', dataIndex: 'step', key: 'step' },
    { title: '位置', dataIndex: 'position', key: 'position' },
    { title: '动作', dataIndex: 'action', key: 'action' },
    { title: '即时奖励', dataIndex: 'reward', key: 'reward',
      render: (val: number) => (
        <span style={{ color: val > 0 ? '#52c41a' : val < 0 ? '#ff4d4f' : '#000' }}>
          {val.toFixed(2)}
        </span>
      )
    },
    { title: '累积奖励', dataIndex: 'cumulativeReward', key: 'cumulativeReward',
      render: (val: number) => val.toFixed(2)
    }
  ];

  const breakdownColumns = [
    { title: '步数', dataIndex: 'step', key: 'step' },
    { title: '步数惩罚', dataIndex: 'step', key: 'stepReward',
      render: (val: number) => val.toFixed(2)
    },
    { title: '目标奖励', dataIndex: 'goal', key: 'goal',
      render: (val: number) => val.toFixed(2)
    },
    { title: '碰撞惩罚', dataIndex: 'collision', key: 'collision',
      render: (val: number) => val.toFixed(2)
    },
    { title: '探索奖励', dataIndex: 'exploration', key: 'exploration',
      render: (val: number) => val.toFixed(2)
    },
    { title: '距离奖励', dataIndex: 'distance', key: 'distance',
      render: (val: number) => val.toFixed(2)
    },
    { title: '总奖励', dataIndex: 'total', key: 'total',
      render: (val: number) => (
        <strong style={{ color: val > 0 ? '#52c41a' : '#ff4d4f' }}>
          {val.toFixed(2)}
        </strong>
      )
    }
  ];

  const rewardChartConfig = {
    data: rewardHistory,
    xField: 'step',
    yField: 'cumulativeReward',
    smooth: true,
    color: '#1890ff',
  };

  const renderGridWorld = () => {
    const cells = [];
    for (let y = 0; y < gridWorld.height; y++) {
      for (let x = 0; x < gridWorld.width; x++) {
        const isStart = x === gridWorld.start[0] && y === gridWorld.start[1];
        const isGoal = x === gridWorld.goal[0] && y === gridWorld.goal[1];
        const isAgent = x === agentState.position[0] && y === agentState.position[1];
        const isObstacle = gridWorld.obstacles.some(obs => obs[0] === x && obs[1] === y);
        const isVisited = agentState.visitedStates.has(`${x},${y}`);
        
        let cellStyle: any = {
          width: '40px',
          height: '40px',
          border: '1px solid #d9d9d9',
          display: 'inline-block',
          textAlign: 'center',
          lineHeight: '38px',
          fontSize: '12px'
        };

        if (isObstacle) cellStyle.backgroundColor = '#434343';
        else if (isAgent) cellStyle.backgroundColor = '#1890ff';
        else if (isGoal) cellStyle.backgroundColor = '#52c41a';
        else if (isStart) cellStyle.backgroundColor = '#faad14';
        else if (isVisited) cellStyle.backgroundColor = '#e6f7ff';
        else cellStyle.backgroundColor = '#fafafa';

        cells.push(
          <div key={`${x}-${y}`} style={cellStyle}>
            {isAgent ? '🤖' : isGoal ? '🎯' : isStart ? '🏠' : isObstacle ? '🧱' : ''}
          </div>
        );
      }
      cells.push(<br key={`br-${y}`} />);
    }
    return cells;
  };

  return (
    <div style={{ padding: '24px', background: '#f0f2f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <ThunderboltOutlined style={{ marginRight: '12px', color: '#1890ff' }} />
          基础奖励函数设计系统
        </Title>
        <Text type="secondary" style={{ fontSize: '16px' }}>
          探索不同奖励函数设计对智能体学习行为的影响
        </Text>
      </div>

      <Row gutter={[24, 24]}>
        <Col span={24}>
          <Card title="奖励函数配置" extra={
            <Space>
              <Button 
                type={isTraining ? "default" : "primary"} 
                icon={isTraining ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
                onClick={() => setIsTraining(!isTraining)}
                size="large"
              >
                {isTraining ? '暂停训练' : '开始训练'}
              </Button>
              <Button icon={<ReloadOutlined />} onClick={resetTraining}>
                重置环境
              </Button>
            </Space>
          }>
            <Row gutter={[16, 16]}>
              <Col span={6}>
                <div>
                  <Text strong>奖励函数类型</Text>
                  <Select 
                    value={rewardType} 
                    onChange={setRewardType}
                    style={{ width: '100%', marginTop: '8px' }}
                    size="large"
                  >
                    {rewardTypes.map(type => (
                      <Option key={type.value} value={type.value}>{type.label}</Option>
                    ))}
                  </Select>
                  <Text type="secondary" style={{ marginTop: '4px', display: 'block' }}>
                    {rewardTypes.find(type => type.value === rewardType)?.description}
                  </Text>
                </div>
              </Col>
              <Col span={6}>
                <div>
                  <Text strong>目标奖励: {goalReward}</Text>
                  <Slider 
                    min={10} 
                    max={200} 
                    value={goalReward} 
                    onChange={setGoalReward}
                    style={{ marginTop: '8px' }}
                  />
                </div>
              </Col>
              <Col span={6}>
                <div>
                  <Text strong>步数惩罚: {stepPenalty}</Text>
                  <Slider 
                    min={-10} 
                    max={0} 
                    step={0.1}
                    value={stepPenalty} 
                    onChange={setStepPenalty}
                    style={{ marginTop: '8px' }}
                  />
                </div>
              </Col>
              <Col span={6}>
                <div>
                  <Text strong>碰撞惩罚: {collisionPenalty}</Text>
                  <Slider 
                    min={-100} 
                    max={-10} 
                    value={collisionPenalty} 
                    onChange={setCollisionPenalty}
                    style={{ marginTop: '8px' }}
                  />
                </div>
              </Col>
            </Row>
          </Card>
        </Col>

        <Col span={12}>
          <Card title="GridWorld环境">
            <div style={{ textAlign: 'center', marginBottom: '16px' }}>
              {renderGridWorld()}
            </div>
            <Row gutter={[16, 16]}>
              <Col span={8}>
                <Statistic 
                  title="当前回合" 
                  value={episode} 
                  prefix={<CalculatorOutlined />}
                />
              </Col>
              <Col span={8}>
                <Statistic 
                  title="当前步数" 
                  value={agentState.stepsInEpisode} 
                />
              </Col>
              <Col span={8}>
                <Statistic 
                  title="累积奖励" 
                  value={agentState.totalReward} 
                  precision={2}
                  valueStyle={{ color: agentState.totalReward >= 0 ? '#3f8600' : '#cf1322' }}
                />
              </Col>
            </Row>
          </Card>
        </Col>

        <Col span={12}>
          <Card title="奖励趋势分析">
            <div style={{ height: '200px', marginBottom: '16px' }}>
              <Line {...rewardChartConfig} />
            </div>
            <Space direction="vertical" style={{ width: '100%' }}>
              <div>
                <Text strong>当前位置: </Text>
                <Tag>({agentState.position[0]}, {agentState.position[1]})</Tag>
              </div>
              <div>
                <Text strong>探索状态数: </Text>
                <Tag color="blue">{agentState.visitedStates.size}</Tag>
              </div>
              <div>
                <Text strong>到目标距离: </Text>
                <Tag color="orange">{manhattanDistance(agentState.position, gridWorld.goal)}</Tag>
              </div>
            </Space>
          </Card>
        </Col>

        <Col span={24}>
          <Tabs defaultActiveKey="history">
            <TabPane tab="奖励历史" key="history">
              <Table 
                columns={rewardColumns} 
                dataSource={rewardHistory} 
                pagination={{ pageSize: 10 }}
                size="small"
              />
            </TabPane>
            
            <TabPane tab="奖励分解" key="breakdown">
              <Table 
                columns={breakdownColumns} 
                dataSource={rewardBreakdown} 
                pagination={{ pageSize: 10 }}
                size="small"
              />
            </TabPane>
            
            <TabPane tab="函数对比" key="comparison">
              <Row gutter={[16, 16]}>
                <Col span={12}>
                  <Card title="奖励函数特性对比" size="small">
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Alert 
                        message="稀疏奖励" 
                        description="优点: 简单直观；缺点: 学习困难，收敛慢"
                        variant="default" 
                      />
                      <Alert 
                        message="密集奖励" 
                        description="优点: 学习快速；缺点: 可能过度引导，局部最优"
                        type="success" 
                      />
                      <Alert 
                        message="塑形奖励" 
                        description="优点: 平衡探索利用；缺点: 设计复杂"
                        variant="warning" 
                      />
                      <Alert 
                        message="势函数奖励" 
                        description="优点: 理论保证；缺点: 需要领域知识"
                        variant="destructive" 
                      />
                    </Space>
                  </Card>
                </Col>
                <Col span={12}>
                  <Card title="奖励函数公式" size="small">
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <div>
                        <Text strong>稀疏奖励:</Text>
                        <br />
                        <Text code>R(s,a) = 100 if goal else -1</Text>
                      </div>
                      <div>
                        <Text strong>密集奖励:</Text>
                        <br />
                        <Text code>R(s,a) = -(dist_new - dist_old) - 1</Text>
                      </div>
                      <div>
                        <Text strong>势函数奖励:</Text>
                        <br />
                        <Text code>R(s,a) = φ(s') - φ(s) - 1</Text>
                      </div>
                    </Space>
                  </Card>
                </Col>
              </Row>
            </TabPane>
          </Tabs>
        </Col>
      </Row>
    </div>
  );
};

export default BasicRewardsPage;