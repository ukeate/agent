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
  Radio,
  Collapse
} from 'antd';
import { Line, Column, Pie } from '@ant-design/charts';
import {
  ApiOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  ReloadOutlined,
  ControlOutlined,
  SettingOutlined,
  ExperimentOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;
const { Panel } = Collapse;

const ActionSpacePage: React.FC = () => {
  const [actionSpaceType, setActionSpaceType] = useState('discrete');
  const [actionCount, setActionCount] = useState(4);
  const [isTraining, setIsTraining] = useState(false);
  const [step, setStep] = useState(0);
  
  const [actionHistory, setActionHistory] = useState<any[]>([]);
  const [actionDistribution, setActionDistribution] = useState<Map<string, number>>(new Map());
  const [actionEffectiveness, setActionEffectiveness] = useState<Map<string, number>>(new Map());
  const [environmentState, setEnvironmentState] = useState({ x: 5, y: 5, goal: { x: 9, y: 9 } });

  const actionSpaceTypes = [
    { value: 'discrete', label: '离散动作空间', description: '有限的动作集合' },
    { value: 'continuous', label: '连续动作空间', description: '无限的动作参数' },
    { value: 'multi_discrete', label: '多重离散动作', description: '多个独立的离散动作' },
    { value: 'hybrid', label: '混合动作空间', description: '离散+连续混合' }
  ];

  // 定义不同类型的动作空间
  const getActionSpace = () => {
    switch (actionSpaceType) {
      case 'discrete':
        return Array.from({ length: actionCount }, (_, i) => ({
          id: i,
          name: ['上移', '下移', '左移', '右移', '停止', '跳跃', '攻击', '防御'][i] || `动作${i}`,
          type: 'discrete',
          params: null
        }));
      
      case 'continuous':
        return [
          { id: 0, name: '移动方向', type: 'continuous', params: { min: 0, max: 360, current: 0 } },
          { id: 1, name: '移动速度', type: 'continuous', params: { min: 0, max: 10, current: 5 } },
          { id: 2, name: '转向角度', type: 'continuous', params: { min: -180, max: 180, current: 0 } }
        ];
      
      case 'multi_discrete':
        return [
          { id: 0, name: '移动动作', type: 'discrete', options: ['不动', '前进', '后退', '左转', '右转'] },
          { id: 1, name: '操作动作', type: 'discrete', options: ['无操作', '抓取', '放置', '推动'] },
          { id: 2, name: '交互动作', type: 'discrete', options: ['无交互', '开启', '关闭', '切换'] }
        ];
      
      case 'hybrid':
        return [
          { id: 0, name: '离散指令', type: 'discrete', options: ['移动', '操作', '等待'] },
          { id: 1, name: '连续参数1', type: 'continuous', params: { min: -1, max: 1, current: 0 } },
          { id: 2, name: '连续参数2', type: 'continuous', params: { min: -1, max: 1, current: 0 } }
        ];
      
      default:
        return [];
    }
  };

  const actionSpace = getActionSpace();

  // 执行动作并计算效果
  const executeAction = (actionId: number) => {
    let newState = { ...environmentState };
    let reward = 0;
    let actionName = '';

    if (actionSpaceType === 'discrete') {
      const actions = ['上移', '下移', '左移', '右移', '停止', '跳跃', '攻击', '防御'];
      actionName = actions[actionId] || `动作${actionId}`;
      
      switch (actionId) {
        case 0: // 上移
          newState.y = Math.max(0, newState.y - 1);
          break;
        case 1: // 下移
          newState.y = Math.min(9, newState.y + 1);
          break;
        case 2: // 左移
          newState.x = Math.max(0, newState.x - 1);
          break;
        case 3: // 右移
          newState.x = Math.min(9, newState.x + 1);
          break;
        case 4: // 停止
          // 位置不变
          break;
        default:
          // 其他动作的特殊效果
          reward += Math.random() * 2 - 1; // 随机奖励
      }
      
      // 计算距离奖励
      const oldDistance = Math.abs(environmentState.x - environmentState.goal.x) + 
                          Math.abs(environmentState.y - environmentState.goal.y);
      const newDistance = Math.abs(newState.x - newState.goal.x) + 
                          Math.abs(newState.y - newState.goal.y);
      reward += (oldDistance - newDistance) * 2;
      
      // 到达目标的额外奖励
      if (newState.x === newState.goal.x && newState.y === newState.goal.y) {
        reward += 100;
        // 重置目标位置
        newState.goal = {
          x: Math.floor(Math.random() * 10),
          y: Math.floor(Math.random() * 10)
        };
      }
      
    } else {
      actionName = `连续动作${actionId}`;
      // 连续动作的简化处理
      const angle = Math.random() * 2 * Math.PI;
      const speed = Math.random() * 2;
      newState.x = Math.max(0, Math.min(9, newState.x + Math.cos(angle) * speed));
      newState.y = Math.max(0, Math.min(9, newState.y + Math.sin(angle) * speed));
      reward = Math.random() * 4 - 2;
    }

    setEnvironmentState(newState);
    
    // 更新动作分布
    setActionDistribution(prev => {
      const newMap = new Map(prev);
      newMap.set(actionName, (newMap.get(actionName) || 0) + 1);
      return newMap;
    });
    
    // 更新动作效果
    setActionEffectiveness(prev => {
      const newMap = new Map(prev);
      const currentEffectiveness = newMap.get(actionName) || 0;
      const newEffectiveness = (currentEffectiveness + reward) / ((prev.get(actionName) || 0) + 1);
      newMap.set(actionName, newEffectiveness);
      return newMap;
    });
    
    return { actionName, reward, newState };
  };

  // 模拟训练过程
  const simulateTraining = () => {
    const interval = setInterval(() => {
      setStep(prev => {
        const newStep = prev + 1;
        
        // 选择动作（这里使用随机策略）
        const actionId = Math.floor(Math.random() * Math.max(actionCount, actionSpace.length));
        const result = executeAction(actionId);
        
        // 记录历史
        setActionHistory(prev => {
          const newData = [...prev, {
            step: newStep,
            actionId,
            actionName: result.actionName,
            reward: result.reward,
            position: `(${result.newState.x.toFixed(1)}, ${result.newState.y.toFixed(1)})`,
            effectiveness: actionEffectiveness.get(result.actionName) || 0
          }];
          return newData.slice(-50);
        });
        
        if (newStep >= 500) {
          setIsTraining(false);
          return newStep;
        }
        return newStep;
      });
    }, 100);

    return () => clearInterval(interval);
  };

  useEffect(() => {
    if (isTraining) {
      const cleanup = simulateTraining();
      return cleanup;
    }
  }, [isTraining]);

  const resetTraining = () => {
    setStep(0);
    setActionHistory([]);
    setActionDistribution(new Map());
    setActionEffectiveness(new Map());
    setEnvironmentState({ x: 5, y: 5, goal: { x: 9, y: 9 } });
  };

  const actionColumns = [
    { title: '步数', dataIndex: 'step', key: 'step' },
    { title: '动作ID', dataIndex: 'actionId', key: 'actionId' },
    { title: '动作名称', dataIndex: 'actionName', key: 'actionName' },
    { title: '即时奖励', dataIndex: 'reward', key: 'reward',
      render: (val: number) => (
        <span style={{ color: val > 0 ? '#52c41a' : '#ff4d4f' }}>
          {val.toFixed(2)}
        </span>
      )
    },
    { title: '智能体位置', dataIndex: 'position', key: 'position' },
    { title: '动作效果', dataIndex: 'effectiveness', key: 'effectiveness',
      render: (val: number) => val.toFixed(3)
    }
  ];

  const distributionData = Array.from(actionDistribution.entries()).map(([action, count]) => ({
    action,
    count,
    percentage: ((count / step) * 100).toFixed(1)
  }));

  const effectivenessData = Array.from(actionEffectiveness.entries()).map(([action, effectiveness]) => ({
    action,
    effectiveness: effectiveness.toFixed(3),
    value: effectiveness
  }));

  const distributionConfig = {
    data: distributionData,
    angleField: 'count',
    colorField: 'action',
    radius: 0.8,
    label: {
      type: 'outer',
      content: '{name} {percentage}',
    },
    interactions: [{ type: 'element-active' }],
  };

  const effectivenessConfig = {
    data: effectivenessData,
    xField: 'action',
    yField: 'value',
    columnStyle: {
      fill: ({ value }: any) => (value > 0 ? '#52c41a' : '#ff4d4f'),
    },
    label: {
      position: 'middle' as const,
      style: {
        fill: '#FFFFFF',
        opacity: 0.6,
      },
    },
  };

  const renderEnvironment = () => {
    const cells = [];
    for (let y = 0; y < 10; y++) {
      for (let x = 0; x < 10; x++) {
        const isAgent = x === Math.floor(environmentState.x) && y === Math.floor(environmentState.y);
        const isGoal = x === environmentState.goal.x && y === environmentState.goal.y;
        
        let cellStyle: any = {
          width: '20px',
          height: '20px',
          border: '1px solid #d9d9d9',
          display: 'inline-block',
          textAlign: 'center',
          lineHeight: '18px',
          fontSize: '10px'
        };

        if (isAgent) cellStyle.backgroundColor = '#1890ff';
        else if (isGoal) cellStyle.backgroundColor = '#52c41a';
        else cellStyle.backgroundColor = '#fafafa';

        cells.push(
          <div key={`${x}-${y}`} style={cellStyle}>
            {isAgent ? '🤖' : isGoal ? '🎯' : ''}
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
          <ApiOutlined style={{ marginRight: '12px', color: '#1890ff' }} />
          动作空间定义与分析系统
        </Title>
        <Text type="secondary" style={{ fontSize: '16px' }}>
          深入探索不同类型动作空间的设计原理和对智能体行为的影响
        </Text>
      </div>

      <Row gutter={[24, 24]}>
        <Col span={24}>
          <Card title="动作空间配置" extra={
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
              <Col span={12}>
                <div>
                  <Text strong>动作空间类型</Text>
                  <Select 
                    value={actionSpaceType} 
                    onChange={setActionSpaceType}
                    style={{ width: '100%', marginTop: '8px' }}
                    size="large"
                  >
                    {actionSpaceTypes.map(type => (
                      <Option key={type.value} value={type.value}>{type.label}</Option>
                    ))}
                  </Select>
                  <Text type="secondary" style={{ marginTop: '4px', display: 'block' }}>
                    {actionSpaceTypes.find(type => type.value === actionSpaceType)?.description}
                  </Text>
                </div>
              </Col>
              <Col span={12}>
                {actionSpaceType === 'discrete' && (
                  <div>
                    <Text strong>动作数量: {actionCount}</Text>
                    <Slider 
                      min={2} 
                      max={8} 
                      value={actionCount} 
                      onChange={setActionCount}
                      style={{ marginTop: '8px' }}
                      disabled={isTraining}
                    />
                  </div>
                )}
                <Space style={{ marginTop: '8px' }}>
                  <Text strong>训练步数: </Text>
                  <Tag color="blue" style={{ fontSize: '14px' }}>{step}</Tag>
                  <Text strong>动作空间大小: </Text>
                  <Tag color="green">{actionSpace.length}</Tag>
                </Space>
              </Col>
            </Row>
          </Card>
        </Col>

        <Col span={12}>
          <Card title="环境状态">
            <div style={{ textAlign: 'center', marginBottom: '16px' }}>
              {renderEnvironment()}
            </div>
            <Row gutter={[16, 16]}>
              <Col span={12}>
                <Statistic 
                  title="智能体位置" 
                  value={`(${environmentState.x.toFixed(1)}, ${environmentState.y.toFixed(1)})`}
                />
              </Col>
              <Col span={12}>
                <Statistic 
                  title="目标位置" 
                  value={`(${environmentState.goal.x}, ${environmentState.goal.y})`}
                />
              </Col>
            </Row>
          </Card>
        </Col>

        <Col span={12}>
          <Card title="动作空间结构">
            <Collapse size="small">
              {actionSpace.map((action, index) => (
                <Panel header={`${action.name} (ID: ${action.id})`} key={index}>
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <div><Text strong>类型:</Text> {action.type}</div>
                    {action.params && (
                      <div>
                        <Text strong>参数范围:</Text> 
                        [{action.params.min}, {action.params.max}]
                      </div>
                    )}
                    {action.options && (
                      <div>
                        <Text strong>可选值:</Text>
                        <Space wrap>
                          {action.options.map((opt, i) => (
                            <Tag key={i}>{opt}</Tag>
                          ))}
                        </Space>
                      </div>
                    )}
                    <div>
                      <Text strong>执行次数:</Text> {actionDistribution.get(action.name) || 0}
                    </div>
                    <div>
                      <Text strong>平均效果:</Text> {(actionEffectiveness.get(action.name) || 0).toFixed(3)}
                    </div>
                  </Space>
                </Panel>
              ))}
            </Collapse>
          </Card>
        </Col>

        <Col span={24}>
          <Tabs defaultActiveKey="distribution">
            <TabPane tab="动作分布" key="distribution" icon={<ControlOutlined />}>
              <Row gutter={[24, 24]}>
                <Col span={12}>
                  <Card title="动作选择分布" size="small">
                    <div style={{ height: '300px' }}>
                      <Pie {...distributionConfig} />
                    </div>
                  </Card>
                </Col>
                <Col span={12}>
                  <Card title="动作效果分析" size="small">
                    <div style={{ height: '300px' }}>
                      <Column {...effectivenessConfig} />
                    </div>
                  </Card>
                </Col>
              </Row>
            </TabPane>
            
            <TabPane tab="动作历史" key="history" icon={<ExperimentOutlined />}>
              <Table 
                columns={actionColumns} 
                dataSource={actionHistory.slice(-20)} 
                pagination={false}
                size="small"
                scroll={{ y: 300 }}
              />
            </TabPane>

            <TabPane tab="设计原则" key="principles" icon={<SettingOutlined />}>
              <Row gutter={[16, 16]}>
                <Col span={12}>
                  <Card title="动作空间设计考虑" size="small">
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Alert 
                        message="完整性" 
                        description="动作空间应能表达所有必要的策略选择"
                        type="success" 
                      />
                      <Alert 
                        message="最小性" 
                        description="避免冗余动作，保持动作空间紧凑"
                        variant="default" 
                      />
                      <Alert 
                        message="可执行性" 
                        description="所有动作都必须在环境中可以实际执行"
                        variant="warning" 
                      />
                      <Alert 
                        message="有界性" 
                        description="连续动作空间需要明确的边界约束"
                        variant="destructive" 
                      />
                    </Space>
                  </Card>
                </Col>
                <Col span={12}>
                  <Card title="动作空间类型对比" size="small">
                    <Collapse size="small">
                      <Panel header="离散动作空间" key="1">
                        <Text>优点: 简单直观，容易优化</Text><br />
                        <Text>缺点: 表达能力有限</Text><br />
                        <Text>适用: 游戏、机器人关节控制</Text>
                      </Panel>
                      <Panel header="连续动作空间" key="2">
                        <Text>优点: 表达能力强，动作精细</Text><br />
                        <Text>缺点: 优化困难，需要特殊算法</Text><br />
                        <Text>适用: 机器人控制、自动驾驶</Text>
                      </Panel>
                      <Panel header="混合动作空间" key="3">
                        <Text>优点: 兼具离散和连续优势</Text><br />
                        <Text>缺点: 复杂度高，算法设计困难</Text><br />
                        <Text>适用: 复杂决策任务</Text>
                      </Panel>
                    </Collapse>
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

export default ActionSpacePage;