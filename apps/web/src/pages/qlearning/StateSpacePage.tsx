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
  Tooltip
} from 'antd';
import { Line, Scatter, Heatmap } from '@ant-design/charts';
import {
  DatabaseOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  ReloadOutlined,
  NodeIndexOutlined,
  SettingOutlined,
  EyeOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;

const StateSpacePage: React.FC = () => {
  const [stateType, setStateType] = useState('discrete');
  const [dimension, setDimension] = useState(2);
  const [resolution, setResolution] = useState(10);
  const [isExploring, setIsExploring] = useState(false);
  const [step, setStep] = useState(0);
  
  const [currentState, setCurrentState] = useState([0, 0]);
  const [stateHistory, setStateHistory] = useState<any[]>([]);
  const [visitationMap, setVisitationMap] = useState<Map<string, number>>(new Map());
  const [stateValues, setStateValues] = useState<Map<string, number>>(new Map());

  const stateTypes = [
    { value: 'discrete', label: '离散状态空间', description: '有限的状态集合' },
    { value: 'continuous', label: '连续状态空间', description: '无限的状态空间' },
    { value: 'hybrid', label: '混合状态空间', description: '离散+连续混合' },
    { value: 'hierarchical', label: '分层状态空间', description: '多层次抽象状态' }
  ];

  // 生成状态空间网格数据
  const generateStateGrid = () => {
    const gridData = [];
    for (let x = 0; x < resolution; x++) {
      for (let y = 0; y < resolution; y++) {
        const stateKey = `${x},${y}`;
        const visitCount = visitationMap.get(stateKey) || 0;
        const value = stateValues.get(stateKey) || Math.random() * 10 - 5;
        
        gridData.push({
          x,
          y,
          visits: visitCount,
          value,
          stateKey,
          type: getStateType(x, y)
        });
      }
    }
    return gridData;
  };

  // 确定状态类型（用于分层状态空间）
  const getStateType = (x: number, y: number): string => {
    if (stateType === 'hierarchical') {
      const centerX = resolution / 2;
      const centerY = resolution / 2;
      const distance = Math.sqrt((x - centerX) ** 2 + (y - centerY) ** 2);
      
      if (distance < resolution * 0.15) return 'goal';
      else if (distance < resolution * 0.3) return 'near_goal';
      else if (x < resolution * 0.2 || y < resolution * 0.2) return 'boundary';
      else return 'normal';
    }
    return 'normal';
  };

  // 状态转移函数
  const transitionState = (currentState: number[], action: number): number[] => {
    let newState = [...currentState];
    
    if (stateType === 'discrete') {
      // 离散状态转移
      const moves = [[-1, 0], [1, 0], [0, -1], [0, 1]]; // 上下左右
      const move = moves[action % 4];
      newState[0] = Math.max(0, Math.min(resolution - 1, currentState[0] + move[0]));
      newState[1] = Math.max(0, Math.min(resolution - 1, currentState[1] + move[1]));
    } else if (stateType === 'continuous') {
      // 连续状态转移（添加噪声）
      const noise = 0.5;
      const moves = [[-1, 0], [1, 0], [0, -1], [0, 1]];
      const move = moves[action % 4];
      newState[0] = Math.max(0, Math.min(resolution - 1, 
        currentState[0] + move[0] + (Math.random() - 0.5) * noise));
      newState[1] = Math.max(0, Math.min(resolution - 1, 
        currentState[1] + move[1] + (Math.random() - 0.5) * noise));
    } else {
      // 其他类型的简化转移
      const moves = [[-1, 0], [1, 0], [0, -1], [0, 1]];
      const move = moves[action % 4];
      newState[0] = Math.max(0, Math.min(resolution - 1, currentState[0] + move[0]));
      newState[1] = Math.max(0, Math.min(resolution - 1, currentState[1] + move[1]));
    }
    
    return newState;
  };

  // 模拟状态探索
  const simulateExploration = () => {
    const interval = setInterval(() => {
      setStep(prev => {
        const newStep = prev + 1;
        
        // 随机选择动作
        const action = Math.floor(Math.random() * 4);
        const newState = transitionState(currentState, action);
        
        // 更新当前状态
        setCurrentState(newState);
        
        // 更新访问计数
        const stateKey = `${Math.floor(newState[0])},${Math.floor(newState[1])}`;
        setVisitationMap(prev => {
          const newMap = new Map(prev);
          newMap.set(stateKey, (newMap.get(stateKey) || 0) + 1);
          return newMap;
        });
        
        // 更新状态价值（简化的价值函数）
        setStateValues(prev => {
          const newMap = new Map(prev);
          const centerX = resolution / 2;
          const centerY = resolution / 2;
          const distance = Math.sqrt(
            (newState[0] - centerX) ** 2 + (newState[1] - centerY) ** 2
          );
          const value = 10 - distance * 0.5 + (Math.random() - 0.5) * 2;
          newMap.set(stateKey, value);
          return newMap;
        });
        
        // 记录历史
        setStateHistory(prev => {
          const newData = [...prev, {
            step: newStep,
            state: [...newState],
            stateKey,
            x: newState[0],
            y: newState[1],
            action,
            type: getStateType(Math.floor(newState[0]), Math.floor(newState[1]))
          }];
          return newData.slice(-100);
        });
        
        if (newStep >= 1000) {
          setIsExploring(false);
          return newStep;
        }
        return newStep;
      });
    }, 100);

    return () => clearInterval(interval);
  };

  useEffect(() => {
    if (isExploring) {
      const cleanup = simulateExploration();
      return cleanup;
    }
  }, [isExploring]);

  const resetExploration = () => {
    setStep(0);
    setCurrentState([Math.floor(resolution / 2), Math.floor(resolution / 2)]);
    setStateHistory([]);
    setVisitationMap(new Map());
    setStateValues(new Map());
  };

  const stateGridData = generateStateGrid();
  
  const visitationConfig = {
    data: stateGridData,
    xField: 'x',
    yField: 'y',
    colorField: 'visits',
    sizeField: 'visits',
    size: [2, 15],
    color: ['#ffffff', '#1890ff', '#0050b3'],
    tooltip: {
      fields: ['visits', 'value', 'type'],
      formatter: (datum: any) => ({
        name: '状态信息',
        value: `访问: ${datum.visits}, 价值: ${datum.value.toFixed(2)}, 类型: ${datum.type}`
      })
    }
  };

  const valueConfig = {
    data: stateGridData,
    xField: 'x',
    yField: 'y',
    colorField: 'value',
    color: ['#ff4d4f', '#faad14', '#52c41a'],
  };

  const trajectoryConfig = {
    data: stateHistory,
    xField: 'x',
    yField: 'y',
    seriesField: 'type',
    size: 2,
    color: ['#1890ff', '#52c41a', '#faad14', '#ff4d4f'],
  };

  const stateColumns = [
    { title: '步数', dataIndex: 'step', key: 'step' },
    { title: 'X坐标', dataIndex: 'x', key: 'x', render: (val: number) => val.toFixed(2) },
    { title: 'Y坐标', dataIndex: 'y', key: 'y', render: (val: number) => val.toFixed(2) },
    { title: '动作', dataIndex: 'action', key: 'action',
      render: (action: number) => {
        const actionNames = ['上', '下', '左', '右'];
        return actionNames[action] || action;
      }
    },
    { title: '状态类型', dataIndex: 'type', key: 'type',
      render: (type: string) => {
        const colors = {
          goal: 'green',
          near_goal: 'blue',
          boundary: 'orange',
          normal: 'default'
        };
        return <Tag color={colors[type as keyof typeof colors]}>{type}</Tag>;
      }
    }
  ];

  const stateStats = {
    totalStates: stateGridData.length,
    visitedStates: visitationMap.size,
    averageVisits: visitationMap.size > 0 ? 
      Array.from(visitationMap.values()).reduce((a, b) => a + b, 0) / visitationMap.size : 0,
    maxVisits: visitationMap.size > 0 ? Math.max(...Array.from(visitationMap.values())) : 0,
    stateSpaceCoverage: (visitationMap.size / stateGridData.length) * 100
  };

  return (
    <div style={{ padding: '24px', background: '#f0f2f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <DatabaseOutlined style={{ marginRight: '12px', color: '#1890ff' }} />
          状态空间设计与分析系统
        </Title>
        <Text type="secondary" style={{ fontSize: '16px' }}>
          深入理解不同类型状态空间的特性和探索策略的影响
        </Text>
      </div>

      <Row gutter={[24, 24]}>
        <Col span={24}>
          <Card title="状态空间配置" extra={
            <Space>
              <Button 
                type={isExploring ? "default" : "primary"} 
                icon={isExploring ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
                onClick={() => setIsExploring(!isExploring)}
                size="large"
              >
                {isExploring ? '暂停探索' : '开始探索'}
              </Button>
              <Button icon={<ReloadOutlined />} onClick={resetExploration}>
                重置状态空间
              </Button>
            </Space>
          }>
            <Row gutter={[16, 16]}>
              <Col span={8}>
                <div>
                  <Text strong>状态空间类型</Text>
                  <Select 
                    value={stateType} 
                    onChange={setStateType}
                    style={{ width: '100%', marginTop: '8px' }}
                    size="large"
                  >
                    {stateTypes.map(type => (
                      <Option key={type.value} value={type.value}>{type.label}</Option>
                    ))}
                  </Select>
                  <Text type="secondary" style={{ marginTop: '4px', display: 'block' }}>
                    {stateTypes.find(type => type.value === stateType)?.description}
                  </Text>
                </div>
              </Col>
              <Col span={8}>
                <div>
                  <Text strong>空间分辨率: {resolution}×{resolution}</Text>
                  <Slider 
                    min={5} 
                    max={20} 
                    value={resolution} 
                    onChange={setResolution}
                    style={{ marginTop: '8px' }}
                    disabled={isExploring}
                  />
                </div>
              </Col>
              <Col span={8}>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <div>
                    <Text strong>探索步数: </Text>
                    <Tag color="blue" style={{ fontSize: '14px' }}>{step}</Tag>
                  </div>
                  <div>
                    <Text strong>当前状态: </Text>
                    <Tag color="green">({currentState[0].toFixed(2)}, {currentState[1].toFixed(2)})</Tag>
                  </div>
                </Space>
              </Col>
            </Row>
          </Card>
        </Col>

        <Col span={16}>
          <Card title="状态空间可视化">
            <Tabs defaultActiveKey="visitation">
              <TabPane tab="访问热图" key="visitation" icon={<EyeOutlined />}>
                <div style={{ height: '300px' }}>
                  <Scatter {...visitationConfig} />
                </div>
              </TabPane>
              <TabPane tab="价值函数" key="value" icon={<NodeIndexOutlined />}>
                <div style={{ height: '300px' }}>
                  <Heatmap {...valueConfig} />
                </div>
              </TabPane>
              <TabPane tab="轨迹分析" key="trajectory" icon={<SettingOutlined />}>
                <div style={{ height: '300px' }}>
                  <Scatter {...trajectoryConfig} />
                </div>
              </TabPane>
            </Tabs>
          </Card>
        </Col>

        <Col span={8}>
          <Card title="状态空间统计">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Statistic 
                title="状态空间大小" 
                value={stateStats.totalStates} 
                prefix={<DatabaseOutlined />}
              />
              <Statistic 
                title="已访问状态" 
                value={stateStats.visitedStates} 
                suffix={`/ ${stateStats.totalStates}`}
              />
              <Statistic 
                title="空间覆盖率" 
                value={stateStats.stateSpaceCoverage} 
                precision={1}
                suffix="%"
                valueStyle={{ color: stateStats.stateSpaceCoverage > 50 ? '#3f8600' : '#cf1322' }}
              />
              <Statistic 
                title="平均访问次数" 
                value={stateStats.averageVisits} 
                precision={2}
              />
              <div style={{ marginTop: '16px' }}>
                <Text strong>探索进度</Text>
                <Progress 
                  percent={Math.min((step / 1000) * 100, 100)}
                  status={isExploring ? 'active' : 'normal'}
                  strokeColor="#52c41a"
                />
              </div>
            </Space>
          </Card>
        </Col>

        <Col span={24}>
          <Tabs defaultActiveKey="analysis">
            <TabPane tab="状态分析" key="analysis">
              <Row gutter={[16, 16]}>
                <Col span={12}>
                  <Card title="状态类型分布" size="small">
                    {stateType === 'hierarchical' && (
                      <Space direction="vertical" style={{ width: '100%' }}>
                        <Alert message="分层状态空间结构" description="根据与目标的距离划分不同层次的状态" variant="default" />
                        <div>
                          <Tag color="green">目标区域</Tag> - 最高价值区域
                        </div>
                        <div>
                          <Tag color="blue">近目标区域</Tag> - 中等价值区域  
                        </div>
                        <div>
                          <Tag color="orange">边界区域</Tag> - 低价值但重要的边界
                        </div>
                        <div>
                          <Tag color="default">普通区域</Tag> - 基础探索区域
                        </div>
                      </Space>
                    )}
                    {stateType === 'discrete' && (
                      <Alert message="离散状态空间" description="状态完全独立，转移确定性强" type="success" />
                    )}
                    {stateType === 'continuous' && (
                      <Alert message="连续状态空间" description="状态间连续变化，需要函数逼近" variant="warning" />
                    )}
                  </Card>
                </Col>
                <Col span={12}>
                  <Card title="维度诅咒分析" size="small">
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <div>
                        <Text strong>当前维度: </Text> {dimension}D
                      </div>
                      <div>
                        <Text strong>状态数量: </Text> {Math.pow(resolution, dimension).toLocaleString()}
                      </div>
                      <div>
                        <Text strong>存储复杂度: </Text> O(n^{dimension})
                      </div>
                      <Alert 
                        message="维度诅咒警告" 
                        description={`${dimension}维空间需要 ${Math.pow(resolution, dimension)} 个状态，高维度将导致指数级增长`}
                        type={dimension > 3 ? "error" : "info"} 
                      />
                    </Space>
                  </Card>
                </Col>
              </Row>
            </TabPane>
            
            <TabPane tab="探索历史" key="history">
              <Table 
                columns={stateColumns} 
                dataSource={stateHistory.slice(-20)} 
                pagination={false}
                size="small"
                scroll={{ y: 300 }}
              />
            </TabPane>

            <TabPane tab="设计原则" key="principles">
              <Row gutter={[16, 16]}>
                <Col span={12}>
                  <Card title="状态空间设计原则" size="small">
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Alert message="马尔可夫性质" description="当前状态包含做决策所需的所有信息" type="success" />
                      <Alert message="完整性" description="所有相关的系统状态都能被表示" variant="default" />
                      <Alert message="最小性" description="避免冗余状态，保持表示的紧凑性" variant="warning" />
                      <Alert message="可观测性" description="智能体能够观测和区分不同状态" variant="destructive" />
                    </Space>
                  </Card>
                </Col>
                <Col span={12}>
                  <Card title="常见问题与解决方案" size="small">
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <div>
                        <Text strong>维度诅咒:</Text> 使用特征选择、降维、函数逼近
                      </div>
                      <div>
                        <Text strong>状态别名:</Text> 增加历史信息、扩展状态表示
                      </div>
                      <div>
                        <Text strong>稀疏奖励:</Text> 状态塑形、分层强化学习
                      </div>
                      <div>
                        <Text strong>探索不充分:</Text> 好奇心驱动、内在动机
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

export default StateSpacePage;