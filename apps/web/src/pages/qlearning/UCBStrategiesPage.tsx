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
  InputNumber
} from 'antd';
import { Line, Column } from '@ant-design/charts';
import {
  LineChartOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  ReloadOutlined,
  CalculatorOutlined,
  TrophyOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;

const UCBStrategiesPage: React.FC = () => {
  const [strategy, setStrategy] = useState('ucb1');
  const [isTraining, setIsTraining] = useState(false);
  const [round, setRound] = useState(0);
  const [c, setC] = useState(2.0); // UCB参数
  const [alpha, setAlpha] = useState(0.5); // UCB-V参数
  
  const [arms] = useState([
    { id: 0, name: '臂1', trueMean: 0.3 },
    { id: 1, name: '臂2', trueMean: 0.5 },
    { id: 2, name: '臂3', trueMean: 0.7 },
    { id: 3, name: '臂4', trueMean: 0.2 }
  ]);

  const [armStats, setArmStats] = useState(
    arms.map(arm => ({
      id: arm.id,
      name: arm.name,
      pulls: 0,
      totalReward: 0,
      avgReward: 0,
      ucbValue: 0,
      confidence: 0
    }))
  );

  const [rewardHistory, setRewardHistory] = useState<any[]>([]);
  const [regretHistory, setRegretHistory] = useState<any[]>([]);
  const [cumulativeRegret, setCumulativeRegret] = useState(0);

  const strategies = [
    { value: 'ucb1', label: 'UCB1', description: '经典上置信界算法' },
    { value: 'ucb2', label: 'UCB2', description: '改进的UCB算法' },
    { value: 'ucb_v', label: 'UCB-V', description: '基于方差的UCB' },
    { value: 'ucb_normal', label: 'UCB-Normal', description: '正态分布UCB' }
  ];

  const calculateUCB = (armStat: any, totalPulls: number, strategy: string) => {
    if (armStat.pulls === 0) return Infinity;
    
    const meanReward = armStat.avgReward;
    const n = armStat.pulls;
    const t = totalPulls;
    
    switch (strategy) {
      case 'ucb1':
        return meanReward + c * Math.sqrt(Math.log(t) / n);
      case 'ucb2':
        return meanReward + Math.sqrt((3 * Math.log(t)) / (2 * n));
      case 'ucb_v':
        // 简化的方差估计
        const variance = Math.max(0.25, armStat.variance || 0.25);
        return meanReward + Math.sqrt((2 * variance * Math.log(t)) / n) + (3 * Math.log(t)) / n;
      case 'ucb_normal':
        return meanReward + Math.sqrt((16 * Math.log(t - 1)) / n);
      default:
        return meanReward + c * Math.sqrt(Math.log(t) / n);
    }
  };

  const selectArm = () => {
    const totalPulls = armStats.reduce((sum, arm) => sum + arm.pulls, 0);
    
    // 确保每个臂至少被拉一次
    const unpulledArm = armStats.find(arm => arm.pulls === 0);
    if (unpulledArm) return unpulledArm.id;
    
    // 计算UCB值并选择最大的
    const ucbValues = armStats.map(arm => ({
      id: arm.id,
      ucb: calculateUCB(arm, totalPulls, strategy)
    }));
    
    return ucbValues.reduce((best, current) => 
      current.ucb > best.ucb ? current : best
    ).id;
  };

  const pullArm = (armId: number) => {
    const arm = arms[armId];
    // 模拟伯努利分布奖励
    const reward = Math.random() < arm.trueMean ? 1 : 0;
    
    setArmStats(prev => {
      const newStats = [...prev];
      const armStat = newStats[armId];
      armStat.pulls += 1;
      armStat.totalReward += reward;
      armStat.avgReward = armStat.totalReward / armStat.pulls;
      
      // 计算新的UCB值
      const totalPulls = newStats.reduce((sum, arm) => sum + arm.pulls, 0);
      newStats.forEach(stat => {
        stat.ucbValue = calculateUCB(stat, totalPulls, strategy);
        stat.confidence = stat.pulls > 0 ? 
          c * Math.sqrt(Math.log(totalPulls) / stat.pulls) : 0;
      });
      
      return newStats;
    });
    
    // 计算瞬时遗憾 (最优臂收益 - 当前选择收益)
    const optimalMean = Math.max(...arms.map(arm => arm.trueMean));
    const instantRegret = optimalMean - arm.trueMean;
    
    setCumulativeRegret(prev => prev + instantRegret);
    
    // 更新历史记录
    setRewardHistory(prev => {
      const newData = [...prev, {
        round: round + 1,
        armId,
        armName: arm.name,
        reward,
        avgReward: armStats[armId].avgReward
      }];
      return newData.slice(-100);
    });
    
    setRegretHistory(prev => {
      const newRegret = prev.length > 0 ? prev[prev.length - 1].cumulative + instantRegret : instantRegret;
      const newData = [...prev, {
        round: round + 1,
        instant: instantRegret,
        cumulative: newRegret
      }];
      return newData.slice(-100);
    });
  };

  const simulateTraining = () => {
    const interval = setInterval(() => {
      setRound(prev => {
        const newRound = prev + 1;
        const selectedArm = selectArm();
        pullArm(selectedArm);
        
        if (newRound >= 1000) {
          setIsTraining(false);
          return newRound;
        }
        return newRound;
      });
    }, 50);

    return () => clearInterval(interval);
  };

  useEffect(() => {
    if (isTraining) {
      const cleanup = simulateTraining();
      return cleanup;
    }
  }, [isTraining]);

  const resetTraining = () => {
    setRound(0);
    setCumulativeRegret(0);
    setArmStats(arms.map(arm => ({
      id: arm.id,
      name: arm.name,
      pulls: 0,
      totalReward: 0,
      avgReward: 0,
      ucbValue: 0,
      confidence: 0
    })));
    setRewardHistory([]);
    setRegretHistory([]);
  };

  const armColumns = [
    { title: '老虎机臂', dataIndex: 'name', key: 'name' },
    { title: '拉取次数', dataIndex: 'pulls', key: 'pulls' },
    { title: '平均奖励', dataIndex: 'avgReward', key: 'avgReward', 
      render: (val: number) => val.toFixed(3) },
    { title: 'UCB值', dataIndex: 'ucbValue', key: 'ucbValue', 
      render: (val: number) => val === Infinity ? '∞' : val.toFixed(3) },
    { title: '置信区间', dataIndex: 'confidence', key: 'confidence', 
      render: (val: number) => `±${val.toFixed(3)}` }
  ];

  const regretConfig = {
    data: regretHistory,
    xField: 'round',
    yField: 'cumulative',
    smooth: true,
    color: '#ff4d4f',
    point: { size: 2 },
  };

  const armRewardConfig = {
    data: armStats.map(arm => ({ name: arm.name, value: arm.avgReward, pulls: arm.pulls })),
    xField: 'name',
    yField: 'value',
    columnStyle: {
      fill: '#1890ff',
    },
    label: {
      position: 'middle' as const,
      style: {
        fill: '#FFFFFF',
        opacity: 0.6,
      },
    },
  };

  return (
    <div style={{ padding: '24px', background: '#f0f2f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <LineChartOutlined style={{ marginRight: '12px', color: '#1890ff' }} />
          Upper Confidence Bound (UCB) 策略系统
        </Title>
        <Text type="secondary" style={{ fontSize: '16px' }}>
          探索-利用权衡的最优解决方案，通过置信上界指导决策制定
        </Text>
      </div>

      <Row gutter={[24, 24]}>
        <Col span={24}>
          <Card title="UCB策略配置" extra={
            <Space>
              <Button 
                type={isTraining ? "default" : "primary"} 
                icon={isTraining ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
                onClick={() => setIsTraining(!isTraining)}
                size="large"
              >
                {isTraining ? '暂停实验' : '开始实验'}
              </Button>
              <Button icon={<ReloadOutlined />} onClick={resetTraining}>
                重置实验
              </Button>
            </Space>
          }>
            <Row gutter={[16, 16]}>
              <Col span={8}>
                <div>
                  <Text strong>UCB策略类型</Text>
                  <Select 
                    value={strategy} 
                    onChange={setStrategy}
                    style={{ width: '100%', marginTop: '8px' }}
                    size="large"
                  >
                    {strategies.map(s => (
                      <Option key={s.value} value={s.value}>{s.label}</Option>
                    ))}
                  </Select>
                  <Text type="secondary" style={{ marginTop: '4px', display: 'block' }}>
                    {strategies.find(s => s.value === strategy)?.description}
                  </Text>
                </div>
              </Col>
              <Col span={8}>
                <div>
                  <Text strong>探索参数 C: {c}</Text>
                  <Slider 
                    min={0.1} 
                    max={5.0} 
                    step={0.1} 
                    value={c} 
                    onChange={setC}
                    style={{ marginTop: '8px' }}
                  />
                  <Text type="secondary">控制探索与利用的平衡</Text>
                </div>
              </Col>
              <Col span={8}>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <div>
                    <Text strong>当前轮次: </Text>
                    <Tag color="blue" style={{ fontSize: '14px' }}>{round}</Tag>
                  </div>
                  <div>
                    <Text strong>累积遗憾: </Text>
                    <Tag color="red" style={{ fontSize: '14px' }}>{cumulativeRegret.toFixed(3)}</Tag>
                  </div>
                </Space>
              </Col>
            </Row>
          </Card>
        </Col>

        <Col span={16}>
          <Card title="老虎机臂状态监控">
            <Table 
              columns={armColumns} 
              dataSource={armStats} 
              pagination={false}
              size="small"
            />
            <div style={{ marginTop: '16px', height: '200px' }}>
              <Column {...armRewardConfig} />
            </div>
          </Card>
        </Col>

        <Col span={8}>
          <Card title="实时统计">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Statistic 
                title="总拉取次数" 
                value={armStats.reduce((sum, arm) => sum + arm.pulls, 0)} 
                prefix={<CalculatorOutlined />}
              />
              <Statistic 
                title="累积遗憾" 
                value={cumulativeRegret} 
                precision={3}
                valueStyle={{ color: '#cf1322' }}
              />
              <Statistic 
                title="最佳臂识别" 
                value={armStats.reduce((best, arm) => 
                  arm.avgReward > best.avgReward ? arm : best, armStats[0]
                ).name}
                prefix={<TrophyOutlined />}
              />
              <div style={{ marginTop: '16px' }}>
                <Text strong>算法收敛性</Text>
                <Progress 
                  percent={Math.min((round / 1000) * 100, 100)}
                  status={isTraining ? 'active' : 'normal'}
                  strokeColor="#52c41a"
                />
              </div>
            </Space>
          </Card>
        </Col>

        <Col span={24}>
          <Tabs defaultActiveKey="regret">
            <TabPane tab="遗憾分析" key="regret">
              <Card title="累积遗憾曲线" size="small">
                <Alert 
                  message="遗憾函数分析" 
                  description="UCB算法的理论遗憾上界为O(log T)，实际性能通常优于理论界限"
                  variant="default" 
                  showIcon 
                  style={{ marginBottom: '16px' }}
                />
                <div style={{ height: '300px' }}>
                  <Line {...regretConfig} />
                </div>
              </Card>
            </TabPane>
            
            <TabPane tab="置信区间" key="confidence">
              <Row gutter={[16, 16]}>
                {armStats.map(arm => (
                  <Col span={12} key={arm.id}>
                    <Card title={arm.name} size="small">
                      <Space direction="vertical" style={{ width: '100%' }}>
                        <div>估计均值: {arm.avgReward.toFixed(3)}</div>
                        <div>置信半径: ±{arm.confidence.toFixed(3)}</div>
                        <div>UCB值: {arm.ucbValue === Infinity ? '∞' : arm.ucbValue.toFixed(3)}</div>
                        <Progress 
                          percent={(arm.pulls / Math.max(...armStats.map(a => a.pulls)) * 100) || 0}
                          size="small"
                          strokeColor={arm.id === 2 ? '#52c41a' : '#1890ff'}
                        />
                      </Space>
                    </Card>
                  </Col>
                ))}
              </Row>
            </TabPane>
            
            <TabPane tab="算法比较" key="comparison">
              <Row gutter={[16, 16]}>
                <Col span={12}>
                  <Card title="UCB算法族比较" size="small">
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Alert message="UCB1" description="适用于奖励分布未知的情况，理论保证强" type="success" />
                      <Alert message="UCB2" description="在某些情况下性能更好，但计算复杂度较高" variant="default" />
                      <Alert message="UCB-V" description="利用方差信息，适合奖励方差已知的场景" variant="warning" />
                      <Alert message="UCB-Normal" description="假设奖励服从正态分布，收敛更快" variant="destructive" />
                    </Space>
                  </Card>
                </Col>
                <Col span={12}>
                  <Card title="性能指标" size="small">
                    <Statistic.Group>
                      <Statistic title="平均遗憾" value={round > 0 ? (cumulativeRegret / round).toFixed(4) : 0} />
                      <Statistic title="最优臂选择率" value={`${((armStats[2]?.pulls || 0) / Math.max(round, 1) * 100).toFixed(1)}%`} />
                    </Statistic.Group>
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

export default UCBStrategiesPage;