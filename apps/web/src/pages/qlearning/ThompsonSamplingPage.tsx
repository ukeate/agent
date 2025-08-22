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
  Tooltip
} from 'antd';
import { Line, Column } from '@ant-design/charts';
import {
  ExperimentOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  ReloadOutlined,
  FunctionOutlined,
  BarChartOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;

const ThompsonSamplingPage: React.FC = () => {
  const [distribution, setDistribution] = useState('beta');
  const [isTraining, setIsTraining] = useState(false);
  const [round, setRound] = useState(0);
  
  const [arms] = useState([
    { id: 0, name: '臂A', trueMean: 0.3, color: '#ff4d4f' },
    { id: 1, name: '臂B', trueMean: 0.5, color: '#1890ff' },
    { id: 2, name: '臂C', trueMean: 0.7, color: '#52c41a' },
    { id: 3, name: '臂D', trueMean: 0.2, color: '#faad14' }
  ]);

  const [armStats, setArmStats] = useState(
    arms.map(arm => ({
      id: arm.id,
      name: arm.name,
      alpha: 1, // Beta分布参数 (成功次数 + 1)
      beta: 1,  // Beta分布参数 (失败次数 + 1)
      pulls: 0,
      successes: 0,
      avgReward: 0,
      sampledValue: 0,
      credibleInterval: [0, 1]
    }))
  );

  const [selectionHistory, setSelectionHistory] = useState<any[]>([]);
  const [distributionData, setDistributionData] = useState<any[]>([]);
  const [cumulativeRegret, setCumulativeRegret] = useState(0);

  const distributions = [
    { value: 'beta', label: 'Beta分布', description: '适用于伯努利奖励' },
    { value: 'gaussian', label: '高斯分布', description: '适用于连续奖励' },
    { value: 'gamma', label: 'Gamma分布', description: '适用于正向连续奖励' }
  ];

  // Beta分布采样
  const sampleFromBeta = (alpha: number, beta: number): number => {
    // 使用简化的Beta分布采样（实际应用中需要更精确的实现）
    const x = Math.random();
    const y = Math.random();
    const logX = Math.log(x) / alpha;
    const logY = Math.log(y) / beta;
    return Math.exp(logX) / (Math.exp(logX) + Math.exp(logY));
  };

  // 计算Beta分布的置信区间
  const getBetaCredibleInterval = (alpha: number, beta: number, confidence = 0.95): [number, number] => {
    // 简化计算，实际应用中应使用更精确的分位数计算
    const mean = alpha / (alpha + beta);
    const variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1));
    const std = Math.sqrt(variance);
    const margin = 1.96 * std; // 95%置信区间
    return [Math.max(0, mean - margin), Math.min(1, mean + margin)];
  };

  const selectArm = (): number => {
    // 从每个臂的后验分布中采样
    const samples = armStats.map(arm => ({
      id: arm.id,
      sample: distribution === 'beta' 
        ? sampleFromBeta(arm.alpha, arm.beta)
        : Math.random() // 简化其他分布
    }));

    // 选择采样值最大的臂
    return samples.reduce((best, current) => 
      current.sample > best.sample ? current : best
    ).id;
  };

  const pullArm = (armId: number): void => {
    const arm = arms[armId];
    const reward = Math.random() < arm.trueMean ? 1 : 0; // 伯努利奖励
    
    setArmStats(prev => {
      const newStats = [...prev];
      const armStat = newStats[armId];
      
      armStat.pulls += 1;
      if (reward === 1) {
        armStat.successes += 1;
        armStat.alpha += 1; // 贝叶斯更新
      } else {
        armStat.beta += 1;   // 贝叶斯更新
      }
      
      armStat.avgReward = armStat.successes / armStat.pulls;
      armStat.credibleInterval = getBetaCredibleInterval(armStat.alpha, armStat.beta);
      
      return newStats;
    });

    // 计算遗憾
    const optimalMean = Math.max(...arms.map(arm => arm.trueMean));
    const instantRegret = optimalMean - arm.trueMean;
    setCumulativeRegret(prev => prev + instantRegret);

    // 记录选择历史
    setSelectionHistory(prev => {
      const newData = [...prev, {
        round: round + 1,
        selectedArm: armId,
        armName: arm.name,
        reward,
        cumulativeRegret: cumulativeRegret + instantRegret
      }];
      return newData.slice(-200);
    });
  };

  const generateDistributionData = (): void => {
    const data: any[] = [];
    const resolution = 100;
    
    for (let i = 0; i <= resolution; i++) {
      const x = i / resolution;
      armStats.forEach(arm => {
        if (arm.pulls > 0) {
          // 简化的Beta分布密度函数
          const alpha = arm.alpha;
          const beta = arm.beta;
          const density = Math.pow(x, alpha - 1) * Math.pow(1 - x, beta - 1);
          
          data.push({
            x,
            density: density * 100, // 缩放以便可视化
            arm: arm.name,
            armId: arm.id
          });
        }
      });
    }
    
    setDistributionData(data);
  };

  const simulateTraining = () => {
    const interval = setInterval(() => {
      setRound(prev => {
        const newRound = prev + 1;
        const selectedArm = selectArm();
        pullArm(selectedArm);
        
        if (newRound % 10 === 0) {
          generateDistributionData();
        }
        
        if (newRound >= 1000) {
          setIsTraining(false);
          return newRound;
        }
        return newRound;
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

  useEffect(() => {
    generateDistributionData();
  }, [armStats]);

  const resetTraining = () => {
    setRound(0);
    setCumulativeRegret(0);
    setArmStats(arms.map(arm => ({
      id: arm.id,
      name: arm.name,
      alpha: 1,
      beta: 1,
      pulls: 0,
      successes: 0,
      avgReward: 0,
      sampledValue: 0,
      credibleInterval: [0, 1]
    })));
    setSelectionHistory([]);
    setDistributionData([]);
  };

  const armColumns = [
    { title: '老虎机臂', dataIndex: 'name', key: 'name' },
    { title: '拉取次数', dataIndex: 'pulls', key: 'pulls' },
    { title: '成功次数', dataIndex: 'successes', key: 'successes' },
    { title: '成功率', dataIndex: 'avgReward', key: 'avgReward',
      render: (val: number) => `${(val * 100).toFixed(1)}%` },
    { title: 'α参数', dataIndex: 'alpha', key: 'alpha' },
    { title: 'β参数', dataIndex: 'beta', key: 'beta' },
    { title: '95%置信区间', dataIndex: 'credibleInterval', key: 'credibleInterval',
      render: (interval: [number, number]) => 
        `[${(interval[0] * 100).toFixed(1)}%, ${(interval[1] * 100).toFixed(1)}%]`
    }
  ];

  const regretConfig = {
    data: selectionHistory,
    xField: 'round',
    yField: 'cumulativeRegret',
    smooth: true,
    color: '#ff4d4f',
  };

  const distributionConfig = {
    data: distributionData,
    xField: 'x',
    yField: 'density',
    seriesField: 'arm',
    smooth: true,
    color: ['#ff4d4f', '#1890ff', '#52c41a', '#faad14'],
  };

  const armSelectionConfig = {
    data: armStats.map(arm => ({ 
      name: arm.name, 
      pulls: arm.pulls,
      rate: armStats.reduce((sum, a) => sum + a.pulls, 0) > 0 
        ? (arm.pulls / armStats.reduce((sum, a) => sum + a.pulls, 0) * 100).toFixed(1)
        : '0'
    })),
    xField: 'name',
    yField: 'pulls',
    columnStyle: {
      fill: '#1890ff',
    },
  };

  return (
    <div style={{ padding: '24px', background: '#f0f2f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <ExperimentOutlined style={{ marginRight: '12px', color: '#1890ff' }} />
          Thompson采样算法系统
        </Title>
        <Text type="secondary" style={{ fontSize: '16px' }}>
          贝叶斯学习方法，通过后验分布采样实现最优的探索-利用平衡
        </Text>
      </div>

      <Row gutter={[24, 24]}>
        <Col span={24}>
          <Card title="Thompson采样配置" extra={
            <Space>
              <Button 
                type={isTraining ? "default" : "primary"} 
                icon={isTraining ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
                onClick={() => setIsTraining(!isTraining)}
                size="large"
              >
                {isTraining ? '暂停采样' : '开始采样'}
              </Button>
              <Button icon={<ReloadOutlined />} onClick={resetTraining}>
                重置实验
              </Button>
            </Space>
          }>
            <Row gutter={[16, 16]}>
              <Col span={12}>
                <div>
                  <Text strong>后验分布类型</Text>
                  <Select 
                    value={distribution} 
                    onChange={setDistribution}
                    style={{ width: '100%', marginTop: '8px' }}
                    size="large"
                  >
                    {distributions.map(d => (
                      <Option key={d.value} value={d.value}>{d.label}</Option>
                    ))}
                  </Select>
                  <Text type="secondary" style={{ marginTop: '4px', display: 'block' }}>
                    {distributions.find(d => d.value === distribution)?.description}
                  </Text>
                </div>
              </Col>
              <Col span={12}>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <div>
                    <Text strong>当前轮次: </Text>
                    <Tag color="blue" style={{ fontSize: '16px' }}>{round}</Tag>
                  </div>
                  <div>
                    <Text strong>累积遗憾: </Text>
                    <Tag color="red" style={{ fontSize: '16px' }}>{cumulativeRegret.toFixed(3)}</Tag>
                  </div>
                  <div>
                    <Text strong>算法状态: </Text>
                    <Tag color={isTraining ? 'green' : 'default'}>
                      {isTraining ? '采样中' : '已停止'}
                    </Tag>
                  </div>
                </Space>
              </Col>
            </Row>
          </Card>
        </Col>

        <Col span={16}>
          <Card title="贝叶斯参数统计">
            <Table 
              columns={armColumns} 
              dataSource={armStats} 
              pagination={false}
              size="small"
            />
          </Card>
        </Col>

        <Col span={8}>
          <Card title="实时统计">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Statistic 
                title="总采样次数" 
                value={armStats.reduce((sum, arm) => sum + arm.pulls, 0)} 
                prefix={<FunctionOutlined />}
              />
              <Statistic 
                title="最佳臂识别" 
                value={armStats.reduce((best, arm) => 
                  arm.avgReward > best.avgReward ? arm : best, armStats[0]
                ).name}
                prefix={<BarChartOutlined />}
              />
              <div style={{ marginTop: '16px' }}>
                <Text strong>收敛程度</Text>
                <Progress 
                  percent={Math.min((round / 1000) * 100, 100)}
                  status={isTraining ? 'active' : 'normal'}
                  strokeColor="#722ed1"
                />
              </div>
            </Space>
          </Card>
        </Col>

        <Col span={24}>
          <Tabs defaultActiveKey="distributions">
            <TabPane tab="后验分布" key="distributions">
              <Card title="Beta分布后验更新" size="small">
                <Alert 
                  message="贝叶斯更新规则" 
                  description="观察到成功时α+=1，失败时β+=1。后验均值为α/(α+β)"
                  variant="default" 
                  showIcon 
                  style={{ marginBottom: '16px' }}
                />
                <div style={{ height: '300px' }}>
                  <Line {...distributionConfig} />
                </div>
              </Card>
            </TabPane>
            
            <TabPane tab="选择频率" key="selection">
              <Row gutter={[16, 16]}>
                <Col span={12}>
                  <Card title="臂选择分布" size="small">
                    <div style={{ height: '250px' }}>
                      <Column {...armSelectionConfig} />
                    </div>
                  </Card>
                </Col>
                <Col span={12}>
                  <Card title="置信区间演化" size="small">
                    {armStats.map((arm, index) => (
                      <div key={arm.id} style={{ marginBottom: '12px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <Text strong style={{ color: arms[index].color }}>{arm.name}</Text>
                          <Text>均值: {(arm.alpha / (arm.alpha + arm.beta)).toFixed(3)}</Text>
                        </div>
                        <Progress 
                          percent={arm.pulls > 0 ? (arm.avgReward * 100) : 0}
                          strokeColor={arms[index].color}
                          size="small"
                          format={() => `${(arm.avgReward * 100).toFixed(1)}%`}
                        />
                      </div>
                    ))}
                  </Card>
                </Col>
              </Row>
            </TabPane>
            
            <TabPane tab="遗憾分析" key="regret">
              <Card title="累积遗憾演化" size="small">
                <Alert 
                  message="Thompson采样的理论性质" 
                  description="在某些条件下，Thompson采样是最优的贝叶斯策略，具有优秀的遗憾界"
                  type="success" 
                  showIcon 
                  style={{ marginBottom: '16px' }}
                />
                <div style={{ height: '300px' }}>
                  <Line {...regretConfig} />
                </div>
              </Card>
            </TabPane>

            <TabPane tab="算法对比" key="comparison">
              <Row gutter={[16, 16]}>
                <Col span={12}>
                  <Card title="Thompson采样 vs UCB" size="small">
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Alert 
                        message="Thompson采样优势" 
                        description="1. 自然的贝叶斯框架 2. 优秀的实验性能 3. 处理延迟奖励能力强"
                        type="success" 
                      />
                      <Alert 
                        message="UCB算法优势" 
                        description="1. 理论保证更强 2. 参数调节简单 3. 计算开销较小"
                        variant="default" 
                      />
                    </Space>
                  </Card>
                </Col>
                <Col span={12}>
                  <Card title="性能指标对比" size="small">
                    <Statistic.Group>
                      <Statistic 
                        title="平均遗憾" 
                        value={round > 0 ? (cumulativeRegret / round).toFixed(4) : 0} 
                        precision={4}
                      />
                      <Statistic 
                        title="最优臂识别率" 
                        value={`${((armStats[2]?.pulls || 0) / Math.max(round, 1) * 100).toFixed(1)}%`}
                      />
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

export default ThompsonSamplingPage;