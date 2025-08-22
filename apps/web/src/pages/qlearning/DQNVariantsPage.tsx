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
  Switch,
  Table
} from 'antd';
import { Line } from '@ant-design/charts';
import {
  ExperimentOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  ReloadOutlined,
  SettingOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;

const DQNVariantsPage: React.FC = () => {
  const [variant, setVariant] = useState('double_dqn');
  const [isTraining, setIsTraining] = useState(false);
  const [episode, setEpisode] = useState(0);
  const [epsilon, setEpsilon] = useState(0.1);
  const [targetUpdateFreq, setTargetUpdateFreq] = useState(1000);
  const [enablePrioritizedReplay, setEnablePrioritizedReplay] = useState(false);
  
  const [metrics, setMetrics] = useState({
    loss: 0,
    avgReward: 0,
    qValue: 0,
    targetQValue: 0
  });

  const [performanceData, setPerformanceData] = useState<any[]>([]);
  const [networkArchitecture, setNetworkArchitecture] = useState({
    layers: [128, 128],
    activation: 'relu',
    optimizer: 'adam'
  });

  const variants = [
    { value: 'double_dqn', label: 'Double DQN', description: '解决Q值过估计问题' },
    { value: 'dueling_dqn', label: 'Dueling DQN', description: '分离状态价值与优势函数' },
    { value: 'prioritized_dqn', label: 'Prioritized DQN', description: '优先经验回放' },
    { value: 'rainbow_dqn', label: 'Rainbow DQN', description: '集成多种改进技术' }
  ];

  const architectureColumns = [
    { title: '层类型', dataIndex: 'type', key: 'type' },
    { title: '神经元数量', dataIndex: 'neurons', key: 'neurons' },
    { title: '激活函数', dataIndex: 'activation', key: 'activation' },
    { title: '参数量', dataIndex: 'params', key: 'params' }
  ];

  const architectureData = [
    { key: '1', type: '输入层', neurons: 84, activation: '-', params: '0' },
    { key: '2', type: '隐藏层1', neurons: 128, activation: 'ReLU', params: '10,880' },
    { key: '3', type: '隐藏层2', neurons: 128, activation: 'ReLU', params: '16,512' },
    { key: '4', type: 'Value流', neurons: 1, activation: 'Linear', params: '129' },
    { key: '5', type: 'Advantage流', neurons: 4, activation: 'Linear', params: '516' }
  ];

  const simulateTraining = () => {
    const interval = setInterval(() => {
      setEpisode(prev => {
        const newEpisode = prev + 1;
        const reward = Math.sin(newEpisode * 0.1) * 10 + 15 + Math.random() * 5;
        const loss = Math.exp(-newEpisode * 0.01) * 2 + Math.random() * 0.5;
        
        setMetrics({
          loss: loss,
          avgReward: reward,
          qValue: Math.random() * 10 + 5,
          targetQValue: Math.random() * 9 + 4.5
        });

        setPerformanceData(prevData => {
          const newData = [...prevData, {
            episode: newEpisode,
            reward: reward,
            loss: loss,
            type: '训练奖励'
          }];
          return newData.slice(-100);
        });

        if (newEpisode >= 1000) {
          setIsTraining(false);
          return newEpisode;
        }
        return newEpisode;
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
    setEpisode(0);
    setPerformanceData([]);
    setMetrics({
      loss: 0,
      avgReward: 0,
      qValue: 0,
      targetQValue: 0
    });
  };

  const chartConfig = {
    data: performanceData,
    xField: 'episode',
    yField: 'reward',
    seriesField: 'type',
    smooth: true,
    animation: {
      appear: {
        animation: 'path-in',
        duration: 1000,
      },
    },
  };

  const getVariantDescription = () => {
    const current = variants.find(v => v.value === variant);
    return current?.description || '';
  };

  return (
    <div style={{ padding: '24px', background: '#f0f2f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <ExperimentOutlined style={{ marginRight: '12px', color: '#1890ff' }} />
          DQN算法变体训练系统
        </Title>
        <Text type="secondary" style={{ fontSize: '16px' }}>
          对比不同DQN变体的性能表现，深入理解各种改进技术的原理和效果
        </Text>
      </div>

      <Row gutter={[24, 24]}>
        <Col span={24}>
          <Card title="变体配置与控制" extra={
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
                重置训练
              </Button>
            </Space>
          }>
            <Row gutter={[16, 16]}>
              <Col span={8}>
                <div>
                  <Text strong>DQN变体选择</Text>
                  <Select 
                    value={variant} 
                    onChange={setVariant}
                    style={{ width: '100%', marginTop: '8px' }}
                    size="large"
                  >
                    {variants.map(v => (
                      <Option key={v.value} value={v.value}>{v.label}</Option>
                    ))}
                  </Select>
                  <Text type="secondary" style={{ marginTop: '4px', display: 'block' }}>
                    {getVariantDescription()}
                  </Text>
                </div>
              </Col>
              <Col span={8}>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <div>
                    <Text strong>Epsilon (探索率): {epsilon}</Text>
                    <Slider 
                      min={0.01} 
                      max={1.0} 
                      step={0.01} 
                      value={epsilon} 
                      onChange={setEpsilon}
                    />
                  </div>
                  <div>
                    <Text strong>目标网络更新频率: {targetUpdateFreq}</Text>
                    <Slider 
                      min={100} 
                      max={2000} 
                      step={100} 
                      value={targetUpdateFreq} 
                      onChange={setTargetUpdateFreq}
                    />
                  </div>
                </Space>
              </Col>
              <Col span={8}>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <div>
                    <Text strong>优先经验回放</Text>
                    <br />
                    <Switch 
                      checked={enablePrioritizedReplay} 
                      onChange={setEnablePrioritizedReplay}
                      style={{ marginTop: '8px' }}
                    />
                  </div>
                  <Tag color={isTraining ? 'green' : 'default'} style={{ marginTop: '8px' }}>
                    {isTraining ? '训练中' : '已停止'}
                  </Tag>
                </Space>
              </Col>
            </Row>
          </Card>
        </Col>

        <Col span={24}>
          <Card title="训练性能监控">
            <Row gutter={[16, 16]}>
              <Col span={6}>
                <Statistic 
                  title="当前回合" 
                  value={episode} 
                  prefix={<PlayCircleOutlined />}
                />
              </Col>
              <Col span={6}>
                <Statistic 
                  title="平均奖励" 
                  value={metrics.avgReward} 
                  precision={2}
                  valueStyle={{ color: metrics.avgReward > 10 ? '#3f8600' : '#cf1322' }}
                />
              </Col>
              <Col span={6}>
                <Statistic 
                  title="训练损失" 
                  value={metrics.loss} 
                  precision={4}
                  valueStyle={{ color: '#1890ff' }}
                />
              </Col>
              <Col span={6}>
                <Statistic 
                  title="Q值估计" 
                  value={metrics.qValue} 
                  precision={2}
                  suffix="/ 目标Q值"
                />
              </Col>
            </Row>
            <div style={{ marginTop: '24px', height: '300px' }}>
              <Line {...chartConfig} />
            </div>
          </Card>
        </Col>

        <Col span={24}>
          <Tabs defaultActiveKey="architecture">
            <TabPane tab="网络架构" key="architecture" icon={<SettingOutlined />}>
              <Row gutter={[24, 24]}>
                <Col span={12}>
                  <Card title="Dueling DQN架构" size="small">
                    <Table 
                      columns={architectureColumns} 
                      dataSource={architectureData} 
                      pagination={false}
                      size="small"
                    />
                  </Card>
                </Col>
                <Col span={12}>
                  <Card title="算法对比" size="small">
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Alert 
                        message="Double DQN" 
                        description="使用主网络选择动作，目标网络评估Q值，有效减少过估计偏差"
                        variant="default" 
                        showIcon 
                      />
                      <Alert 
                        message="Dueling DQN" 
                        description="将Q网络分解为状态价值函数V(s)和优势函数A(s,a)两个流"
                        type="success" 
                        showIcon 
                      />
                      <Alert 
                        message="Prioritized Replay" 
                        description="根据TD误差优先采样重要经验，提高学习效率"
                        variant="warning" 
                        showIcon 
                      />
                    </Space>
                  </Card>
                </Col>
              </Row>
            </TabPane>
            
            <TabPane tab="训练进度" key="progress">
              <Row gutter={[16, 16]}>
                <Col span={12}>
                  <Card title="训练进度" size="small">
                    <Progress 
                      percent={Math.min((episode / 1000) * 100, 100)} 
                      status={isTraining ? 'active' : 'normal'}
                      strokeColor={{
                        '0%': '#108ee9',
                        '100%': '#87d068',
                      }}
                    />
                    <div style={{ marginTop: '16px' }}>
                      <Text>总回合数: {episode} / 1000</Text>
                    </div>
                  </Card>
                </Col>
                <Col span={12}>
                  <Card title="算法状态" size="small">
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <div>探索策略: Epsilon-Greedy (ε={epsilon})</div>
                      <div>目标网络更新: 每{targetUpdateFreq}步</div>
                      <div>经验回放: {enablePrioritizedReplay ? '优先' : '均匀'}采样</div>
                      <div>当前变体: <Tag color="blue">{variants.find(v => v.value === variant)?.label}</Tag></div>
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

export default DQNVariantsPage;