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
  Switch,
  Timeline
} from 'antd';
import { Line, Column } from '@ant-design/charts';
import {
  ControlOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  ReloadOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;

const TrainingManagerPage: React.FC = () => {
  const [isTraining, setIsTraining] = useState(false);
  const [currentEpisode, setCurrentEpisode] = useState(0);
  const [totalEpisodes] = useState(1000);
  const [batchSize, setBatchSize] = useState(32);
  const [learningRate, setLearningRate] = useState(0.001);
  const [saveInterval, setSaveInterval] = useState(100);
  
  const [trainingMetrics, setTrainingMetrics] = useState({
    loss: 0,
    reward: 0,
    epsilon: 1.0,
    qValue: 0
  });
  
  const [trainingHistory, setTrainingHistory] = useState<any[]>([]);
  const [checkpoints, setCheckpoints] = useState<any[]>([]);
  const [trainingJobs, setTrainingJobs] = useState([
    { id: 1, name: 'DQN-CartPole', status: 'completed', progress: 100, startTime: '2024-01-15 10:00' },
    { id: 2, name: 'A3C-Atari', status: 'running', progress: 65, startTime: '2024-01-15 14:30' },
    { id: 3, name: 'PPO-Walker', status: 'pending', progress: 0, startTime: null }
  ]);

  const [systemResources, setSystemResources] = useState({
    cpuUsage: 75,
    memoryUsage: 60,
    gpuUsage: 85,
    diskUsage: 45
  });

  const simulateTraining = () => {
    const interval = setInterval(() => {
      setCurrentEpisode(prev => {
        const newEpisode = prev + 1;
        
        // 模拟训练指标
        const loss = Math.max(0.01, 5.0 * Math.exp(-newEpisode * 0.005) + Math.random() * 0.1);
        const reward = Math.sin(newEpisode * 0.01) * 20 + 50 + Math.random() * 10;
        const epsilon = Math.max(0.05, 1.0 - newEpisode * 0.001);
        const qValue = Math.random() * 15 + 5;
        
        setTrainingMetrics({ loss, reward, epsilon, qValue });
        
        // 更新训练历史
        setTrainingHistory(prev => {
          const newData = [...prev, {
            episode: newEpisode,
            loss,
            reward,
            epsilon,
            qValue,
            timestamp: new Date().toLocaleTimeString()
          }];
          return newData.slice(-200);
        });
        
        // 创建检查点
        if (newEpisode % saveInterval === 0) {
          setCheckpoints(prev => [...prev, {
            episode: newEpisode,
            timestamp: new Date().toLocaleString(),
            metrics: { loss, reward, epsilon, qValue },
            id: Date.now()
          }].slice(-10));
        }
        
        // 模拟系统资源变化
        setSystemResources(prev => ({
          cpuUsage: Math.max(10, Math.min(95, prev.cpuUsage + (Math.random() - 0.5) * 10)),
          memoryUsage: Math.max(10, Math.min(90, prev.memoryUsage + (Math.random() - 0.5) * 5)),
          gpuUsage: Math.max(20, Math.min(98, prev.gpuUsage + (Math.random() - 0.5) * 8)),
          diskUsage: Math.max(5, Math.min(80, prev.diskUsage + (Math.random() - 0.5) * 2))
        }));
        
        if (newEpisode >= totalEpisodes) {
          setIsTraining(false);
          return newEpisode;
        }
        return newEpisode;
      });
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
    setCurrentEpisode(0);
    setTrainingHistory([]);
    setCheckpoints([]);
    setTrainingMetrics({ loss: 0, reward: 0, epsilon: 1.0, qValue: 0 });
  };

  const metricsConfig = {
    data: trainingHistory,
    xField: 'episode',
    yField: 'reward',
    seriesField: 'type',
    smooth: true,
  };

  const lossConfig = {
    data: trainingHistory,
    xField: 'episode',
    yField: 'loss',
    smooth: true,
    color: '#ff4d4f',
  };

  const jobColumns = [
    { title: '任务名称', dataIndex: 'name', key: 'name' },
    { title: '状态', dataIndex: 'status', key: 'status',
      render: (status: string) => {
        const colors = { completed: 'green', running: 'blue', pending: 'orange', failed: 'red' };
        const icons = { 
          completed: <CheckCircleOutlined />, 
          running: <ClockCircleOutlined />, 
          pending: <ExclamationCircleOutlined />,
          failed: <ExclamationCircleOutlined />
        };
        return <Tag color={colors[status as keyof typeof colors]} icon={icons[status as keyof typeof icons]}>{status}</Tag>;
      }
    },
    { title: '进度', dataIndex: 'progress', key: 'progress',
      render: (progress: number) => <Progress percent={progress} size="small" />
    },
    { title: '开始时间', dataIndex: 'startTime', key: 'startTime' },
    { title: '操作', key: 'action',
      render: (_: any, record: any) => (
        <Space>
          {record.status === 'running' && <Button size="small" danger>停止</Button>}
          {record.status === 'pending' && <Button size="small" type="primary">启动</Button>}
          <Button size="small">查看日志</Button>
        </Space>
      )
    }
  ];

  const checkpointColumns = [
    { title: '回合数', dataIndex: 'episode', key: 'episode' },
    { title: '保存时间', dataIndex: 'timestamp', key: 'timestamp' },
    { title: '损失', dataIndex: 'metrics', key: 'loss',
      render: (metrics: any) => metrics.loss.toFixed(4)
    },
    { title: '奖励', dataIndex: 'metrics', key: 'reward',
      render: (metrics: any) => metrics.reward.toFixed(2)
    },
    { title: '操作', key: 'action',
      render: (_: any, record: any) => (
        <Space>
          <Button size="small" type="primary">加载</Button>
          <Button size="small">下载</Button>
        </Space>
      )
    }
  ];

  return (
    <div style={{ padding: '24px', background: '#f0f2f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <ControlOutlined style={{ marginRight: '12px', color: '#1890ff' }} />
          训练调度管理系统
        </Title>
        <Text type="secondary" style={{ fontSize: '16px' }}>
          统一管理多个强化学习训练任务，监控训练过程和系统资源
        </Text>
      </div>

      <Row gutter={[24, 24]}>
        <Col span={24}>
          <Card title="当前训练任务控制" extra={
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
              <Col span={6}>
                <div>
                  <Text strong>批量大小: {batchSize}</Text>
                  <Slider 
                    min={16} 
                    max={128} 
                    value={batchSize} 
                    onChange={setBatchSize}
                    disabled={isTraining}
                    style={{ marginTop: '8px' }}
                  />
                </div>
              </Col>
              <Col span={6}>
                <div>
                  <Text strong>学习率: {learningRate}</Text>
                  <Slider 
                    min={0.0001} 
                    max={0.01} 
                    step={0.0001}
                    value={learningRate} 
                    onChange={setLearningRate}
                    disabled={isTraining}
                    style={{ marginTop: '8px' }}
                  />
                </div>
              </Col>
              <Col span={6}>
                <div>
                  <Text strong>保存间隔: {saveInterval} 回合</Text>
                  <Slider 
                    min={50} 
                    max={500} 
                    step={50}
                    value={saveInterval} 
                    onChange={setSaveInterval}
                    disabled={isTraining}
                    style={{ marginTop: '8px' }}
                  />
                </div>
              </Col>
              <Col span={6}>
                <Space direction="vertical">
                  <Text strong>训练进度: {currentEpisode}/{totalEpisodes}</Text>
                  <Progress 
                    percent={Math.min((currentEpisode / totalEpisodes) * 100, 100)}
                    status={isTraining ? 'active' : 'normal'}
                    strokeColor="#52c41a"
                  />
                </Space>
              </Col>
            </Row>
          </Card>
        </Col>

        <Col span={18}>
          <Card title="训练指标监控">
            <Row gutter={[16, 16]} style={{ marginBottom: '16px' }}>
              <Col span={6}>
                <Statistic 
                  title="当前损失" 
                  value={trainingMetrics.loss} 
                  precision={4}
                  valueStyle={{ color: '#ff4d4f' }}
                />
              </Col>
              <Col span={6}>
                <Statistic 
                  title="平均奖励" 
                  value={trainingMetrics.reward} 
                  precision={2}
                  valueStyle={{ color: trainingMetrics.reward > 50 ? '#3f8600' : '#cf1322' }}
                />
              </Col>
              <Col span={6}>
                <Statistic 
                  title="Epsilon值" 
                  value={trainingMetrics.epsilon} 
                  precision={3}
                  valueStyle={{ color: '#1890ff' }}
                />
              </Col>
              <Col span={6}>
                <Statistic 
                  title="Q值估计" 
                  value={trainingMetrics.qValue} 
                  precision={2}
                  valueStyle={{ color: '#722ed1' }}
                />
              </Col>
            </Row>
            
            <Tabs defaultActiveKey="reward">
              <TabPane tab="奖励曲线" key="reward">
                <div style={{ height: '250px' }}>
                  <Line {...metricsConfig} />
                </div>
              </TabPane>
              <TabPane tab="损失曲线" key="loss">
                <div style={{ height: '250px' }}>
                  <Line {...lossConfig} />
                </div>
              </TabPane>
            </Tabs>
          </Card>
        </Col>

        <Col span={6}>
          <Card title="系统资源监控">
            <Space direction="vertical" style={{ width: '100%' }}>
              <div>
                <Text strong>CPU使用率</Text>
                <Progress 
                  percent={systemResources.cpuUsage} 
                  strokeColor={systemResources.cpuUsage > 80 ? '#ff4d4f' : '#52c41a'}
                  size="small"
                />
              </div>
              <div>
                <Text strong>内存使用率</Text>
                <Progress 
                  percent={systemResources.memoryUsage} 
                  strokeColor={systemResources.memoryUsage > 80 ? '#ff4d4f' : '#1890ff'}
                  size="small"
                />
              </div>
              <div>
                <Text strong>GPU使用率</Text>
                <Progress 
                  percent={systemResources.gpuUsage} 
                  strokeColor={systemResources.gpuUsage > 90 ? '#ff4d4f' : '#722ed1'}
                  size="small"
                />
              </div>
              <div>
                <Text strong>磁盘使用率</Text>
                <Progress 
                  percent={systemResources.diskUsage} 
                  strokeColor={systemResources.diskUsage > 70 ? '#faad14' : '#52c41a'}
                  size="small"
                />
              </div>
            </Space>
            
            <Alert 
              message="资源警告" 
              description={systemResources.gpuUsage > 90 ? "GPU使用率过高" : "系统资源正常"}
              type={systemResources.gpuUsage > 90 ? "warning" : "success"}
              style={{ marginTop: '16px' }}
            />
          </Card>
        </Col>

        <Col span={24}>
          <Tabs defaultActiveKey="jobs">
            <TabPane tab="训练任务管理" key="jobs">
              <Table 
                columns={jobColumns} 
                dataSource={trainingJobs} 
                pagination={false}
                size="small"
              />
            </TabPane>
            
            <TabPane tab="模型检查点" key="checkpoints">
              <Table 
                columns={checkpointColumns} 
                dataSource={checkpoints} 
                pagination={false}
                size="small"
              />
            </TabPane>
            
            <TabPane tab="训练日志" key="logs">
              <Timeline>
                {trainingHistory.slice(-10).reverse().map((entry, index) => (
                  <Timeline.Item key={index} color={entry.reward > 50 ? 'green' : 'red'}>
                    <div>
                      <Text strong>回合 {entry.episode}</Text> - {entry.timestamp}
                      <br />
                      <Text type="secondary">
                        奖励: {entry.reward.toFixed(2)}, 损失: {entry.loss.toFixed(4)}, Epsilon: {entry.epsilon.toFixed(3)}
                      </Text>
                    </div>
                  </Timeline.Item>
                ))}
              </Timeline>
            </TabPane>
            
            <TabPane tab="训练配置" key="config">
              <Row gutter={[16, 16]}>
                <Col span={12}>
                  <Card title="超参数配置" size="small">
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <div>学习率: {learningRate}</div>
                      <div>批量大小: {batchSize}</div>
                      <div>目标网络更新频率: 1000</div>
                      <div>经验回放缓冲区大小: 100000</div>
                      <div>Epsilon衰减率: 0.995</div>
                    </Space>
                  </Card>
                </Col>
                <Col span={12}>
                  <Card title="训练策略" size="small">
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <div>算法: DQN</div>
                      <div>网络架构: [128, 128]</div>
                      <div>激活函数: ReLU</div>
                      <div>优化器: Adam</div>
                      <div>损失函数: MSE</div>
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

export default TrainingManagerPage;