import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Row, 
  Col, 
  Typography, 
  Space, 
  Button, 
  Progress,
  Statistic,
  Tag,
  Descriptions,
  Alert,
  Tabs,
  Select,
  InputNumber,
  Divider
} from 'antd';
import { Line } from '@ant-design/charts';
import {
  RobotOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  LineChartOutlined,
  BranchesOutlined,
  DatabaseOutlined,
  ThunderboltOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;

const DQNPage: React.FC = () => {
  const [isTraining, setIsTraining] = useState(false);
  const [episode, setEpisode] = useState(0);
  const [networkType, setNetworkType] = useState('dqn');
  const [lossData, setLossData] = useState<any[]>([]);
  const [rewardData, setRewardData] = useState<any[]>([]);
  const [replayBufferSize, setReplayBufferSize] = useState(0);
  const [targetNetworkUpdates, setTargetNetworkUpdates] = useState(0);

  useEffect(() => {
    if (isTraining) {
      simulateTraining();
    }
  }, [isTraining]);

  const simulateTraining = () => {
    const interval = setInterval(() => {
      setEpisode(prev => {
        const newEpisode = prev + 1;
        
        // 模拟损失函数数据
        const newLoss = Math.exp(-newEpisode / 100) + Math.random() * 0.1;
        setLossData(prev => [...prev.slice(-99), { episode: newEpisode, loss: newLoss }]);
        
        // 模拟奖励数据
        const newReward = (newEpisode / 50) + Math.random() * 5;
        setRewardData(prev => [...prev.slice(-99), { episode: newEpisode, reward: newReward }]);
        
        // 模拟经验回放缓冲区大小
        setReplayBufferSize(Math.min(10000, newEpisode * 100));
        
        // 模拟目标网络更新
        if (newEpisode % 10 === 0) {
          setTargetNetworkUpdates(prev => prev + 1);
        }

        if (newEpisode >= 500) {
          setIsTraining(false);
          clearInterval(interval);
        }

        return newEpisode;
      });
    }, 50);
  };

  const lossConfig = {
    data: lossData,
    xField: 'episode',
    yField: 'loss',
    smooth: true,
    color: '#ff4d4f',
    point: { size: 2 },
    line: { size: 2 },
  };

  const rewardConfig = {
    data: rewardData,
    xField: 'episode',
    yField: 'reward',
    smooth: true,
    color: '#52c41a',
    point: { size: 2 },
    line: { size: 2 },
  };

  return (
    <div style={{ padding: '24px', background: '#f0f2f5', minHeight: '100vh' }}>
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        
        {/* 页面标题 */}
        <Card>
          <Space align="center">
            <RobotOutlined style={{ fontSize: '24px', color: '#1890ff' }} />
            <Title level={2} style={{ margin: 0 }}>Deep Q-Network (DQN)</Title>
          </Space>
          <Text type="secondary">
            使用深度神经网络逼近Q函数的强化学习算法，能处理大规模和连续状态空间
          </Text>
        </Card>

        <Row gutter={[16, 16]}>
          
          {/* 网络配置 */}
          <Col span={8}>
            <Card title="网络架构配置" extra={<BranchesOutlined />}>
              <Space direction="vertical" style={{ width: '100%' }}>
                
                <div>
                  <Text strong>网络类型:</Text>
                  <Select 
                    value={networkType} 
                    onChange={setNetworkType}
                    style={{ width: '100%' }}
                    disabled={isTraining}
                  >
                    <Option value="dqn">标准 DQN</Option>
                    <Option value="double_dqn">Double DQN</Option>
                    <Option value="dueling_dqn">Dueling DQN</Option>
                  </Select>
                </div>

                <Alert 
                  message="当前网络结构" 
                  description={
                    <div>
                      <Text>输入层: 84 x 84 x 4 (状态)</Text><br/>
                      <Text>卷积层1: 32 filters, 8x8, stride=4</Text><br/>
                      <Text>卷积层2: 64 filters, 4x4, stride=2</Text><br/>
                      <Text>卷积层3: 64 filters, 3x3, stride=1</Text><br/>
                      <Text>全连接层: 512 units</Text><br/>
                      <Text>输出层: 4 actions</Text>
                    </div>
                  }
                  variant="default"
                />

                <Descriptions column={1} size="small">
                  <Descriptions.Item label="总参数数量">1,686,548</Descriptions.Item>
                  <Descriptions.Item label="可训练参数">1,686,548</Descriptions.Item>
                  <Descriptions.Item label="优化器">Adam</Descriptions.Item>
                  <Descriptions.Item label="学习率">0.00025</Descriptions.Item>
                </Descriptions>

              </Space>
            </Card>
          </Col>

          {/* 训练状态 */}
          <Col span={8}>
            <Card title="训练状态监控">
              <Space direction="vertical" style={{ width: '100%' }}>
                
                <Statistic 
                  title="训练回合" 
                  value={episode} 
                  suffix="/ 500"
                />
                
                <Progress 
                  percent={(episode / 500) * 100} 
                  status={isTraining ? 'active' : 'normal'}
                />

                <Row gutter={16}>
                  <Col span={12}>
                    <Statistic 
                      title="经验回放缓冲区" 
                      value={replayBufferSize}
                      suffix="/ 10000"
                    />
                  </Col>
                  <Col span={12}>
                    <Statistic 
                      title="目标网络更新" 
                      value={targetNetworkUpdates}
                    />
                  </Col>
                </Row>

                <Space>
                  <Button 
                    type="primary" 
                    icon={<PlayCircleOutlined />}
                    onClick={() => setIsTraining(!isTraining)}
                    disabled={episode >= 500}
                  >
                    {isTraining ? '暂停训练' : '开始训练'}
                  </Button>
                  
                  <Button 
                    icon={<DatabaseOutlined />}
                    onClick={() => {
                      setEpisode(0);
                      setLossData([]);
                      setRewardData([]);
                      setReplayBufferSize(0);
                      setTargetNetworkUpdates(0);
                    }}
                    disabled={isTraining}
                  >
                    重置
                  </Button>
                </Space>

              </Space>
            </Card>
          </Col>

          {/* DQN核心技术 */}
          <Col span={8}>
            <Card title="DQN核心技术" extra={<ThunderboltOutlined />}>
              <Space direction="vertical" style={{ width: '100%' }}>
                
                <Tag color="blue">经验回放 (Experience Replay)</Tag>
                <Text type="secondary">打破数据相关性，提高样本利用率</Text>
                
                <Tag color="green">目标网络 (Target Network)</Tag>
                <Text type="secondary">稳定训练过程，减少目标值波动</Text>
                
                <Tag color="orange">卷积神经网络 (CNN)</Tag>
                <Text type="secondary">自动提取空间特征，处理原始像素输入</Text>

                <Divider />

                <Alert 
                  message="DQN损失函数" 
                  description="L = E[(r + γ max Q_target(s',a') - Q(s,a))²]"
                  type="success"
                />

              </Space>
            </Card>
          </Col>
        </Row>

        {/* 训练曲线可视化 */}
        <Row gutter={[16, 16]}>
          <Col span={12}>
            <Card title="损失函数曲线" extra={<LineChartOutlined />}>
              {lossData.length > 0 ? (
                <Line {...lossConfig} height={300} />
              ) : (
                <div style={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Text type="secondary">开始训练以查看损失曲线</Text>
                </div>
              )}
            </Card>
          </Col>
          
          <Col span={12}>
            <Card title="奖励收敛曲线">
              {rewardData.length > 0 ? (
                <Line {...rewardConfig} height={300} />
              ) : (
                <div style={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Text type="secondary">开始训练以查看奖励曲线</Text>
                </div>
              )}
            </Card>
          </Col>
        </Row>

        {/* DQN详细原理 */}
        <Card title="DQN算法原理详解">
          <Tabs defaultActiveKey="1">
            
            <TabPane tab="经验回放机制" key="1">
              <Row gutter={[16, 16]}>
                <Col span={12}>
                  <Space direction="vertical">
                    <Title level={4}>经验回放 (Experience Replay)</Title>
                    <Text>
                      传统的在线学习存在样本相关性问题，DQN引入经验回放来解决：
                    </Text>
                    <ul>
                      <li><strong>存储经验</strong>: 将 (s, a, r, s') 四元组存入回放缓冲区</li>
                      <li><strong>随机采样</strong>: 从缓冲区随机抽取batch进行训练</li>
                      <li><strong>打破相关性</strong>: 避免连续样本的时间相关性</li>
                      <li><strong>提高效率</strong>: 每个经验可以被多次使用</li>
                    </ul>
                  </Space>
                </Col>
                <Col span={12}>
                  <Alert 
                    message="回放缓冲区结构"
                    description={
                      <div>
                        <Text>容量: 1,000,000 transitions</Text><br/>
                        <Text>采样大小: 32 transitions</Text><br/>
                        <Text>存储格式: (state, action, reward, next_state, done)</Text><br/>
                        <Text>采样策略: 均匀随机采样</Text>
                      </div>
                    }
                    variant="default"
                  />
                </Col>
              </Row>
            </TabPane>

            <TabPane tab="目标网络稳定化" key="2">
              <Row gutter={[16, 16]}>
                <Col span={12}>
                  <Space direction="vertical">
                    <Title level={4}>目标网络 (Target Network)</Title>
                    <Text>
                      DQN使用两个网络来稳定训练过程：
                    </Text>
                    <ul>
                      <li><strong>主网络</strong>: 实时更新的Q网络，用于动作选择</li>
                      <li><strong>目标网络</strong>: 延迟更新的网络，用于计算目标Q值</li>
                      <li><strong>定期同步</strong>: 每C步将主网络参数复制到目标网络</li>
                      <li><strong>稳定训练</strong>: 避免目标值的快速变化</li>
                    </ul>
                  </Space>
                </Col>
                <Col span={12}>
                  <Alert 
                    message="目标网络更新策略"
                    description={
                      <div>
                        <Text>更新频率: 每10,000步</Text><br/>
                        <Text>更新方式: 硬更新 (完全复制)</Text><br/>
                        <Text>目标Q计算: Q_target(s', argmax_a Q(s',a))</Text>
                      </div>
                    }
                    variant="warning"
                  />
                </Col>
              </Row>
            </TabPane>

            <TabPane tab="网络架构设计" key="3">
              <Space direction="vertical" style={{ width: '100%' }}>
                <Title level={4}>卷积神经网络架构</Title>
                <Text>
                  DQN使用卷积神经网络直接从原始像素学习：
                </Text>
                
                <Row gutter={[16, 16]}>
                  <Col span={8}>
                    <Card size="small" title="输入层">
                      <Text>84 × 84 × 4</Text><br/>
                      <Text type="secondary">4帧灰度图像堆叠</Text>
                    </Card>
                  </Col>
                  <Col span={8}>
                    <Card size="small" title="卷积层">
                      <Text>Conv1: 32@8×8, stride=4</Text><br/>
                      <Text>Conv2: 64@4×4, stride=2</Text><br/>
                      <Text>Conv3: 64@3×3, stride=1</Text>
                    </Card>
                  </Col>
                  <Col span={8}>
                    <Card size="small" title="全连接层">
                      <Text>FC1: 512 units</Text><br/>
                      <Text>Output: 4 Q-values</Text><br/>
                      <Text type="secondary">对应4个动作</Text>
                    </Card>
                  </Col>
                </Row>
              </Space>
            </TabPane>

          </Tabs>
        </Card>

        {/* 算法对比 */}
        <Row gutter={[16, 16]}>
          <Col span={12}>
            <Card title="DQN vs 表格Q-Learning">
              <table style={{ width: '100%', fontSize: '14px' }}>
                <thead>
                  <tr style={{ background: '#fafafa' }}>
                    <th style={{ padding: '8px', border: '1px solid #d9d9d9' }}>特性</th>
                    <th style={{ padding: '8px', border: '1px solid #d9d9d9' }}>表格Q-Learning</th>
                    <th style={{ padding: '8px', border: '1px solid #d9d9d9' }}>DQN</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td style={{ padding: '8px', border: '1px solid #d9d9d9' }}>状态表示</td>
                    <td style={{ padding: '8px', border: '1px solid #d9d9d9' }}>离散索引</td>
                    <td style={{ padding: '8px', border: '1px solid #d9d9d9' }}>高维向量</td>
                  </tr>
                  <tr>
                    <td style={{ padding: '8px', border: '1px solid #d9d9d9' }}>函数逼近</td>
                    <td style={{ padding: '8px', border: '1px solid #d9d9d9' }}>查表</td>
                    <td style={{ padding: '8px', border: '1px solid #d9d9d9' }}>神经网络</td>
                  </tr>
                  <tr>
                    <td style={{ padding: '8px', border: '1px solid #d9d9d9' }}>泛化能力</td>
                    <td style={{ padding: '8px', border: '1px solid #d9d9d9' }}>无</td>
                    <td style={{ padding: '8px', border: '1px solid #d9d9d9' }}>强</td>
                  </tr>
                  <tr>
                    <td style={{ padding: '8px', border: '1px solid #d9d9d9' }}>状态空间规模</td>
                    <td style={{ padding: '8px', border: '1px solid #d9d9d9' }}>小规模</td>
                    <td style={{ padding: '8px', border: '1px solid #d9d9d9' }}>大规模</td>
                  </tr>
                </tbody>
              </table>
            </Card>
          </Col>

          <Col span={12}>
            <Card title="DQN的优势与局限">
              <Space direction="vertical">
                <div>
                  <Text strong style={{ color: '#52c41a' }}>优势:</Text>
                  <ul>
                    <li>能处理高维状态空间 (如图像)</li>
                    <li>具有良好的泛化能力</li>
                    <li>突破了传统RL的状态空间限制</li>
                    <li>在Atari游戏中表现优异</li>
                  </ul>
                </div>
                
                <div>
                  <Text strong style={{ color: '#ff4d4f' }}>局限:</Text>
                  <ul>
                    <li>训练不稳定，容易发散</li>
                    <li>过高估计Q值 (overestimation bias)</li>
                    <li>样本效率相对较低</li>
                    <li>对超参数敏感</li>
                  </ul>
                </div>
              </Space>
            </Card>
          </Col>
        </Row>

      </Space>
    </div>
  );
};

export default DQNPage;