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
  Descriptions,
  Tabs,
  Table
} from 'antd';
import { Line } from '@ant-design/charts';
import {
  SearchOutlined,
  LineChartOutlined,
  ExperimentOutlined,
  ControlOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;

const ExplorationStrategiesPage: React.FC = () => {
  const [strategy, setStrategy] = useState('epsilon_greedy');
  const [epsilon, setEpsilon] = useState(0.1);
  const [isTraining, setIsTraining] = useState(false);
  const [episode, setEpisode] = useState(0);
  const [explorationData, setExplorationData] = useState<any[]>([]);
  const [actionCounts, setActionCounts] = useState([0, 0, 0, 0]);

  useEffect(() => {
    if (isTraining) {
      simulateExploration();
    }
  }, [isTraining]);

  const simulateExploration = () => {
    const interval = setInterval(() => {
      setEpisode(prev => {
        const newEpisode = prev + 1;
        
        // 模拟探索率衰减
        let currentExploration = epsilon;
        if (strategy === 'decaying_epsilon') {
          currentExploration = Math.max(0.01, epsilon * Math.exp(-newEpisode / 1000));
        }
        
        setExplorationData(prev => [...prev.slice(-99), { 
          episode: newEpisode, 
          exploration: currentExploration,
          reward: Math.random() * 20 + (1 - currentExploration) * 30
        }]);
        
        // 模拟动作选择统计
        const selectedAction = Math.random() < currentExploration ? 
          Math.floor(Math.random() * 4) : 0; // 假设动作0是最优的
        
        setActionCounts(prev => {
          const newCounts = [...prev];
          newCounts[selectedAction]++;
          return newCounts;
        });

        if (newEpisode >= 1000) {
          setIsTraining(false);
          clearInterval(interval);
        }

        return newEpisode;
      });
    }, 10);
  };

  const explorationConfig = {
    data: explorationData,
    xField: 'episode',
    yField: 'exploration',
    smooth: true,
    color: '#1890ff',
    point: { size: 2 },
    line: { size: 2 },
  };

  const rewardConfig = {
    data: explorationData,
    xField: 'episode',
    yField: 'reward',
    smooth: true,
    color: '#52c41a',
    point: { size: 2 },
    line: { size: 2 },
  };

  const actionColumns = [
    { title: '动作', dataIndex: 'action', key: 'action' },
    { title: '选择次数', dataIndex: 'count', key: 'count' },
    { title: '选择比例', dataIndex: 'ratio', key: 'ratio', render: (val: number) => `${(val * 100).toFixed(1)}%` },
  ];

  const totalActions = actionCounts.reduce((sum, count) => sum + count, 0);
  const actionData = actionCounts.map((count, index) => ({
    key: index,
    action: `动作 ${index}`,
    count,
    ratio: totalActions > 0 ? count / totalActions : 0
  }));

  return (
    <div style={{ padding: '24px', background: '#f0f2f5', minHeight: '100vh' }}>
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        
        {/* 页面标题 */}
        <Card>
          <Space align="center">
            <SearchOutlined style={{ fontSize: '24px', color: '#1890ff' }} />
            <Title level={2} style={{ margin: 0 }}>探索策略系统 (Exploration Strategies)</Title>
          </Space>
          <Text type="secondary">
            平衡探索与利用的核心机制，决定智能体如何在未知环境中学习最优策略
          </Text>
        </Card>

        <Row gutter={[16, 16]}>
          
          {/* 策略配置 */}
          <Col span={8}>
            <Card title="探索策略配置" extra={<ControlOutlined />}>
              <Space direction="vertical" style={{ width: '100%' }}>
                
                <div>
                  <Text strong>策略类型:</Text>
                  <Select 
                    value={strategy} 
                    onChange={setStrategy}
                    style={{ width: '100%' }}
                    disabled={isTraining}
                  >
                    <Option value="epsilon_greedy">Epsilon-Greedy</Option>
                    <Option value="decaying_epsilon">衰减Epsilon-Greedy</Option>
                    <Option value="ucb">Upper Confidence Bound</Option>
                    <Option value="thompson">Thompson Sampling</Option>
                    <Option value="boltzmann">Boltzmann探索</Option>
                  </Select>
                </div>

                <div>
                  <Text strong>探索率 (ε): {epsilon}</Text>
                  <Slider 
                    min={0} 
                    max={1} 
                    step={0.01} 
                    value={epsilon}
                    onChange={setEpsilon}
                    disabled={isTraining}
                  />
                  <Text type="secondary">
                    {strategy === 'epsilon_greedy' ? '固定探索率' : 
                     strategy === 'decaying_epsilon' ? '初始探索率，会衰减' : 
                     '探索参数'}
                  </Text>
                </div>

                <Alert 
                  message={`当前策略: ${strategy === 'epsilon_greedy' ? 'Epsilon-Greedy' : 
                    strategy === 'decaying_epsilon' ? '衰减Epsilon-Greedy' :
                    strategy === 'ucb' ? 'Upper Confidence Bound' :
                    strategy === 'thompson' ? 'Thompson Sampling' : 'Boltzmann探索'}`}
                  description={
                    strategy === 'epsilon_greedy' ? '以ε概率随机探索，1-ε概率贪婪选择' :
                    strategy === 'decaying_epsilon' ? '探索率随训练过程衰减' :
                    strategy === 'ucb' ? '基于置信区间的探索策略' :
                    strategy === 'thompson' ? '基于贝叶斯后验分布采样' : 
                    '基于动作价值的softmax分布'
                  }
                  variant="default"
                />

              </Space>
            </Card>
          </Col>

          {/* 训练状态 */}
          <Col span={8}>
            <Card title="训练状态监控">
              <Space direction="vertical" style={{ width: '100%' }}>
                
                <Statistic 
                  title="训练步数" 
                  value={episode} 
                  suffix="/ 1000"
                />
                
                <Progress 
                  percent={(episode / 1000) * 100} 
                  status={isTraining ? 'active' : 'normal'}
                />

                <Row gutter={16}>
                  <Col span={12}>
                    <Statistic 
                      title="当前探索率" 
                      value={strategy === 'decaying_epsilon' && episode > 0 ? 
                        Math.max(0.01, epsilon * Math.exp(-episode / 1000)) : epsilon}
                      precision={3}
                    />
                  </Col>
                  <Col span={12}>
                    <Statistic 
                      title="总动作数" 
                      value={totalActions}
                    />
                  </Col>
                </Row>

                <Space>
                  <Button 
                    type="primary" 
                    icon={<PlayCircleOutlined />}
                    onClick={() => setIsTraining(!isTraining)}
                    disabled={episode >= 1000}
                  >
                    {isTraining ? '暂停' : '开始训练'}
                  </Button>
                  
                  <Button 
                    onClick={() => {
                      setEpisode(0);
                      setExplorationData([]);
                      setActionCounts([0, 0, 0, 0]);
                    }}
                    disabled={isTraining}
                  >
                    重置
                  </Button>
                </Space>

              </Space>
            </Card>
          </Col>

          {/* 探索vs利用平衡 */}
          <Col span={8}>
            <Card title="探索vs利用平衡">
              <Space direction="vertical" style={{ width: '100%' }}>
                
                <div>
                  <Text strong>探索 (Exploration):</Text>
                  <ul style={{ margin: '8px 0', paddingLeft: '20px' }}>
                    <li>尝试未知的动作</li>
                    <li>发现潜在的更优策略</li>
                    <li>避免陷入局部最优</li>
                  </ul>
                </div>

                <div>
                  <Text strong>利用 (Exploitation):</Text>
                  <ul style={{ margin: '8px 0', paddingLeft: '20px' }}>
                    <li>选择当前已知的最优动作</li>
                    <li>最大化即时奖励</li>
                    <li>基于已有经验做决策</li>
                  </ul>
                </div>

                <Alert 
                  message="平衡原则" 
                  description="有效的探索策略需要在探索新动作和利用已知最优动作之间找到平衡"
                  variant="warning"
                />

              </Space>
            </Card>
          </Col>
        </Row>

        {/* 策略效果可视化 */}
        <Row gutter={[16, 16]}>
          <Col span={12}>
            <Card title="探索率变化曲线" extra={<LineChartOutlined />}>
              {explorationData.length > 0 ? (
                <Line {...explorationConfig} height={300} />
              ) : (
                <div style={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Text type="secondary">开始训练以查看探索率变化</Text>
                </div>
              )}
            </Card>
          </Col>
          
          <Col span={12}>
            <Card title="奖励收敛情况">
              {explorationData.length > 0 ? (
                <Line {...rewardConfig} height={300} />
              ) : (
                <div style={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Text type="secondary">开始训练以查看奖励变化</Text>
                </div>
              )}
            </Card>
          </Col>
        </Row>

        {/* 动作选择统计 */}
        <Card title="动作选择统计" extra={<ExperimentOutlined />}>
          <Row gutter={[16, 16]}>
            <Col span={12}>
              <Table 
                columns={actionColumns}
                dataSource={actionData}
                pagination={false}
                size="small"
              />
            </Col>
            <Col span={12}>
              <Alert 
                message="动作选择分析" 
                description={
                  totalActions > 0 ? (
                    <div>
                      <Text>最常选择: 动作 {actionCounts.indexOf(Math.max(...actionCounts))}</Text><br/>
                      <Text>选择次数: {Math.max(...actionCounts)}</Text><br/>
                      <Text>探索比例: {((totalActions - Math.max(...actionCounts)) / totalActions * 100).toFixed(1)}%</Text>
                    </div>
                  ) : '开始训练以查看统计信息'
                }
                variant="default"
              />
            </Col>
          </Row>
        </Card>

        {/* 探索策略详解 */}
        <Card title="探索策略详细原理">
          <Tabs defaultActiveKey="1">
            
            <TabPane tab="Epsilon-Greedy" key="1">
              <Row gutter={[16, 16]}>
                <Col span={12}>
                  <Space direction="vertical">
                    <Title level={4}>Epsilon-Greedy策略</Title>
                    <Text>最简单也是最常用的探索策略：</Text>
                    <ul>
                      <li>以 ε 的概率随机选择动作 (探索)</li>
                      <li>以 1-ε 的概率选择当前最优动作 (利用)</li>
                      <li>参数简单，易于理解和实现</li>
                      <li>在很多环境中表现良好</li>
                    </ul>
                    
                    <Alert 
                      message="数学表达式" 
                      description="a = argmax Q(s,a) with prob. 1-ε; random action with prob. ε"
                      type="success"
                    />
                  </Space>
                </Col>
                <Col span={12}>
                  <Space direction="vertical">
                    <Title level={5}>参数设置建议:</Title>
                    <Descriptions column={1} size="small">
                      <Descriptions.Item label="初学阶段">ε = 0.5 ~ 1.0 (多探索)</Descriptions.Item>
                      <Descriptions.Item label="训练中期">ε = 0.1 ~ 0.3 (平衡)</Descriptions.Item>
                      <Descriptions.Item label="后期评估">ε = 0.01 ~ 0.05 (多利用)</Descriptions.Item>
                    </Descriptions>
                    
                    <Tag color="blue">优点: 简单有效</Tag>
                    <Tag color="orange">缺点: 探索无目标性</Tag>
                  </Space>
                </Col>
              </Row>
            </TabPane>

            <TabPane tab="衰减Epsilon-Greedy" key="2">
              <Row gutter={[16, 16]}>
                <Col span={12}>
                  <Space direction="vertical">
                    <Title level={4}>衰减Epsilon-Greedy策略</Title>
                    <Text>动态调整探索率的改进策略：</Text>
                    <ul>
                      <li>初期高探索率，充分探索环境</li>
                      <li>随训练进行逐渐降低探索率</li>
                      <li>后期低探索率，专注利用已学知识</li>
                      <li>符合学习的自然规律</li>
                    </ul>
                  </Space>
                </Col>
                <Col span={12}>
                  <Space direction="vertical">
                    <Title level={5}>常见衰减策略:</Title>
                    <ul>
                      <li><strong>指数衰减</strong>: ε(t) = ε₀ × e^(-λt)</li>
                      <li><strong>线性衰减</strong>: ε(t) = ε₀ × (1 - t/T)</li>
                      <li><strong>逆时间衰减</strong>: ε(t) = ε₀ / (1 + λt)</li>
                    </ul>
                    
                    <Tag color="green">优点: 适应学习过程</Tag>
                    <Tag color="red">缺点: 参数调节复杂</Tag>
                  </Space>
                </Col>
              </Row>
            </TabPane>

            <TabPane tab="Upper Confidence Bound" key="3">
              <Space direction="vertical" style={{ width: '100%' }}>
                <Title level={4}>Upper Confidence Bound (UCB)</Title>
                <Text>基于置信区间的智能探索策略：</Text>
                
                <Row gutter={[16, 16]}>
                  <Col span={12}>
                    <ul>
                      <li>考虑动作的不确定性</li>
                      <li>优先探索不确定性高的动作</li>
                      <li>理论基础扎实，有遗憾界保证</li>
                      <li>无需手动设置探索参数</li>
                    </ul>
                  </Col>
                  <Col span={12}>
                    <Alert 
                      message="UCB公式" 
                      description="UCB(a) = Q(a) + c × √(ln(t) / N(a))"
                      variant="default"
                    />
                    <Text type="secondary">其中 c 是探索常数，t 是总时间步，N(a) 是动作a的选择次数</Text>
                  </Col>
                </Row>
              </Space>
            </TabPane>

            <TabPane tab="Thompson Sampling" key="4">
              <Space direction="vertical" style={{ width: '100%' }}>
                <Title level={4}>Thompson Sampling</Title>
                <Text>基于贝叶斯推理的概率性探索策略：</Text>
                
                <Row gutter={[16, 16]}>
                  <Col span={12}>
                    <ul>
                      <li>维护每个动作价值的后验分布</li>
                      <li>从后验分布中采样来选择动作</li>
                      <li>自然地平衡探索与利用</li>
                      <li>在多臂赌博机问题中表现优异</li>
                    </ul>
                  </Col>
                  <Col span={12}>
                    <Space direction="vertical">
                      <Text strong>算法步骤:</Text>
                      <ol>
                        <li>为每个动作维护Beta分布 Beta(α, β)</li>
                        <li>从各动作的分布中采样得到估计值</li>
                        <li>选择估计值最大的动作</li>
                        <li>根据奖励更新对应的分布参数</li>
                      </ol>
                    </Space>
                  </Col>
                </Row>
              </Space>
            </TabPane>

          </Tabs>
        </Card>

      </Space>
    </div>
  );
};

export default ExplorationStrategiesPage;