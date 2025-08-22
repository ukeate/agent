import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Row, 
  Col, 
  Typography, 
  Space, 
  Button, 
  Table, 
  InputNumber, 
  Slider,
  Alert,
  Progress,
  Statistic,
  Tag,
  Descriptions,
  Divider
} from 'antd';
import {
  DatabaseOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  ReloadOutlined,
  LineChartOutlined,
  SettingOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;

const TabularQLearningPage: React.FC = () => {
  const [qTable, setQTable] = useState<number[][]>([]);
  const [learningRate, setLearningRate] = useState(0.1);
  const [gamma, setGamma] = useState(0.99);
  const [epsilon, setEpsilon] = useState(0.1);
  const [episode, setEpisode] = useState(0);
  const [isTraining, setIsTraining] = useState(false);
  const [convergenceData, setConvergenceData] = useState<number[]>([]);

  useEffect(() => {
    // 初始化4x4的Q表
    initializeQTable();
  }, []);

  const initializeQTable = () => {
    const newQTable = Array(16).fill(null).map(() => Array(4).fill(0));
    setQTable(newQTable);
  };

  const handleStartTraining = () => {
    setIsTraining(true);
    simulateTraining();
  };

  const simulateTraining = () => {
    const interval = setInterval(() => {
      setEpisode(prev => {
        const newEpisode = prev + 1;
        
        // 模拟Q值更新
        setQTable(prev => {
          const newTable = prev.map(row => 
            row.map(val => val + (Math.random() - 0.5) * 0.1)
          );
          return newTable;
        });

        // 模拟收敛数据
        setConvergenceData(prev => [...prev.slice(-49), Math.random() * 20 + 10]);

        if (newEpisode >= 100) {
          setIsTraining(false);
          clearInterval(interval);
        }

        return newEpisode;
      });
    }, 100);
  };

  const qTableColumns = [
    { title: '状态', dataIndex: 'state', key: 'state' },
    { title: '↑ (向上)', dataIndex: 'up', key: 'up', render: (val: number) => val.toFixed(3) },
    { title: '→ (向右)', dataIndex: 'right', key: 'right', render: (val: number) => val.toFixed(3) },
    { title: '↓ (向下)', dataIndex: 'down', key: 'down', render: (val: number) => val.toFixed(3) },
    { title: '← (向左)', dataIndex: 'left', key: 'left', render: (val: number) => val.toFixed(3) },
  ];

  const qTableData = qTable.map((row, index) => ({
    key: index,
    state: `S${index}`,
    up: row[0],
    right: row[1],
    down: row[2],
    left: row[3]
  }));

  return (
    <div style={{ padding: '24px', background: '#f0f2f5', minHeight: '100vh' }}>
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        
        {/* 页面标题 */}
        <Card>
          <Space align="center">
            <DatabaseOutlined style={{ fontSize: '24px', color: '#1890ff' }} />
            <Title level={2} style={{ margin: 0 }}>表格Q-Learning (Tabular Q-Learning)</Title>
          </Space>
          <Text type="secondary">
            经典的表格式Q-Learning算法，通过Q表存储每个状态-动作对的Q值，适用于小规模离散状态空间
          </Text>
        </Card>

        <Row gutter={[16, 16]}>
          
          {/* 算法参数配置 */}
          <Col span={8}>
            <Card title="算法参数配置" extra={<SettingOutlined />}>
              <Space direction="vertical" style={{ width: '100%' }}>
                
                <div>
                  <Text strong>学习率 (Learning Rate): {learningRate}</Text>
                  <Slider 
                    min={0.01} 
                    max={1} 
                    step={0.01} 
                    value={learningRate}
                    onChange={setLearningRate}
                    disabled={isTraining}
                  />
                  <Text type="secondary">控制新信息对已有知识的影响程度</Text>
                </div>

                <div>
                  <Text strong>折扣因子 (Gamma): {gamma}</Text>
                  <Slider 
                    min={0} 
                    max={1} 
                    step={0.01} 
                    value={gamma}
                    onChange={setGamma}
                    disabled={isTraining}
                  />
                  <Text type="secondary">未来奖励的重要性权重</Text>
                </div>

                <div>
                  <Text strong>探索率 (Epsilon): {epsilon}</Text>
                  <Slider 
                    min={0} 
                    max={1} 
                    step={0.01} 
                    value={epsilon}
                    onChange={setEpsilon}
                    disabled={isTraining}
                  />
                  <Text type="secondary">随机探索vs贪婪利用的平衡</Text>
                </div>

              </Space>
            </Card>
          </Col>

          {/* 训练状态监控 */}
          <Col span={8}>
            <Card title="训练状态监控">
              <Space direction="vertical" style={{ width: '100%' }}>
                
                <Statistic 
                  title="训练回合" 
                  value={episode} 
                  suffix="/ 100"
                />
                
                <Progress 
                  percent={(episode / 100) * 100} 
                  status={isTraining ? 'active' : 'normal'}
                />

                <Space>
                  <Button 
                    type="primary" 
                    icon={<PlayCircleOutlined />}
                    onClick={handleStartTraining}
                    disabled={isTraining}
                  >
                    开始训练
                  </Button>
                  
                  <Button 
                    icon={<ReloadOutlined />}
                    onClick={() => {
                      setEpisode(0);
                      setConvergenceData([]);
                      initializeQTable();
                    }}
                    disabled={isTraining}
                  >
                    重置
                  </Button>
                </Space>

                {isTraining && (
                  <Alert 
                    message="正在训练中..." 
                    description="智能体正在学习最优策略，Q值实时更新中"
                    variant="default" 
                    showIcon 
                  />
                )}

              </Space>
            </Card>
          </Col>

          {/* Q-Learning核心原理 */}
          <Col span={8}>
            <Card title="核心算法原理">
              <Space direction="vertical" style={{ width: '100%' }}>
                
                <Alert 
                  message="Q-Learning更新公式" 
                  description="Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]"
                  type="success"
                />

                <Descriptions column={1} size="small">
                  <Descriptions.Item label="Q(s,a)">当前状态s下执行动作a的Q值</Descriptions.Item>
                  <Descriptions.Item label="α (alpha)">学习率，控制更新幅度</Descriptions.Item>
                  <Descriptions.Item label="r">即时奖励</Descriptions.Item>
                  <Descriptions.Item label="γ (gamma)">折扣因子，衡量未来奖励重要性</Descriptions.Item>
                  <Descriptions.Item label="max Q(s',a')">下一状态的最大Q值</Descriptions.Item>
                </Descriptions>

                <Tag color="blue">Model-Free</Tag>
                <Tag color="green">Off-Policy</Tag>
                <Tag color="orange">Value-Based</Tag>

              </Space>
            </Card>
          </Col>
        </Row>

        {/* Q表可视化 */}
        <Card title="Q表可视化 (4x4 GridWorld)" extra={<DatabaseOutlined />}>
          <Alert 
            message="Q表说明" 
            description="表格显示每个状态下各动作的Q值。数值越大表示该动作在该状态下的价值越高。"
            variant="default"
            style={{ marginBottom: 16 }}
          />
          
          <Table 
            columns={qTableColumns}
            dataSource={qTableData}
            pagination={false}
            size="small"
            scroll={{ x: 600 }}
          />
        </Card>

        {/* 算法特点和适用场景 */}
        <Row gutter={[16, 16]}>
          <Col span={12}>
            <Card title="算法特点" extra={<LineChartOutlined />}>
              <Space direction="vertical">
                <Text><strong>优点:</strong></Text>
                <ul>
                  <li>原理简单，易于理解和实现</li>
                  <li>理论基础扎实，有收敛性保证</li>
                  <li>不需要环境模型，适用性广</li>
                  <li>能学习到最优策略</li>
                </ul>
                
                <Text><strong>缺点:</strong></Text>
                <ul>
                  <li>只适用于小规模离散状态空间</li>
                  <li>无法处理连续状态空间</li>
                  <li>状态空间大时收敛很慢</li>
                  <li>需要访问所有状态-动作对</li>
                </ul>
              </Space>
            </Card>
          </Col>

          <Col span={12}>
            <Card title="适用场景">
              <Space direction="vertical">
                <Text><strong>理想应用:</strong></Text>
                <ul>
                  <li>小规模网格世界 (GridWorld)</li>
                  <li>简单的游戏环境 (如Tic-Tac-Toe)</li>
                  <li>状态数量有限的决策问题</li>
                  <li>教学和算法原理演示</li>
                </ul>
                
                <Text><strong>不适用场景:</strong></Text>
                <ul>
                  <li>大规模状态空间 (如图像处理)</li>
                  <li>连续状态空间 (如机器人控制)</li>
                  <li>高维状态表示</li>
                  <li>实时性要求很高的应用</li>
                </ul>
              </Space>
            </Card>
          </Col>
        </Row>

      </Space>
    </div>
  );
};

export default TabularQLearningPage;