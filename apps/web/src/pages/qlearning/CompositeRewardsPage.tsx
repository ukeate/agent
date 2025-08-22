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
  Divider
} from 'antd';
import { Line, Column, Pie } from '@ant-design/charts';
import {
  NodeIndexOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  ReloadOutlined,
  CalculatorOutlined,
  FunctionOutlined,
  PartitionOutlined,
  BranchesOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;

const CompositeRewardsPage: React.FC = () => {
  const [isTraining, setIsTraining] = useState(false);
  const [episode, setEpisode] = useState(0);
  
  // 复合奖励函数权重
  const [taskWeight, setTaskWeight] = useState(0.4);
  const [explorationWeight, setExplorationWeight] = useState(0.3);
  const [efficiencyWeight, setEfficiencyWeight] = useState(0.2);
  const [safetyWeight, setSafetyWeight] = useState(0.1);
  
  // 子奖励函数参数
  const [taskReward, setTaskReward] = useState(100);
  const [explorationBonus, setExplorationBonus] = useState(10);
  const [efficiencyBonus, setEfficiencyBonus] = useState(5);
  const [safetyPenalty, setSafetyPenalty] = useState(-20);

  const [performanceData, setPerformanceData] = useState([
    { episode: 0, taskReward: 0, explorationReward: 0, efficiencyReward: 0, safetyReward: 0, totalReward: 0 },
    { episode: 10, taskReward: 20, explorationReward: 15, efficiencyReward: 8, safetyReward: -5, totalReward: 38 },
    { episode: 20, taskReward: 45, explorationReward: 25, efficiencyReward: 12, safetyReward: -2, totalReward: 80 },
    { episode: 30, taskReward: 70, explorationReward: 20, efficiencyReward: 18, safetyReward: 0, totalReward: 108 },
    { episode: 40, taskReward: 85, explorationReward: 15, efficiencyReward: 22, safetyReward: 2, totalReward: 124 },
  ]);

  const [weightHistory, setWeightHistory] = useState([
    { episode: 0, task: 0.4, exploration: 0.3, efficiency: 0.2, safety: 0.1 },
    { episode: 10, task: 0.45, exploration: 0.25, efficiency: 0.2, safety: 0.1 },
    { episode: 20, task: 0.5, exploration: 0.2, efficiency: 0.2, safety: 0.1 },
    { episode: 30, task: 0.55, exploration: 0.15, efficiency: 0.2, safety: 0.1 },
    { episode: 40, task: 0.6, exploration: 0.1, efficiency: 0.2, safety: 0.1 },
  ]);

  const rewardComponents = [
    { type: '任务完成奖励', value: taskWeight, color: '#1890ff' },
    { type: '探索奖励', value: explorationWeight, color: '#52c41a' },
    { type: '效率奖励', value: efficiencyWeight, color: '#faad14' },
    { type: '安全奖励', value: safetyWeight, color: '#f5222d' },
  ];

  const lineConfig = {
    data: performanceData,
    xField: 'episode',
    yField: 'totalReward',
    seriesField: 'type',
    smooth: true,
    point: {
      size: 5,
      shape: 'diamond',
    },
    tooltip: {
      showMarkers: false,
    },
    meta: {
      totalReward: {
        alias: '总奖励',
      },
    },
  };

  const pieConfig = {
    appendPadding: 10,
    data: rewardComponents,
    angleField: 'value',
    colorField: 'type',
    radius: 0.8,
    label: {
      type: 'outer',
      content: '{name} {percentage}',
    },
    interactions: [
      {
        type: 'element-active',
      },
    ],
  };

  const componentTableData = [
    {
      key: '1',
      component: '任务完成奖励',
      weight: taskWeight,
      baseValue: taskReward,
      currentValue: taskReward * taskWeight,
      description: '完成主要任务目标时获得的奖励',
    },
    {
      key: '2', 
      component: '探索奖励',
      weight: explorationWeight,
      baseValue: explorationBonus,
      currentValue: explorationBonus * explorationWeight,
      description: '访问新状态或未探索区域的奖励',
    },
    {
      key: '3',
      component: '效率奖励', 
      weight: efficiencyWeight,
      baseValue: efficiencyBonus,
      currentValue: efficiencyBonus * efficiencyWeight,
      description: '快速完成任务或优化路径的奖励',
    },
    {
      key: '4',
      component: '安全奖励',
      weight: safetyWeight, 
      baseValue: safetyPenalty,
      currentValue: safetyPenalty * safetyWeight,
      description: '避免危险状态或违规行为的奖励/惩罚',
    },
  ];

  const componentColumns = [
    {
      title: '奖励组件',
      dataIndex: 'component',
      key: 'component',
    },
    {
      title: '权重',
      dataIndex: 'weight',
      key: 'weight',
      render: (value: number) => <Tag color="blue">{value.toFixed(2)}</Tag>,
    },
    {
      title: '基础值',
      dataIndex: 'baseValue',
      key: 'baseValue',
      render: (value: number) => <Text>{value}</Text>,
    },
    {
      title: '加权值',
      dataIndex: 'currentValue',
      key: 'currentValue',
      render: (value: number) => <Text strong>{value.toFixed(2)}</Text>,
    },
    {
      title: '描述',
      dataIndex: 'description',
      key: 'description',
    },
  ];

  const handleWeightChange = (component: string, value: number) => {
    const newWeight = value / 100;
    switch (component) {
      case 'task':
        setTaskWeight(newWeight);
        break;
      case 'exploration':
        setExplorationWeight(newWeight);
        break;
      case 'efficiency':
        setEfficiencyWeight(newWeight);
        break;
      case 'safety':
        setSafetyWeight(newWeight);
        break;
    }
  };

  const startTraining = () => {
    setIsTraining(true);
    const interval = setInterval(() => {
      setEpisode(prev => {
        if (prev >= 50) {
          setIsTraining(false);
          clearInterval(interval);
          return prev;
        }
        return prev + 1;
      });
    }, 200);
  };

  const resetTraining = () => {
    setIsTraining(false);
    setEpisode(0);
  };

  const totalWeight = taskWeight + explorationWeight + efficiencyWeight + safetyWeight;

  return (
    <div style={{ padding: '24px', background: '#f0f2f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <NodeIndexOutlined style={{ marginRight: '8px', color: '#1890ff' }} />
          复合奖励系统
        </Title>
        <Text type="secondary">
          设计和优化多组件复合奖励函数，平衡不同目标间的权重关系
        </Text>
      </div>

      <Row gutter={[24, 24]}>
        {/* 权重配置卡片 */}
        <Col span={24}>
          <Card title="奖励组件权重配置" extra={
            <Space>
              <Text>总权重: </Text>
              <Tag color={totalWeight === 1.0 ? 'green' : 'orange'}>
                {totalWeight.toFixed(2)}
              </Tag>
            </Space>
          }>
            {totalWeight !== 1.0 && (
              <Alert
                message="权重总和建议为1.0"
                description="当前权重总和不等于1.0，可能影响奖励函数的稳定性"
                variant="warning"
                showIcon
                style={{ marginBottom: 16 }}
              />
            )}
            <Row gutter={[16, 16]}>
              <Col span={6}>
                <Text strong>任务完成权重: {taskWeight.toFixed(2)}</Text>
                <Slider
                  min={0}
                  max={100}
                  value={taskWeight * 100}
                  onChange={(value) => handleWeightChange('task', value)}
                  marks={{
                    0: '0%',
                    25: '25%',
                    50: '50%',
                    75: '75%',
                    100: '100%'
                  }}
                />
              </Col>
              <Col span={6}>
                <Text strong>探索权重: {explorationWeight.toFixed(2)}</Text>
                <Slider
                  min={0}
                  max={100}
                  value={explorationWeight * 100}
                  onChange={(value) => handleWeightChange('exploration', value)}
                  marks={{
                    0: '0%',
                    25: '25%',
                    50: '50%',
                    75: '75%',
                    100: '100%'
                  }}
                />
              </Col>
              <Col span={6}>
                <Text strong>效率权重: {efficiencyWeight.toFixed(2)}</Text>
                <Slider
                  min={0}
                  max={100}
                  value={efficiencyWeight * 100}
                  onChange={(value) => handleWeightChange('efficiency', value)}
                  marks={{
                    0: '0%',
                    25: '25%',
                    50: '50%',
                    75: '75%',
                    100: '100%'
                  }}
                />
              </Col>
              <Col span={6}>
                <Text strong>安全权重: {safetyWeight.toFixed(2)}</Text>
                <Slider
                  min={0}
                  max={100}
                  value={safetyWeight * 100}
                  onChange={(value) => handleWeightChange('safety', value)}
                  marks={{
                    0: '0%',
                    25: '25%',
                    50: '50%',
                    75: '75%',
                    100: '100%'
                  }}
                />
              </Col>
            </Row>
          </Card>
        </Col>

        {/* 训练控制和状态 */}
        <Col span={8}>
          <Card title="训练控制">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Statistic title="当前回合" value={episode} />
              <Progress percent={(episode / 50) * 100} status={isTraining ? 'active' : 'normal'} />
              <Space>
                <Button
                  type="primary"
                  icon={<PlayCircleOutlined />}
                  onClick={startTraining}
                  disabled={isTraining}
                >
                  开始训练
                </Button>
                <Button
                  icon={<ReloadOutlined />}
                  onClick={resetTraining}
                  disabled={isTraining}
                >
                  重置
                </Button>
              </Space>
            </Space>
          </Card>
        </Col>

        {/* 权重分布图 */}
        <Col span={8}>
          <Card title="权重分布">
            <div style={{ height: 250 }}>
              <Pie {...pieConfig} />
            </div>
          </Card>
        </Col>

        {/* 实时统计 */}
        <Col span={8}>
          <Card title="奖励统计">
            <Row gutter={[16, 16]}>
              <Col span={12}>
                <Statistic
                  title="任务奖励"
                  value={taskReward * taskWeight}
                  precision={2}
                  valueStyle={{ color: '#1890ff' }}
                />
              </Col>
              <Col span={12}>
                <Statistic
                  title="探索奖励"
                  value={explorationBonus * explorationWeight}
                  precision={2}
                  valueStyle={{ color: '#52c41a' }}
                />
              </Col>
              <Col span={12}>
                <Statistic
                  title="效率奖励"
                  value={efficiencyBonus * efficiencyWeight}
                  precision={2}
                  valueStyle={{ color: '#faad14' }}
                />
              </Col>
              <Col span={12}>
                <Statistic
                  title="安全奖励"
                  value={safetyPenalty * safetyWeight}
                  precision={2}
                  valueStyle={{ color: '#f5222d' }}
                />
              </Col>
            </Row>
          </Card>
        </Col>

        {/* 性能趋势图 */}
        <Col span={16}>
          <Card title="训练性能趋势">
            <div style={{ height: 300 }}>
              <Line {...lineConfig} />
            </div>
          </Card>
        </Col>

        {/* 奖励组件详情 */}
        <Col span={8}>
          <Card title="组件参数设置">
            <Space direction="vertical" style={{ width: '100%' }}>
              <div>
                <Text strong>任务完成奖励</Text>
                <Slider
                  min={50}
                  max={200}
                  value={taskReward}
                  onChange={setTaskReward}
                  marks={{ 50: '50', 100: '100', 150: '150', 200: '200' }}
                />
              </div>
              <div>
                <Text strong>探索奖励</Text>
                <Slider
                  min={1}
                  max={20}
                  value={explorationBonus}
                  onChange={setExplorationBonus}
                  marks={{ 1: '1', 5: '5', 10: '10', 15: '15', 20: '20' }}
                />
              </div>
              <div>
                <Text strong>效率奖励</Text>
                <Slider
                  min={1}
                  max={10}
                  value={efficiencyBonus}
                  onChange={setEfficiencyBonus}
                  marks={{ 1: '1', 3: '3', 5: '5', 8: '8', 10: '10' }}
                />
              </div>
              <div>
                <Text strong>安全惩罚</Text>
                <Slider
                  min={-50}
                  max={0}
                  value={safetyPenalty}
                  onChange={setSafetyPenalty}
                  marks={{ '-50': '-50', '-30': '-30', '-20': '-20', '-10': '-10', '0': '0' }}
                />
              </div>
            </Space>
          </Card>
        </Col>

        {/* 组件详情表格 */}
        <Col span={24}>
          <Card title="奖励组件详情">
            <Table
              dataSource={componentTableData}
              columns={componentColumns}
              pagination={false}
              size="middle"
            />
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default CompositeRewardsPage;