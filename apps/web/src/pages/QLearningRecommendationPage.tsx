import React, { useState, useEffect } from 'react'
import { 
  Card, 
  Row, 
  Col, 
  Button, 
  Space, 
  Table, 
  Select,
  Tag,
  Statistic,
  Alert,
  Progress,
  Typography,
  Divider,
  Form,
  Slider,
  Switch,
  Tooltip,
  Modal,
  Radio,
  Tabs
} from 'antd'
import { 
  PieChart, 
  Pie, 
  Cell, 
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Legend,
  BarChart,
  Bar,
  AreaChart,
  Area
} from 'recharts'
import {
  ExperimentOutlined,
  ThunderboltOutlined,
  SettingOutlined,
  TrophyOutlined,
  SyncOutlined,
  BarChartOutlined,
  PieChartOutlined,
  LineChartOutlined,
  BulbOutlined,
  ClockCircleOutlined,
  FireOutlined
} from '@ant-design/icons'

const { Title, Text, Paragraph } = Typography
const { Option } = Select
const { TabPane } = Tabs

// 策略组合模式
const COMBINATION_MODES = {
  WEIGHTED_AVERAGE: 'weighted_average',
  EPSILON_SWITCHING: 'epsilon_switching', 
  CONTEXTUAL_SELECTION: 'contextual_selection',
  HIERARCHICAL: 'hierarchical',
  ENSEMBLE_VOTING: 'ensemble_voting'
}

// 生成混合推荐数据
const generateHybridRecommendations = () => {
  const decisions = []
  const sources = ['q_learning', 'bandit', 'hybrid']
  const actions = ['推荐A', '推荐B', '推荐C', '推荐D']
  
  for (let i = 0; i < 100; i++) {
    const source = sources[Math.floor(Math.random() * sources.length)]
    const action = Math.floor(Math.random() * 4)
    
    decisions.push({
      id: i + 1,
      timestamp: new Date(Date.now() - (100 - i) * 60000),
      action: action,
      actionName: actions[action],
      decisionSource: source,
      confidenceScore: Math.random() * 100,
      qLearningScore: Math.random() * 100,
      banditScore: Math.random() * 100,
      combinationScore: Math.random() * 100,
      reward: Math.random() * 10 - 2,
      inferenceTime: Math.random() * 50 + 10,
      success: Math.random() > 0.3
    })
  }
  
  return decisions
}

// 生成策略性能数据
const generateStrategyPerformance = () => [
  {
    strategy: 'Q-Learning',
    totalDecisions: 3250,
    successfulDecisions: 2840,
    averageReward: 4.2,
    averageConfidence: 85.3,
    averageInferenceTime: 12.5
  },
  {
    strategy: 'Bandit',
    totalDecisions: 2180,
    successfulDecisions: 1950,
    averageReward: 3.8,
    averageConfidence: 78.9,
    averageInferenceTime: 8.3
  },
  {
    strategy: 'Hybrid',
    totalDecisions: 4420,
    successfulDecisions: 4020,
    averageReward: 5.1,
    averageConfidence: 89.7,
    averageInferenceTime: 15.2
  }
]

const QLearningRecommendationPage: React.FC = () => {
  const [combinationMode, setCombinationMode] = useState(COMBINATION_MODES.WEIGHTED_AVERAGE)
  const [qLearningWeight, setQLearningWeight] = useState(0.6)
  const [banditWeight, setBanditWeight] = useState(0.4)
  const [adaptiveWeights, setAdaptiveWeights] = useState(false)
  const [hybridData, setHybridData] = useState(() => generateHybridRecommendations())
  const [performanceData] = useState(() => generateStrategyPerformance())
  const [isRunning, setIsRunning] = useState(false)
  const [currentDecision, setCurrentDecision] = useState<any>(null)

  // 模拟实时推荐
  useEffect(() => {
    let interval: NodeJS.Timeout | null = null
    
    if (isRunning) {
      interval = setInterval(() => {
        const newDecision = generateHybridRecommendations().slice(-1)[0]
        setCurrentDecision(newDecision)
        setHybridData(prev => [...prev.slice(-99), newDecision])
      }, 2000)
    }
    
    return () => {
      if (interval) clearInterval(interval)
    }
  }, [isRunning])

  const handleStartRecommendation = () => {
    setIsRunning(true)
  }

  const handleStopRecommendation = () => {
    setIsRunning(false)
  }

  const handleWeightChange = (qWeight: number) => {
    setQLearningWeight(qWeight)
    setBanditWeight(1 - qWeight)
  }

  // 策略配置面板
  const StrategyConfigPanel = () => (
    <Card title="策略配置" size="small">
      <Space direction="vertical" style={{ width: '100%' }}>
        <div>
          <Text strong>组合模式:</Text>
          <Select 
            value={combinationMode} 
            onChange={setCombinationMode}
            style={{ width: '100%', marginTop: 8 }}
          >
            <Option value={COMBINATION_MODES.WEIGHTED_AVERAGE}>加权平均</Option>
            <Option value={COMBINATION_MODES.EPSILON_SWITCHING}>ε-切换</Option>
            <Option value={COMBINATION_MODES.CONTEXTUAL_SELECTION}>上下文选择</Option>
            <Option value={COMBINATION_MODES.HIERARCHICAL}>层次结构</Option>
            <Option value={COMBINATION_MODES.ENSEMBLE_VOTING}>集成投票</Option>
          </Select>
        </div>

        <div>
          <Text strong>权重配置:</Text>
          <div style={{ marginTop: 16 }}>
            <Text>Q-Learning权重: {qLearningWeight.toFixed(2)}</Text>
            <Slider
              min={0}
              max={1}
              step={0.1}
              value={qLearningWeight}
              onChange={handleWeightChange}
              disabled={adaptiveWeights}
            />
            <Text>Bandit权重: {banditWeight.toFixed(2)}</Text>
          </div>
        </div>

        <div>
          <Text strong>自适应权重:</Text>
          <Switch 
            checked={adaptiveWeights} 
            onChange={setAdaptiveWeights}
            style={{ marginLeft: 8 }}
          />
          <div style={{ marginTop: 8 }}>
            <Text type="secondary" style={{ fontSize: '12px' }}>
              {adaptiveWeights ? '根据历史性能自动调整权重' : '使用手动设置的固定权重'}
            </Text>
          </div>
        </div>

        <Space>
          <Button 
            type="primary" 
            icon={<ThunderboltOutlined />}
            onClick={handleStartRecommendation}
            disabled={isRunning}
          >
            开始推荐
          </Button>
          <Button 
            onClick={handleStopRecommendation}
            disabled={!isRunning}
          >
            停止推荐
          </Button>
        </Space>
      </Space>
    </Card>
  )

  // 实时决策展示
  const RealTimeDecision = () => (
    <Card 
      title={
        <Space>
          <SyncOutlined spin={isRunning} />
          实时决策
        </Space>
      } 
      size="small"
    >
      {currentDecision ? (
        <Space direction="vertical" style={{ width: '100%' }}>
          <Row gutter={16}>
            <Col span={6}>
              <Statistic
                title="推荐动作"
                value={currentDecision.actionName}
                prefix={<TrophyOutlined />}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="决策来源"
                value={currentDecision.decisionSource.toUpperCase()}
                prefix={<BulbOutlined />}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="置信度"
                value={currentDecision.confidenceScore.toFixed(1)}
                suffix="%"
                prefix={<BarChartOutlined />}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="推理时间"
                value={currentDecision.inferenceTime.toFixed(1)}
                suffix="ms"
                prefix={<ClockCircleOutlined />}
              />
            </Col>
          </Row>
          
          <div>
            <Text strong>策略评分:</Text>
            <Row gutter={8} style={{ marginTop: 8 }}>
              <Col span={8}>
                <Progress
                  percent={currentDecision.qLearningScore}
                  format={percent => `Q: ${percent?.toFixed(0)}`}
                  strokeColor="#1890ff"
                />
              </Col>
              <Col span={8}>
                <Progress
                  percent={currentDecision.banditScore}
                  format={percent => `B: ${percent?.toFixed(0)}`}
                  strokeColor="#52c41a"
                />
              </Col>
              <Col span={8}>
                <Progress
                  percent={currentDecision.combinationScore}
                  format={percent => `H: ${percent?.toFixed(0)}`}
                  strokeColor="#fa8c16"
                />
              </Col>
            </Row>
          </div>
        </Space>
      ) : (
        <Alert 
          message="等待决策数据" 
          description="点击开始推荐按钮启动混合推荐系统" 
          variant="default" 
        />
      )}
    </Card>
  )

  // 决策来源分布饼图
  const DecisionSourceDistribution = () => {
    const sourceCount = hybridData.reduce((acc, item) => {
      acc[item.decisionSource] = (acc[item.decisionSource] || 0) + 1
      return acc
    }, {} as Record<string, number>)

    const pieData = Object.entries(sourceCount).map(([source, count]) => ({
      name: source.toUpperCase(),
      value: count,
      percent: (count / hybridData.length * 100).toFixed(1)
    }))

    const COLORS = ['#1890ff', '#52c41a', '#fa8c16', '#722ed1']

    return (
      <Card title="决策来源分布" size="small">
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie
              data={pieData}
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={100}
              dataKey="value"
              label={({name, percent}) => `${name}: ${percent}%`}
            >
              {pieData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </Card>
    )
  }

  // 性能对比表
  const PerformanceTable = () => {
    const columns = [
      {
        title: '策略',
        dataIndex: 'strategy',
        key: 'strategy',
        render: (strategy: string) => <Tag color="blue">{strategy}</Tag>
      },
      {
        title: '总决策数',
        dataIndex: 'totalDecisions',
        key: 'totalDecisions',
      },
      {
        title: '成功率',
        key: 'successRate',
        render: (record: any) => {
          const rate = (record.successfulDecisions / record.totalDecisions * 100)
          return <Progress percent={rate} format={percent => `${percent?.toFixed(1)}%`} />
        }
      },
      {
        title: '平均奖励',
        dataIndex: 'averageReward',
        key: 'averageReward',
        render: (reward: number) => reward.toFixed(2)
      },
      {
        title: '平均置信度',
        dataIndex: 'averageConfidence',
        key: 'averageConfidence',
        render: (confidence: number) => `${confidence.toFixed(1)}%`
      },
      {
        title: '平均推理时间',
        dataIndex: 'averageInferenceTime',
        key: 'averageInferenceTime',
        render: (time: number) => `${time.toFixed(1)}ms`
      }
    ]

    return (
      <Card title="策略性能对比" size="small">
        <Table
          columns={columns}
          dataSource={performanceData}
          rowKey="strategy"
          size="small"
          pagination={false}
        />
      </Card>
    )
  }

  // 奖励趋势图
  const RewardTrendChart = () => {
    const trendData = hybridData.slice(-20).map((item, index) => ({
      index: index + 1,
      reward: item.reward,
      qLearningScore: item.qLearningScore,
      banditScore: item.banditScore,
      combinationScore: item.combinationScore
    }))

    return (
      <Card title="奖励趋势分析" size="small">
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={trendData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="index" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="reward" stroke="#ff4d4f" name="实际奖励" strokeWidth={2} />
            <Line type="monotone" dataKey="qLearningScore" stroke="#1890ff" name="Q-Learning评分" strokeWidth={1} />
            <Line type="monotone" dataKey="banditScore" stroke="#52c41a" name="Bandit评分" strokeWidth={1} />
            <Line type="monotone" dataKey="combinationScore" stroke="#fa8c16" name="混合评分" strokeWidth={1} />
          </LineChart>
        </ResponsiveContainer>
      </Card>
    )
  }

  // 决策历史表
  const DecisionHistoryTable = () => {
    const columns = [
      {
        title: 'ID',
        dataIndex: 'id',
        key: 'id',
        width: 60,
      },
      {
        title: '时间',
        dataIndex: 'timestamp',
        key: 'timestamp',
        render: (time: Date) => time.toLocaleTimeString(),
        width: 100,
      },
      {
        title: '推荐动作',
        dataIndex: 'actionName',
        key: 'actionName',
        render: (action: string) => <Tag>{action}</Tag>
      },
      {
        title: '决策来源',
        dataIndex: 'decisionSource',
        key: 'decisionSource',
        render: (source: string) => {
          const colors = {
            q_learning: 'blue',
            bandit: 'green', 
            hybrid: 'orange'
          }
          return <Tag color={(colors as any)[source]}>{source.toUpperCase()}</Tag>
        }
      },
      {
        title: '置信度',
        dataIndex: 'confidenceScore',
        key: 'confidenceScore',
        render: (score: number) => `${score.toFixed(1)}%`
      },
      {
        title: '奖励',
        dataIndex: 'reward',
        key: 'reward',
        render: (reward: number) => (
          <Text style={{ color: reward > 0 ? '#52c41a' : '#ff4d4f' }}>
            {reward.toFixed(2)}
          </Text>
        )
      },
      {
        title: '成功',
        dataIndex: 'success',
        key: 'success',
        render: (success: boolean) => (
          <Tag color={success ? 'green' : 'red'}>
            {success ? '成功' : '失败'}
          </Tag>
        )
      }
    ]

    return (
      <Card title="决策历史" size="small">
        <Table
          columns={columns}
          dataSource={hybridData.slice(-20)}
          rowKey="id"
          size="small"
          pagination={{ pageSize: 10 }}
          scroll={{ x: 800 }}
        />
      </Card>
    )
  }

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>
        <ExperimentOutlined /> Q-Learning混合推荐
      </Title>
      <Paragraph type="secondary">
        结合Q-Learning策略推理与多臂老虎机算法，实现智能化的混合推荐决策系统
      </Paragraph>
      
      <Divider />

      <Tabs defaultActiveKey="1">
        <TabPane tab="实时推荐" key="1">
          <Space direction="vertical" size="large" style={{ width: '100%' }}>
            <Row gutter={16}>
              <Col span={8}>
                <StrategyConfigPanel />
              </Col>
              <Col span={16}>
                <RealTimeDecision />
              </Col>
            </Row>

            <Row gutter={16}>
              <Col span={12}>
                <DecisionSourceDistribution />
              </Col>
              <Col span={12}>
                <RewardTrendChart />
              </Col>
            </Row>
          </Space>
        </TabPane>

        <TabPane tab="性能分析" key="2">
          <Space direction="vertical" size="large" style={{ width: '100%' }}>
            <PerformanceTable />
            
            <Row gutter={16}>
              <Col span={8}>
                <Card>
                  <Statistic
                    title="总决策数"
                    value={hybridData.length}
                    prefix={<FireOutlined />}
                  />
                </Card>
              </Col>
              <Col span={8}>
                <Card>
                  <Statistic
                    title="平均奖励"
                    value={hybridData.reduce((sum, item) => sum + item.reward, 0) / hybridData.length}
                    precision={2}
                    prefix={<TrophyOutlined />}
                    valueStyle={{ 
                      color: hybridData.reduce((sum, item) => sum + item.reward, 0) > 0 ? '#3f8600' : '#cf1322' 
                    }}
                  />
                </Card>
              </Col>
              <Col span={8}>
                <Card>
                  <Statistic
                    title="成功率"
                    value={hybridData.filter(item => item.success).length / hybridData.length * 100}
                    precision={1}
                    suffix="%"
                    prefix={<BarChartOutlined />}
                  />
                </Card>
              </Col>
            </Row>

            <Card title="策略权重演化" size="small">
              <ResponsiveContainer width="100%" height={200}>
                <AreaChart data={hybridData.slice(-20).map((_, i) => ({
                  index: i + 1,
                  qLearning: qLearningWeight * 100,
                  bandit: banditWeight * 100
                }))}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="index" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Area type="monotone" dataKey="qLearning" stackId="1" stroke="#1890ff" fill="#1890ff" name="Q-Learning权重%" />
                  <Area type="monotone" dataKey="bandit" stackId="1" stroke="#52c41a" fill="#52c41a" name="Bandit权重%" />
                </AreaChart>
              </ResponsiveContainer>
            </Card>
          </Space>
        </TabPane>

        <TabPane tab="决策历史" key="3">
          <DecisionHistoryTable />
        </TabPane>
      </Tabs>
    </div>
  )
}

export default QLearningRecommendationPage