import React, { useState, useEffect } from 'react'
import { 
  Card, 
  Row, 
  Col, 
  Statistic, 
  Progress, 
  Table, 
  Button, 
  Space, 
  Tag, 
  Timeline,
  Alert,
  Select,
  Switch,
  Typography,
  Divider
} from 'antd'
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar
} from 'recharts'
import {
  PlayCircleOutlined,
  PauseOutlined,
  StopOutlined,
  ReloadOutlined,
  MonitorOutlined,
  TrophyOutlined,
  ExperimentOutlined,
  LineChartOutlined
} from '@ant-design/icons'

const { Title, Text } = Typography
const { Option } = Select

// 模拟训练数据
const generateTrainingData = () => {
  const data = []
  let reward = 0
  let loss = 1.0
  
  for (let i = 0; i < 100; i++) {
    reward += (Math.random() - 0.3) * 10
    loss = Math.max(0.01, loss * (0.98 + Math.random() * 0.04))
    
    data.push({
      episode: i + 1,
      reward: Math.max(-100, Math.min(100, reward)),
      loss: loss,
      explorationRate: Math.max(0.01, 1.0 - (i * 0.01)),
      qValue: Math.random() * 50,
      averageReward: reward / (i + 1)
    })
  }
  return data
}

const QLearningTrainingPage: React.FC = () => {
  const [isTraining, setIsTraining] = useState(false)
  const [isPaused, setIsPaused] = useState(false)
  const [currentEpisode, setCurrentEpisode] = useState(0)
  const [selectedAgent, setSelectedAgent] = useState('agent-1')
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [trainingData, setTrainingData] = useState(() => generateTrainingData())
  
  // 模拟训练状态
  useEffect(() => {
    let interval: NodeJS.Timeout | null = null
    
    if (isTraining && !isPaused && autoRefresh) {
      interval = setInterval(() => {
        setCurrentEpisode(prev => prev + 1)
        setTrainingData(generateTrainingData())
      }, 1000)
    }
    
    return () => {
      if (interval) clearInterval(interval)
    }
  }, [isTraining, isPaused, autoRefresh])

  const handleStartTraining = () => {
    setIsTraining(true)
    setIsPaused(false)
    setCurrentEpisode(0)
  }

  const handlePauseTraining = () => {
    setIsPaused(!isPaused)
  }

  const handleStopTraining = () => {
    setIsTraining(false)
    setIsPaused(false)
    setCurrentEpisode(0)
  }

  const handleResetData = () => {
    setTrainingData(generateTrainingData())
    setCurrentEpisode(0)
  }

  // 训练控制面板
  const TrainingControlPanel = () => (
    <Card title="训练控制面板" size="small">
      <Row gutter={16}>
        <Col span={12}>
          <Space direction="vertical" style={{ width: '100%' }}>
            <div>
              <Text strong>智能体选择:</Text>
              <Select 
                value={selectedAgent} 
                onChange={setSelectedAgent}
                style={{ width: '100%', marginTop: 8 }}
              >
                <Option value="agent-1">DQN Agent #1</Option>
                <Option value="agent-2">Double DQN Agent #2</Option>
                <Option value="agent-3">Dueling DQN Agent #3</Option>
                <Option value="agent-4">TabularQ Agent #4</Option>
              </Select>
            </div>
            <div>
              <Text strong>自动刷新:</Text>
              <Switch 
                checked={autoRefresh} 
                onChange={setAutoRefresh}
                style={{ marginLeft: 8 }}
              />
            </div>
          </Space>
        </Col>
        <Col span={12}>
          <Space wrap>
            {!isTraining ? (
              <Button 
                type="primary" 
                icon={<PlayCircleOutlined />}
                onClick={handleStartTraining}
              >
                开始训练
              </Button>
            ) : (
              <>
                <Button 
                  icon={isPaused ? <PlayCircleOutlined /> : <PauseOutlined />}
                  onClick={handlePauseTraining}
                >
                  {isPaused ? '继续' : '暂停'}
                </Button>
                <Button 
                  icon={<StopOutlined />}
                  onClick={handleStopTraining}
                  danger
                >
                  停止
                </Button>
              </>
            )}
            <Button 
              icon={<ReloadOutlined />}
              onClick={handleResetData}
            >
              重置数据
            </Button>
          </Space>
        </Col>
      </Row>
    </Card>
  )

  // 训练状态卡片
  const TrainingStatusCards = () => {
    const currentData = trainingData[trainingData.length - 1]
    
    return (
      <Row gutter={16}>
        <Col span={6}>
          <Card>
            <Statistic
              title="当前Episode"
              value={currentEpisode}
              prefix={<ExperimentOutlined />}
            />
            <Progress 
              percent={(currentEpisode % 100)} 
              size="small" 
              showInfo={false}
              style={{ marginTop: 8 }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="当前奖励"
              value={currentData?.reward || 0}
              precision={2}
              prefix={<TrophyOutlined />}
              valueStyle={{ 
                color: (currentData?.reward || 0) > 0 ? '#3f8600' : '#cf1322' 
              }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="平均奖励"
              value={currentData?.averageReward || 0}
              precision={2}
              prefix={<LineChartOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="探索率"
              value={(currentData?.explorationRate || 0) * 100}
              precision={1}
              suffix="%"
              prefix={<MonitorOutlined />}
            />
          </Card>
        </Col>
      </Row>
    )
  }

  // 训练状态指示器
  const TrainingStatusAlert = () => {
    if (!isTraining) {
      return (
        <Alert
          message="训练已停止"
          description="点击开始训练按钮启动Q-Learning智能体训练"
          variant="default"
          showIcon
        />
      )
    }
    
    if (isPaused) {
      return (
        <Alert
          message="训练已暂停"
          description="训练进程已暂停，点击继续按钮恢复训练"
          variant="warning"
          showIcon
        />
      )
    }
    
    return (
      <Alert
        message="训练进行中"
        description={`智能体 ${selectedAgent} 正在进行第 ${currentEpisode} 轮训练`}
        type="success"
        showIcon
      />
    )
  }

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>
        <ExperimentOutlined /> Q-Learning训练监控
      </Title>
      <Text type="secondary">
        实时监控Q-Learning智能体的训练过程，包括奖励变化、损失函数、探索策略等关键指标
      </Text>
      
      <Divider />

      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        {/* 训练状态提醒 */}
        <TrainingStatusAlert />

        {/* 训练控制面板 */}
        <TrainingControlPanel />

        {/* 训练状态卡片 */}
        <TrainingStatusCards />

        {/* 训练图表 */}
        <Row gutter={16}>
          <Col span={12}>
            <Card title="奖励曲线" size="small">
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={trainingData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="episode" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="reward" 
                    stroke="#1890ff" 
                    name="当前奖励"
                    dot={false}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="averageReward" 
                    stroke="#52c41a" 
                    name="平均奖励"
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          </Col>
          
          <Col span={12}>
            <Card title="损失函数" size="small">
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={trainingData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="episode" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Area
                    type="monotone"
                    dataKey="loss"
                    stroke="#ff4d4f"
                    fill="#ff4d4f"
                    fillOpacity={0.3}
                    name="损失值"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </Card>
          </Col>
        </Row>

        <Row gutter={16}>
          <Col span={12}>
            <Card title="探索率变化" size="small">
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={trainingData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="episode" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="explorationRate" 
                    stroke="#722ed1" 
                    name="探索率"
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          </Col>
          
          <Col span={12}>
            <Card title="Q值分布" size="small">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={trainingData.slice(-20)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="episode" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="qValue" fill="#fa8c16" name="平均Q值" />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </Col>
        </Row>

        {/* 训练日志时间线 */}
        <Card title="训练日志" size="small">
          <Timeline
            items={[
              {
                children: '智能体初始化完成',
                color: 'blue',
              },
              {
                children: `Episode ${Math.max(0, currentEpisode - 10)} - 开始探索阶段`,
                color: 'green',
              },
              {
                children: `Episode ${Math.max(0, currentEpisode - 5)} - 策略开始收敛`,
                color: 'yellow',
              },
              {
                children: isTraining ? 
                  `Episode ${currentEpisode} - 训练进行中${isPaused ? ' (已暂停)' : ''}` :
                  '训练已停止',
                color: isTraining ? (isPaused ? 'orange' : 'green') : 'red',
              },
            ]}
          />
        </Card>
      </Space>
    </div>
  )
}

export default QLearningTrainingPage