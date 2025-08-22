import React, { useState, useEffect } from 'react'
import { Card, Row, Col, Progress, Button, Space, Switch, Typography, Table, Tag, Statistic, Alert, Timeline, Slider } from 'antd'
import { 
  ExperimentOutlined,
  BranchesOutlined,
  RiseOutlined,
  BulbOutlined,
  SyncOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  ReloadOutlined,
  BarChartOutlined,
  LineChartOutlined,
  AimOutlined,
  RobotOutlined,
  ThunderboltOutlined,
  ClockCircleOutlined
} from '@ant-design/icons'
import { Line, Column, Gauge } from '@ant-design/plots'
import type { ColumnsType } from 'antd/es/table'

const { Title, Text, Paragraph } = Typography

interface LearningSession {
  id: string
  algorithm: string
  status: 'running' | 'paused' | 'completed' | 'failed'
  startTime: string
  duration: string
  iterations: number
  currentReward: number
  bestReward: number
  convergence: number
}

interface OnlineMetric {
  timestamp: string
  reward: number
  loss: number
  accuracy: number
  exploration_rate: number
}

const PersonalizationLearningPage: React.FC = () => {
  const [isLearning, setIsLearning] = useState(true)
  const [learningRate, setLearningRate] = useState(0.01)
  const [explorationRate, setExplorationRate] = useState(0.1)
  const [batchSize, setBatchSize] = useState(32)
  const [autoTuning, setAutoTuning] = useState(true)

  const [sessions, setSessions] = useState<LearningSession[]>([
    {
      id: '1',
      algorithm: 'Q-Learning',
      status: 'running',
      startTime: '2024-01-15 10:30:00',
      duration: '2h 15m',
      iterations: 15420,
      currentReward: 0.825,
      bestReward: 0.847,
      convergence: 0.89
    },
    {
      id: '2',
      algorithm: 'Thompson Sampling',
      status: 'completed',
      startTime: '2024-01-15 08:00:00',
      duration: '1h 45m',
      iterations: 12300,
      currentReward: 0.792,
      bestReward: 0.798,
      convergence: 0.95
    },
    {
      id: '3',
      algorithm: 'UCB',
      status: 'paused',
      startTime: '2024-01-15 09:15:00',
      duration: '3h 02m',
      iterations: 18750,
      currentReward: 0.734,
      bestReward: 0.756,
      convergence: 0.67
    }
  ])

  const [metrics, setMetrics] = useState<OnlineMetric[]>([])

  // 生成模拟学习指标数据
  useEffect(() => {
    const generateMetrics = () => {
      const newMetrics: OnlineMetric[] = []
      const now = Date.now()
      
      for (let i = 59; i >= 0; i--) {
        const timestamp = new Date(now - i * 60000).toLocaleTimeString()
        newMetrics.push({
          timestamp,
          reward: 0.5 + Math.random() * 0.4 + i * 0.002, // 递增趋势
          loss: 1.0 - i * 0.01 + Math.random() * 0.1, // 递减趋势
          accuracy: 0.6 + i * 0.005 + Math.random() * 0.1,
          exploration_rate: Math.max(0.01, explorationRate - i * 0.001)
        })
      }
      
      setMetrics(newMetrics)
    }

    generateMetrics()
    
    if (isLearning) {
      const interval = setInterval(generateMetrics, 5000)
      return () => clearInterval(interval)
    }
  }, [isLearning, explorationRate])

  // 模拟学习进度更新
  useEffect(() => {
    if (!isLearning) return

    const interval = setInterval(() => {
      setSessions(prev => prev.map(session => {
        if (session.status === 'running') {
          return {
            ...session,
            iterations: session.iterations + Math.floor(Math.random() * 100),
            currentReward: Math.min(1, session.currentReward + (Math.random() - 0.5) * 0.01),
            convergence: Math.min(1, session.convergence + Math.random() * 0.01)
          }
        }
        return session
      }))
    }, 2000)

    return () => clearInterval(interval)
  }, [isLearning])

  const sessionColumns: ColumnsType<LearningSession> = [
    {
      title: '算法',
      dataIndex: 'algorithm',
      key: 'algorithm',
      render: (text) => <Text strong>{text}</Text>
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status) => {
        const config = {
          running: { color: 'processing', text: '运行中' },
          paused: { color: 'warning', text: '已暂停' },
          completed: { color: 'success', text: '已完成' },
          failed: { color: 'error', text: '失败' }
        }
        return <Tag color={config[status].color}>{config[status].text}</Tag>
      }
    },
    {
      title: '迭代次数',
      dataIndex: 'iterations',
      key: 'iterations',
      render: (value) => value.toLocaleString()
    },
    {
      title: '当前奖励',
      dataIndex: 'currentReward',
      key: 'currentReward',
      render: (value) => (
        <Progress 
          percent={value * 100} 
          size="small" 
          format={(v) => `${(value).toFixed(3)}`}
        />
      )
    },
    {
      title: '最佳奖励',
      dataIndex: 'bestReward',
      key: 'bestReward',
      render: (value) => <Text type="success">{value.toFixed(3)}</Text>
    },
    {
      title: '收敛度',
      dataIndex: 'convergence',
      key: 'convergence',
      render: (value) => (
        <Progress 
          percent={value * 100} 
          size="small"
          strokeColor={{
            '0%': '#ff4d4f',
            '50%': '#faad14',
            '100%': '#52c41a',
          }}
        />
      )
    },
    {
      title: '持续时间',
      dataIndex: 'duration',
      key: 'duration',
      render: (text) => <Text type="secondary">{text}</Text>
    }
  ]

  // 奖励趋势配置
  const rewardConfig = {
    data: metrics,
    xField: 'timestamp',
    yField: 'reward',
    smooth: true,
    color: '#52c41a',
    point: {
      size: 3,
      shape: 'circle'
    },
    yAxis: {
      title: { text: '奖励值' },
      min: 0,
      max: 1
    }
  }

  // 损失趋势配置
  const lossConfig = {
    data: metrics,
    xField: 'timestamp',
    yField: 'loss',
    smooth: true,
    color: '#ff4d4f',
    yAxis: {
      title: { text: '损失值' }
    }
  }

  // 准确率仪表盘配置
  const accuracyGaugeConfig = {
    percent: metrics.length > 0 ? metrics[metrics.length - 1].accuracy : 0.75,
    range: {
      color: 'l(0) 0:#ff4d4f 0.5:#faad14 1:#52c41a',
    },
    statistic: {
      content: {
        formatter: ({ percent }: any) => `准确率 ${(percent * 100).toFixed(1)}%`,
      },
    },
  }

  const handleToggleLearning = () => {
    setIsLearning(!isLearning)
    
    // 更新会话状态
    setSessions(prev => prev.map(session => ({
      ...session,
      status: session.status === 'running' ? 'paused' : 
              session.status === 'paused' ? 'running' : session.status
    })))
  }

  const handleResetLearning = () => {
    setSessions(prev => prev.map(session => ({
      ...session,
      iterations: 0,
      currentReward: 0.5,
      convergence: 0,
      status: 'running'
    })))
    setMetrics([])
  }

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>
        <BranchesOutlined /> 在线学习管理
      </Title>
      <Paragraph type="secondary">
        实时监控和管理个性化引擎的在线学习过程，支持动态调参和策略优化
      </Paragraph>

      {/* 学习状态告警 */}
      <Alert
        message={
          <Space>
            {isLearning ? <SyncOutlined spin /> : <PauseCircleOutlined />}
            <Text>学习状态: {isLearning ? '正在学习' : '已暂停'}</Text>
            <Text type="secondary">| 活跃会话: {sessions.filter(s => s.status === 'running').length}</Text>
          </Space>
        }
        type={isLearning ? 'success' : 'warning'}
        style={{ marginBottom: 24 }}
        action={
          <Space>
            <Button 
              type="primary" 
              icon={isLearning ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
              onClick={handleToggleLearning}
            >
              {isLearning ? '暂停' : '继续'}
            </Button>
            <Button 
              icon={<ReloadOutlined />}
              onClick={handleResetLearning}
            >
              重置
            </Button>
          </Space>
        }
      />

      {/* 学习配置 */}
      <Card title="学习参数配置" style={{ marginBottom: 24 }}>
        <Row gutter={[24, 16]}>
          <Col span={6}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Text>学习率: {learningRate}</Text>
              <Slider
                min={0.001}
                max={0.1}
                step={0.001}
                value={learningRate}
                onChange={setLearningRate}
                marks={{
                  0.001: '0.001',
                  0.01: '0.01',
                  0.1: '0.1'
                }}
              />
            </Space>
          </Col>
          <Col span={6}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Text>探索率: {explorationRate}</Text>
              <Slider
                min={0.01}
                max={0.5}
                step={0.01}
                value={explorationRate}
                onChange={setExplorationRate}
                marks={{
                  0.01: '1%',
                  0.1: '10%',
                  0.5: '50%'
                }}
              />
            </Space>
          </Col>
          <Col span={6}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Text>批大小: {batchSize}</Text>
              <Slider
                min={8}
                max={128}
                step={8}
                value={batchSize}
                onChange={setBatchSize}
                marks={{
                  8: '8',
                  32: '32',
                  64: '64',
                  128: '128'
                }}
              />
            </Space>
          </Col>
          <Col span={6}>
            <Space direction="vertical">
              <Text>自动调参</Text>
              <Switch 
                checked={autoTuning}
                onChange={setAutoTuning}
                checkedChildren="开启"
                unCheckedChildren="关闭"
              />
            </Space>
          </Col>
        </Row>
      </Card>

      {/* 关键指标 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="平均奖励"
              value={metrics.length > 0 ? metrics[metrics.length - 1].reward : 0}
              precision={3}
              prefix={<RiseOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="总迭代次数"
              value={sessions.reduce((acc, s) => acc + s.iterations, 0)}
              prefix={<SyncOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="收敛度"
              value={sessions.reduce((acc, s) => acc + s.convergence, 0) / sessions.length * 100}
              suffix="%"
              precision={1}
              prefix={<AimOutlined />}
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="学习效率"
              value={85.2}
              suffix="%"
              prefix={<ThunderboltOutlined />}
              valueStyle={{ color: '#fa8c16' }}
            />
          </Card>
        </Col>
      </Row>

      {/* 学习趋势图表 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={8}>
          <Card title="奖励趋势">
            <Line {...rewardConfig} height={200} />
          </Card>
        </Col>
        <Col span={8}>
          <Card title="损失趋势">
            <Line {...lossConfig} height={200} />
          </Card>
        </Col>
        <Col span={8}>
          <Card title="当前准确率">
            <Gauge {...accuracyGaugeConfig} height={200} />
          </Card>
        </Col>
      </Row>

      {/* 学习会话表格 */}
      <Card title="学习会话" style={{ marginBottom: 24 }}>
        <Table 
          columns={sessionColumns}
          dataSource={sessions}
          pagination={false}
          size="middle"
        />
      </Card>

      {/* 学习历史时间轴 */}
      <Card title="学习历史">
        <Timeline>
          <Timeline.Item color="green" dot={<BulbOutlined />}>
            <Space direction="vertical" size="small">
              <Text strong>自适应学习率优化</Text>
              <Text type="secondary">10:45 - 学习率自动调整为 0.005，收敛速度提升 15%</Text>
            </Space>
          </Timeline.Item>
          <Timeline.Item color="blue" dot={<RobotOutlined />}>
            <Space direction="vertical" size="small">
              <Text strong>新算法启动</Text>
              <Text type="secondary">10:30 - 启动 Thompson Sampling 算法学习会话</Text>
            </Space>
          </Timeline.Item>
          <Timeline.Item color="orange" dot={<ExperimentOutlined />}>
            <Space direction="vertical" size="small">
              <Text strong>A/B测试完成</Text>
              <Text type="secondary">10:15 - Q-Learning vs UCB 对比测试结束，Q-Learning 胜出</Text>
            </Space>
          </Timeline.Item>
          <Timeline.Item dot={<ClockCircleOutlined />}>
            <Space direction="vertical" size="small">
              <Text strong>学习会话开始</Text>
              <Text type="secondary">10:00 - 启动新一轮在线学习，目标提升推荐准确率</Text>
            </Space>
          </Timeline.Item>
        </Timeline>
      </Card>
    </div>
  )
}

export default PersonalizationLearningPage