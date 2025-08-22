import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { 
  Card, 
  Row, 
  Col, 
  Tabs, 
  Typography, 
  Space, 
  Button, 
  Select, 
  Alert, 
  Spin,
  Statistic,
  Progress,
  Tag
} from 'antd'
import {
  ExperimentOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  ReloadOutlined,
  LineChartOutlined,
  RobotOutlined,
  ThunderboltOutlined,
  DatabaseOutlined,
  DashboardOutlined
} from '@ant-design/icons'
import { QLearningAgentPanel } from '../components/qlearning/QLearningAgentPanel'
import { QLearningTrainingPanel } from '../components/qlearning/QLearningTrainingPanel'
import { QLearningVisualization } from '../components/qlearning/QLearningVisualization'
import { QLearningEnvironmentPanel } from '../components/qlearning/QLearningEnvironmentPanel'

const { Title, Text, Paragraph } = Typography
const { TabPane } = Tabs

interface AgentStats {
  agent_id: string
  algorithm_type: string
  episode_count: number
  step_count: number
  current_epsilon: number
  total_reward: number
  average_reward: number
  recent_rewards: number[]
}

interface TrainingSession {
  session_id: string
  agent_id: string
  status: 'idle' | 'running' | 'paused' | 'completed'
  total_episodes: number
  current_episode: number
  average_reward: number
  convergence_achieved: boolean
  start_time: string
  elapsed_time?: number
}

const QLearningPage: React.FC = () => {
  const navigate = useNavigate()
  const [activeTab, setActiveTab] = useState('overview')
  const [agents, setAgents] = useState<AgentStats[]>([])
  const [trainingSessions, setTrainingSessions] = useState<TrainingSession[]>([])
  const [loading, setLoading] = useState(false)
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null)

  // 模拟数据加载
  useEffect(() => {
    loadAgentData()
    loadTrainingData()
    
    // 模拟实时更新
    const interval = setInterval(() => {
      if (trainingSessions.some(s => s.status === 'running')) {
        updateTrainingProgress()
      }
    }, 2000)
    
    return () => clearInterval(interval)
  }, [])

  const loadAgentData = async () => {
    setLoading(true)
    try {
      // 模拟API调用
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      const mockAgents: AgentStats[] = [
        {
          agent_id: 'classic-q-001',
          algorithm_type: 'q_learning',
          episode_count: 150,
          step_count: 45000,
          current_epsilon: 0.15,
          total_reward: 2340.5,
          average_reward: 15.6,
          recent_rewards: [12.3, 18.9, 14.2, 16.8, 13.5, 19.1, 17.4, 15.9, 16.2, 14.7]
        },
        {
          agent_id: 'dqn-001',
          algorithm_type: 'dqn',
          episode_count: 85,
          step_count: 28500,
          current_epsilon: 0.08,
          total_reward: 1890.2,
          average_reward: 22.2,
          recent_rewards: [19.8, 25.3, 21.7, 24.1, 20.9, 26.4, 23.8, 22.5, 24.9, 21.2]
        },
        {
          agent_id: 'dueling-dqn-001',
          algorithm_type: 'dueling_dqn',
          episode_count: 42,
          step_count: 15600,
          current_epsilon: 0.12,
          total_reward: 1245.8,
          average_reward: 29.7,
          recent_rewards: [27.3, 32.1, 28.9, 31.5, 26.8, 33.2, 30.4, 29.1, 32.8, 28.7]
        }
      ]
      
      setAgents(mockAgents)
    } catch (error) {
      console.error('Failed to load agent data:', error)
    } finally {
      setLoading(false)
    }
  }

  const loadTrainingData = async () => {
    try {
      const mockSessions: TrainingSession[] = [
        {
          session_id: 'train-001',
          agent_id: 'classic-q-001',
          status: 'running',
          total_episodes: 200,
          current_episode: 150,
          average_reward: 15.6,
          convergence_achieved: false,
          start_time: '2025-08-19T10:30:00Z',
          elapsed_time: 1800
        },
        {
          session_id: 'train-002', 
          agent_id: 'dqn-001',
          status: 'completed',
          total_episodes: 100,
          current_episode: 100,
          average_reward: 22.2,
          convergence_achieved: true,
          start_time: '2025-08-19T09:15:00Z',
          elapsed_time: 3600
        }
      ]
      
      setTrainingSessions(mockSessions)
    } catch (error) {
      console.error('Failed to load training data:', error)
    }
  }

  const updateTrainingProgress = () => {
    setTrainingSessions(prev => prev.map(session => {
      if (session.status === 'running') {
        const newEpisode = Math.min(session.current_episode + 1, session.total_episodes)
        const newReward = session.average_reward + (Math.random() - 0.5) * 2
        
        return {
          ...session,
          current_episode: newEpisode,
          average_reward: Math.max(0, newReward),
          status: newEpisode >= session.total_episodes ? 'completed' : 'running',
          convergence_achieved: newReward > session.average_reward * 1.1,
          elapsed_time: (session.elapsed_time || 0) + 2
        }
      }
      return session
    }))

    // 同步更新agent数据
    setAgents(prev => prev.map(agent => {
      const session = trainingSessions.find(s => s.agent_id === agent.agent_id && s.status === 'running')
      if (session) {
        return {
          ...agent,
          episode_count: session.current_episode,
          average_reward: session.average_reward,
          current_epsilon: Math.max(0.01, agent.current_epsilon * 0.999)
        }
      }
      return agent
    }))
  }

  const createNewAgent = async (algorithmType: string) => {
    try {
      setLoading(true)
      // 模拟API调用
      await new Promise(resolve => setTimeout(resolve, 1500))
      
      const newAgent: AgentStats = {
        agent_id: `${algorithmType}-${Date.now()}`,
        algorithm_type: algorithmType,
        episode_count: 0,
        step_count: 0,
        current_epsilon: 1.0,
        total_reward: 0,
        average_reward: 0,
        recent_rewards: []
      }
      
      setAgents(prev => [...prev, newAgent])
    } catch (error) {
      console.error('Failed to create agent:', error)
    } finally {
      setLoading(false)
    }
  }

  const startTraining = async (agentId: string, episodes: number = 100) => {
    try {
      const newSession: TrainingSession = {
        session_id: `train-${Date.now()}`,
        agent_id: agentId,
        status: 'running',
        total_episodes: episodes,
        current_episode: 0,
        average_reward: 0,
        convergence_achieved: false,
        start_time: new Date().toISOString(),
        elapsed_time: 0
      }
      
      setTrainingSessions(prev => [...prev, newSession])
    } catch (error) {
      console.error('Failed to start training:', error)
    }
  }

  const getAlgorithmColor = (algorithm: string) => {
    switch (algorithm) {
      case 'q_learning': return 'blue'
      case 'dqn': return 'green'
      case 'double_dqn': return 'orange'
      case 'dueling_dqn': return 'purple'
      default: return 'default'
    }
  }

  const getAlgorithmName = (algorithm: string) => {
    switch (algorithm) {
      case 'q_learning': return 'Classic Q-Learning'
      case 'dqn': return 'Deep Q-Network'
      case 'double_dqn': return 'Double DQN'  
      case 'dueling_dqn': return 'Dueling DQN'
      default: return algorithm
    }
  }

  const runningSessionsCount = trainingSessions.filter(s => s.status === 'running').length
  const completedSessionsCount = trainingSessions.filter(s => s.status === 'completed').length
  const totalAgents = agents.length
  const avgPerformance = agents.length > 0 
    ? agents.reduce((sum, agent) => sum + agent.average_reward, 0) / agents.length 
    : 0

  return (
    <div style={{ padding: '24px', background: '#f5f5f5', minHeight: '100vh' }}>
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        {/* 页面头部 */}
        <Card>
          <Space align="center" style={{ width: '100%', justifyContent: 'space-between' }}>
            <Space align="center">
              <ExperimentOutlined style={{ fontSize: '24px', color: '#1890ff' }} />
              <div>
                <Title level={2} style={{ margin: 0 }}>Q-Learning策略优化系统</Title>
                <Text type="secondary">强化学习智能体训练与策略优化平台</Text>
              </div>
            </Space>
            
            <Space>
              <Button 
                type="primary" 
                icon={<PlayCircleOutlined />}
                onClick={() => setActiveTab('training')}
              >
                开始训练
              </Button>
              <Button 
                icon={<RobotOutlined />}
                onClick={() => setActiveTab('agents')}
              >
                管理智能体
              </Button>
              <Button icon={<ReloadOutlined />} onClick={loadAgentData} loading={loading}>
                刷新数据
              </Button>
            </Space>
          </Space>
        </Card>

        {/* 系统概览 */}
        <Row gutter={16}>
          <Col span={6}>
            <Card>
              <Statistic
                title="活跃智能体"
                value={totalAgents}
                prefix={<RobotOutlined style={{ color: '#1890ff' }} />}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="训练中会话"
                value={runningSessionsCount}
                prefix={<ThunderboltOutlined style={{ color: '#52c41a' }} />}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="已完成训练"
                value={completedSessionsCount}
                prefix={<LineChartOutlined style={{ color: '#722ed1' }} />}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="平均性能"
                value={avgPerformance}
                precision={1}
                prefix={<ExperimentOutlined style={{ color: '#faad14' }} />}
              />
            </Card>
          </Col>
        </Row>

        {/* 算法介绍 */}
        <Card title="算法技术栈" size="small">
          <Paragraph>
            本系统实现了完整的Q-Learning算法家族，从经典的表格Q-Learning到最新的深度强化学习算法：
          </Paragraph>
          <Space wrap>
            <Tag color="blue" icon={<ExperimentOutlined />}>Classic Q-Learning - 表格型算法，适用于离散小状态空间</Tag>
            <Tag color="green" icon={<ThunderboltOutlined />}>Deep Q-Network (DQN) - 神经网络逼近，处理高维状态</Tag>
            <Tag color="orange" icon={<LineChartOutlined />}>Double DQN - 减少Q值高估偏差</Tag>
            <Tag color="purple" icon={<RobotOutlined />}>Dueling DQN - 分离状态价值和优势函数</Tag>
          </Space>
        </Card>

        {/* 主要功能标签页 */}
        <Card>
          <Tabs activeKey={activeTab} onChange={setActiveTab}>
            <TabPane tab={
              <span>
                <ThunderboltOutlined />
                功能总览
              </span>
            } key="overview">
              <div style={{ padding: '20px 0' }}>
                <Row gutter={[24, 24]}>
                  
                  {/* Q-Learning算法家族 */}
                  <Col span={24}>
                    <Card title="Q-Learning算法家族" size="small">
                      <Row gutter={[16, 16]}>
                        <Col span={6}>
                          <Card 
                            hoverable 
                            size="small"
                            onClick={() => navigate('/qlearning/tabular')}
                            style={{ textAlign: 'center', cursor: 'pointer' }}
                          >
                            <DatabaseOutlined style={{ fontSize: '32px', color: '#1890ff', marginBottom: '8px' }} />
                            <div><strong>表格Q-Learning</strong></div>
                            <div style={{ fontSize: '12px', color: '#666' }}>经典表格式算法</div>
                          </Card>
                        </Col>
                        <Col span={6}>
                          <Card 
                            hoverable 
                            size="small"
                            onClick={() => navigate('/qlearning/dqn')}
                            style={{ textAlign: 'center', cursor: 'pointer' }}
                          >
                            <RobotOutlined style={{ fontSize: '32px', color: '#52c41a', marginBottom: '8px' }} />
                            <div><strong>Deep Q-Network</strong></div>
                            <div style={{ fontSize: '12px', color: '#666' }}>神经网络逼近</div>
                          </Card>
                        </Col>
                        <Col span={6}>
                          <Card 
                            hoverable 
                            size="small"
                            onClick={() => navigate('/qlearning/variants')}
                            style={{ textAlign: 'center', cursor: 'pointer' }}
                          >
                            <ExperimentOutlined style={{ fontSize: '32px', color: '#fa8c16', marginBottom: '8px' }} />
                            <div><strong>DQN变体</strong></div>
                            <div style={{ fontSize: '12px', color: '#666' }}>Double/Dueling DQN</div>
                          </Card>
                        </Col>
                      </Row>
                    </Card>
                  </Col>

                  {/* 探索策略系统 */}
                  <Col span={12}>
                    <Card title="探索策略系统" size="small">
                      <Row gutter={[12, 12]}>
                        <Col span={12}>
                          <Button block onClick={() => navigate('/exploration-strategies')}>
                            Epsilon-Greedy系列
                          </Button>
                        </Col>
                        <Col span={12}>
                          <Button block onClick={() => navigate('/ucb-strategies')}>
                            Upper Confidence Bound
                          </Button>
                        </Col>
                        <Col span={12}>
                          <Button block onClick={() => navigate('/thompson-sampling')}>
                            Thompson Sampling
                          </Button>
                        </Col>
                        <Col span={12}>
                          <Button block onClick={() => navigate('/adaptive-exploration')}>
                            自适应探索策略
                          </Button>
                        </Col>
                      </Row>
                    </Card>
                  </Col>

                  {/* 奖励函数系统 */}
                  <Col span={12}>
                    <Card title="奖励函数系统" size="small">
                      <Row gutter={[12, 12]}>
                        <Col span={12}>
                          <Button block onClick={() => navigate('/basic-rewards')}>
                            基础奖励函数
                          </Button>
                        </Col>
                        <Col span={12}>
                          <Button block onClick={() => navigate('/composite-rewards')}>
                            复合奖励系统
                          </Button>
                        </Col>
                        <Col span={12}>
                          <Button block onClick={() => navigate('/adaptive-rewards')}>
                            自适应奖励调整
                          </Button>
                        </Col>
                        <Col span={12}>
                          <Button block onClick={() => navigate('/reward-shaping')}>
                            奖励塑形技术
                          </Button>
                        </Col>
                      </Row>
                    </Card>
                  </Col>

                  {/* 环境建模系统 */}
                  <Col span={12}>
                    <Card title="环境建模系统" size="small">
                      <Row gutter={[12, 12]}>
                        <Col span={12}>
                          <Button block onClick={() => navigate('/state-space')}>
                            状态空间设计
                          </Button>
                        </Col>
                        <Col span={12}>
                          <Button block onClick={() => navigate('/action-space')}>
                            动作空间定义
                          </Button>
                        </Col>
                        <Col span={12}>
                          <Button block onClick={() => navigate('/environment-simulator')}>
                            环境模拟器
                          </Button>
                        </Col>
                        <Col span={12}>
                          <Button block onClick={() => navigate('/grid-world')}>
                            GridWorld环境
                          </Button>
                        </Col>
                      </Row>
                    </Card>
                  </Col>

                  {/* 训练管理系统 */}
                  <Col span={12}>
                    <Card title="训练管理系统" size="small">
                      <Row gutter={[12, 12]}>
                        <Col span={12}>
                          <Button block onClick={() => navigate('/training-manager')}>
                            训练调度管理
                          </Button>
                        </Col>
                        <Col span={12}>
                          <Button block onClick={() => navigate('/learning-rate-scheduler')}>
                            学习率调度器
                          </Button>
                        </Col>
                        <Col span={12}>
                          <Button block onClick={() => navigate('/early-stopping')}>
                            早停机制
                          </Button>
                        </Col>
                        <Col span={12}>
                          <Button block onClick={() => navigate('/performance-tracker')}>
                            性能追踪器
                          </Button>
                        </Col>
                      </Row>
                    </Card>
                  </Col>

                </Row>
              </div>
            </TabPane>
            
            <TabPane tab={
              <span>
                <RobotOutlined />
                智能体管理
              </span>
            } key="agents">
              <QLearningAgentPanel
                agents={agents}
                loading={loading}
                onCreateAgent={createNewAgent}
                onStartTraining={startTraining}
                onSelectAgent={setSelectedAgent}
                selectedAgent={selectedAgent}
              />
            </TabPane>
            
            <TabPane tab={
              <span>
                <PlayCircleOutlined />
                训练监控
              </span>
            } key="training">
              <QLearningTrainingPanel
                trainingSessions={trainingSessions}
                agents={agents}
                onStartTraining={startTraining}
                loading={loading}
              />
            </TabPane>
            
            <TabPane tab={
              <span>
                <LineChartOutlined />
                性能可视化
              </span>
            } key="visualization">
              <QLearningVisualization
                agents={agents}
                trainingSessions={trainingSessions}
                selectedAgent={selectedAgent}
              />
            </TabPane>
            
            <TabPane tab={
              <span>
                <ThunderboltOutlined />
                环境配置
              </span>
            } key="environment">
              <QLearningEnvironmentPanel />
            </TabPane>
          </Tabs>
        </Card>
      </Space>
    </div>
  )
}

export default QLearningPage