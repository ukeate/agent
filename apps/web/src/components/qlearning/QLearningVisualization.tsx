import React, { useState, useEffect } from 'react'
import { 
  Card, 
  Row, 
  Col, 
  Select, 
  Space, 
  Tag,
  Typography,
  Alert
} from 'antd'
import {
  LineChartOutlined,
  BarChartOutlined,
  PieChartOutlined,
  HeatMapOutlined
} from '@ant-design/icons'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  ScatterChart,
  Scatter,
  Legend,
  Area,
  AreaChart
} from 'recharts'

const { Title, Text } = Typography
const { Option } = Select

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

interface QLearningVisualizationProps {
  agents: AgentStats[]
  trainingSessions: TrainingSession[]
  selectedAgent: string | null
}

const QLearningVisualization: React.FC<QLearningVisualizationProps> = ({
  agents,
  trainingSessions,
  selectedAgent
}) => {
  const [selectedAgentForChart, setSelectedAgentForChart] = useState<string | null>(selectedAgent)
  const [chartType, setChartType] = useState<string>('rewards')

  useEffect(() => {
    if (selectedAgent && !selectedAgentForChart) {
      setSelectedAgentForChart(selectedAgent)
    }
  }, [selectedAgent, selectedAgentForChart])

  const getAlgorithmName = (algorithm: string) => {
    switch (algorithm) {
      case 'q_learning': return 'Classic Q-Learning'
      case 'dqn': return 'Deep Q-Network'
      case 'double_dqn': return 'Double DQN'  
      case 'dueling_dqn': return 'Dueling DQN'
      default: return algorithm
    }
  }

  const getAlgorithmColor = (algorithm: string) => {
    switch (algorithm) {
      case 'q_learning': return '#1890ff'
      case 'dqn': return '#52c41a'
      case 'double_dqn': return '#faad14'
      case 'dueling_dqn': return '#722ed1'
      default: return '#d9d9d9'
    }
  }

  // 生成奖励趋势数据
  const generateRewardTrendData = (agent: AgentStats) => {
    const data = []
    const rewards = agent.recent_rewards || []
    
    for (let i = 0; i < rewards.length; i++) {
      data.push({
        episode: agent.episode_count - rewards.length + i + 1,
        reward: rewards[i],
        averageReward: rewards.slice(0, i + 1).reduce((sum, r) => sum + r, 0) / (i + 1)
      })
    }
    
    return data
  }

  // 算法性能对比数据
  const generateAlgorithmComparisonData = () => {
    return agents.map(agent => ({
      algorithm: getAlgorithmName(agent.algorithm_type),
      averageReward: agent.average_reward,
      episodes: agent.episode_count,
      epsilon: agent.current_epsilon,
      color: getAlgorithmColor(agent.algorithm_type)
    }))
  }

  // 探索率变化数据
  const generateEpsilonDecayData = () => {
    const selectedAgent = agents.find(a => a.agent_id === selectedAgentForChart)
    if (!selectedAgent) return []

    const data = []
    const currentEpsilon = selectedAgent.current_epsilon
    const episodes = selectedAgent.episode_count
    
    // 模拟epsilon衰减过程
    for (let i = 0; i <= episodes; i += Math.max(1, Math.floor(episodes / 20))) {
      const epsilon = Math.max(0.01, 1.0 * Math.pow(0.995, i))
      data.push({
        episode: i,
        epsilon: epsilon,
        explorationRate: epsilon * 100
      })
    }
    
    return data
  }

  // Q值分布数据（模拟）
  const generateQValueDistribution = () => {
    const data = []
    const actions = ['Action 1', 'Action 2', 'Action 3', 'Action 4', 'Action 5']
    
    actions.forEach(action => {
      data.push({
        action: action,
        qValue: Math.random() * 20 - 10,
        visits: Math.floor(Math.random() * 1000) + 100
      })
    })
    
    return data
  }

  const selectedAgentData = selectedAgentForChart 
    ? agents.find(a => a.agent_id === selectedAgentForChart) 
    : null

  const rewardTrendData = selectedAgentData ? generateRewardTrendData(selectedAgentData) : []
  const algorithmComparisonData = generateAlgorithmComparisonData()
  const epsilonDecayData = generateEpsilonDecayData()
  const qValueDistribution = generateQValueDistribution()

  const colors = ['#1890ff', '#52c41a', '#faad14', '#722ed1', '#13c2c2']

  return (
    <Row gutter={[16, 16]}>
      {/* 控制面板 */}
      <Col span={24}>
        <Card size="small">
          <Space style={{ width: '100%', justifyContent: 'space-between' }}>
            <Space>
              <LineChartOutlined style={{ fontSize: '16px', color: '#1890ff' }} />
              <Text strong>性能可视化分析</Text>
            </Space>
            
            <Space>
              <Text>选择智能体:</Text>
              <Select
                style={{ width: 200 }}
                placeholder="选择要分析的智能体"
                value={selectedAgentForChart}
                onChange={setSelectedAgentForChart}
                allowClear
              >
                {agents.map(agent => (
                  <Option key={agent.agent_id} value={agent.agent_id}>
                    <Space>
                      <code style={{ fontSize: '11px' }}>
                        {agent.agent_id.slice(-8)}
                      </code>
                      <Tag size="small" color={getAlgorithmColor(agent.algorithm_type)}>
                        {getAlgorithmName(agent.algorithm_type)}
                      </Tag>
                    </Space>
                  </Option>
                ))}
              </Select>
              
              <Text>图表类型:</Text>
              <Select value={chartType} onChange={setChartType} style={{ width: 120 }}>
                <Option value="rewards">奖励趋势</Option>
                <Option value="comparison">算法对比</Option>
                <Option value="epsilon">探索率</Option>
                <Option value="qvalues">Q值分布</Option>
              </Select>
            </Space>
          </Space>
        </Card>
      </Col>

      {/* 主要图表区域 */}
      <Col span={16}>
        <Card
          title={
            <Space>
              {chartType === 'rewards' && <LineChartOutlined />}
              {chartType === 'comparison' && <BarChartOutlined />}
              {chartType === 'epsilon' && <LineChartOutlined />}
              {chartType === 'qvalues' && <BarChartOutlined />}
              
              {chartType === 'rewards' && '奖励趋势分析'}
              {chartType === 'comparison' && '算法性能对比'}
              {chartType === 'epsilon' && '探索率衰减'}
              {chartType === 'qvalues' && 'Q值分布'}
            </Space>
          }
        >
          {chartType === 'rewards' && (
            selectedAgentData ? (
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={rewardTrendData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="episode" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="reward" 
                    stroke="#1890ff" 
                    strokeWidth={2}
                    dot={{ r: 3 }}
                    name="单轮奖励"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="averageReward" 
                    stroke="#52c41a" 
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    name="平均奖励"
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <Alert message="请选择一个智能体查看奖励趋势" variant="default" />
            )
          )}
          
          {chartType === 'comparison' && (
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={algorithmComparisonData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="algorithm" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="averageReward" name="平均奖励">
                  {algorithmComparisonData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
          
          {chartType === 'epsilon' && (
            <ResponsiveContainer width="100%" height={400}>
              <AreaChart data={epsilonDecayData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="episode" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Area
                  type="monotone"
                  dataKey="explorationRate"
                  stroke="#faad14"
                  fill="#faad14"
                  fillOpacity={0.3}
                  name="探索率 (%)"
                />
              </AreaChart>
            </ResponsiveContainer>
          )}
          
          {chartType === 'qvalues' && (
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={qValueDistribution}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="action" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="qValue" fill="#722ed1" name="Q值" />
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>
      </Col>

      {/* 侧边统计信息 */}
      <Col span={8}>
        <Space direction="vertical" style={{ width: '100%' }}>
          {/* 智能体详情 */}
          {selectedAgentData && (
            <Card title="智能体详情" size="small">
              <Space direction="vertical" style={{ width: '100%' }}>
                <div>
                  <Text type="secondary">智能体ID:</Text>
                  <br />
                  <code style={{ fontSize: '12px' }}>{selectedAgentData.agent_id}</code>
                </div>
                
                <div>
                  <Text type="secondary">算法类型:</Text>
                  <br />
                  <Tag color={getAlgorithmColor(selectedAgentData.algorithm_type)}>
                    {getAlgorithmName(selectedAgentData.algorithm_type)}
                  </Tag>
                </div>
                
                <Row gutter={8}>
                  <Col span={12}>
                    <Text type="secondary">训练轮数</Text>
                    <br />
                    <Text strong>{selectedAgentData.episode_count}</Text>
                  </Col>
                  <Col span={12}>
                    <Text type="secondary">总步数</Text>
                    <br />
                    <Text strong>{selectedAgentData.step_count.toLocaleString()}</Text>
                  </Col>
                </Row>
                
                <Row gutter={8}>
                  <Col span={12}>
                    <Text type="secondary">探索率</Text>
                    <br />
                    <Text strong>{(selectedAgentData.current_epsilon * 100).toFixed(1)}%</Text>
                  </Col>
                  <Col span={12}>
                    <Text type="secondary">平均奖励</Text>
                    <br />
                    <Text strong>{selectedAgentData.average_reward.toFixed(2)}</Text>
                  </Col>
                </Row>
              </Space>
            </Card>
          )}

          {/* 算法对比统计 */}
          <Card title="算法性能对比" size="small">
            <Space direction="vertical" style={{ width: '100%' }}>
              {algorithmComparisonData.map(data => (
                <Row key={data.algorithm} style={{ width: '100%' }} align="middle">
                  <Col span={12}>
                    <Tag size="small" color={data.color}>
                      {data.algorithm.replace(/^.* /, '')}
                    </Tag>
                  </Col>
                  <Col span={12} style={{ textAlign: 'right' }}>
                    <Text strong>{data.averageReward.toFixed(1)}</Text>
                  </Col>
                </Row>
              ))}
            </Space>
          </Card>

          {/* 训练状态统计 */}
          <Card title="训练状态统计" size="small">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Row>
                <Col span={12}>
                  <Text type="secondary">活跃智能体</Text>
                </Col>
                <Col span={12} style={{ textAlign: 'right' }}>
                  <Text strong>{agents.length}</Text>
                </Col>
              </Row>
              
              <Row>
                <Col span={12}>
                  <Text type="secondary">训练会话</Text>
                </Col>
                <Col span={12} style={{ textAlign: 'right' }}>
                  <Text strong>{trainingSessions.length}</Text>
                </Col>
              </Row>
              
              <Row>
                <Col span={12}>
                  <Text type="secondary">运行中</Text>
                </Col>
                <Col span={12} style={{ textAlign: 'right' }}>
                  <Text strong style={{ color: '#52c41a' }}>
                    {trainingSessions.filter(s => s.status === 'running').length}
                  </Text>
                </Col>
              </Row>
              
              <Row>
                <Col span={12}>
                  <Text type="secondary">已完成</Text>
                </Col>
                <Col span={12} style={{ textAlign: 'right' }}>
                  <Text strong style={{ color: '#1890ff' }}>
                    {trainingSessions.filter(s => s.status === 'completed').length}
                  </Text>
                </Col>
              </Row>
            </Space>
          </Card>
        </Space>
      </Col>
    </Row>
  )
}

export { QLearningVisualization }