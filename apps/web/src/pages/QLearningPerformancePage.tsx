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
  Tabs,
  Tooltip,
  Radio,
  DatePicker,
  List,
  Avatar
} from 'antd'
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Legend, 
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
  ComposedChart,
  Tooltip as RechartsTooltip
} from 'recharts'
import {
  MonitorOutlined,
  DashboardOutlined,
  ClockCircleOutlined,
  ThunderboltOutlined,
  TrophyOutlined,
  BarChartOutlined,
  LineChartOutlined,
  PieChartOutlined,
  FireOutlined,
  RocketOutlined,
  ExperimentOutlined,
  SettingOutlined,
  AlertOutlined
} from '@ant-design/icons'

const { Title, Text, Paragraph } = Typography
const { Option } = Select
const { TabPane } = Tabs
const { RangePicker } = DatePicker

// 生成性能指标数据
const generatePerformanceMetrics = () => {
  const metrics = []
  const agents = ['DQN-1', 'Double-DQN-2', 'Dueling-DQN-3', 'Tabular-Q-4']
  
  agents.forEach((agent, idx) => {
    const basePerformance = 0.7 + idx * 0.05
    
    for (let i = 0; i < 100; i++) {
      metrics.push({
        timestamp: new Date(Date.now() - (100 - i) * 3600000),
        agent,
        agentType: agent.split('-')[0],
        episode: i + 1,
        reward: (Math.random() - 0.3) * 20 + basePerformance * 50,
        loss: Math.max(0.01, Math.random() * 2),
        explorationRate: Math.max(0.01, 1.0 - (i * 0.01)),
        inferenceTime: Math.random() * 50 + 10,
        memoryUsage: Math.random() * 1000 + 200,
        cpuUsage: Math.random() * 80 + 10,
        throughput: Math.random() * 1000 + 500,
        accuracy: basePerformance + Math.random() * 0.2,
        cacheHitRate: Math.random() * 40 + 60
      })
    }
  })
  
  return metrics
}

// 生成系统资源使用数据
const generateSystemMetrics = () => {
  const data = []
  
  for (let i = 0; i < 50; i++) {
    data.push({
      time: new Date(Date.now() - (50 - i) * 60000),
      cpuUsage: Math.random() * 80 + 10,
      memoryUsage: Math.random() * 8000 + 1000,
      gpuUsage: Math.random() * 90 + 5,
      diskIO: Math.random() * 100,
      networkIO: Math.random() * 1000,
      inferenceQPS: Math.random() * 500 + 100
    })
  }
  
  return data
}

// 生成性能异常事件
const generatePerformanceAlerts = () => [
  {
    id: 1,
    timestamp: new Date(Date.now() - 3600000),
    level: 'warning',
    agent: 'DQN-1',
    message: '推理时间超过阈值 (>100ms)',
    metric: 'inference_time',
    value: 125.6,
    threshold: 100
  },
  {
    id: 2,
    timestamp: new Date(Date.now() - 7200000),
    level: 'error',
    agent: 'Double-DQN-2',
    message: 'CPU使用率过高 (>90%)',
    metric: 'cpu_usage',
    value: 94.3,
    threshold: 90
  },
  {
    id: 3,
    timestamp: new Date(Date.now() - 10800000),
    level: 'info',
    agent: 'Dueling-DQN-3',
    message: '缓存命中率提升',
    metric: 'cache_hit_rate',
    value: 89.5,
    threshold: 85
  }
]

const QLearningPerformancePage: React.FC = () => {
  const [selectedAgent, setSelectedAgent] = useState('all')
  const [selectedMetric, setSelectedMetric] = useState('reward')
  const [timeRange, setTimeRange] = useState('24h')
  const [performanceData] = useState(() => generatePerformanceMetrics())
  const [systemData] = useState(() => generateSystemMetrics())
  const [alerts] = useState(() => generatePerformanceAlerts())
  const [refreshing, setRefreshing] = useState(false)

  const handleRefresh = async () => {
    setRefreshing(true)
    // 模拟刷新延迟
    await new Promise(resolve => setTimeout(resolve, 1000))
    setRefreshing(false)
  }

  // 过滤数据
  const filteredData = performanceData.filter(item => 
    selectedAgent === 'all' || item.agent === selectedAgent
  )

  // 计算关键指标
  const calculateKPIs = () => {
    if (filteredData.length === 0) return {}
    
    const recentData = filteredData.slice(-20)
    
    return {
      avgReward: recentData.reduce((sum, item) => sum + item.reward, 0) / recentData.length,
      avgInferenceTime: recentData.reduce((sum, item) => sum + item.inferenceTime, 0) / recentData.length,
      avgAccuracy: recentData.reduce((sum, item) => sum + item.accuracy, 0) / recentData.length,
      avgThroughput: recentData.reduce((sum, item) => sum + item.throughput, 0) / recentData.length,
      totalEpisodes: Math.max(...filteredData.map(item => item.episode)),
      avgCacheHit: recentData.reduce((sum, item) => sum + item.cacheHitRate, 0) / recentData.length
    }
  }

  const kpis = calculateKPIs()

  // KPI卡片组
  const KPICards = () => (
    <Row gutter={16}>
      <Col span={4}>
        <Card>
          <Statistic
            title="平均奖励"
            value={kpis.avgReward || 0}
            precision={2}
            prefix={<TrophyOutlined />}
            valueStyle={{ 
              color: (kpis.avgReward || 0) > 0 ? '#3f8600' : '#cf1322' 
            }}
          />
        </Card>
      </Col>
      <Col span={4}>
        <Card>
          <Statistic
            title="推理时间"
            value={kpis.avgInferenceTime || 0}
            precision={1}
            suffix="ms"
            prefix={<ClockCircleOutlined />}
          />
        </Card>
      </Col>
      <Col span={4}>
        <Card>
          <Statistic
            title="准确率"
            value={(kpis.avgAccuracy || 0) * 100}
            precision={1}
            suffix="%"
            prefix={<BarChartOutlined />}
          />
        </Card>
      </Col>
      <Col span={4}>
        <Card>
          <Statistic
            title="吞吐量"
            value={kpis.avgThroughput || 0}
            precision={0}
            suffix="req/s"
            prefix={<ThunderboltOutlined />}
          />
        </Card>
      </Col>
      <Col span={4}>
        <Card>
          <Statistic
            title="训练Episodes"
            value={kpis.totalEpisodes || 0}
            prefix={<ExperimentOutlined />}
          />
        </Card>
      </Col>
      <Col span={4}>
        <Card>
          <Statistic
            title="缓存命中率"
            value={kpis.avgCacheHit || 0}
            precision={1}
            suffix="%"
            prefix={<RocketOutlined />}
          />
        </Card>
      </Col>
    </Row>
  )

  // 性能趋势图
  const PerformanceTrendChart = () => {
    const chartData = filteredData.slice(-50)
    
    return (
      <Card 
        title="性能趋势分析"
        size="small"
        extra={
          <Space>
            <Select value={selectedMetric} onChange={setSelectedMetric} size="small">
              <Option value="reward">奖励</Option>
              <Option value="loss">损失</Option>
              <Option value="accuracy">准确率</Option>
              <Option value="inferenceTime">推理时间</Option>
            </Select>
          </Space>
        }
      >
        <ResponsiveContainer width="100%" height={350}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="episode" />
            <YAxis />
            <RechartsTooltip />
            <Legend />
            <Line 
              type="monotone" 
              dataKey={selectedMetric} 
              stroke="#1890ff" 
              strokeWidth={2}
              dot={false}
              name={selectedMetric === 'reward' ? '奖励' : 
                   selectedMetric === 'loss' ? '损失' : 
                   selectedMetric === 'accuracy' ? '准确率' : '推理时间'}
            />
          </LineChart>
        </ResponsiveContainer>
      </Card>
    )
  }

  // 智能体对比图
  const AgentComparisonChart = () => {
    const agentStats = ['DQN-1', 'Double-DQN-2', 'Dueling-DQN-3', 'Tabular-Q-4'].map(agent => {
      const agentData = performanceData.filter(item => item.agent === agent).slice(-20)
      return {
        agent: agent,
        avgReward: agentData.reduce((sum, item) => sum + item.reward, 0) / agentData.length,
        avgAccuracy: agentData.reduce((sum, item) => sum + item.accuracy, 0) / agentData.length * 100,
        avgInferenceTime: agentData.reduce((sum, item) => sum + item.inferenceTime, 0) / agentData.length
      }
    })

    return (
      <Card title="智能体性能对比" size="small">
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={agentStats}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="agent" />
            <YAxis />
            <RechartsTooltip />
            <Legend />
            <Bar dataKey="avgReward" fill="#1890ff" name="平均奖励" />
            <Bar dataKey="avgAccuracy" fill="#52c41a" name="准确率(%)" />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    )
  }

  // 系统资源监控
  const SystemResourceChart = () => (
    <Card title="系统资源监控" size="small">
      <ResponsiveContainer width="100%" height={300}>
        <ComposedChart data={systemData.slice(-20)}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="time" tickFormatter={(time) => new Date(time).toLocaleTimeString()} />
          <YAxis yAxisId="left" />
          <YAxis yAxisId="right" orientation="right" />
          <RechartsTooltip />
          <Legend />
          <Area 
            yAxisId="left"
            type="monotone" 
            dataKey="cpuUsage" 
            fill="#ff7875" 
            stroke="#ff4d4f" 
            name="CPU使用率(%)"
            fillOpacity={0.3}
          />
          <Line 
            yAxisId="right"
            type="monotone" 
            dataKey="memoryUsage" 
            stroke="#1890ff" 
            name="内存使用(MB)"
          />
        </ComposedChart>
      </ResponsiveContainer>
    </Card>
  )

  // 推理性能散点图
  const InferencePerformanceScatter = () => {
    const scatterData = filteredData.slice(-100).map(item => ({
      inferenceTime: item.inferenceTime,
      accuracy: item.accuracy * 100,
      reward: item.reward,
      agent: item.agent
    }))

    return (
      <Card title="推理性能分析" size="small">
        <ResponsiveContainer width="100%" height={300}>
          <ScatterChart data={scatterData}>
            <CartesianGrid />
            <XAxis dataKey="inferenceTime" name="推理时间(ms)" />
            <YAxis dataKey="accuracy" name="准确率(%)" />
            <RechartsTooltip cursor={{ strokeDasharray: '3 3' }} />
            <Scatter name="性能点" dataKey="accuracy" fill="#8884d8" />
          </ScatterChart>
        </ResponsiveContainer>
      </Card>
    )
  }

  // 性能警报列表
  const PerformanceAlerts = () => (
    <Card title="性能警报" size="small">
      <List
        dataSource={alerts}
        renderItem={(item) => (
          <List.Item>
            <List.Item.Meta
              avatar={
                <Avatar 
                  icon={<AlertOutlined />} 
                  style={{ 
                    backgroundColor: 
                      item.level === 'error' ? '#ff4d4f' : 
                      item.level === 'warning' ? '#fa8c16' : '#52c41a'
                  }} 
                />
              }
              title={
                <Space>
                  <Tag color={item.level === 'error' ? 'red' : item.level === 'warning' ? 'orange' : 'green'}>
                    {item.level.toUpperCase()}
                  </Tag>
                  {item.agent}
                </Space>
              }
              description={
                <div>
                  <Text>{item.message}</Text>
                  <br />
                  <Text type="secondary">
                    值: {item.value} | 阈值: {item.threshold} | {item.timestamp.toLocaleString()}
                  </Text>
                </div>
              }
            />
          </List.Item>
        )}
      />
    </Card>
  )

  // 性能详细表格
  const PerformanceDetailTable = () => {
    const columns = [
      {
        title: '智能体',
        dataIndex: 'agent',
        key: 'agent',
        render: (agent: string) => <Tag color="blue">{agent}</Tag>
      },
      {
        title: 'Episode',
        dataIndex: 'episode',
        key: 'episode',
      },
      {
        title: '奖励',
        dataIndex: 'reward',
        key: 'reward',
        render: (reward: number) => (
          <Text style={{ color: reward > 0 ? '#52c41a' : '#ff4d4f' }}>
            {reward.toFixed(2)}
          </Text>
        ),
        sorter: (a: any, b: any) => a.reward - b.reward,
      },
      {
        title: '准确率',
        dataIndex: 'accuracy',
        key: 'accuracy',
        render: (accuracy: number) => `${(accuracy * 100).toFixed(1)}%`,
        sorter: (a: any, b: any) => a.accuracy - b.accuracy,
      },
      {
        title: '推理时间',
        dataIndex: 'inferenceTime',
        key: 'inferenceTime',
        render: (time: number) => `${time.toFixed(1)}ms`,
        sorter: (a: any, b: any) => a.inferenceTime - b.inferenceTime,
      },
      {
        title: 'CPU使用率',
        dataIndex: 'cpuUsage',
        key: 'cpuUsage',
        render: (usage: number) => (
          <Progress 
            percent={usage} 
            size="small" 
            status={usage > 80 ? 'exception' : 'normal'}
          />
        )
      },
      {
        title: '内存使用',
        dataIndex: 'memoryUsage',
        key: 'memoryUsage',
        render: (memory: number) => `${memory.toFixed(0)}MB`
      }
    ]

    return (
      <Card title="性能详细数据" size="small">
        <Table
          columns={columns}
          dataSource={filteredData.slice(-20)}
          rowKey={(record, index) => `${record.agent}-${record.episode}-${index}`}
          size="small"
          pagination={{ pageSize: 10 }}
          scroll={{ x: 1000 }}
        />
      </Card>
    )
  }

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>
        <MonitorOutlined /> Q-Learning性能分析
      </Title>
      <Paragraph type="secondary">
        全面分析Q-Learning智能体系统的性能表现，包括推理效率、资源使用、准确率等关键指标
      </Paragraph>
      
      <Divider />

      {/* 控制面板 */}
      <Card size="small" style={{ marginBottom: 16 }}>
        <Row gutter={16} align="middle">
          <Col span={4}>
            <Text strong>智能体选择:</Text>
            <Select 
              value={selectedAgent} 
              onChange={setSelectedAgent}
              style={{ width: '100%', marginTop: 4 }}
              size="small"
            >
              <Option value="all">全部智能体</Option>
              <Option value="DQN-1">DQN-1</Option>
              <Option value="Double-DQN-2">Double-DQN-2</Option>
              <Option value="Dueling-DQN-3">Dueling-DQN-3</Option>
              <Option value="Tabular-Q-4">Tabular-Q-4</Option>
            </Select>
          </Col>
          <Col span={4}>
            <Text strong>时间范围:</Text>
            <Radio.Group 
              value={timeRange} 
              onChange={(e) => setTimeRange(e.target.value)}
              size="small"
              style={{ marginTop: 4 }}
            >
              <Radio.Button value="1h">1小时</Radio.Button>
              <Radio.Button value="24h">24小时</Radio.Button>
              <Radio.Button value="7d">7天</Radio.Button>
            </Radio.Group>
          </Col>
          <Col span={4}>
            <Button 
              type="primary" 
              icon={<DashboardOutlined />}
              loading={refreshing}
              onClick={handleRefresh}
              size="small"
            >
              刷新数据
            </Button>
          </Col>
        </Row>
      </Card>

      <Tabs defaultActiveKey="1">
        <TabPane tab="性能总览" key="1">
          <Space direction="vertical" size="large" style={{ width: '100%' }}>
            <KPICards />
            <PerformanceTrendChart />
            
            <Row gutter={16}>
              <Col span={12}>
                <AgentComparisonChart />
              </Col>
              <Col span={12}>
                <SystemResourceChart />
              </Col>
            </Row>
          </Space>
        </TabPane>

        <TabPane tab="详细分析" key="2">
          <Row gutter={16}>
            <Col span={12}>
              <InferencePerformanceScatter />
            </Col>
            <Col span={12}>
              <PerformanceAlerts />
            </Col>
          </Row>
          
          <div style={{ marginTop: 16 }}>
            <PerformanceDetailTable />
          </div>
        </TabPane>

        <TabPane tab="系统监控" key="3">
          <Row gutter={16}>
            <Col span={6}>
              <Card>
                <Statistic
                  title="系统负载"
                  value={75.3}
                  precision={1}
                  suffix="%"
                  prefix={<MonitorOutlined />}
                  valueStyle={{ color: '#fa8c16' }}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="GPU使用率"
                  value={85.7}
                  precision={1}
                  suffix="%"
                  prefix={<FireOutlined />}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="网络IO"
                  value={234.5}
                  precision={1}
                  suffix="MB/s"
                  prefix={<ThunderboltOutlined />}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="磁盘IO"
                  value={45.2}
                  precision={1}
                  suffix="MB/s"
                  prefix={<SettingOutlined />}
                />
              </Card>
            </Col>
          </Row>
          
          <div style={{ marginTop: 16 }}>
            <Card title="实时系统指标" size="small">
              <ResponsiveContainer width="100%" height={400}>
                <ComposedChart data={systemData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" tickFormatter={(time) => new Date(time).toLocaleTimeString()} />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" />
                  <RechartsTooltip />
                  <Legend />
                  <Area 
                    yAxisId="left"
                    type="monotone" 
                    dataKey="cpuUsage" 
                    fill="#1890ff" 
                    stroke="#1890ff" 
                    name="CPU(%)"
                    fillOpacity={0.3}
                  />
                  <Area 
                    yAxisId="left"
                    type="monotone" 
                    dataKey="gpuUsage" 
                    fill="#52c41a" 
                    stroke="#52c41a" 
                    name="GPU(%)"
                    fillOpacity={0.3}
                  />
                  <Line 
                    yAxisId="right"
                    type="monotone" 
                    dataKey="inferenceQPS" 
                    stroke="#fa8c16" 
                    name="推理QPS"
                    strokeWidth={2}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </Card>
          </div>
        </TabPane>
      </Tabs>
    </div>
  )
}

export default QLearningPerformancePage