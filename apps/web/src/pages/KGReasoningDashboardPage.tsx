import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Statistic,
  Progress,
  Badge,
  Table,
  Tabs,
  Alert,
  Tag,
  Button,
  Space,
  Typography,
  Divider,
  Timeline,
  List,
  Avatar,
} from 'antd'
import {
  ThunderboltOutlined,
  RobotOutlined,
  NodeIndexOutlined,
  ShareAltOutlined,
  ExperimentOutlined,
  DashboardOutlined,
  TrophyOutlined,
  LineChartOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  PlayCircleOutlined,
  DatabaseOutlined,
  BulbOutlined,
  SettingOutlined,
} from '@ant-design/icons'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, BarChart, Bar } from 'recharts'

const { Title, Text, Paragraph } = Typography
const { TabPane } = Tabs

const KGReasoningDashboardPage: React.FC = () => {
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('overview')

  // 模拟数据
  const reasoningEngineStats = {
    totalQueries: 15648,
    successRate: 94.2,
    avgResponseTime: 1.8,
    activeStrategies: 8,
    totalRules: 342,
    totalEmbeddings: 128000,
    pathsCached: 95640,
    uncertaintyAnalysis: 2847
  }

  const strategyPerformance = [
    { name: 'Ensemble', accuracy: 95.2, usage: 35, responseTime: 2.1, color: '#1890ff' },
    { name: 'Adaptive', accuracy: 92.8, usage: 28, responseTime: 1.7, color: '#52c41a' },
    { name: 'Rule Only', accuracy: 89.4, usage: 15, responseTime: 0.9, color: '#fa541c' },
    { name: 'Embedding Only', accuracy: 87.6, usage: 12, responseTime: 1.2, color: '#722ed1' },
    { name: 'Path Only', accuracy: 85.3, usage: 8, responseTime: 1.5, color: '#eb2f96' },
    { name: 'Cascading', accuracy: 91.7, usage: 2, responseTime: 2.8, color: '#13c2c2' },
  ]

  const performanceData = [
    { time: '00:00', queries: 120, accuracy: 94.5, responseTime: 1.8 },
    { time: '04:00', queries: 85, accuracy: 95.2, responseTime: 1.6 },
    { time: '08:00', queries: 450, accuracy: 93.8, responseTime: 2.1 },
    { time: '12:00', queries: 680, accuracy: 94.1, responseTime: 1.9 },
    { time: '16:00', queries: 520, accuracy: 95.0, responseTime: 1.7 },
    { time: '20:00', queries: 380, accuracy: 93.9, responseTime: 2.0 },
  ]

  const recentActivities = [
    {
      time: '2 分钟前',
      type: 'success',
      title: '规则推理成功',
      description: '复杂关系推导完成，置信度 0.92',
      icon: <CheckCircleOutlined style={{ color: '#52c41a' }} />,
    },
    {
      time: '5 分钟前',
      type: 'processing',
      title: '批量推理处理中',
      description: '正在处理 1,250 个查询请求',
      icon: <PlayCircleOutlined style={{ color: '#1890ff' }} />,
    },
    {
      time: '8 分钟前',
      type: 'warning',
      title: '路径推理性能警告',
      description: '响应时间超过阈值 (3.2s)',
      icon: <ExclamationCircleOutlined style={{ color: '#faad14' }} />,
    },
    {
      time: '12 分钟前',
      type: 'success',
      title: '嵌入模型更新',
      description: 'TransE模型训练完成，准确率提升至 87.6%',
      icon: <CheckCircleOutlined style={{ color: '#52c41a' }} />,
    },
    {
      time: '15 分钟前',
      type: 'info',
      title: '不确定性分析',
      description: '贝叶斯推理完成，后验概率计算就绪',
      icon: <BulbOutlined style={{ color: '#722ed1' }} />,
    },
  ]

  const engineStatus = [
    { name: '混合推理引擎', status: 'running', load: 78, queries: 8540 },
    { name: '规则推理引擎', status: 'running', load: 65, queries: 3210 },
    { name: '嵌入推理引擎', status: 'running', load: 82, queries: 2890 },
    { name: '路径推理引擎', status: 'running', load: 45, queries: 980 },
    { name: '不确定性推理', status: 'running', load: 35, queries: 28 },
  ]

  const topQueries = [
    { query: 'person(X) → human(X)', type: 'Rule', confidence: 0.95, executions: 1205 },
    { query: 'similar_to(company, organization)', type: 'Embedding', confidence: 0.87, executions: 892 },
    { query: 'path(Alice, works_at, Company)', type: 'Path', confidence: 0.91, executions: 745 },
    { query: 'probability(rain | weather_forecast)', type: 'Uncertainty', confidence: 0.78, executions: 432 },
    { query: 'entity_type(X) ∧ located_in(X, Y)', type: 'Rule', confidence: 0.89, executions: 321 },
  ]

  const COLORS = ['#1890ff', '#52c41a', '#fa541c', '#722ed1', '#eb2f96', '#13c2c2']

  return (
    <div style={{ padding: '24px', background: '#f0f2f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <ThunderboltOutlined /> 知识图推理引擎总览
        </Title>
        <Paragraph>
          监控和管理混合推理引擎的性能、策略执行状态和系统健康度
        </Paragraph>
      </div>

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="系统概览" key="overview">
          {/* 核心指标卡片 */}
          <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
            <Col xs={24} sm={12} lg={6}>
              <Card>
                <Statistic
                  title="总查询数"
                  value={reasoningEngineStats.totalQueries}
                  prefix={<DashboardOutlined />}
                  valueStyle={{ color: '#1890ff' }}
                />
                <Progress percent={85} strokeColor="#1890ff" size="small" showInfo={false} />
              </Card>
            </Col>
            <Col xs={24} sm={12} lg={6}>
              <Card>
                <Statistic
                  title="成功率"
                  value={reasoningEngineStats.successRate}
                  suffix="%"
                  prefix={<TrophyOutlined />}
                  valueStyle={{ color: '#52c41a' }}
                />
                <Progress percent={reasoningEngineStats.successRate} strokeColor="#52c41a" size="small" showInfo={false} />
              </Card>
            </Col>
            <Col xs={24} sm={12} lg={6}>
              <Card>
                <Statistic
                  title="平均响应时间"
                  value={reasoningEngineStats.avgResponseTime}
                  suffix="s"
                  prefix={<ClockCircleOutlined />}
                  valueStyle={{ color: '#fa541c' }}
                />
                <Progress percent={75} strokeColor="#fa541c" size="small" showInfo={false} />
              </Card>
            </Col>
            <Col xs={24} sm={12} lg={6}>
              <Card>
                <Statistic
                  title="活跃策略"
                  value={reasoningEngineStats.activeStrategies}
                  prefix={<SettingOutlined />}
                  valueStyle={{ color: '#722ed1' }}
                />
                <Progress percent={100} strokeColor="#722ed1" size="small" showInfo={false} />
              </Card>
            </Col>
          </Row>

          {/* 推理引擎状态 */}
          <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
            <Col xs={24} lg={16}>
              <Card title="推理引擎状态" extra={<Badge status="processing" text="实时监控" />}>
                <Table
                  dataSource={engineStatus}
                  rowKey="name"
                  pagination={false}
                  size="small"
                  columns={[
                    {
                      title: '引擎名称',
                      dataIndex: 'name',
                      key: 'name',
                      render: (text) => (
                        <Space>
                          <RobotOutlined />
                          <Text strong>{text}</Text>
                        </Space>
                      ),
                    },
                    {
                      title: '状态',
                      dataIndex: 'status',
                      key: 'status',
                      render: (status) => (
                        <Badge
                          status={status === 'running' ? 'success' : 'error'}
                          text={status === 'running' ? '运行中' : '已停止'}
                        />
                      ),
                    },
                    {
                      title: '负载率',
                      dataIndex: 'load',
                      key: 'load',
                      render: (load) => (
                        <Progress
                          percent={load}
                          size="small"
                          strokeColor={load > 80 ? '#ff4d4f' : load > 60 ? '#faad14' : '#52c41a'}
                        />
                      ),
                    },
                    {
                      title: '查询数',
                      dataIndex: 'queries',
                      key: 'queries',
                      render: (queries) => <Text>{queries.toLocaleString()}</Text>,
                    },
                  ]}
                />
              </Card>
            </Col>
            <Col xs={24} lg={8}>
              <Card title="策略分布" extra={<LineChartOutlined />}>
                <ResponsiveContainer width="100%" height={200}>
                  <PieChart>
                    <Pie
                      data={strategyPerformance}
                      cx="50%"
                      cy="50%"
                      outerRadius={60}
                      fill="#8884d8"
                      dataKey="usage"
                      label={({ name, usage }) => `${name}: ${usage}%`}
                    >
                      {strategyPerformance.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </Card>
            </Col>
          </Row>

          {/* 性能趋势 */}
          <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
            <Col xs={24}>
              <Card title="24小时性能趋势" extra={<Badge status="processing" text="实时更新" />}>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={performanceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis yAxisId="left" />
                    <YAxis yAxisId="right" orientation="right" />
                    <Tooltip />
                    <Line
                      yAxisId="left"
                      type="monotone"
                      dataKey="queries"
                      stroke="#1890ff"
                      strokeWidth={2}
                      name="查询数"
                    />
                    <Line
                      yAxisId="right"
                      type="monotone"
                      dataKey="accuracy"
                      stroke="#52c41a"
                      strokeWidth={2}
                      name="准确率(%)"
                    />
                    <Line
                      yAxisId="right"
                      type="monotone"
                      dataKey="responseTime"
                      stroke="#fa541c"
                      strokeWidth={2}
                      name="响应时间(s)"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab="策略性能" key="strategies">
          <Row gutter={[16, 16]}>
            <Col xs={24} lg={16}>
              <Card title="推理策略性能对比" extra={<TrophyOutlined />}>
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={strategyPerformance}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="accuracy" fill="#52c41a" name="准确率(%)" />
                    <Bar dataKey="usage" fill="#1890ff" name="使用率(%)" />
                  </BarChart>
                </ResponsiveContainer>
              </Card>
            </Col>
            <Col xs={24} lg={8}>
              <Card title="策略详情" size="small">
                <List
                  dataSource={strategyPerformance}
                  renderItem={(item) => (
                    <List.Item>
                      <List.Item.Meta
                        avatar={
                          <Avatar
                            style={{ backgroundColor: item.color }}
                            icon={<ThunderboltOutlined />}
                          />
                        }
                        title={
                          <Space>
                            <Text strong>{item.name}</Text>
                            <Tag color={item.accuracy > 90 ? 'green' : item.accuracy > 85 ? 'orange' : 'red'}>
                              {item.accuracy}%
                            </Tag>
                          </Space>
                        }
                        description={
                          <Space direction="vertical" size={2}>
                            <Text type="secondary">使用率: {item.usage}%</Text>
                            <Text type="secondary">响应时间: {item.responseTime}s</Text>
                          </Space>
                        }
                      />
                    </List.Item>
                  )}
                />
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab="热门查询" key="queries">
          <Card title="热门推理查询" extra={<DatabaseOutlined />}>
            <Table
              dataSource={topQueries}
              rowKey="query"
              pagination={false}
              columns={[
                {
                  title: '查询表达式',
                  dataIndex: 'query',
                  key: 'query',
                  render: (text) => <Text code>{text}</Text>,
                },
                {
                  title: '类型',
                  dataIndex: 'type',
                  key: 'type',
                  render: (type) => {
                    const typeColors = {
                      Rule: 'blue',
                      Embedding: 'green',
                      Path: 'orange',
                      Uncertainty: 'purple',
                    }
                    return <Tag color={typeColors[type]}>{type}</Tag>
                  },
                },
                {
                  title: '置信度',
                  dataIndex: 'confidence',
                  key: 'confidence',
                  render: (confidence) => (
                    <Progress
                      percent={confidence * 100}
                      size="small"
                      format={() => confidence.toFixed(2)}
                    />
                  ),
                },
                {
                  title: '执行次数',
                  dataIndex: 'executions',
                  key: 'executions',
                  render: (executions) => <Text>{executions.toLocaleString()}</Text>,
                },
              ]}
            />
          </Card>
        </TabPane>

        <TabPane tab="活动日志" key="activities">
          <Card title="实时活动日志" extra={<ClockCircleOutlined />}>
            <Timeline
              items={recentActivities.map((activity) => ({
                color: activity.type === 'success' ? 'green' : 
                       activity.type === 'warning' ? 'orange' :
                       activity.type === 'processing' ? 'blue' : 'gray',
                dot: activity.icon,
                children: (
                  <div>
                    <div style={{ marginBottom: '4px' }}>
                      <Text strong>{activity.title}</Text>
                      <Text type="secondary" style={{ marginLeft: '12px' }}>
                        {activity.time}
                      </Text>
                    </div>
                    <Text type="secondary">{activity.description}</Text>
                  </div>
                ),
              }))}
            />
          </Card>
        </TabPane>
      </Tabs>

      {/* 底部警告信息 */}
      <Alert
        message="推理引擎监控"
        description="系统正在监控所有推理引擎的性能指标。如发现异常，请及时检查相应的引擎配置。"
        type="info"
        showIcon
        style={{ marginTop: '24px' }}
      />
    </div>
  )
}

export default KGReasoningDashboardPage