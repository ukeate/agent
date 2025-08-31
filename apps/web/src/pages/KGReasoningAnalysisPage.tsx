import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Button,
  Table,
  Tabs,
  Space,
  Typography,
  Input,
  Select,
  DatePicker,
  Modal,
  Tag,
  Badge,
  Progress,
  Statistic,
  Alert,
  Tree,
  List,
  Avatar,
  Tooltip,
  Drawer,
  Collapse,
  Timeline,
  Rate,
  Switch,
  Slider,
  Form,
  Divider,
  notification,
} from 'antd'
import {
  BarChartOutlined,
  PieChartOutlined,
  LineChartOutlined,
  SearchOutlined,
  FilterOutlined,
  ExportOutlined,
  EyeOutlined,
  DownloadOutlined,
  ShareAltOutlined,
  BookOutlined,
  BulbOutlined,
  ThunderboltOutlined,
  TrophyOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  InfoCircleOutlined,
  NodeIndexOutlined,
  DatabaseOutlined,
  FileTextOutlined,
  SettingOutlined,
  SyncOutlined,
  ReloadOutlined,
  StarOutlined,
  HeartOutlined,
  CommentOutlined,
} from '@ant-design/icons'
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip as RechartsTooltip, 
  ResponsiveContainer, 
  AreaChart, 
  Area, 
  BarChart, 
  Bar, 
  PieChart, 
  Pie, 
  Cell,
  ScatterChart,
  Scatter,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  TreeMap
} from 'recharts'

const { Title, Text, Paragraph } = Typography
const { TabPane } = Tabs
const { RangePicker } = DatePicker
const { Panel } = Collapse
const { Search } = Input
const { Option } = Select

interface ReasoningResult {
  id: string
  query: string
  strategy: string
  results: any[]
  confidence: number
  responseTime: number
  accuracy?: number
  timestamp: string
  userId: string
  feedback?: {
    rating: number
    comment: string
    helpful: boolean
  }
  explanation?: {
    steps: string[]
    evidence: string[]
    confidence_breakdown: Record<string, number>
  }
}

interface AnalysisMetric {
  name: string
  value: number
  trend: 'up' | 'down' | 'stable'
  changePercent: number
  description: string
}

const KGReasoningAnalysisPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState('overview')
  const [selectedResult, setSelectedResult] = useState<ReasoningResult | null>(null)
  const [detailDrawerVisible, setDetailDrawerVisible] = useState(false)
  const [filterModalVisible, setFilterModalVisible] = useState(false)
  const [dateRange, setDateRange] = useState<any[]>([])
  const [selectedStrategies, setSelectedStrategies] = useState<string[]>([])
  const [confidenceRange, setConfidenceRange] = useState<[number, number]>([0, 1])

  // 分析指标数据
  const analysisMetrics: AnalysisMetric[] = [
    {
      name: '平均准确率',
      value: 94.2,
      trend: 'up',
      changePercent: 2.3,
      description: '过去7天推理结果的平均准确率'
    },
    {
      name: '用户满意度',
      value: 4.6,
      trend: 'up', 
      changePercent: 0.8,
      description: '基于用户反馈的满意度评分'
    },
    {
      name: '响应速度',
      value: 1.8,
      trend: 'down',
      changePercent: -12.5,
      description: '平均查询响应时间(秒)'
    },
    {
      name: '成功率',
      value: 96.8,
      trend: 'stable',
      changePercent: 0.2,
      description: '推理查询的成功执行率'
    }
  ]

  // 模拟推理结果数据
  const mockResults: ReasoningResult[] = [
    {
      id: 'result_001',
      query: 'person(X) → human(X)',
      strategy: 'rule_only',
      results: [
        { entity: 'Alice', confidence: 0.98, type: 'person' },
        { entity: 'Bob', confidence: 0.95, type: 'person' }
      ],
      confidence: 0.96,
      responseTime: 0.8,
      accuracy: 98.5,
      timestamp: '2024-01-20 10:30:15',
      userId: 'user_001',
      feedback: {
        rating: 5,
        comment: '结果非常准确，推理过程清晰',
        helpful: true
      },
      explanation: {
        steps: ['规则匹配', '实体识别', '置信度计算', '结果验证'],
        evidence: ['规则: person(X) → human(X)', '实体库匹配', '语义验证'],
        confidence_breakdown: { 'rule_match': 0.98, 'entity_match': 0.95, 'semantic_check': 0.94 }
      }
    },
    {
      id: 'result_002',
      query: 'similar_to(?, "artificial intelligence")',
      strategy: 'embedding_only',
      results: [
        { entity: 'machine_learning', similarity: 0.89, type: 'concept' },
        { entity: 'deep_learning', similarity: 0.85, type: 'concept' },
        { entity: 'neural_networks', similarity: 0.82, type: 'concept' }
      ],
      confidence: 0.85,
      responseTime: 1.2,
      accuracy: 87.3,
      timestamp: '2024-01-20 10:25:30',
      userId: 'user_002',
      feedback: {
        rating: 4,
        comment: '相关性较高，但可以更准确',
        helpful: true
      },
      explanation: {
        steps: ['向量检索', '相似度计算', '结果排序', '阈值过滤'],
        evidence: ['TransE嵌入模型', '余弦相似度计算', 'Top-K检索'],
        confidence_breakdown: { 'embedding_quality': 0.87, 'similarity_threshold': 0.85, 'context_match': 0.83 }
      }
    },
    {
      id: 'result_003',
      query: 'path(Alice, works_at, ?Company)',
      strategy: 'path_only',
      results: [
        { path: ['Alice', 'works_at', 'TechCorp'], confidence: 0.92, hops: 1 },
        { path: ['Alice', 'member_of', 'Team_A', 'part_of', 'TechCorp'], confidence: 0.78, hops: 3 }
      ],
      confidence: 0.85,
      responseTime: 2.1,
      accuracy: 89.2,
      timestamp: '2024-01-20 10:20:45',
      userId: 'user_003',
      feedback: {
        rating: 4,
        comment: '找到了正确的路径关系',
        helpful: true
      },
      explanation: {
        steps: ['路径搜索', '多跳推理', '路径验证', '置信度评估'],
        evidence: ['BFS搜索算法', '路径权重计算', '关系验证'],
        confidence_breakdown: { 'path_existence': 0.92, 'relationship_strength': 0.85, 'path_length': 0.78 }
      }
    }
  ]

  // 策略性能对比数据
  const strategyPerformance = [
    { strategy: '规则推理', accuracy: 95.2, speed: 90, satisfaction: 4.7, usage: 35 },
    { strategy: '嵌入推理', accuracy: 87.6, speed: 85, satisfaction: 4.3, usage: 28 },
    { strategy: '路径推理', accuracy: 89.4, speed: 75, satisfaction: 4.1, usage: 20 },
    { strategy: '集成策略', accuracy: 96.8, speed: 65, satisfaction: 4.8, usage: 17 }
  ]

  // 准确率趋势数据
  const accuracyTrends = [
    { date: '01-14', rule: 94.2, embedding: 86.1, path: 88.5, ensemble: 95.8 },
    { date: '01-15', rule: 94.8, embedding: 86.8, path: 89.1, ensemble: 96.2 },
    { date: '01-16', rule: 95.1, embedding: 87.2, path: 89.6, ensemble: 96.5 },
    { date: '01-17', rule: 94.9, embedding: 87.8, path: 88.9, ensemble: 96.8 },
    { date: '01-18', rule: 95.3, embedding: 88.1, path: 90.2, ensemble: 97.1 },
    { date: '01-19', rule: 95.0, embedding: 87.9, path: 89.8, ensemble: 96.9 },
    { date: '01-20', rule: 95.2, embedding: 87.6, path: 89.4, ensemble: 96.8 }
  ]

  // 用户满意度分布
  const satisfactionDistribution = [
    { rating: '5星', count: 245, percentage: 68.2 },
    { rating: '4星', count: 89, percentage: 24.8 },
    { rating: '3星', count: 18, percentage: 5.0 },
    { rating: '2星', count: 5, percentage: 1.4 },
    { rating: '1星', count: 2, percentage: 0.6 }
  ]

  // 查询类型分析
  const queryTypeAnalysis = [
    { type: '实体推理', count: 1250, accuracy: 96.2, avgTime: 0.8 },
    { type: '关系推理', count: 890, accuracy: 92.5, avgTime: 1.5 },
    { type: '相似度查询', count: 675, accuracy: 87.8, avgTime: 1.2 },
    { type: '路径查询', count: 445, accuracy: 89.4, avgTime: 2.1 },
    { type: '复杂推理', count: 180, accuracy: 94.8, avgTime: 3.2 }
  ]

  const handleExportReport = () => {
    notification.success({
      message: '报告导出成功',
      description: '分析报告已生成并开始下载'
    })
  }

  const handleViewDetail = (result: ReasoningResult) => {
    setSelectedResult(result)
    setDetailDrawerVisible(true)
  }

  const getTrendIcon = (trend: string, changePercent: number) => {
    if (trend === 'up') {
      return <TrophyOutlined style={{ color: '#52c41a' }} />
    } else if (trend === 'down') {
      return changePercent < 0 ? <CheckCircleOutlined style={{ color: '#52c41a' }} /> : <WarningOutlined style={{ color: '#ff4d4f' }} />
    }
    return <InfoCircleOutlined style={{ color: '#1890ff' }} />
  }

  const resultColumns = [
    {
      title: 'Query ID',
      dataIndex: 'id',
      key: 'id',
      width: 120,
      render: (id: string) => <Text code>{id}</Text>
    },
    {
      title: '查询内容',
      dataIndex: 'query',
      key: 'query',
      render: (query: string) => (
        <Tooltip title={query}>
          <Text style={{ maxWidth: 200, display: 'block', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            {query}
          </Text>
        </Tooltip>
      )
    },
    {
      title: '策略',
      dataIndex: 'strategy',
      key: 'strategy',
      render: (strategy: string) => <Tag color="blue">{strategy}</Tag>
    },
    {
      title: '置信度',
      dataIndex: 'confidence',
      key: 'confidence',
      render: (confidence: number) => (
        <Progress 
          percent={confidence * 100} 
          size="small" 
          format={() => confidence.toFixed(2)}
          strokeColor={confidence > 0.9 ? '#52c41a' : confidence > 0.7 ? '#faad14' : '#ff4d4f'}
        />
      )
    },
    {
      title: '准确率',
      dataIndex: 'accuracy',
      key: 'accuracy',
      render: (accuracy?: number) => (
        accuracy ? <Text>{accuracy.toFixed(1)}%</Text> : <Text type="secondary">-</Text>
      )
    },
    {
      title: '响应时间',
      dataIndex: 'responseTime',
      key: 'responseTime',
      render: (time: number) => <Text>{time.toFixed(1)}s</Text>
    },
    {
      title: '用户评分',
      key: 'feedback',
      render: (_, record: ReasoningResult) => (
        record.feedback ? (
          <Space>
            <Rate disabled defaultValue={record.feedback.rating} style={{ fontSize: '14px' }} />
            <Text style={{ fontSize: '12px' }}>({record.feedback.rating})</Text>
          </Space>
        ) : (
          <Text type="secondary">未评分</Text>
        )
      )
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record: ReasoningResult) => (
        <Space>
          <Button 
            size="small" 
            icon={<EyeOutlined />}
            onClick={() => handleViewDetail(record)}
          >
            详情
          </Button>
          <Button size="small" icon={<ShareAltOutlined />}>分享</Button>
        </Space>
      )
    }
  ]

  const COLORS = ['#1890ff', '#52c41a', '#fa541c', '#722ed1', '#eb2f96', '#13c2c2']

  return (
    <div style={{ padding: '24px', background: '#f0f2f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <BarChartOutlined /> 推理结果分析
        </Title>
        <Paragraph>
          深度分析推理结果质量、用户反馈和系统性能表现
        </Paragraph>
      </div>

      {/* 核心指标卡片 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        {analysisMetrics.map((metric, index) => (
          <Col xs={24} sm={12} lg={6} key={metric.name}>
            <Card>
              <Statistic
                title={metric.name}
                value={metric.value}
                suffix={metric.name.includes('率') || metric.name.includes('度') ? '%' : 
                       metric.name.includes('速度') ? 's' : ''}
                valueStyle={{ 
                  color: metric.trend === 'up' && metric.changePercent > 0 ? '#52c41a' :
                         metric.trend === 'down' && metric.changePercent < 0 && metric.name.includes('速度') ? '#52c41a' :
                         metric.trend === 'down' && metric.changePercent < 0 ? '#ff4d4f' : '#1890ff'
                }}
                prefix={getTrendIcon(metric.trend, metric.changePercent)}
              />
              <div style={{ marginTop: '8px', fontSize: '12px' }}>
                <Text type="secondary">{metric.description}</Text>
                <br />
                <Text 
                  style={{ 
                    color: metric.changePercent > 0 ? '#52c41a' : '#ff4d4f'
                  }}
                >
                  {metric.changePercent > 0 ? '+' : ''}{metric.changePercent}%
                </Text>
              </div>
            </Card>
          </Col>
        ))}
      </Row>

      {/* 快速操作栏 */}
      <Card style={{ marginBottom: '24px' }}>
        <Space size="large" style={{ width: '100%', justifyContent: 'space-between' }}>
          <Space>
            <Search 
              placeholder="搜索查询内容..." 
              style={{ width: 300 }}
              onSearch={(value) => {
                // 处理搜索逻辑
              }}
            />
            <Button 
              icon={<FilterOutlined />}
              onClick={() => setFilterModalVisible(true)}
            >
              高级筛选
            </Button>
          </Space>
          <Space>
            <Button icon={<ExportOutlined />} onClick={handleExportReport}>
              导出报告
            </Button>
            <Button icon={<ReloadOutlined />}>刷新数据</Button>
          </Space>
        </Space>
      </Card>

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="综合分析" key="overview">
          <Row gutter={[16, 16]}>
            {/* 准确率趋势图 */}
            <Col xs={24} lg={16}>
              <Card title="准确率趋势分析" extra={<LineChartOutlined />}>
                <ResponsiveContainer width="100%" height={350}>
                  <LineChart data={accuracyTrends}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis domain={[80, 100]} />
                    <RechartsTooltip />
                    <Line type="monotone" dataKey="rule" stroke="#1890ff" name="规则推理" />
                    <Line type="monotone" dataKey="embedding" stroke="#52c41a" name="嵌入推理" />
                    <Line type="monotone" dataKey="path" stroke="#fa541c" name="路径推理" />
                    <Line type="monotone" dataKey="ensemble" stroke="#722ed1" name="集成策略" />
                  </LineChart>
                </ResponsiveContainer>
              </Card>
            </Col>

            {/* 策略性能雷达图 */}
            <Col xs={24} lg={8}>
              <Card title="策略性能对比" extra={<ThunderboltOutlined />}>
                <ResponsiveContainer width="100%" height={350}>
                  <RadarChart data={strategyPerformance}>
                    <PolarGrid />
                    <PolarAngleAxis dataKey="strategy" />
                    <PolarRadiusAxis angle={90} domain={[0, 100]} />
                    <Radar
                      name="准确率"
                      dataKey="accuracy"
                      stroke="#1890ff"
                      fill="#1890ff"
                      fillOpacity={0.3}
                    />
                    <Radar
                      name="速度"
                      dataKey="speed"
                      stroke="#52c41a"
                      fill="transparent"
                      strokeDasharray="3 3"
                    />
                    <RechartsTooltip />
                  </RadarChart>
                </ResponsiveContainer>
              </Card>
            </Col>
          </Row>

          {/* 查询类型分析 */}
          <Row gutter={[16, 16]} style={{ marginTop: '16px' }}>
            <Col xs={24} lg={12}>
              <Card title="查询类型分析">
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={queryTypeAnalysis}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="type" />
                    <YAxis yAxisId="left" />
                    <YAxis yAxisId="right" orientation="right" />
                    <RechartsTooltip />
                    <Bar yAxisId="left" dataKey="count" fill="#1890ff" name="查询数量" />
                    <Bar yAxisId="right" dataKey="accuracy" fill="#52c41a" name="准确率%" />
                  </BarChart>
                </ResponsiveContainer>
              </Card>
            </Col>

            <Col xs={24} lg={12}>
              <Card title="用户满意度分布">
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie
                      data={satisfactionDistribution}
                      cx="50%"
                      cy="50%"
                      outerRadius={80}
                      dataKey="count"
                      label={({ rating, percentage }) => `${rating}: ${percentage}%`}
                    >
                      {satisfactionDistribution.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <RechartsTooltip />
                  </PieChart>
                </ResponsiveContainer>
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab="详细结果" key="results">
          <Card title="推理结果详情列表">
            <Table
              dataSource={mockResults}
              columns={resultColumns}
              rowKey="id"
              pagination={{
                showSizeChanger: true,
                showQuickJumper: true,
                showTotal: (total) => `共 ${total} 条结果`
              }}
              expandable={{
                expandedRowRender: (record: ReasoningResult) => (
                  <Row gutter={16}>
                    <Col span={12}>
                      <Card size="small" title="推理结果">
                        <List
                          dataSource={record.results}
                          renderItem={(item: any) => (
                            <List.Item>
                              <List.Item.Meta
                                title={item.entity || item.path?.join(' → ')}
                                description={
                                  <Space>
                                    <Text>置信度: {(item.confidence || item.similarity)?.toFixed(2)}</Text>
                                    {item.type && <Tag>{item.type}</Tag>}
                                    {item.hops && <Tag>跳数: {item.hops}</Tag>}
                                  </Space>
                                }
                              />
                            </List.Item>
                          )}
                        />
                      </Card>
                    </Col>
                    <Col span={12}>
                      {record.explanation && (
                        <Card size="small" title="推理解释">
                          <Collapse size="small">
                            <Panel header="推理步骤" key="steps">
                              <Timeline size="small">
                                {record.explanation.steps.map((step, index) => (
                                  <Timeline.Item key={index}>{step}</Timeline.Item>
                                ))}
                              </Timeline>
                            </Panel>
                            <Panel header="证据信息" key="evidence">
                              {record.explanation.evidence.map((evidence, index) => (
                                <Alert 
                                  key={index}
                                  message={evidence} 
                                  type="info" 
                                  size="small" 
                                  style={{ marginBottom: '4px' }}
                                />
                              ))}
                            </Panel>
                            <Panel header="置信度分解" key="confidence">
                              {Object.entries(record.explanation.confidence_breakdown).map(([key, value]) => (
                                <div key={key} style={{ marginBottom: '8px' }}>
                                  <Text>{key}:</Text>
                                  <Progress percent={value * 100} size="small" />
                                </div>
                              ))}
                            </Panel>
                          </Collapse>
                        </Card>
                      )}
                    </Col>
                  </Row>
                )
              }}
            />
          </Card>
        </TabPane>

        <TabPane tab="用户反馈" key="feedback">
          <Row gutter={[16, 16]}>
            <Col xs={24} lg={16}>
              <Card title="用户反馈详情">
                <List
                  dataSource={mockResults.filter(r => r.feedback)}
                  renderItem={(result) => (
                    <List.Item
                      actions={[
                        <Button size="small" icon={<HeartOutlined />}>有用</Button>,
                        <Button size="small" icon={<CommentOutlined />}>回复</Button>
                      ]}
                    >
                      <List.Item.Meta
                        avatar={
                          <Avatar style={{ backgroundColor: '#87d068' }}>
                            {result.userId.slice(-1)}
                          </Avatar>
                        }
                        title={
                          <Space>
                            <Text strong>查询: {result.query.substring(0, 50)}...</Text>
                            <Rate disabled value={result.feedback!.rating} style={{ fontSize: '14px' }} />
                          </Space>
                        }
                        description={
                          <Space direction="vertical" style={{ width: '100%' }}>
                            <Text>{result.feedback!.comment}</Text>
                            <Space>
                              <Text type="secondary">用户: {result.userId}</Text>
                              <Text type="secondary">时间: {result.timestamp}</Text>
                              <Text type="secondary">策略: {result.strategy}</Text>
                              {result.feedback!.helpful && <Badge status="success" text="标记为有用" />}
                            </Space>
                          </Space>
                        }
                      />
                    </List.Item>
                  )}
                />
              </Card>
            </Col>
            <Col xs={24} lg={8}>
              <Card title="反馈统计" style={{ marginBottom: '16px' }}>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Text>平均评分:</Text>
                    <Space>
                      <Rate disabled value={4.6} allowHalf style={{ fontSize: '14px' }} />
                      <Text strong>4.6/5</Text>
                    </Space>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Text>总反馈数:</Text>
                    <Text strong>359</Text>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Text>有用标记:</Text>
                    <Text strong style={{ color: '#52c41a' }}>312 (87%)</Text>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Text>改进建议:</Text>
                    <Text strong style={{ color: '#faad14' }}>47</Text>
                  </div>
                </Space>
              </Card>
              <Card title="反馈趋势" size="small">
                <ResponsiveContainer width="100%" height={200}>
                  <AreaChart data={[
                    { date: '01-16', rating: 4.3, count: 45 },
                    { date: '01-17', rating: 4.5, count: 52 },
                    { date: '01-18', rating: 4.6, count: 61 },
                    { date: '01-19', rating: 4.7, count: 58 },
                    { date: '01-20', rating: 4.6, count: 63 }
                  ]}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <RechartsTooltip />
                    <Area type="monotone" dataKey="rating" stackId="1" stroke="#1890ff" fill="#1890ff" />
                  </AreaChart>
                </ResponsiveContainer>
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab="质量洞察" key="insights">
          <Row gutter={[16, 16]}>
            <Col xs={24} lg={12}>
              <Card title="质量洞察报告" extra={<BulbOutlined />}>
                <Space direction="vertical" style={{ width: '100%' }} size="large">
                  <Alert
                    message="推理准确率持续提升"
                    description="过去7天平均准确率从92.1%提升至94.2%，主要得益于集成策略的优化"
                    type="success"
                    showIcon
                  />
                  <Alert
                    message="路径推理性能有待改进"
                    description="路径推理的平均响应时间为2.1秒，建议优化搜索算法以提高效率"
                    type="warning"
                    showIcon
                  />
                  <Alert
                    message="用户满意度表现优秀"
                    description="68.2%的用户给出5星评价，用户反馈整体积极正面"
                    type="info"
                    showIcon
                  />
                  <Alert
                    message="嵌入模型需要更新"
                    description="嵌入推理的准确率相对较低，建议升级到更新的预训练模型"
                    type="warning"
                    showIcon
                  />
                </Space>
              </Card>
            </Col>
            <Col xs={24} lg={12}>
              <Card title="改进建议" extra={<StarOutlined />}>
                <List
                  dataSource={[
                    {
                      title: '优化路径搜索算法',
                      description: '实现A*搜索算法，预计可提升30%性能',
                      priority: 'high',
                      impact: 'high'
                    },
                    {
                      title: '更新嵌入模型',
                      description: '升级到最新的预训练模型，提高语义理解能力',
                      priority: 'medium',
                      impact: 'high'
                    },
                    {
                      title: '增强缓存策略',
                      description: '实现智能缓存预热，减少冷启动延迟',
                      priority: 'medium',
                      impact: 'medium'
                    },
                    {
                      title: '改进用户界面',
                      description: '基于用户反馈优化查询构建器的易用性',
                      priority: 'low',
                      impact: 'medium'
                    }
                  ]}
                  renderItem={(item) => (
                    <List.Item>
                      <List.Item.Meta
                        title={
                          <Space>
                            <Text strong>{item.title}</Text>
                            <Tag color={item.priority === 'high' ? 'red' : item.priority === 'medium' ? 'orange' : 'green'}>
                              {item.priority}
                            </Tag>
                            <Tag color={item.impact === 'high' ? 'purple' : 'blue'}>
                              影响: {item.impact}
                            </Tag>
                          </Space>
                        }
                        description={item.description}
                      />
                    </List.Item>
                  )}
                />
              </Card>
            </Col>
          </Row>
        </TabPane>
      </Tabs>

      {/* 结果详情抽屉 */}
      <Drawer
        title="推理结果详情"
        width={800}
        visible={detailDrawerVisible}
        onClose={() => setDetailDrawerVisible(false)}
      >
        {selectedResult && (
          <Space direction="vertical" style={{ width: '100%' }} size="large">
            <Card title="查询信息" size="small">
              <Row gutter={[16, 8]}>
                <Col span={8}><Text type="secondary">查询ID:</Text></Col>
                <Col span={16}><Text code>{selectedResult.id}</Text></Col>
                <Col span={8}><Text type="secondary">查询内容:</Text></Col>
                <Col span={16}><Text>{selectedResult.query}</Text></Col>
                <Col span={8}><Text type="secondary">推理策略:</Text></Col>
                <Col span={16}><Tag color="blue">{selectedResult.strategy}</Tag></Col>
                <Col span={8}><Text type="secondary">执行时间:</Text></Col>
                <Col span={16}><Text>{selectedResult.timestamp}</Text></Col>
              </Row>
            </Card>

            <Card title="推理结果" size="small">
              <List
                dataSource={selectedResult.results}
                renderItem={(item: any) => (
                  <List.Item>
                    <List.Item.Meta
                      title={item.entity || item.path?.join(' → ')}
                      description={
                        <Space>
                          <Progress 
                            percent={(item.confidence || item.similarity) * 100} 
                            size="small"
                            format={() => (item.confidence || item.similarity)?.toFixed(3)}
                          />
                          {item.type && <Tag>{item.type}</Tag>}
                        </Space>
                      }
                    />
                  </List.Item>
                )}
              />
            </Card>

            {selectedResult.explanation && (
              <Card title="推理解释" size="small">
                <Tabs size="small">
                  <TabPane tab="推理步骤" key="steps">
                    <Timeline>
                      {selectedResult.explanation.steps.map((step, index) => (
                        <Timeline.Item key={index}>{step}</Timeline.Item>
                      ))}
                    </Timeline>
                  </TabPane>
                  <TabPane tab="置信度分析" key="confidence">
                    {Object.entries(selectedResult.explanation.confidence_breakdown).map(([key, value]) => (
                      <div key={key} style={{ marginBottom: '16px' }}>
                        <Text strong>{key}:</Text>
                        <Progress 
                          percent={value * 100} 
                          strokeColor="#1890ff"
                          format={() => value.toFixed(3)}
                        />
                      </div>
                    ))}
                  </TabPane>
                  <TabPane tab="证据链" key="evidence">
                    {selectedResult.explanation.evidence.map((evidence, index) => (
                      <Alert 
                        key={index}
                        message={evidence} 
                        type="info" 
                        style={{ marginBottom: '8px' }}
                      />
                    ))}
                  </TabPane>
                </Tabs>
              </Card>
            )}

            {selectedResult.feedback && (
              <Card title="用户反馈" size="small">
                <Space direction="vertical" style={{ width: '100%' }}>
                  <div>
                    <Text strong>评分: </Text>
                    <Rate disabled value={selectedResult.feedback.rating} />
                    <Text style={{ marginLeft: '8px' }}>({selectedResult.feedback.rating}/5)</Text>
                  </div>
                  <div>
                    <Text strong>评论: </Text>
                    <Text>{selectedResult.feedback.comment}</Text>
                  </div>
                  <div>
                    <Text strong>有用性: </Text>
                    {selectedResult.feedback.helpful ? 
                      <Badge status="success" text="标记为有用" /> : 
                      <Badge status="default" text="未标记" />
                    }
                  </div>
                </Space>
              </Card>
            )}
          </Space>
        )}
      </Drawer>

      {/* 高级筛选对话框 */}
      <Modal
        title="高级筛选"
        visible={filterModalVisible}
        onCancel={() => setFilterModalVisible(false)}
        onOk={() => setFilterModalVisible(false)}
        width={600}
      >
        <Form layout="vertical">
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item label="时间范围">
                <RangePicker 
                  value={dateRange}
                  onChange={setDateRange}
                  style={{ width: '100%' }}
                />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="推理策略">
                <Select
                  mode="multiple"
                  value={selectedStrategies}
                  onChange={setSelectedStrategies}
                  style={{ width: '100%' }}
                >
                  <Option value="rule_only">规则推理</Option>
                  <Option value="embedding_only">嵌入推理</Option>
                  <Option value="path_only">路径推理</Option>
                  <Option value="ensemble">集成策略</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item label="置信度范围">
                <Slider
                  range
                  min={0}
                  max={1}
                  step={0.1}
                  value={confidenceRange}
                  onChange={setConfidenceRange}
                  marks={{ 0: '0', 0.5: '0.5', 1: '1' }}
                />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="仅显示有反馈的结果">
                <Switch />
              </Form.Item>
            </Col>
          </Row>
        </Form>
      </Modal>
    </div>
  )
}

export default KGReasoningAnalysisPage