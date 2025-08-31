import React, { useState, useEffect } from 'react'
import { Card, Row, Col, Typography, Space, Statistic, Progress, Table, Button, Tag, Timeline, Alert, Tabs, List, Avatar, Badge, Select, DatePicker, Input, Drawer, message, Descriptions, Radio, Slider } from 'antd'
import {
  LineChartOutlined,
  ThunderboltOutlined,
  AlertOutlined,
  RiseOutlined,
  FallOutlined,
  CalendarOutlined,
  ClockCircleOutlined,
  FireOutlined,
  ExclamationCircleOutlined,
  CheckCircleOutlined,
  WarningOutlined,
  FilterOutlined,
  ExportOutlined,
  AnalyticsOutlined,
  NodeIndexOutlined,
  ForkOutlined
} from '@ant-design/icons'
import ReactEcharts from 'echarts-for-react'
import type { ColumnsType } from 'antd/es/table'

const { Title, Text, Paragraph } = Typography
const { TabPane } = Tabs
const { RangePicker } = DatePicker

interface EmotionalEvent {
  id: string
  timestamp: string
  eventType: '突变' | '周期' | '异常' | '转折'
  startTime: string
  endTime: string
  duration: number
  peakIntensity: number
  emotionSequence: string[]
  causalFactors: string[]
  significance: number
  recoveryTime: number
  impactScope: string[]
  confidence: number
}

interface CausalRelation {
  source: string
  target: string
  strength: number
  confidence: number
  evidence: string[]
}

const EmotionalEventAnalysisPage: React.FC = () => {
  const [events, setEvents] = useState<EmotionalEvent[]>([])
  const [relations, setRelations] = useState<CausalRelation[]>([])
  const [selectedTimeRange, setSelectedTimeRange] = useState<[string, string]>(['', ''])
  const [eventTypeFilter, setEventTypeFilter] = useState<string>('all')
  const [detailDrawerVisible, setDetailDrawerVisible] = useState(false)
  const [selectedEvent, setSelectedEvent] = useState<EmotionalEvent | null>(null)

  useEffect(() => {
    // 模拟数据加载
    const mockEvents: EmotionalEvent[] = [
      {
        id: 'evt-001',
        timestamp: '2025-01-23T10:30:00Z',
        eventType: '突变',
        startTime: '2025-01-23T10:00:00Z',
        endTime: '2025-01-23T11:00:00Z',
        duration: 60,
        peakIntensity: 0.92,
        emotionSequence: ['紧张', '兴奋', '快乐', '满足'],
        causalFactors: ['项目完成', '客户认可', '团队庆祝'],
        significance: 0.88,
        recoveryTime: 30,
        impactScope: ['工作', '社交', '自信'],
        confidence: 0.85
      },
      {
        id: 'evt-002',
        timestamp: '2025-01-22T14:30:00Z',
        eventType: '周期',
        startTime: '2025-01-22T14:00:00Z',
        endTime: '2025-01-22T18:00:00Z',
        duration: 240,
        peakIntensity: 0.75,
        emotionSequence: ['疲劳', '焦虑', '压力', '缓解'],
        causalFactors: ['工作负荷', '截止期限', '下午疲劳期'],
        significance: 0.65,
        recoveryTime: 120,
        impactScope: ['工作效率', '情绪状态'],
        confidence: 0.78
      },
      {
        id: 'evt-003',
        timestamp: '2025-01-21T09:00:00Z',
        eventType: '转折',
        startTime: '2025-01-21T08:30:00Z',
        endTime: '2025-01-21T09:30:00Z',
        duration: 60,
        peakIntensity: 0.68,
        emotionSequence: ['沮丧', '思考', '顿悟', '乐观'],
        causalFactors: ['问题解决', '新视角', '灵感突现'],
        significance: 0.72,
        recoveryTime: 15,
        impactScope: ['创造力', '问题解决'],
        confidence: 0.82
      }
    ]

    const mockRelations: CausalRelation[] = [
      {
        source: '工作压力',
        target: '焦虑情绪',
        strength: 0.85,
        confidence: 0.9,
        evidence: ['历史数据', '模式匹配', '时间关联']
      },
      {
        source: '项目完成',
        target: '成就感',
        strength: 0.92,
        confidence: 0.95,
        evidence: ['直接因果', '情感记录', '用户反馈']
      }
    ]

    setEvents(mockEvents)
    setRelations(mockRelations)
  }, [])

  // 情感强度变化图表
  const emotionIntensityOption = {
    title: { text: '情感强度变化曲线', left: 'center' },
    tooltip: { trigger: 'axis' },
    legend: { bottom: '5%' },
    xAxis: {
      type: 'time',
      boundaryGap: false
    },
    yAxis: {
      type: 'value',
      name: '强度',
      max: 1,
      min: -1
    },
    series: [
      {
        name: '正面情感',
        type: 'line',
        smooth: true,
        data: [
          ['2025-01-20 08:00', 0.3],
          ['2025-01-20 12:00', 0.5],
          ['2025-01-21 08:00', 0.7],
          ['2025-01-21 12:00', 0.4],
          ['2025-01-22 08:00', 0.6],
          ['2025-01-22 12:00', 0.3],
          ['2025-01-23 08:00', 0.8],
          ['2025-01-23 12:00', 0.9]
        ],
        itemStyle: { color: '#52c41a' },
        areaStyle: { opacity: 0.3 }
      },
      {
        name: '负面情感',
        type: 'line',
        smooth: true,
        data: [
          ['2025-01-20 08:00', -0.2],
          ['2025-01-20 12:00', -0.1],
          ['2025-01-21 08:00', -0.3],
          ['2025-01-21 12:00', -0.5],
          ['2025-01-22 08:00', -0.4],
          ['2025-01-22 12:00', -0.6],
          ['2025-01-23 08:00', -0.2],
          ['2025-01-23 12:00', -0.1]
        ],
        itemStyle: { color: '#ff4d4f' },
        areaStyle: { opacity: 0.3 }
      }
    ],
    dataZoom: [
      {
        type: 'inside',
        start: 0,
        end: 100
      },
      {
        start: 0,
        end: 100
      }
    ]
  }

  // 因果关系网络图
  const causalNetworkOption = {
    title: { text: '情感因果关系网络', left: 'center' },
    tooltip: {},
    animationDurationUpdate: 1500,
    animationEasingUpdate: 'quinticInOut',
    series: [{
      type: 'graph',
      layout: 'force',
      data: [
        { name: '工作压力', symbolSize: 60, category: 0, value: 10 },
        { name: '焦虑', symbolSize: 50, category: 1, value: 8 },
        { name: '项目完成', symbolSize: 55, category: 0, value: 9 },
        { name: '快乐', symbolSize: 50, category: 1, value: 9 },
        { name: '团队合作', symbolSize: 45, category: 0, value: 7 },
        { name: '成就感', symbolSize: 50, category: 1, value: 8 },
        { name: '疲劳', symbolSize: 40, category: 1, value: 6 },
        { name: '休息', symbolSize: 35, category: 0, value: 5 },
        { name: '恢复', symbolSize: 40, category: 1, value: 6 }
      ],
      links: [
        { source: '工作压力', target: '焦虑', value: 0.85 },
        { source: '工作压力', target: '疲劳', value: 0.7 },
        { source: '项目完成', target: '快乐', value: 0.9 },
        { source: '项目完成', target: '成就感', value: 0.88 },
        { source: '团队合作', target: '成就感', value: 0.75 },
        { source: '休息', target: '恢复', value: 0.8 },
        { source: '疲劳', target: '休息', value: 0.6 }
      ],
      categories: [
        { name: '触发因素' },
        { name: '情感结果' }
      ],
      roam: true,
      label: {
        show: true,
        position: 'right',
        formatter: '{b}'
      },
      lineStyle: {
        color: 'source',
        curveness: 0.3
      },
      emphasis: {
        focus: 'adjacency',
        lineStyle: {
          width: 10
        }
      },
      force: {
        repulsion: 100,
        gravity: 0.1,
        edgeLength: 100
      }
    }]
  }

  // 事件表格列定义
  const eventColumns: ColumnsType<EmotionalEvent> = [
    {
      title: '时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      render: (text) => new Date(text).toLocaleString('zh-CN'),
      sorter: (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    },
    {
      title: '事件类型',
      dataIndex: 'eventType',
      key: 'eventType',
      render: (text) => {
        const colors = {
          '突变': 'red',
          '周期': 'blue',
          '异常': 'orange',
          '转折': 'green'
        }
        return <Tag color={colors[text as keyof typeof colors]}>{text}</Tag>
      },
      filters: [
        { text: '突变', value: '突变' },
        { text: '周期', value: '周期' },
        { text: '异常', value: '异常' },
        { text: '转折', value: '转折' }
      ],
      onFilter: (value, record) => record.eventType === value
    },
    {
      title: '持续时间',
      dataIndex: 'duration',
      key: 'duration',
      render: (val) => `${val}分钟`,
      sorter: (a, b) => a.duration - b.duration
    },
    {
      title: '峰值强度',
      dataIndex: 'peakIntensity',
      key: 'peakIntensity',
      render: (val) => <Progress percent={val * 100} size="small" />,
      sorter: (a, b) => a.peakIntensity - b.peakIntensity
    },
    {
      title: '重要性',
      dataIndex: 'significance',
      key: 'significance',
      render: (val) => (
        <Badge
          status={val > 0.8 ? "error" : val > 0.5 ? "warning" : "default"}
          text={`${(val * 100).toFixed(0)}%`}
        />
      )
    },
    {
      title: '置信度',
      dataIndex: 'confidence',
      key: 'confidence',
      render: (val) => `${(val * 100).toFixed(0)}%`
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record) => (
        <Space>
          <Button type="link" size="small" onClick={() => handleViewEvent(record)}>
            详情
          </Button>
          <Button type="link" size="small" onClick={() => handleAnalyzeEvent(record)}>
            分析
          </Button>
        </Space>
      )
    }
  ]

  const handleViewEvent = (event: EmotionalEvent) => {
    setSelectedEvent(event)
    setDetailDrawerVisible(true)
  }

  const handleAnalyzeEvent = (event: EmotionalEvent) => {
    message.info(`正在深度分析事件 ${event.id}...`)
  }

  return (
    <div style={{ padding: '24px' }}>
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        {/* 页面标题 */}
        <Card>
          <Row justify="space-between" align="middle">
            <Col>
              <Space>
                <LineChartOutlined style={{ fontSize: 32, color: '#1890ff' }} />
                <div>
                  <Title level={3} style={{ margin: 0 }}>情感事件分析引擎</Title>
                  <Text type="secondary">自动识别重要情感事件和因果关系</Text>
                </div>
              </Space>
            </Col>
            <Col>
              <Space>
                <Button icon={<FilterOutlined />}>高级筛选</Button>
                <Button icon={<ExportOutlined />}>导出报告</Button>
                <Button type="primary" icon={<AnalyticsOutlined />}>
                  深度分析
                </Button>
              </Space>
            </Col>
          </Row>
        </Card>

        {/* 核心指标 */}
        <Row gutter={[16, 16]}>
          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="识别事件"
                value={156}
                prefix={<FireOutlined />}
                suffix="个"
                valueStyle={{ color: '#1890ff' }}
              />
              <Text type="secondary">本月累计</Text>
            </Card>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="重要事件"
                value={42}
                prefix={<ExclamationCircleOutlined />}
                suffix="个"
                valueStyle={{ color: '#ff4d4f' }}
              />
              <Space>
                <RiseOutlined style={{ color: '#52c41a' }} />
                <Text type="secondary">较上月 +15%</Text>
              </Space>
            </Card>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="因果关联"
                value={89}
                prefix={<NodeIndexOutlined />}
                suffix="%"
                valueStyle={{ color: '#52c41a' }}
              />
              <Text type="secondary">关联准确率</Text>
            </Card>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="分析延迟"
                value={120}
                prefix={<ThunderboltOutlined />}
                suffix="ms"
                valueStyle={{ color: '#faad14' }}
              />
              <Text type="secondary">平均响应</Text>
            </Card>
          </Col>
        </Row>

        {/* 主要内容区 */}
        <Card>
          <Tabs defaultActiveKey="timeline">
            <TabPane tab="事件时间线" key="timeline">
              <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                {/* 筛选栏 */}
                <Row gutter={16}>
                  <Col span={6}>
                    <Select
                      style={{ width: '100%' }}
                      placeholder="事件类型"
                      value={eventTypeFilter}
                      onChange={setEventTypeFilter}
                    >
                      <Select.Option value="all">全部类型</Select.Option>
                      <Select.Option value="突变">情感突变</Select.Option>
                      <Select.Option value="周期">周期模式</Select.Option>
                      <Select.Option value="异常">异常波动</Select.Option>
                      <Select.Option value="转折">情感转折</Select.Option>
                    </Select>
                  </Col>
                  <Col span={8}>
                    <RangePicker style={{ width: '100%' }} />
                  </Col>
                  <Col span={6}>
                    <Slider
                      range
                      defaultValue={[0.5, 1.0]}
                      min={0}
                      max={1}
                      step={0.1}
                      marks={{
                        0: '低',
                        0.5: '中',
                        1: '高'
                      }}
                    />
                    <Text type="secondary">重要性筛选</Text>
                  </Col>
                </Row>

                {/* 时间线展示 */}
                <Timeline mode="alternate">
                  {events.map((event, index) => (
                    <Timeline.Item
                      key={event.id}
                      color={event.significance > 0.7 ? 'red' : 'blue'}
                      label={new Date(event.timestamp).toLocaleDateString()}
                      dot={
                        event.eventType === '突变' ? <ThunderboltOutlined /> :
                        event.eventType === '异常' ? <WarningOutlined /> :
                        event.eventType === '转折' ? <ForkOutlined /> :
                        <ClockCircleOutlined />
                      }
                    >
                      <Card size="small" hoverable onClick={() => handleViewEvent(event)}>
                        <Space direction="vertical" size="small">
                          <Row justify="space-between">
                            <Text strong>{event.eventType}事件</Text>
                            <Badge
                              status={event.significance > 0.7 ? "error" : "warning"}
                              text={`重要性: ${(event.significance * 100).toFixed(0)}%`}
                            />
                          </Row>
                          <Text>情感序列: {event.emotionSequence.join(' → ')}</Text>
                          <Text type="secondary">
                            峰值强度: {(event.peakIntensity * 100).toFixed(0)}% | 
                            持续: {event.duration}分钟 | 
                            恢复: {event.recoveryTime}分钟
                          </Text>
                          <Space wrap>
                            {event.causalFactors.map(factor => (
                              <Tag key={factor} color="blue">{factor}</Tag>
                            ))}
                          </Space>
                        </Space>
                      </Card>
                    </Timeline.Item>
                  ))}
                </Timeline>
              </Space>
            </TabPane>

            <TabPane tab="强度分析" key="intensity">
              <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                <ReactEcharts option={emotionIntensityOption} style={{ height: 400 }} />
                
                <Card title="情感波动统计" size="small">
                  <Row gutter={16}>
                    <Col span={8}>
                      <Statistic
                        title="平均强度"
                        value={0.65}
                        precision={2}
                        valueStyle={{ color: '#3f8600' }}
                        prefix={<RiseOutlined />}
                      />
                    </Col>
                    <Col span={8}>
                      <Statistic
                        title="波动幅度"
                        value={0.42}
                        precision={2}
                        valueStyle={{ color: '#faad14' }}
                      />
                    </Col>
                    <Col span={8}>
                      <Statistic
                        title="稳定性指数"
                        value={78}
                        suffix="%"
                        valueStyle={{ color: '#1890ff' }}
                      />
                    </Col>
                  </Row>
                </Card>
              </Space>
            </TabPane>

            <TabPane tab="因果分析" key="causal">
              <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                <Alert
                  message="因果关系自动发现"
                  description="系统已识别 89 对因果关系，置信度大于 70% 的有 56 对"
                  type="info"
                  showIcon
                />
                
                <ReactEcharts option={causalNetworkOption} style={{ height: 500 }} />
                
                <Card title="高置信度因果关系" size="small">
                  <List
                    dataSource={relations}
                    renderItem={(item) => (
                      <List.Item>
                        <List.Item.Meta
                          avatar={<Avatar icon={<NodeIndexOutlined />} />}
                          title={
                            <Space>
                              <Text>{item.source}</Text>
                              <Text type="secondary">→</Text>
                              <Text strong>{item.target}</Text>
                            </Space>
                          }
                          description={
                            <Space direction="vertical" size="small">
                              <Row gutter={16}>
                                <Col span={8}>
                                  <Text type="secondary">关联强度: </Text>
                                  <Progress
                                    percent={item.strength * 100}
                                    size="small"
                                    strokeColor="#52c41a"
                                  />
                                </Col>
                                <Col span={8}>
                                  <Text type="secondary">置信度: </Text>
                                  <Progress
                                    percent={item.confidence * 100}
                                    size="small"
                                    strokeColor="#1890ff"
                                  />
                                </Col>
                              </Row>
                              <Space wrap>
                                {item.evidence.map(e => (
                                  <Tag key={e} color="geekblue">{e}</Tag>
                                ))}
                              </Space>
                            </Space>
                          }
                        />
                      </List.Item>
                    )}
                  />
                </Card>
              </Space>
            </TabPane>

            <TabPane tab="事件列表" key="list">
              <Table
                columns={eventColumns}
                dataSource={events}
                rowKey="id"
                pagination={{ pageSize: 10 }}
              />
            </TabPane>
          </Tabs>
        </Card>

        {/* 详情抽屉 */}
        <Drawer
          title="事件详细分析"
          placement="right"
          width={600}
          onClose={() => setDetailDrawerVisible(false)}
          visible={detailDrawerVisible}
        >
          {selectedEvent && (
            <Descriptions bordered column={1} size="small">
              <Descriptions.Item label="事件ID">{selectedEvent.id}</Descriptions.Item>
              <Descriptions.Item label="事件类型">
                <Tag color="blue">{selectedEvent.eventType}</Tag>
              </Descriptions.Item>
              <Descriptions.Item label="开始时间">
                {new Date(selectedEvent.startTime).toLocaleString('zh-CN')}
              </Descriptions.Item>
              <Descriptions.Item label="结束时间">
                {new Date(selectedEvent.endTime).toLocaleString('zh-CN')}
              </Descriptions.Item>
              <Descriptions.Item label="持续时间">{selectedEvent.duration} 分钟</Descriptions.Item>
              <Descriptions.Item label="峰值强度">
                <Progress percent={selectedEvent.peakIntensity * 100} />
              </Descriptions.Item>
              <Descriptions.Item label="情感序列">
                {selectedEvent.emotionSequence.join(' → ')}
              </Descriptions.Item>
              <Descriptions.Item label="因果因素">
                <Space wrap>
                  {selectedEvent.causalFactors.map(factor => (
                    <Tag key={factor}>{factor}</Tag>
                  ))}
                </Space>
              </Descriptions.Item>
              <Descriptions.Item label="影响范围">
                <Space wrap>
                  {selectedEvent.impactScope.map(scope => (
                    <Tag key={scope} color="green">{scope}</Tag>
                  ))}
                </Space>
              </Descriptions.Item>
              <Descriptions.Item label="重要性">
                <Badge
                  status={selectedEvent.significance > 0.8 ? "error" : "warning"}
                  text={`${(selectedEvent.significance * 100).toFixed(0)}%`}
                />
              </Descriptions.Item>
              <Descriptions.Item label="恢复时间">{selectedEvent.recoveryTime} 分钟</Descriptions.Item>
              <Descriptions.Item label="置信度">{(selectedEvent.confidence * 100).toFixed(0)}%</Descriptions.Item>
            </Descriptions>
          )}
        </Drawer>
      </Space>
    </div>
  )
}

export default EmotionalEventAnalysisPage