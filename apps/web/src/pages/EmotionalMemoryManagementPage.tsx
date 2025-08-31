import React, { useState, useEffect } from 'react'
import { Card, Row, Col, Typography, Space, Statistic, Progress, Table, Button, Tag, Timeline, Alert, Tabs, List, Avatar, Tooltip, Badge, Divider, Select, DatePicker, Input, Switch, Modal, message, Descriptions } from 'antd'
import { 
  BrainOutlined, 
  HeartOutlined, 
  ClockCircleOutlined, 
  DatabaseOutlined, 
  SearchOutlined,
  LineChartOutlined,
  UserOutlined,
  CalendarOutlined,
  TagOutlined,
  LinkOutlined,
  SafetyOutlined,
  ThunderboltOutlined,
  CloudServerOutlined,
  FireOutlined,
  RiseOutlined,
  FallOutlined,
  ExperimentOutlined,
  HistoryOutlined,
  FilterOutlined,
  ExportOutlined,
  ImportOutlined,
  SettingOutlined,
  InfoCircleOutlined,
  DeleteOutlined,
  AimOutlined,
  AlertOutlined,
  NetworkOutlined
} from '@ant-design/icons'
import ReactEcharts from 'echarts-for-react'
import type { ColumnsType } from 'antd/es/table'

const { Title, Text, Paragraph } = Typography
const { TabPane } = Tabs
const { RangePicker } = DatePicker
const { Search } = Input

interface EmotionalMemory {
  id: string
  timestamp: string
  emotionType: string
  intensity: number
  context: string
  triggerFactors: string[]
  importance: number
  relatedMemories: string[]
  storageLayer: 'hot' | 'warm' | 'cold'
  tags: string[]
}

interface EmotionalEvent {
  id: string
  eventType: string
  startTime: string
  endTime: string
  peakIntensity: number
  emotionSequence: string[]
  causalFactors: string[]
  significance: number
  recoveryTime: number
}

interface UserPreference {
  category: string
  preference: string
  confidence: number
  lastUpdated: string
  effectiveness: number
}

interface TriggerPattern {
  id: string
  patternType: string
  triggerConditions: Record<string, any>
  frequency: number
  confidence: number
  lastOccurred: string
  predictionAccuracy: number
}

const EmotionalMemoryManagementPage: React.FC = () => {
  const [memories, setMemories] = useState<EmotionalMemory[]>([])
  const [events, setEvents] = useState<EmotionalEvent[]>([])
  const [preferences, setPreferences] = useState<UserPreference[]>([])
  const [patterns, setPatterns] = useState<TriggerPattern[]>([])
  const [loading, setLoading] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedLayer, setSelectedLayer] = useState<string>('all')
  const [privacyMode, setPrivacyMode] = useState(false)

  // 模拟数据
  useEffect(() => {
    const mockMemories: EmotionalMemory[] = [
      {
        id: 'mem-001',
        timestamp: '2025-01-23T10:30:00Z',
        emotionType: '快乐',
        intensity: 0.85,
        context: '完成重要项目里程碑',
        triggerFactors: ['成就', '团队合作'],
        importance: 0.9,
        relatedMemories: ['mem-002', 'mem-003'],
        storageLayer: 'hot',
        tags: ['工作', '成功', '团队']
      },
      {
        id: 'mem-002',
        timestamp: '2025-01-22T15:20:00Z',
        emotionType: '焦虑',
        intensity: 0.65,
        context: '项目截止日期临近',
        triggerFactors: ['时间压力', '责任'],
        importance: 0.7,
        relatedMemories: ['mem-001'],
        storageLayer: 'warm',
        tags: ['工作', '压力']
      },
      {
        id: 'mem-003',
        timestamp: '2025-01-20T09:00:00Z',
        emotionType: '平静',
        intensity: 0.4,
        context: '早晨冥想练习',
        triggerFactors: ['例行活动', '自我关怀'],
        importance: 0.5,
        relatedMemories: [],
        storageLayer: 'cold',
        tags: ['健康', '日常']
      }
    ]

    const mockEvents: EmotionalEvent[] = [
      {
        id: 'evt-001',
        eventType: '情感突变',
        startTime: '2025-01-23T10:00:00Z',
        endTime: '2025-01-23T11:00:00Z',
        peakIntensity: 0.9,
        emotionSequence: ['紧张', '兴奋', '快乐'],
        causalFactors: ['项目完成', '获得认可'],
        significance: 0.85,
        recoveryTime: 30
      },
      {
        id: 'evt-002',
        eventType: '情感周期',
        startTime: '2025-01-22T14:00:00Z',
        endTime: '2025-01-22T18:00:00Z',
        peakIntensity: 0.7,
        emotionSequence: ['压力', '焦虑', '疲惫'],
        causalFactors: ['工作负荷', '时间限制'],
        significance: 0.6,
        recoveryTime: 120
      }
    ]

    const mockPreferences: UserPreference[] = [
      {
        category: '情感支持',
        preference: '积极鼓励',
        confidence: 0.85,
        lastUpdated: '2025-01-23',
        effectiveness: 0.9
      },
      {
        category: '沟通风格',
        preference: '直接简洁',
        confidence: 0.78,
        lastUpdated: '2025-01-22',
        effectiveness: 0.82
      },
      {
        category: '安慰方式',
        preference: '理性分析',
        confidence: 0.72,
        lastUpdated: '2025-01-21',
        effectiveness: 0.75
      }
    ]

    const mockPatterns: TriggerPattern[] = [
      {
        id: 'pat-001',
        patternType: '工作压力触发',
        triggerConditions: {
          timeOfDay: '14:00-18:00',
          workload: 'high',
          deadline: '<2days'
        },
        frequency: 0.65,
        confidence: 0.82,
        lastOccurred: '2025-01-22',
        predictionAccuracy: 0.78
      },
      {
        id: 'pat-002',
        patternType: '成就快乐触发',
        triggerConditions: {
          achievement: 'completed',
          teamWork: true,
          recognition: true
        },
        frequency: 0.45,
        confidence: 0.88,
        lastOccurred: '2025-01-23',
        predictionAccuracy: 0.85
      }
    ]

    setMemories(mockMemories)
    setEvents(mockEvents)
    setPreferences(mockPreferences)
    setPatterns(mockPatterns)
  }, [])

  // 存储层分布图表配置
  const storageDistributionOption = {
    title: { text: '记忆存储层分布', left: 'center' },
    tooltip: { trigger: 'item' },
    legend: { bottom: '5%' },
    series: [{
      type: 'pie',
      radius: ['40%', '70%'],
      data: [
        { value: 35, name: '热存储 (7天内)', itemStyle: { color: '#ff4d4f' } },
        { value: 45, name: '温存储 (6个月内)', itemStyle: { color: '#ffa940' } },
        { value: 20, name: '冷存储 (长期)', itemStyle: { color: '#1890ff' } }
      ]
    }]
  }

  // 情感强度时间线图表配置
  const emotionTimelineOption = {
    title: { text: '情感强度时间线', left: 'center' },
    tooltip: { trigger: 'axis' },
    xAxis: { 
      type: 'category',
      data: ['1月18日', '1月19日', '1月20日', '1月21日', '1月22日', '1月23日']
    },
    yAxis: { type: 'value', name: '强度', max: 1 },
    series: [
      {
        name: '快乐',
        type: 'line',
        smooth: true,
        data: [0.3, 0.4, 0.5, 0.6, 0.7, 0.85],
        itemStyle: { color: '#52c41a' }
      },
      {
        name: '焦虑',
        type: 'line',
        smooth: true,
        data: [0.2, 0.25, 0.35, 0.45, 0.65, 0.4],
        itemStyle: { color: '#faad14' }
      },
      {
        name: '平静',
        type: 'line',
        smooth: true,
        data: [0.6, 0.5, 0.4, 0.35, 0.3, 0.35],
        itemStyle: { color: '#1890ff' }
      }
    ]
  }

  // 触发模式热力图配置
  const triggerHeatmapOption = {
    title: { text: '情感触发模式热力图', left: 'center' },
    tooltip: { position: 'top' },
    grid: { height: '50%', top: '10%' },
    xAxis: {
      type: 'category',
      data: ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    },
    yAxis: {
      type: 'category',
      data: ['早晨', '上午', '中午', '下午', '傍晚', '夜晚']
    },
    visualMap: {
      min: 0,
      max: 10,
      calculable: true,
      orient: 'horizontal',
      left: 'center',
      bottom: '5%'
    },
    series: [{
      name: '触发频率',
      type: 'heatmap',
      data: [
        [0, 0, 5], [1, 0, 3], [2, 0, 4], [3, 0, 6], [4, 0, 7], [5, 0, 2], [6, 0, 1],
        [0, 1, 7], [1, 1, 8], [2, 1, 9], [3, 1, 8], [4, 1, 9], [5, 1, 3], [6, 1, 2],
        [0, 2, 3], [1, 2, 4], [2, 2, 5], [3, 2, 4], [4, 2, 5], [5, 2, 2], [6, 2, 1],
        [0, 3, 8], [1, 3, 9], [2, 3, 10], [3, 3, 9], [4, 3, 10], [5, 3, 4], [6, 3, 3],
        [0, 4, 6], [1, 4, 7], [2, 4, 8], [3, 4, 7], [4, 4, 8], [5, 4, 3], [6, 4, 2],
        [0, 5, 2], [1, 5, 3], [2, 5, 4], [3, 5, 3], [4, 5, 4], [5, 5, 1], [6, 5, 1]
      ].map(function (item) {
        return [item[0], item[1], item[2] || '-']
      })
    }]
  }

  // 表格列定义
  const memoryColumns: ColumnsType<EmotionalMemory> = [
    {
      title: '时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      render: (text) => new Date(text).toLocaleString('zh-CN'),
      sorter: (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    },
    {
      title: '情感类型',
      dataIndex: 'emotionType',
      key: 'emotionType',
      render: (text) => <Tag color="blue">{text}</Tag>
    },
    {
      title: '强度',
      dataIndex: 'intensity',
      key: 'intensity',
      render: (val) => <Progress percent={val * 100} size="small" />,
      sorter: (a, b) => a.intensity - b.intensity
    },
    {
      title: '重要性',
      dataIndex: 'importance',
      key: 'importance',
      render: (val) => (
        <Badge 
          status={val > 0.8 ? "error" : val > 0.5 ? "warning" : "default"} 
          text={`${(val * 100).toFixed(0)}%`} 
        />
      )
    },
    {
      title: '存储层',
      dataIndex: 'storageLayer',
      key: 'storageLayer',
      render: (text) => {
        const colors = { hot: 'red', warm: 'orange', cold: 'blue' }
        return <Tag color={colors[text as keyof typeof colors]}>{text}</Tag>
      }
    },
    {
      title: '标签',
      dataIndex: 'tags',
      key: 'tags',
      render: (tags: string[]) => (
        <Space size={[0, 8]} wrap>
          {tags.map(tag => <Tag key={tag}>{tag}</Tag>)}
        </Space>
      )
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record) => (
        <Space>
          <Button type="link" size="small" onClick={() => handleViewMemory(record)}>
            查看
          </Button>
          <Button type="link" size="small" onClick={() => handleAnalyzeMemory(record)}>
            分析
          </Button>
        </Space>
      )
    }
  ]

  const handleViewMemory = (memory: EmotionalMemory) => {
    Modal.info({
      title: '情感记忆详情',
      width: 600,
      content: (
        <div>
          <Descriptions bordered size="small">
            <Descriptions.Item label="记忆ID" span={3}>{memory.id}</Descriptions.Item>
            <Descriptions.Item label="时间戳" span={3}>
              {new Date(memory.timestamp).toLocaleString('zh-CN')}
            </Descriptions.Item>
            <Descriptions.Item label="情感类型" span={3}>
              <Tag color="blue">{memory.emotionType}</Tag>
            </Descriptions.Item>
            <Descriptions.Item label="强度" span={3}>
              <Progress percent={memory.intensity * 100} />
            </Descriptions.Item>
            <Descriptions.Item label="上下文" span={3}>{memory.context}</Descriptions.Item>
            <Descriptions.Item label="触发因素" span={3}>
              {memory.triggerFactors.join(', ')}
            </Descriptions.Item>
            <Descriptions.Item label="关联记忆" span={3}>
              {memory.relatedMemories.join(', ') || '无'}
            </Descriptions.Item>
          </Descriptions>
        </div>
      )
    })
  }

  const handleAnalyzeMemory = (memory: EmotionalMemory) => {
    message.info(`正在分析记忆 ${memory.id} 的模式和关联...`)
  }

  const handleSearchMemories = (value: string) => {
    setSearchQuery(value)
    message.info(`搜索记忆: ${value}`)
  }

  const handleExportMemories = () => {
    message.success('正在导出情感记忆数据...')
  }

  const handlePrivacyToggle = (checked: boolean) => {
    setPrivacyMode(checked)
    message.info(checked ? '隐私模式已开启' : '隐私模式已关闭')
  }

  return (
    <div style={{ padding: '24px' }}>
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        {/* 页面标题 */}
        <Card>
          <Row justify="space-between" align="middle">
            <Col>
              <Space>
                <BrainOutlined style={{ fontSize: 32, color: '#1890ff' }} />
                <div>
                  <Title level={3} style={{ margin: 0 }}>情感记忆管理系统</Title>
                  <Text type="secondary">长期情感记忆存储、检索和模式分析</Text>
                </div>
              </Space>
            </Col>
            <Col>
              <Space>
                <Switch
                  checkedChildren="隐私"
                  unCheckedChildren="公开"
                  checked={privacyMode}
                  onChange={handlePrivacyToggle}
                />
                <Button icon={<ExportOutlined />} onClick={handleExportMemories}>
                  导出数据
                </Button>
                <Button type="primary" icon={<SettingOutlined />}>
                  系统设置
                </Button>
              </Space>
            </Col>
          </Row>
        </Card>

        {/* 核心指标卡片 */}
        <Row gutter={[16, 16]}>
          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="总记忆数量"
                value={1256}
                prefix={<DatabaseOutlined />}
                suffix="条"
                valueStyle={{ color: '#1890ff' }}
              />
              <Progress percent={78} size="small" />
              <Text type="secondary">存储使用率</Text>
            </Card>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="重要事件"
                value={42}
                prefix={<FireOutlined />}
                suffix="个"
                valueStyle={{ color: '#ff4d4f' }}
              />
              <Space style={{ marginTop: 8 }}>
                <RiseOutlined style={{ color: '#52c41a' }} />
                <Text type="secondary">本周新增 12</Text>
              </Space>
            </Card>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="模式识别"
                value={85}
                prefix={<ExperimentOutlined />}
                suffix="%"
                valueStyle={{ color: '#52c41a' }}
              />
              <Space style={{ marginTop: 8 }}>
                <Text type="secondary">准确率</Text>
                <Tooltip title="基于历史验证数据">
                  <InfoCircleOutlined />
                </Tooltip>
              </Space>
            </Card>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="检索响应"
                value={85}
                prefix={<ThunderboltOutlined />}
                suffix="ms"
                valueStyle={{ color: '#faad14' }}
              />
              <Text type="secondary">平均延迟</Text>
            </Card>
          </Col>
        </Row>

        {/* 主要功能标签页 */}
        <Card>
          <Tabs defaultActiveKey="memories">
            <TabPane 
              tab={<span><DatabaseOutlined /> 记忆存储</span>} 
              key="memories"
            >
              <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                {/* 搜索和筛选栏 */}
                <Row gutter={16}>
                  <Col span={8}>
                    <Search
                      placeholder="搜索情感记忆..."
                      onSearch={handleSearchMemories}
                      enterButton
                    />
                  </Col>
                  <Col span={4}>
                    <Select
                      style={{ width: '100%' }}
                      placeholder="存储层"
                      value={selectedLayer}
                      onChange={setSelectedLayer}
                    >
                      <Select.Option value="all">全部</Select.Option>
                      <Select.Option value="hot">热存储</Select.Option>
                      <Select.Option value="warm">温存储</Select.Option>
                      <Select.Option value="cold">冷存储</Select.Option>
                    </Select>
                  </Col>
                  <Col span={6}>
                    <RangePicker style={{ width: '100%' }} />
                  </Col>
                  <Col span={6}>
                    <Space>
                      <Button icon={<FilterOutlined />}>高级筛选</Button>
                      <Button icon={<ImportOutlined />}>导入</Button>
                    </Space>
                  </Col>
                </Row>

                {/* 记忆表格 */}
                <Table
                  columns={memoryColumns}
                  dataSource={memories}
                  rowKey="id"
                  pagination={{ pageSize: 5 }}
                />

                {/* 存储分布图 */}
                <Row gutter={16}>
                  <Col span={12}>
                    <ReactEcharts option={storageDistributionOption} style={{ height: 300 }} />
                  </Col>
                  <Col span={12}>
                    <ReactEcharts option={emotionTimelineOption} style={{ height: 300 }} />
                  </Col>
                </Row>
              </Space>
            </TabPane>

            <TabPane 
              tab={<span><LineChartOutlined /> 事件分析</span>} 
              key="events"
            >
              <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                <Alert
                  message="情感事件自动识别"
                  description="系统已识别 42 个重要情感事件，包括情感突变、周期性模式和异常波动"
                  type="info"
                  showIcon
                />

                {/* 事件时间线 */}
                <Card title="重要情感事件时间线" size="small">
                  <Timeline mode="alternate">
                    {events.map(event => (
                      <Timeline.Item
                        key={event.id}
                        color={event.significance > 0.7 ? 'red' : 'blue'}
                        label={new Date(event.startTime).toLocaleDateString()}
                      >
                        <Card size="small">
                          <Space direction="vertical" size="small">
                            <Text strong>{event.eventType}</Text>
                            <Text>强度峰值: {(event.peakIntensity * 100).toFixed(0)}%</Text>
                            <Text>情感序列: {event.emotionSequence.join(' → ')}</Text>
                            <Text type="secondary">恢复时间: {event.recoveryTime}分钟</Text>
                          </Space>
                        </Card>
                      </Timeline.Item>
                    ))}
                  </Timeline>
                </Card>

                {/* 因果关系分析 */}
                <Card title="因果关系网络" size="small">
                  <ReactEcharts
                    option={{
                      tooltip: {},
                      series: [{
                        type: 'graph',
                        layout: 'force',
                        data: [
                          { name: '工作压力', symbolSize: 50, category: 0 },
                          { name: '焦虑', symbolSize: 40, category: 1 },
                          { name: '项目完成', symbolSize: 45, category: 0 },
                          { name: '快乐', symbolSize: 40, category: 1 },
                          { name: '团队合作', symbolSize: 35, category: 0 },
                          { name: '成就感', symbolSize: 40, category: 1 }
                        ],
                        links: [
                          { source: '工作压力', target: '焦虑' },
                          { source: '项目完成', target: '快乐' },
                          { source: '团队合作', target: '成就感' },
                          { source: '成就感', target: '快乐' }
                        ],
                        categories: [
                          { name: '触发因素' },
                          { name: '情感结果' }
                        ],
                        roam: true,
                        force: {
                          repulsion: 100
                        }
                      }]
                    }}
                    style={{ height: 400 }}
                  />
                </Card>
              </Space>
            </TabPane>

            <TabPane 
              tab={<span><UserOutlined /> 偏好学习</span>} 
              key="preferences"
            >
              <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                <Alert
                  message="个人情感偏好已更新"
                  description="基于最近30天的交互数据，系统已学习并更新您的情感支持偏好"
                  type="success"
                  showIcon
                />

                {/* 偏好列表 */}
                <List
                  itemLayout="horizontal"
                  dataSource={preferences}
                  renderItem={(item) => (
                    <List.Item
                      actions={[
                        <Progress
                          type="circle"
                          percent={item.confidence * 100}
                          width={50}
                          format={(percent) => `${percent?.toFixed(0)}%`}
                        />
                      ]}
                    >
                      <List.Item.Meta
                        avatar={<Avatar icon={<HeartOutlined />} />}
                        title={item.category}
                        description={
                          <Space direction="vertical" size="small">
                            <Text>{item.preference}</Text>
                            <Space>
                              <Text type="secondary">有效性: {(item.effectiveness * 100).toFixed(0)}%</Text>
                              <Divider type="vertical" />
                              <Text type="secondary">更新: {item.lastUpdated}</Text>
                            </Space>
                          </Space>
                        }
                      />
                    </List.Item>
                  )}
                />

                {/* 偏好词云 */}
                <Card title="情感表达词汇偏好" size="small">
                  <div style={{ textAlign: 'center', padding: '20px' }}>
                    <Space wrap size={[8, 16]}>
                      {['积极', '鼓励', '理性', '简洁', '直接', '温暖', '支持', '共情'].map((word, index) => (
                        <Tag
                          key={word}
                          color={index % 2 === 0 ? 'blue' : 'green'}
                          style={{ fontSize: 14 + Math.random() * 8, padding: '4px 12px' }}
                        >
                          {word}
                        </Tag>
                      ))}
                    </Space>
                  </div>
                </Card>
              </Space>
            </TabPane>

            <TabPane 
              tab={<span><ExperimentOutlined /> 触发模式</span>} 
              key="patterns"
            >
              <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                {/* 模式识别统计 */}
                <Row gutter={16}>
                  <Col span={8}>
                    <Card size="small">
                      <Statistic
                        title="已识别模式"
                        value={patterns.length}
                        prefix={<TagOutlined />}
                      />
                    </Card>
                  </Col>
                  <Col span={8}>
                    <Card size="small">
                      <Statistic
                        title="预测准确率"
                        value={78}
                        suffix="%"
                        prefix={<AimOutlined />}
                      />
                    </Card>
                  </Col>
                  <Col span={8}>
                    <Card size="small">
                      <Statistic
                        title="风险预警"
                        value={3}
                        valueStyle={{ color: '#ff4d4f' }}
                        prefix={<AlertOutlined />}
                      />
                    </Card>
                  </Col>
                </Row>

                {/* 触发模式热力图 */}
                <Card title="触发模式时空分布" size="small">
                  <ReactEcharts option={triggerHeatmapOption} style={{ height: 400 }} />
                </Card>

                {/* 模式详情卡片 */}
                <Row gutter={16}>
                  {patterns.map(pattern => (
                    <Col span={12} key={pattern.id}>
                      <Card 
                        title={pattern.patternType}
                        extra={
                          <Badge 
                            status={pattern.confidence > 0.8 ? "success" : "warning"} 
                            text={`置信度 ${(pattern.confidence * 100).toFixed(0)}%`}
                          />
                        }
                        size="small"
                      >
                        <Space direction="vertical" size="small" style={{ width: '100%' }}>
                          <Row justify="space-between">
                            <Text>发生频率</Text>
                            <Text strong>{(pattern.frequency * 100).toFixed(0)}%</Text>
                          </Row>
                          <Row justify="space-between">
                            <Text>预测准确</Text>
                            <Text strong>{(pattern.predictionAccuracy * 100).toFixed(0)}%</Text>
                          </Row>
                          <Row justify="space-between">
                            <Text>最近发生</Text>
                            <Text strong>{pattern.lastOccurred}</Text>
                          </Row>
                          <Divider style={{ margin: '8px 0' }} />
                          <Text type="secondary">触发条件:</Text>
                          <div>
                            {Object.entries(pattern.triggerConditions).map(([key, value]) => (
                              <Tag key={key} color="blue" style={{ marginBottom: 4 }}>
                                {key}: {String(value)}
                              </Tag>
                            ))}
                          </div>
                        </Space>
                      </Card>
                    </Col>
                  ))}
                </Row>
              </Space>
            </TabPane>

            <TabPane 
              tab={<span><SearchOutlined /> 记忆检索</span>} 
              key="retrieval"
            >
              <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                <Alert
                  message="语义检索引擎"
                  description="使用自然语言查询历史情感记忆，支持模糊匹配和关联搜索"
                  type="info"
                  showIcon
                />

                {/* 高级检索界面 */}
                <Card title="智能记忆检索" size="small">
                  <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                    <Input.TextArea
                      placeholder="输入您想要查询的情感记忆描述，例如：'上周的快乐时刻' 或 '工作相关的焦虑情绪'"
                      rows={3}
                    />
                    <Row gutter={16}>
                      <Col span={6}>
                        <Select style={{ width: '100%' }} placeholder="情感类型">
                          <Select.Option value="all">全部</Select.Option>
                          <Select.Option value="happy">快乐</Select.Option>
                          <Select.Option value="sad">悲伤</Select.Option>
                          <Select.Option value="anxious">焦虑</Select.Option>
                          <Select.Option value="calm">平静</Select.Option>
                        </Select>
                      </Col>
                      <Col span={6}>
                        <Select style={{ width: '100%' }} placeholder="重要性">
                          <Select.Option value="all">全部</Select.Option>
                          <Select.Option value="high">高</Select.Option>
                          <Select.Option value="medium">中</Select.Option>
                          <Select.Option value="low">低</Select.Option>
                        </Select>
                      </Col>
                      <Col span={6}>
                        <RangePicker style={{ width: '100%' }} />
                      </Col>
                      <Col span={6}>
                        <Button type="primary" icon={<SearchOutlined />} block>
                          语义搜索
                        </Button>
                      </Col>
                    </Row>
                  </Space>
                </Card>

                {/* 检索结果展示 */}
                <Card title="检索结果 (相关度排序)" size="small">
                  <List
                    itemLayout="vertical"
                    dataSource={memories.slice(0, 3)}
                    renderItem={(item) => (
                      <List.Item
                        key={item.id}
                        actions={[
                          <Space>
                            <Text type="secondary">相关度: 92%</Text>
                            <LinkOutlined />
                          </Space>
                        ]}
                      >
                        <List.Item.Meta
                          avatar={<Avatar icon={<HeartOutlined />} style={{ backgroundColor: '#1890ff' }} />}
                          title={
                            <Space>
                              <Text>{new Date(item.timestamp).toLocaleString('zh-CN')}</Text>
                              <Tag color="blue">{item.emotionType}</Tag>
                            </Space>
                          }
                          description={item.context}
                        />
                        <Space wrap>
                          {item.tags.map(tag => <Tag key={tag}>{tag}</Tag>)}
                        </Space>
                      </List.Item>
                    )}
                  />
                </Card>
              </Space>
            </TabPane>

            <TabPane 
              tab={<span><SafetyOutlined /> 隐私安全</span>} 
              key="privacy"
            >
              <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                <Alert
                  message="隐私保护状态"
                  description="所有情感记忆数据已加密存储，访问日志已开启审计"
                  type="success"
                  showIcon
                  action={
                    <Button size="small" type="primary">
                      查看审计日志
                    </Button>
                  }
                />

                {/* 隐私设置 */}
                <Card title="隐私与安全设置" size="small">
                  <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                    <Row justify="space-between" align="middle">
                      <Col span={18}>
                        <Space direction="vertical" size="small">
                          <Text strong>端到端加密</Text>
                          <Text type="secondary">使用AES-256加密所有情感记忆数据</Text>
                        </Space>
                      </Col>
                      <Col span={6} style={{ textAlign: 'right' }}>
                        <Switch defaultChecked />
                      </Col>
                    </Row>
                    <Divider />
                    <Row justify="space-between" align="middle">
                      <Col span={18}>
                        <Space direction="vertical" size="small">
                          <Text strong>匿名化处理</Text>
                          <Text type="secondary">自动脱敏个人身份信息</Text>
                        </Space>
                      </Col>
                      <Col span={6} style={{ textAlign: 'right' }}>
                        <Switch defaultChecked />
                      </Col>
                    </Row>
                    <Divider />
                    <Row justify="space-between" align="middle">
                      <Col span={18}>
                        <Space direction="vertical" size="small">
                          <Text strong>访问审计</Text>
                          <Text type="secondary">记录所有数据访问操作</Text>
                        </Space>
                      </Col>
                      <Col span={6} style={{ textAlign: 'right' }}>
                        <Switch defaultChecked />
                      </Col>
                    </Row>
                    <Divider />
                    <Row justify="space-between" align="middle">
                      <Col span={18}>
                        <Space direction="vertical" size="small">
                          <Text strong>自动清理</Text>
                          <Text type="secondary">90天后自动清理低重要性记忆</Text>
                        </Space>
                      </Col>
                      <Col span={6} style={{ textAlign: 'right' }}>
                        <Switch />
                      </Col>
                    </Row>
                  </Space>
                </Card>

                {/* 数据管理 */}
                <Card title="数据管理" size="small">
                  <Space wrap>
                    <Button icon={<ExportOutlined />}>导出我的数据</Button>
                    <Button icon={<ImportOutlined />}>导入备份</Button>
                    <Button icon={<DeleteOutlined />} danger>清除所有记忆</Button>
                    <Button icon={<LockOutlined />}>更改加密密钥</Button>
                  </Space>
                </Card>
              </Space>
            </TabPane>
          </Tabs>
        </Card>
      </Space>
    </div>
  )
}

export default EmotionalMemoryManagementPage