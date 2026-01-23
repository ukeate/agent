import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useState, useEffect, useRef } from 'react'
import { logger } from '../utils/logger'
import {
  Card,
  Tabs,
  Button,
  Input,
  Select,
  Row,
  Col,
  Typography,
  Space,
  Tag,
  Alert,
  Timeline,
  Progress,
  Divider,
  Badge,
  message,
  Table,
  Modal,
  Form,
  InputNumber,
  Switch,
  Slider,
  List,
  Avatar,
  Rate,
  Checkbox,
  DatePicker,
  Upload,
  Tooltip,
  Spin,
} from 'antd'
import {
  LineChartOutlined,
  BarChartOutlined,
  NodeIndexOutlined,
  ExperimentOutlined,
  SyncOutlined,
  AlertOutlined,
  BulbOutlined,
  SettingOutlined,
  EyeOutlined,
  ThunderboltOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  DownloadOutlined,
  UploadOutlined,
  FireOutlined,
  TrophyOutlined,
  HeartOutlined,
  TeamOutlined,
  ClockCircleOutlined,
  ArrowUpOutlined,
  ArrowDownOutlined,
  MinusOutlined,
} from '@ant-design/icons'
import * as d3 from 'd3'

const { Title, Text, Paragraph } = Typography
const { TextArea } = Input
const { TabPane } = Tabs
const { Option } = Select
const { RangePicker } = DatePicker

// 情感流数据类型
interface EmotionFlowPoint {
  timestamp: string
  user_id: string
  emotion: string
  intensity: number
  valence: number
  arousal: number
  context?: Record<string, any>
}

interface EmotionFlow {
  session_id: string
  participants: string[]
  start_time: string
  end_time: string
  flow_points: EmotionFlowPoint[]
  emotional_peaks: Array<{
    timestamp: string
    user_id: string
    emotion: string
    intensity: number
    context?: Record<string, any>
    type: string
  }>
  emotional_valleys: Array<{
    timestamp: string
    user_id: string
    emotion: string
    intensity: number
    context?: Record<string, any>
    type: string
  }>
  turning_points: Array<{
    timestamp: string
    user_id: string
    emotion: string
    intensity: number
    trend_change: number
    influence_factor: string
    type: string
  }>
  overall_trend: string
  dominant_emotions: Record<string, number>
}

interface SessionAnalysisData {
  session_id: string
  duration: number
  participant_count: number
  message_count: number
  average_emotion_intensity: number
  peak_count: number
  valley_count: number
  turning_point_count: number
  trend: string
  health_score: number
}

// API 客户端
const emotionFlowApi = {
  async analyzeEmotionFlow(sessionId: string, conversationData: any[]) {
    try {
      const response = await apiFetch(
        buildApiUrl(`/api/v1/social-emotion/analyze`),
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user_id: 'current_user',
            session_id: sessionId,
            emotion_data: {
              emotions: { neutral: 0.5 },
              intensity: 0.5,
              confidence: 0.8,
            },
            social_context: {
              participants: ['user1', 'user2'],
              scenario: 'meeting',
            },
            analysis_type: ['emotion_flow'],
            cultural_context: 'zh-CN',
            privacy_consent: true,
          }),
        }
      )

      const result = await response.json()

      return {
        success: true,
        data: {
          emotion_flow: result.results?.emotion_flow || null,
        },
      }
    } catch (error) {
      logger.error('情感流分析失败:', error)
      return {
        success: false,
        error: error.message,
      }
    }
  },

  async getRealTimeFlow(sessionId: string) {
    try {
      const response = await apiFetch(
        buildApiUrl(`/api/v1/social-emotion/status`)
      )

      const result = await response.json()
      return {
        success: true,
        data: result,
      }
    } catch (error) {
      return {
        success: false,
      }
    }
  },

  async getSessionAnalytics(sessionId: string) {
    try {
      const response = await apiFetch(
        buildApiUrl(`/api/v1/social-emotion/dashboard`)
      )

      return {
        success: true,
        data: await response.json(),
      }
    } catch (error) {
      return {
        success: false,
      }
    }
  },

  async exportFlowData(sessionId: string, format: string) {
    try {
      const response = await apiFetch(buildApiUrl('/social-emotion/export'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: 'current_user',
          data_types: ['emotion_flow', 'analytics'],
          format_type: format,
        }),
      })

      return await response.json()
    } catch (error) {
      logger.error('导出失败:', error)
      return { success: false, error: error.message }
    }
  },
}

// 已移除本地情感流模拟数据，界面依赖真实API返回

const EmotionFlowAnalysisPage: React.FC = () => {
  const [currentFlow, setCurrentFlow] = useState<EmotionFlow | null>(null)
  const [sessionAnalytics, setSessionAnalytics] =
    useState<SessionAnalysisData | null>(null)
  const [loading, setLoading] = useState(false)
  const [realTimeMode, setRealTimeMode] = useState(false)
  const [selectedSession, setSelectedSession] = useState('session_1')
  const [showAnalysisModal, setShowAnalysisModal] = useState(false)
  const [showExportModal, setShowExportModal] = useState(false)

  const [analysisForm] = Form.useForm()
  const [exportForm] = Form.useForm()
  const chartRef = useRef<SVGSVGElement>(null)
  const realTimeChartRef = useRef<SVGSVGElement>(null)

  // 会话选项
  const sessionOptions = [
    { value: 'session_1', label: '团队会议-2024-01-15' },
    { value: 'session_2', label: '产品讨论-2024-01-14' },
    { value: 'session_3', label: '技术评审-2024-01-13' },
    { value: 'session_4', label: '项目规划-2024-01-12' },
    { value: 'session_5', label: '客户沟通-2024-01-11' },
  ]

  useEffect(() => {
    loadFlowData()
  }, [selectedSession])

  useEffect(() => {
    let interval: ReturnType<typeof setTimeout>
    if (realTimeMode) {
      interval = setInterval(updateRealTimeData, 3000)
    }
    return () => {
      if (interval) clearInterval(interval)
    }
  }, [realTimeMode])

  useEffect(() => {
    if (currentFlow && chartRef.current) {
      renderFlowChart()
    }
  }, [currentFlow])

  const loadFlowData = async () => {
    setLoading(true)
    try {
      const [flowResult, analyticsResult] = await Promise.all([
        emotionFlowApi.analyzeEmotionFlow(selectedSession, []),
        emotionFlowApi.getSessionAnalytics(selectedSession),
      ])

      if (flowResult.data?.emotion_flow) {
        setCurrentFlow(flowResult.data.emotion_flow)
      }

      if (analyticsResult.data) {
        setSessionAnalytics(analyticsResult.data)
      }
      if (!flowResult.success) {
        message.error('情感流数据加载失败')
      } else {
        message.success('情感流数据加载成功')
      }
    } catch (error) {
      logger.error('加载失败:', error)
      message.error('数据加载失败')
    } finally {
      setLoading(false)
    }
  }

  const updateRealTimeData = async () => {
    if (!realTimeMode) return

    try {
      const result = await emotionFlowApi.getRealTimeFlow(selectedSession)
      if (result.data) {
        // 更新实时数据
        renderRealTimeChart(result.data)
      }
    } catch (error) {
      logger.error('实时数据更新失败:', error)
    }
  }

  const renderFlowChart = () => {
    if (!chartRef.current || !currentFlow) return

    const svg = d3.select(chartRef.current)
    svg.selectAll('*').remove()

    const margin = { top: 20, right: 30, bottom: 40, left: 50 }
    const width = 800 - margin.left - margin.right
    const height = 400 - margin.top - margin.bottom

    const g = svg
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    // 时间比例尺
    const xScale = d3
      .scaleTime()
      .domain(
        d3.extent(currentFlow.flow_points, d => new Date(d.timestamp)) as [
          Date,
          Date,
        ]
      )
      .range([0, width])

    // 强度比例尺
    const yScale = d3.scaleLinear().domain([0, 1]).range([height, 0])

    // 颜色比例尺
    const colorScale = d3
      .scaleOrdinal(d3.schemeSet3)
      .domain(currentFlow.participants)

    // 添加坐标轴
    g.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale).tickFormat(d3.timeFormat('%H:%M')))

    g.append('g').call(d3.axisLeft(yScale))

    // 按用户分组绘制线条
    currentFlow.participants.forEach(participant => {
      const participantData = currentFlow.flow_points
        .filter(p => p.user_id === participant)
        .sort(
          (a, b) =>
            new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
        )

      if (participantData.length === 0) return

      const line = d3
        .line<EmotionFlowPoint>()
        .x(d => xScale(new Date(d.timestamp)))
        .y(d => yScale(d.intensity))
        .curve(d3.curveMonotoneX)

      g.append('path')
        .datum(participantData)
        .attr('fill', 'none')
        .attr('stroke', colorScale(participant))
        .attr('stroke-width', 2)
        .attr('d', line)

      // 添加数据点
      g.selectAll(`.point-${participant.replace(/\s+/g, '')}`)
        .data(participantData)
        .enter()
        .append('circle')
        .attr('class', `point-${participant.replace(/\s+/g, '')}`)
        .attr('cx', d => xScale(new Date(d.timestamp)))
        .attr('cy', d => yScale(d.intensity))
        .attr('r', 4)
        .attr('fill', colorScale(participant))
        .on('mouseover', function (event, d) {
          d3.select(this).attr('r', 6)
          // 显示tooltip (简化版)
        })
        .on('mouseout', function () {
          d3.select(this).attr('r', 4)
        })
    })

    // 标记峰值
    currentFlow.emotional_peaks.forEach(peak => {
      g.append('circle')
        .attr('cx', xScale(new Date(peak.timestamp)))
        .attr('cy', yScale(peak.intensity))
        .attr('r', 8)
        .attr('fill', 'red')
        .attr('opacity', 0.7)
        .append('title')
        .text(`峰值: ${peak.emotion} (${peak.intensity.toFixed(2)})`)
    })

    // 标记低谷
    currentFlow.emotional_valleys.forEach(valley => {
      g.append('circle')
        .attr('cx', xScale(new Date(valley.timestamp)))
        .attr('cy', yScale(valley.intensity))
        .attr('r', 8)
        .attr('fill', 'blue')
        .attr('opacity', 0.7)
        .append('title')
        .text(`低谷: ${valley.emotion} (${valley.intensity.toFixed(2)})`)
    })

    // 添加图例
    const legend = g
      .append('g')
      .attr('transform', `translate(${width - 120}, 20)`)

    currentFlow.participants.forEach((participant, i) => {
      const legendRow = legend
        .append('g')
        .attr('transform', `translate(0, ${i * 20})`)

      legendRow
        .append('circle')
        .attr('r', 5)
        .attr('fill', colorScale(participant))

      legendRow
        .append('text')
        .attr('x', 10)
        .attr('y', 5)
        .style('font-size', '12px')
        .text(participant)
    })
  }

  const renderRealTimeChart = (realtimeData: any) => {
    if (!realTimeChartRef.current) return

    // 简化的实时图表渲染
    const svg = d3.select(realTimeChartRef.current)
    svg.selectAll('*').remove()

    // 添加实时情感显示
    svg
      .append('text')
      .attr('x', 50)
      .attr('y', 30)
      .style('font-size', '16px')
      .style('font-weight', 'bold')
      .text(`当前情感: ${realtimeData.current_emotion}`)

    svg
      .append('text')
      .attr('x', 50)
      .attr('y', 50)
      .style('font-size', '14px')
      .text(`强度: ${(realtimeData.current_intensity * 100).toFixed(1)}%`)
  }

  const performNewAnalysis = async (values: any) => {
    setLoading(true)
    try {
      const result = await emotionFlowApi.analyzeEmotionFlow(
        values.session_id || selectedSession,
        [] // 实际应传入对话数据
      )

      if (result.data?.emotion_flow) {
        setCurrentFlow(result.data.emotion_flow)
        message.success('分析完成')
        setShowAnalysisModal(false)
      }
    } catch (error) {
      message.error('分析失败')
    } finally {
      setLoading(false)
    }
  }

  const exportFlowData = async (values: any) => {
    try {
      const result = await emotionFlowApi.exportFlowData(
        selectedSession,
        values.format
      )
      if (result.success) {
        message.success('导出成功')
        setShowExportModal(false)
      } else {
        message.error('导出失败')
      }
    } catch (error) {
      message.error('导出失败')
    }
  }

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'improving':
        return <ArrowUpOutlined style={{ color: '#52c41a' }} />
      case 'declining':
        return <ArrowDownOutlined style={{ color: '#ff4d4f' }} />
      default:
        return <MinusOutlined style={{ color: '#1890ff' }} />
    }
  }

  const getTrendColor = (trend: string) => {
    switch (trend) {
      case 'improving':
        return '#52c41a'
      case 'declining':
        return '#ff4d4f'
      default:
        return '#1890ff'
    }
  }

  const renderOverviewCards = () => (
    <Row gutter={16}>
      <Col span={6}>
        <Card>
          <div style={{ textAlign: 'center' }}>
            <TeamOutlined
              style={{ fontSize: 24, color: '#1890ff', marginBottom: 8 }}
            />
            <div style={{ fontSize: 24, fontWeight: 'bold', color: '#1890ff' }}>
              {currentFlow?.participants.length || 0}
            </div>
            <div style={{ color: '#8c8c8c' }}>参与者</div>
          </div>
        </Card>
      </Col>
      <Col span={6}>
        <Card>
          <div style={{ textAlign: 'center' }}>
            <ClockCircleOutlined
              style={{ fontSize: 24, color: '#52c41a', marginBottom: 8 }}
            />
            <div style={{ fontSize: 24, fontWeight: 'bold', color: '#52c41a' }}>
              {sessionAnalytics ? Math.round(sessionAnalytics.duration) : 0}
            </div>
            <div style={{ color: '#8c8c8c' }}>时长(分钟)</div>
          </div>
        </Card>
      </Col>
      <Col span={6}>
        <Card>
          <div style={{ textAlign: 'center' }}>
            <FireOutlined
              style={{ fontSize: 24, color: '#fa8c16', marginBottom: 8 }}
            />
            <div style={{ fontSize: 24, fontWeight: 'bold', color: '#fa8c16' }}>
              {currentFlow?.emotional_peaks.length || 0}
            </div>
            <div style={{ color: '#8c8c8c' }}>情感高峰</div>
          </div>
        </Card>
      </Col>
      <Col span={6}>
        <Card>
          <div style={{ textAlign: 'center' }}>
            <TrophyOutlined
              style={{ fontSize: 24, color: '#722ed1', marginBottom: 8 }}
            />
            <div style={{ fontSize: 20, fontWeight: 'bold', color: '#722ed1' }}>
              {sessionAnalytics
                ? Math.round(sessionAnalytics.health_score * 100)
                : 0}
              %
            </div>
            <div style={{ color: '#8c8c8c' }}>健康度</div>
          </div>
        </Card>
      </Col>
    </Row>
  )

  const renderFlowChart = () => (
    <Card
      title={
        <span>
          <LineChartOutlined style={{ marginRight: 8 }} />
          情感流变化图表
        </span>
      }
    >
      {currentFlow ? (
        <div>
          <div style={{ marginBottom: 16, textAlign: 'center' }}>
            <Space>
              <span>整体趋势:</span>
              <Tag
                color={getTrendColor(currentFlow.overall_trend)}
                icon={getTrendIcon(currentFlow.overall_trend)}
              >
                {currentFlow.overall_trend === 'improving'
                  ? '改善中'
                  : currentFlow.overall_trend === 'declining'
                    ? '下降中'
                    : '稳定'}
              </Tag>
            </Space>
          </div>
          <svg ref={chartRef}></svg>
        </div>
      ) : (
        <div style={{ textAlign: 'center', padding: 60 }}>
          <Spin spinning={loading}>
            <Text type="secondary">暂无情感流数据</Text>
          </Spin>
        </div>
      )}
    </Card>
  )

  const renderKeyMoments = () => {
    const columns = [
      {
        title: '时间',
        dataIndex: 'timestamp',
        key: 'timestamp',
        render: (timestamp: string) => (
          <Text style={{ fontSize: '12px' }}>
            {new Date(timestamp).toLocaleTimeString()}
          </Text>
        ),
      },
      {
        title: '类型',
        dataIndex: 'type',
        key: 'type',
        render: (type: string) => {
          const typeMap = {
            peak: { color: 'red', icon: <ArrowUpOutlined />, text: '情感高峰' },
            valley: {
              color: 'blue',
              icon: <ArrowDownOutlined />,
              text: '情感低谷',
            },
            turning_point: {
              color: 'orange',
              icon: <MinusOutlined />,
              text: '转折点',
            },
          }
          const config = typeMap[type as keyof typeof typeMap]
          return (
            <Tag color={config.color} icon={config.icon}>
              {config.text}
            </Tag>
          )
        },
      },
      {
        title: '用户',
        dataIndex: 'user_id',
        key: 'user_id',
        render: (userId: string) => (
          <Avatar size="small" style={{ backgroundColor: '#87d068' }}>
            {userId.charAt(0).toUpperCase()}
          </Avatar>
        ),
      },
      {
        title: '情感',
        dataIndex: 'emotion',
        key: 'emotion',
        render: (emotion: string) => <Tag color="purple">{emotion}</Tag>,
      },
      {
        title: '强度',
        dataIndex: 'intensity',
        key: 'intensity',
        render: (intensity: number) => (
          <Progress
            percent={Math.round(intensity * 100)}
            size="small"
            strokeColor="#1890ff"
            style={{ width: 80 }}
          />
        ),
      },
      {
        title: '影响因素',
        dataIndex: 'influence_factor',
        key: 'influence_factor',
        render: (factor: string) => factor && <Tag color="cyan">{factor}</Tag>,
      },
    ]

    const keyMoments = [
      ...(currentFlow?.emotional_peaks.map(peak => ({
        ...peak,
        type: 'peak',
      })) || []),
      ...(currentFlow?.emotional_valleys.map(valley => ({
        ...valley,
        type: 'valley',
      })) || []),
      ...(currentFlow?.turning_points.map(point => ({
        ...point,
        type: 'turning_point',
      })) || []),
    ].sort(
      (a, b) =>
        new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
    )

    return (
      <Card
        title={
          <span>
            <AlertOutlined style={{ marginRight: 8 }} />
            关键时刻 ({keyMoments.length})
          </span>
        }
      >
        <Table
          columns={columns}
          dataSource={keyMoments}
          rowKey={(record, index) => `${record.type}-${index}`}
          pagination={{ pageSize: 10 }}
          size="small"
        />
      </Card>
    )
  }

  const renderDominantEmotions = () => (
    <Card
      title={
        <span>
          <HeartOutlined style={{ marginRight: 8 }} />
          主导情感分布
        </span>
      }
    >
      {currentFlow?.dominant_emotions ? (
        <div>
          {Object.entries(currentFlow.dominant_emotions)
            .sort(([, a], [, b]) => b - a)
            .slice(0, 8)
            .map(([emotion, value], index) => (
              <div key={emotion} style={{ marginBottom: 16 }}>
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    marginBottom: 4,
                  }}
                >
                  <Text style={{ textTransform: 'capitalize' }}>{emotion}</Text>
                  <Text>{(value * 100).toFixed(1)}%</Text>
                </div>
                <Progress
                  percent={Math.round(value * 100)}
                  strokeColor={`hsl(${index * 45}, 70%, 50%)`}
                  size="small"
                />
              </div>
            ))}
        </div>
      ) : (
        <Text type="secondary">暂无数据</Text>
      )}
    </Card>
  )

  const renderRealTimeMonitor = () => (
    <Card
      title={
        <span>
          <ThunderboltOutlined style={{ marginRight: 8 }} />
          实时监控
        </span>
      }
      extra={
        <Switch
          checked={realTimeMode}
          onChange={setRealTimeMode}
          checkedChildren="开启"
          unCheckedChildren="关闭"
        />
      }
    >
      {realTimeMode ? (
        <div>
          <Alert
            message="实时监控已启用"
            description="正在监控当前会话的情感变化"
            type="success"
            showIcon
            style={{ marginBottom: 16 }}
          />
          <svg ref={realTimeChartRef} width={300} height={100}></svg>
        </div>
      ) : (
        <div style={{ textAlign: 'center', padding: 40 }}>
          <Text type="secondary">启用实时监控以查看动态情感变化</Text>
        </div>
      )}
    </Card>
  )

  const renderAnalysisModal = () => (
    <Modal
      title="新建情感流分析"
      open={showAnalysisModal}
      onCancel={() => setShowAnalysisModal(false)}
      footer={[
        <Button key="cancel" onClick={() => setShowAnalysisModal(false)}>
          取消
        </Button>,
        <Button
          key="analyze"
          type="primary"
          loading={loading}
          onClick={() => analysisForm.submit()}
        >
          开始分析
        </Button>,
      ]}
      width={600}
    >
      <Form form={analysisForm} layout="vertical" onFinish={performNewAnalysis}>
        <Alert
          message="情感流分析"
          description="分析会话中的情感变化模式，识别关键时刻和趋势"
          type="info"
          showIcon
          style={{ marginBottom: 24 }}
        />

        <Form.Item
          label="会话选择"
          name="session_id"
          rules={[{ required: true, message: '请选择分析的会话' }]}
        >
          <Select placeholder="选择要分析的会话">
            {sessionOptions.map(session => (
              <Option key={session.value} value={session.value}>
                {session.label}
              </Option>
            ))}
          </Select>
        </Form.Item>

        <Form.Item label="分析时间范围" name="time_range">
          <RangePicker showTime />
        </Form.Item>

        <Form.Item label="分析深度" name="analysis_depth" initialValue={70}>
          <Slider
            min={30}
            max={100}
            marks={{
              30: '快速',
              50: '标准',
              70: '深入',
              100: '详细',
            }}
          />
        </Form.Item>

        <Form.Item name="include_context" valuePropName="checked">
          <Checkbox>包含上下文分析</Checkbox>
        </Form.Item>

        <Form.Item name="detect_anomalies" valuePropName="checked">
          <Checkbox>异常情感检测</Checkbox>
        </Form.Item>

        <Form.Item name="real_time_updates" valuePropName="checked">
          <Checkbox>启用实时更新</Checkbox>
        </Form.Item>
      </Form>
    </Modal>
  )

  const renderExportModal = () => (
    <Modal
      title="导出情感流数据"
      open={showExportModal}
      onCancel={() => setShowExportModal(false)}
      footer={[
        <Button key="cancel" onClick={() => setShowExportModal(false)}>
          取消
        </Button>,
        <Button key="export" type="primary" onClick={() => exportForm.submit()}>
          导出
        </Button>,
      ]}
    >
      <Form form={exportForm} layout="vertical" onFinish={exportFlowData}>
        <Form.Item
          label="导出格式"
          name="format"
          initialValue="json"
          rules={[{ required: true, message: '请选择导出格式' }]}
        >
          <Select>
            <Option value="json">JSON</Option>
            <Option value="csv">CSV</Option>
            <Option value="xlsx">Excel</Option>
          </Select>
        </Form.Item>

        <Form.Item label="导出内容" name="content_types">
          <Checkbox.Group>
            <Row>
              <Col span={24}>
                <Checkbox value="flow_points">情感流数据点</Checkbox>
              </Col>
              <Col span={24}>
                <Checkbox value="key_moments">关键时刻</Checkbox>
              </Col>
              <Col span={24}>
                <Checkbox value="analytics">统计分析</Checkbox>
              </Col>
              <Col span={24}>
                <Checkbox value="participants">参与者信息</Checkbox>
              </Col>
            </Row>
          </Checkbox.Group>
        </Form.Item>
      </Form>
    </Modal>
  )

  return (
    <div style={{ padding: '24px' }}>
      <div
        style={{
          marginBottom: 24,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <Title level={2}>
          <LineChartOutlined style={{ marginRight: 12, color: '#1890ff' }} />
          情感流分析
        </Title>
        <Space>
          <Select
            style={{ width: 250 }}
            value={selectedSession}
            onChange={setSelectedSession}
            placeholder="选择会话"
          >
            {sessionOptions.map(session => (
              <Option key={session.value} value={session.value}>
                {session.label}
              </Option>
            ))}
          </Select>
          <Button
            type="primary"
            icon={<ExperimentOutlined />}
            onClick={() => setShowAnalysisModal(true)}
          >
            新建分析
          </Button>
          <Button
            icon={<DownloadOutlined />}
            onClick={() => setShowExportModal(true)}
          >
            导出数据
          </Button>
          <Button
            icon={<SyncOutlined />}
            loading={loading}
            onClick={loadFlowData}
          >
            刷新
          </Button>
        </Space>
      </div>

      <div style={{ marginBottom: 24 }}>{renderOverviewCards()}</div>

      <Tabs defaultActiveKey="flow-chart">
        <TabPane tab="情感流图表" key="flow-chart">
          {renderFlowChart()}
        </TabPane>

        <TabPane tab="关键时刻" key="key-moments">
          {renderKeyMoments()}
        </TabPane>

        <TabPane tab="情感分布" key="emotion-distribution">
          <Row gutter={24}>
            <Col span={16}>{renderDominantEmotions()}</Col>
            <Col span={8}>{renderRealTimeMonitor()}</Col>
          </Row>
        </TabPane>

        <TabPane tab="趋势分析" key="trend-analysis">
          <Card title="情感趋势分析">
            <Alert
              message="趋势分析功能"
              description="深度分析情感流的长期趋势和模式"
              type="info"
              showIcon
              style={{ marginBottom: 24 }}
            />
            <div style={{ textAlign: 'center', padding: 60 }}>
              <Text type="secondary">趋势分析功能正在开发中...</Text>
            </div>
          </Card>
        </TabPane>
      </Tabs>

      {renderAnalysisModal()}
      {renderExportModal()}
    </div>
  )
}

export default EmotionFlowAnalysisPage
